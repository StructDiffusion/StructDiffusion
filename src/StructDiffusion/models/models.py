import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from StructDiffusion.models.encoders import EncoderMLP, DropoutSampler, SinusoidalPositionEmbeddings
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from StructDiffusion.models.point_transformer import PointTransformerEncoderSmall
from StructDiffusion.models.point_transformer_large import PointTransformerCls


class TransformerDiffusionModel(torch.nn.Module):

    def __init__(self, vocab_size,
                 # transformer params
                 encoder_input_dim=256,
                 num_attention_heads=8, encoder_hidden_dim=16, encoder_dropout=0.1, encoder_activation="relu",
                 encoder_num_layers=8,
                 # output head params
                 structure_dropout=0.0, object_dropout=0.0,
                 # pc encoder params
                 ignore_rgb=False, pc_emb_dim=256, posed_pc_emb_dim=80,
                 pose_emb_dim=80,
                 max_seq_size=7, max_token_type_size=4,
                 seq_pos_emb_dim=8, seq_type_emb_dim=8,
                 word_emb_dim=160,
                 time_emb_dim=80,
                 use_virtual_structure_frame=True,
                 ):
        super(TransformerDiffusionModel, self).__init__()

        assert posed_pc_emb_dim + pose_emb_dim == word_emb_dim
        assert encoder_input_dim == word_emb_dim + time_emb_dim + seq_pos_emb_dim + seq_type_emb_dim

        # 3D translation + 6D rotation
        action_dim = 3 + 6

        # default:
        # 256 = 80 (point cloud) + 80 (position) + 80 (time) + 8 (position idx) + 8 (token idx)
        # 256 = 160 (word embedding) + 80 (time) + 8 (position idx) + 8 (token idx)

        # PC
        self.ignore_rgb = ignore_rgb
        if ignore_rgb:
            self.pc_encoder = PointTransformerEncoderSmall(output_dim=pc_emb_dim, input_dim=3, mean_center=True)
        else:
            self.pc_encoder = PointTransformerEncoderSmall(output_dim=pc_emb_dim, input_dim=6, mean_center=True)
        self.posed_pc_encoder = EncoderMLP(pc_emb_dim, posed_pc_emb_dim, uses_pt=True)

        # for virtual structure frame
        self.use_virtual_structure_frame = use_virtual_structure_frame
        if use_virtual_structure_frame:
            self.virtual_frame_embed = nn.Parameter(torch.randn(1, 1, posed_pc_emb_dim))  # B, 1, posed_pc_emb_dim

        # for language
        self.word_embeddings = torch.nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)

        # for diffusion
        self.pose_encoder = nn.Sequential(nn.Linear(action_dim, pose_emb_dim))
        self.time_embeddings = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # for transformer
        self.position_embeddings = torch.nn.Embedding(max_seq_size, seq_pos_emb_dim)
        self.type_embeddings = torch.nn.Embedding(max_token_type_size, seq_type_emb_dim)

        encoder_layers = TransformerEncoderLayer(encoder_input_dim, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)

        self.struct_head = DropoutSampler(encoder_input_dim, action_dim, dropout_rate=structure_dropout)
        self.obj_head = DropoutSampler(encoder_input_dim, action_dim, dropout_rate=object_dropout)

    def encode_posed_pc(self, pcs, batch_size, num_objects):
        if self.ignore_rgb:
            center_xyz, x = self.pc_encoder(pcs[:, :, :3], None)
        else:
            center_xyz, x = self.pc_encoder(pcs[:, :, :3], pcs[:, :, 3:])
        posed_pc_embed = self.posed_pc_encoder(x, center_xyz)
        posed_pc_embed = posed_pc_embed.reshape(batch_size, num_objects, -1)
        return posed_pc_embed

    def forward(self, t, pcs, sentence, poses, type_index, position_index, pad_mask):

        batch_size, num_objects, num_pts, _ = pcs.shape
        _, num_poses, _ = poses.shape
        _, sentence_len = sentence.shape
        _, total_len = type_index.shape

        pcs = pcs.reshape(batch_size * num_objects, num_pts, -1)
        posed_pc_embed = self.encode_posed_pc(pcs, batch_size, num_objects)

        pose_embed = self.pose_encoder(poses)

        if self.use_virtual_structure_frame:
            virtual_frame_embed = self.virtual_frame_embed.repeat(batch_size, 1, 1)
            posed_pc_embed = torch.cat([virtual_frame_embed, posed_pc_embed], dim=1)
        tgt_obj_embed = torch.cat([pose_embed, posed_pc_embed], dim=-1)

        #########################
        sentence_embed = self.word_embeddings(sentence)

        #########################

        # transformer time dim: sentence, struct, obj
        # transformer feat dim: obj pc + pose / word, time, token type, position

        time_embed = self.time_embeddings(t)  # B, dim
        time_embed = time_embed.unsqueeze(1).repeat(1, total_len, 1)  # B, L, dim

        position_embed = self.position_embeddings(position_index)
        type_embed = self.type_embeddings(type_index)

        tgt_sequence_encode = torch.cat([sentence_embed, tgt_obj_embed], dim=1)
        tgt_sequence_encode = torch.cat([tgt_sequence_encode, time_embed, position_embed, type_embed], dim=-1)

        tgt_pad_mask = pad_mask

        #########################
        # sequence_encode: [batch size, sequence_length, encoder input dimension]
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        tgt_sequence_encode = tgt_sequence_encode.transpose(1, 0)
        # convert to bool
        tgt_pad_mask = (tgt_pad_mask == 1)
        # encode: [sequence_length, batch_size, embedding_size]
        encode = self.encoder(tgt_sequence_encode, src_key_padding_mask=tgt_pad_mask)
        encode = encode.transpose(1, 0)
        #########################

        target_encodes = encode[:, -num_poses:, :]
        if self.use_virtual_structure_frame:
            obj_encodes = target_encodes[:, 1:, :]
            pred_obj_poses = self.obj_head(obj_encodes)  # B, N, 3 + 6
            struct_encode = encode[:, 0, :].unsqueeze(1)
            # use a different sampler for struct prediction since it should have larger variance than object predictions
            pred_struct_pose = self.struct_head(struct_encode)  # B, 1, 3 + 6
            pred_poses = torch.cat([pred_struct_pose, pred_obj_poses], dim=1)
        else:
            pred_poses = self.obj_head(target_encodes)  # B, N, 3 + 6

        assert pred_poses.shape == poses.shape

        return pred_poses


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=.25):
        super(FocalLoss, self).__init__()
        # self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        # targets = targets.type(torch.long)
        # at = self.alpha.gather(0, targets.data.view(-1))
        # F_loss = at*(1-pt)**self.gamma * BCE_loss
        F_loss = (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()


class PCTDiscriminator(torch.nn.Module):

    def __init__(self, max_num_objects, include_env_pc=False, pct_random_sampling=False):

        super(PCTDiscriminator, self).__init__()

        # input_dim: xyz + one hot for each object
        if include_env_pc:
            self.classifier = PointTransformerCls(input_dim=max_num_objects + 1 + 3, output_dim=1, use_random_sampling=pct_random_sampling)
        else:
            self.classifier = PointTransformerCls(input_dim=max_num_objects + 3, output_dim=1, use_random_sampling=pct_random_sampling)

    def forward(self, scene_xyz):
        label = self.classifier(scene_xyz)
        return label

    def convert_logits(self, logits):
        return torch.sigmoid(logits)

