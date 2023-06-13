import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from StructDiffusion.models.encoders import EncoderMLP, DropoutSampler, SinusoidalPositionEmbeddings
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from StructDiffusion.models.point_transformer import PointTransformerEncoderSmall


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
        self.virtual_frame_embeddings = torch.nn.Embedding(1, posed_pc_emb_dim)

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

    def encode_posed_pc(self, xyzs, rgbs, batch_size, num_objects):
        if self.ignore_rgb:
            center_xyz, x = self.pc_encoder(xyzs, None)
        else:
            center_xyz, x = self.pc_encoder(xyzs, rgbs)
        posed_pc_embed = self.posed_pc_encoder(x, center_xyz)
        posed_pc_embed = posed_pc_embed.reshape(batch_size, num_objects, -1)
        return posed_pc_embed

    def forward(self, t, xyzs, obj_xyztheta_inputs, struct_xyztheta_inputs, sentence,
                position_index, struct_position_index,
                token_type_index, struct_token_type_index,
                start_token,
                object_pad_mask, struct_pad_mask, sentence_pad_mask):

        # ToDo: add time embeddings

        # print(xyzs.shape)
        # print(object_pad_mask.shape)
        # print(sentence.shape)
        # print(sentence_pad_mask.shape)
        # print(token_type_index.shape)
        # print(obj_x_inputs.shape)
        # print(obj_y_inputs.shape)
        # print(obj_theta_inputs.shape)
        # print(position_index.shape)

        batch_size, num_target_objects, num_pts, _ = xyzs.shape
        _, sentence_len = sentence.shape

        xyzs = xyzs.reshape(batch_size * num_target_objects, num_pts, -1)
        posed_pc_embed = self.encode_posed_pc(xyzs, None, batch_size, num_target_objects)

        xyztheta_embed = self.pose_encoder(torch.cat([struct_xyztheta_inputs, obj_xyztheta_inputs], dim=1))

        # at this point, obj_pc_embed has size [batch size, num objs, some dim], obj_xytheta_embed [batch size, num objs, some dim]
        # combine them into [batch size, num objs, some dim]
        # we need to shift obj_xytheta_embed to the right by one position and add a start token
        start_token_embed = self.virtual_frame_embeddings(start_token)

        tgt_obj_embed = torch.cat([start_token_embed, posed_pc_embed], dim=1)
        tgt_obj_embed = torch.cat([xyztheta_embed, tgt_obj_embed], dim=-1)

        #########################
        sentence_embed = self.word_embeddings(sentence)

        #########################

        # transformer time dim: sentence, struct, obj
        # transformer feat dim: obj pc + pose / word, time, token type, position

        time_embed = self.time_embeddings(t)  # B, dim
        time_embed = time_embed.unsqueeze(1).repeat(1, sentence_len + 1 + num_target_objects, 1)  # B, L + 1 + N, dim
        position_embed = self.position_embeddings(torch.cat([position_index[:, :sentence_len], struct_position_index, position_index[:, -num_target_objects:]], dim=1))
        type_embed = self.type_embeddings(torch.cat([token_type_index[:, :sentence_len], struct_token_type_index, token_type_index[:, -num_target_objects:]], dim=1))

        tgt_sequence_encode = torch.cat([sentence_embed, tgt_obj_embed], dim=1)
        tgt_sequence_encode = torch.cat([tgt_sequence_encode, time_embed, position_embed, type_embed], dim=-1)
        tgt_pad_mask = torch.cat([sentence_pad_mask, struct_pad_mask, object_pad_mask], dim=1)  # B, L + 1 + N

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

        obj_encodes = encode[:, sentence_len + 1:, :]
        obj_xyztheta_outputs = self.obj_head(obj_encodes)  # B, N, 3 + 6

        struct_encodes = encode[:, sentence_len, :].unsqueeze(1)  # B, 1, 3 + 6
        # use a different sampler for struct prediction since it should have larger variance than object predictions
        struct_xyztheta_outputs = self.struct_head(struct_encodes)

        # predictions = {"obj_xyztheta_outputs": obj_xyztheta_outputs,
        #                "struct_xyztheta_outputs": struct_xyztheta_outputs}

        return struct_xyztheta_outputs, obj_xyztheta_outputs