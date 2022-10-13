import torch
import torch.nn as nn

from StructDiffusion.models.encoders import EncoderMLP, DropoutSampler
from torch.nn import Transformer
from StructDiffusion.models.point_transformer import PointTransformerEncoderSmall
from StructDiffusion.utils.rotation_continuity import compute_rotation_matrix_from_ortho6d


class PriorContinuousOutEncoderDecoderStructPCT6DDropoutAllObjects(torch.nn.Module):
    """
    This model takes in point clouds of all objects
    """

    def __init__(self, vocab_size,
                 num_attention_heads=8, encoder_hidden_dim=16, encoder_dropout=0.1, encoder_activation="relu",
                 encoder_num_layers=8, structure_dropout=0.0, object_dropout=0.0, theta_loss_divide=None, ignore_rgb=False):
        super(PriorContinuousOutEncoderDecoderStructPCT6DDropoutAllObjects, self).__init__()

        print("Transformer Encoder Decoder Struct with Point Transformer 6D All Objects")
        print("structure dropout", structure_dropout)
        print("object dropout:", object_dropout)
        print("theta loss divide:", theta_loss_divide)
        print("ignore rgb:", ignore_rgb)

        self.theta_loss_divide = theta_loss_divide
        self.ignore_rgb = ignore_rgb

        # object encode will have dim 256
        if ignore_rgb:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=3, mean_center=True)
        else:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=True)

        # 256 = 120 (point cloud) + 120 (position) + 8 (position idx) + 8 (token type)
        # Important: we set uses_pt to true because we want the model to consider the positions of objects that
        #  don't need to be rearranged.
        self.mlp = EncoderMLP(256, 240, uses_pt=True)
        self.position_encoder = nn.Sequential(nn.Linear(3 + 3 * 3, 120))
        self.start_token_embeddings = torch.nn.Embedding(1, 240)

        self.point_cloud_downscale = torch.nn.Linear(240, 120)

        self.word_embeddings = torch.nn.Embedding(vocab_size, 240, padding_idx=0)
        # type sentence, other obj pc, target object pc, struct
        self.token_type_embeddings = torch.nn.Embedding(4, 8)
        # max number of objects or max length of sentence is 7
        self.position_embeddings = torch.nn.Embedding(7, 8)

        print(num_attention_heads)
        print(encoder_hidden_dim)
        print(encoder_dropout)
        print(encoder_activation)

        # encoder_layers = TransformerEncoderLayer(256, num_attention_heads,
        #                                          encoder_hidden_dim, encoder_dropout, encoder_activation)
        # self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)
        self.encoder = Transformer(d_model=256,
                                   nhead=num_attention_heads,
                                   num_encoder_layers=encoder_num_layers,
                                   num_decoder_layers=encoder_num_layers,
                                   dim_feedforward=encoder_hidden_dim,
                                   dropout=encoder_dropout)

        self.struct_dist = DropoutSampler(256, 3 + 6, dropout_rate=structure_dropout)
        self.obj_dist = DropoutSampler(256, 3 + 6, dropout_rate=object_dropout)

    def encode_pc(self, xyzs, rgbs, batch_size, num_objects):
        if self.ignore_rgb:
            center_xyz, x = self.object_encoder(xyzs, None)
        else:
            center_xyz, x = self.object_encoder(xyzs, rgbs)
        obj_pc_embed = self.mlp(x, center_xyz)
        obj_pc_embed = obj_pc_embed.reshape(batch_size, num_objects, -1)
        return obj_pc_embed

    def forward(self, xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                sentence, sentence_pad_mask, token_type_index,
                obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index, tgt_mask, start_token,
                struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                struct_position_index, struct_token_type_index, struct_pad_mask):

        # print(xyzs.shape)
        # print(object_pad_mask.shape)
        # print(sentence.shape)
        # print(sentence_pad_mask.shape)
        # print(token_type_index.shape)
        # print(obj_x_inputs.shape)
        # print(obj_y_inputs.shape)
        # print(obj_theta_inputs.shape)
        # print(position_index.shape)

        batch_size = object_pad_mask.shape[0]
        num_target_objects = object_pad_mask.shape[1]
        num_other_objects = other_object_pad_mask.shape[1]

        #########################
        obj_pc_embed = self.encode_pc(xyzs, rgbs, batch_size, num_target_objects)
        other_obj_pc_embed = self.encode_pc(other_xyzs, other_rgbs, batch_size, num_other_objects)

        obj_xytheta_inputs = torch.cat([obj_x_inputs.reshape(obj_x_inputs.shape[0], obj_x_inputs.shape[1], -1),
                                    obj_y_inputs.reshape(obj_y_inputs.shape[0], obj_y_inputs.shape[1], -1),
                                    obj_z_inputs.reshape(obj_z_inputs.shape[0], obj_z_inputs.shape[1], -1),
                                    obj_theta_inputs.reshape(obj_theta_inputs.shape[0], obj_theta_inputs.shape[1], -1)],
                                    dim=-1)
        struct_xytheta_inputs = torch.cat([struct_x_inputs.reshape(struct_x_inputs.shape[0], struct_x_inputs.shape[1], -1),
                                           struct_y_inputs.reshape(struct_y_inputs.shape[0], struct_y_inputs.shape[1], -1),
                                           struct_z_inputs.reshape(struct_z_inputs.shape[0], struct_z_inputs.shape[1], -1),
                                           struct_theta_inputs.reshape(struct_theta_inputs.shape[0], struct_theta_inputs.shape[1], -1)],
                                           dim=-1)

        xytheta_embed = self.position_encoder(torch.cat([struct_xytheta_inputs, obj_xytheta_inputs], dim=1))

        # at this point, obj_pc_embed has size [batch size, num objs, 240], obj_xytheta_embed [batch size, num objs, 120]
        # combine them into [batch size, num objs, 240]
        # we need to shift obj_xytheta_embed to the right by one position and add a start token
        start_token_embed = self.start_token_embeddings(start_token)
        tgt_obj_embed = torch.cat([xytheta_embed[:, :-1, :], self.point_cloud_downscale(obj_pc_embed)], dim=-1)
        tgt_obj_embed = torch.cat([start_token_embed, tgt_obj_embed], dim=1)

        # src can't have access to groundtruth position information
        # src should encode both target objects and other objects
        src_obj_embed = torch.cat([other_obj_pc_embed, obj_pc_embed], dim=1)

        #########################
        word_embed = self.word_embeddings(sentence)

        #########################
        position_embed = self.position_embeddings(position_index)
        token_type_embed = self.token_type_embeddings(token_type_index)
        struct_position_embed = self.position_embeddings(struct_position_index)
        struct_token_type_embed = self.token_type_embeddings(struct_token_type_index)

        src_sequence_encode = torch.cat([word_embed, src_obj_embed], dim=1)
        src_sequence_encode = torch.cat([src_sequence_encode, position_embed, token_type_embed], dim=-1)
        src_pad_mask = torch.cat([sentence_pad_mask, other_object_pad_mask, object_pad_mask], dim=1)

        tgt_sequence_encode = tgt_obj_embed
        tgt_position_embed = torch.cat([struct_position_embed, position_embed[:, -num_target_objects:, :]], dim=1)
        tgt_token_type_embed = torch.cat([struct_token_type_embed, token_type_embed[:, -num_target_objects:, :]], dim=1)
        tgt_sequence_encode = torch.cat([tgt_sequence_encode, tgt_position_embed, tgt_token_type_embed], dim=-1)
        tgt_pad_mask = torch.cat([struct_pad_mask, object_pad_mask], dim=1)

        assert tgt_mask.shape[0] == tgt_sequence_encode.shape[1], "sequence length of target mask and target sequence encodes don't match"

        #########################
        # sequence_encode: [batch size, sequence_length, encoder input dimension]
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        src_sequence_encode = src_sequence_encode.transpose(1, 0)
        tgt_sequence_encode = tgt_sequence_encode.transpose(1, 0)

        # convert to bool
        src_pad_mask = (src_pad_mask == 1)
        tgt_pad_mask = (tgt_pad_mask == 1)

        # encode: [sequence_length, batch_size, embedding_size]
        encode = self.encoder(src=src_sequence_encode, tgt=tgt_sequence_encode, tgt_mask=tgt_mask,
                              src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask,
                              memory_key_padding_mask=src_pad_mask)
        encode = encode.transpose(1, 0)
        #########################
        obj_encodes = encode[:, -num_target_objects:, :]
        obj_encodes = obj_encodes.reshape(-1, obj_encodes.shape[-1])
        # dim: [batch_size, num_features]
        obj_xyztheta_outputs = self.obj_dist(obj_encodes)

        struct_encodes = encode[:, -num_target_objects-1, :]
        struct_encodes = struct_encodes.reshape(-1, struct_encodes.shape[-1])
        # use a different sampler for struct prediction since it should have larger variance than object predictions
        struct_xyztheta_outputs = self.struct_dist(struct_encodes)

        ########################
        # input: batch * 6, output: batch * 3 * 3
        obj_theta_outputs = compute_rotation_matrix_from_ortho6d(obj_xyztheta_outputs[:, 3:]).reshape(-1, 9)
        struct_theta_inputs = compute_rotation_matrix_from_ortho6d(struct_xyztheta_outputs[:, 3:]).reshape(-1, 9)

        predictions = {"obj_x_outputs": obj_xyztheta_outputs[:, 0].unsqueeze(1),
                       "obj_y_outputs": obj_xyztheta_outputs[:, 1].unsqueeze(1),
                       "obj_z_outputs": obj_xyztheta_outputs[:, 2].unsqueeze(1),
                       "obj_theta_outputs": obj_theta_outputs,
                       "struct_x_inputs": struct_xyztheta_outputs[:, 0].unsqueeze(1),
                       "struct_y_inputs": struct_xyztheta_outputs[:, 1].unsqueeze(1),
                       "struct_z_inputs": struct_xyztheta_outputs[:, 2].unsqueeze(1),
                       "struct_theta_inputs": struct_theta_inputs}

        return predictions

    def criterion(self, predictions, labels):

        loss = 0
        for key in predictions:
            preds = predictions[key]
            gts = labels[key]

            if self.theta_loss_divide is None:
                loss += self.mse_loss(preds, gts)
            else:
                if "theta" in key:
                    loss += self.mse_loss(preds, gts) / self.theta_loss_divide
                else:
                    loss += self.mse_loss(preds, gts)

        return loss

    def mse_loss(self, input, target, ignored_index=-100, reduction="mean"):

        mask = target == ignored_index

        # mask_index = torch.any(target == ignored_index, dim=1)
        out = (input[~mask] - target[~mask]) ** 2
        if reduction == "mean":
            return out.mean()
        elif reduction == "None":
            return out