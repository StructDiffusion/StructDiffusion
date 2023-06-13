import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from StructDiffusion.models.encoders import EncoderMLP, DropoutSampler
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from StructDiffusion.models.point_transformer import PointTransformerEncoderSmall



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TransformerDiffuser(torch.nn.Module):
    """
    This model takes in point clouds of all objects
    """

    def __init__(self, num_attention_heads=8, encoder_hidden_dim=16, encoder_dropout=0.1, encoder_activation="relu",
                 encoder_num_layers=8, structure_dropout=0.0, object_dropout=0.0, ignore_rgb=False):
        super(TransformerDiffuser, self).__init__()

        print("Transformer Decoder Struct with Point Transformer 6D All Objects")
        print("ignore rgb:", ignore_rgb)

        self.ignore_rgb = ignore_rgb

        # object encode will have dim 256
        if ignore_rgb:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=3, mean_center=True)
        else:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=True)

        # 256 = 80 (point cloud) + 80 (position) + 80 (time) + 16 (position idx)
        # Important: we set uses_pt to true because we want the model to consider the positions of objects that
        #  don't need to be rearranged.
        self.mlp = EncoderMLP(256, 80, uses_pt=True)
        self.position_encoder = nn.Sequential(nn.Linear(3 + 6, 80))
        self.start_token_embeddings = torch.nn.Embedding(1, 80)

        # max number of objects or max length of sentence is 7
        self.position_embeddings = torch.nn.Embedding(7, 16)

        print(num_attention_heads)
        print(encoder_hidden_dim)
        print(encoder_dropout)
        print(encoder_activation)

        encoder_layers = TransformerEncoderLayer(256, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(80),
            nn.Linear(80, 80),
            nn.GELU(),
            nn.Linear(80, 80),
        )

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

    def forward(self, t, xyzs, obj_xyztheta_inputs, struct_xyztheta_inputs,
                position_index, struct_position_index, start_token):

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

        #########################
        xyzs = xyzs.reshape(batch_size * num_target_objects, num_pts, -1)
        obj_pc_embed = self.encode_pc(xyzs, None, batch_size, num_target_objects)

        xytheta_embed = self.position_encoder(torch.cat([struct_xyztheta_inputs, obj_xyztheta_inputs], dim=1))

        # at this point, obj_pc_embed has size [batch size, num objs, 240], obj_xytheta_embed [batch size, num objs, 120]
        # combine them into [batch size, num objs, 240]
        # we need to shift obj_xytheta_embed to the right by one position and add a start token
        start_token_embed = self.start_token_embeddings(start_token)

        tgt_obj_embed = torch.cat([start_token_embed, obj_pc_embed], dim=1)
        tgt_obj_embed = torch.cat([xytheta_embed, tgt_obj_embed], dim=-1)

        #########################
        time_embed = self.time_mlp(t)  # B, dim
        time_embed = time_embed.unsqueeze(1).repeat(1, num_target_objects + 1, 1)  # B, N, dim
        position_embed = self.position_embeddings(position_index)
        struct_position_embed = self.position_embeddings(struct_position_index)

        tgt_position_embed = torch.cat([struct_position_embed, position_embed[:, -num_target_objects:, :]], dim=1)
        tgt_sequence_encode = torch.cat([time_embed, tgt_obj_embed, tgt_position_embed], dim=-1)

        #########################
        # sequence_encode: [batch size, sequence_length, encoder input dimension]
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        tgt_sequence_encode = tgt_sequence_encode.transpose(1, 0)

        # encode: [sequence_length, batch_size, embedding_size]
        encode = self.encoder(tgt_sequence_encode)
        encode = encode.transpose(1, 0)
        #########################
        obj_encodes = encode[:, 1:, :]
        obj_xyztheta_outputs = self.obj_dist(obj_encodes)  # B, N, 3 + 6

        struct_encodes = encode[:, 0, :].unsqueeze(1)  # B, 1, 3 + 6
        # use a different sampler for struct prediction since it should have larger variance than object predictions
        struct_xyztheta_outputs = self.struct_dist(struct_encodes)

        # predictions = {"obj_xyztheta_outputs": obj_xyztheta_outputs,
        #                "struct_xyztheta_outputs": struct_xyztheta_outputs}

        return struct_xyztheta_outputs, obj_xyztheta_outputs


class TransformerDiffuserVariableNumObjs(torch.nn.Module):
    """
    This model takes in point clouds of all objects
    """

    def __init__(self, num_attention_heads=8, encoder_hidden_dim=16, encoder_dropout=0.1, encoder_activation="relu",
                 encoder_num_layers=8, structure_dropout=0.0, object_dropout=0.0, ignore_rgb=False):
        super(TransformerDiffuserVariableNumObjs, self).__init__()

        print("Transformer Decoder Struct with Point Transformer 6D All Objects")
        print("ignore rgb:", ignore_rgb)

        self.ignore_rgb = ignore_rgb

        # object encode will have dim 256
        if ignore_rgb:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=3, mean_center=True)
        else:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=True)

        # 256 = 80 (point cloud) + 80 (position) + 80 (time) + 16 (position idx)
        # Important: we set uses_pt to true because we want the model to consider the positions of objects that
        #  don't need to be rearranged.
        self.mlp = EncoderMLP(256, 80, uses_pt=True)
        self.position_encoder = nn.Sequential(nn.Linear(3 + 6, 80))
        self.start_token_embeddings = torch.nn.Embedding(1, 80)

        # max number of objects or max length of sentence is 7
        self.position_embeddings = torch.nn.Embedding(7, 16)

        print(num_attention_heads)
        print(encoder_hidden_dim)
        print(encoder_dropout)
        print(encoder_activation)

        encoder_layers = TransformerEncoderLayer(256, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(80),
            nn.Linear(80, 80),
            nn.GELU(),
            nn.Linear(80, 80),
        )

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

    def forward(self, t, xyzs, obj_xyztheta_inputs, struct_xyztheta_inputs,
                position_index, struct_position_index, start_token, object_pad_mask, struct_pad_mask):

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

        #########################
        xyzs = xyzs.reshape(batch_size * num_target_objects, num_pts, -1)
        obj_pc_embed = self.encode_pc(xyzs, None, batch_size, num_target_objects)

        xytheta_embed = self.position_encoder(torch.cat([struct_xyztheta_inputs, obj_xyztheta_inputs], dim=1))

        # at this point, obj_pc_embed has size [batch size, num objs, 240], obj_xytheta_embed [batch size, num objs, 120]
        # combine them into [batch size, num objs, 240]
        # we need to shift obj_xytheta_embed to the right by one position and add a start token
        start_token_embed = self.start_token_embeddings(start_token)

        tgt_obj_embed = torch.cat([start_token_embed, obj_pc_embed], dim=1)
        tgt_obj_embed = torch.cat([xytheta_embed, tgt_obj_embed], dim=-1)

        #########################
        time_embed = self.time_mlp(t)  # B, dim
        time_embed = time_embed.unsqueeze(1).repeat(1, num_target_objects + 1, 1)  # B, 1 + N, dim
        position_embed = self.position_embeddings(position_index)
        struct_position_embed = self.position_embeddings(struct_position_index)

        tgt_position_embed = torch.cat([struct_position_embed, position_embed[:, -num_target_objects:, :]], dim=1)
        tgt_sequence_encode = torch.cat([time_embed, tgt_obj_embed, tgt_position_embed], dim=-1)
        tgt_pad_mask = torch.cat([struct_pad_mask, object_pad_mask], dim=1)  # B, 1 + N

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
        obj_encodes = encode[:, 1:, :]
        obj_xyztheta_outputs = self.obj_dist(obj_encodes)  # B, N, 3 + 6

        struct_encodes = encode[:, 0, :].unsqueeze(1)  # B, 1, 3 + 6
        # use a different sampler for struct prediction since it should have larger variance than object predictions
        struct_xyztheta_outputs = self.struct_dist(struct_encodes)

        # predictions = {"obj_xyztheta_outputs": obj_xyztheta_outputs,
        #                "struct_xyztheta_outputs": struct_xyztheta_outputs}

        return struct_xyztheta_outputs, obj_xyztheta_outputs


class TransformerDiffuserLang(torch.nn.Module):
    """
    This model takes in point clouds of all objects
    """

    def __init__(self, vocab_size, num_attention_heads=8, encoder_hidden_dim=16, encoder_dropout=0.1, encoder_activation="relu",
                 encoder_num_layers=8, structure_dropout=0.0, object_dropout=0.0, ignore_rgb=False):
        super(TransformerDiffuserLang, self).__init__()

        print("Transformer Decoder Struct with Point Transformer 6D All Objects")
        print("ignore rgb:", ignore_rgb)

        self.ignore_rgb = ignore_rgb

        # object encode will have dim 256
        if ignore_rgb:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=3, mean_center=True)
        else:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=True)

        # 256 = 80 (point cloud) + 80 (position) + 80 (time) + 8 (position idx) + 8 (token idx)
        # 256 = 160 (word embedding) + 80 (time) + 8 (position idx) + 8 (token idx)
        # Important: we set uses_pt to true because we want the model to consider the positions of objects that
        #  don't need to be rearranged.
        self.mlp = EncoderMLP(256, 80, uses_pt=True)
        self.position_encoder = nn.Sequential(nn.Linear(3 + 6, 80))
        self.start_token_embeddings = torch.nn.Embedding(1, 80)

        # max number of objects or max length of sentence is 7
        self.position_embeddings = torch.nn.Embedding(7, 8)
        self.token_type_embeddings = torch.nn.Embedding(4, 8)

        self.word_embeddings = torch.nn.Embedding(vocab_size, 160, padding_idx=0)

        print(num_attention_heads)
        print(encoder_hidden_dim)
        print(encoder_dropout)
        print(encoder_activation)

        encoder_layers = TransformerEncoderLayer(256, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(80),
            nn.Linear(80, 80),
            nn.GELU(),
            nn.Linear(80, 80),
        )

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

        #########################
        xyzs = xyzs.reshape(batch_size * num_target_objects, num_pts, -1)
        obj_pc_embed = self.encode_pc(xyzs, None, batch_size, num_target_objects)

        xytheta_embed = self.position_encoder(torch.cat([struct_xyztheta_inputs, obj_xyztheta_inputs], dim=1))

        # at this point, obj_pc_embed has size [batch size, num objs, 80], obj_xytheta_embed [batch size, num objs, 80]
        # combine them into [batch size, num objs, 160]
        # we need to shift obj_xytheta_embed to the right by one position and add a start token
        start_token_embed = self.start_token_embeddings(start_token)

        tgt_obj_embed = torch.cat([start_token_embed, obj_pc_embed], dim=1)
        tgt_obj_embed = torch.cat([xytheta_embed, tgt_obj_embed], dim=-1)

        #########################
        sentence_embed = self.word_embeddings(sentence)

        #########################

        # transformer time dim: sentence, struct, obj
        # transformer feat dim: obj pc + pose / word, time, token type, position

        time_embed = self.time_mlp(t)  # B, dim
        time_embed = time_embed.unsqueeze(1).repeat(1, sentence_len + 1 + num_target_objects, 1)  # B, L + 1 + N, dim
        position_embed = self.position_embeddings(torch.cat([position_index[:, :sentence_len], struct_position_index, position_index[:, -num_target_objects:]], dim=1))
        token_type_embed = self.token_type_embeddings(torch.cat([token_type_index[:, :sentence_len], struct_token_type_index, token_type_index[:, -num_target_objects:]], dim=1))

        tgt_sequence_encode = torch.cat([sentence_embed, tgt_obj_embed], dim=1)
        tgt_sequence_encode = torch.cat([tgt_sequence_encode, time_embed, position_embed, token_type_embed], dim=-1)
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
        obj_xyztheta_outputs = self.obj_dist(obj_encodes)  # B, N, 3 + 6

        struct_encodes = encode[:, sentence_len, :].unsqueeze(1)  # B, 1, 3 + 6
        # use a different sampler for struct prediction since it should have larger variance than object predictions
        struct_xyztheta_outputs = self.struct_dist(struct_encodes)

        # predictions = {"obj_xyztheta_outputs": obj_xyztheta_outputs,
        #                "struct_xyztheta_outputs": struct_xyztheta_outputs}

        return struct_xyztheta_outputs, obj_xyztheta_outputs



class TransformerDiffuserLangSentenceEmb(torch.nn.Module):
    """
    This model takes in point clouds of all objects
    """

    def __init__(self, sentence_embedding_size, num_attention_heads=8, encoder_hidden_dim=16, encoder_dropout=0.1, encoder_activation="relu",
                 encoder_num_layers=8, structure_dropout=0.0, object_dropout=0.0, ignore_rgb=False):
        super(TransformerDiffuserLangSentenceEmb, self).__init__()

        print("Transformer Decoder Struct with Point Transformer 6D All Objects")
        print("ignore rgb:", ignore_rgb)

        self.ignore_rgb = ignore_rgb

        # object encode will have dim 256
        if ignore_rgb:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=3, mean_center=True)
        else:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=True)

        # 256 = 80 (point cloud) + 80 (position) + 80 (time) + 8 (position idx) + 8 (token idx)
        # 256 = 160 (word embedding) + 80 (time) + 8 (position idx) + 8 (token idx)
        # Important: we set uses_pt to true because we want the model to consider the positions of objects that
        #  don't need to be rearranged.
        self.mlp = EncoderMLP(256, 80, uses_pt=True)
        self.position_encoder = nn.Sequential(nn.Linear(3 + 6, 80))
        self.start_token_embeddings = torch.nn.Embedding(1, 80)

        # max number of objects or max length of sentence is 7
        self.position_embeddings = torch.nn.Embedding(7, 8)
        self.token_type_embeddings = torch.nn.Embedding(4, 8)

        self.sentence_down_sample = torch.nn.Linear(sentence_embedding_size, 160)

        print(num_attention_heads)
        print(encoder_hidden_dim)
        print(encoder_dropout)
        print(encoder_activation)

        encoder_layers = TransformerEncoderLayer(256, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(80),
            nn.Linear(80, 80),
            nn.GELU(),
            nn.Linear(80, 80),
        )

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

    def forward(self, t, xyzs, obj_xyztheta_inputs, struct_xyztheta_inputs, sentence_embedding,
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
        sentence_len = 1

        #########################
        xyzs = xyzs.reshape(batch_size * num_target_objects, num_pts, -1)
        obj_pc_embed = self.encode_pc(xyzs, None, batch_size, num_target_objects)

        xytheta_embed = self.position_encoder(torch.cat([struct_xyztheta_inputs, obj_xyztheta_inputs], dim=1))

        # at this point, obj_pc_embed has size [batch size, num objs, 80], obj_xytheta_embed [batch size, num objs, 80]
        # combine them into [batch size, num objs, 160]
        # we need to shift obj_xytheta_embed to the right by one position and add a start token
        start_token_embed = self.start_token_embeddings(start_token)

        tgt_obj_embed = torch.cat([start_token_embed, obj_pc_embed], dim=1)
        tgt_obj_embed = torch.cat([xytheta_embed, tgt_obj_embed], dim=-1)

        #########################
        sentence_embed = self.sentence_down_sample(sentence_embedding)

        #########################

        # transformer time dim: sentence, struct, obj
        # transformer feat dim: obj pc + pose / word, time, token type, position

        time_embed = self.time_mlp(t)  # B, dim
        time_embed = time_embed.unsqueeze(1).repeat(1, sentence_len + 1 + num_target_objects, 1)  # B, L + 1 + N, dim
        position_embed = self.position_embeddings(torch.cat([position_index[:, :sentence_len], struct_position_index, position_index[:, -num_target_objects:]], dim=1))
        token_type_embed = self.token_type_embeddings(torch.cat([token_type_index[:, :sentence_len], struct_token_type_index, token_type_index[:, -num_target_objects:]], dim=1))

        tgt_sequence_encode = torch.cat([sentence_embed, tgt_obj_embed], dim=1)
        tgt_sequence_encode = torch.cat([tgt_sequence_encode, time_embed, position_embed, token_type_embed], dim=-1)
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
        obj_xyztheta_outputs = self.obj_dist(obj_encodes)  # B, N, 3 + 6

        struct_encodes = encode[:, sentence_len, :].unsqueeze(1)  # B, 1, 3 + 6
        # use a different sampler for struct prediction since it should have larger variance than object predictions
        struct_xyztheta_outputs = self.struct_dist(struct_encodes)

        # predictions = {"obj_xyztheta_outputs": obj_xyztheta_outputs,
        #                "struct_xyztheta_outputs": struct_xyztheta_outputs}

        return struct_xyztheta_outputs, obj_xyztheta_outputs


class TransformerNoisedClassifier(torch.nn.Module):
    """
    This model takes in point clouds of all objects
    """

    def __init__(self, vocab_size, num_attention_heads=8, encoder_hidden_dim=16, encoder_dropout=0.1, encoder_activation="relu",
                 encoder_num_layers=8, ignore_rgb=False, final_mean_embedding=False):
        super(TransformerNoisedClassifier, self).__init__()

        print("Transformer Decoder Struct with Point Transformer 6D All Objects")
        print("ignore rgb:", ignore_rgb)

        self.ignore_rgb = ignore_rgb
        self.final_mean_embedding = final_mean_embedding

        # object encode will have dim 256
        if ignore_rgb:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=3, mean_center=True)
        else:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=True)

        # 256 = 80 (point cloud) + 80 (position) + 80 (time) + 8 (position idx) + 8 (token idx)
        # 256 = 160 (word embedding) + 80 (time) + 8 (position idx) + 8 (token idx)
        # Important: we set uses_pt to true because we want the model to consider the positions of objects that
        #  don't need to be rearranged.
        self.mlp = EncoderMLP(256, 80, uses_pt=True)
        self.position_encoder = nn.Sequential(nn.Linear(3 + 6, 80))
        self.start_token_embeddings = torch.nn.Embedding(1, 80)

        # max number of objects or max length of sentence is 7
        self.position_embeddings = torch.nn.Embedding(7, 8)
        self.token_type_embeddings = torch.nn.Embedding(4, 8)

        self.word_embeddings = torch.nn.Embedding(vocab_size, 160, padding_idx=0)

        print(num_attention_heads)
        print(encoder_hidden_dim)
        print(encoder_dropout)
        print(encoder_activation)

        encoder_layers = TransformerEncoderLayer(256, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(80),
            nn.Linear(80, 80),
            nn.GELU(),
            nn.Linear(80, 80),
        )

        self.pred_fc1 = nn.Linear(256, 256)
        self.pred_fc2 = nn.Linear(256, 1)

    def encode_pc(self, xyzs, rgbs, batch_size, num_objects):
        if self.ignore_rgb:
            center_xyz, x = self.object_encoder(xyzs, None)
        else:
            center_xyz, x = self.object_encoder(xyzs, rgbs)
        obj_pc_embed = self.mlp(x, center_xyz)
        obj_pc_embed = obj_pc_embed.reshape(batch_size, num_objects, -1)
        return obj_pc_embed

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

        #########################
        xyzs = xyzs.reshape(batch_size * num_target_objects, num_pts, -1)
        obj_pc_embed = self.encode_pc(xyzs, None, batch_size, num_target_objects)

        xytheta_embed = self.position_encoder(torch.cat([struct_xyztheta_inputs, obj_xyztheta_inputs], dim=1))

        # at this point, obj_pc_embed has size [batch size, num objs, 80], obj_xytheta_embed [batch size, num objs, 80]
        # combine them into [batch size, num objs, 160]
        # we need to shift obj_xytheta_embed to the right by one position and add a start token
        start_token_embed = self.start_token_embeddings(start_token)

        tgt_obj_embed = torch.cat([start_token_embed, obj_pc_embed], dim=1)
        tgt_obj_embed = torch.cat([xytheta_embed, tgt_obj_embed], dim=-1)

        #########################
        sentence_embed = self.word_embeddings(sentence)

        #########################

        # transformer time dim: sentence, struct, obj
        # transformer feat dim: obj pc + pose / word, time, token type, position

        time_embed = self.time_mlp(t)  # B, dim
        time_embed = time_embed.unsqueeze(1).repeat(1, sentence_len + 1 + num_target_objects, 1)  # B, L + 1 + N, dim
        position_embed = self.position_embeddings(torch.cat([position_index[:, :sentence_len], struct_position_index, position_index[:, -num_target_objects:]], dim=1))
        token_type_embed = self.token_type_embeddings(torch.cat([token_type_index[:, :sentence_len], struct_token_type_index, token_type_index[:, -num_target_objects:]], dim=1))

        tgt_sequence_encode = torch.cat([sentence_embed, tgt_obj_embed], dim=1)
        tgt_sequence_encode = torch.cat([tgt_sequence_encode, time_embed, position_embed, token_type_embed], dim=-1)
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

        if self.final_mean_embedding:
            # Take an average pool
            encode = encode.mean(dim=1)
        else:
            # Take from the structure position
            encode = encode[:, sentence_len]

        h = F.relu(self.pred_fc1(encode))
        logit = self.pred_fc2(h)

        logit = F.sigmoid(logit)

        return logit