import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, Independent, kl_divergence
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer, TransformerDecoderLayer, TransformerDecoder

from StructDiffusion.models.encoders import EncoderMLP, DropoutSampler
from StructDiffusion.models.point_transformer import PointTransformerEncoderSmall
from StructDiffusion.utils.rotation_continuity import compute_rotation_matrix_from_ortho6d


class PriorContinuousOutEncoderDecoderStructPCT6DDropoutAllObjects(torch.nn.Module):
    """
    This model takes in point clouds of all objects
    """

    def __init__(self, vocab_size,
                 num_attention_heads=8, encoder_hidden_dim=16, encoder_dropout=0.1, encoder_activation="relu",
                 encoder_num_layers=8, structure_dropout=0.0, object_dropout=0.0, theta_loss_divide=None, ignore_rgb=False, ignore_other_objects=False):
        super(PriorContinuousOutEncoderDecoderStructPCT6DDropoutAllObjects, self).__init__()

        print("Transformer Encoder Decoder Struct with Point Transformer 6D All Objects")
        print("structure dropout", structure_dropout)
        print("object dropout:", object_dropout)
        print("theta loss divide:", theta_loss_divide)
        print("ignore rgb:", ignore_rgb)
        print("ignore other objects:", ignore_other_objects)

        self.theta_loss_divide = theta_loss_divide
        self.ignore_rgb = ignore_rgb
        self.ignore_other_objects = ignore_other_objects

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

        if not self.ignore_other_objects:
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
        if not self.ignore_other_objects:
            src_obj_embed = torch.cat([other_obj_pc_embed, obj_pc_embed], dim=1)
        else:
            src_obj_embed = obj_pc_embed

        #########################
        word_embed = self.word_embeddings(sentence)

        #########################
        position_embed = self.position_embeddings(position_index)
        token_type_embed = self.token_type_embeddings(token_type_index)
        if self.ignore_other_objects:
            position_embed = torch.cat([position_embed[:, 0:-num_target_objects-num_other_objects, :], position_embed[:, -num_target_objects:, :]], dim=1)
            token_type_embed = torch.cat([token_type_embed[:, 0:-num_target_objects - num_other_objects, :], token_type_embed[:, -num_target_objects:, :]], dim=1)

        struct_position_embed = self.position_embeddings(struct_position_index)
        struct_token_type_embed = self.token_type_embeddings(struct_token_type_index)

        src_sequence_encode = torch.cat([word_embed, src_obj_embed], dim=1)
        src_sequence_encode = torch.cat([src_sequence_encode, position_embed, token_type_embed], dim=-1)
        if not self.ignore_other_objects:
            src_pad_mask = torch.cat([sentence_pad_mask, other_object_pad_mask, object_pad_mask], dim=1)
        else:
            src_pad_mask = torch.cat([sentence_pad_mask, object_pad_mask], dim=1)

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


class ACTORVAE3Language(torch.nn.Module):

    def __init__(self, vocab_size,
                 num_attention_heads=8, encoder_hidden_dim=16, encoder_dropout=0.0, encoder_activation="relu",
                 encoder_num_layers=8, structure_dropout=0.0, object_dropout=0.0, theta_loss_weight=1, kl_loss_weight=1):
        super(ACTORVAE3Language, self).__init__()

        print("ACTOR VAE with 3 networks")
        print("object dropout:", object_dropout)
        print("structure dropout:", structure_dropout)
        print("transformer dropout:", encoder_dropout)
        print("theta loss weight:", theta_loss_weight)
        print("kl loss weight:", kl_loss_weight)

        self.theta_loss_weight = theta_loss_weight
        self.ignore_rgb = True
        self.kl_loss_weight = kl_loss_weight

        # -------------------------------------
        # base networks for extracting features
        self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=3, mean_center=True)
        self.mlp = EncoderMLP(256, 240, uses_pt=False)
        self.point_cloud_downscale = torch.nn.Linear(240, 120)

        self.position_encoder = nn.Sequential(nn.Linear(3 + 3 * 3, 120))
        self.start_token_embeddings = torch.nn.Embedding(1, 240)
        self.start_token_embeddings_encoder = torch.nn.Embedding(1, 120)
        self.mu_embeddings = torch.nn.Embedding(1, 256)
        self.sigma_embeddings = torch.nn.Embedding(1, 256)

        self.word_embeddings = torch.nn.Embedding(vocab_size, 240, padding_idx=0)
        # type sentence, other obj pc, target object pc, struct
        self.token_type_embeddings = torch.nn.Embedding(4, 8)
        # max number of objects or max length of sentence is 7
        self.position_embeddings = torch.nn.Embedding(7, 8)

        # -------------------------------------
        # recognition network
        recognition_layers = TransformerEncoderLayer(256, num_attention_heads,
                                                     encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.vae_recognition = TransformerEncoder(recognition_layers, encoder_num_layers)

        # -------------------------------------
        # prior network
        prior_layers = TransformerEncoderLayer(256, num_attention_heads,
                                               encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.vae_prior = TransformerEncoder(prior_layers, encoder_num_layers)

        # -------------------------------------
        # decoder network
        generation_layers = TransformerDecoderLayer(256, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.vae_generation = TransformerDecoder(generation_layers, encoder_num_layers)

        self.struct_dist = DropoutSampler(256, 3 + 6, dropout_rate=structure_dropout)
        self.obj_dist = DropoutSampler(256, 3 + 6, dropout_rate=object_dropout)

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

    def encode_pc(self, xyzs, rgbs, batch_size, num_objects):
        if self.ignore_rgb:
            center_xyz, x = self.object_encoder(xyzs, None)
        else:
            center_xyz, x = self.object_encoder(xyzs, rgbs)
        obj_pc_embed = self.mlp(x, center_xyz)
        obj_pc_embed = obj_pc_embed.reshape(batch_size, num_objects, -1)
        return obj_pc_embed

    def feature_forward(self, xyzs, rgbs, sentence,
                        obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs,
                        struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                        position_index, struct_position_index,
                        token_type_index, struct_token_type_index,
                        batch_size, num_target_objects):

        obj_pc_embed = self.encode_pc(xyzs, rgbs, batch_size, num_target_objects)

        obj_xytheta_inputs = torch.cat([obj_x_inputs.reshape(obj_x_inputs.shape[0], obj_x_inputs.shape[1], -1),
                                        obj_y_inputs.reshape(obj_y_inputs.shape[0], obj_y_inputs.shape[1], -1),
                                        obj_z_inputs.reshape(obj_z_inputs.shape[0], obj_z_inputs.shape[1], -1),
                                        obj_theta_inputs.reshape(obj_theta_inputs.shape[0], obj_theta_inputs.shape[1],
                                                                 -1)],
                                       dim=-1)
        struct_xytheta_inputs = torch.cat(
            [struct_x_inputs.reshape(struct_x_inputs.shape[0], struct_x_inputs.shape[1], -1),
             struct_y_inputs.reshape(struct_y_inputs.shape[0], struct_y_inputs.shape[1], -1),
             struct_z_inputs.reshape(struct_z_inputs.shape[0], struct_z_inputs.shape[1], -1),
             struct_theta_inputs.reshape(struct_theta_inputs.shape[0], struct_theta_inputs.shape[1], -1)],
            dim=-1)
        xytheta_embed = self.position_encoder(torch.cat([struct_xytheta_inputs, obj_xytheta_inputs], dim=1))

        word_embed = self.word_embeddings(sentence)

        position_embed = self.position_embeddings(position_index)
        struct_position_embed = self.position_embeddings(struct_position_index)
        token_type_embed = self.token_type_embeddings(token_type_index)
        struct_token_type_embed = self.token_type_embeddings(struct_token_type_index)

        return obj_pc_embed, word_embed, xytheta_embed, position_embed, struct_position_embed, token_type_embed, struct_token_type_embed

    def recognition_forward(self, obj_pc_embed, word_embed, xytheta_embed,
                                         position_embed, struct_position_embed,
                                         token_type_embed, struct_token_type_embed,
                                         object_pad_mask, struct_pad_mask, sentence_pad_mask,
                                         start_token,
                                         num_target_objects, num_words):

        # sequence: mu, sigma, word1, word2,... , struct, obj1, obj2, ...

        mu_embed = self.mu_embeddings(start_token)  # 256
        sigma_embed = self.sigma_embeddings(start_token)  # 256
        start_token_embed = self.start_token_embeddings_encoder(start_token)  # 120

        obj_pc_embed = self.point_cloud_downscale(obj_pc_embed)  # 120
        # append virtual structure frame
        obj_embed = torch.cat([start_token_embed, obj_pc_embed], dim=1)  # 120
        # add target positions of objects
        obj_embed = torch.cat([xytheta_embed, obj_embed], dim=-1)  # 120 + 120

        # append word
        word_obj_embed = torch.cat([word_embed, obj_embed], dim=1)  # 240

        position_embed = torch.cat([position_embed[:, :num_words, :], struct_position_embed, position_embed[:, -num_target_objects:, :]], dim=1)  # 8
        token_embed = torch.cat([token_type_embed[:, :num_words, :], struct_token_type_embed, token_type_embed[:, -num_target_objects:, :]], dim=1)  # 8
        word_obj_embed = torch.cat([word_obj_embed, position_embed, token_embed], dim=-1)  # 240 + 8 + 8

        src_sequence_encode = torch.cat([mu_embed, sigma_embed, word_obj_embed], dim=1)
        src_pad_mask = torch.cat([struct_pad_mask, struct_pad_mask, sentence_pad_mask, struct_pad_mask, object_pad_mask], dim=1)

        #########################
        # prepare sequence input for transformer
        # sequence_encode: [batch size, sequence_length, encoder input dimension]
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        src_sequence_encode = src_sequence_encode.transpose(1, 0)

        # convert to bool
        src_pad_mask = (src_pad_mask == 1)

        #########################
        # VAE encode
        # encode: [sequence_length, batch_size, embedding_size]
        encode = self.vae_recognition(src=src_sequence_encode, src_key_padding_mask=src_pad_mask)
        encode = encode.transpose(1, 0)

        mu = encode[:, 0]
        logvar = encode[:, 1]
        std = torch.exp(0.5 * logvar)
        dist = Independent(Normal(mu, std), 1)

        return dist

    def prior_forward(self, obj_pc_embed, word_embed,
                      position_embed,
                      token_type_embed,
                      object_pad_mask, struct_pad_mask, sentence_pad_mask,
                      start_token,
                      num_target_objects, num_words):

        # sequence: mu, sigma, word1, word2,... , obj1, obj2, ...

        mu_embed = self.mu_embeddings(start_token)  # 256
        sigma_embed = self.sigma_embeddings(start_token)  # 256

        # append word
        word_obj_embed = torch.cat([word_embed, obj_pc_embed], dim=1)  # 240
        position_embed = torch.cat([position_embed[:, :num_words, :], position_embed[:, -num_target_objects:, :]], dim=1)  # 8
        token_embed = torch.cat([token_type_embed[:, :num_words, :], token_type_embed[:, -num_target_objects:, :]],dim=1)  # 8
        word_obj_embed = torch.cat([word_obj_embed, position_embed, token_embed], dim=-1)  # 240 + 8 + 8

        src_sequence_encode = torch.cat([mu_embed, sigma_embed, word_obj_embed], dim=1)
        src_pad_mask = torch.cat([struct_pad_mask, struct_pad_mask, sentence_pad_mask, object_pad_mask], dim=1)

        #########################
        # prepare sequence input for transformer
        # sequence_encode: [batch size, sequence_length, encoder input dimension]
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        src_sequence_encode = src_sequence_encode.transpose(1, 0)

        # convert to bool
        src_pad_mask = (src_pad_mask == 1)

        #########################
        # VAE encode
        # encode: [sequence_length, batch_size, embedding_size]
        encode = self.vae_prior(src=src_sequence_encode, src_key_padding_mask=src_pad_mask)
        encode = encode.transpose(1, 0)

        mu = encode[:, 0]
        logvar = encode[:, 1]
        std = torch.exp(0.5 * logvar)
        dist = Independent(Normal(mu, std), 1)

        return dist

    def generation_forward(self, latent_code,
                          obj_pc_embed, word_embed,
                          position_embed, struct_position_embed,
                          token_type_embed, struct_token_type_embed,
                          object_pad_mask, struct_pad_mask, sentence_pad_mask,
                          start_token,
                          num_target_objects, num_words):

        # sequence: word1, word2,... , struct, obj1, obj2, ...
        start_token_embed = self.start_token_embeddings(start_token)  # 240
        word_obj_embed = torch.cat([word_embed, start_token_embed, obj_pc_embed], dim=1)  # 240

        position_embed = torch.cat([position_embed[:, :num_words, :], struct_position_embed, position_embed[:, -num_target_objects:, :]], dim=1)  # 8
        token_embed = torch.cat([token_type_embed[:, :num_words, :], struct_token_type_embed, token_type_embed[:, -num_target_objects:, :]], dim=1)  # 8
        word_obj_embed = torch.cat([word_obj_embed, position_embed, token_embed], dim=-1)  # 240 + 8 + 8

        tgt_sequence_encode = word_obj_embed
        tgt_pad_mask = torch.cat([struct_pad_mask, sentence_pad_mask, object_pad_mask], dim=1)

        #########################
        # prepare sequence input for transformer
        # sequence_encode: [batch size, sequence_length, encoder input dimension]
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        tgt_sequence_encode = tgt_sequence_encode.transpose(1, 0)

        # convert to bool
        tgt_pad_mask = (tgt_pad_mask == 1)

        # latent_code: [batch_size, embedding_size]
        latent_code = latent_code.unsqueeze(1)  # [batch_size, sequence_length=1, embedding_size]
        latent_code = latent_code.transpose(1, 0)

        encode = self.vae_generation(tgt=tgt_sequence_encode, memory=latent_code,
                                  tgt_key_padding_mask=tgt_pad_mask)
        encode = encode.transpose(1, 0)

        obj_encodes = encode[:, -num_target_objects:, :]
        obj_encodes = obj_encodes.reshape(-1, obj_encodes.shape[-1])
        # dim: [batch_size, num_features]
        obj_xyztheta_outputs = self.obj_dist(obj_encodes)

        struct_encodes = encode[:, -num_target_objects - 1, :]
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

    def forward(self, xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                sentence, sentence_pad_mask, token_type_index,
                obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index, tgt_mask, start_token,
                struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                struct_position_index, struct_token_type_index, struct_pad_mask):

        # token_type_index = [0] * (self.max_num_shape_parameters) + [1] * (self.max_num_other_objects) + [2] * self.max_num_objects
        # position_index = list(range(self.max_num_shape_parameters)) + list(range(self.max_num_other_objects)) + list(range(self.max_num_objects))
        # struct_position_index = [0]
        # struct_token_type_index = [3]
        # struct_pad_mask = [0]

        batch_size = object_pad_mask.shape[0]
        num_target_objects = object_pad_mask.shape[1]
        num_other_objects = other_object_pad_mask.shape[1]
        num_words = sentence_pad_mask.shape[1]

        # compute features
        obj_pc_embed, word_embed, xytheta_embed, position_embed, struct_position_embed, token_type_embed, struct_token_type_embed = self.feature_forward(
            xyzs, rgbs, sentence,
            obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs,
            struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
            position_index, struct_position_index,
            token_type_index, struct_token_type_index,
            batch_size, num_target_objects)

        # vae recognition
        recognition_dist = self.recognition_forward(obj_pc_embed, word_embed, xytheta_embed,
                                                 position_embed, struct_position_embed,
                                                 token_type_embed, struct_token_type_embed,
                                                 object_pad_mask, struct_pad_mask, sentence_pad_mask,
                                                 start_token,
                                                 num_target_objects, num_words)

        # vae prior
        prior_dist = self.prior_forward(obj_pc_embed, word_embed,
                                        position_embed,
                                        token_type_embed,
                                        object_pad_mask, struct_pad_mask, sentence_pad_mask,
                                        start_token,
                                        num_target_objects, num_words)

        if self.training:
            # reparameterize
            latent_code = recognition_dist.rsample()
        else:
            # ToDo: this can be from prior as well. We should evaluate samples both from prior and recognition
            latent_code = recognition_dist.sample()

        # vae generation
        predictions = self.generation_forward(latent_code,
                                          obj_pc_embed, word_embed,
                                          position_embed, struct_position_embed,
                                          token_type_embed, struct_token_type_embed,
                                          object_pad_mask, struct_pad_mask, sentence_pad_mask,
                                          start_token,
                                          num_target_objects, num_words)

        predictions["recognition_dist"] = recognition_dist
        predictions["prior_dist"] = prior_dist

        return predictions

    def sample_rearrangement(self, latent_code, xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                             sentence, sentence_pad_mask, token_type_index,
                             obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index, tgt_mask, start_token,
                             struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                             struct_position_index, struct_token_type_index, struct_pad_mask):

        batch_size = object_pad_mask.shape[0]
        num_target_objects = object_pad_mask.shape[1]
        num_other_objects = other_object_pad_mask.shape[1]
        num_words = sentence_pad_mask.shape[1]

        # compute features
        obj_pc_embed, word_embed, xytheta_embed, position_embed, struct_position_embed, token_type_embed, struct_token_type_embed = self.feature_forward(
            xyzs, rgbs, sentence,
            obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs,
            struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
            position_index, struct_position_index,
            token_type_index, struct_token_type_index,
            batch_size, num_target_objects)

        # vae decode
        predictions = self.generation_forward(latent_code,
                                          obj_pc_embed, word_embed,
                                          position_embed, struct_position_embed,
                                          token_type_embed, struct_token_type_embed,
                                          object_pad_mask, struct_pad_mask, sentence_pad_mask,
                                          start_token,
                                          num_target_objects, num_words)

        return predictions

    def get_prior_distribution(self, xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                               sentence, sentence_pad_mask, token_type_index,
                               obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index, tgt_mask, start_token,
                               struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                               struct_position_index, struct_token_type_index, struct_pad_mask):

        batch_size = object_pad_mask.shape[0]
        num_target_objects = object_pad_mask.shape[1]
        num_other_objects = other_object_pad_mask.shape[1]
        num_words = sentence_pad_mask.shape[1]

        # compute features
        obj_pc_embed, word_embed, xytheta_embed, position_embed, struct_position_embed, token_type_embed, struct_token_type_embed = self.feature_forward(
            xyzs, rgbs, sentence,
            obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs,
            struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
            position_index, struct_position_index,
            token_type_index, struct_token_type_index,
            batch_size, num_target_objects)

        prior_dist = self.prior_forward(obj_pc_embed, word_embed,
                                        position_embed,
                                        token_type_embed,
                                        object_pad_mask, struct_pad_mask, sentence_pad_mask,
                                        start_token,
                                        num_target_objects, num_words)

        return prior_dist

    def criterion(self, predictions, labels):

        loss = 0
        for key in predictions:
            if key == "recognition_dist" or key == "prior_dist":
                continue

            preds = predictions[key]
            gts = labels[key]

            if "theta" in key:
                loss += self.mse_loss(preds, gts) * self.theta_loss_weight
            else:
                loss += self.mse_loss(preds, gts)

        # probably need to add a constant to balance loss
        KLD = torch.mean(kl_divergence(predictions["recognition_dist"], predictions["prior_dist"]), dim=-1)
        loss += self.kl_loss_weight * KLD

        return loss

    def mse_loss(self, input, target, ignored_index=-100, reduction="mean"):

        mask = target == ignored_index

        # mask_index = torch.any(target == ignored_index, dim=1)
        out = (input[~mask] - target[~mask]) ** 2
        if reduction == "mean":
            return out.mean()
        elif reduction == "None":
            return out