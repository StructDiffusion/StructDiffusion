import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer

from StructDiffusion.models.encoders import EncoderMLP
from StructDiffusion.models.point_transformer_large import PointTransformerCls
from StructDiffusion.models.point_transformer import PointTransformerEncoderSmall


class FocalLoss(nn.Module):
    "Focal Loss"

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


class DiscriminatorWholeScene(torch.nn.Module):

    """
    This circle predictor uses point clouds of objects to predict whether they form a circle
    """

    def __init__(self, max_num_objects, use_focal_loss=False, focal_loss_gamma=2, use_regression_loss=False,
                 include_env_pc=False, pct_random_sampling=False):

        super(DiscriminatorWholeScene, self).__init__()

        # just to ensure that we specify this parameter in omega config
        assert pct_random_sampling is not None

        # input_dim: xyz + one hot for each object
        if include_env_pc:
            self.classifier = PointTransformerCls(input_dim=max_num_objects + 1 + 3, output_dim=1, use_random_sampling=pct_random_sampling)
        else:
            self.classifier = PointTransformerCls(input_dim=max_num_objects + 3, output_dim=1, use_random_sampling=pct_random_sampling)

        ###########################
        assert int(use_regression_loss) + int(use_focal_loss) <= 1

        self.use_regression_loss = use_regression_loss
        if use_focal_loss:
            print("use focal loss with gamma {}".format(focal_loss_gamma))
            self.loss = FocalLoss(gamma=focal_loss_gamma)
        elif use_regression_loss:
            print("use regression L2 loss")
            self.loss = torch.nn.MSELoss()
        else:
            print("use standard BCE logit loss")
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, scene_xyz):

        is_circle = self.classifier(scene_xyz)

        is_circle = is_circle.squeeze(dim=1)

        predictions = {"is_circle": is_circle}

        return predictions

    def criterion(self, predictions, labels):

        if self.use_regression_loss:
            predictions = self.convert_logits(predictions)

        loss = 0
        for key in predictions:

            preds = predictions[key]
            gts = labels[key]

            loss += self.loss(preds, gts)

        return loss

    def convert_logits(self, predictions):

        for key in predictions:
            if key == "is_circle":
                predictions[key] = torch.sigmoid(predictions[key])

        return predictions


class DiscriminatorWholeSceneMultihead(torch.nn.Module):

    """
    This circle predictor uses point clouds of objects to predict whether they form a circle
    """

    def __init__(self, max_num_objects, use_focal_loss=False, focal_loss_gamma=2, use_regression_loss=False,
                 include_env_pc=False, pct_random_sampling=False, num_output_heads=4):

        super(DiscriminatorWholeSceneMultihead, self).__init__()

        # just to ensure that we specify this parameter in omega config
        assert pct_random_sampling is not None

        # input_dim: xyz + one hot for each object
        if include_env_pc:
            self.classifier = PointTransformerCls(input_dim=max_num_objects + 1 + 3, output_dim=num_output_heads, use_random_sampling=pct_random_sampling)
        else:
            self.classifier = PointTransformerCls(input_dim=max_num_objects + 3, output_dim=num_output_heads, use_random_sampling=pct_random_sampling)

        ###########################
        assert int(use_regression_loss) + int(use_focal_loss) <= 1

        self.use_regression_loss = use_regression_loss
        if use_focal_loss:
            print("use focal loss with gamma {}".format(focal_loss_gamma))
            self.loss = FocalLoss(gamma=focal_loss_gamma)
        elif use_regression_loss:
            print("use regression L2 loss")
            self.loss = torch.nn.MSELoss()
        else:
            print("use standard BCE logit loss")
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, scene_xyz, structure_type_index):

        # B, 1

        is_circle = self.classifier(scene_xyz)  # B, 4
        # https://medium.com/@mbednarski/understanding-indexing-with-pytorch-gather-33717a84ebc4
        is_circle = torch.gather(is_circle, dim=1, index=structure_type_index)  # B, 1

        is_circle = is_circle.squeeze(dim=1)

        predictions = {"is_circle": is_circle}

        return predictions

    def criterion(self, predictions, labels):

        if self.use_regression_loss:
            predictions = self.convert_logits(predictions)

        loss = 0
        for key in predictions:

            preds = predictions[key]
            gts = labels[key]

            loss += self.loss(preds, gts)

        return loss

    def convert_logits(self, predictions):

        for key in predictions:
            if key == "is_circle":
                predictions[key] = torch.sigmoid(predictions[key])

        return predictions


class DiscriminatorWholeSceneLanguageMLPFusionBN(torch.nn.Module):

    """
    This circle predictor uses point clouds of objects to predict whether they form a circle
    """

    def __init__(self, vocab_size, max_num_objects, use_focal_loss=False, focal_loss_gamma=2, use_regression_loss=False,
                 include_env_pc=False, pct_random_sampling=False,
                 # transformer params below
                 num_attention_heads=4, encoder_hidden_dim=16, encoder_dropout=0.1,
                 encoder_activation="relu", encoder_num_layers=4,
                 ):

        super(DiscriminatorWholeSceneLanguageMLPFusionBN, self).__init__()

        # just to ensure that we specify this parameter in omega config
        assert pct_random_sampling is not None

        # input_dim: xyz + one hot for each object
        if include_env_pc:
            self.classifier = PointTransformerCls(input_dim=max_num_objects + 1 + 3, output_dim=256, use_random_sampling=pct_random_sampling)
        else:
            self.classifier = PointTransformerCls(input_dim=max_num_objects + 3, output_dim=256, use_random_sampling=pct_random_sampling)

        ###########################

        self.word_embeddings = torch.nn.Embedding(vocab_size, 240, padding_idx=0)
        # max number of objects or max length of sentence is 7
        self.position_embeddings = torch.nn.Embedding(7, 16)

        encoder_layers = TransformerEncoderLayer(256, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)

        ###########################
        self.final_fc = nn.Sequential(nn.Linear(256 * 2, 512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(),
                                       nn.Linear(512, 512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(),
                                       nn.Linear(512, 1))

        ###########################
        assert int(use_regression_loss) + int(use_focal_loss) <= 1

        self.use_regression_loss = use_regression_loss
        if use_focal_loss:
            print("use focal loss with gamma {}".format(focal_loss_gamma))
            self.loss = FocalLoss(gamma=focal_loss_gamma)
        elif use_regression_loss:
            print("use regression L2 loss")
            self.loss = torch.nn.MSELoss()
        else:
            print("use standard BCE logit loss")
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, scene_xyz, sentence, sentence_pad_mask, position_index):

        scene_pc_embed = self.classifier(scene_xyz)

        ############################################
        # encode language into a fixed dimension embedding
        sequence_encode = torch.cat([self.word_embeddings(sentence), self.position_embeddings(position_index)], dim=-1)

        # sequence_encode: [batch size, sequence_length, encoder input dimension]
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        sequence_encode = sequence_encode.transpose(1, 0)

        # convert to bool
        pad_mask = (sentence_pad_mask == 1)

        # encode: [sequence_length, batch_size, embedding_size]
        encode = self.encoder(sequence_encode, src_key_padding_mask=pad_mask)
        encode = encode.transpose(1, 0)

        valid_encode_mask = 1 - sentence_pad_mask
        # mean pooling from all positions for each sequence: [batch_size, embedding_size]
        mean_encode = torch.mean(encode * valid_encode_mask.unsqueeze(-1), dim=1)

        #############################################

        is_circle = self.final_fc(torch.cat([mean_encode, scene_pc_embed], dim=-1))

        is_circle = is_circle.squeeze(dim=1)

        predictions = {"is_circle": is_circle}

        return predictions

    def criterion(self, predictions, labels):

        if self.use_regression_loss:
            predictions = self.convert_logits(predictions)

        loss = 0
        for key in predictions:

            preds = predictions[key]
            gts = labels[key]

            loss += self.loss(preds, gts)

        return loss

    def convert_logits(self, predictions):

        for key in predictions:
            if key == "is_circle":
                predictions[key] = torch.sigmoid(predictions[key])

        return predictions


class DiscriminatorWholeSceneLanguage(torch.nn.Module):

    """
    This circle predictor uses point clouds of objects to predict whether they form a circle
    """

    def __init__(self, vocab_size, max_num_objects, use_focal_loss=False, focal_loss_gamma=2, use_regression_loss=False,
                 include_env_pc=False, pct_random_sampling=False,
                 # transformer params below
                 num_attention_heads=4, encoder_hidden_dim=16, encoder_dropout=0.1,
                 encoder_activation="relu", encoder_num_layers=4,
                 ):

        super(DiscriminatorWholeSceneLanguage, self).__init__()

        # just to ensure that we specify this parameter in omega config
        assert pct_random_sampling is not None

        # input_dim: xyz + one hot for each object
        if include_env_pc:
            self.classifier = PointTransformerCls(input_dim=max_num_objects + 1 + 3, output_dim=256, use_random_sampling=pct_random_sampling)
        else:
            self.classifier = PointTransformerCls(input_dim=max_num_objects + 3, output_dim=256, use_random_sampling=pct_random_sampling)

        ###########################

        self.word_embeddings = torch.nn.Embedding(vocab_size, 240, padding_idx=0)
        # max number of objects or max length of sentence is 7
        self.position_embeddings = torch.nn.Embedding(7, 16)

        encoder_layers = TransformerEncoderLayer(256, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)

        ###########################
        self.final_fc = nn.Sequential(nn.Linear(256 * 2, 256),
                                       nn.LayerNorm(256),
                                       nn.ReLU(),
                                       nn.Linear(256, 128),
                                       nn.LayerNorm(128),
                                       nn.ReLU(),
                                       nn.Linear(128, 1))

        ###########################
        assert int(use_regression_loss) + int(use_focal_loss) <= 1

        self.use_regression_loss = use_regression_loss
        if use_focal_loss:
            print("use focal loss with gamma {}".format(focal_loss_gamma))
            self.loss = FocalLoss(gamma=focal_loss_gamma)
        elif use_regression_loss:
            print("use regression L2 loss")
            self.loss = torch.nn.MSELoss()
        else:
            print("use standard BCE logit loss")
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, scene_xyz, sentence, sentence_pad_mask, position_index):

        scene_pc_embed = self.classifier(scene_xyz)

        ############################################
        # encode language into a fixed dimension embedding
        sequence_encode = torch.cat([self.word_embeddings(sentence), self.position_embeddings(position_index)], dim=-1)

        # sequence_encode: [batch size, sequence_length, encoder input dimension]
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        sequence_encode = sequence_encode.transpose(1, 0)

        # convert to bool
        pad_mask = (sentence_pad_mask == 1)

        # encode: [sequence_length, batch_size, embedding_size]
        encode = self.encoder(sequence_encode, src_key_padding_mask=pad_mask)
        encode = encode.transpose(1, 0)

        valid_encode_mask = 1 - sentence_pad_mask
        # mean pooling from all positions for each sequence: [batch_size, embedding_size]
        mean_encode = torch.mean(encode * valid_encode_mask.unsqueeze(-1), dim=1)

        #############################################

        is_circle = self.final_fc(torch.cat([mean_encode, scene_pc_embed], dim=-1))

        is_circle = is_circle.squeeze(dim=1)

        predictions = {"is_circle": is_circle}

        return predictions

    def criterion(self, predictions, labels):

        if self.use_regression_loss:
            predictions = self.convert_logits(predictions)

        loss = 0
        for key in predictions:

            preds = predictions[key]
            gts = labels[key]

            loss += self.loss(preds, gts)

        return loss

    def convert_logits(self, predictions):

        for key in predictions:
            if key == "is_circle":
                predictions[key] = torch.sigmoid(predictions[key])

        return predictions


class DiscriminatorWholeSceneObject(torch.nn.Module):

    """
    This circle predictor uses point clouds of objects to predict whether they form a circle
    """

    def __init__(self, max_num_objects, use_focal_loss=False, focal_loss_gamma=2, use_regression_loss=False, include_env_pc=False,
                 num_attention_heads=4, encoder_hidden_dim=32, encoder_dropout=0.1, encoder_activation="relu",
                 encoder_num_layers=4, pct_random_sampling=False):

        super(DiscriminatorWholeSceneObject, self).__init__()

        # just to ensure that we specify this parameter in omega config
        assert pct_random_sampling is not None

        self.ignore_rgb = True

        # encoding whole scene
        # input_dim: xyz + one hot for each object (+ env background)
        if include_env_pc:
            self.scene_encoder = PointTransformerCls(input_dim=max_num_objects + 1 + 3, output_dim=256, use_random_sampling=pct_random_sampling)
        else:
            self.scene_encoder = PointTransformerCls(input_dim=max_num_objects + 3, output_dim=256, use_random_sampling=pct_random_sampling)

        # encoding each object
        self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=3, mean_center=True)
        self.mlp = EncoderMLP(256, 240, uses_pt=True)

        ###########################
        self.query_embeddings = torch.nn.Embedding(2, 8)
        self.position_embeddings = torch.nn.Embedding(max_num_objects, 8)

        ###########################
        self.encoder = Transformer(d_model=256,
                                   nhead=num_attention_heads,
                                   num_encoder_layers=encoder_num_layers,
                                   num_decoder_layers=encoder_num_layers,
                                   dim_feedforward=encoder_hidden_dim,
                                   dropout=encoder_dropout)

        self.final_fc = nn.Sequential(nn.Linear(256, 256),
                                       nn.LayerNorm(256),
                                       nn.ReLU(),
                                       nn.Linear(256, 128),
                                       nn.LayerNorm(128),
                                       nn.ReLU(),
                                       nn.Linear(128, 1))
        ###########################
        assert int(use_regression_loss) + int(use_focal_loss) <= 1

        self.use_regression_loss = use_regression_loss
        if use_focal_loss:
            print("use focal loss with gamma {}".format(focal_loss_gamma))
            self.loss = FocalLoss(gamma=focal_loss_gamma)
        elif use_regression_loss:
            print("use regression L2 loss")
            self.loss = torch.nn.MSELoss()
        else:
            print("use standard BCE logit loss")
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def encode_pc(self, xyzs, rgbs, batch_size, num_objects):
        if self.ignore_rgb:
            center_xyz, x = self.object_encoder(xyzs, None)
        else:
            center_xyz, x = self.object_encoder(xyzs, rgbs)
        obj_pc_embed = self.mlp(x, center_xyz)
        obj_pc_embed = obj_pc_embed.reshape(batch_size, num_objects, -1)
        return obj_pc_embed

    def forward(self, scene_xyz, obj_xyzs, position_index, query_index, object_pad_mask):

        batch_size = object_pad_mask.shape[0]
        max_num_target_objects = object_pad_mask.shape[1]

        scene_embed = self.scene_encoder(scene_xyz)  # batch_size, 256
        obj_pc_embed = self.encode_pc(obj_xyzs, None, batch_size, max_num_target_objects)  # batch_size, num_objs, 240

        position_embed = self.position_embeddings(position_index)
        query_embed = self.query_embeddings(query_index)

        #########################
        src_sequence_encode = scene_embed.unsqueeze(1)  # batch_size, 1, 256
        tgt_sequence_encode = torch.cat([obj_pc_embed, position_embed, query_embed], dim=-1)  # batch_size, num_objs, 240 + 8 + 8
        tgt_pad_mask = object_pad_mask

        #########################
        # sequence_encode: [batch size, sequence_length, encoder input dimension]
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        src_sequence_encode = src_sequence_encode.transpose(1, 0)
        tgt_sequence_encode = tgt_sequence_encode.transpose(1, 0)

        # convert to bool
        tgt_pad_mask = (tgt_pad_mask == 1)

        # encode: [sequence_length, batch_size, embedding_size]
        encode = self.encoder(src=src_sequence_encode, tgt=tgt_sequence_encode, tgt_key_padding_mask=tgt_pad_mask)
        encode = encode.transpose(1, 0)
        #########################

        # query_index: [batch_size, num_objs]
        # encode: [batch_size, num_objs, dim]
        # need to use query_index == 1 because logical indexing requires boolean type tensor
        query_obj_encode = encode[query_index == 1]  # batch_size, dim
        is_circle = self.final_fc(query_obj_encode)  # batch_size, 1

        is_circle = is_circle.squeeze(dim=1)

        predictions = {"is_circle": is_circle}

        return predictions

    def criterion(self, predictions, labels):

        if self.use_regression_loss:
            predictions = self.convert_logits(predictions)

        loss = 0
        for key in predictions:

            preds = predictions[key]
            gts = labels[key]

            loss += self.loss(preds, gts)

        return loss

    def convert_logits(self, predictions):

        for key in predictions:
            if key == "is_circle":
                predictions[key] = torch.sigmoid(predictions[key])

        return predictions


class DiscriminatorWholeSceneWhich(torch.nn.Module):

    """
    This circle predictor uses point clouds of objects to predict whether they form a circle
    """

    def __init__(self, max_num_objects, use_focal_loss=False, focal_loss_gamma=2):

        super(DiscriminatorWholeSceneWhich, self).__init__()

        print("Discriminator Whole Scene Which")

        # input_dim: xyz + one hot for each object
        self.encoder = PointTransformerCls(input_dim=max_num_objects+3+1, output_dim=256)

        self.structure_fier = nn.Sequential(nn.Linear(256, 256),
                                               nn.LayerNorm(256),
                                               nn.ReLU(),
                                               nn.Linear(256, 128),
                                               nn.LayerNorm(128),
                                               nn.ReLU(),
                                               nn.Linear(128, 1))

        self.perturb_fier = nn.Sequential(nn.Linear(256, 256),
                                           nn.LayerNorm(256),
                                           nn.ReLU(),
                                           nn.Linear(256, 128),
                                           nn.LayerNorm(128),
                                           nn.ReLU(),
                                           nn.Linear(128, max_num_objects))

        ###########################
        if use_focal_loss:
            print("use focal loss")
            print("focal loss gamma", focal_loss_gamma)
            self.loss = FocalLoss(gamma=focal_loss_gamma)
        else:
            print("use standard BCE logit loss")
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, scene_xyz):

        encode = self.encoder(scene_xyz)
        is_structure = self.structure_fier(encode)
        perturbed_ind = self.perturb_fier(encode)

        is_structure = is_structure.squeeze(dim=1)

        predictions = {"is_structure": is_structure,
                       "perturbed_ind": perturbed_ind}

        return predictions

    def criterion(self, predictions, labels):

        loss = 0
        for key in predictions:

            preds = predictions[key]
            gts = labels[key]

            loss += self.loss(preds, gts)

        return loss

    def convert_logits(self, predictions):

        for key in predictions:
            predictions[key] = torch.sigmoid(predictions[key])

        return predictions


class RearrangeObjectsPredictorPCT(torch.nn.Module):

    def __init__(self, vocab_size,
                 num_attention_heads=8, encoder_hidden_dim=16, encoder_dropout=0.1, encoder_activation="relu", encoder_num_layers=8,
                 use_focal_loss=False, focal_loss_gamma=2):
        super(RearrangeObjectsPredictorPCT, self).__init__()

        print("Transformer with Point Transformer")

        # object encode will have dim 256
        self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=True)

        # 256 = 240 (point cloud) + 8 (position idx) + 8 (token type)
        self.mlp = EncoderMLP(256, 240, uses_pt=False)

        self.word_embeddings = torch.nn.Embedding(vocab_size, 240, padding_idx=0)
        self.token_type_embeddings = torch.nn.Embedding(2, 8)
        # max number of objects 7
        self.position_embeddings = torch.nn.Embedding(11, 8)

        encoder_layers = TransformerEncoderLayer(256, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)

        self.rearrange_object_fier = nn.Sequential(nn.Linear(256, 256),
                                                   nn.LayerNorm(256),
                                                   nn.ReLU(),
                                                   nn.Linear(256, 128),
                                                   nn.LayerNorm(128),
                                                   nn.ReLU(),
                                                   nn.Linear(128, 1))

        ###########################
        if use_focal_loss:
            print("use focal loss")
            self.loss = FocalLoss(gamma=focal_loss_gamma)
        else:
            print("use standard BCE logit loss")
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, xyzs, rgbs, object_pad_mask, sentence, sentence_pad_mask, token_type_index, position_index):

        batch_size = object_pad_mask.shape[0]
        num_objects = object_pad_mask.shape[1]
        num_words = sentence_pad_mask.shape[1]

        #########################
        center_xyz, x = self.object_encoder(xyzs, rgbs)
        x = self.mlp(x, center_xyz)
        x = x.reshape(batch_size, num_objects, -1)

        #########################
        sentence = self.word_embeddings(sentence)

        #########################

        # Important: we want to use position index here bc it might be useful for counting

        position_embed = self.position_embeddings(position_index)
        token_type_embed = self.token_type_embeddings(token_type_index)
        pad_mask = torch.cat([sentence_pad_mask, object_pad_mask], dim=1)

        sequence_encode = torch.cat([sentence, x], dim=1)
        sequence_encode = torch.cat([sequence_encode, position_embed, token_type_embed], dim=-1)
        #########################
        # sequence_encode: [batch size, sequence_length, encoder input dimension]
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        sequence_encode = sequence_encode.transpose(1, 0)

        # convert to bool
        pad_mask = (pad_mask == 1)

        # encode: [sequence_length, batch_size, embedding_size]
        encode = self.encoder(sequence_encode, src_key_padding_mask=pad_mask)
        encode = encode.transpose(1, 0)
        #########################
        obj_encodes = encode[:, -num_objects:, :]
        obj_encodes = obj_encodes.reshape(-1, obj_encodes.shape[-1])

        rearrange_obj_labels = self.rearrange_object_fier(obj_encodes).squeeze(dim=1)  # batch_size * num_objects

        predictions = {"rearrange_obj_labels": rearrange_obj_labels}

        return predictions

    def criterion(self, predictions, labels):

        loss = 0
        for key in predictions:

            preds = predictions[key]
            gts = labels[key]

            mask = gts == -100
            preds = preds[~mask]
            gts = gts[~mask]

            loss += self.loss(preds, gts)

        return loss

    def convert_logits(self, predictions):

        for key in predictions:
            if key == "rearrange_obj_labels":
                predictions[key] = torch.sigmoid(predictions[key])

        return predictions


class OrderPredictorPCT(torch.nn.Module):

    def __init__(self, vocab_size, max_num_objects,
                 num_attention_heads=8, encoder_hidden_dim=16, encoder_dropout=0.1, encoder_activation="relu",
                 encoder_num_layers=8):
        super(OrderPredictorPCT, self).__init__()

        print("Transformer with Point Transformer")

        # object encode will have dim 256
        self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=True)

        # 256 = 240 (word) + 8 (position idx) + 8 (token type) = 248 (pc embedding) + 8 (token type)
        self.mlp = EncoderMLP(256, 248, uses_pt=False)

        self.word_embeddings = torch.nn.Embedding(vocab_size, 240, padding_idx=0)
        self.token_type_embeddings = torch.nn.Embedding(2, 8)
        # max number of objects 11
        self.position_embeddings = torch.nn.Embedding(max_num_objects, 8)

        encoder_layers = TransformerEncoderLayer(256, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)

        ##########################
        # Sinkhorn params
        self.latent_dim = 256
        self.K = max_num_objects
        self.n_samples = 5
        self.noise_factor = 1.0
        self.temp = 1.0
        self.n_iters = 20

        self.criterion = nn.MSELoss(reduction='none')

        self.sinknet = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim),
                                     nn.LayerNorm(self.latent_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.latent_dim, self.K))

    def forward(self, xyzs, rgbs, object_pad_mask, sentence, sentence_pad_mask, token_type_index, position_index,
                normal_samples):

        batch_size = object_pad_mask.shape[0]
        num_objects = object_pad_mask.shape[1]
        num_words = sentence_pad_mask.shape[1]

        #########################
        center_xyz, x = self.object_encoder(xyzs, rgbs)
        x = self.mlp(x, center_xyz)
        x = x.reshape(batch_size, num_objects, -1)

        token_type_embed = self.token_type_embeddings(token_type_index[:, num_words:])
        x = torch.cat([x, token_type_embed], dim=-1)

        #########################
        sentence = self.word_embeddings(sentence)

        position_embed = self.position_embeddings(position_index[:, :num_words])
        token_type_embed = self.token_type_embeddings(token_type_index[:, :num_words])
        sentence = torch.cat([sentence, position_embed, token_type_embed], dim=-1)

        #########################
        pad_mask = torch.cat([sentence_pad_mask, object_pad_mask], dim=1)
        sequence_encode = torch.cat([sentence, x], dim=1)

        #########################
        # sequence_encode: [batch size, sequence_length, encoder input dimension]
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        sequence_encode = sequence_encode.transpose(1, 0)

        # convert to bool
        pad_mask = (pad_mask == 1)

        # encode: [sequence_length, batch_size, embedding_size]
        encode = self.encoder(sequence_encode, src_key_padding_mask=pad_mask)
        encode = encode.transpose(1, 0)
        #########################
        obj_encodes = encode[:, -num_objects:, :]  # batch_size, max_num_objects (k), latent_dim

        #########################
        log_alpha = self.sinknet(obj_encodes)  # batch_size, max_num_objects (k), k
        log_alpha = log_alpha.reshape(-1, self.K, self.K)  # insurance

        soft_perms_inf, log_alpha_w_noise = self.gumbel_sinkhorn(log_alpha, normal_samples)

        P = self.inv_soft_pers_flattened(soft_perms_inf, self.K)  # n_samples * batch_size, k, k

        # conveniently, we just use position index here
        seq = position_index[:, num_words:].float()
        seq_tiled = seq.repeat(self.n_samples, 1)  # n_samples * batch_size, k
        seq_pred = self.permute(torch.unsqueeze(seq_tiled, dim=-1), P)  # n_samples * batch_size, k, 1
        seq_pred = torch.squeeze(seq_pred)  # n_samples * batch_size, k
        # important: the axes for seq_pred are n_samples, batch_size, k

        predictions = {"rearrange_obj_sequence_samples": seq_pred,
                       "P": P}

        return predictions

    def loss(self, seq_pred, seq_gt, stopping_mask, seq_len):

        # seq_pred: n_samples * batch_size, k
        # seq_gt: batch_size, k
        # stopping_mask: n_samples * batch_size, k
        # seq_len: batch_size

        recon_loss = self.criterion(seq_pred * stopping_mask, seq_gt.repeat(self.n_samples, 1) * stopping_mask)  # n_samples * batch_size, k
        recon_loss = (recon_loss.sum(dim=-1) / seq_len.repeat(self.n_samples).int().float()).mean()

        return recon_loss

    def permute(self, seq, P):

        return torch.matmul(P, seq)

    def inv_soft_pers_flattened(self, soft_perms_inf, n_numbers):

        # soft_perms_inf: batch_size, n_samples, k, k
        inv_soft_perms = torch.transpose(soft_perms_inf, 2, 3)
        inv_soft_perms = torch.transpose(inv_soft_perms, 0, 1)  # n_samples, batch_size, k, k

        inv_soft_perms_flat = inv_soft_perms.view(-1, n_numbers, n_numbers)  # n_samples * batch_size, k, k
        return inv_soft_perms_flat

    def sample_gumbel(self, normal_samples, shape, eps=1e-20):
        assert list(normal_samples.shape) == list(shape)
        return -torch.log(eps - torch.log(normal_samples + eps))

    def gumbel_sinkhorn(self, log_alpha, normal_samples):

        n = log_alpha.size()[1]
        log_alpha = log_alpha.view(-1, n, n)
        batch_size = log_alpha.size()[0]

        log_alpha_w_noise = log_alpha.repeat(self.n_samples, 1, 1)  # n_samples * batch_size, k, k

        if self.noise_factor == 0:
            noise = 0.0
        else:
            noise = self.sample_gumbel(normal_samples, [self.n_samples * batch_size, n, n]) * self.noise_factor  # n_samples * batch_size, k, k

        log_alpha_w_noise = log_alpha_w_noise + noise
        log_alpha_w_noise = log_alpha_w_noise / self.temp

        my_log_alpha_w_noise = log_alpha_w_noise.clone()

        sink = self.sinkhorn(my_log_alpha_w_noise)

        sink = sink.view(self.n_samples, batch_size, n, n)
        sink = torch.transpose(sink, 1, 0)  # batch_size, n_samples, k, k
        log_alpha_w_noise = log_alpha_w_noise.view(self.n_samples, batch_size, n, n)
        log_alpha_w_noise = torch.transpose(log_alpha_w_noise, 1, 0)  # batch_size, n_samples, k, k

        return sink, log_alpha_w_noise

    def sinkhorn(self, log_alpha, n_iters=20):

        n = log_alpha.size()[1]
        log_alpha = log_alpha.view(-1, n, n)  # n_samples * batch_size, k, k

        for i in range(n_iters):
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
        return torch.exp(log_alpha)