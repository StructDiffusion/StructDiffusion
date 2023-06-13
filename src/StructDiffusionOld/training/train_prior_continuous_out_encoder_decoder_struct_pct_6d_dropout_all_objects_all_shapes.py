import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler
import numpy as np
import time
import os
import tqdm
import argparse
from omegaconf import OmegaConf
from collections import defaultdict

from torch.utils.data import DataLoader
from StructDiffusion.data.dataset_v23_continuous_out_ar_6d_all_objects_all_shapes import SemanticArrangementDataset
from StructDiffusion.models.baselines import PriorContinuousOutEncoderDecoderStructPCT6DDropoutAllObjects
from StructDiffusion.data.tokenizer import Tokenizer
from StructDiffusion.utils.rotation_continuity import compute_geodesic_distance_from_two_matrices
from StructDiffusion.utils.rearrangement import evaluate_prior_prediction


def evaluate(gts, predictions, keys, debug=False):
    """
    :param gts: expect a list of tensors
    :param predictions: expect a list of tensor
    :return:
    """

    total_mses = 0
    for key in keys:
        # predictions[key][0]: [batch_size * number_of_objects, dim]
        predictions_for_key = torch.cat(predictions[key], dim=0)
        # gts[key][0]: [batch_size * number_of_objects, dim]
        gts_for_key = torch.cat(gts[key], dim=0)

        assert gts_for_key.shape == predictions_for_key.shape

        target_indices = gts_for_key != -100
        gts_for_key = gts_for_key[target_indices]
        predictions_for_key = predictions_for_key[target_indices]
        num_objects = len(predictions_for_key)

        distances = predictions_for_key - gts_for_key

        me = torch.mean(torch.abs(distances))
        mse = torch.mean(distances ** 2)

        if debug:
            print("Groundtruths:")
            print(gts_for_key[:100])
            print("Predictions")
            print(predictions_for_key[:100])

        print("{} ME for {} objects: {}".format(key, num_objects, me))
        print("{} MSE for {} objects: {}".format(key, num_objects, mse))
        total_mses += mse

        if "theta" in key:
            predictions_for_key = predictions_for_key.reshape(-1, 3, 3)
            gts_for_key = gts_for_key.reshape(-1, 3, 3)
            geodesic_distance = compute_geodesic_distance_from_two_matrices(predictions_for_key, gts_for_key)
            geodesic_distance = torch.rad2deg(geodesic_distance)
            mgd = torch.mean(geodesic_distance)
            print("{} Mean Geodesic Distance for {} objects: {}".format(key, num_objects, mgd))

    return -total_mses

    # for key in ["obj_x_outputs", "obj_y_outputs", "obj_theta_outputs"]:
    #     predicted_values = np.array(predictions[key])
    #     gt_values = np.array(gts[key])
    #     assert len(gt_values) == len(predicted_values)
    #     mask = gt_values == -100
    #
    #     if cfg.obj_xytheta_predict_target_only:
    #         gt_values = gt_values[~mask]
    #         predicted_values = predicted_values[~mask]
    #         mean_dist = np.mean(np.abs(predicted_values - gt_values))
    #         mean_squared_dist = np.mean((predicted_values - gt_values) ** 2)
    #         print("{}: ME {} MSE {} per object".format(key, mean_dist, mean_squared_dist))
    #     else:
    #         obj_xyztheta_indicators_all = np.array(obj_xyztheta_indicators_all, dtype=bool)
    #         target_gt_values = gt_values[~mask & obj_xyztheta_indicators_all]
    #         target_predicted_values = predicted_values[~mask & obj_xyztheta_indicators_all]
    #         target_mean_dist = np.mean(np.abs(target_predicted_values - target_gt_values))
    #         target_mean_squared_dist = np.mean((target_predicted_values - target_gt_values) ** 2)
    #         print(target_gt_values[:100])
    #         print("target object {}: ME {} MSE {} per object".format(key, target_mean_dist, target_mean_squared_dist))
    #         # loss for objects that shouldn't be moved should be very small
    #         other_gt_values = gt_values[~mask & ~obj_xyztheta_indicators_all]
    #         other_predicted_values = predicted_values[~mask & ~obj_xyztheta_indicators_all]
    #         other_mean_dist = np.mean(np.abs(other_predicted_values - other_gt_values))
    #         other_mean_squared_dist = np.mean((other_predicted_values - other_gt_values) ** 2)
    #         print(other_gt_values[:100])
    #         print("other object {}: ME {} MSE {} per object".format(key, other_mean_dist, other_mean_squared_dist))


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def train_model(cfg, model, data_iter, optimizer, warmup, num_epochs, device, save_best_model, grad_clipping=1.0):

    if save_best_model:
        best_model_dir = os.path.join(cfg.experiment_dir, "best_model")
        print("best model will be saved to {}".format(best_model_dir))
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        best_score = -np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        epoch_loss = 0
        gts = defaultdict(list)
        predictions = defaultdict(list)

        with tqdm.tqdm(total=len(data_iter["train"])) as pbar:
            for step, batch in enumerate(data_iter["train"]):
                optimizer.zero_grad()
                # input
                xyzs = batch["xyzs"].to(device, non_blocking=True)
                rgbs = batch["rgbs"].to(device, non_blocking=True)
                object_pad_mask = batch["object_pad_mask"].to(device, non_blocking=True)
                other_xyzs = batch["other_xyzs"].to(device, non_blocking=True)
                other_rgbs = batch["other_rgbs"].to(device, non_blocking=True)
                other_object_pad_mask = batch["other_object_pad_mask"].to(device, non_blocking=True)
                sentence = batch["sentence"].to(device, non_blocking=True)
                sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)
                token_type_index = batch["token_type_index"].to(device, non_blocking=True)
                position_index = batch["position_index"].to(device, non_blocking=True)
                obj_x_inputs = batch["obj_x_inputs"].to(device, non_blocking=True)
                obj_y_inputs = batch["obj_y_inputs"].to(device, non_blocking=True)
                obj_z_inputs = batch["obj_z_inputs"].to(device, non_blocking=True)
                obj_theta_inputs = batch["obj_theta_inputs"].to(device, non_blocking=True)

                struct_x_inputs = batch["struct_x_inputs"].to(device, non_blocking=True)
                struct_y_inputs = batch["struct_y_inputs"].to(device, non_blocking=True)
                struct_z_inputs = batch["struct_z_inputs"].to(device, non_blocking=True)
                struct_theta_inputs = batch["struct_theta_inputs"].to(device, non_blocking=True)
                struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
                struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
                struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)

                tgt_mask = generate_square_subsequent_mask(object_pad_mask.shape[1] + 1).to(device, non_blocking=True)
                start_token = torch.zeros((object_pad_mask.shape[0], 1), dtype=torch.long).to(device, non_blocking=True)

                # output
                targets = {}
                for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                            "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
                    targets[key] = batch[key].to(device, non_blocking=True)
                    targets[key] = targets[key].reshape(targets[key].shape[0] * targets[key].shape[1], -1)

                preds = model.forward(xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                                      sentence, sentence_pad_mask, token_type_index,
                                      obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index,
                                      tgt_mask, start_token,
                                      struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                                      struct_position_index, struct_token_type_index, struct_pad_mask)

                loss = model.criterion(preds, targets)
                loss.backward()

                if grad_clipping != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

                optimizer.step()
                epoch_loss += loss

                for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                            "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
                    gts[key].append(targets[key].detach())
                    predictions[key].append(preds[key].detach())

                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss})

        warmup.step()

        print('[Epoch:{}]:  Training Loss:{:.4}'.format(epoch, epoch_loss))
        evaluate(gts, predictions, ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                                    "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"])

        score = validate(cfg, model, data_iter["valid"], epoch, device)
        if save_best_model and score > best_score:
            print("Saving best model so far...")
            best_score = score
            save_model(best_model_dir, cfg, epoch, model)

    return model


def validate(cfg, model, data_iter, epoch, device):
    """
    helper function to evaluate the model

    :param model:
    :param data_iter:
    :param epoch:
    :param device:
    :return:
    """

    model.eval()

    epoch_loss = 0
    gts = defaultdict(list)
    predictions = defaultdict(list)
    with torch.no_grad():

        with tqdm.tqdm(total=len(data_iter)) as pbar:
            for step, batch in enumerate(data_iter):

                # input
                xyzs = batch["xyzs"].to(device, non_blocking=True)
                rgbs = batch["rgbs"].to(device, non_blocking=True)
                object_pad_mask = batch["object_pad_mask"].to(device, non_blocking=True)
                other_xyzs = batch["other_xyzs"].to(device, non_blocking=True)
                other_rgbs = batch["other_rgbs"].to(device, non_blocking=True)
                other_object_pad_mask = batch["other_object_pad_mask"].to(device, non_blocking=True)
                sentence = batch["sentence"].to(device, non_blocking=True)
                sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)
                token_type_index = batch["token_type_index"].to(device, non_blocking=True)
                position_index = batch["position_index"].to(device, non_blocking=True)
                obj_x_inputs = batch["obj_x_inputs"].to(device, non_blocking=True)
                obj_y_inputs = batch["obj_y_inputs"].to(device, non_blocking=True)
                obj_z_inputs = batch["obj_z_inputs"].to(device, non_blocking=True)
                obj_theta_inputs = batch["obj_theta_inputs"].to(device, non_blocking=True)

                struct_x_inputs = batch["struct_x_inputs"].to(device, non_blocking=True)
                struct_y_inputs = batch["struct_y_inputs"].to(device, non_blocking=True)
                struct_z_inputs = batch["struct_z_inputs"].to(device, non_blocking=True)
                struct_theta_inputs = batch["struct_theta_inputs"].to(device, non_blocking=True)
                struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
                struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
                struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)

                tgt_mask = generate_square_subsequent_mask(object_pad_mask.shape[1] + 1).to(device, non_blocking=True)
                start_token = torch.zeros((object_pad_mask.shape[0], 1), dtype=torch.long).to(device, non_blocking=True)

                # output
                targets = {}
                for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                            "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
                    targets[key] = batch[key].to(device, non_blocking=True)
                    targets[key] = targets[key].reshape(targets[key].shape[0] * targets[key].shape[1], -1)

                preds = model.forward(xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                                      sentence, sentence_pad_mask, token_type_index,
                                      obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index,
                                      tgt_mask, start_token,
                                      struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                                      struct_position_index, struct_token_type_index, struct_pad_mask)
                loss = model.criterion(preds, targets)

                for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                            "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
                    gts[key].append(targets[key])
                    predictions[key].append(preds[key])

                epoch_loss += loss
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss})

    print('[Epoch:{}]:  Val Loss:{:.4}'.format(epoch, epoch_loss))

    score = evaluate_prior_prediction(gts, predictions,
                     ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                      "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"])
    return score


def infer_once(cfg, model, batch, device):

    model.eval()

    predictions = defaultdict(list)
    with torch.no_grad():

        # input
        xyzs = batch["xyzs"].to(device, non_blocking=True)
        rgbs = batch["rgbs"].to(device, non_blocking=True)
        object_pad_mask = batch["object_pad_mask"].to(device, non_blocking=True)
        other_xyzs = batch["other_xyzs"].to(device, non_blocking=True)
        other_rgbs = batch["other_rgbs"].to(device, non_blocking=True)
        other_object_pad_mask = batch["other_object_pad_mask"].to(device, non_blocking=True)
        sentence = batch["sentence"].to(device, non_blocking=True)
        sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)
        token_type_index = batch["token_type_index"].to(device, non_blocking=True)
        position_index = batch["position_index"].to(device, non_blocking=True)

        struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
        struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
        struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)

        obj_x_inputs = batch["obj_x_inputs"].to(device, non_blocking=True)
        obj_y_inputs = batch["obj_y_inputs"].to(device, non_blocking=True)
        obj_z_inputs = batch["obj_z_inputs"].to(device, non_blocking=True)
        obj_theta_inputs = batch["obj_theta_inputs"].to(device, non_blocking=True)

        struct_x_inputs = batch["struct_x_inputs"].to(device, non_blocking=True)
        struct_y_inputs = batch["struct_y_inputs"].to(device, non_blocking=True)
        struct_z_inputs = batch["struct_z_inputs"].to(device, non_blocking=True)
        struct_theta_inputs = batch["struct_theta_inputs"].to(device, non_blocking=True)

        tgt_mask = generate_square_subsequent_mask(object_pad_mask.shape[1] + 1).to(device, non_blocking=True)
        start_token = torch.zeros((object_pad_mask.shape[0], 1), dtype=torch.long).to(device, non_blocking=True)

        preds = model.forward(xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                              sentence, sentence_pad_mask, token_type_index,
                              obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index,
                              tgt_mask, start_token,
                              struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                              struct_position_index, struct_token_type_index, struct_pad_mask)

        for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                    "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
            predictions[key].append(preds[key])

    return predictions


def save_model(model_dir, cfg, epoch, model, optimizer=None, scheduler=None):
    state_dict = {'epoch': epoch,
                  'model_state_dict': model.state_dict()}
    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state_dict, os.path.join(model_dir, "model.tar"))
    OmegaConf.save(cfg, os.path.join(model_dir, "config.yaml"))


def load_model(model_dir, ngc_vocab=False):
    """
    Load transformer model
    Important: to use the model, call model.eval() or model.train()
    :param model_dir:
    :return:
    """
    # load dictionaries
    cfg = OmegaConf.load(os.path.join(model_dir, "config.yaml"))

    data_cfg = cfg.dataset
    # if not ngc_vocab:
    #     tokenizer = Tokenizer(os.path.join(data_cfg.dir, "type_vocabs.json"))
    # else:
    #     tokenizer = Tokenizer(os.path.join(data_cfg.dir, "type_vocabs_ngc.json"))
    tokenizer = Tokenizer(data_cfg.vocab_dir)
    vocab_size = tokenizer.get_vocab_size()

    # initialize model
    model_cfg = cfg.model
    if cfg.degree_continuity_correction is None:
        cfg.degree_continuity_correction = False
    model = PriorContinuousOutEncoderDecoderStructPCT6DDropoutAllObjects(vocab_size,
                                                                         num_attention_heads=model_cfg.num_attention_heads,
                                                                         encoder_hidden_dim=model_cfg.encoder_hidden_dim,
                                                                         encoder_dropout=model_cfg.encoder_dropout,
                                                                         encoder_activation=model_cfg.encoder_activation,
                                                                         encoder_num_layers=model_cfg.encoder_num_layers,
                                                                         structure_dropout=model_cfg.structure_dropout,
                                                                         object_dropout=model_cfg.object_dropout,
                                                                         theta_loss_divide=model_cfg.theta_loss_divide,
                                                                         ignore_rgb=model_cfg.ignore_rgb)
    model.to(cfg.device)

    # load state dicts
    checkpoint = torch.load(os.path.join(model_dir, "model.tar"))
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = None
    if "optimizer_state_dict" in checkpoint:
        training_cfg = cfg.training
        optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = None
    if "scheduler_state_dict" in checkpoint:
        scheduler = None
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    return cfg, tokenizer, model, optimizer, scheduler, epoch


def run_model(cfg):

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)
        torch.backends.cudnn.deterministic = True

    data_cfg = cfg.dataset
    tokenizer = Tokenizer(data_cfg.vocab_dir)
    vocab_size = tokenizer.get_vocab_size()

    train_dataset = SemanticArrangementDataset(data_cfg.dirs, data_cfg.index_dirs, "train", tokenizer,
                                               data_cfg.max_num_objects,
                                               data_cfg.max_num_other_objects,
                                               data_cfg.max_num_shape_parameters,
                                               data_cfg.max_num_rearrange_features,
                                               data_cfg.max_num_anchor_features,
                                               data_cfg.num_pts)
    valid_dataset = SemanticArrangementDataset(data_cfg.dirs, data_cfg.index_dirs, "valid", tokenizer,
                                               data_cfg.max_num_objects,
                                               data_cfg.max_num_other_objects,
                                               data_cfg.max_num_shape_parameters,
                                               data_cfg.max_num_rearrange_features,
                                               data_cfg.max_num_anchor_features,
                                               data_cfg.num_pts)

    data_iter = {}
    data_iter["train"] = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True,
                                    collate_fn=SemanticArrangementDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)
    data_iter["valid"] = DataLoader(valid_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                    collate_fn=SemanticArrangementDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)

    # load model
    model_cfg = cfg.model
    model = PriorContinuousOutEncoderDecoderStructPCT6DDropoutAllObjects(vocab_size,
                                                                         num_attention_heads=model_cfg.num_attention_heads,
                                                                         encoder_hidden_dim=model_cfg.encoder_hidden_dim,
                                                                         encoder_dropout=model_cfg.encoder_dropout,
                                                                         encoder_activation=model_cfg.encoder_activation,
                                                                         encoder_num_layers=model_cfg.encoder_num_layers,
                                                                         structure_dropout=model_cfg.structure_dropout,
                                                                         object_dropout=model_cfg.object_dropout,
                                                                         theta_loss_divide=model_cfg.theta_loss_divide,
                                                                         ignore_rgb=model_cfg.ignore_rgb,
                                                                         ignore_other_objects=model_cfg.ignore_other_objects)
    model.to(cfg.device)

    training_cfg = cfg.training
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate, weight_decay=training_cfg.l2)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg.lr_restart)
    warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=training_cfg.warmup,
                                    after_scheduler=scheduler)

    train_model(cfg, model, data_iter, optimizer, warmup, training_cfg.max_epochs, cfg.device, cfg.save_best_model)

    # save model
    if cfg.save_model:
        model_dir = os.path.join(cfg.experiment_dir, "model")
        print("Saving model to {}".format(model_dir))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_model(model_dir, cfg, cfg.max_epochs, model, optimizer, scheduler)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--base_config_file", help='base config yaml file',
                        default='../../../configs/base.yaml',
                        type=str)
    parser.add_argument("--config_file", help='config yaml file',
                        default='../../../configs/encoderdecoder.yaml',
                        type=str)
    args = parser.parse_args()
    assert os.path.exists(args.base_config_file), "Cannot find base config yaml file at {}".format(args.config_file)
    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)
    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")
    base_cfg = OmegaConf.load(args.base_config_file)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(base_cfg, cfg)
    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)
    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

    run_model(cfg)