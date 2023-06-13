import torch
import torch.optim as optim
import numpy as np
import time
import os
import tqdm
import argparse
from omegaconf import OmegaConf
from collections import defaultdict
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from StructDiffusion.data.dataset_pairwise_collision import PairwiseCollisionDataset
from StructDiffusion.models.semantic_rearrangement_discriminators import DiscriminatorWholeScene


def evaluate(gts, predictions, key, regression_mode=False, return_classification_dict=False, debug=True,
             dataset_idxs=None):
    """
    :param gts: expect a list of tensors
    :param predictions: expect a list of tensor
    :return: a score indicating how good the model is.
    """

    predictions_for_key = torch.cat(predictions[key], dim=0)
    gts_for_key = torch.cat(gts[key], dim=0)
    if dataset_idxs is not None:
        dataset_idxs = torch.cat(dataset_idxs, dim=0)  # dim: num_scenes

    assert len(gts_for_key) == len(predictions_for_key)
    num_scenes = len(predictions_for_key)

    if not regression_mode:
        predicted_classes = predictions_for_key > 0.5
        accuracy = torch.sum(gts_for_key == predicted_classes) / len(gts_for_key)

        if debug:
            print("Groundtruths:")
            print(gts_for_key[:100])
            print("Predictions")
            print(predictions_for_key[:100])
            print("Predicted Classes")
            print(predicted_classes[:100])
        print("{} scenes -- {} accuracy: {}".format(num_scenes, key, accuracy))
        report = classification_report(gts_for_key.detach().cpu().numpy(), predicted_classes.detach().cpu().numpy(),
                                    output_dict=True)
        print(report)

        # compute score for each dataset
        if dataset_idxs is not None:
            for di in torch.unique(dataset_idxs):
                print("***")
                print("for dataset", di)

                dataset_labels = gts_for_key[dataset_idxs == di]
                dataset_predictions = predictions_for_key[dataset_idxs == di]
                dataset_predicted_classes = predicted_classes[dataset_idxs == di]
                dataset_report = classification_report(dataset_labels.detach().cpu().numpy(), dataset_predicted_classes.detach().cpu().numpy(),
                                               output_dict=True)
                if debug:
                    print("Groundtruths:")
                    print(dataset_labels[:100])
                    print("Predictions")
                    print(dataset_predictions[:100])
                    print("Predicted Classes")
                    print(dataset_predicted_classes[:100])
                print(dataset_report)

        if not return_classification_dict:
            return accuracy
        else:
            return report

    else:
        distances = predictions_for_key - gts_for_key

        me = torch.mean(torch.abs(distances))
        mse = torch.mean(distances ** 2)
        med = torch.median(torch.abs(distances))

        if debug:
            print("Groundtruths:")
            print(gts_for_key[:100])
            print("Predictions")
            print(predictions_for_key[:100])

        print("ME for {} scenes: {}".format(num_scenes, me))
        print("MSE for {} scenes: {}".format(num_scenes, mse))
        print("MEDIAN for {} scenes: {}".format(num_scenes, med))

        if dataset_idxs is not None:
            raise Exception("per dataset evaluation not implemented yet")

        return -mse


def train_model(cfg, model, data_iter, optimizer, scheduler, num_epochs, device, save_best_model, grad_clipping=1.0, tqdm_update_rate=1):

    # validate first
    validate(model, data_iter["valid"], 0, device, regression_mode=cfg.regression_mode)

    if save_best_model:
        best_model_dir = os.path.join(cfg.experiment_dir, "best_model")
        print("best model will be saved to {}".format(best_model_dir))
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        best_score = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        epoch_loss = 0
        gts = defaultdict(list)
        predictions = defaultdict(list)

        with tqdm.tqdm(total=len(data_iter["train"])) as pbar:
            # batch_timer_start = time.time()
            for step, batch in enumerate(data_iter["train"]):
                # batch_timer_stop = time.time()
                # elapsed = batch_timer_stop - batch_timer_start
                # print("Load data time (step {}): {}".format(step, elapsed))

                optimizer.zero_grad()

                # input
                scene_xyz = batch["scene_xyz"].to(device, non_blocking=True)

                # output
                targets = {}
                for key in ["is_circle"]:
                    targets[key] = batch[key].to(device, non_blocking=True)
                # print(targets[key])

                preds = model.forward(scene_xyz)
                loss = model.criterion(preds, targets)
                loss.backward()

                if grad_clipping != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

                optimizer.step()
                epoch_loss += loss
                # batch_timer_start = time.time()

                preds = model.convert_logits(preds)
                for key in ["is_circle"]:
                    gts[key].append(targets[key])
                    predictions[key].append(preds[key])

                if step != 0 and step % tqdm_update_rate == 0:
                    pbar.update(tqdm_update_rate)
                    pbar.set_postfix({"Batch loss": loss})

        if scheduler is not None:
            scheduler.step()

        print('[Epoch:{}]:  Training Loss:{:.4}'.format(epoch, epoch_loss))
        evaluate(gts, predictions, "is_circle", regression_mode=cfg.regression_mode)

        score = validate(model, data_iter["valid"], epoch, device, regression_mode=cfg.regression_mode)
        if save_best_model and score > best_score:
            print("Saving best model so far...")
            best_score = score
            save_model(best_model_dir, cfg, epoch, model, optimizer)

    return model


def validate(model, data_iter, epoch, device, regression_mode, tqdm_update_rate=1, return_classification_dict=False):
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
                scene_xyz = batch["scene_xyz"].to(device, non_blocking=True)

                # output
                targets = {}
                for key in ["is_circle"]:
                    targets[key] = batch[key].to(device, non_blocking=True)
                # print(targets[key])

                preds = model.forward(scene_xyz)
                loss = model.criterion(preds, targets)

                preds = model.convert_logits(preds)
                for key in ["is_circle"]:
                    gts[key].append(targets[key])
                    predictions[key].append(preds[key])

                epoch_loss += loss
                if step != 0 and step % tqdm_update_rate == 0:
                    pbar.update(tqdm_update_rate)
                    pbar.set_postfix({"Batch loss": loss})

    print('[Epoch:{}]:  Val Loss:{:.4}'.format(epoch, epoch_loss))
    if not return_classification_dict:
        score = evaluate(gts, predictions, "is_circle", regression_mode)
    else:
        score = evaluate(gts, predictions, "is_circle", return_classification_dict=True)
    return score


def infer_once(model, batch, device, verbose=True):

    model.eval()
    with torch.no_grad():

        # input
        scene_xyz = batch["scene_xyz"].to(device, non_blocking=True)
        preds = model.forward(scene_xyz)
        preds = model.convert_logits(preds)

    return preds


def save_model(model_dir, cfg, epoch, model, optimizer=None, scheduler=None):
    state_dict = {'epoch': epoch,
                  'model_state_dict': model.state_dict()}
    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state_dict, os.path.join(model_dir, "model.tar"))
    OmegaConf.save(cfg, os.path.join(model_dir, "config.yaml"))


def load_model(model_dir):
    """
    Load transformer model
    Important: to use the model, call model.eval() or model.train()
    :param model_dir:
    :return:
    """

    # load dictionaries
    cfg = OmegaConf.load(os.path.join(model_dir, "config.yaml"))

    # initialize model
    model_cfg = cfg.model
    model = DiscriminatorWholeScene(max_num_objects=cfg.dataset.max_num_objects,
                                    use_focal_loss=model_cfg.use_focal_loss,
                                    focal_loss_gamma=model_cfg.focal_loss_gamma,
                                    use_regression_loss=cfg.regression_mode,
                                    pct_random_sampling=model_cfg.pct_random_sampling)
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
    return cfg, model, optimizer, scheduler, epoch


def run_model(cfg):

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)
        torch.backends.cudnn.deterministic = True

    data_cfg = cfg.dataset

    full_dataset = PairwiseCollisionDataset(data_cfg.dirs, data_cfg.index_dirs,
                                            data_cfg.urdf_pc_idx_file,
                             data_cfg.collision_data_dir,
                             num_pts=data_cfg.num_pts,
                             num_scene_pts=data_cfg.num_scene_pts,
                             normalize_pc=data_cfg.normalize_pc,
                             random_rotation = data_cfg.random_rotation,
                             data_augmentation = data_cfg.data_augmentation)

    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset) * 0.7),
                                                                              len(full_dataset) - int(
                                                                                  len(full_dataset) * 0.7)],
                                                               generator=torch.Generator().manual_seed(cfg.random_seed))

    data_iter = {}
    data_iter["train"] = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True,
                                    num_workers=data_cfg.num_workers,
                                    collate_fn=PairwiseCollisionDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory)
    data_iter["valid"] = DataLoader(valid_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                    num_workers=data_cfg.num_workers,
                                    collate_fn=PairwiseCollisionDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory)

    # load model
    model_cfg = cfg.model
    training_cfg = cfg.training
    # cfg.dataset.positive_ratio = train_dataset.get_positive_ratio()
    model = DiscriminatorWholeScene(max_num_objects=cfg.dataset.max_num_objects,
                                    use_focal_loss=model_cfg.use_focal_loss,
                                    focal_loss_gamma=model_cfg.focal_loss_gamma,
                                    use_regression_loss=cfg.regression_mode,
                                    pct_random_sampling=model_cfg.pct_random_sampling)
    model.to(cfg.device)

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)

    # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scheduler = None

    train_model(cfg, model, data_iter, optimizer, scheduler, training_cfg.max_epochs, cfg.device, cfg.save_best_model)

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
                        default='../../../configs/collision_detector.yaml',
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