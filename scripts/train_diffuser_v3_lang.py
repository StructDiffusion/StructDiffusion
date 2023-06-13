import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import os
import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from StructDiffusion.data.semantic_arrangement import SemanticArrangementDataset
from StructDiffusion.language.tokenizer import Tokenizer
from StructDiffusion.models.models import TransformerDiffusionModel

from StructDiffusion.diffusion.noise_schedule import NoiseSchedule, q_sample
from StructDiffusion.diffusion.pose_conversion import get_diffusion_variables, get_struct_objs_poses
from StructDiffusion.utils.batch_inference import move_pc_and_create_scene_simple, visualize_batch_pcs


class ConditionalPoseDiffusionModel(pl.LightningModule):

    def __init__(self, vocab_size, model_cfg, loss_cfg, noise_scheduler_cfg, optimizer_cfg):
        super().__init__()
        self.save_hyperparameters()

        self.model = TransformerDiffusionModel(vocab_size, **model_cfg)

        self.noise_schedule = NoiseSchedule(**noise_scheduler_cfg)

        self.loss_type = loss_cfg.type

        self.optimizer_cfg = optimizer_cfg
        self.configure_optimizers()

    def forward(self, batch):

        # input
        xyzs = batch["xyzs"]
        B = xyzs.shape[0]
        # obj_pad_mask: we don't need it now since we are testing
        obj_xyztheta_inputs = batch["obj_xyztheta_inputs"]
        struct_xyztheta_inputs = batch["struct_xyztheta_inputs"]
        position_index = batch["position_index"]
        struct_position_index = batch["struct_position_index"]
        object_pad_mask = batch["obj_pad_mask"]
        struct_pad_mask = batch["struct_pad_mask"]
        sentence = batch["sentence"]
        token_type_index = batch["token_type_index"]
        struct_token_type_index = batch["struct_token_type_index"]
        sentence_pad_mask = batch["sentence_pad_mask"]

        start_token = torch.zeros((B, 1), dtype=torch.long).to(self.device)
        t = torch.randint(0, self.noise_schedule.timesteps, (B,), dtype=torch.long).to(self.device)

        # --------------
        x_start = get_diffusion_variables(struct_xyztheta_inputs, obj_xyztheta_inputs)
        noise = torch.randn_like(x_start, device=self.device)
        x_noisy = q_sample(x_start=x_start, t=t, noise_schedule=self.noise_schedule, noise=noise)

        struct_xyztheta_inputs = x_noisy[:, 0, :].unsqueeze(1)  # B, 1, 3 + 6
        obj_xyztheta_inputs = x_noisy[:, 1:, :]  # B, N, 3 + 6
        struct_xyztheta_outputs, obj_xyztheta_outputs = self.model.forward(t, xyzs, obj_xyztheta_inputs,
                                                                      struct_xyztheta_inputs, sentence,
                                                                      position_index, struct_position_index,
                                                                      token_type_index, struct_token_type_index,
                                                                      start_token,
                                                                      object_pad_mask, struct_pad_mask,
                                                                      sentence_pad_mask)
        predicted_noise = torch.cat([struct_xyztheta_outputs, obj_xyztheta_outputs], dim=1)  # B, 1+N, 9

        # important: skip computing loss for masked positions
        pad_mask = torch.cat([struct_pad_mask, object_pad_mask], dim=1)  # B, 1+N
        keep_mask = (pad_mask == 0)
        noise = noise[keep_mask]  # dim: number of positions that need loss calculation
        predicted_noise = predicted_noise[keep_mask]

        return noise, predicted_noise

    def compute_loss(self, noise, predicted_noise, prefix="train/"):
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        self.log(prefix + "loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        noise, pred_noise, aux_data = self.forward(batch)
        loss = self.compute_loss(noise, pred_noise, prefix="train/")
        return loss

    def validation_step(self, batch, batch_idx):
        noise, pred_noise, aux_data = self.forward(batch)
        loss = self.compute_loss(noise, pred_noise, prefix="val/")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.optimizer_cfg.lr, self.optimizer_cfg.weight_decay) # 1e-5
        return optimizer






def train_model(cfg, model, data_iter, noise_schedule, optimizer, warmup, num_epochs, device, save_best_model,
                summary_writer, grad_clipping=1.0):

    loss_type = cfg.diffusion.loss_type

    # if save_best_model:
    #     best_model_dir = os.path.join(cfg.experiment_dir, "best_model")
    #     print("best model will be saved to {}".format(best_model_dir))
    #     if not os.path.exists(best_model_dir):
    #         os.makedirs(best_model_dir)
    #     best_score = -np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        epoch_loss = 0

        with tqdm.tqdm(total=len(data_iter["train"])) as pbar:
            for step, batch in enumerate(data_iter["train"]):
                optimizer.zero_grad()

                # input
                xyzs = batch["xyzs"].to(device, non_blocking=True)
                B = xyzs.shape[0]
                # obj_pad_mask: we don't need it now since we are testing
                obj_xyztheta_inputs = batch["obj_xyztheta_inputs"].to(device, non_blocking=True)
                struct_xyztheta_inputs = batch["struct_xyztheta_inputs"].to(device, non_blocking=True)
                position_index = batch["position_index"].to(device, non_blocking=True)
                struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
                object_pad_mask = batch["obj_pad_mask"].to(device, non_blocking=True)
                struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)
                sentence = batch["sentence"].to(device, non_blocking=True)
                token_type_index = batch["token_type_index"].to(device, non_blocking=True)
                struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
                sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)

                start_token = torch.zeros((B, 1), dtype=torch.long).to(device, non_blocking=True)
                t = torch.randint(0, noise_schedule.timesteps, (B,), dtype=torch.long).to(device, non_blocking=True)

                #--------------
                x_start = get_diffusion_variables(struct_xyztheta_inputs, obj_xyztheta_inputs)
                noise = torch.randn_like(x_start, device=device)
                x_noisy = q_sample(x_start=x_start, t=t, noise_schedule=noise_schedule, noise=noise)

                struct_xyztheta_inputs = x_noisy[:, 0, :].unsqueeze(1)  # B, 1, 3 + 6
                obj_xyztheta_inputs = x_noisy[:, 1:, :]  # B, N, 3 + 6
                struct_xyztheta_outputs, obj_xyztheta_outputs = model.forward(t, xyzs, obj_xyztheta_inputs, struct_xyztheta_inputs, sentence,
                                                                              position_index, struct_position_index,
                                                                              token_type_index, struct_token_type_index,
                                                                              start_token,
                                                                              object_pad_mask, struct_pad_mask, sentence_pad_mask)

                predicted_noise = torch.cat([struct_xyztheta_outputs, obj_xyztheta_outputs], dim=1)  # B, 1+N, 9

                # important: skip computing loss for masked positions
                pad_mask = torch.cat([struct_pad_mask, object_pad_mask], dim=1)  # B, 1+N
                keep_mask = (pad_mask == 0)
                noise = noise[keep_mask]  # dim: number of positions that need loss calculation
                predicted_noise = predicted_noise[keep_mask]

                if loss_type == 'l1':
                    loss = F.l1_loss(noise, predicted_noise)
                elif loss_type == 'l2':
                    loss = F.mse_loss(noise, predicted_noise)
                elif loss_type == "huber":
                    loss = F.smooth_l1_loss(noise, predicted_noise)
                else:
                    raise NotImplementedError()
                # --------------

                loss.backward()
                if grad_clipping != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
                optimizer.step()
                epoch_loss += loss
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss})

                summary_writer.add_scalar("train_loss", loss, epoch * len(data_iter["train"]) + step)
                summary_writer.add_scalar("epoch", epoch, epoch * len(data_iter["train"]) + step)

        if warmup is not None:
            warmup.step()

        print('[Epoch:{}]:  Training Loss:{:.4}'.format(epoch, epoch_loss))

        validate_model(cfg, model, data_iter, noise_schedule, epoch, device, summary_writer)

        # evaluate(gts, predictions, ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
        #                             "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"])
        #
        # score = validate(cfg, model, data_iter["valid"], epoch, device)
        # if save_best_model and score > best_score:
        #     print("Saving best model so far...")
        #     best_score = score
        #     save_model(best_model_dir, cfg, epoch, model)

    return model


def validate_model(cfg, model, data_iter, noise_schedule, epoch, device, summary_writer):

    loss_type = cfg.diffusion.loss_type

    model.eval()

    epoch_loss = 0
    # gts = defaultdict(list)
    # predictions = defaultdict(list)
    with torch.no_grad():

        with tqdm.tqdm(total=len(data_iter["valid"])) as pbar:
            for step, batch in enumerate(data_iter["valid"]):

                # input
                xyzs = batch["xyzs"].to(device, non_blocking=True)
                B = xyzs.shape[0]
                # obj_pad_mask: we don't need it now since we are testing
                obj_xyztheta_inputs = batch["obj_xyztheta_inputs"].to(device, non_blocking=True)
                struct_xyztheta_inputs = batch["struct_xyztheta_inputs"].to(device, non_blocking=True)
                position_index = batch["position_index"].to(device, non_blocking=True)
                struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
                object_pad_mask = batch["obj_pad_mask"].to(device, non_blocking=True)
                struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)
                sentence = batch["sentence"].to(device, non_blocking=True)
                token_type_index = batch["token_type_index"].to(device, non_blocking=True)
                struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
                sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)

                start_token = torch.zeros((B, 1), dtype=torch.long).to(device, non_blocking=True)
                t = torch.randint(0, noise_schedule.timesteps, (B,), dtype=torch.long).to(device, non_blocking=True)

                # --------------
                x_start = get_diffusion_variables(struct_xyztheta_inputs, obj_xyztheta_inputs)
                noise = torch.randn_like(x_start, device=device)
                x_noisy = q_sample(x_start=x_start, t=t, noise_schedule=noise_schedule, noise=noise)

                struct_xyztheta_inputs = x_noisy[:, 0, :].unsqueeze(1)  # B, 1, 3 + 6
                obj_xyztheta_inputs = x_noisy[:, 1:, :]  # B, N, 3 + 6
                struct_xyztheta_outputs, obj_xyztheta_outputs = model.forward(t, xyzs, obj_xyztheta_inputs, struct_xyztheta_inputs, sentence,
                                                                              position_index, struct_position_index,
                                                                              token_type_index, struct_token_type_index,
                                                                              start_token,
                                                                              object_pad_mask, struct_pad_mask, sentence_pad_mask)
                predicted_noise = torch.cat([struct_xyztheta_outputs, obj_xyztheta_outputs], dim=1)

                # important: skip computing loss for masked positions
                pad_mask = torch.cat([struct_pad_mask, object_pad_mask], dim=1)  # B, 1+N
                keep_mask = (pad_mask == 0)
                noise = noise[keep_mask]  # dim: number of positions that need loss calculation
                predicted_noise = predicted_noise[keep_mask]

                if loss_type == 'l1':
                    loss = F.l1_loss(noise, predicted_noise)
                elif loss_type == 'l2':
                    loss = F.mse_loss(noise, predicted_noise)
                elif loss_type == "huber":
                    loss = F.smooth_l1_loss(noise, predicted_noise)
                else:
                    raise NotImplementedError()

                epoch_loss += loss
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss})
                summary_writer.add_scalar("valid_loss", loss, epoch * len(data_iter["valid"]) + step)

    print('[Epoch:{}]:  Val Loss:{:.4}'.format(epoch, epoch_loss))
    # summary_writer.add_scalar("valid_loss", epoch_loss, epoch)


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

    data_cfg = cfg.dataset
    tokenizer = Tokenizer(data_cfg.vocab_dir)
    vocab_size = tokenizer.get_vocab_size()

    # initialize model
    model_cfg = cfg.model
    model = TransformerDiffuserLang(vocab_size,
                                    num_attention_heads=model_cfg.num_attention_heads,
                                encoder_hidden_dim=model_cfg.encoder_hidden_dim,
                                encoder_dropout=model_cfg.encoder_dropout,
                                encoder_activation=model_cfg.encoder_activation,
                                encoder_num_layers=model_cfg.encoder_num_layers,
                                structure_dropout=model_cfg.structure_dropout,
                                object_dropout=model_cfg.object_dropout,
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

    noise_schedule = NoiseSchedule(cfg.diffusion.time_steps)

    epoch = checkpoint['epoch']
    return cfg, tokenizer, model, noise_schedule, optimizer, scheduler, epoch


def main(cfg):

    pl.seed_everything(cfg.random_seed)
    # torch.backends.cudnn.deterministic = True

    wandb_logger = WandbLogger(**cfg.WANDB)
    wandb_logger.experiment.config.update(cfg)
    checkpoint_callback = ModelCheckpoint()

    tokenizer = Tokenizer(cfg.DATASET.vocab_dir)
    vocab_size = tokenizer.get_vocab_size()

    train_dataset = SemanticArrangementDataset(split="train", tokenizer=tokenizer, **cfg.DATASET)
    valid_dataset = SemanticArrangementDataset(split="val", tokenizer=tokenizer, **cfg.DATASET)


    # train_dataset = SemanticArrangementDataset(data_roots=data_cfg.dirs,
    #                                            index_roots=data_cfg.index_dirs,
    #                                            split="train",
    #                                            tokenizer=tokenizer,
    #                                            max_num_objects=data_cfg.max_num_objects,
    #                                            max_num_other_objects=data_cfg.max_num_other_objects,
    #                                            max_num_shape_parameters=data_cfg.max_num_shape_parameters,
    #                                            max_num_rearrange_features=data_cfg.max_num_rearrange_features,
    #                                            max_num_anchor_features=data_cfg.max_num_anchor_features,
    #                                            num_pts=data_cfg.num_pts,
    #                                            filter_num_moved_objects_range=data_cfg.filter_num_moved_objects_range,
    #                                            data_augmentation=False,
    #                                            shuffle_object_index=data_cfg.shuffle_object_index)
    #
    # valid_dataset = SemanticArrangementDataset(data_roots=data_cfg.dirs,
    #                                            index_roots=data_cfg.index_dirs,
    #                                            split="valid",
    #                                            tokenizer=tokenizer,
    #                                            max_num_objects=data_cfg.max_num_objects,
    #                                            max_num_other_objects=data_cfg.max_num_other_objects,
    #                                            max_num_shape_parameters=data_cfg.max_num_shape_parameters,
    #                                            max_num_rearrange_features=data_cfg.max_num_rearrange_features,
    #                                            max_num_anchor_features=data_cfg.max_num_anchor_features,
    #                                            num_pts=data_cfg.num_pts,
    #                                            filter_num_moved_objects_range=data_cfg.filter_num_moved_objects_range,
    #                                            data_augmentation=False,
    #                                            shuffle_object_index=data_cfg.shuffle_object_index)

    # data_iter = {}
    train_dataloader = DataLoader(train_dataset, shuffle=False, **cfg.DATALOADER)
    valid_dataloader = DataLoader(valid_dataset, shuffle=True, **cfg.DATALOADER)
    #     data_iter["valid"] = DataLoader(valid_dataset, batch_size=data_cfg.batch_size, shuffle=False,
    #                                     pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)

    # load model
    model = ConditionalPoseDiffusionModel(vocab_size, cfg.MODEL, cfg.LOSS, cfg.NOISE_SCHEDULE, cfg.OPTIMIZER)

    # model = ConditionalPoseDiffusionModel(vocab_size,
    #                                 num_attention_heads=model_cfg.num_attention_heads,
    #                             encoder_hidden_dim=model_cfg.encoder_hidden_dim,
    #                             encoder_dropout=model_cfg.encoder_dropout,
    #                             encoder_activation=model_cfg.encoder_activation,
    #                             encoder_num_layers=model_cfg.encoder_num_layers,
    #                             structure_dropout=model_cfg.structure_dropout,
    #                             object_dropout=model_cfg.object_dropout,
    #                             ignore_rgb=model_cfg.ignore_rgb)
    # model.to(cfg.device)


    trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback], **cfg.TRAINING.TRAINER)

    # training_cfg = cfg.training
    # optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate, weight_decay=training_cfg.l2)
    # scheduler = None
    # warmup = None
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg.lr_restart)
    # warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=training_cfg.warmup,
    #                                 after_scheduler=scheduler)

    # noise_schedule = NoiseSchedule(cfg.diffusion.time_steps)

    # train_model(cfg, model, data_iter, noise_schedule, optimizer, warmup, training_cfg.max_epochs, cfg.device,
    #             cfg.save_best_model, summary_writer)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # # save model
    # if cfg.save_model:
    #     model_dir = os.path.join(cfg.experiment_dir, "model")
    #     print("Saving model to {}".format(model_dir))
    #     if not os.path.exists(model_dir):
    #         os.makedirs(model_dir)
    #     save_model(model_dir, cfg, cfg.max_epochs, model, optimizer, scheduler)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--base_config_file", help='base config yaml file',
                        default='../configs/base.yaml',
                        type=str)
    parser.add_argument("--config_file", help='config yaml file',
                        default='../configs/conditional_pose_diffusion.yaml',
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

    main(cfg)