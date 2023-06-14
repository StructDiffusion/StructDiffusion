import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from StructDiffusion.models.models import TransformerDiffusionModel, PCTDiscriminator, FocalLoss

from StructDiffusion.diffusion.noise_schedule import NoiseSchedule, q_sample
from StructDiffusion.diffusion.pose_conversion import get_diffusion_variables_from_H, get_diffusion_variables_from_9D_actions


class ConditionalPoseDiffusionModel(pl.LightningModule):

    def __init__(self, vocab_size, model_cfg, loss_cfg, noise_scheduler_cfg, optimizer_cfg):
        super().__init__()
        self.save_hyperparameters()

        self.model = TransformerDiffusionModel(vocab_size, **model_cfg)

        self.noise_schedule = NoiseSchedule(**noise_scheduler_cfg)

        self.loss_type = loss_cfg.type

        self.optimizer_cfg = optimizer_cfg
        self.configure_optimizers()

        self.batch_size = None

    def forward(self, batch):

        # input
        pcs = batch["pcs"]
        B = pcs.shape[0]
        self.batch_size = B
        sentence = batch["sentence"]
        goal_poses = batch["goal_poses"]
        type_index = batch["type_index"]
        position_index = batch["position_index"]
        pad_mask = batch["pad_mask"]

        t = torch.randint(0, self.noise_schedule.timesteps, (B,), dtype=torch.long).to(self.device)

        # --------------
        x_start = get_diffusion_variables_from_H(goal_poses)
        noise = torch.randn_like(x_start, device=self.device)
        x_noisy = q_sample(x_start=x_start, t=t, noise_schedule=self.noise_schedule, noise=noise)

        predicted_noise = self.model.forward(t, pcs, sentence, x_noisy, type_index, position_index, pad_mask)

        # important: skip computing loss for masked positions
        num_poses = goal_poses.shape[1]  # B, N, 4, 4
        pose_pad_mask = pad_mask[:, -num_poses:]
        keep_mask = (pose_pad_mask == 0)
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

        self.log(prefix + "loss", loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        noise, pred_noise = self.forward(batch)
        loss = self.compute_loss(noise, pred_noise, prefix="train/")
        return loss

    def validation_step(self, batch, batch_idx):
        noise, pred_noise = self.forward(batch)
        loss = self.compute_loss(noise, pred_noise, prefix="val/")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr, weight_decay=self.optimizer_cfg.weight_decay) # 1e-5
        return optimizer


class PairwiseCollisionModel(pl.LightningModule):

    def __init__(self, model_cfg, loss_cfg, optimizer_cfg):
        super().__init__()
        self.save_hyperparameters()

        self.model = PCTDiscriminator(**model_cfg)

        self.loss_cfg = loss_cfg
        self.loss = None
        self.configure_loss()

        self.optimizer_cfg = optimizer_cfg
        self.configure_optimizers()

    def forward(self, batch):
        label = batch["label"]
        predicted_label = self.model.forward(batch["scene_xyz"])
        return label, predicted_label

    def compute_loss(self, label, predicted_label, prefix="train/"):
        if self.loss_cfg.type == "MSE":
            predicted_label = torch.sigmoid(predicted_label)
        loss = self.loss(predicted_label, label)
        self.log(prefix + "loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        label, predicted_label = self.forward(batch)
        loss = self.compute_loss(label, predicted_label, prefix="train/")
        return loss

    def validation_step(self, batch, batch_idx):
        label, predicted_label = self.forward(batch)
        loss = self.compute_loss(label, predicted_label, prefix="val/")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr, weight_decay=self.optimizer_cfg.weight_decay) # 1e-5
        return optimizer

    def configure_loss(self):
        if self.loss_cfg.type == "Focal":
            print("use focal loss with gamma {}".format(self.loss_cfg.focal_gamma))
            self.loss = FocalLoss(gamma=self.loss_cfg.focal_gamma)
        elif self.loss_cfg.type == "MSE":
            print("use regression L2 loss")
            self.loss = torch.nn.MSELoss()
        elif self.loss_cfg.type == "BCE":
            print("use standard BCE logit loss")
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")