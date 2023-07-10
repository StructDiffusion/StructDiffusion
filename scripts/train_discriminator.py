import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from StructDiffusion.data.pairwise_collision import PairwiseCollisionDataset
from StructDiffusion.models.pl_models import PairwiseCollisionModel


def main(cfg):

    pl.seed_everything(cfg.random_seed)

    wandb_logger = WandbLogger(**cfg.WANDB)
    wandb_logger.experiment.config.update(cfg)
    checkpoint_callback = ModelCheckpoint()

    full_dataset = PairwiseCollisionDataset(**cfg.DATASET)
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset) * 0.7), len(full_dataset) - int(len(full_dataset) * 0.7)])
    train_dataloader = DataLoader(train_dataset, shuffle=True, **cfg.DATALOADER)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, **cfg.DATALOADER)

    model = PairwiseCollisionModel(cfg.MODEL, cfg.LOSS, cfg.OPTIMIZER, cfg.DATASET)

    trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback], **cfg.TRAINER)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--base_config_file", help='base config yaml file',
                        default='../configs/base.yaml',
                        type=str)
    parser.add_argument("--config_file", help='config yaml file',
                        default='../configs/pairwise_collision.yaml',
                        type=str)
    args = parser.parse_args()
    base_cfg = OmegaConf.load(args.base_config_file)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(base_cfg, cfg)

    main(cfg)