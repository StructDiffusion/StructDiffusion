from torch.utils.data import DataLoader
import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from StructDiffusion.data.semantic_arrangement import SemanticArrangementDataset
from StructDiffusion.language.tokenizer import Tokenizer
from StructDiffusion.models.pl_models import ConditionalPoseDiffusionModel


def main(cfg):

    pl.seed_everything(cfg.random_seed)

    wandb_logger = WandbLogger(**cfg.WANDB)
    wandb_logger.experiment.config.update(cfg)
    checkpoint_callback = ModelCheckpoint()

    tokenizer = Tokenizer(cfg.DATASET.vocab_dir)
    vocab_size = tokenizer.get_vocab_size()

    train_dataset = SemanticArrangementDataset(split="train", tokenizer=tokenizer, **cfg.DATASET)
    valid_dataset = SemanticArrangementDataset(split="valid", tokenizer=tokenizer, **cfg.DATASET)
    train_dataloader = DataLoader(train_dataset, shuffle=True, **cfg.DATALOADER)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, **cfg.DATALOADER)

    model = ConditionalPoseDiffusionModel(vocab_size, cfg.MODEL, cfg.LOSS, cfg.NOISE_SCHEDULE, cfg.OPTIMIZER)

    trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback], **cfg.TRAINER)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--base_config_file", help='base config yaml file',
                        default='../configs/base.yaml',
                        type=str)
    parser.add_argument("--config_file", help='config yaml file',
                        default='../configs/conditional_pose_diffusion.yaml',
                        type=str)
    args = parser.parse_args()
    base_cfg = OmegaConf.load(args.base_config_file)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(base_cfg, cfg)

    main(cfg)