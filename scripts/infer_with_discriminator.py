import os
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf

from StructDiffusion.data.semantic_arrangement import SemanticArrangementDataset
from StructDiffusion.language.tokenizer import Tokenizer
from StructDiffusion.models.pl_models import ConditionalPoseDiffusionModel, PairwiseCollisionModel
from StructDiffusion.diffusion.sampler import SamplerV2
from StructDiffusion.utils.files import get_checkpoint_path_from_dir, replace_config_for_testing_data
from StructDiffusion.utils.batch_inference import move_pc_and_create_scene_simple, visualize_batch_pcs


def main(args, cfg):

    pl.seed_everything(args.eval_random_seed)

    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    diffusion_checkpoint_path = get_checkpoint_path_from_dir(os.path.join(cfg.WANDB.save_dir, cfg.WANDB.project, args.diffusion_checkpoint_id, "checkpoints"))
    collision_checkpoint_path = get_checkpoint_path_from_dir(os.path.join(cfg.WANDB.save_dir, cfg.WANDB.project, args.collision_checkpoint_id, "checkpoints"))

    if args.eval_mode == "infer":

        tokenizer = Tokenizer(cfg.DATASET.vocab_dir)
        # override ignore_rgb for visualization
        cfg.DATASET.ignore_rgb = False
        dataset = SemanticArrangementDataset(split="test", tokenizer=tokenizer, **cfg.DATASET)

        sampler = SamplerV2(ConditionalPoseDiffusionModel, diffusion_checkpoint_path,
                            PairwiseCollisionModel, collision_checkpoint_path, device)

        data_idxs = np.random.permutation(len(dataset))
        for di in data_idxs:
            raw_datum = dataset.get_raw_data(di)
            print(tokenizer.convert_structure_params_to_natural_language(raw_datum["sentence"]))
            datum = dataset.convert_to_tensors(raw_datum, tokenizer)
            batch = dataset.single_datum_to_batch(datum, args.num_samples, device, inference_mode=True)

            num_poses = datum["goal_poses"].shape[0]
            struct_pose, pc_poses_in_struct = sampler.sample(batch, num_poses)

            new_obj_xyzs = move_pc_and_create_scene_simple(batch["pcs"], struct_pose, pc_poses_in_struct)
            visualize_batch_pcs(new_obj_xyzs, args.num_samples, limit_B=10, trimesh=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer")
    parser.add_argument("--base_config_file", help='base config yaml file',
                        default='../configs/base.yaml',
                        type=str)
    parser.add_argument("--config_file", help='config yaml file',
                        default='../configs/conditional_pose_diffusion.yaml',
                        type=str)
    parser.add_argument("--testing_data_config_file", help='config yaml file',
                        default='../configs/testing_data.yaml',
                        type=str)
    parser.add_argument("--diffusion_checkpoint_id",
                        default="ConditionalPoseDiffusion",
                        type=str)
    parser.add_argument("--collision_checkpoint_id",
                        default="CollisionDiscriminator",
                        type=str)
    parser.add_argument("--eval_mode",
                        default="infer",
                        type=str)
    parser.add_argument("--eval_random_seed",
                        default=42,
                        type=int)
    parser.add_argument("--num_samples",
                        default=10,
                        type=int)
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.base_config_file)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(base_cfg, cfg)

    testing_data_cfg = OmegaConf.load(args.testing_data_config_file)
    testing_data_cfg = OmegaConf.merge(base_cfg, testing_data_cfg)
    replace_config_for_testing_data(cfg, testing_data_cfg)

    main(args, cfg)


