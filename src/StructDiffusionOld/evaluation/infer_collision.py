import torch
from torch.utils.data import DataLoader

import StructDiffusion.data.dataset_pairwise_collision as collision_dataset
import StructDiffusion.training.train_pairwise_collision_detector as collision_model


class CollisionInference:

    def __init__(self, model_dir, empty_dataset=False):

        cfg, model, _, _, _ = collision_model.load_model(model_dir)

        data_cfg = cfg.dataset

        if empty_dataset:
            data_cfg.dirs = []
            data_cfg.index_dirs = []
            data_cfg.urdf_pc_idx_file = None
            data_cfg.collision_data_dir = None

        full_dataset = collision_dataset.PairwiseCollisionDataset(data_cfg.dirs, data_cfg.index_dirs, data_cfg.urdf_pc_idx_file,
                                                data_cfg.collision_data_dir,
                                                num_pts=data_cfg.num_pts,
                                                num_scene_pts=data_cfg.num_scene_pts,
                                                normalize_pc=data_cfg.normalize_pc,
                                                random_rotation=data_cfg.random_rotation,
                                                data_augmentation=data_cfg.data_augmentation)

        if not empty_dataset:
            train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset) * 0.7),
                                                                                        len(full_dataset) - int(
                                                                                            len(full_dataset) * 0.7)],
                                                                         generator=torch.Generator().manual_seed(
                                                                             cfg.random_seed))
            self.dataset = valid_dataset
        else:
            self.dataset = full_dataset

        self.model = model
        self.cfg = cfg
        self.epoch = 0  # not important
        self.device = cfg.device
        self.full_dataset = full_dataset

    # def try_this(self):
    #
    #     self.model.eval()
    #     for d in self.dataset:
    #         raw_scene_xyz = d["scene_xyz"]
    #         scene_xyz = d["scene_xyz"].unsqueeze(0).to(self.device)
    #         gt_label = d["is_circle"].unsqueeze(0).to(self.device)
    #
    #     # self.model.eval()
    #     # for batch in data_iter:
    #     #     scene_xyz = batch["scene_xyz"].to(self.device, non_blocking=True)
    #     #     gt_label = batch["is_circle"].to(self.device, non_blocking=True)
    #         print(scene_xyz.shape)
    #         print(gt_label.shape)
    #         with torch.no_grad():
    #             pred = self.model.forward(scene_xyz)
    #             pred_label = self.model.convert_logits(pred)
    #         print("gt:", gt_label)
    #         print("pred:", pred_label)
    #
    #         show_pcs([raw_scene_xyz[:, 0:3]], [np.tile(np.array([0, 1, 0], dtype=np.float), (raw_scene_xyz.shape[0], 1))],
    #                  add_coordinate_frame=True)