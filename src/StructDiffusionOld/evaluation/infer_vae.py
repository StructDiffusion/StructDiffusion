import torch
from torch.utils.data import DataLoader

import StructDiffusion.data.dataset_v23_continuous_out_ar_6d_all_objects_all_shapes as prior_dataset
import StructDiffusion.training.train_actor_vae_3networks_language_all_shapes as prior_model

class PriorInference:

    def __init__(self, model_dir, data_split="test", test_specific_shape=None, override_data_dirs=None, override_index_dirs=None):
        # load prior
        cfg, tokenizer, model, optimizer, scheduler, epoch = prior_model.load_model(model_dir, ngc_vocab=False)

        data_cfg = cfg.dataset

        # data_cfg.dir = "/home/weiyu/data_drive/examples_local_200/result"
        # data_cfg.index_dir = "index_200"
        if test_specific_shape is not None:
            dirs = data_cfg.dirs
            index_dirs = data_cfg.index_dirs
            shape_dirs_index_dirs = []
            for d, id in zip(dirs, index_dirs):
                if test_specific_shape in d:
                    shape_dirs_index_dirs.append((d, id))
            data_cfg.dirs = [s[0] for s in shape_dirs_index_dirs]
            data_cfg.index_dirs = [s[1] for s in shape_dirs_index_dirs]

        if override_data_dirs is None:
            override_data_dirs = data_cfg.dirs
        if override_index_dirs is None:
            override_index_dirs = data_cfg.index_dirs

        dataset = prior_dataset.SemanticArrangementDataset(override_data_dirs, override_index_dirs, data_split, tokenizer,
                                                           data_cfg.max_num_objects,
                                                           data_cfg.max_num_other_objects,
                                                           data_cfg.max_num_shape_parameters,
                                                           data_cfg.max_num_rearrange_features,
                                                           data_cfg.max_num_anchor_features,
                                                           data_cfg.num_pts,
                                                           data_augmentation=data_cfg.data_augmentation)

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model
        self.cfg = cfg
        self.dataset = dataset
        self.epoch = epoch

    def validate(self):
        data_cfg = self.cfg.dataset
        data_iter = DataLoader(self.dataset, batch_size=data_cfg.batch_size, shuffle=False,
                               collate_fn=prior_dataset.SemanticArrangementDataset.collate_fn,
                               pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)

        prior_model.validate(self.cfg, self.model, data_iter, self.epoch, self.cfg.device)

    def limited_batch_inference(self, data, verbose=True, convert_to_tensors=True, return_numpy=True):
        """
        This function makes the assumption that scenes in the batch have the same number of objects that need to be
        rearranged

        :param data:
        :param model:
        :param test_dataset:
        :param tokenizer:
        :param cfg:
        :param num_samples:
        :param verbose:
        :return:
        """
        data_size = len(data)
        batch_size = self.cfg.dataset.batch_size
        if verbose:
            print("data size:", data_size)
            print("batch size:", batch_size)

        num_batches = int(data_size / batch_size)
        if data_size % batch_size != 0:
            num_batches += 1

        all_obj_preds = []
        all_struct_preds = []
        for b in range(num_batches):
            if b + 1 == num_batches:
                # last batch
                batch = data[b * batch_size:]
            else:
                batch = data[b * batch_size: (b+1) * batch_size]
            if convert_to_tensors:
                data_tensors = [self.dataset.convert_to_tensors(d, self.tokenizer) for d in batch]
            else:
                data_tensors = batch
            data_tensors = self.dataset.collate_fn(data_tensors)
            predictions = prior_model.infer_once(self.cfg, self.model, data_tensors, self.cfg.device)

            obj_x_preds = torch.cat(predictions["obj_x_outputs"], dim=0)
            obj_y_preds = torch.cat(predictions["obj_y_outputs"], dim=0)
            obj_z_preds = torch.cat(predictions["obj_z_outputs"], dim=0)
            obj_theta_preds = torch.cat(predictions["obj_theta_outputs"], dim=0)
            obj_preds = torch.cat([obj_x_preds, obj_y_preds, obj_z_preds, obj_theta_preds], dim=1)  # batch_size * max num objects, output_dim

            struct_x_preds = torch.cat(predictions["struct_x_inputs"], dim=0)
            struct_y_preds = torch.cat(predictions["struct_y_inputs"], dim=0)
            struct_z_preds = torch.cat(predictions["struct_z_inputs"], dim=0)
            struct_theta_preds = torch.cat(predictions["struct_theta_inputs"], dim=0)
            struct_preds = torch.cat([struct_x_preds, struct_y_preds, struct_z_preds, struct_theta_preds], dim=1) # batch_size, output_dim

            all_obj_preds.append(obj_preds)
            all_struct_preds.append(struct_preds)

        obj_preds = torch.cat(all_obj_preds, dim=0)  # data_size * max num objects, output_dim
        struct_preds = torch.cat(all_struct_preds, dim=0)  # data_size, output_dim

        if return_numpy:
            obj_preds = obj_preds.detach().cpu().numpy()
            struct_preds = struct_preds.detach().cpu().numpy()

        obj_preds = obj_preds.reshape(data_size, -1, obj_preds.shape[-1])  # batch_size, max num objects, output_dim

        return struct_preds, obj_preds