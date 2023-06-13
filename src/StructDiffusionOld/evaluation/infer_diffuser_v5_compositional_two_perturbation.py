import tqdm
import torch
from torch.utils.data import DataLoader

from StructDiffusion.utils.torch_data import default_collate
from StructDiffusion.training.train_diffuser_v5_compositional_two_perturbation import load_model, get_diffusion_variables, extract, get_struct_objs_poses, move_pc_and_create_scene, visualize_batch_pcs
from StructDiffusion.data.dataset_v3_diffuser_compositional_two_perturbation import SemanticArrangementDataset


class DiffuserInference:

    def __init__(self, model_dir, data_split="test", override_data_dirs=None, override_index_dirs=None, invalid_noise_weight=2.0):

        # load prior
        cfg, tokenizer, model, noise_schedule, optimizer, scheduler, epoch = load_model(model_dir)

        data_cfg = cfg.dataset
        if override_data_dirs is None:
            override_data_dirs = data_cfg.dirs
        if override_index_dirs is None:
            override_index_dirs = data_cfg.index_dirs
        dataset = SemanticArrangementDataset(
            data_roots=override_data_dirs,
            index_roots=override_index_dirs,
            split=data_split,
            tokenizer=tokenizer,
            # important: set probability to sample negative data
            perturbation_probability=0.0,
            perturbation_mode=data_cfg.perturbation_mode,
            min_translation=data_cfg.min_translation,
            max_translation=data_cfg.max_translation,
            min_rotation=data_cfg.min_rotation,
            max_rotation=data_cfg.max_rotation,
            max_num_objects=data_cfg.max_num_objects,
            max_num_other_objects=data_cfg.max_num_other_objects,
            max_num_shape_parameters=data_cfg.max_num_shape_parameters,
            max_num_rearrange_features=data_cfg.max_num_rearrange_features,
            max_num_anchor_features=data_cfg.max_num_anchor_features,
            num_pts=data_cfg.num_pts,
            filter_num_moved_objects_range=data_cfg.filter_num_moved_objects_range,
            data_augmentation=False,
            shuffle_object_index=False)

        self.noise_schedule = noise_schedule
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model
        self.dataset = dataset
        self.epoch = epoch

        self.invalid_noise_weight = invalid_noise_weight

    def limited_batch_inference(self, data, num_samples, convert_to_tensors=True, return_numpy=True):
        """

        :param data: raw data dict from the dataloader
        :param num_samples: how many samples to generate for the given scene
        :param convert_to_tensors:
        :param return_numpy:
        :return:
        """

        device = self.cfg.device
        noise_schedule = self.noise_schedule

        self.model.eval()
        with torch.no_grad():

            if convert_to_tensors:
                data = self.dataset.convert_to_tensors(data, self.tokenizer)
            batch = default_collate([data])

            # input
            xyzs = batch["xyzs"].to(device, non_blocking=True)
            B, N, P, _ = xyzs.shape
            obj_xyztheta_inputs = batch["obj_xyztheta_inputs"].to(device, non_blocking=True)
            struct_xyztheta_inputs = batch["struct_xyztheta_inputs"].to(device, non_blocking=True)
            position_index = batch["position_index"].to(device, non_blocking=True)
            struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
            start_token = torch.zeros((B, 1), dtype=torch.long).to(device, non_blocking=True)
            object_pad_mask = batch["obj_pad_mask"].to(device, non_blocking=True)
            struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)
            valid_sentence = batch["sentence"].to(device, non_blocking=True)
            token_type_index = batch["token_type_index"].to(device, non_blocking=True)
            struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
            sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)

            raw_valid_sentence = eval(batch["raw_sentence"][0])
            raw_invalid_sentence = []
            for tv, tt in raw_valid_sentence:
                if tt == "physics":
                    assert tv == "valid"
                    raw_invalid_sentence.append(("invalid", tt))
                else:
                    raw_invalid_sentence.append((tv, tt))
            print("raw valid sentence:", raw_valid_sentence)
            print("raw invalid sentence:", raw_invalid_sentence)
            invalid_sentence = torch.LongTensor([self.tokenizer.tokenize(*i) for i in raw_invalid_sentence]).to(device, non_blocking=True)

            # repeat num_samples times for this scene
            assert B == 1
            xyzs = xyzs.repeat(num_samples, 1, 1, 1)
            obj_xyztheta_inputs = obj_xyztheta_inputs.repeat(num_samples, 1, 1)
            struct_xyztheta_inputs = struct_xyztheta_inputs.repeat(num_samples, 1, 1)
            position_index = position_index.repeat(num_samples, 1)
            struct_position_index = struct_position_index.repeat(num_samples, 1)
            start_token = start_token.repeat(num_samples, 1)
            object_pad_mask = object_pad_mask.repeat(num_samples, 1)
            struct_pad_mask = struct_pad_mask.repeat(num_samples, 1)

            valid_sentence = valid_sentence.repeat(num_samples, 1)
            invalid_sentence = invalid_sentence.repeat(num_samples, 1)

            token_type_index = token_type_index.repeat(num_samples, 1)
            struct_token_type_index = struct_token_type_index.repeat(num_samples, 1)
            sentence_pad_mask = sentence_pad_mask.repeat(num_samples, 1)
            B = num_samples

            # start diffusion
            x_gt = get_diffusion_variables(struct_xyztheta_inputs, obj_xyztheta_inputs)
            x = torch.randn_like(x_gt, device=device)
            # TODO: have the option to only save the prev step to save memory
            xs = []
            for t_index in tqdm.tqdm(reversed(range(0, noise_schedule.timesteps)), desc='sampling loop time step',
                                     total=noise_schedule.timesteps):

                # get noise params
                t = torch.full((B,), t_index, device=device, dtype=torch.long)
                betas_t = extract(noise_schedule.betas, t, x.shape)
                sqrt_one_minus_alphas_cumprod_t = extract(noise_schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
                sqrt_recip_alphas_t = extract(noise_schedule.sqrt_recip_alphas, t, x.shape)

                # predict noise
                struct_xyztheta_inputs = x[:, 0, :].unsqueeze(1)  # B, 1, 3 + 6
                obj_xyztheta_inputs = x[:, 1:, :]  # B, N, 3 + 6

                # run valid
                struct_xyztheta_outputs, obj_xyztheta_outputs = self.model.forward(t, xyzs, obj_xyztheta_inputs,
                                                                              struct_xyztheta_inputs, valid_sentence,
                                                                              position_index, struct_position_index,
                                                                              token_type_index, struct_token_type_index,
                                                                              start_token,
                                                                              object_pad_mask, struct_pad_mask,
                                                                              sentence_pad_mask)
                predicted_noise_valid = torch.cat([struct_xyztheta_outputs, obj_xyztheta_outputs], dim=1)
                # run invalid
                struct_xyztheta_outputs, obj_xyztheta_outputs = self.model.forward(t, xyzs, obj_xyztheta_inputs,
                                                                              struct_xyztheta_inputs, invalid_sentence,
                                                                              position_index, struct_position_index,
                                                                              token_type_index, struct_token_type_index,
                                                                              start_token,
                                                                              object_pad_mask, struct_pad_mask,
                                                                              sentence_pad_mask)
                predicted_noise_invalid = torch.cat([struct_xyztheta_outputs, obj_xyztheta_outputs], dim=1)
                # combine valid and invalid
                predicted_noise = (1 + self.invalid_noise_weight) * predicted_noise_valid - self.invalid_noise_weight * predicted_noise_invalid

                # compute noisy x at t
                model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
                if t_index == 0:
                    x = model_mean
                else:
                    posterior_variance_t = extract(noise_schedule.posterior_variance, t, x.shape)
                    noise = torch.randn_like(x)
                    # Algorithm 2 line 4:
                    x = model_mean + torch.sqrt(posterior_variance_t) * noise

                xs.append(x)

            # get final result
            xs = list(reversed(xs))
            struct_pose, pc_poses_in_struct = get_struct_objs_poses(xs[0])
            # struct_pose: B, 1, 4, 4
            # pc_poses_in_struct: B, N, 4, 4

        if return_numpy:
            struct_pose = struct_pose.detach().cpu().numpy()
            pc_poses_in_struct = pc_poses_in_struct.detach().cpu().numpy()

        return struct_pose, pc_poses_in_struct





def run(model_dir, num_samples=10):

    # load model
    cfg, tokenizer, model, noise_schedule, optimizer, scheduler, epoch = load_model(model_dir)
    model.eval()
    device = cfg.device

    # load data
    data_cfg = cfg.dataset
    # test_dataset = SemanticArrangementDataset(data_roots=["/home/weiyu/data_drive/data_test_objects/dinner_data/result"],
    #                                           index_roots=["/home/weiyu/data_drive/data_test_objects/dinner_data/result/index"],
    #                                           # data_roots=data_cfg.dirs,
    #                                           # index_roots=data_cfg.index_dirs,
    #                                           split="test",
    #                                           tokenizer=tokenizer,
    #                                           max_num_objects=data_cfg.max_num_objects,
    #                                           max_num_other_objects=data_cfg.max_num_other_objects,
    #                                           max_num_shape_parameters=data_cfg.max_num_shape_parameters,
    #                                           max_num_rearrange_features=data_cfg.max_num_rearrange_features,
    #                                           max_num_anchor_features=data_cfg.max_num_anchor_features,
    #                                           num_pts=data_cfg.num_pts,
    #                                           filter_num_moved_objects_range=data_cfg.filter_num_moved_objects_range,
    #                                           data_augmentation=False,
    #                                           shuffle_object_index=False)

    test_dataset = SemanticArrangementDataset(data_roots=["/home/weiyu/data_drive/data_test_objects/dinner_data/result"],
                                              index_roots=["/home/weiyu/data_drive/data_test_objects/dinner_data/result/index"],
                                              # data_roots=data_cfg.dirs,
                                              #index_roots=data_cfg.index_dirs,
                                               split="test",
                                               tokenizer=tokenizer,
                                              # important: set probability to sample negative data
                                               perturbation_probability=1.0,
                                               perturbation_mode=data_cfg.perturbation_mode,
                                               min_translation=data_cfg.min_translation,
                                               max_translation=data_cfg.max_translation,
                                               min_rotation=data_cfg.min_rotation,
                                               max_rotation=data_cfg.max_rotation,
                                               max_num_objects=data_cfg.max_num_objects,
                                               max_num_other_objects=data_cfg.max_num_other_objects,
                                               max_num_shape_parameters=data_cfg.max_num_shape_parameters,
                                               max_num_rearrange_features=data_cfg.max_num_rearrange_features,
                                               max_num_anchor_features=data_cfg.max_num_anchor_features,
                                               num_pts=data_cfg.num_pts,
                                               filter_num_moved_objects_range=data_cfg.filter_num_moved_objects_range,
                                               data_augmentation=False,
                                               shuffle_object_index=data_cfg.shuffle_object_index)


    data_iter = {}
    data_iter["test"] = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                   # collate_fn=SemanticArrangementDataset.collate_fn,
                                   pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)


    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter["test"]):

            # input
            xyzs = batch["xyzs"].to(device, non_blocking=True)
            B, N, P, _ = xyzs.shape
            obj_xyztheta_inputs = batch["obj_xyztheta_inputs"].to(device, non_blocking=True)
            struct_xyztheta_inputs = batch["struct_xyztheta_inputs"].to(device, non_blocking=True)
            position_index = batch["position_index"].to(device, non_blocking=True)
            struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
            start_token = torch.zeros((B, 1), dtype=torch.long).to(device, non_blocking=True)
            object_pad_mask = batch["obj_pad_mask"].to(device, non_blocking=True)
            struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)
            sentence = batch["sentence"].to(device, non_blocking=True)
            token_type_index = batch["token_type_index"].to(device, non_blocking=True)
            struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
            sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)

            raw_sentence = eval(batch["raw_sentence"][0])
            print("raw sentence:", raw_sentence)

            # repeat num_samples times for this scene
            assert B == 1
            xyzs = xyzs.repeat(num_samples, 1, 1, 1)
            obj_xyztheta_inputs = obj_xyztheta_inputs.repeat(num_samples, 1, 1)
            struct_xyztheta_inputs = struct_xyztheta_inputs.repeat(num_samples, 1, 1)
            position_index = position_index.repeat(num_samples, 1)
            struct_position_index = struct_position_index.repeat(num_samples, 1)
            start_token = start_token.repeat(num_samples, 1)
            object_pad_mask = object_pad_mask.repeat(num_samples, 1)
            struct_pad_mask = struct_pad_mask.repeat(num_samples, 1)
            sentence = sentence.repeat(num_samples, 1)
            token_type_index = token_type_index.repeat(num_samples, 1)
            struct_token_type_index = struct_token_type_index.repeat(num_samples, 1)
            sentence_pad_mask = sentence_pad_mask.repeat(num_samples, 1)
            B = num_samples

            # start diffusion
            x_gt = get_diffusion_variables(struct_xyztheta_inputs, obj_xyztheta_inputs)
            x = torch.randn_like(x_gt, device=device)
            xs = []
            for t_index in tqdm.tqdm(reversed(range(0, noise_schedule.timesteps)), desc='sampling loop time step',
                                     total=noise_schedule.timesteps):
            # for t_index in tqdm.tqdm(reversed(range(0, 1)), desc='sampling loop time step',total=1):

                # get noise params
                t = torch.full((B,), t_index, device=device, dtype=torch.long)
                betas_t = extract(noise_schedule.betas, t, x.shape)
                sqrt_one_minus_alphas_cumprod_t = extract(noise_schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
                sqrt_recip_alphas_t = extract(noise_schedule.sqrt_recip_alphas, t, x.shape)

                # predict noise
                struct_xyztheta_inputs = x[:, 0, :].unsqueeze(1)  # B, 1, 3 + 6
                obj_xyztheta_inputs = x[:, 1:, :]  # B, N, 3 + 6
                struct_xyztheta_outputs, obj_xyztheta_outputs = model.forward(t, xyzs, obj_xyztheta_inputs, struct_xyztheta_inputs, sentence,
                                                                              position_index, struct_position_index,
                                                                              token_type_index, struct_token_type_index,
                                                                              start_token,
                                                                              object_pad_mask, struct_pad_mask, sentence_pad_mask)

                predicted_noise = torch.cat([struct_xyztheta_outputs, obj_xyztheta_outputs], dim=1)

                # compute noisy x at t
                model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
                if t_index == 0:
                    x = model_mean
                else:
                    posterior_variance_t = extract(noise_schedule.posterior_variance, t, x.shape)
                    noise = torch.randn_like(x)
                    # Algorithm 2 line 4:
                    x = model_mean + torch.sqrt(posterior_variance_t) * noise

                xs.append(x)

            # for t_index in tqdm.tqdm(reversed(range(0, 5)), desc='sampling loop time step',
            #                          total=5):
            #
            #     # get noise params
            #     t = torch.full((B,), t_index, device=device, dtype=torch.long)
            #     betas_t = extract(noise_schedule.betas, t, x.shape)
            #     sqrt_one_minus_alphas_cumprod_t = extract(noise_schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
            #     sqrt_recip_alphas_t = extract(noise_schedule.sqrt_recip_alphas, t, x.shape)
            #
            #     # predict noise
            #     struct_xyztheta_inputs = x[:, 0, :].unsqueeze(1)  # B, 1, 3 + 6
            #     obj_xyztheta_inputs = x[:, 1:, :]  # B, N, 3 + 6
            #     struct_xyztheta_outputs, obj_xyztheta_outputs = model.forward(t, xyzs, obj_xyztheta_inputs,
            #                                                                   struct_xyztheta_inputs,
            #                                                                   position_index, struct_position_index,
            #                                                                   start_token)
            #     predicted_noise = torch.cat([struct_xyztheta_outputs, obj_xyztheta_outputs], dim=1)
            #
            #     # compute noisy x at t
            #     model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
            #     if t_index == 0:
            #         x = model_mean
            #     else:
            #         posterior_variance_t = extract(noise_schedule.posterior_variance, t, x.shape)
            #         noise = torch.randn_like(x)
            #         # Algorithm 2 line 4:
            #         x = model_mean + torch.sqrt(posterior_variance_t) * noise
            #
            #     xs.append(x)

            xs = list(reversed(xs))

            # visualize x
            # for vis_t in tqdm.tqdm([int(tt) for tt in np.ceil(np.linspace(0**0.5, 199**0.5, 20) ** 2)], desc='visualize iteration time step',):
            # # for vis_t in [0]:
            #     print(vis_t)
            #     struct_pose, pc_poses_in_struct = get_struct_objs_poses(xs[vis_t])
            #     # print(struct_pose)
            #     # print(pc_poses_in_struct)
            #     new_obj_xyzs = move_pc_and_create_scene(xyzs, struct_pose, pc_poses_in_struct)
            #     num_target_objs = torch.sum(object_pad_mask == 0)
            #     # print(object_pad_mask)
            #     # print(num_target_objs)
            #     new_obj_xyzs = new_obj_xyzs[:, :num_target_objs]
            #     # print(new_obj_xyzs.shape)
            #     # visualize_batch_pcs(new_obj_xyzs, B, num_target_objs, P, verbose=False, limit_B=num_samples,)
            #     #                     # save_dir=os.path.join("/home/weiyu/Research/intern/StructDiffuser/imgs/shapes", "d{}/t{}".format(batch_idx, vis_t)))
            #
            #     new_obj_xyzs = new_obj_xyzs.cpu().numpy()
            #     new_obj_xyzs = new_obj_xyzs[0]  # num_target_objs, P, 3
            #     obj_rgbs = batch["rgbs"].numpy()[0, :num_target_objs]
            #     # print(obj_rgbs.shape)
            #     vis_pcs = [trimesh.PointCloud(obj_xyz, colors=np.concatenate([obj_rgb * 255, np.ones([P, 1])*255], axis=-1)) for obj_xyz, obj_rgb in zip(new_obj_xyzs, obj_rgbs)]
            #
            #     scene = trimesh.Scene()
            #     # add the coordinate frame first
            #     geom = trimesh.creation.axis(0.01)
            #     scene.add_geometry(geom)
            #     table = trimesh.creation.box(extents=[1.0, 1.0, 0.02])
            #     table.apply_translation([0.5, 0, -0.01])
            #     table.visual.vertex_colors = [150, 111, 87, 125]
            #     scene.add_geometry(table)
            #     # bounds = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
            #     bounds = trimesh.creation.icosphere(subdivisions=3, radius=3.1)
            #     bounds.apply_translation([0, 0, 0])
            #     bounds.visual.vertex_colors = [30, 30, 30, 30]
            #     scene.add_geometry(bounds)
            #     scene.add_geometry(vis_pcs)
            #
            #     RT_4x4 = np.array([[-0.7147778097036409, -0.6987369263935487, 0.02931536200292423, 0.3434544782290732],
            #                        [-0.47073865286968597, 0.4496990781074231, -0.7590624874434035, 0.10599949513304896],
            #                        [0.5172018981497449, -0.5563608962206371, -0.6503574015161744, 5.32058832987803],
            #                        [0.0, 0.0, 0.0, 1.0]])
            #     RT_4x4 = np.linalg.inv(RT_4x4)
            #     RT_4x4 = RT_4x4 @ np.diag([1, -1, -1, 1])
            #     scene.camera_transform = RT_4x4
            #
            #     scene.show()



            struct_pose, pc_poses_in_struct = get_struct_objs_poses(xs[0])
            new_obj_xyzs = move_pc_and_create_scene(xyzs, struct_pose, pc_poses_in_struct)
            visualize_batch_pcs(new_obj_xyzs, B, N, P, verbose=False, limit_B=num_samples)


def run_compositional(model_dir, num_samples=10, invalid_noise_weight=0.2):

    # load model
    cfg, tokenizer, model, noise_schedule, optimizer, scheduler, epoch = load_model(model_dir)
    model.eval()
    device = cfg.device

    # load data
    data_cfg = cfg.dataset

    test_dataset = SemanticArrangementDataset(data_roots=["/home/weiyu/data_drive/data_test_objects/line_data/result"],
                                              index_roots=["/home/weiyu/data_drive/data_test_objects/line_data/result/index"],
                                              # data_roots=data_cfg.dirs,
                                              #index_roots=data_cfg.index_dirs,
                                               split="test",
                                               tokenizer=tokenizer,
                                              # important: set probability to sample negative data
                                               perturbation_probability=0.0,
                                               perturbation_mode=data_cfg.perturbation_mode,
                                               min_translation=data_cfg.min_translation,
                                               max_translation=data_cfg.max_translation,
                                               min_rotation=data_cfg.min_rotation,
                                               max_rotation=data_cfg.max_rotation,
                                               max_num_objects=data_cfg.max_num_objects,
                                               max_num_other_objects=data_cfg.max_num_other_objects,
                                               max_num_shape_parameters=data_cfg.max_num_shape_parameters,
                                               max_num_rearrange_features=data_cfg.max_num_rearrange_features,
                                               max_num_anchor_features=data_cfg.max_num_anchor_features,
                                               num_pts=data_cfg.num_pts,
                                               filter_num_moved_objects_range=data_cfg.filter_num_moved_objects_range,
                                               data_augmentation=False,
                                               shuffle_object_index=data_cfg.shuffle_object_index)


    data_iter = {}
    data_iter["test"] = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                   # collate_fn=SemanticArrangementDataset.collate_fn,
                                   pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)


    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter["test"]):

            # input
            xyzs = batch["xyzs"].to(device, non_blocking=True)
            B, N, P, _ = xyzs.shape
            obj_xyztheta_inputs = batch["obj_xyztheta_inputs"].to(device, non_blocking=True)
            struct_xyztheta_inputs = batch["struct_xyztheta_inputs"].to(device, non_blocking=True)
            position_index = batch["position_index"].to(device, non_blocking=True)
            struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
            start_token = torch.zeros((B, 1), dtype=torch.long).to(device, non_blocking=True)
            object_pad_mask = batch["obj_pad_mask"].to(device, non_blocking=True)
            struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)
            valid_sentence = batch["sentence"].to(device, non_blocking=True)
            token_type_index = batch["token_type_index"].to(device, non_blocking=True)
            struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
            sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)


            raw_valid_sentence = eval(batch["raw_sentence"][0])

            raw_invalid_sentence = []
            for tv, tt in raw_valid_sentence:
                if tt == "physics":
                    assert tv == "valid"
                    raw_invalid_sentence.append(("invalid", tt))
                else:
                    raw_invalid_sentence.append((tv, tt))
            print("raw valid sentence:", raw_valid_sentence)
            print("raw invalid sentence:", raw_invalid_sentence)

            invalid_sentence = torch.LongTensor([tokenizer.tokenize(*i) for i in raw_invalid_sentence]).to(device, non_blocking=True)

            # repeat num_samples times for this scene
            assert B == 1
            xyzs = xyzs.repeat(num_samples, 1, 1, 1)
            obj_xyztheta_inputs = obj_xyztheta_inputs.repeat(num_samples, 1, 1)
            struct_xyztheta_inputs = struct_xyztheta_inputs.repeat(num_samples, 1, 1)
            position_index = position_index.repeat(num_samples, 1)
            struct_position_index = struct_position_index.repeat(num_samples, 1)
            start_token = start_token.repeat(num_samples, 1)
            object_pad_mask = object_pad_mask.repeat(num_samples, 1)
            struct_pad_mask = struct_pad_mask.repeat(num_samples, 1)

            valid_sentence = valid_sentence.repeat(num_samples, 1)
            invalid_sentence = invalid_sentence.repeat(num_samples, 1)

            token_type_index = token_type_index.repeat(num_samples, 1)
            struct_token_type_index = struct_token_type_index.repeat(num_samples, 1)
            sentence_pad_mask = sentence_pad_mask.repeat(num_samples, 1)
            B = num_samples

            # start diffusion
            x_gt = get_diffusion_variables(struct_xyztheta_inputs, obj_xyztheta_inputs)
            x = torch.randn_like(x_gt, device=device)
            xs = []
            for t_index in tqdm.tqdm(reversed(range(0, noise_schedule.timesteps)), desc='sampling loop time step',
                                     total=noise_schedule.timesteps):
            # for t_index in tqdm.tqdm(reversed(range(0, 1)), desc='sampling loop time step',total=1):

                # get noise params
                t = torch.full((B,), t_index, device=device, dtype=torch.long)
                betas_t = extract(noise_schedule.betas, t, x.shape)
                sqrt_one_minus_alphas_cumprod_t = extract(noise_schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
                sqrt_recip_alphas_t = extract(noise_schedule.sqrt_recip_alphas, t, x.shape)

                # predict noise
                struct_xyztheta_inputs = x[:, 0, :].unsqueeze(1)  # B, 1, 3 + 6
                obj_xyztheta_inputs = x[:, 1:, :]  # B, N, 3 + 6

                # run valid
                struct_xyztheta_outputs, obj_xyztheta_outputs = model.forward(t, xyzs, obj_xyztheta_inputs, struct_xyztheta_inputs, valid_sentence,
                                                                              position_index, struct_position_index,
                                                                              token_type_index, struct_token_type_index,
                                                                              start_token,
                                                                              object_pad_mask, struct_pad_mask, sentence_pad_mask)
                predicted_noise_valid = torch.cat([struct_xyztheta_outputs, obj_xyztheta_outputs], dim=1)

                if invalid_noise_weight > 0.0:
                    # run invalid
                    struct_xyztheta_outputs, obj_xyztheta_outputs = model.forward(t, xyzs, obj_xyztheta_inputs,
                                                                                  struct_xyztheta_inputs, invalid_sentence,
                                                                                  position_index, struct_position_index,
                                                                                  token_type_index, struct_token_type_index,
                                                                                  start_token,
                                                                                  object_pad_mask, struct_pad_mask,
                                                                                  sentence_pad_mask)
                    predicted_noise_invalid = torch.cat([struct_xyztheta_outputs, obj_xyztheta_outputs], dim=1)

                    # combine valid and invalid
                    predicted_noise = (1 + invalid_noise_weight) * predicted_noise_valid - invalid_noise_weight * predicted_noise_invalid
                else:
                    predicted_noise = predicted_noise_valid

                # compute noisy x at t
                model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
                if t_index == 0:
                    x = model_mean
                else:
                    posterior_variance_t = extract(noise_schedule.posterior_variance, t, x.shape)
                    noise = torch.randn_like(x)
                    # Algorithm 2 line 4:
                    x = model_mean + torch.sqrt(posterior_variance_t) * noise

                xs.append(x)

            xs = list(reversed(xs))

            struct_pose, pc_poses_in_struct = get_struct_objs_poses(xs[0])
            new_obj_xyzs = move_pc_and_create_scene(xyzs, struct_pose, pc_poses_in_struct)
            visualize_batch_pcs(new_obj_xyzs, B, N, P, verbose=False, limit_B=num_samples)



if __name__ == "__main__":
    model_dir = "/home/weiyu/Research/intern/StructDiffuser/experiments/20220909-213428/model"
    run_compositional(model_dir, num_samples=10, invalid_noise_weight=2.0)