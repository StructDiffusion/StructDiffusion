import tqdm
import torch
from torch.utils.data import DataLoader

from StructDiffusion.training.train_diffuser_v2_variable_num_objs import load_model, get_diffusion_variables, extract, get_struct_objs_poses, move_pc_and_create_scene, visualize_batch_pcs
from StructDiffusion.data.dataset_v1_diffuser import SemanticArrangementDataset


def run(model_dir, num_samples=10):

    # load model
    cfg, tokenizer, model, noise_schedule, optimizer, scheduler, epoch = load_model(model_dir)
    model.eval()
    device = cfg.device

    # load data
    data_cfg = cfg.dataset
    test_dataset = SemanticArrangementDataset(data_roots=data_cfg.dirs,
                                              index_roots=data_cfg.index_dirs,
                                              split="test",
                                              tokenizer=tokenizer,
                                              max_num_objects=data_cfg.max_num_objects,
                                              max_num_other_objects=data_cfg.max_num_other_objects,
                                              max_num_shape_parameters=data_cfg.max_num_shape_parameters,
                                              max_num_rearrange_features=data_cfg.max_num_rearrange_features,
                                              max_num_anchor_features=data_cfg.max_num_anchor_features,
                                              num_pts=data_cfg.num_pts,
                                              filter_num_moved_objects_range=data_cfg.filter_num_moved_objects_range,
                                              data_augmentation=False,
                                              shuffle_object_index=False)

    data_iter = {}
    data_iter["test"] = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                   # collate_fn=SemanticArrangementDataset.collate_fn,
                                   pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)


    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter["test"]):

            # input
            xyzs = batch["xyzs"].to(device, non_blocking=True)
            B, N, P, _ = xyzs.shape
            # obj_pad_mask: we don't need it now since we are testing
            obj_xyztheta_inputs = batch["obj_xyztheta_inputs"].to(device, non_blocking=True)
            struct_xyztheta_inputs = batch["struct_xyztheta_inputs"].to(device, non_blocking=True)
            position_index = batch["position_index"].to(device, non_blocking=True)
            struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
            start_token = torch.zeros((B, 1), dtype=torch.long).to(device, non_blocking=True)
            object_pad_mask = batch["obj_pad_mask"].to(device, non_blocking=True)
            struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)

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
            B = num_samples

            # start diffusion
            x_gt = get_diffusion_variables(struct_xyztheta_inputs, obj_xyztheta_inputs)
            x = torch.randn_like(x_gt, device=device)
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
                struct_xyztheta_outputs, obj_xyztheta_outputs = model.forward(t, xyzs, obj_xyztheta_inputs,
                                                                              struct_xyztheta_inputs,
                                                                              position_index, struct_position_index,
                                                                              start_token, object_pad_mask, struct_pad_mask)
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
            #     struct_pose, pc_poses_in_struct = get_struct_objs_poses(xs[vis_t])
            #     new_obj_xyzs = move_pc_and_create_scene(xyzs, struct_pose, pc_poses_in_struct)
            #     visualize_batch_pcs(new_obj_xyzs, B, N, P, verbose=False, limit_B=num_samples,
            #                         save_dir=os.path.join("/home/weiyu/Research/intern/StructDiffuser/imgs/shapes", "d{}/t{}".format(batch_idx, vis_t)))

            struct_pose, pc_poses_in_struct = get_struct_objs_poses(xs[0])
            new_obj_xyzs = move_pc_and_create_scene(xyzs, struct_pose, pc_poses_in_struct)
            visualize_batch_pcs(new_obj_xyzs, B, N, P, verbose=False, limit_B=num_samples)




if __name__ == "__main__":
    # model_dir = "/home/weiyu/Research/intern/StructDiffuser/experiments/20220711-074345/model"
    model_dir = "/home/weiyu/Research/intern/StructDiffuser/experiments/20220711-225253/model"
    run(model_dir, num_samples=5)