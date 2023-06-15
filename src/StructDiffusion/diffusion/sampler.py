import torch
from tqdm import tqdm
import pytorch3d as tra3d

from StructDiffusion.diffusion.noise_schedule import extract
from StructDiffusion.diffusion.pose_conversion import get_struct_objs_poses
from StructDiffusion.utils.batch_inference import move_pc_and_create_scene_simple, visualize_batch_pcs, move_pc_and_create_scene_new

class Sampler:

    def __init__(self, model_class, checkpoint_path, device, debug=False):

        self.debug = debug
        self.device = device

        self.model = model_class.load_from_checkpoint(checkpoint_path)
        self.backbone = self.model.model
        self.backbone.to(device)
        self.backbone.eval()

    def sample(self, batch, num_poses):

        noise_schedule = self.model.noise_schedule

        B = batch["pcs"].shape[0]

        x_noisy = torch.randn((B, num_poses, 9), device=self.device)

        xs = []
        for t_index in tqdm(reversed(range(0, noise_schedule.timesteps)),
                            desc='sampling loop time step', total=noise_schedule.timesteps):

            t = torch.full((B,), t_index, device=self.device, dtype=torch.long)

            # noise schedule
            betas_t = extract(noise_schedule.betas, t, x_noisy.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(noise_schedule.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
            sqrt_recip_alphas_t = extract(noise_schedule.sqrt_recip_alphas, t, x_noisy.shape)

            # predict noise
            pcs = batch["pcs"]
            sentence = batch["sentence"]
            type_index = batch["type_index"]
            position_index = batch["position_index"]
            pad_mask = batch["pad_mask"]
            # calling the backbone instead of the pytorch-lightning model
            with torch.no_grad():
                predicted_noise = self.backbone.forward(t, pcs, sentence, x_noisy, type_index, position_index, pad_mask)

            # compute noisy x at t
            model_mean = sqrt_recip_alphas_t * (x_noisy - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
            if t_index == 0:
                x_noisy = model_mean
            else:
                posterior_variance_t = extract(noise_schedule.posterior_variance, t, x_noisy.shape)
                noise = torch.randn_like(x_noisy)
                x_noisy = model_mean + torch.sqrt(posterior_variance_t) * noise

            xs.append(x_noisy)

        xs = list(reversed(xs))
        return xs

class SamplerV2:

    def __init__(self, diffusion_model_class, diffusion_checkpoint_path,
                 collision_model_class, collision_checkpoint_path,
                 device, debug=False):

        self.debug = debug
        self.device = device

        self.diffusion_model = diffusion_model_class.load_from_checkpoint(diffusion_checkpoint_path)
        self.diffusion_backbone = self.diffusion_model.model
        self.diffusion_backbone.to(device)
        self.diffusion_backbone.eval()

        self.collision_model = collision_model_class.load_from_checkpoint(collision_checkpoint_path)
        self.collision_backbone = self.collision_model.model
        self.collision_backbone.to(device)
        self.collision_backbone.eval()

    def sample(self, batch, num_poses):

        noise_schedule = self.diffusion_model.noise_schedule

        B = batch["pcs"].shape[0]

        x_noisy = torch.randn((B, num_poses, 9), device=self.device)

        xs = []
        for t_index in tqdm(reversed(range(0, noise_schedule.timesteps)),
                            desc='sampling loop time step', total=noise_schedule.timesteps):

            t = torch.full((B,), t_index, device=self.device, dtype=torch.long)

            # noise schedule
            betas_t = extract(noise_schedule.betas, t, x_noisy.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(noise_schedule.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
            sqrt_recip_alphas_t = extract(noise_schedule.sqrt_recip_alphas, t, x_noisy.shape)

            # predict noise
            pcs = batch["pcs"]
            sentence = batch["sentence"]
            type_index = batch["type_index"]
            position_index = batch["position_index"]
            pad_mask = batch["pad_mask"]
            # calling the backbone instead of the pytorch-lightning model
            with torch.no_grad():
                predicted_noise = self.backbone.forward(t, pcs, sentence, x_noisy, type_index, position_index, pad_mask)

            # compute noisy x at t
            model_mean = sqrt_recip_alphas_t * (x_noisy - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
            if t_index == 0:
                x_noisy = model_mean
            else:
                posterior_variance_t = extract(noise_schedule.posterior_variance, t, x_noisy.shape)
                noise = torch.randn_like(x_noisy)
                x_noisy = model_mean + torch.sqrt(posterior_variance_t) * noise

            xs.append(x_noisy)

        xs = list(reversed(xs))

        visualize = True

        struct_pose, pc_poses_in_struct = get_struct_objs_poses(xs[0])
        # struct_pose: B, 1, 4, 4
        # pc_poses_in_struct: B, N, 4, 4

        S = B
        num_elite = 10
        ####################################################
        # only keep one copy

        # N, P, 3
        obj_xyzs = batch["pcs"][0][:, :, :3]
        print("obj_xyzs shape", obj_xyzs.shape)

        # 1, N
        # object_pad_mask: padding location has 1
        num_target_objs = num_poses
        if self.diffusion_backbone.use_virtual_structure_frame:
            num_target_objs -= 1
        object_pad_mask = batch["pad_mask"][0][-num_target_objs:].unsqueeze(0)
        target_object_inds = 1 - object_pad_mask
        print("target_object_inds shape", target_object_inds.shape)
        print("target_object_inds", target_object_inds)

        N, P, _ = obj_xyzs.shape
        print("S, N, P: {}, {}, {}".format(S, N, P))

        if visualize:
            print("visualizing initial scene")
            visualize_batch_pcs(obj_xyzs, 1, N, P)

        ####################################################
        # S, N, ...

        struct_pose = struct_pose.repeat(1, N, 1, 1)  # S, N, 4, 4
        struct_pose = struct_pose.reshape(S * N, 4, 4)  # S x N, 4, 4

        new_obj_xyzs = obj_xyzs.repeat(S, 1, 1, 1)  # S, N, P, 3
        current_pc_pose = torch.eye(4).repeat(S, N, 1, 1).to(self.device)  # S, N, 4, 4
        current_pc_pose[:, :, :3, 3] = torch.mean(new_obj_xyzs, dim=2)  # S, N, 4, 4
        current_pc_pose = current_pc_pose.reshape(S * N, 4, 4)  # S x N, 4, 4

        # optimize xyzrpy
        obj_params = torch.zeros((S, N, 6)).to(self.device)
        obj_params[:, :, :3] = pc_poses_in_struct[:, :, :3, 3]
        obj_params[:, :, 3:] = tra3d.matrix_to_euler_angles(pc_poses_in_struct[:, :, :3, :3], "XYZ")  # S, N, 6
        #
        # new_obj_xyzs_before_cem, goal_pc_pose_before_cem = move_pc(obj_xyzs, obj_params, struct_pose, current_pc_pose, device)
        #
        # if visualize:
        #     print("visualizing rearrangements predicted by the generator")
        #     visualize_batch_pcs(new_obj_xyzs_before_cem, S, N, P, limit_B=5)

        ####################################################
        # rank

        # evaluate in batches
        scores = torch.zeros(S).to(self.device)
        no_intersection_scores = torch.zeros(S).to(self.device)  # the higher the better
        num_batches = int(S / B)
        if S % B != 0:
            num_batches += 1
        for b in range(num_batches):
            if b + 1 == num_batches:
                cur_batch_idxs_start = b * B
                cur_batch_idxs_end = S
            else:
                cur_batch_idxs_start = b * B
                cur_batch_idxs_end = (b + 1) * B
            cur_batch_size = cur_batch_idxs_end - cur_batch_idxs_start

            # print("current batch idxs start", cur_batch_idxs_start)
            # print("current batch idxs end", cur_batch_idxs_end)
            # print("size of the current batch", cur_batch_size)

            batch_obj_params = obj_params[cur_batch_idxs_start: cur_batch_idxs_end]
            batch_struct_pose = struct_pose[cur_batch_idxs_start * N: cur_batch_idxs_end * N]
            batch_current_pc_pose = current_pc_pose[cur_batch_idxs_start * N:cur_batch_idxs_end * N]

            new_obj_xyzs, _, subsampled_scene_xyz, _, obj_pair_xyzs = \
                move_pc_and_create_scene_new(obj_xyzs, batch_obj_params, batch_struct_pose, batch_current_pc_pose,
                                             target_object_inds, self.device,
                                             return_scene_pts=False,
                                             return_scene_pts_and_pc_idxs=False,
                                             num_scene_pts=False,
                                             normalize_pc=False,
                                             return_pair_pc=True,
                                             num_pair_pc_pts=self.collision_model.data_cfg.num_scene_pts,
                                             normalize_pair_pc=self.collision_model.data_cfg.normalize_pc)

            #######################################
            # predict whether there are pairwise collisions
            # if collision_score_weight > 0:
            with torch.no_grad():
                _, num_comb, num_pair_pc_pts, _ = obj_pair_xyzs.shape
                # obj_pair_xyzs = obj_pair_xyzs.reshape(cur_batch_size * num_comb, num_pair_pc_pts, -1)
                collision_logits = self.collision_backbone.forward(obj_pair_xyzs.reshape(cur_batch_size * num_comb, num_pair_pc_pts, -1))
                collision_scores = self.collision_backbone.convert_logits(collision_logits).reshape(cur_batch_size, num_comb)  # cur_batch_size, num_comb

                # debug
                # for bi, this_obj_pair_xyzs in enumerate(obj_pair_xyzs):
                #     print("batch id", bi)
                #     for pi, obj_pair_xyz in enumerate(this_obj_pair_xyzs):
                #         print("pair", pi)
                #         # obj_pair_xyzs: 2 * P, 5
                #         print("collision score", collision_scores[bi, pi])
                #         trimesh.PointCloud(obj_pair_xyz[:, :3].cpu()).show()

                # 1 - mean() since the collision model predicts 1 if there is a collision
                no_intersection_scores[cur_batch_idxs_start:cur_batch_idxs_end] = 1 - torch.mean(collision_scores, dim=1)
            if visualize:
                print("no intersection scores", no_intersection_scores)
            # #######################################
            # if discriminator_score_weight > 0:
            #     # # debug:
            #     # print(subsampled_scene_xyz.shape)
            #     # print(subsampled_scene_xyz[0])
            #     # trimesh.PointCloud(subsampled_scene_xyz[0, :, :3].cpu().numpy()).show()
            #     #
            #     with torch.no_grad():
            #
            #         # Important: since this discriminator only uses local structure param, takes sentence from the first and last position
            #         # local_sentence = sentence[:, [0, 4]]
            #         # local_sentence_pad_mask = sentence_pad_mask[:, [0, 4]]
            #         # sentence_disc, sentence_pad_mask_disc, position_index_dic = discriminator_inference.dataset.tensorfy_sentence(raw_sentence_discriminator, raw_sentence_pad_mask_discriminator, raw_position_index_discriminator)
            #
            #         sentence_disc = torch.LongTensor(
            #             [discriminator_tokenizer.tokenize(*i) for i in raw_sentence_discriminator])
            #         sentence_pad_mask_disc = torch.LongTensor(raw_sentence_pad_mask_discriminator)
            #         position_index_dic = torch.LongTensor(raw_position_index_discriminator)
            #
            #         preds = discriminator_model.forward(subsampled_scene_xyz,
            #                                             sentence_disc.unsqueeze(0).repeat(cur_batch_size, 1).to(device),
            #                                             sentence_pad_mask_disc.unsqueeze(0).repeat(cur_batch_size,
            #                                                                                        1).to(device),
            #                                             position_index_dic.unsqueeze(0).repeat(cur_batch_size, 1).to(
            #                                                 device))
            #         # preds = discriminator_model.forward(subsampled_scene_xyz)
            #         preds = discriminator_model.convert_logits(preds)
            #         preds = preds["is_circle"]  # cur_batch_size,
            #         scores[cur_batch_idxs_start:cur_batch_idxs_end] = preds
            #     if visualize:
            #         print("discriminator scores", scores)

        # scores = scores * discriminator_score_weight + no_intersection_scores * collision_score_weight
        scores = no_intersection_scores
        sort_idx = torch.argsort(scores).flip(dims=[0])[:num_elite]
        elite_obj_params = obj_params[sort_idx]  # num_elite, N, 6
        elite_struct_poses = struct_pose.reshape(S, N, 4, 4)[sort_idx]  # num_elite, N, 4, 4
        elite_struct_poses = elite_struct_poses.reshape(num_elite * N, 4, 4)  # num_elite x N, 4, 4
        elite_scores = scores[sort_idx]
        print("elite scores:", elite_scores)

        ####################################################
        # visualize best samples
        num_scene_pts = 4096 # if discriminator_num_scene_pts is None else discriminator_num_scene_pts
        batch_current_pc_pose = current_pc_pose[0: num_elite * N]
        best_new_obj_xyzs, best_goal_pc_pose, best_subsampled_scene_xyz, _, _ = \
            move_pc_and_create_scene_new(obj_xyzs, elite_obj_params, elite_struct_poses, batch_current_pc_pose,
                                         target_object_inds, self.device,
                                         return_scene_pts=True, num_scene_pts=num_scene_pts, normalize_pc=True)
        if visualize:
            print("visualizing elite rearrangements ranked by collision model/discriminator")
            visualize_batch_pcs(best_new_obj_xyzs, num_elite, N, P, limit_B=num_elite)


        return xs