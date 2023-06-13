import os
import tqdm
import torch
from torch.utils.data import DataLoader
import pytorch3d.transforms as tra3d

from StructDiffusion.utils.torch_data import default_collate
from StructDiffusion.training.train_diffuser_v3_lang import load_model, get_diffusion_variables, extract, get_struct_objs_poses, move_pc_and_create_scene, visualize_batch_pcs
from StructDiffusion.data.dataset_v1_diffuser import SemanticArrangementDataset

from StructDiffusion.evaluation.infer_diffuser_v3_lang import DiffuserInference
from StructDiffusion.evaluation.infer_collision import CollisionInference
from StructDiffusion.evaluation.infer_discriminator import DiscriminatorInference
from StructDiffusion.utils.batch_inference import move_pc, move_pc_and_create_scene_new





def run(model_dir, num_samples=10):

    # load model
    # cfg, tokenizer, model, noise_schedule, optimizer, scheduler, epoch = load_model(model_dir)
    # model.eval()

    # TODO: make them params
    discriminator_score_weight = 1.0
    discriminator_model_dir = "/home/weiyu/data_drive/models_0914/discriminator_lan_local_shape_param/best_model"
    collision_score_weight = 0.0
    collision_model_dir = "/home/weiyu/data_drive/models_0914/collision/best_model"
    data_split = "test"
    override_data_dirs = ["/home/weiyu/data_drive/data_test_objects/line_data/result"]
    override_index_dirs = ["/home/weiyu/data_drive/data_test_objects/line_data/result/index"]
    visualize = True

    discriminator_inference_batch_size = 10
    num_elite = 5



    prior_inference = DiffuserInference(model_dir, data_split, override_data_dirs, override_index_dirs)
    prior_dataset = prior_inference.dataset

    device = prior_inference.cfg.device
    print("device", device)

    if discriminator_score_weight > 0:
        discriminator_inference = DiscriminatorInference(discriminator_model_dir)
        discriminator_model = discriminator_inference.model
        discriminator_cfg = discriminator_inference.cfg
        discriminator_tokenizer = discriminator_inference.dataset.tokenizer

        discriminator_model.eval()
        discriminator_num_scene_pts = discriminator_cfg.dataset.num_scene_pts
        discriminator_normalize_pc = discriminator_cfg.dataset.normalize_pc
    else:
        discriminator_num_scene_pts = None
        discriminator_normalize_pc = False

    if collision_score_weight > 0:
        collision_inference = CollisionInference(collision_model_dir, empty_dataset=True)
        collision_model = collision_inference.model
        collision_cfg = collision_inference.cfg
        collision_model.eval()
        collision_num_pair_pc_pts = collision_cfg.dataset.num_scene_pts
        collision_normalize_pc = collision_cfg.dataset.normalize_pc
    else:
        collision_num_pair_pc_pts = None
        collision_normalize_pc = False

    S = num_samples
    B = discriminator_inference_batch_size



    for idx, data_idx in enumerate(range(len(prior_dataset))):

        print("\n" + "*" * 50)

        ####################################################
        # sample S predictions

        sample_raw_data = prior_dataset.get_raw_data(data_idx, inference_mode=True)
        sample_file_id = os.path.splitext(os.path.basename(sample_raw_data["filename"]))[0]
        print(sample_file_id)
        sample_tensor_data = prior_dataset.convert_to_tensors(sample_raw_data, prior_dataset.tokenizer)

        struct_pose, pc_poses_in_struct = prior_inference.limited_batch_inference(sample_tensor_data, num_samples=S,
                                                                                  convert_to_tensors=False,
                                                                                  return_numpy=False)
        # struct_pose: S, 1, 4, 4
        # pc_poses_in_struct: S, N, 4, 4

        ####################################################
        # only keep one copy
        print("sentence", sample_raw_data["sentence"])

        # prepare for discriminator
        if discriminator_score_weight > 0:
            raw_sentence_discriminator = [sample_raw_data["sentence"][si] for si in [0, 4]]
            raw_sentence_pad_mask_discriminator = [sample_raw_data["sentence_pad_mask"][si] for si in [0, 4]]
            raw_position_index_discriminator = list(range(discriminator_cfg.dataset.max_num_shape_parameters))
            print("raw_sentence_discriminator", raw_sentence_discriminator)
            print("raw_sentence_pad_mask_discriminator", raw_sentence_pad_mask_discriminator)
            print("raw_position_index_discriminator", raw_position_index_discriminator)

        ####################################################
        # only keep one copy

        # N, P, 3
        obj_xyzs = sample_tensor_data["xyzs"].to(device)
        print("obj_xyzs shape", obj_xyzs.shape)

        # 1, N
        # object_pad_mask: padding location has 1
        object_pad_mask = sample_tensor_data["obj_pad_mask"].to(device).unsqueeze(0)
        target_object_inds = 1 - object_pad_mask
        print("target_object_inds shape", target_object_inds.shape)
        print("target_object_inds", target_object_inds)

        N, P, _ = obj_xyzs.shape
        print("S, N, P: {}, {}, {}".format(S, N, P))

        if visualize:
            print("visualizing initial scene")
            visualize_batch_pcs(obj_xyzs, 1, N, P)
        # visualize_batch_pcs(obj_xyzs, 1, N, P)

        ####################################################
        # S, N, ...

        struct_pose = struct_pose.repeat(1, N, 1, 1)  # S, N, 4, 4
        struct_pose = struct_pose.reshape(S * N, 4, 4)  # S x N, 4, 4

        new_obj_xyzs = obj_xyzs.repeat(S, 1, 1, 1)  # S, N, P, 3
        current_pc_pose = torch.eye(4).repeat(S, N, 1, 1).to(device)  # S, N, 4, 4
        # print(torch.mean(obj_xyzs, dim=2).shape)
        current_pc_pose[:, :, :3, 3] = torch.mean(new_obj_xyzs, dim=2)  # S, N, 4, 4
        current_pc_pose = current_pc_pose.reshape(S * N, 4, 4)  # S x N, 4, 4

        # optimize xyzrpy
        obj_params = torch.zeros((S, N, 6)).to(device)
        obj_params[:, :, :3] = pc_poses_in_struct[:, :, :3, 3]
        obj_params[:, :, 3:] = tra3d.matrix_to_euler_angles(pc_poses_in_struct[:, :, :3, :3], "XYZ")  # S, N, 6

        new_obj_xyzs_before_cem, goal_pc_pose_before_cem = move_pc(obj_xyzs, obj_params, struct_pose, current_pc_pose,
                                                                   device)

        if visualize:
            print("visualizing rearrangements predicted by the generator")
            visualize_batch_pcs(new_obj_xyzs_before_cem, S, N, P, limit_B=5)

        ####################################################
        # rank

        # evaluate in batches
        scores = torch.zeros(S).to(device)
        no_intersection_scores = torch.zeros(S).to(device)  # the higher the better
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
                                             target_object_inds, device,
                                             return_scene_pts=discriminator_score_weight > 0,
                                             return_scene_pts_and_pc_idxs=False,
                                             num_scene_pts=discriminator_num_scene_pts,
                                             normalize_pc=discriminator_normalize_pc,
                                             return_pair_pc=collision_score_weight > 0,
                                             num_pair_pc_pts=collision_num_pair_pc_pts,
                                             normalize_pair_pc=collision_normalize_pc)

            #######################################
            # predict whether there are pairwise collisions
            if collision_score_weight > 0:
                with torch.no_grad():
                    _, num_comb, num_pair_pc_pts, _ = obj_pair_xyzs.shape
                    # obj_pair_xyzs = obj_pair_xyzs.reshape(cur_batch_size * num_comb, num_pair_pc_pts, -1)
                    collision_logits = collision_model.forward(
                        obj_pair_xyzs.reshape(cur_batch_size * num_comb, num_pair_pc_pts, -1))
                    collision_scores = collision_model.convert_logits(collision_logits)["is_circle"].reshape(
                        cur_batch_size, num_comb)  # cur_batch_size, num_comb

                    # debug
                    # for bi, this_obj_pair_xyzs in enumerate(obj_pair_xyzs):
                    #     print("batch id", bi)
                    #     for pi, obj_pair_xyz in enumerate(this_obj_pair_xyzs):
                    #         print("pair", pi)
                    #         # obj_pair_xyzs: 2 * P, 5
                    #         print("collision score", collision_scores[bi, pi])
                    #         trimesh.PointCloud(obj_pair_xyz[:, :3].cpu()).show()

                    # 1 - mean() since the collision model predicts 1 if there is a collision
                    no_intersection_scores[cur_batch_idxs_start:cur_batch_idxs_end] = 1 - torch.mean(collision_scores,
                                                                                                     dim=1)
                if visualize:
                    print("no intersection scores", no_intersection_scores)
            #######################################
            if discriminator_score_weight > 0:
                # # debug:
                # print(subsampled_scene_xyz.shape)
                # print(subsampled_scene_xyz[0])
                # trimesh.PointCloud(subsampled_scene_xyz[0, :, :3].cpu().numpy()).show()
                #
                with torch.no_grad():

                    # Important: since this discriminator only uses local structure param, takes sentence from the first and last position
                    # local_sentence = sentence[:, [0, 4]]
                    # local_sentence_pad_mask = sentence_pad_mask[:, [0, 4]]
                    # sentence_disc, sentence_pad_mask_disc, position_index_dic = discriminator_inference.dataset.tensorfy_sentence(raw_sentence_discriminator, raw_sentence_pad_mask_discriminator, raw_position_index_discriminator)

                    sentence_disc = torch.LongTensor(
                        [discriminator_tokenizer.tokenize(*i) for i in raw_sentence_discriminator])
                    sentence_pad_mask_disc = torch.LongTensor(raw_sentence_pad_mask_discriminator)
                    position_index_dic = torch.LongTensor(raw_position_index_discriminator)

                    preds = discriminator_model.forward(subsampled_scene_xyz,
                                                        sentence_disc.unsqueeze(0).repeat(cur_batch_size, 1).to(device),
                                                        sentence_pad_mask_disc.unsqueeze(0).repeat(cur_batch_size,
                                                                                                   1).to(device),
                                                        position_index_dic.unsqueeze(0).repeat(cur_batch_size, 1).to(
                                                            device))
                    # preds = discriminator_model.forward(subsampled_scene_xyz)
                    preds = discriminator_model.convert_logits(preds)
                    preds = preds["is_circle"]  # cur_batch_size,
                    scores[cur_batch_idxs_start:cur_batch_idxs_end] = preds
                if visualize:
                    print("discriminator scores", scores)

        scores = scores * discriminator_score_weight + no_intersection_scores * collision_score_weight
        sort_idx = torch.argsort(scores).flip(dims=[0])[:num_elite]
        elite_obj_params = obj_params[sort_idx]  # num_elite, N, 6
        elite_struct_poses = struct_pose.reshape(S, N, 4, 4)[sort_idx]  # num_elite, N, 4, 4
        elite_struct_poses = elite_struct_poses.reshape(num_elite * N, 4, 4)  # num_elite x N, 4, 4
        elite_scores = scores[sort_idx]
        print("elite scores:", elite_scores)

        ####################################################
        # visualize best samples
        num_scene_pts = 4096 if discriminator_num_scene_pts is None else discriminator_num_scene_pts
        batch_current_pc_pose = current_pc_pose[0: num_elite * N]
        best_new_obj_xyzs, best_goal_pc_pose, best_subsampled_scene_xyz, _, _ = \
            move_pc_and_create_scene_new(obj_xyzs, elite_obj_params, elite_struct_poses, batch_current_pc_pose,
                                         target_object_inds, device,
                                         return_scene_pts=True, num_scene_pts=num_scene_pts, normalize_pc=True)
        if visualize:
            print("visualizing elite rearrangements ranked by collision model/discriminator")
            visualize_batch_pcs(best_new_obj_xyzs, num_elite, N, P, limit_B=num_elite)

        best_goal_pc_pose = best_goal_pc_pose.cpu().numpy()
        best_subsampled_scene_xyz = best_subsampled_scene_xyz.cpu().numpy()
        best_new_obj_xyzs = best_new_obj_xyzs.cpu().numpy()
        best_score_so_far = elite_scores.cpu().numpy()
        ####################################################




if __name__ == "__main__":
    # model_dir = "/home/weiyu/Research/intern/StructDiffuser/experiments/20220903-222727/model"
    model_dir = "/home/weiyu/data_drive/models_0914/diffuser/model"
    run(model_dir, num_samples=50)