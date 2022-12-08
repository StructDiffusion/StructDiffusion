import os
import torch
import numpy as np
import pytorch3d.transforms as tra3d
import json
import argparse
from omegaconf import OmegaConf

# diffusion model
from StructDiffusion.evaluation.infer_diffuser_v3_lang import DiffuserInference

# physics eval
from StructDiffusion.utils.physics_eval import switch_stdout, visualize_batch_pcs, convert_bool, save_dict_to_h5, move_pc_and_create_scene_new, move_pc
from rearrangement_gym.semantic_rearrangement.physics_verification_dinner import verify_datum_in_simulation

# discriminators
from StructDiffusion.evaluation.infer_collision import CollisionInference
from StructDiffusion.evaluation.infer_discriminator import DiscriminatorInference


def evaluate(random_seed, structure_type, generator_model_dir, data_split, data_root,
             discriminator_model_dir=None, discriminator_model=None, discriminator_cfg=None, discriminator_tokenizer=None,
             collision_model_dir=None,
             collision_score_weight=0.5, discriminator_score_weight=0.5,
             num_samples=50, num_elite=10,
             discriminator_inference_batch_size=64,
             assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
             object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
             redirect_stdout=False, shuffle=False, summary_writer=None,
             max_num_eval=10, visualize=False,
             override_data_dirs=None, override_index_dirs=None, physics_eval_early_stop=True):
    
    assert 0 <= collision_score_weight <= 1
    assert 0 <= discriminator_score_weight <= 1

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

    save_dir = os.path.join(data_root, data_split)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    OmegaConf.save(cfg, os.path.join(save_dir, "experiment_config.yaml"))

    if redirect_stdout:
        stdout_filename = os.path.join(data_root, "{}_log.txt".format(data_split))
    else:
        stdout_filename = None
    switch_stdout(stdout_filename)

    prior_inference = DiffuserInference(generator_model_dir, data_split, override_data_dirs, override_index_dirs)
    prior_dataset = prior_inference.dataset

    if discriminator_score_weight > 0:
        if discriminator_model_dir is not None:
            discriminator_inference = DiscriminatorInference(discriminator_model_dir)
            discriminator_model = discriminator_inference.model
            discriminator_cfg = discriminator_inference.cfg
            discriminator_tokenizer = discriminator_inference.dataset.tokenizer
        else:
            assert discriminator_model is not None
            assert discriminator_cfg is not None
            assert discriminator_tokenizer is not None
        discriminator_model.eval()
        discriminator_num_scene_pts = discriminator_cfg.dataset.num_scene_pts
        discriminator_normalize_pc = discriminator_cfg.dataset.normalize_pc
    else:
        discriminator_num_scene_pts = None
        discriminator_normalize_pc = False

    if collision_score_weight > 0:
        collision_inference = CollisionInference(collision_model_dir)
        collision_model = collision_inference.model
        collision_cfg = collision_inference.cfg
        collision_model.eval()
        collision_num_pair_pc_pts = collision_cfg.dataset.num_scene_pts
        collision_normalize_pc = collision_cfg.dataset.normalize_pc
    else:
        collision_num_pair_pc_pts = None
        collision_normalize_pc = False

    device = prior_inference.cfg.device
    print("device", device)

    S = num_samples
    B = discriminator_inference_batch_size

    if shuffle:
        prior_dataset_idxs = np.random.permutation(len(prior_dataset))
    else:
        prior_dataset_idxs = list(range(len(prior_dataset)))

    all_eval_idxs = []
    success_eval_idxs = []
    all_scores_average = []
    for idx, data_idx in enumerate(prior_dataset_idxs):

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

        new_obj_xyzs_before_cem, goal_pc_pose_before_cem = move_pc(obj_xyzs, obj_params, struct_pose, current_pc_pose, device)

        if visualize:
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
                                             return_scene_pts=discriminator_score_weight>0,
                                             return_scene_pts_and_pc_idxs=False,
                                             num_scene_pts=discriminator_num_scene_pts, normalize_pc=discriminator_normalize_pc,
                                             return_pair_pc=collision_score_weight>0, num_pair_pc_pts=collision_num_pair_pc_pts, normalize_pair_pc=collision_normalize_pc)

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
                    no_intersection_scores[cur_batch_idxs_start:cur_batch_idxs_end] = 1 - torch.mean(collision_scores, dim=1)
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

                    sentence_disc = torch.LongTensor([discriminator_tokenizer.tokenize(*i) for i in raw_sentence_discriminator])
                    sentence_pad_mask_disc = torch.LongTensor(raw_sentence_pad_mask_discriminator)
                    position_index_dic = torch.LongTensor(raw_position_index_discriminator)

                    preds = discriminator_model.forward(subsampled_scene_xyz,
                                                        sentence_disc.unsqueeze(0).repeat(cur_batch_size, 1).to(device),
                                                        sentence_pad_mask_disc.unsqueeze(0).repeat(cur_batch_size, 1).to(device),
                                                        position_index_dic.unsqueeze(0).repeat(cur_batch_size, 1).to(device))
                    # preds = discriminator_model.forward(subsampled_scene_xyz)
                    preds = discriminator_model.convert_logits(preds)
                    preds = preds["is_circle"]  # cur_batch_size,
                    scores[cur_batch_idxs_start:cur_batch_idxs_end] = preds
                if visualize:
                    print("scores", scores)

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
            move_pc_and_create_scene_new(obj_xyzs, elite_obj_params, elite_struct_poses, batch_current_pc_pose, target_object_inds, device,
                                     return_scene_pts=True, num_scene_pts=num_scene_pts, normalize_pc=True)
        if visualize:
            visualize_batch_pcs(best_new_obj_xyzs, num_elite, N, P, limit_B=num_elite)

        best_goal_pc_pose = best_goal_pc_pose.cpu().numpy()
        best_subsampled_scene_xyz = best_subsampled_scene_xyz.cpu().numpy()
        best_new_obj_xyzs = best_new_obj_xyzs.cpu().numpy()
        best_score_so_far = elite_scores.cpu().numpy()
        ####################################################
        # verify in physics simulation
        d = {}
        d["is_circle"] = 0
        d["goal_specification"] = sample_raw_data["goal_specification"]
        d["target_objs"] = sample_raw_data["target_objs"]
        d["current_obj_poses"] = sample_raw_data["current_obj_poses"]
        d["current_pc_poses"] = sample_raw_data["current_pc_poses"]
        d["obj_perturbation_matrices"] = None

        physics_results = []
        for goal_pc_pose in best_goal_pc_pose:
            d["goal_pc_poses"] = goal_pc_pose
            check, check_dict = verify_datum_in_simulation(d, assets_path, object_model_dir,
                                                           structure_type=structure_type,
                                                           early_stop=physics_eval_early_stop, visualize=False)
            print("data point {}, check {}".format(data_idx, bool(check)))
            physics_results.append((check, check_dict))

        for check, check_dict in physics_results:
            print(check, check_dict)

        ####################################################
        # save data

        sd = {}
        sd["json_goal_specification"] = json.dumps(sample_raw_data["goal_specification"])
        sd["json_target_objs"] = json.dumps(sample_raw_data["target_objs"])
        # a list of numpy arrays to automatically be concatenated when storing in h5
        sd["current_obj_poses"] = sample_raw_data["current_obj_poses"][:len(sample_raw_data["target_objs"])]
        # numpy
        sd["current_pc_poses"] = sample_raw_data["current_pc_poses"]
        # sd["obj_perturbation_matrices"] = None
        sd["json_sentence"] = json.dumps(sample_raw_data["sentence"])
        sd["sentence_pad_mask"] = sample_raw_data["sentence_pad_mask"]
        # sd["position_index"] = list(range(discriminator_cfg.dataset.max_num_shape_parameters))
        sd["object_pad_mask"] = sample_raw_data["obj_pad_mask"]
        # for discriminator
        if discriminator_score_weight > 0:
            sd["json_raw_sentence_discriminator"] = json.dumps(raw_sentence_discriminator)
            sd["raw_sentence_pad_mask_discriminator"] = raw_sentence_pad_mask_discriminator
            sd["raw_position_index_discriminator"] = raw_position_index_discriminator

        # only save the best one
        for bsi in range(len(best_score_so_far)):

            sd["check"] = physics_results[bsi][0]
            sd["json_check_dict"] = json.dumps(convert_bool(physics_results[bsi][1]))
            sd["goal_pc_poses"] = best_goal_pc_pose[bsi]
            sd["new_obj_xyzs"] = best_new_obj_xyzs[bsi]
            sd["subsampled_scene_xyz"] = best_subsampled_scene_xyz[bsi]
            sd["discriminator_score"] = best_score_so_far[bsi]

            # for k in sd:
            #     if type(sd[k]).__module__ == np.__name__:
            #         print(k, sd[k].shape)
            #     else:
            #         print(k, sd[k])
            save_dict_to_h5(sd, filename=os.path.join(save_dir, "{}_cem{}.h5".format(sample_file_id, bsi)))

            all_eval_idxs.append(data_idx)
            if physics_results[bsi][0]:
                success_eval_idxs.append(data_idx)

        print("All:", all_eval_idxs)
        print("Success:", success_eval_idxs)

        if len(all_eval_idxs) > max_num_eval:
            break

    switch_stdout()

    return success_eval_idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval")
    parser.add_argument("--config_file", help='config yaml file',
                        default='../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_discriminator/dinner.yaml',
                        type=str)
    args = parser.parse_args()
    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)
    cfg = OmegaConf.load(args.config_file)

    cfg.physics_eval_early_stop = False

    evaluate(**cfg)

    # ####################################################################################################################
    # # batch testing for testing objects
    # ####################################################################################################################
    # model_dir = "/home/weiyu/Research/intern/StructDiffuser/experiments/20220903-222727/model"
    # discriminator_model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220809/20220813-220054_epoch46/best_model"
    # collision_model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220809/20220809-002824/best_model"
    #
    # evaluate(structure_type="dinner",
    #          generator_model_dir=model_dir, data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_diffuser_dinner_10k_test_objects_discriminator_local_shape_param_collision_100",
    #          discriminator_model_dir=discriminator_model_dir,
    #          collision_model_dir=collision_model_dir,
    #          collision_score_weight=0.5, discriminator_score_weight=0.5,
    #          num_samples=100, num_elite=5,
    #          discriminator_inference_batch_size=24,  # 64 runs out of memoery
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=False, shuffle=False, summary_writer=None,
    #          max_num_eval=100, visualize=True,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/dinner_data/result"],
    #          override_index_dirs=["index"])
    #
    # evaluate(structure_type="circle",
    #          generator_model_dir=model_dir, data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_diffuser_circle_10k_test_objects_discriminator_local_shape_param_collision_100",
    #          discriminator_model_dir=discriminator_model_dir,
    #          collision_model_dir=collision_model_dir,
    #          collision_score_weight=0.5, discriminator_score_weight=0.5,
    #          num_samples=100, num_elite=5,
    #          discriminator_inference_batch_size=24,  # 64 runs out of memoery
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=True, shuffle=False, summary_writer=None,
    #          max_num_eval=100, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/circle_data/result"],
    #          override_index_dirs=["index"])
    #
    # evaluate(structure_type="line",
    #          generator_model_dir=model_dir, data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_diffuser_line_10k_test_objects_discriminator_local_shape_param_collision_100",
    #          discriminator_model_dir=discriminator_model_dir,
    #          collision_model_dir=collision_model_dir,
    #          collision_score_weight=0.5, discriminator_score_weight=0.5,
    #          num_samples=100, num_elite=5,
    #          discriminator_inference_batch_size=24,  # 64 runs out of memoery
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=True, shuffle=False, summary_writer=None,
    #          max_num_eval=100, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/line_data/result"],
    #          override_index_dirs=["index"])
    #
    # evaluate(structure_type="tower",
    #          generator_model_dir=model_dir, data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_diffuser_stacking_10k_test_objects_discriminator_local_shape_param_collision_100",
    #          discriminator_model_dir=discriminator_model_dir,
    #          collision_model_dir=collision_model_dir,
    #          collision_score_weight=0.5, discriminator_score_weight=0.5,
    #          num_samples=100, num_elite=5,
    #          discriminator_inference_batch_size=24,  # 64 runs out of memoery
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=True, shuffle=False, summary_writer=None,
    #          max_num_eval=100, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/stacking_data/result"],
    #          override_index_dirs=["index"])







