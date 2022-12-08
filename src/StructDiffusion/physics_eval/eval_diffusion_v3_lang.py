import torch
import numpy as np
import os
import pytorch3d.transforms as tra3d
import json
import argparse
from omegaconf import OmegaConf

# diffusion model
from StructDiffusion.evaluation.infer_diffuser_v3_lang import DiffuserInference

# physics eval
from StructDiffusion.utils.physics_eval import switch_stdout, visualize_batch_pcs, convert_bool, save_dict_to_h5
from rearrangement_gym.semantic_rearrangement.physics_verification_dinner import verify_datum_in_simulation


def evaluate(random_seed, structure_type, generator_model_dir, data_split, data_root,
            assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
            object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
            redirect_stdout=False, shuffle=False, summary_writer=None, max_num_eval=10, visualize=False,
            override_data_dirs=None, override_index_dirs=None, physics_eval_early_stop=True):

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

    device = prior_inference.cfg.device
    print("device", device)

    if shuffle:
        prior_dataset_idxs = np.random.permutation(len(prior_dataset))
    else:
        prior_dataset_idxs = list(range(len(prior_dataset)))

    B = 1
    all_eval_idxs = []
    success_eval_idxs = []
    for idx, data_idx in enumerate(prior_dataset_idxs):
        print("\n" + "*" * 50)

        sample_raw_data = prior_dataset.get_raw_data(data_idx, inference_mode=True)
        sample_file_id = os.path.splitext(os.path.basename(sample_raw_data["filename"]))[0]
        print(sample_file_id)
        sample_tensor_data = prior_dataset.convert_to_tensors(sample_raw_data, prior_dataset.tokenizer)

        struct_pose, pc_poses_in_struct = prior_inference.limited_batch_inference(sample_tensor_data, num_samples=1,
                                                                          convert_to_tensors=False,
                                                                          return_numpy=False)
        # struct_pose: B, 1, 4, 4
        # pc_poses_in_struct: B, N, 4, 4



        ####################################################
        # obj_xyzs: N, P, 3

        # N, P, 3
        obj_xyzs = sample_tensor_data["xyzs"].to(device)
        print("obj_xyzs shape", obj_xyzs.shape)
        B, N, _, _ = pc_poses_in_struct.shape
        _, P, _ = obj_xyzs.shape
        print("B, N, P: {}, {}, {}".format(B, N, P))

        new_obj_xyzs = obj_xyzs.repeat(B, 1, 1, 1)  # B, N, P, 3
        if visualize:
            visualize_batch_pcs(new_obj_xyzs, 1, N, P)

        current_pc_poses = torch.eye(4).repeat(B, N, 1, 1).to(device)  # B, N, 4, 4
        # print(torch.mean(obj_xyzs, dim=2).shape)
        current_pc_poses[:, :, :3, 3] = torch.mean(new_obj_xyzs, dim=2)  # B, N, 4, 4
        current_pc_poses = current_pc_poses.reshape(B * N, 4, 4)  # B x N, 4, 4

        struct_pose = struct_pose.repeat(1, N, 1, 1)  # B, N, 4, 4
        struct_pose = struct_pose.reshape(B * N, 4, 4)  # B x 1, 4, 4
        pc_poses_in_struct = pc_poses_in_struct.reshape(B * N, 4, 4)  # B x N, 4, 4

        goal_pc_pose = struct_pose @ pc_poses_in_struct  # B x N, 4, 4
        goal_pc_transform = goal_pc_pose @ torch.inverse(current_pc_poses)  # B x N, 4, 4

        # important: pytorch3d uses row-major ordering, need to transpose each transformation matrix
        transpose = tra3d.Transform3d(matrix=goal_pc_transform.transpose(1, 2))

        new_obj_xyzs = new_obj_xyzs.reshape(B * N, P, 3)  # B x N, P, 3
        new_obj_xyzs = transpose.transform_points(new_obj_xyzs)

        # put it back to B, N, P, 3
        new_obj_xyzs = new_obj_xyzs.reshape(B, N, P, 3)
        # put it back to B, N, 4, 4
        goal_pc_pose = goal_pc_pose.reshape(B, N, 4, 4)

        if visualize:
            visualize_batch_pcs(new_obj_xyzs, B, N, P)

        best_goal_pc_pose = goal_pc_pose.cpu().numpy()
        best_new_obj_xyzs = new_obj_xyzs.cpu().numpy()

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
        sd["object_pad_mask"] = sample_raw_data["obj_pad_mask"]

        for bsi in range(B):

            sd["check"] = physics_results[bsi][0]
            sd["json_check_dict"] = json.dumps(convert_bool(physics_results[bsi][1]))
            sd["goal_pc_poses"] = best_goal_pc_pose[bsi]
            sd["new_obj_xyzs"] = best_new_obj_xyzs[bsi]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval")
    parser.add_argument("--config_file", help='config yaml file',
                        default='../configs/physics_eval/diffusion_v3_lang/dinner.yaml',
                        type=str)
    args = parser.parse_args()
    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)
    cfg = OmegaConf.load(args.config_file)

    cfg.physics_eval_early_stop = False

    evaluate(**cfg)

    ####################################################################################################################
    # batch testing for testing objects
    ####################################################################################################################
    # # dinner 10k
    # evaluate(structure_type="dinner",
    #         generator_model_dir="/home/weiyu/Research/intern/StructDiffuser/experiments/20220903-222727/model",
    #         data_split="train", data_root="/home/weiyu/data_drive/physics_eval_diffuser_dinner_10k_test_objects_100",
    #         test_specific_shape=None, assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #         object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #         redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #         override_data_dirs=["/home/weiyu/data_drive/data_test_objects/dinner_data/result"],
    #         override_index_dirs=["index"])
    #
    # evaluate(structure_type="line",
    #         generator_model_dir="/home/weiyu/Research/intern/StructDiffuser/experiments/20220903-222727/model",
    #         data_split="train", data_root="/home/weiyu/data_drive/physics_eval_diffuser_line_10k_test_objects_100",
    #         test_specific_shape=None, assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #         object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #         redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #         override_data_dirs=["/home/weiyu/data_drive/data_test_objects/line_data/result"],
    #         override_index_dirs=["index"])
    #
    # evaluate(structure_type="circle",
    #         generator_model_dir="/home/weiyu/Research/intern/StructDiffuser/experiments/20220903-222727/model",
    #         data_split="train", data_root="/home/weiyu/data_drive/physics_eval_diffuser_circle_10k_test_objects_100",
    #         test_specific_shape=None, assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #         object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #         redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #         override_data_dirs=["/home/weiyu/data_drive/data_test_objects/circle_data/result"],
    #         override_index_dirs=["index"])
    #
    # evaluate(structure_type="tower",
    #         generator_model_dir="/home/weiyu/Research/intern/StructDiffuser/experiments/20220903-222727/model",
    #         data_split="train", data_root="/home/weiyu/data_drive/physics_eval_diffuser_stacking_10k_test_objects_100",
    #         test_specific_shape=None, assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #         object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #         redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #         override_data_dirs=["/home/weiyu/data_drive/data_test_objects/stacking_data/result"],
    #         override_index_dirs=["index"])






