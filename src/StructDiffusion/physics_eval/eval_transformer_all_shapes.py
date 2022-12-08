import torch
import numpy as np
import os
import pytorch3d.transforms as tra3d
import json
import argparse
from omegaconf import OmegaConf

from src.generative_models.try_langevin_actor_vae_3networks_language_all_shapes_discriminator_7 import switch_stdout, move_pc, visualize_batch_pcs, convert_bool, save_dict_to_h5
from src.test_prior_continuous_out_encoder_decoder_struct_pct_6d_dropout_all_objects_all_shapes import PriorInference
from brain2.semantic_rearrangement.physics_verification_dinner import verify_datum_in_simulation


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

    prior_inference = PriorInference(generator_model_dir, data_split=data_split,
                                     override_data_dirs=override_data_dirs,
                                     override_index_dirs=override_index_dirs)
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
        sample_data = prior_dataset.convert_to_tensors(sample_raw_data, prior_dataset.tokenizer)

        ####################################################
        # sample S predictions

        # only running the following line once is incorrect because transformer is autoregressive
        # struct_preds, obj_preds = prior_inference.limited_batch_inference([sample_raw_data], verbose=False, return_numpy=False)

        # important: iterative sampling
        beam_data = [sample_raw_data] * B
        # first predict structure pose
        beam_goal_struct_pose, target_object_preds = prior_inference.limited_batch_inference(beam_data)
        for b in range(B):
            datum = beam_data[b]
            datum["struct_x_inputs"] = [beam_goal_struct_pose[b][0]]
            datum["struct_y_inputs"] = [beam_goal_struct_pose[b][1]]
            datum["struct_z_inputs"] = [beam_goal_struct_pose[b][2]]
            datum["struct_theta_inputs"] = [beam_goal_struct_pose[b][3:]]

        # then iteratively predict pose of each object
        beam_goal_obj_poses = []
        for obj_idx in range(prior_dataset.max_num_objects):
            struct_preds, target_object_preds = prior_inference.limited_batch_inference(beam_data)
            beam_goal_obj_poses.append(target_object_preds[:, obj_idx])
            for b in range(B):
                datum = beam_data[b]
                datum["obj_x_inputs"][obj_idx] = target_object_preds[b][obj_idx][0]
                datum["obj_y_inputs"][obj_idx] = target_object_preds[b][obj_idx][1]
                datum["obj_z_inputs"][obj_idx] = target_object_preds[b][obj_idx][2]
                datum["obj_theta_inputs"][obj_idx] = target_object_preds[b][obj_idx][3:]
        # concat in the object dim
        beam_goal_obj_poses = np.stack(beam_goal_obj_poses, axis=0)
        # swap axis
        beam_goal_obj_poses = np.swapaxes(beam_goal_obj_poses, 1, 0)  # batch size, number of target objects, pose dim

        struct_preds = torch.FloatTensor(beam_goal_struct_pose)
        obj_preds = torch.FloatTensor(beam_goal_obj_poses)

        print(struct_preds.shape)
        print(obj_preds.shape)

        # struct_preds: B, 12
        # obj_preds: B, N, 12
        ####################################################

        # #*************************************************************
        # # test obj poses
        # #*************************************************************
        # import trimesh
        # def load_object_mesh_from_object_info(assets_dir, object_urdf):
        #     mesh_path = os.path.join(assets_dir, "visual", object_urdf[:-5] + ".obj")
        #     object_visual_mesh = trimesh.load(mesh_path)
        #     return object_visual_mesh
        #
        # goal_specification = sample_raw_data["goal_specification"]
        # obj_xyzs = sample_raw_data["xyzs"]
        # current_obj_poses = sample_raw_data["current_obj_poses"]
        # target_objs = sample_raw_data["target_objs"]
        #
        # target_obj_urdfs = [obj_spec["urdf"] for obj_spec in goal_specification["rearrange"]["objects"]]
        # structure_parameters = goal_specification["shape"]
        # if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
        #     target_obj_urdfs = target_obj_urdfs[::-1]
        # elif structure_parameters["type"] == "tower" or structure_parameters["type"] == "dinner":
        #     target_obj_urdfs = target_obj_urdfs
        # else:
        #     raise KeyError("{} structure is not recognized".format(structure_parameters["type"]))
        #
        # target_obj_vis_meshes = [load_object_mesh_from_object_info(object_model_dir, u) for u in target_obj_urdfs]
        # for i, obj_name in enumerate(target_objs):
        #     current_obj_pose = current_obj_poses[i]
        #     target_obj_vis_meshes[i].apply_transform(current_obj_pose)
        #
        # obj_pcs_vis = [trimesh.PointCloud(pc_obj[:, :3], colors=[255, 0, 0, 255]) for pc_obj in obj_xyzs]
        # scene = trimesh.Scene()
        # # add the coordinate frame first
        # geom = trimesh.creation.axis(0.01)
        # scene.add_geometry(geom)
        # table = trimesh.creation.box(extents=[1.0, 1.0, 0.02])
        # table.apply_translation([0.5, 0, -0.01])
        # table.visual.vertex_colors = [150, 111, 87, 125]
        # scene.add_geometry(table)
        #
        # for obj_vis_mesh in target_obj_vis_meshes:
        #     obj_vis_mesh.visual.vertex_colors = [50, 50, 50, 100]
        #
        # scene.add_geometry(target_obj_vis_meshes)
        # scene.add_geometry(obj_pcs_vis)
        # scene.show()
        # # *************************************************************

        ####################################################
        # obj_xyzs: N, P, 3
        # obj_params: B, N, 6
        # struct_pose: B x N, 4, 4
        # current_pc_pose: B x N, 4, 4
        # target_object_inds: 1, N

        # N, P, 3
        obj_xyzs = sample_data["xyzs"].to(device)
        print("obj_xyzs shape", obj_xyzs.shape)
        N, P, _ = obj_xyzs.shape
        print("B, N, P: {}, {}, {}".format(B, N, P))
        if visualize:
            visualize_batch_pcs(obj_xyzs, 1, N, P)

        struct_pose = torch.eye(4).repeat(B, 1, 1).to(device)  # S, 4, 4
        struct_pose[:, :3, :3] = struct_preds[:, 3:].reshape(-1, 3, 3)
        struct_pose[:, :3, 3] = struct_preds[:, :3]
        struct_pose = struct_pose.repeat_interleave(N, dim=0)  # S x N, 4, 4

        current_pc_pose = torch.eye(4).repeat(N, 1, 1).to(device)  # N, 4, 4
        current_pc_pose[:, :3, 3] = torch.mean(obj_xyzs, dim=1)  # N, 4, 4
        current_pc_pose = current_pc_pose.repeat(B, 1, 1)  # S x N, 4, 4

        obj_params = torch.zeros((B, N, 6)).to(device)
        obj_preds = obj_preds.reshape(B, N, -1)  # S, N, 12
        obj_params[:, :, :3] = obj_preds[:, :, :3]
        obj_params[:, :, 3:] = tra3d.matrix_to_euler_angles(obj_preds[:, :, 3:].reshape(B, N, 3, 3), "XYZ")

        best_new_obj_xyzs, best_goal_pc_pose = move_pc(obj_xyzs, obj_params, struct_pose, current_pc_pose, device)

        if visualize:
            visualize_batch_pcs(best_new_obj_xyzs, B, N, P)

        # visualize_batch_pcs(best_new_obj_xyzs, B, N, P)

        best_goal_pc_pose = best_goal_pc_pose.cpu().numpy()
        best_new_obj_xyzs = best_new_obj_xyzs.cpu().numpy()

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

        # preds = discriminator_model.forward(subsampled_scene_xyz,
        #                                     sentence.repeat(cur_batch_size, 1),
        #                                     sentence_pad_mask.repeat(cur_batch_size, 1),
        #                                     position_index.repeat(cur_batch_size, 1))

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
        sd["object_pad_mask"] = sample_raw_data["object_pad_mask"]

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
                        default='../configs/physics_eval/transformer/dinner.yaml',
                        type=str)
    args = parser.parse_args()
    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)
    cfg = OmegaConf.load(args.config_file)

    cfg.physics_eval_early_stop = False

    evaluate(**cfg)



    ####################################################################################################################
    # # line 5k for test discriminator
    # evaluate(generator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220717-143907/best_model",
    #          data_split="test", data_root="/home/weiyu/data_drive/physics_eval_transformer_line_test_discriminator_5k",
    #          test_specific_shape=None, assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
    #          redirect_stdout=False, shuffle=False, summary_writer=None, max_num_eval=30, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_new_objects/examples_line_new_objects_test_discriminator/result"],
    #          override_index_dirs=["index_5k"])

    # ####################################################################################################################
    # # batch testing
    # ####################################################################################################################
    # # stacking 10k
    # evaluate(structure_type="tower",
    #          generator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220705-225536/best_model",
    #          data_split="test", data_root="/home/weiyu/data_drive/physics_eval_transformer_stacking_10k",
    #          test_specific_shape=None, assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_new_objects/examples_stacking_new_objects/result"],
    #          override_index_dirs=["index_10k"])
    # ####################################################################################################################
    # # dinner 10k
    # evaluate(structure_type="dinner",
    #         generator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220805-153259/best_model",
    #         data_split="test", data_root="/home/weiyu/data_drive/physics_eval_transformer_dinner_10k",
    #         test_specific_shape=None, assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #         object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
    #         redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #         override_data_dirs=["/home/weiyu/data_drive/data_new_objects/examples_dinner_new_objects/result"],
    #         override_index_dirs=["index_10k"])
    #
    # # use physics_verification_dinner
    # # Success: [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 24, 25, 28, 29, 30]
    #
    # # ####################################################################################################################
    # # circle 10k
    # evaluate(structure_type="circle",
    #     generator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220814-223710/best_model",
    #         data_split="test", data_root="/home/weiyu/data_drive/physics_eval_transformer_circle_10k",
    #         test_specific_shape=None, assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #         object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
    #         redirect_stdout=False, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #         override_data_dirs=["/home/weiyu/data_drive/data_new_objects/examples_circle_new_objects/result"],
    #         override_index_dirs=["index_10k"])
    #
    # # circle: [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 59, 60, 61, 63, 64, 65, 68, 70, 72, 74, 75, 76, 77, 78, 80, 82, 83, 86, 87, 88, 90, 91, 93, 97, 98, 99, 100]
    #
    # # ####################################################################################################################
    # # line 10k
    # evaluate(structure_type="line",
    #          generator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220814-223820/best_model",
    #         data_split="test", data_root="/home/weiyu/data_drive/physics_eval_transformer_line_10k",
    #         test_specific_shape=None, assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #         object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
    #         redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #         override_data_dirs=["/home/weiyu/data_drive/data_new_objects/examples_line_new_objects/result"],
    #         override_index_dirs=["index_10k"])
    # ####################################################################################################################







    # ####################################################################################################################
    # # batch testing for testing objects
    # ####################################################################################################################
    # # dinner 10k
    # evaluate(structure_type="dinner",
    #         generator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220805-153259/best_model",
    #         data_split="train", data_root="/home/weiyu/data_drive/physics_eval_transformer_dinner_10k_test_objects",
    #         assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #         object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #         redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #         override_data_dirs=["/home/weiyu/data_drive/data_test_objects/dinner_data/result"],
    #         override_index_dirs=["index"])
    #
    # # 10: [0, 1, 2, 6, 12, 14, 15, 16, 18, 19]
    # ###################################################################################################################
    # # stacking 10k
    # evaluate(structure_type="tower",
    #          generator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220809/20220815-144822/best_model",
    #         data_split="train", data_root="/home/weiyu/data_drive/physics_eval_transformer_stacking_10k_test_objects",
    #         assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #         object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #         redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #         override_data_dirs=["/home/weiyu/data_drive/data_test_objects/stacking_data/result"],
    #         override_index_dirs=["index"])
    # # 14: [0, 4, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 20]
    # ###################################################################################################################
    # # circle 10k
    # evaluate(structure_type="circle",
    #          generator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220814-223710/best_model",
    #         data_split="train", data_root="/home/weiyu/data_drive/physics_eval_transformer_circle_10k_test_objects",
    #         assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #         object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #         redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #         override_data_dirs=["/home/weiyu/data_drive/data_test_objects/circle_data/result"],
    #         override_index_dirs=["index"])
    # # 16: [0, 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20]
    # ###################################################################################################################
    # # line 10k
    # evaluate(structure_type="line",
    #          generator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220814-223820/best_model",
    #         data_split="train", data_root="/home/weiyu/data_drive/physics_eval_transformer_line_10k_test_objects",
    #         assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #         object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #         redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #         override_data_dirs=["/home/weiyu/data_drive/data_test_objects/line_data/result"],
    #         override_index_dirs=["index"])
    # # 11: [1, 3, 5, 6, 7, 9, 10, 11, 13, 14, 18]
    # ####################################################################################################################
    # ####################################################################################################################


