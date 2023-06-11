import torch
import numpy as np
import os
import pytorch3d.transforms as tra3d
import json
import argparse
from omegaconf import OmegaConf

# physics eval
from StructDiffusion.utils.physics_eval import switch_stdout, visualize_batch_pcs, convert_bool, save_dict_to_h5, move_pc
from rearrangement_gym.semantic_rearrangement.physics_verification_dinner import verify_datum_in_simulation

# inference
from StructDiffusion.evaluation.infer_vae import PriorInference


def evaluate(random_seed, structure_type, generator_model_dir, data_split, data_root,
            assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
            object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
            redirect_stdout=False, shuffle=False, summary_writer=None, max_num_eval=10, visualize=False,
            override_data_dirs=None, override_index_dirs=None, physics_eval_early_stop=True, **kwargs):

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
                                     override_data_dirs=override_data_dirs, override_index_dirs=override_index_dirs)
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

        ####################################################
        # sample S predictions

        sample_raw_data = prior_dataset.get_raw_data(data_idx, inference_mode=True)
        sample_file_id = os.path.splitext(os.path.basename(sample_raw_data["filename"]))[0]
        print(sample_file_id)
        samples_data = [prior_dataset.convert_to_tensors(sample_raw_data, prior_dataset.tokenizer)] * B
        # for _ in range(num_samples):
        #     datum = test_dataset[data_idx]
        #     samples_data.append(datum)

        struct_preds, obj_preds = prior_inference.limited_batch_inference(samples_data, verbose=False,
                                                                                    convert_to_tensors=False,
                                                                                    return_numpy=False)

        # struct_preds: B, 12
        # obj_preds: B, N, 12
        ####################################################
        # only keep one copy

        sentence = samples_data[0]["sentence"].to(device).unsqueeze(0)  # 1, sentence_length
        sentence_pad_mask = samples_data[0]["sentence_pad_mask"].to(device).unsqueeze(0)  # 1, sentence_length
        ####################################################
        # only keep one copy

        # N, P, 3
        obj_xyzs = samples_data[0]["xyzs"].to(device)
        print("obj_xyzs shape", obj_xyzs.shape)

        # 1, N
        # object_pad_mask: padding location has 1
        object_pad_mask = samples_data[0]["object_pad_mask"].to(device).unsqueeze(0)
        target_object_inds = 1 - object_pad_mask
        print("target_object_inds shape", target_object_inds.shape)
        print("target_object_inds", target_object_inds)

        N, P, _ = obj_xyzs.shape
        print("B, N, P: {}, {}, {}".format(B, N, P))

        if visualize:
            visualize_batch_pcs(obj_xyzs, 1, N, P)
        # visualize_batch_pcs(obj_xyzs, 1, N, P)

        ####################################################
        # S, N, ...

        # Important: we don't optimize structure
        struct_pose = torch.eye(4).repeat(B, 1, 1).to(device)  # B, 4, 4
        struct_pose[:, :3, :3] = struct_preds[:, 3:].reshape(-1, 3, 3)
        struct_pose[:, :3, 3] = struct_preds[:, :3]
        struct_pose = struct_pose.repeat_interleave(N, dim=0)  # B x N, 4, 4

        current_pc_pose = torch.eye(4).repeat(N, 1, 1).to(device)  # N, 4, 4
        current_pc_pose[:, :3, 3] = torch.mean(obj_xyzs, dim=1)  # N, 4, 4
        current_pc_pose = current_pc_pose.repeat(B, 1, 1)  # B x N, 4, 4

        # struct_params = torch.rand((B, 6)).to(device)
        # struct_params[:, :3] = struct_preds[:, :3]
        # # Important: tra3d.matrix_to_euler_angles return [-pi, pi]
        # struct_params[:, 3:] = tra3d.matrix_to_euler_angles(struct_preds[:, 3:].reshape(B, 3, 3), "XYZ")

        obj_params = torch.zeros((B, N, 6)).to(device)
        obj_preds = obj_preds.reshape(B, N, -1)  # B, N, 12
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
            check, check_dict = verify_datum_in_simulation(d, assets_path, object_model_dir, structure_type=structure_type,
                                                           early_stop=physics_eval_early_stop, visualize=False)
            print("data point {}, check {}".format(data_idx, bool(check)))
            physics_results.append((check, check_dict))

        for check, check_dict in physics_results:
            print(check, check_dict)
        # visualize_batch_pcs(best_new_obj_xyzs, num_best_so_far, N, P)

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
    parser.add_argument("--base_config_file", help='base config yaml file',
                        default='../../../configs/physics_eval/dataset_housekeep_custom/base.yaml',
                        type=str)
    parser.add_argument("--config_file", help='config yaml file',
                        default='../../../configs/physics_eval/dataset_housekeep_custom/vae/circle_test.yaml',
                        type=str)
    args = parser.parse_args()
    assert os.path.exists(args.base_config_file), "Cannot find base config yaml file at {}".format(args.config_file)
    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)
    base_cfg = OmegaConf.load(args.base_config_file)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(base_cfg, cfg)

    cfg.physics_eval_early_stop = False

    evaluate(**cfg)

    # all shapes
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/models/generative_models/actor_vae_language/best_model"
    # circle vae object shuffle
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220223-225258/best_model"
    # test_model_batch(args.model_dir)
    # exit()
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/models/actor_vae_language_all_shapes_new_stacking/best_model"

    # args.discriminator_model_dir = None
    # tower
    # args.discriminator_model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220209-224034/best_model"
    # circle
    # args.discriminator_model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220215-212829/best_model"
    # dinner
    # args.discriminator_model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220216-101729/best_model"

    # tower transformer pct random sampling, using random sampling branch, 4096 scene pts, language
    # args.discriminator_model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220515-120043/best_model"
    # line transformer pct random sampling, using random sampling branch, 4096 scene pts, language
    # args.discriminator_model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220426-092948/best_model"

    # tower 2 objs
    # args.discriminator_model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220209-231107/best_model"
    # tower 3 objs
    # args.discriminator_model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220210-100522/best_model"


    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/models/actor_vae_language_all_shapes_new_stacking/best_model"
    # evaluate(generator_model_dir=args.model_dir, data_split="test", data_root="/home/weiyu/data_drive/physics_eval_vae",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
    #          redirect_stdout=False, shuffle=False, summary_writer=None, max_num_eval=30, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_new_objects/examples_line_new_objects/result"],
    #          override_index_dirs=["index_10k"])
    ####################################################################################################################
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/models/actor_vae_language_all_shapes_new_stacking/best_model"
    # evaluate(generator_model_dir=args.model_dir, data_split="test", data_root="/home/weiyu/data_drive/physics_eval_vae_stacking",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
    #          redirect_stdout=False, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_new_objects/examples_stacking_new_objects/result"],
    #          override_index_dirs=["index_10k"])

    ####################################################################################################################
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220718-115940/best_model"
    # evaluate(generator_model_dir=args.model_dir, data_split="test",
    #          data_root="/home/weiyu/data_drive/physics_eval_vae_line_discriminator",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
    #          redirect_stdout=False, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_new_objects/examples_line_new_objects_test_discriminator/result"],
    #          override_index_dirs=["index_10k"])

    ####################################################################################################################
    # batch testing
    ####################################################################################################################
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/models/actor_vae_language_all_shapes_new_stacking/best_model"
    # evaluate(structure_type="line",
    #          generator_model_dir=args.model_dir, data_split="test",
    #          data_root="/home/weiyu/data_drive/physics_eval_vae_line",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_new_objects/examples_line_new_objects/result"],
    #          override_index_dirs=["index_10k"])
    #
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/models/actor_vae_language_all_shapes_new_stacking/best_model"
    # evaluate(structure_type="dinner",
    #          generator_model_dir=args.model_dir, data_split="test",
    #          data_root="/home/weiyu/data_drive/physics_eval_vae_dinner",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_new_objects/examples_dinner_new_objects/result"],
    #          override_index_dirs=["index_10k"])
    #
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/models/actor_vae_language_all_shapes_new_stacking/best_model"
    # evaluate(structure_type="tower",
    #          generator_model_dir=args.model_dir, data_split="test",
    #          data_root="/home/weiyu/data_drive/physics_eval_vae_stacking",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_new_objects/examples_stacking_new_objects/result"],
    #          override_index_dirs=["index_10k"])
    #
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/models/actor_vae_language_all_shapes_new_stacking/best_model"
    # evaluate(structure_type="circle",
    #          generator_model_dir=args.model_dir, data_split="test",
    #          data_root="/home/weiyu/data_drive/physics_eval_vae_circle",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_new_objects/examples_circle_new_objects/result"],
    #          override_index_dirs=["index_10k"])
    ####################################################################################################################
    ####################################################################################################################





    # ####################################################################################################################
    # # batch testing for test objects
    # ####################################################################################################################
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/models/actor_vae_language_all_shapes_new_stacking/best_model"
    # evaluate(structure_type="line",
    #          generator_model_dir=args.model_dir, data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_vae_line",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=50, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/line_data/result"],
    #          override_index_dirs=["index"])
    #
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/models/actor_vae_language_all_shapes_new_stacking/best_model"
    # evaluate(structure_type="dinner",
    #          generator_model_dir=args.model_dir, data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_vae_dinner",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=50, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/dinner_data/result"],
    #          override_index_dirs=["index"])
    #
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/models/actor_vae_language_all_shapes_new_stacking/best_model"
    # evaluate(structure_type="tower",
    #          generator_model_dir=args.model_dir, data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_vae_stacking",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=50, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/stacking_data/result"],
    #          override_index_dirs=["index"])
    #
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/models/actor_vae_language_all_shapes_new_stacking/best_model"
    # evaluate(structure_type="circle",
    #          generator_model_dir=args.model_dir, data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_vae_circle",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=50, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/circle_data/result"],
    #          override_index_dirs=["index"])
    # ####################################################################################################################
    # ####################################################################################################################







    # ####################################################################################################################
    # # batch testing for test objects
    # ####################################################################################################################
    # args.model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220831-112256/best_model"
    # evaluate(structure_type="dinner",
    #          generator_model_dir=args.model_dir, data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_vae_dinner_24k_TESTESTEST",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=False, shuffle=False, summary_writer=None, max_num_eval=50, visualize=True,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/dinner_data/result"],
    #          override_index_dirs=["index"])
    #
    # evaluate(structure_type="line",
    #          generator_model_dir=args.model_dir, data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_vae_line_24k",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=50, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/line_data/result"],
    #          override_index_dirs=["index"])
    #
    # evaluate(structure_type="tower",
    #          generator_model_dir=args.model_dir, data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_vae_stacking_24k",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=50, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/stacking_data/result"],
    #          override_index_dirs=["index"])
    #
    # evaluate(structure_type="circle",
    #          generator_model_dir=args.model_dir, data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_vae_circle_24k",
    #          test_specific_shape=None,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=50, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/circle_data/result"],
    #          override_index_dirs=["index"])
    # ####################################################################################################################
    # ####################################################################################################################