import h5py
import trimesh
import json
import os
import numpy as np
from src.rearrangement_utils import show_pcs_color_order
import pandas as pd
import brain2.utils.transformations as tra
import time

from src.tokenizer import Tokenizer

def load_h5_key(h5, key):
    if key in h5:
        return h5[key][()]
    elif "json_" + key in h5:
        return json.loads(h5["json_" + key][()])
    else:
        return None


def load_object_mesh_from_object_info(assets_dir, object_urdf):
    mesh_path = os.path.join(assets_dir, "visual", object_urdf[:-5], "model.obj")
    object_visual_mesh = trimesh.load(mesh_path, force="mesh")
    return object_visual_mesh


def visualize_result(filename, assets_dir, save_file_prefix=None, verbose=False, tokenizer=None):

    h5 = h5py.File(filename, 'r')
    # moved_objs = h5['moved_objs'][()].split(',')
    goal_specification = load_h5_key(h5, "goal_specification")
    num_rearrange_objs = len(goal_specification["rearrange"]["objects"])

    target_obj_urdfs = [obj_spec["urdf"] for obj_spec in goal_specification["rearrange"]["objects"]]

    structure_parameters = goal_specification["shape"]
    if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
        target_obj_urdfs = target_obj_urdfs[::-1]
    elif structure_parameters["type"] == "tower" or structure_parameters["type"] == "dinner":
        target_obj_urdfs = target_obj_urdfs
    else:
        raise KeyError("{} structure is not recognized".format(structure_parameters["type"]))

    # target_obj_urdfs = target_obj_urdfs[::-1]

    target_obj_vis_meshes = [load_object_mesh_from_object_info(assets_dir, u) for u in target_obj_urdfs]

    # for mesh in target_obj_vis_meshes:
    #     mesh.show()

    target_objs = load_h5_key(h5, "target_objs")
    current_obj_poses = load_h5_key(h5, "current_obj_poses")
    current_pc_poses = load_h5_key(h5, "current_pc_poses")
    goal_pc_poses = load_h5_key(h5, "goal_pc_poses")

    # compute goal poses
    predicted_goal_obj_poses = []
    for i, obj_name in enumerate(target_objs):

        current_obj_pose = current_obj_poses[i]
        current_pc_pose = current_pc_poses[i]
        goal_pc_pose = goal_pc_poses[i]

        predicted_goal_obj_pose = goal_pc_pose @ np.linalg.inv(current_pc_pose) @ current_obj_pose
        predicted_goal_obj_poses.append(predicted_goal_obj_pose)

        target_obj_vis_meshes[i].apply_transform(predicted_goal_obj_pose)

        if verbose:
            print(obj_name)
            print("current pc pose", current_pc_pose)
            print("goal pc pose", goal_pc_pose)
            print("current object pose", current_obj_pose)
            print("goal object pose", predicted_goal_obj_pose)

    sentence = load_h5_key(h5, "sentence")
    if tokenizer:
        print(tokenizer.convert_structure_params_to_natural_language(sentence))

    check = load_h5_key(h5, "check")
    discriminator_score = load_h5_key(h5, "discriminator_score")
    check_dict = load_h5_key(h5, "check_dict")

    print("\n" + "-"*50)
    print("filename", filename)
    print("sentence", sentence)
    print("check", check)
    print("discriminator score", discriminator_score)
    if verbose:
        print("check dict", check_dict)

    # print(check_dict)
    error_types = []
    if check == False:
        for key in check_dict:
            key_checks = check_dict[key]
            if type(key_checks) == list:
                if False in key_checks:
                    error_types.append(key)
            else:
                if key_checks == False:
                    error_types.append(key)
        print("error types:", error_types)

    num_target_objs = len(load_h5_key(h5, "target_objs"))

    pcs = load_h5_key(h5, "new_obj_xyzs")[:num_target_objs]
    obj_pcs_vis = [trimesh.PointCloud(pc_obj[:, :3], colors=[255, 0, 0, 255]) for pc_obj in pcs]
    # trimesh.Scene(obj_pcs_vis).show()

    if save_file_prefix is not None:
        save_pc_filename = save_file_prefix + "_pc.png"
        save_mesh_filename = save_file_prefix + "_mesh.png"
    else:
        save_pc_filename = None
        save_mesh_filename = None

    time.sleep(1)
    show_pcs_color_order([xyz[:, :3] for xyz in pcs], None, visualize=(save_pc_filename is None), add_coordinate_frame=False,
                         side_view=False, save_path=save_pc_filename, add_table=True)

    # scene_pc = load_h5_key(h5, "subsampled_scene_xyz")
    # trimesh.PointCloud(scene_pc[:, :3]).show()

    scene = trimesh.Scene()
    # add the coordinate frame first
    geom = trimesh.creation.axis(0.01)
    # scene.add_geometry(geom)
    table = trimesh.creation.box(extents=[1.0, 1.0, 0.02])
    table.apply_translation([0.5, 0, -0.01])
    table.visual.vertex_colors = [150, 111, 87, 125]
    scene.add_geometry(table)

    # for obj_vis_mesh in target_obj_vis_meshes:
    #     obj_vis_mesh.visual.vertex_colors = [50, 50, 50, 100]

    # scene.add_geometry(obj_pcs_vis)

    # RT_4x4 = np.array([[-0.7147778097036409, -0.6987369263935487, 0.02931536200292423, 0.3434544782290732],
    #                    [-0.47073865286968597, 0.4496990781074231, -0.7590624874434035, 0.10599949513304896],
    #                    [0.5172018981497449, -0.5563608962206371, -0.6503574015161744, 5.32058832987803],
    #                    [0.0, 0.0, 0.0, 1.0]])
    # RT_4x4 = np.linalg.inv(RT_4x4)
    # RT_4x4 = RT_4x4 @ np.diag([1, -1, -1, 1])
    # scene.camera_transform = RT_4x4

    # RT_4x4 = np.array([[-0.005378542346186285, 0.9998161380851034, -0.018405469481106367, -0.00822956735846642],
    #                    [0.9144495496588564, -0.0025306059632757057, -0.40469200283941076, -0.5900283926985573],
    #                    [-0.40466417238365116, -0.019007526352692254, -0.9142678062422447, 1.636231273015809],
    #                    [0.0, 0.0, 0.0, 1.0]])
    # RT_4x4 = np.linalg.inv(RT_4x4)
    # RT_4x4 = RT_4x4 @ np.diag([1, -1, -1, 1])
    # scene.camera_transform = RT_4x4

    # if save_mesh_filename:
    #     scene.add_geometry(target_obj_vis_meshes)
    #     # scene.apply_transform(tra.euler_matrix(0, 0, np.pi / 2))
    #
    #     RT_4x4 = np.array([[-0.45176964096260663, -0.8692415544879025, 0.20080665192162478, 0.1763777650396091],
    #                        [-0.641881546728535, 0.16037991603449953, -0.7498442254909686, 0.3439043544971707],
    #                        [0.6195904062151876, -0.4676509408567697, -0.6304048905599298, 1.2323325145731152],
    #                        [0.0, 0.0, 0.0, 1.0]])
    #     RT_4x4 = np.linalg.inv(RT_4x4)
    #     RT_4x4 = RT_4x4 @ np.diag([1, -1, -1, 1])
    #     scene.camera_transform = RT_4x4
    #
    #     png = scene.save_image(resolution=[1000, 1000],
    #                            visible=True)
    #     with open(save_mesh_filename, 'wb') as f:
    #         f.write(png)
    # else:
    #     for obj_vis_mesh in target_obj_vis_meshes:
    #         obj_vis_mesh.visual.vertex_colors = [50, 50, 50, 100]
    #     scene.add_geometry(target_obj_vis_meshes)
    #     scene.add_geometry(obj_pcs_vis)
    #
    #     RT_4x4 = np.array([[-0.45176964096260663, -0.8692415544879025, 0.20080665192162478, 0.1763777650396091],
    #      [-0.641881546728535, 0.16037991603449953, -0.7498442254909686, 0.3439043544971707],
    #      [0.6195904062151876, -0.4676509408567697, -0.6304048905599298, 1.2323325145731152], [0.0, 0.0, 0.0, 1.0]])
    #     RT_4x4 = np.linalg.inv(RT_4x4)
    #     RT_4x4 = RT_4x4 @ np.diag([1, -1, -1, 1])
    #     scene.camera_transform = RT_4x4
    #
    #     scene.show()


def performance_breakdown(filename):
    h5 = h5py.File(filename, 'r')
    goal_specification = load_h5_key(h5, "goal_specification")
    num_rearrange_objs = len(goal_specification["rearrange"]["objects"])

    check = load_h5_key(h5, "check")
    check_dict = load_h5_key(h5, "check_dict")

    # print(check_dict)
    error_types = []
    if check == False:
        for key in check_dict:
            key_checks = check_dict[key]
            if type(key_checks) == list:
                if False in key_checks:
                    error_types.append(key)
            else:
                if key_checks == False:
                    error_types.append(key)
        # print("error types:", error_types)

    # if "pvel_norm_velocity" in error_types or "rvel_max_velocity" in error_types:
    #     if "pvel_norm_velocity" in error_types:
    #         error_types.remove("pvel_norm_velocity")
    #     if "rvel_max_velocity" in error_types:
    #         error_types.remove("rvel_max_velocity")
    #     error_types.append("placing_velocity")

    # place_error = False
    # place_error_types = ["pvel_norm_velocity", "rvel_max_velocity", "placing_no_intersection"]
    # for place_error_type in place_error_types:
    #     if place_error_type in error_types:
    #         place_error = True
    #         error_types.remove(place_error_type)
    # if place_error:
    #     error_types = ["placing"]

    if "placing_no_intersection" in error_types:
        error_types = ["placing_no_intersection"]
    if "pvel_norm_velocity" in error_types or "rvel_max_velocity" in error_types:
        error_types = ["placing_velocity"]


    return error_types


def iterate_files(base_dir, assets_dir):

    filenames = sorted(os.listdir(base_dir))

    for filename in filenames:
        visualize_result(os.path.join(base_dir, filename), assets_dir)


def iterate_files_performance(base_dir):
    filenames = sorted(os.listdir(base_dir))

    all_error_types = []
    num_failures = 0
    num_trails = 0
    for filename in filenames:
        if ".h5" not in filename:
            continue
        errors_types = performance_breakdown(os.path.join(base_dir, filename))
        if errors_types:
            all_error_types.extend(errors_types)
            num_failures += 1
        num_trails += 1

    num_successes = num_trails - num_failures
    print("Success rate {}/{} ({}%)".format(num_successes, num_trails, num_successes * 100.0 / num_trails))

    # for error_type in ["placing_no_intersection", "placing_velocity", "no_intersection", "above_table", "upright", "contact_with_table", "close_to_line", "line order", "close_to_circle", "stacking", "z_rotation"]:
    error_counts = []
    # errors_types = ["placing", "no_intersection", "above_table", "upright", "contact_with_table", "close_to_line", "line order", "close_to_circle", "stacking", "z_rotation"]
    errors_types = ["success rate", "placing_no_intersection", "placing_velocity", "no_intersection", "above_table", "upright", "contact_with_table", "close_to_line", "line order", "close_to_circle", "stacking", "z_rotation"]
    for error_type in errors_types:
        if error_type in all_error_types:
            num_type_failure = all_error_types.count(error_type)
            print("Failure due to {}: {}/{} ({}%)".format(error_type, num_type_failure, num_trails, num_type_failure*100.0/num_trails))
            error_counts.append(num_type_failure*1.0/num_trails)
        else:
            if error_type == "success rate":
                error_counts.append(num_successes*1.0/num_trails)
            else:
                error_counts.append(0)
    return errors_types, error_counts


def iterate_files_conditional(base_dir, assets_dir):

    filenames = sorted(os.listdir(base_dir))

    num_success = 0.0
    num_all = 0.0
    for filename in filenames:

        filename = os.path.join(base_dir, filename)

        h5 = h5py.File(filename, 'r')
        check = load_h5_key(h5, "check")
        discriminator_score = load_h5_key(h5, "discriminator_score")

        # if check != True:
        #     continue

        if check != False:
            continue

        visualize_result(filename, assets_dir)

        if check == True:
            num_success += 1
        num_all += 1

    print("Success rate:", num_success/num_all)



def iterate_files_conditional_comparison(base_dir1, base_dir2, assets_dir):
    filenames1 = sorted(os.listdir(base_dir1))
    filenames2 = sorted(os.listdir(base_dir2))

    for filename1, filename2 in zip(filenames1, filenames2):

        filename1 = os.path.join(base_dir1, filename1)
        filename2 = os.path.join(base_dir2, filename2)

        h51 = h5py.File(filename1, 'r')
        check1 = load_h5_key(h51, "check")
        discriminator_score1 = load_h5_key(h51, "discriminator_score")

        h52 = h5py.File(filename2, 'r')
        check2 = load_h5_key(h52, "check")
        discriminator_score2 = load_h5_key(h52, "discriminator_score")

        if check1 == True and check2 == False:
        # if check1 == True and check2 == False:

            visualize_result(filename1, assets_dir)
            visualize_result(filename2, assets_dir)


def visualize_all_models():
    root_dir = "/home/weiyu/data_drive/physics_eval_1016"
    result_dir = "/home/weiyu/Research/intern/semantic-rearrangement/src/physics_eval/results/1016"
    assets_dir = "/home/weiyu/data_drive/structformer_assets/housekeep_custom_handpicked_small"
    structure_types = ["dinner"] # ["circle", "stacking", "line", "dinner"]
    start_file_index = 0
    save_dir = None # "/home/weiyu/Desktop/StructDiffusion_Qualitative_2"
    file_idxs = [83]
    tokenizer = Tokenizer("/home/weiyu/data_drive/data_new_objects/type_vocabs_coarse.json")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    methods = sorted(os.listdir(root_dir))
    for structure_type in structure_types:

        if save_dir is not None:
            structure_save_dir = os.path.join(save_dir, structure_type)
            if not os.path.exists(structure_save_dir):
                os.makedirs(structure_save_dir)

        print(structure_type)
        method_to_files = {}
        for method in methods:
            method_data_dir = os.path.join(root_dir, method, "train")
            if structure_type in method:
                filenames = sorted(os.listdir(method_data_dir))
                method_to_files[method] = [os.path.join(method_data_dir, filename) for filename in filenames]

        methods = sorted(method_to_files.keys())
        print(methods)

        methods = ['transformer_{}', 'vae_{}', 'diffuser_{}', 'transformer_discriminator_cem_{}', 'vae_discriminator_cem_{}', 'transformer_collision_{}', 'vae_collision_{}', 'diffuser_{}_collision']
        # methods = ['vae_{}']
        methods = [m.format(structure_type) for m in methods]

        if not file_idxs:
            file_idxs = list(range(len(method_to_files[methods[0]])))
        for fi in file_idxs:

            if fi < start_file_index:
                continue

            print("-" * 100)
            print("file index", fi)
            for method in methods:
                filename = method_to_files[method][fi]
                h5 = h5py.File(filename, 'r')
                check = load_h5_key(h5, "check")
                print("Method {} success {}".format(method, check))
            print("-" * 100)

            if save_dir is not None or "y" == input("visualize this? hit y: "):
                for method in methods:
                    filename = method_to_files[method][fi]
                    print("visualize {} {}".format(method, filename))

                    if save_dir is not None:
                        save_file_prefix = os.path.join(structure_save_dir, "{}_{}".format(fi, method))
                    else:
                        save_file_prefix = None
                    visualize_result(filename, assets_dir, save_file_prefix=save_file_prefix, tokenizer=tokenizer)


def write_results():
    root_dir = "/home/weiyu/data_drive/physics_eval_1016"
    result_dir = "/home/weiyu/Research/intern/semantic-rearrangement/src/physics_eval/results/1016"
    assets_dir = "/home/weiyu/data_drive/structformer_assets/housekeep_custom_handpicked_small"
    base_names = sorted(os.listdir(root_dir))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    structure_types = ["circle", "stacking", "line", "dinner"]

    for structure_type in structure_types:

        method_to_result = {}
        all_error_types = None
        for base_name in base_names:
            print("\n\n" + "*"*50)
            print(base_name)
            if structure_type in base_name:
                error_types, error_counts = iterate_files_performance(os.path.join(root_dir, base_name, "train"))
                # iterate_files(os.path.join(root_dir, base_name, "train"), assets_dir)
                method_to_result[base_name] = error_counts
                if all_error_types is None:
                    all_error_types = error_types
                else:
                    assert all_error_types == error_types

        df = pd.DataFrame.from_dict(method_to_result, orient='index', columns=all_error_types)
        print(df)
        df.to_excel(os.path.join(result_dir, "{}.xlsx".format(structure_type)))

        normalized_df = df / df.max()
        normalized_df.to_excel(os.path.join(result_dir, "{}_norm.xlsx".format(structure_type)))

if __name__ == "__main__":

    # filename = "/home/weiyu/data_drive/physics_eval_vae_dinner_discriminator_test_objects_24k/train/data00000001_cem0.h5"
    # assets_dir = "/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4"



    # base_dir1 = "/home/weiyu/data_drive/physics_eval_vae_dinner_discriminator_test_objects_24k/train"
    # base_dir2 = "/home/weiyu/data_drive/physics_eval_vae_cem_all_shapes_dinner_lang_discriminator_local_shape_param_test_objects_24k/train"
    # base_dir3 = "/home/weiyu/data_drive/physics_eval_vae_dinner_24k/train"
    # iterate_files_conditional(base_dir, assets_dir)



    # base_dir1 = "/home/weiyu/data_drive/physics_eval_vae_dinner_discriminator_test_objects_24k/train"
    # base_dir2 = "/home/weiyu/data_drive/physics_eval_vae_cem_all_shapes_dinner_lang_discriminator_local_shape_param_test_objects_24k/train"
    # iterate_files_conditional_comparison(base_dir1, base_dir2, assets_dir)


    # base_dir1 = "/home/weiyu/data_drive/physics_eval_vae_circle_24k/train"
    # base_dir2 = "/home/weiyu/data_drive/physics_eval_vae_cem_all_shapes_circle_lang_discriminator_local_shape_param_test_objects_24k/train"
    # iterate_files_conditional_comparison(base_dir1, base_dir2, assets_dir)

    # base_dir = "/home/weiyu/data_drive/physics_eval_diffuser_dinner_10k_test_objects/train"
    # base_dir = "/home/weiyu/data_drive/physics_eval_diffuser_dinner_10k_test_objects_discriminator_local_shape_param_collision_random2/train"
    # base_dir = "/home/weiyu/data_drive/physics_eval_diffuser_line_10k_test_objects_discriminator_local_shape_param_collision/train"
    # base_dir = "/home/weiyu/data_drive/physics_eval_diffuser_template_sentence_dinner_10k_test_objects_discriminator_local_shape_param_collision_100/train"
    # iterate_files(base_dir, assets_dir)
    # iterate_files_conditional(base_dir, assets_dir)


    # base_dir1 = "/home/weiyu/data_drive/physics_eval_diffuser_stacking_10k_test_objects_discriminator_local_shape_param_collision/train"
    # base_dir2 = "/home/weiyu/data_drive/physics_eval_vae_cem_all_shapes_stacking_lang_discriminator_local_shape_param_test_objects_24k/train"
    # iterate_files_conditional_comparison(base_dir1, base_dir2, assets_dir)

    # base_dir = "/home/weiyu/data_drive/physics_eval_vae_cem_all_shapes_stacking_lang_discriminator_local_shape_param_test_objects_24k/train"
    # iterate_files(base_dir, assets_dir)

    # results_dict = {"test": "/home/weiyu/data_drive/physics_eval_vae_all_shapes_line_lang_discriminator_local_shape_param_collision_100_TESTESTEST/train",
    #                 # "structformer": "/home/weiyu/data_drive/physics_eval_transformer_{}_10k_test_objects/train",
    #                 # "structformer + collision": "/home/weiyu/data_drive/physics_eval_transformer_lang_{}_10k_test_objects_collision_100/train",
    #                 # "structformer + cem": "/home/weiyu/data_drive/physics_eval_transformer_lang_cem_{}_10k_test_objects/train",
    #                 # "vae": "/home/weiyu/data_drive/physics_eval_vae_{}/train",
    #                 # "vae + collision": "/home/weiyu/data_drive/physics_eval_vae_all_shapes_{}_lang_discriminator_local_shape_param_collision_100/train",
    #                 # "vae + lang cem": "/home/weiyu/data_drive/physics_eval_vae_cem_all_shapes_{}_lang_discriminator_local_shape_param_test_objects/train",
    #                 # "vae + single task cem": "/home/weiyu/data_drive/physics_eval_vae_{}_discriminator_test_objects/train",
    #                 # "diffuser": "/home/weiyu/data_drive/physics_eval_diffuser_{}_10k_test_objects_100/train",
    #                 # "diffuser + collision": "/home/weiyu/data_drive/physics_eval_diffuser_{}_10k_test_objects_discriminator_local_shape_param_collision_100/train"
    #                 }
    #
    # for structure_type in ["dinner"]:# ["dinner", "line", "stacking", "circle"]:
    #     print(structure_type)
    #     # # structformer
    #     # base_dir = "/home/weiyu/data_drive/physics_eval_transformer_{}_10k_test_objects/train".format(structure_type)
    #     # # structformer + cem
    #     # base_dir = "/home/weiyu/data_drive/physics_eval_transformer_lang_cem_{}_10k_test_objects"
    #     # # vae
    #     # base_dir = "/home/weiyu/data_drive/physics_eval_vae_{}"
    #     # # vae + lang cem
    #     # base_dir = "/home/weiyu/data_drive/physics_eval_vae_cem_all_shapes_{}_lang_discriminator_local_shape_param_test_objects"
    #     # # vae + single task cem
    #     # base_dir = "/home/weiyu/data_drive/physics_eval_vae_{}_discriminator_test_objects"
    #     # # diffuser + collision
    #     # base_dir = "/home/weiyu/data_drive/physics_eval_diffuser_{}_10k_test_objects_discriminator_local_shape_param_collision_100/train".format(structure_type)
    #     # # diffuser
    #     # base_dir = "/home/weiyu/data_drive/physics_eval_diffuser_{}_10k_test_objects_100/train".format(structure_type)
    #     # # structformer + collision
    #     # base_dir = "/home/weiyu/data_drive/physics_eval_transformer_lang_{}_10k_test_objects_collision_100/train".format(structure_type)
    #     # # vae + collision
    #     # base_dir = "/home/weiyu/data_drive/physics_eval_vae_all_shapes_{}_lang_discriminator_local_shape_param_collision_100/train".format(structure_type)
    #
    #     for model_name in results_dict:
    #         print("\n\n" + "*"*100)
    #         print(model_name + "\n")
    #         base_dir = results_dict[model_name].format(structure_type)
    #
    #         # iterate_files_performance(base_dir)
    #         iterate_files(base_dir, assets_dir)

        # iterate_files(base_dir1, assets_dir)
        # iterate_files_performance(base_dir)
        # iterate_files_conditional(base_dir, assets_dir)
    #
    #     iterate_files_conditional_comparison(base_dir1, base_dir2, assets_dir)




    # for structure_type in ["dinner", "line", "stacking", "circle"]:
    #     print("\n" + "*" * 100)
    #     print(structure_type)
    #     base_dir = "/home/weiyu/data_drive/physics_eval_diffuser_template_sentence_{}_10k_test_objects_discriminator_local_shape_param_collision_100/train".format(structure_type)
    #     iterate_files_performance(base_dir)




    # for structure_type in ["dinner"]: #["dinner", "line", "stacking", "circle"]:
    #         print("\n" + "*" * 100)
    #         print(structure_type)
    #         base_dir = "/home/weiyu/data_drive/physics_eval_diffuser_d1c0_{}_10k_test_objects_discriminator_local_shape_param_collision_100/train".format(structure_type)
    #         iterate_files_performance(base_dir)




    # for inw in [0.1, 0.5, 1.0, 2.0]:
    #     print("\n\n" + "#" * 100)
    #     print(inw)
    #     for structure_type in ["dinner", "line", "stacking", "circle"]:
    #         print("\n" + "*" * 100)
    #         print(structure_type)
    #         base_dir = "/home/weiyu/data_drive/physics_eval_diffuser_compositional_two_perturbation_i{}_{}_10k_test_objects_100/train".format(inw, structure_type)
    #         iterate_files_performance(base_dir)

    # root_dir = "/home/weiyu/data_drive/physics_eval_0914"
    # base_names = sorted(os.listdir(root_dir))
    # for base_name in base_names:
    #     print("\n\n" + "*"*50)
    #     print(base_name)
    #     iterate_files_performance(os.path.join(root_dir, base_name, "train"))
    #     # iterate_files(os.path.join(root_dir, base_name, "train"), assets_dir)

    # root_dir = "/home/weiyu/data_drive/physics_eval_1016"
    # result_dir = "/home/weiyu/Research/intern/semantic-rearrangement/src/physics_eval/results/1016"
    # assets_dir = "/home/weiyu/data_drive/structformer_assets/housekeep_custom_handpicked_small"
    # base_names = sorted(os.listdir(root_dir))
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)
    #
    # structure_types = ["circle", "stacking", "line", "dinner"]
    #
    # for structure_type in structure_types:
    #
    #     method_to_result = {}
    #     all_error_types = None
    #     for base_name in base_names:
    #         print("\n\n" + "*"*50)
    #         print(base_name)
    #         if structure_type in base_name:
    #             error_types, error_counts = iterate_files_performance(os.path.join(root_dir, base_name, "train"))
    #             iterate_files(os.path.join(root_dir, base_name, "train"), assets_dir)
    #             method_to_result[base_name] = error_counts
    #             if all_error_types is None:
    #                 all_error_types = error_types
    #             else:
    #                 assert all_error_types == error_types
    #
    #     df = pd.DataFrame.from_dict(method_to_result, orient='index', columns=all_error_types)
    #     print(df)
    #     df.to_excel(os.path.join(result_dir, "{}.xlsx".format(structure_type)))
    #
    #     normalized_df = df / df.max()
    #     normalized_df.to_excel(os.path.join(result_dir, "{}_norm.xlsx".format(structure_type)))

    # write_results()
    # visualize_all_models()

    assets_dir = "/home/weiyu/data_drive/structformer_assets/housekeep_custom_handpicked_small"
    base_dir = "/home/weiyu/data_drive/physics_eval_vae_cem_all_shapes_stacking_lang_discriminator_local_shape_param_test_objects_24k/train"
    iterate_files(base_dir, assets_dir)