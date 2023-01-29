import copy
import cv2
import h5py
import numpy as np
import os
import trimesh
import torch
from tqdm import tqdm
import json
import random

# Local imports
from StructDiffusion.utils.rearrangement import show_pcs, get_pts, combine_and_sample_xyzs, random_move_obj_xyzs, array_to_tensor
from StructDiffusion.data.tokenizer import Tokenizer
from StructDiffusion.data.MeshSceneViewer import MeshSceneViewer

import StructDiffusion.utils.brain2.camera as cam
import StructDiffusion.utils.brain2.image as img
import StructDiffusion.utils.transformations as tra


class SemanticArrangementDataset(torch.utils.data.Dataset):

    def __init__(self, data_roots, index_roots, split, tokenizer,
                 min_translation, max_translation, min_rotation, max_rotation, perturbation_mode,
                 num_random_perturbations_per_positive=1, oversample_positive=False,
                 max_num_objects=11, max_num_shape_parameters=7,
                 num_pts=1024, filter_num_moved_objects_range=None, debug=False, shuffle_object_index=False,
                 data_augmentation=True, mesh_scene_viewer=None):
        """
        Note: setting filter_num_moved_objects_range=[k, k] and max_num_objects=k will create no padding for target objs
        :param shuffle_object_index: whether to shuffle the positions of target objects and other objects in the sequence
        """

        print("data augmentation is set to ", data_augmentation)

        self.data_roots = data_roots
        self.num_pts = num_pts
        print("data dirs:", self.data_roots)
        self.debug = debug

        self.max_num_objects = max_num_objects
        self.max_num_shape_parameters = max_num_shape_parameters
        self.shuffle_object_index = shuffle_object_index

        # parameters for generating negative examples
        self.perturbation_mode = perturbation_mode
        self.min_translation = min_translation
        self.max_translation = max_translation
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation

        self.tokenizer = tokenizer
        self.mesh_scene_viewer = mesh_scene_viewer

        self.arrangement_data = []
        # for data_root, index_root in zip(data_roots, index_roots):
        for ddx in range(len(data_roots)):
            data_root = data_roots[ddx]
            index_root = index_roots[ddx]
            arrangement_indices_file = os.path.join(data_root, index_root, "{}_arrangement_indices_file_all.txt".format(split))
            if os.path.exists(arrangement_indices_file):
                with open(arrangement_indices_file, "r") as fh:
                    self.arrangement_data.extend([(os.path.join(data_root, f[0]), f[1]) for f in eval(fh.readline().strip())])
            else:
                print("{} does not exist".format(arrangement_indices_file))

        # filter based on num objs
        if filter_num_moved_objects_range is not None:
            self.arrangement_data = self.filter_based_on_number_of_moved_objects(filter_num_moved_objects_range)
        print("{} valid sequences".format(len(self.arrangement_data)))

        # create positive and negative examples
        # each data point is a tuple of (filename, step_t, perturbation_label), where label is True or False
        arrangement_data = []
        print("Adding {} perturbations per arrangement step".format(num_random_perturbations_per_positive))
        for filename, step_t in self.arrangement_data:

            # sample n negative examples from each step
            for _ in range(num_random_perturbations_per_positive):
                arrangement_data.append((filename, step_t, True))

            if not oversample_positive:
                arrangement_data.append((filename, step_t, False))
            else:
                # balance positive and negative examples
                for _ in range(num_random_perturbations_per_positive):
                    arrangement_data.append((filename, step_t, False))
        print("{} original and perturbed examples in total".format(len(arrangement_data)))
        self.arrangement_data = arrangement_data

        # Noise
        self.data_augmentation = data_augmentation
        # additive noise
        self.gp_rescale_factor_range = [12, 20]
        self.gaussian_scale_range = [0., 0.003]
        # multiplicative noise
        self.gamma_shape = 1000.
        self.gamma_scale = 0.001

    def filter_based_on_number_of_moved_objects(self, filter_num_moved_objects_range):
        assert len(list(filter_num_moved_objects_range)) == 2
        min_num, max_num = filter_num_moved_objects_range
        print("Remove scenes that have less than {} or more than {} objects being moved".format(min_num, max_num))
        ok_data = []
        for filename, step_t in self.arrangement_data:
            h5 = h5py.File(filename, 'r')
            moved_objs = h5['moved_objs'][()].split(',')
            if min_num <= len(moved_objs) <= max_num:
                ok_data.append((filename, step_t))
        print("{} valid sequences left".format(len(ok_data)))
        return ok_data

    def helper_iterate_through_files(self):
        xs = []
        ys = []
        thetas = []
        for idx in tqdm(range(len(self.arrangement_data))):
            filename, t = self.arrangement_data[idx]
            h5 = h5py.File(filename, 'r')
            ids = self._get_ids(h5)
            # moved_objs = h5['moved_objs'][()].split(',')
            all_objs = sorted([o for o in ids.keys() if "object_" in o])

            for obj in all_objs:
                obj_pose = h5[obj][t]
                T = np.eye(4)
                T[:3, :3] = obj_pose[:3, :3]
                _, _, theta = tra.euler_from_matrix(T)
                # print("{} t{} obj {}: x {}, y {}, theta {}".format(filename, t, obj, obj_pose[0, 3], obj_pose[1, 3], theta))
                xs.append(obj_pose[0, 3])
                ys.append(obj_pose[1, 3])
                thetas.append(theta)

        print("x min {} max {}".format(min(xs), max(xs)))
        print("y min {} max {}".format(min(ys), max(ys)))
        print("theta min {} max {}".format(min(thetas), max(thetas)))

    def get_data_idx(self, idx):
        # Create the datum to return
        file_idx = np.argmax(idx < self.file_to_count)
        data = h5py.File(self.data_files[file_idx], 'r')
        if file_idx > 0:
            # for lang2sym, idx is always 0
            idx = idx - self.file_to_count[file_idx - 1]
        return data, idx, file_idx

    def add_noise_to_depth(self, depth_img):
        """ add depth noise """
        multiplicative_noise = np.random.gamma(self.gamma_shape, self.gamma_scale)
        depth_img = multiplicative_noise * depth_img
        return depth_img

    def add_noise_to_xyz(self, xyz_img, depth_img):
        """ TODO: remove this code or at least celean it up"""
        xyz_img = xyz_img.copy()
        H, W, C = xyz_img.shape
        gp_rescale_factor = np.random.randint(self.gp_rescale_factor_range[0],
                                              self.gp_rescale_factor_range[1])
        gp_scale = np.random.uniform(self.gaussian_scale_range[0],
                                     self.gaussian_scale_range[1])
        small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
        additive_noise = np.random.normal(loc=0.0, scale=gp_scale, size=(small_H, small_W, C))
        additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
        xyz_img[depth_img > 0, :] += additive_noise[depth_img > 0, :]
        return xyz_img

    def random_index(self):
        return self[np.random.randint(len(self))]

    def _get_rgb(self, h5, idx, ee=True):
        RGB = "ee_rgb" if ee else "rgb"
        rgb1 = img.PNGToNumpy(h5[RGB][idx])[:, :, :3] / 255.  # remove alpha
        return rgb1

    def _get_depth(self, h5, idx, ee=True):
        DEPTH = "ee_depth" if ee else "depth"

    def _get_images(self, h5, idx, ee=True):
        if ee:
            RGB, DEPTH, SEG = "ee_rgb", "ee_depth", "ee_seg"
            DMIN, DMAX = "ee_depth_min", "ee_depth_max"
        else:
            RGB, DEPTH, SEG = "rgb", "depth", "seg"
            DMIN, DMAX = "depth_min", "depth_max"
        dmin = h5[DMIN][idx]
        dmax = h5[DMAX][idx]
        rgb1 = img.PNGToNumpy(h5[RGB][idx])[:, :, :3] / 255.  # remove alpha
        depth1 = h5[DEPTH][idx] / 20000. * (dmax - dmin) + dmin
        seg1 = img.PNGToNumpy(h5[SEG][idx])

        valid1 = np.logical_and(depth1 > 0.1, depth1 < 2.)

        # proj_matrix = h5['proj_matrix'][()]
        camera = cam.get_camera_from_h5(h5)
        if self.data_augmentation:
            depth1 = self.add_noise_to_depth(depth1)

        xyz1 = cam.compute_xyz(depth1, camera)
        if self.data_augmentation:
            xyz1 = self.add_noise_to_xyz(xyz1, depth1)

        # Transform the point cloud
        # Here it is...
        # CAM_POSE = "ee_cam_pose" if ee else "cam_pose"
        CAM_POSE = "ee_camera_view" if ee else "camera_view"
        cam_pose = h5[CAM_POSE][idx]
        if ee:
            # ee_camera_view has 0s for x, y, z
            cam_pos = h5["ee_cam_pose"][:][:3, 3]
            cam_pose[:3, 3] = cam_pos

        # Get transformed point cloud
        h, w, d = xyz1.shape
        xyz1 = xyz1.reshape(h * w, -1)
        xyz1 = trimesh.transform_points(xyz1, cam_pose)
        xyz1 = xyz1.reshape(h, w, -1)

        scene1 = rgb1, depth1, seg1, valid1, xyz1

        return scene1

    def __len__(self):
        return len(self.arrangement_data)

    def _get_ids(self, h5):
        """
        get object ids

        @param h5:
        @return:
        """
        ids = {}
        for k in h5.keys():
            if k.startswith("id_"):
                ids[k[3:]] = h5[k][()]
        return ids

    def get_positive_ratio(self):
        num_pos = 0
        for d in self.arrangement_data:
            filename, step_t = d
            if step_t == 0:
                num_pos += 1
        return (len(self.arrangement_data) - num_pos) * 1.0 / num_pos

    def get_object_position_vocab_sizes(self):
        return self.tokenizer.get_object_position_vocab_sizes()

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_data_index(self, idx):
        filename = self.arrangement_data[idx]
        return filename

    def get_raw_data(self, idx, inference_mode=False, shuffle_object_index=False):
        # shuffle_object_index can be used to test different orders of objects

        filename, step_t, perturbation_label = self.arrangement_data[idx]

        h5 = h5py.File(filename, 'r')
        ids = self._get_ids(h5)
        # moved_objs = h5['moved_objs'][()].split(',')
        all_objs = sorted([o for o in ids.keys() if "object_" in o])
        goal_specification = json.loads(str(np.array(h5["goal_specification"])))
        num_rearrange_objs = len(goal_specification["rearrange"]["objects"])
        target_objs = all_objs[:num_rearrange_objs]

        structure_parameters = goal_specification["shape"]
        # Important: ensure the order is correct
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            target_objs = target_objs[::-1]
        elif structure_parameters["type"] == "tower" or structure_parameters["type"] == "dinner":
            target_objs = target_objs
        else:
            raise KeyError("{} structure is not recognized".format(structure_parameters["type"]))

        ###################################
        # getting scene images and point clouds
        scene = self._get_images(h5, step_t, ee=True)
        rgb, depth, seg, valid, xyz = scene
        if inference_mode:
            initial_scene = scene

        # getting object point clouds
        obj_xyzs = []
        obj_rgbs = []
        obj_pad_mask = []
        obj_pc_centers = []
        for obj in target_objs:
            obj_mask = np.logical_and(seg == ids[obj], valid)
            if np.sum(obj_mask) <= 0:
                raise Exception
            ok, obj_xyz, obj_rgb, _ = get_pts(xyz, rgb, obj_mask, num_pts=self.num_pts, to_tensor=False)
            obj_xyzs.append(obj_xyz)
            obj_rgbs.append(obj_rgb)
            obj_pc_centers.append(np.mean(obj_xyz, axis=0))
            obj_pad_mask.append(0)

        ###################################
        # code in this block computes goal positions for objects
        # Important: because of the noises we added to point clouds, the rearranged point clouds will not be perfect
        structure_pose = tra.euler_matrix(structure_parameters["rotation"][0], structure_parameters["rotation"][1],
                                          structure_parameters["rotation"][2])
        structure_pose[:3, 3] = [structure_parameters["position"][0], structure_parameters["position"][1],
                                 structure_parameters["position"][2]]
        structure_pose_inv = np.linalg.inv(structure_pose)

        current_pc_poses = []
        # goal_pc_poses = []
        goal_obj_poses = []
        current_obj_poses = []
        goal_pc_poses_in_struct = []
        for obj, obj_pc_center in zip(target_objs, obj_pc_centers):
            current_pc_pose = np.eye(4)
            current_pc_pose[:3, 3] = obj_pc_center[:3]
            current_pc_poses.append(current_pc_pose)

            goal_pose = h5[obj][0]
            current_pose = h5[obj][step_t]
            goal_obj_poses.append(goal_pose)
            current_obj_poses.append(current_pose)

            # find goal position of the current point cloud center in the structure frame
            goal_pc_pose = goal_pose @ np.linalg.inv(current_pose) @ current_pc_pose
            goal_pc_pose_in_struct = structure_pose_inv @ goal_pc_pose

            # goal_pc_poses.append(goal_pc_pose)
            goal_pc_poses_in_struct.append(goal_pc_pose_in_struct)

        # code below transform current object point cloud to the goal point cloud in the world frame (more efficiently)
        if self.debug:
            goal_obj_xyzs = []
            for i, (obj, obj_xyz) in enumerate(zip(target_objs, obj_xyzs)):
                goal_pc_pose_in_struct = goal_pc_poses_in_struct[i]
                current_pc_pose = current_pc_poses[i]

                goal_pc_transform = structure_pose @ goal_pc_pose_in_struct @ np.linalg.inv(current_pc_pose)
                new_obj_xyz = trimesh.transform_points(obj_xyz, goal_pc_transform)
                goal_obj_xyzs.append(new_obj_xyz)
            show_pcs(goal_obj_xyzs, obj_rgbs, add_coordinate_frame=True)

        ###################################
        # if negative example, perturb object point cloud
        if perturbation_label:
            _, moved_obj_idxs, obj_perturbation_matrices = random_move_obj_xyzs(obj_xyzs,
                                            min_translation=self.min_translation, max_translation=self.max_translation,
                                            min_rotation=self.min_rotation, max_rotation=self.max_rotation,
                                            mode=self.perturbation_mode, return_perturbation=True, return_moved_obj_idxs=True,
                                            return_perturbed_obj_xyzs=False)

            perturbed_goal_pc_poses_in_struct = []
            for goal_pc_poses_in_struct, pm in zip(goal_pc_poses_in_struct, obj_perturbation_matrices):
                perturbed_goal_pc_pose_in_struct = goal_pc_poses_in_struct @ pm
                perturbed_goal_pc_poses_in_struct.append(perturbed_goal_pc_pose_in_struct)
            goal_pc_poses_in_struct = perturbed_goal_pc_poses_in_struct

            if self.debug:
                print("perturbing {} objects".format(len(moved_obj_idxs)))
                perturbed_obj_xyzs = []
                for goal_pc_pose_in_struct, obj_xyz, current_pc_pose in zip(goal_pc_poses_in_struct, obj_xyzs, current_pc_poses):
                    goal_pc_transform = structure_pose @ goal_pc_pose_in_struct @ np.linalg.inv(current_pc_pose)
                    new_obj_xyz = trimesh.transform_points(obj_xyz, goal_pc_transform)
                    perturbed_obj_xyzs.append(new_obj_xyz)
                show_pcs(perturbed_obj_xyzs, obj_rgbs, add_coordinate_frame=True)

        ###################################
        obj_xyzs = [array_to_tensor(x) for x in obj_xyzs]
        obj_rgbs = [array_to_tensor(x) for x in obj_rgbs]

        # pad data after perturbation because we don't want to perturb padding objects
        for i in range(self.max_num_objects - len(target_objs)):
            obj_xyzs.append(torch.zeros([1024, 3], dtype=torch.float32))
            obj_rgbs.append(torch.zeros([1024, 3], dtype=torch.float32))
            obj_pad_mask.append(1)

        ###################################
        # Important: IGNORE key is used to avoid computing loss. -100 is the default ignore_index for NLLLoss and MSE Loss
        obj_xyztheta_inputs = []
        struct_xyztheta_inputs = []

        struct_xyz_theta = structure_pose[:3, 3].tolist() + structure_pose[:3, :3].flatten().tolist()
        struct_xyztheta_inputs.append(struct_xyz_theta)

        # objects that need to be rearranged
        for obj_idx, obj in enumerate(target_objs):
            obj_xyztheta = goal_pc_poses_in_struct[obj_idx][:3, 3].tolist() + goal_pc_poses_in_struct[obj_idx][:3, :3].flatten().tolist()
            obj_xyztheta_inputs.append(obj_xyztheta)

        # paddings
        for i in range(self.max_num_objects - len(target_objs)):
            # xyztheta_inputs.append([-100] * 12)
            obj_xyztheta_inputs.append([0] * 12)

        ###################################
        # preparing sentence
        sentence = []
        sentence_pad_mask = []

        # structure parameters
        # 5 parameters
        structure_parameters = goal_specification["shape"]
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))
            if structure_parameters["type"] == "circle":
                sentence.append((structure_parameters["radius"], "radius"))
            elif structure_parameters["type"] == "line":
                sentence.append((structure_parameters["length"] / 2.0, "radius"))
            for _ in range(5):
                sentence_pad_mask.append(0)
        else:
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))
            for _ in range(4):
                sentence_pad_mask.append(0)
            sentence.append(("PAD", None))
            sentence_pad_mask.append(1)

        ###################################
        # check collision
        has_collision = False
        if self.mesh_scene_viewer:
            if perturbation_label:
                perturbed_goal_pc_poses = []
                for goal_pc_pose_in_struct, pm in zip(goal_pc_poses_in_struct, obj_perturbation_matrices):
                    pgpcp = structure_pose @ goal_pc_pose_in_struct
                    perturbed_goal_pc_poses.append(pgpcp)
                scene = mesh_scene_viewer.load_mesh_scene(target_objs, goal_specification, current_obj_poses,
                                                          current_pc_poses, perturbed_goal_pc_poses, visualize=False)
                has_collision = mesh_scene_viewer.check_scene_collision(scene)

                if self.debug:
                    print("scene has collision", has_collision)
                    pc_vis = [trimesh.PointCloud(obj_xyz) for obj_xyz in perturbed_obj_xyzs]
                    trimesh.Scene([pc_vis, scene]).show()

        ###################################
        if self.debug:
            print("---")
            print("all objects:", all_objs)
            print("target objects:", target_objs)
            # print("goal specification:", goal_specification)
            print("sentence:", sentence)
            print("obj_xyztheta_inputs", obj_xyztheta_inputs)
            print("struct_xyztheta_inputs", struct_xyztheta_inputs)

        assert len(obj_xyzs) == len(obj_xyztheta_inputs)

        ###################################

        # used to indicate whether the token is an object point cloud or a part of the instruction
        # token_type_index = [0] * (self.max_num_shape_parameters) + [1] * (self.max_num_other_objects) + [2] * self.max_num_objects
        # position_index = list(range(self.max_num_shape_parameters)) + list(range(self.max_num_other_objects)) + list(range(self.max_num_objects))
        token_type_index = [0] * (self.max_num_shape_parameters) + [2] * self.max_num_objects
        position_index = list(range(self.max_num_shape_parameters)) + list(range(self.max_num_objects))
        # object_pad_mask

        struct_position_index = [0]
        struct_token_type_index = [3]
        struct_pad_mask = [0]

        # shuffle the position of objects
        if shuffle_object_index:
            shuffle_target_object_indices = list(range(len(target_objs)))
            random.shuffle(shuffle_target_object_indices)
            shuffle_object_indices = shuffle_target_object_indices + list(range(len(target_objs), self.max_num_objects))
            obj_xyzs = [obj_xyzs[i] for i in shuffle_object_indices]
            obj_rgbs = [obj_rgbs[i] for i in shuffle_object_indices]
            obj_xyztheta_inputs = [obj_xyztheta_inputs[i] for i in shuffle_object_indices]
            obj_pad_mask = [obj_pad_mask[i] for i in shuffle_object_indices]
            if inference_mode:
                goal_obj_poses = [goal_obj_poses[i] for i in shuffle_object_indices]
                current_obj_poses = [current_obj_poses[i] for i in shuffle_object_indices]
                target_objs = [target_objs[i] for i in shuffle_target_object_indices]
                current_pc_poses = [current_pc_poses[i] for i in shuffle_object_indices]

        datum = {
            "xyzs": obj_xyzs,
            "rgbs": obj_rgbs,
            "obj_pad_mask": obj_pad_mask,
            "sentence": sentence,
            "sentence_pad_mask": sentence_pad_mask,
            "obj_xyztheta_inputs": obj_xyztheta_inputs,
            "token_type_index": token_type_index,
            "position_index": position_index,
            "struct_xyztheta_inputs": struct_xyztheta_inputs,
            "struct_position_index": struct_position_index,
            "struct_token_type_index": struct_token_type_index,
            "struct_pad_mask": struct_pad_mask,
            "t": step_t,
            "filename": filename,
            "has_collision": int(has_collision),
            "perturbation_label": int(perturbation_label)
        }

        if inference_mode:
            datum["rgb"] = rgb
            datum["goal_obj_poses"] = goal_obj_poses
            datum["current_obj_poses"] = current_obj_poses
            datum["target_objs"] = target_objs
            datum["initial_scene"] = initial_scene
            datum["ids"] = ids
            datum["goal_specification"] = goal_specification
            datum["current_pc_poses"] = current_pc_poses

        return datum

    # def prepare_test_data(self, obj_xyzs, obj_rgbs, other_obj_xyzs, other_obj_rgbs, structure_parameters, initial_scene=None, ids=None):
    #
    #     object_pad_mask = []
    #     other_object_pad_mask = []
    #     for obj in obj_xyzs:
    #         object_pad_mask.append(0)
    #     for obj in other_obj_xyzs:
    #         other_object_pad_mask.append(0)
    #     for i in range(self.max_num_objects - len(obj_xyzs)):
    #         obj_xyzs.append(torch.zeros([1024, 3], dtype=torch.float32))
    #         obj_rgbs.append(torch.zeros([1024, 3], dtype=torch.float32))
    #         object_pad_mask.append(1)
    #     for i in range(self.max_num_other_objects - len(other_obj_xyzs)):
    #         other_obj_xyzs.append(torch.zeros([1024, 3], dtype=torch.float32))
    #         other_obj_rgbs.append(torch.zeros([1024, 3], dtype=torch.float32))
    #         other_object_pad_mask.append(1)
    #
    #     # language instruction
    #     sentence = []
    #     sentence_pad_mask = []
    #     # structure parameters
    #     # 5 parameters
    #     if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
    #         sentence.append((structure_parameters["type"], "shape"))
    #         sentence.append((structure_parameters["rotation"][2], "rotation"))
    #         sentence.append((structure_parameters["position"][0], "position_x"))
    #         sentence.append((structure_parameters["position"][1], "position_y"))
    #         if structure_parameters["type"] == "circle":
    #             sentence.append((structure_parameters["radius"], "radius"))
    #         elif structure_parameters["type"] == "line":
    #             sentence.append((structure_parameters["length"] / 2.0, "radius"))
    #         for _ in range(5):
    #             sentence_pad_mask.append(0)
    #     else:
    #         sentence.append((structure_parameters["type"], "shape"))
    #         sentence.append((structure_parameters["rotation"][2], "rotation"))
    #         sentence.append((structure_parameters["position"][0], "position_x"))
    #         sentence.append((structure_parameters["position"][1], "position_y"))
    #         for _ in range(4):
    #             sentence_pad_mask.append(0)
    #         sentence.append(("PAD", None))
    #         sentence_pad_mask.append(1)
    #
    #     # placeholder for pose predictions
    #     obj_x_outputs = [0] * self.max_num_objects
    #     obj_y_outputs = [0] * self.max_num_objects
    #     obj_z_outputs = [0] * self.max_num_objects
    #     obj_theta_outputs = [[0] * 9] * self.max_num_objects
    #     obj_x_inputs = [0] * self.max_num_objects
    #     obj_y_inputs = [0] * self.max_num_objects
    #     obj_z_inputs = [0] * self.max_num_objects
    #     obj_theta_inputs = [[0] * 9] * self.max_num_objects
    #     struct_x_inputs = [0]
    #     struct_y_inputs = [0]
    #     struct_z_inputs = [0]
    #     struct_theta_inputs = [[0] * 9]
    #
    #     # used to indicate whether the token is an object point cloud or a part of the instruction
    #     token_type_index = [0] * (self.max_num_shape_parameters) + [1] * (self.max_num_other_objects) + [2] * self.max_num_objects
    #     position_index = list(range(self.max_num_shape_parameters)) + list(range(self.max_num_other_objects)) + list(range(self.max_num_objects))
    #     struct_position_index = [0]
    #     struct_token_type_index = [3]
    #     struct_pad_mask = [0]
    #
    #     datum = {
    #         "xyzs": obj_xyzs,
    #         "rgbs": obj_rgbs,
    #         "object_pad_mask": object_pad_mask,
    #         "other_xyzs": other_obj_xyzs,
    #         "other_rgbs": other_obj_rgbs,
    #         "other_object_pad_mask": other_object_pad_mask,
    #         "sentence": sentence,
    #         "sentence_pad_mask": sentence_pad_mask,
    #         "token_type_index": token_type_index,
    #         "obj_x_outputs": obj_x_outputs,
    #         "obj_y_outputs": obj_y_outputs,
    #         "obj_z_outputs": obj_z_outputs,
    #         "obj_theta_outputs": obj_theta_outputs,
    #         "obj_x_inputs": obj_x_inputs,
    #         "obj_y_inputs": obj_y_inputs,
    #         "obj_z_inputs": obj_z_inputs,
    #         "obj_theta_inputs": obj_theta_inputs,
    #         "position_index": position_index,
    #         "struct_position_index": struct_position_index,
    #         "struct_token_type_index": struct_token_type_index,
    #         "struct_pad_mask": struct_pad_mask,
    #         "struct_x_inputs": struct_x_inputs,
    #         "struct_y_inputs": struct_y_inputs,
    #         "struct_z_inputs": struct_z_inputs,
    #         "struct_theta_inputs": struct_theta_inputs,
    #         "t": 0,
    #         "filename": ""
    #     }
    #
    #     if initial_scene:
    #         datum["initial_scene"] = initial_scene
    #     if ids:
    #         datum["ids"] = ids
    #
    #     return datum

    @staticmethod
    def convert_to_tensors(datum, tokenizer, robot_mode=False):

        if robot_mode:
            tensors = {
                "xyzs": torch.stack(datum["xyzs"], dim=0),
                "obj_pad_mask": torch.LongTensor(datum["obj_pad_mask"]),
                "sentence": torch.LongTensor([tokenizer.tokenize(*i) for i in datum["sentence"]]),
                "sentence_pad_mask": torch.LongTensor(datum["sentence_pad_mask"]),
                "token_type_index": torch.LongTensor(datum["token_type_index"]),
                "position_index": torch.LongTensor(datum["position_index"]),
                "struct_position_index": torch.LongTensor(datum["struct_position_index"]),
                "struct_token_type_index": torch.LongTensor(datum["struct_token_type_index"]),
                "struct_pad_mask": torch.LongTensor(datum["struct_pad_mask"]),
                "obj_xyztheta_inputs": torch.FloatTensor(datum["obj_xyztheta_inputs"]),
                "struct_xyztheta_inputs": torch.FloatTensor(datum["struct_xyztheta_inputs"]),
            }
        else:
            tensors = {
                "xyzs": torch.stack(datum["xyzs"], dim=0),
                "rgbs": torch.stack(datum["rgbs"], dim=0),
                "obj_pad_mask": torch.LongTensor(datum["obj_pad_mask"]),
                "sentence": torch.LongTensor([tokenizer.tokenize(*i) for i in datum["sentence"]]),
                "sentence_pad_mask": torch.LongTensor(datum["sentence_pad_mask"]),
                "token_type_index": torch.LongTensor(datum["token_type_index"]),
                "position_index": torch.LongTensor(datum["position_index"]),
                "struct_position_index": torch.LongTensor(datum["struct_position_index"]),
                "struct_token_type_index": torch.LongTensor(datum["struct_token_type_index"]),
                "struct_pad_mask": torch.LongTensor(datum["struct_pad_mask"]),
                "obj_xyztheta_inputs": torch.FloatTensor(datum["obj_xyztheta_inputs"]),
                "struct_xyztheta_inputs": torch.FloatTensor(datum["struct_xyztheta_inputs"]),
                "t": datum["t"],
                "filename": datum["filename"],
                "has_collision": torch.LongTensor([datum["has_collision"]]),
                "perturbation_label": torch.LongTensor([datum["perturbation_label"]]),
            }

        # for k in tensors:
        #     if isinstance(tensors[k], torch.Tensor):
        #         print("--size", k, tensors[k].shape)

        return tensors

    def __getitem__(self, idx):

        datum = self.convert_to_tensors(self.get_raw_data(idx, shuffle_object_index=self.shuffle_object_index),
                                        self.tokenizer)

        return datum

    # # @staticmethod
    # def collate_fn(data):
    #     """
    #     :param data:
    #     :return:
    #     """
    #
    #     batched_data_dict = {}
    #     # for key in ["xyzs", "rgbs", "other_xyzs", "other_rgbs"]:
    #     #     batched_data_dict[key] = torch.cat([dict[key] for dict in data], dim=0)
    #     for key in ["xyzs", "rgbs", "other_xyzs", "other_rgbs",
    #                 "obj_pad_mask", "other_obj_pad_mask", "sentence", "sentence_pad_mask", "token_type_index",
    #                 "obj_xyztheta_inputs", "position_index",
    #                 "struct_position_index", "struct_token_type_index", "struct_pad_mask",
    #                 "struct_xyztheta_inputs"]:
    #         batched_data_dict[key] = torch.stack([dict[key] for dict in data], dim=0)
    #
    #     return batched_data_dict


def compute_min_max(dataloader):

    # tensor([-0.3557, -0.3847,  0.0000, -1.0000, -1.0000, -0.4759, -1.0000, -1.0000,
    #         -0.9079, -0.8668, -0.9105, -0.4186])
    # tensor([0.3915, 0.3494, 0.3267, 1.0000, 1.0000, 0.8961, 1.0000, 1.0000, 0.8194,
    #         0.4787, 0.6421, 1.0000])
    # tensor([0.0918, -0.3758, 0.0000, -1.0000, -1.0000, 0.0000, -1.0000, -1.0000,
    #         -0.0000, 0.0000, 0.0000, 1.0000])
    # tensor([0.9199, 0.3710, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 1.0000, -0.0000,
    #         0.0000, 0.0000, 1.0000])

    obj_min = torch.ones(12) * 10000
    obj_max = torch.ones(12) * -10000
    for d in tqdm(dataloader):
        obj_xyz_theta = d["obj_xyztheta_inputs"]
        obj_xyz_theta = obj_xyz_theta.reshape(-1, 12)
        current_max, _ = torch.max(obj_xyz_theta, dim=0)
        current_min, _ = torch.min(obj_xyz_theta, dim=0)
        obj_max[obj_max < current_max] = current_max[obj_max < current_max]
        obj_min[obj_min > current_min] = current_min[obj_min > current_min]
    print(obj_min)
    print(obj_max)

    struct_min = torch.ones(12) * 10000
    struct_max = torch.ones(12) * -10000
    for d in tqdm(dataloader):
        struct_xyz_theta = d["struct_xyztheta_inputs"]
        struct_xyz_theta = struct_xyz_theta.reshape(-1, 12)
        current_max, _ = torch.max(struct_xyz_theta, dim=0)
        current_min, _ = torch.min(struct_xyz_theta, dim=0)
        struct_max[struct_max < current_max] = current_max[struct_max < current_max]
        struct_min[struct_min > current_min] = current_min[struct_min > current_min]
    print(struct_min)
    print(struct_max)


if __name__ == "__main__":

    tokenizer = Tokenizer("/home/weiyu/data_drive/data_new_objects/type_vocabs_coarse.json")

    data_roots = []
    index_roots = []
    for shape, index in [("dinner", "index_10k")]: # [("circle", "index_34k"), ("line", "index_42k"), ("tower", "index_13k"), ("dinner", "index_24k")]:
        data_roots.append("/home/weiyu/data_drive/data_new_objects/examples_{}_new_objects/result".format(shape))
        index_roots.append(index)

    mesh_scene_viewer = MeshSceneViewer(assets_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large", cache_mesh=True)

    dataset = SemanticArrangementDataset(data_roots=data_roots,
                                         index_roots=index_roots,
                                         split="train", tokenizer=tokenizer,
                                         min_translation=0.01, max_translation=0.08, min_rotation=0.1, max_rotation=0.5, perturbation_mode="6d",
                                         num_random_perturbations_per_positive=10, oversample_positive=False,
                                         max_num_objects=7,
                                         max_num_shape_parameters=5,
                                         num_pts=1024,
                                         filter_num_moved_objects_range=None,  # [5, 5]
                                         data_augmentation=False,
                                         shuffle_object_index=False,
                                         debug=False, mesh_scene_viewer=mesh_scene_viewer)

    # print(len(dataset))
    # for d in dataset:
    #     print("\n\n" + "="*100)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False, num_workers=8)
    for i, d in enumerate(dataloader):
        print(i)
        # for k in d:
        #     if isinstance(d[k], torch.Tensor):
        #         print("--size", k, d[k].shape)
        # for k in d:
        #     print(k, d[k])
        #
        # input("next?")

    # tokenizer = Tokenizer("/home/weiyu/data_drive/data_new_objects/type_vocabs.json")
    # for shape, index in [("circle", "index_34k"), ("line", "index_42k"), ("tower", "index_13k"), ("dinner", "index_24k")]:
    #     for split in ["train", "valid", "test"]:
    #         dataset = SemanticArrangementDataset(data_root="/home/weiyu/data_drive/data_new_objects/examples_{}_new_objects/result".format(shape),
    #                                              index_root=index,
    #                                              split=split, tokenizer=tokenizer,
    #                                              max_num_objects=7,
    #                                              max_num_other_objects=5,
    #                                              max_num_shape_parameters=5,
    #                                              max_num_rearrange_features=0,
    #                                              max_num_anchor_features=0,
    #                                              num_pts=1024,
    #                                              debug=True)
    #
    #         for i in range(0, 1):
    #             d = dataset.get_raw_data(i)
    #             d = dataset.convert_to_tensors(d, dataset.tokenizer)
    #             for k in d:
    #                 if torch.is_tensor(d[k]):
    #                     print("--size", k, d[k].shape)
    #             for k in d:
    #                 print(k, d[k])
    #             input("next?")

            # dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8,
            #                         collate_fn=SemanticArrangementDataset.collate_fn)
            # for d in tqdm(dataloader):
            #     pass