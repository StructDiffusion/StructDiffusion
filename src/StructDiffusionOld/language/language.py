test_parrot_paraphrase.py# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import argparse
import copy
import csv
import cv2
import h5py
import numpy as np
import open3d
import os
import PIL
import scipy
import scipy.io
import sys
import trimesh

import torch
import pytorch3d.transforms as tra3d
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from itertools import combinations

# Local imports
from predictor_v2 import show_pcs, get_pts
from tokenizer import Tokenizer

from brain2.utils.info import logwarn
import brain2.utils.image as img
import brain2.utils.transformations as tra
import brain2.utils.camera as cam

"""
This datset is used for predicting location of objects autoregressively. 
This dataset provides objects that do not need to be rearranged.
"""

class SemanticArrangementDataset(torch.utils.data.Dataset):

    def __init__(self, data_roots, index_roots, split, tokenizer,
                 max_num_objects=11, max_num_other_objects=5,
                 max_num_shape_parameters=7, max_num_rearrange_features=1, max_num_anchor_features=3,
                 num_pts=1024, filter_num_moved_objects_range=None, debug=False, shuffle_object_index=False,
                 data_augmentation=True, **kwargs):
        """

        :param data_root:
        :param split: train, valid, or test
        :param shuffle_object_index: whether to shuffle the positions of target objects and other objects in the sequence
        :param debug:
        :param max_num_shape_parameters:
        :param max_num_objects:
        :param max_num_rearrange_features:
        :param max_num_anchor_features:
        :param num_pts:
        :param use_stored_arrangement_indices:
        :param kwargs:
        """

        print("data augmentation is set to ", data_augmentation)

        self.data_roots = data_roots
        self.num_pts = num_pts
        print("data dirs:", self.data_roots)
        self.debug = debug

        self.max_num_objects = max_num_objects
        self.max_num_other_objects = max_num_other_objects
        self.max_num_shape_parameters = max_num_shape_parameters
        self.max_num_rearrange_features = max_num_rearrange_features
        self.max_num_anchor_features = max_num_anchor_features
        self.shuffle_object_index = shuffle_object_index

        self.tokenizer = tokenizer

        self.arrangement_data = []
        for data_root, index_root in zip(data_roots, index_roots):
            arrangement_indices_file = os.path.join(data_root, index_root, "{}_sequence_indices_file.txt".format(split))
            if os.path.exists(arrangement_indices_file):
                with open(arrangement_indices_file, "r") as fh:
                    self.arrangement_data.extend([os.path.join(data_root, f) for f in eval(fh.readline().strip())])
            else:
                # raise FileNotFoundError("{} does not exist".format(arrangement_indices_file))
                print("{} does not exist".format(arrangement_indices_file))

        # # remove rearranged scenes
        # arrangement_data = []
        # for d in self.arrangement_data:
        #     if d[1] != 0:
        #         arrangement_data.append(d)
        # self.arrangement_data = arrangement_data

        if filter_num_moved_objects_range is not None:
            self.arrangement_data = self.filter_based_on_number_of_moved_objects(filter_num_moved_objects_range)
        print("{} valid sequences".format(len(self.arrangement_data)))

        # Are we going to learn how to rotate things?
        self.use_rotations = True

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
        for filename in self.arrangement_data:
            h5 = h5py.File(filename, 'r')
            moved_objs = h5['moved_objs'][()].split(',')
            if min_num <= len(moved_objs) <= max_num:
                ok_data.append(filename)
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

    def get_template_language(self, idx):
        filename = self.arrangement_data[idx]

        h5 = h5py.File(filename, 'r')
        goal_specification = json.loads(str(np.array(h5["goal_specification"])))

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
        else:
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))

        # Put the [objects] in a [size][shape] on the [x][y] of the table facing [rotation].
        # Build a [size][shape] of the [objects] on the [x][y] of the table facing [rotation].
        # Put the [objects] on the [x][y] of the table and make a [shape] facing [rotation].
        # Rearrange the [objects] into a [shape], and put the structure on the [x][y] of the table facing [rotation].
        # Could you ...
        # Please ...
        # Pick up the objects, put them into a [size][shape], place the [shape] on the [x][y] of table, make sure the [shape] is facing [rotation].

        return sentence


    def get_raw_data(self, idx, inference_mode=False, shuffle_object_index=False):
        # shuffle_object_index can be used to test different orders of objects

        filename = self.arrangement_data[idx]

        h5 = h5py.File(filename, 'r')
        ids = self._get_ids(h5)
        # moved_objs = h5['moved_objs'][()].split(',')
        all_objs = sorted([o for o in ids.keys() if "object_" in o])
        goal_specification = json.loads(str(np.array(h5["goal_specification"])))
        num_rearrange_objs = len(goal_specification["rearrange"]["objects"])
        num_other_objs = len(goal_specification["anchor"]["objects"] + goal_specification["distract"]["objects"])

        assert len(all_objs) == num_rearrange_objs + num_other_objs, "{}, {}".format(len(all_objs), num_rearrange_objs + num_other_objs)
        assert num_rearrange_objs <= self.max_num_objects
        assert num_other_objs <= self.max_num_other_objects

        step_t = num_rearrange_objs

        target_objs = all_objs[:num_rearrange_objs]
        other_objs = all_objs[num_rearrange_objs:]

        structure_parameters = goal_specification["shape"]

        # Important: ensure the order is correct
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            target_objs = target_objs[::-1]
        elif structure_parameters["type"] == "tower" or structure_parameters["type"] == "dinner":
            target_objs = target_objs
        else:
            raise KeyError("{} structure is not recognized".format(structure_parameters["type"]))
        all_objs = target_objs + other_objs

        # Important: temporary fix for a bug in data generation for tower structure
        if "rotation" not in structure_parameters and "position" not in structure_parameters:
            # get bottom object rotation and position
            bottom_obj = 'object_00'
            bottom_obj_pose = h5[bottom_obj][0]
            bottom_obj_rotation = tra.euler_from_matrix(bottom_obj_pose)
            structure_parameters["rotation"] = bottom_obj_rotation
            structure_parameters["position"] = bottom_obj_pose[:3, 3]
            structure_parameters["position"][2] = 0

        ###################################
        # getting scene images and point clouds
        scene = self._get_images(h5, step_t, ee=True)
        rgb, depth, seg, valid, xyz = scene
        if inference_mode:
            initial_scene = scene

        # getting object point clouds
        obj_xyzs = []
        obj_rgbs = []
        object_pad_mask = []
        obj_pc_centers = []
        other_obj_xyzs = []
        other_obj_rgbs = []
        other_object_pad_mask = []
        for obj in all_objs:
            obj_mask = np.logical_and(seg == ids[obj], valid)
            if np.sum(obj_mask) <= 0:
                raise Exception
            ok, obj_xyz, obj_rgb, _ = get_pts(xyz, rgb, obj_mask, num_pts=self.num_pts)
            if not ok:
                raise Exception

            if obj in target_objs:
                obj_xyzs.append(obj_xyz)
                obj_rgbs.append(obj_rgb)
                object_pad_mask.append(0)
                obj_pc_centers.append(torch.mean(obj_xyz, dim=0).numpy())
            elif obj in other_objs:
                other_obj_xyzs.append(obj_xyz)
                other_obj_rgbs.append(obj_rgb)
                other_object_pad_mask.append(0)
            else:
                raise Exception

        if inference_mode:
            goal_scene = self._get_images(h5, 0, ee=True)
            goal_rgb, goal_depth, goal_seg, goal_valid, goal_xyz = goal_scene

        #---------------------------------
        # code below moves the goal point cloud to the structure frame
        # goal_scene = self._get_images(h5, 0, ee=True)
        # goal_rgb, goal_depth, goal_seg, goal_valid, goal_xyz = goal_scene
        #
        # goal_obj_xyzs = []
        # goal_obj_rgbs = []
        # for obj in target_objs:
        #     goal_obj_mask = np.logical_and(goal_seg == ids[obj], goal_valid)
        #     if np.sum(goal_obj_mask) <= 0:
        #         raise Exception
        #     ok, goal_obj_xyz, goal_obj_rgb, _ = get_pts(goal_xyz, goal_rgb, goal_obj_mask, num_pts=self.num_pts)
        #     goal_obj_xyzs.append(goal_obj_xyz)
        #     goal_obj_rgbs.append(goal_obj_rgb)
        #
        # new_obj_xyzs = []
        # for obj_xyz in goal_obj_xyzs:
        #     # new_obj_xyz = obj_xyz - torch.tensor([structure_parameters["position"][0], structure_parameters["position"][1], 0])
        #     # rotation = tra.euler_matrix(0, 0, -structure_parameters["rotation"][2])
        #     T = tra.euler_matrix(structure_parameters["rotation"][0], structure_parameters["rotation"][1],
        #                          structure_parameters["rotation"][2])
        #     T[:3, 3] = [structure_parameters["position"][0], structure_parameters["position"][1],
        #                 structure_parameters["position"][2]]
        #     new_obj_xyz = trimesh.transform_points(obj_xyz, np.linalg.inv(T))
        #     # print(obj_xyz.shape)
        #     # print(torch.mean(obj_xyz, dim=0).unsqueeze(0))
        #     # print(trimesh.transform_points(torch.mean(obj_xyz, dim=0).unsqueeze(0), np.linalg.inv(T)))
        #     new_obj_xyz = torch.tensor(new_obj_xyz, dtype=obj_xyz.dtype)
        #     # print(torch.mean(new_obj_xyz, dim=0))
        #     new_obj_xyzs.append(new_obj_xyz)
        #     print("goal pc center:", torch.mean(new_obj_xyz, dim=0))
        # goal_obj_xyzs = new_obj_xyzs
        # show_pcs(goal_obj_xyzs, goal_obj_rgbs, add_coordinate_frame=True)
        #-----------------------------------
        # code in this block computes goal positions for objects
        # Important: because of the noises we added to point clouds, the rearranged point clouds will not be perfect
        structure_pose = tra.euler_matrix(structure_parameters["rotation"][0], structure_parameters["rotation"][1],
                                          structure_parameters["rotation"][2])
        structure_pose[:3, 3] = [structure_parameters["position"][0], structure_parameters["position"][1],
                                 structure_parameters["position"][2]]
        structure_pose_inv = np.linalg.inv(structure_pose)

        goal_obj_pc_centers = []
        obj_pc_rotations = []
        for obj, obj_pc_center in zip(target_objs, obj_pc_centers):
            current_pc_pose = np.eye(4)
            current_pc_pose[:3, 3] = obj_pc_center[:3]

            goal_pose = h5[obj][0]
            current_pose = h5[obj][step_t]

            # find goal position of the current point cloud center in the structure frame
            goal_pc_pose = structure_pose_inv @ goal_pose @ np.linalg.inv(current_pose) @ current_pc_pose

            goal_obj_center = goal_pc_pose[:3, 3]
            goal_obj_pc_centers.append(goal_obj_center)

            R = np.eye(4)
            R[:3, :3] = goal_pc_pose[:3, :3]
            obj_pc_rotations.append(R)
        # -----------------------------------
        # code below transform current object point cloud to the goal point cloud in the structure frame
        # if self.debug:
        #     new_obj_xyzs = []
        #     for i, obj_xyz in enumerate(obj_xyzs):
        #
        #         # translating to the goal point cloud center
        #         t = np.eye(4)
        #         t[:3, 3] = [goal_obj_pc_centers[i][0] - obj_pc_centers[i][0], goal_obj_pc_centers[i][1] - obj_pc_centers[i][1], goal_obj_pc_centers[i][2] - obj_pc_centers[i][2]]
        #         new_obj_xyz = trimesh.transform_points(obj_xyz, t)
        #
        #         # rotating in place
        #         # R = tra.euler_matrix(0, 0, obj_pc_rotations[i])
        #         R = obj_pc_rotations[i]
        #         obj_center = np.mean(new_obj_xyz, axis=0)
        #         centered_obj_xyz = new_obj_xyz - obj_center
        #         new_centered_obj_xyz = trimesh.transform_points(centered_obj_xyz, R, translate=True)
        #         new_obj_xyz = new_centered_obj_xyz + obj_center
        #         new_obj_xyz = torch.tensor(new_obj_xyz, dtype=obj_xyz.dtype)
        #         new_obj_xyzs.append(new_obj_xyz)
        #         # print(obj_pc_rotations[i], np.rad2deg(obj_pc_rotations[i]))
        #         # show_pcs([new_obj_xyz, obj_xyz, goal_obj_xyzs[i]], [obj_rgbs[i], obj_rgbs[i], goal_obj_rgbs[i]], add_coordinate_frame=True)
        #
        #     show_pcs(new_obj_xyzs, obj_rgbs, add_coordinate_frame=True)
        #-----------------------------------
        # code below transform current object point cloud to the goal point cloud in the world frame
        if self.debug:
            new_obj_xyzs = copy.deepcopy(obj_xyzs)
            new_obj_rgbs = copy.deepcopy(obj_rgbs)
            for i, obj_xyz in enumerate(obj_xyzs):

                # translating to the goal point cloud center
                t = np.eye(4)
                t[:3, 3] = [goal_obj_pc_centers[i][0] - obj_pc_centers[i][0],
                            goal_obj_pc_centers[i][1] - obj_pc_centers[i][1],
                            goal_obj_pc_centers[i][2] - obj_pc_centers[i][2]]
                new_obj_xyz = trimesh.transform_points(obj_xyz, t)

                # rotating in place
                # R = tra.euler_matrix(0, 0, obj_pc_rotations[i])
                R = obj_pc_rotations[i]
                obj_center = np.mean(new_obj_xyz, axis=0)
                centered_obj_xyz = new_obj_xyz - obj_center
                new_centered_obj_xyz = trimesh.transform_points(centered_obj_xyz, R, translate=True)
                new_obj_xyz = new_centered_obj_xyz + obj_center

                new_obj_xyz = trimesh.transform_points(new_obj_xyz, structure_pose)

                new_obj_xyz = torch.tensor(new_obj_xyz, dtype=obj_xyz.dtype)
                new_obj_xyzs[i] = new_obj_xyz
                # print(obj_pc_rotations[i], np.rad2deg(obj_pc_rotations[i]))
                # show_pcs([new_obj_xyz, obj_xyz, goal_obj_xyzs[i]], [obj_rgbs[i], obj_rgbs[i], goal_obj_rgbs[i]], add_coordinate_frame=True)

                new_obj_rgbs[i] = np.tile(np.array([1, 0, 0], dtype=np.float), (new_obj_xyz.shape[0], 1))

                show_pcs(new_obj_xyzs + other_obj_xyzs, new_obj_rgbs + other_obj_rgbs, add_coordinate_frame=True)
        # -----------------------------------

        # pad data
        for i in range(self.max_num_objects - len(target_objs)):
            obj_xyzs.append(torch.zeros([1024, 3], dtype=torch.float32))
            obj_rgbs.append(torch.zeros([1024, 3], dtype=torch.float32))
            object_pad_mask.append(1)
        for i in range(self.max_num_other_objects - len(other_objs)):
            other_obj_xyzs.append(torch.zeros([1024, 3], dtype=torch.float32))
            other_obj_rgbs.append(torch.zeros([1024, 3], dtype=torch.float32))
            other_object_pad_mask.append(1)

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
        else:
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))

        # make a copy of the complete sentence
        complete_sentence = copy.deepcopy(sentence)

        random.shuffle(sentence)
        sentence = sentence[:np.random.randint(1, len(sentence) + 1)]
        for _ in range(len(sentence)):
            sentence_pad_mask.append(0)
        for _ in range(self.max_num_shape_parameters - len(sentence)):
            sentence.append(("PAD", None))
            sentence_pad_mask.append(1)

        ###################################
        # Important: IGNORE key is used to avoid computing loss. -100 is the default ignore_index for NLLLoss and MSE Loss
        obj_xytheta_outputs = []
        obj_xytheta_inputs = []

        # add structure prediction
        struct_xytheta_outputs = []
        struct_xytheta_inputs = []
        x = structure_parameters["position"][0]
        y = structure_parameters["position"][1]
        z = structure_parameters["position"][2]
        theta = tra.euler_matrix(structure_parameters["rotation"][0], structure_parameters["rotation"][1],
                                 structure_parameters["rotation"][2])[:3, :3].flatten().tolist()
        struct_xytheta_outputs.append([x, y, z] + theta)
        struct_xytheta_inputs.append([x, y, z] + theta)

        # objects that need to be rearranged
        goal_obj_poses = []
        current_obj_poses = []
        for obj_idx, obj in enumerate(target_objs):
            # use the values we computed above
            theta = obj_pc_rotations[obj_idx][:3, :3].flatten().tolist()
            x = goal_obj_pc_centers[obj_idx][0]
            y = goal_obj_pc_centers[obj_idx][1]
            z = goal_obj_pc_centers[obj_idx][2]

            obj_xytheta_outputs.append([x, y, z] + theta)
            obj_xytheta_inputs.append([x, y, z] + theta)

            if inference_mode:
                goal_obj_poses.append(h5[obj][0])
                current_obj_poses.append(h5[obj][step_t])

        # paddings
        for i in range(self.max_num_objects - len(target_objs)):
            obj_xytheta_outputs.append([-100] * 12)
            obj_xytheta_inputs.append([0] * 12)

            if inference_mode:
                goal_obj_poses.append(None)
                current_obj_poses.append(None)

        ###################################
        if self.debug:
            print("---")
            print("all objects:", all_objs)
            print("target objects:", target_objs)
            print("other objects:", other_objs)
            print(goal_specification)
            print("complete sentence", complete_sentence)
            print("sentence:", sentence)
            print("obj_xyztheta_inputs", obj_xytheta_inputs)
            print("obj_xytheta_outputs", obj_xytheta_outputs)
            print("struct_xyztheta_inputs", struct_xytheta_inputs)
            print("struct_xyztheta_outputs", struct_xytheta_outputs)
            # plt.figure()
            # plt.imshow(rgb)
            # plt.show()
            #
            # init_scene = self._get_images(h5, 0, ee=True)
            # plt.figure()
            # plt.imshow(init_scene[0])
            # plt.show()
            show_pcs(obj_xyzs + other_obj_xyzs, obj_rgbs + other_obj_rgbs, add_coordinate_frame=True)

        assert len(obj_xyzs) == len(obj_xytheta_outputs)
        ###################################

        # used to indicate whether the token is an object point cloud or a part of the instruction
        token_type_index = [0] * (self.max_num_shape_parameters) + [1] * (self.max_num_other_objects) + [2] * self.max_num_objects
        position_index = list(range(self.max_num_shape_parameters)) + list(range(self.max_num_other_objects)) + list(range(self.max_num_objects))

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
            obj_xytheta_outputs = [obj_xytheta_outputs[i] for i in shuffle_object_indices]
            obj_xytheta_inputs = [obj_xytheta_inputs[i] for i in shuffle_object_indices]
            object_pad_mask = [object_pad_mask[i] for i in shuffle_object_indices]
            if inference_mode:
                goal_obj_poses = [goal_obj_poses[i] for i in shuffle_object_indices]
                current_obj_poses = [current_obj_poses[i] for i in shuffle_object_indices]
                target_objs = [target_objs[i] for i in shuffle_target_object_indices]

        # convert to torch data
        obj_x_outputs = [i[0] for i in obj_xytheta_outputs]
        obj_y_outputs = [i[1] for i in obj_xytheta_outputs]
        obj_z_outputs = [i[2] for i in obj_xytheta_outputs]
        obj_theta_outputs = [i[3:] for i in obj_xytheta_outputs]
        obj_x_inputs = [i[0] for i in obj_xytheta_inputs]
        obj_y_inputs = [i[1] for i in obj_xytheta_inputs]
        obj_z_inputs = [i[2] for i in obj_xytheta_inputs]
        obj_theta_inputs = [i[3:] for i in obj_xytheta_inputs]
        struct_x_inputs = [i[0] for i in struct_xytheta_inputs]
        struct_y_inputs = [i[1] for i in struct_xytheta_inputs]
        struct_z_inputs = [i[2] for i in struct_xytheta_inputs]
        struct_theta_inputs = [i[3:] for i in struct_xytheta_inputs]

        datum = {
            "xyzs": obj_xyzs,
            "rgbs": obj_rgbs,
            "object_pad_mask": object_pad_mask,
            "other_xyzs": other_obj_xyzs,
            "other_rgbs": other_obj_rgbs,
            "other_object_pad_mask": other_object_pad_mask,
            "sentence": sentence,
            "sentence_pad_mask": sentence_pad_mask,
            "token_type_index": token_type_index,
            "obj_x_outputs": obj_x_outputs,
            "obj_y_outputs": obj_y_outputs,
            "obj_z_outputs": obj_z_outputs,
            "obj_theta_outputs": obj_theta_outputs,
            "obj_x_inputs": obj_x_inputs,
            "obj_y_inputs": obj_y_inputs,
            "obj_z_inputs": obj_z_inputs,
            "obj_theta_inputs": obj_theta_inputs,
            "position_index": position_index,
            "struct_position_index": struct_position_index,
            "struct_token_type_index": struct_token_type_index,
            "struct_pad_mask": struct_pad_mask,
            "struct_x_inputs": struct_x_inputs,
            "struct_y_inputs": struct_y_inputs,
            "struct_z_inputs": struct_z_inputs,
            "struct_theta_inputs": struct_theta_inputs,
            "t": step_t,
            "filename": filename,
            "complete_sentence": complete_sentence
        }

        if inference_mode:
            datum["rgb"] = rgb
            datum["goal_rgb"] = goal_rgb
            datum["goal_obj_poses"] = goal_obj_poses
            datum["current_obj_poses"] = current_obj_poses
            datum["target_objs"] = target_objs
            datum["initial_scene"] = initial_scene
            datum["ids"] = ids

        return datum

    def prepare_test_data(self, obj_xyzs, obj_rgbs, other_obj_xyzs, other_obj_rgbs, structure_parameters, initial_scene=None, ids=None):

        object_pad_mask = []
        other_object_pad_mask = []
        for obj in obj_xyzs:
            object_pad_mask.append(0)
        for obj in other_obj_xyzs:
            other_object_pad_mask.append(0)
        for i in range(self.max_num_objects - len(obj_xyzs)):
            obj_xyzs.append(torch.zeros([1024, 3], dtype=torch.float32))
            obj_rgbs.append(torch.zeros([1024, 3], dtype=torch.float32))
            object_pad_mask.append(1)
        for i in range(self.max_num_other_objects - len(other_obj_xyzs)):
            other_obj_xyzs.append(torch.zeros([1024, 3], dtype=torch.float32))
            other_obj_rgbs.append(torch.zeros([1024, 3], dtype=torch.float32))
            other_object_pad_mask.append(1)

        # language instruction
        sentence = []
        sentence_pad_mask = []
        # structure parameters
        # 5 parameters
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

        # placeholder for pose predictions
        obj_x_outputs = [0] * self.max_num_objects
        obj_y_outputs = [0] * self.max_num_objects
        obj_z_outputs = [0] * self.max_num_objects
        obj_theta_outputs = [[0] * 9] * self.max_num_objects
        obj_x_inputs = [0] * self.max_num_objects
        obj_y_inputs = [0] * self.max_num_objects
        obj_z_inputs = [0] * self.max_num_objects
        obj_theta_inputs = [[0] * 9] * self.max_num_objects
        struct_x_inputs = [0]
        struct_y_inputs = [0]
        struct_z_inputs = [0]
        struct_theta_inputs = [[0] * 9]

        # used to indicate whether the token is an object point cloud or a part of the instruction
        token_type_index = [0] * (self.max_num_shape_parameters) + [1] * (self.max_num_other_objects) + [2] * self.max_num_objects
        position_index = list(range(self.max_num_shape_parameters)) + list(range(self.max_num_other_objects)) + list(range(self.max_num_objects))
        struct_position_index = [0]
        struct_token_type_index = [3]
        struct_pad_mask = [0]

        datum = {
            "xyzs": obj_xyzs,
            "rgbs": obj_rgbs,
            "object_pad_mask": object_pad_mask,
            "other_xyzs": other_obj_xyzs,
            "other_rgbs": other_obj_rgbs,
            "other_object_pad_mask": other_object_pad_mask,
            "sentence": sentence,
            "sentence_pad_mask": sentence_pad_mask,
            "token_type_index": token_type_index,
            "obj_x_outputs": obj_x_outputs,
            "obj_y_outputs": obj_y_outputs,
            "obj_z_outputs": obj_z_outputs,
            "obj_theta_outputs": obj_theta_outputs,
            "obj_x_inputs": obj_x_inputs,
            "obj_y_inputs": obj_y_inputs,
            "obj_z_inputs": obj_z_inputs,
            "obj_theta_inputs": obj_theta_inputs,
            "position_index": position_index,
            "struct_position_index": struct_position_index,
            "struct_token_type_index": struct_token_type_index,
            "struct_pad_mask": struct_pad_mask,
            "struct_x_inputs": struct_x_inputs,
            "struct_y_inputs": struct_y_inputs,
            "struct_z_inputs": struct_z_inputs,
            "struct_theta_inputs": struct_theta_inputs,
            "t": 0,
            "filename": ""
        }

        if initial_scene:
            datum["initial_scene"] = initial_scene
        if ids:
            datum["ids"] = ids

        return datum

    @staticmethod
    def convert_to_tensors(datum, tokenizer):

        object_pad_mask = torch.LongTensor(datum["object_pad_mask"])
        other_object_pad_mask = torch.LongTensor(datum["other_object_pad_mask"])
        sentence = torch.LongTensor([tokenizer.tokenize(*i) for i in datum["sentence"]])
        sentence_pad_mask = torch.LongTensor(datum["sentence_pad_mask"])
        token_type_index = torch.LongTensor(datum["token_type_index"])
        obj_x_outputs = torch.FloatTensor(datum["obj_x_outputs"])
        obj_y_outputs = torch.FloatTensor(datum["obj_y_outputs"])
        obj_z_outputs = torch.FloatTensor(datum["obj_z_outputs"])
        obj_theta_outputs = torch.FloatTensor(datum["obj_theta_outputs"])
        obj_x_inputs = torch.FloatTensor(datum["obj_x_inputs"])
        obj_y_inputs = torch.FloatTensor(datum["obj_y_inputs"])
        obj_z_inputs = torch.FloatTensor(datum["obj_z_inputs"])
        obj_theta_inputs = torch.FloatTensor(datum["obj_theta_inputs"])
        position_index = torch.LongTensor(datum["position_index"])
        struct_position_index = torch.LongTensor(datum["struct_position_index"])
        struct_token_type_index = torch.LongTensor(datum["struct_token_type_index"])
        struct_pad_mask = torch.LongTensor(datum["struct_pad_mask"])
        struct_x_inputs = torch.FloatTensor(datum["struct_x_inputs"])
        struct_y_inputs = torch.FloatTensor(datum["struct_y_inputs"])
        struct_z_inputs = torch.FloatTensor(datum["struct_z_inputs"])
        struct_theta_inputs = torch.FloatTensor(datum["struct_theta_inputs"])

        tensors = {
            "xyzs": torch.stack(datum["xyzs"], dim=0),
            "rgbs": torch.stack(datum["rgbs"], dim=0),
            "object_pad_mask": object_pad_mask,
            "other_xyzs": torch.stack(datum["other_xyzs"], dim=0),
            "other_rgbs": torch.stack(datum["other_rgbs"], dim=0),
            "other_object_pad_mask": other_object_pad_mask,
            "sentence": sentence,
            "sentence_pad_mask": sentence_pad_mask,
            "token_type_index": token_type_index,
            "obj_x_outputs": obj_x_outputs,
            "obj_y_outputs": obj_y_outputs,
            "obj_z_outputs": obj_z_outputs,
            "obj_theta_outputs": obj_theta_outputs,
            "obj_x_inputs": obj_x_inputs,
            "obj_y_inputs": obj_y_inputs,
            "obj_z_inputs": obj_z_inputs,
            "obj_theta_inputs": obj_theta_inputs,
            "position_index": position_index,
            "struct_position_index": struct_position_index,
            "struct_token_type_index": struct_token_type_index,
            "struct_pad_mask": struct_pad_mask,
            "struct_x_inputs": struct_x_inputs,
            "struct_y_inputs": struct_y_inputs,
            "struct_z_inputs": struct_z_inputs,
            "struct_theta_inputs": struct_theta_inputs,
            "t": datum["t"],
            "filename": datum["filename"]
        }

        return tensors

    def __getitem__(self, idx):

        datum = self.convert_to_tensors(self.get_raw_data(idx, shuffle_object_index=self.shuffle_object_index),
                                        self.tokenizer)

        return datum

    @staticmethod
    def collate_fn(data):
        """
        :param data:
        :return:
        """

        batched_data_dict = {}
        for key in ["xyzs", "rgbs", "other_xyzs", "other_rgbs"]:
            batched_data_dict[key] = torch.cat([dict[key] for dict in data], dim=0)
        for key in ["object_pad_mask", "other_object_pad_mask", "sentence", "sentence_pad_mask", "token_type_index",
                    "obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                    "obj_x_inputs", "obj_y_inputs", "obj_z_inputs", "obj_theta_inputs", "position_index",
                    "struct_position_index", "struct_token_type_index", "struct_pad_mask",
                    "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
            batched_data_dict[key] = torch.stack([dict[key] for dict in data], dim=0)

        return batched_data_dict

    # def test(self):
    #
    #     h5 = h5py.File("/home/weiyu/data_drive/examples_v4/leonardo/data00000005.h5", "r")
    #
    #     goal_specification = json.loads(str(np.array(h5["goal_specification"])))
    #     print(goal_specification)
    #
    #     moved_objects = h5['moved_objs'][()].split(',')
    #     ids = self._get_ids(h5)
    #     object_names = [o for o in ids.keys() if "object_" in o]
    #     print(object_names)
    #
    #     t = 0
    #     scene, scene_after = self._get_images(h5, t, ee=True)
    #     rgb, depth, seg, valid, xyz = scene
    #     rgb_after, depth_after, seg_after, valid_after, xyz_after = scene_after
    #
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(rgb)
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(rgb_after)
    #     plt.show()
    #
    #     obj_xyzs = []
    #     obj_rgbs = []
    #     for object_name in object_names:
    #         obj_mask = np.logical_and(seg == ids[object_name], valid)
    #         ok, obj_xyz, obj_rgb, _ = get_pts(xyz, rgb, obj_mask, num_pts=self.num_pts)
    #         print("getting points for {}: {}".format(object_name, ok))
    #         obj_xyzs.append(obj_xyz)
    #         obj_rgbs.append(obj_rgb)
    #     show_pcs(obj_xyzs, obj_rgbs, add_coordinate_frame=True)
    #
    #     obj_xyzs_after = []
    #     obj_rgbs_after = []
    #     for object_name in object_names:
    #         obj_mask = np.logical_and(seg_after == ids[object_name], valid_after)
    #         ok, obj_xyz, obj_rgb, _ = get_pts(xyz_after, rgb_after, obj_mask, num_pts=self.num_pts)
    #         print("getting points for {}: {}".format(object_name, ok))
    #         obj_xyzs_after.append(obj_xyz)
    #         obj_rgbs_after.append(obj_rgb)
    #     show_pcs(obj_xyzs_after, obj_rgbs_after, add_coordinate_frame=True)
    #
    #     # move object 1's point cloud
    #     moved_obj1 = moved_objects[0]
    #     print("first moved object is", moved_obj1)
    #     obj1_pose_before = h5[moved_obj1][t]
    #     obj1_pose_after = h5[moved_obj1][t + 1]
    #
    #     print(obj1_pose_before)
    #     T = np.eye(4)
    #     T[:3, :3] = obj1_pose_before[:3, :3]
    #     a, b, theta = tra.euler_from_matrix(T)
    #     print("before pose (x, y, z)", a, b, theta)
    #     print(obj1_pose_after)
    #     T = np.eye(4)
    #     T[:3, :3] = obj1_pose_after[:3, :3]
    #     a, b, theta = tra.euler_from_matrix(T)
    #     print("after pose (x, y, z)", a, b, theta)
    #
    #     # from before to after
    #     R = obj1_pose_before[:3, :3].T @ obj1_pose_after[:3, :3]
    #     T = np.eye(4)
    #     T[:3, :3] = R
    #     a, b, theta = tra.euler_from_matrix(T)
    #     print(theta)
    #     rotation = tra3d.euler_angles_to_matrix(torch.FloatTensor([0, 0, theta]), "XYZ").T
    #     center_after = torch.FloatTensor(obj1_pose_after[:3, 3])
    #
    #     obj_mask_before = np.logical_and(seg == ids[moved_obj1], valid)
    #     ok1, obj1_xyz_before, obj1_rgb_before, _ = get_pts(xyz, rgb, obj_mask_before, num_pts=self.num_pts)
    #
    #     q_center = torch.FloatTensor(obj1_pose_before[:3, 3])
    #     obj_xyz_before_centered = obj1_xyz_before - q_center.view(1, 3).repeat(self.num_pts, 1)
    #     obj_xyz_after_centered = torch.matmul(obj_xyz_before_centered, rotation)
    #     obj1_xyz_after_imagine = obj_xyz_after_centered + center_after.view(1, 3).repeat(self.num_pts, 1)
    #
    #     not_obj_mask_before = np.logical_and(seg != ids[moved_obj1], valid)
    #     _, bg_xyz_before, bg_rgb_before, _ = get_pts(xyz, rgb, not_obj_mask_before, num_pts=10000)
    #
    #     show_pcs([obj1_xyz_after_imagine, bg_xyz_before], [obj1_rgb_before, bg_rgb_before], add_coordinate_frame=True)
    #
    #     # show imagined and actual
    #     show_pcs([obj_xyzs_after[0], obj1_xyz_after_imagine], [obj_rgbs_after[0], obj1_rgb_before])


def process_sentences():

    sentences = []
    with open("/home/weiyu/Research/intern/semantic-rearrangement/src/natural_sentences.txt", "r") as fh:
        for line in fh:
            sentences.append(line.strip())

    token_combos = set()
    for sentence in sentences:
        tokens = sentence.split(" ")
        if "circle" in tokens or "line" in tokens:
            size = tokens[0]
            shape = tokens[1]
            vertical_pos = tokens[4]
            horizontal_pos = tokens[5]
            orientation = tokens[-1]
            print(size, shape, vertical_pos, horizontal_pos, orientation)
            token_combos.add(tuple([size, shape, vertical_pos, horizontal_pos, orientation]))
        elif "tower" in tokens or "dinner" in tokens:
            # tower in the middle center of the table facing south
            shape = tokens[0]
            vertical_pos = tokens[3]
            horizontal_pos = tokens[4]
            orientation = tokens[-1]
            print(shape, vertical_pos, horizontal_pos, orientation)
            token_combos.add(tuple([shape, vertical_pos, horizontal_pos, orientation]))
        else:
            raise KeyError

    print(len(token_combos))

    incomplete_token_combos = set()
    for token_combo in token_combos:
        for num in range(1, len(token_combo) + 1):
            incomplete_token_combos.update(combinations(token_combo, num))

    print(len(incomplete_token_combos))
    print(incomplete_token_combos)




        # Put the  [objects] in a [size][shape] on the [x][y] of the table facing [rotation].
        # Build a [size][shape] of the [objects] on the [x][y] of the table facing [rotation].
        # Put the [objects] on the [x][y] of the table and make a [shape] facing [rotation].
        # Rearrange the [objects] into a [shape], and put the structure on the [x][y] of the table facing [rotation].
        # Could you ...
        # Please ...
        # Pick up the objects, put them into a [size][shape], place the [shape] on the [x][y] of table, make sure the [shape] is facing [rotation].

#     for sentence in unique_sentences:
#         fh.write("{}\n".format(sentence))


if __name__ == "__main__":
    process_sentences()

    # tokenizer = Tokenizer("/home/weiyu/data_drive/data_new_objects/type_vocabs_coarse.json")
    # data_roots = []
    # index_roots = []
    # for shape, index in [("circle", "index_34k"), ("line", "index_42k"), ("tower", "index_13k"), ("dinner", "index_24k")]:
    #     data_roots.append("/home/weiyu/data_drive/data_new_objects/examples_{}_new_objects/result".format(shape))
    #     index_roots.append(index)
    # dataset = SemanticArrangementDataset(data_roots=data_roots,
    #                                      index_roots=index_roots,
    #                                      split="train", tokenizer=tokenizer,
    #                                      max_num_objects=7,
    #                                      max_num_other_objects=5,
    #                                      max_num_shape_parameters=5,
    #                                      max_num_rearrange_features=0,
    #                                      max_num_anchor_features=0,
    #                                      num_pts=1024,
    #                                      debug=True, data_augmentation=False, shuffle_object_index=True)
    #
    # unique_natural_sentences = set()
    # unique_sentences = set()
    # for i in tqdm(range(len(dataset))):
    #     sentence = dataset.get_template_language(i)
    #     # print(sentence)
    #     # print(tokenizer.convert_structure_params_to_natural_language(sentence))
    #     unique_natural_sentences.add(tokenizer.convert_structure_params_to_natural_language(sentence))
    #     unique_sentences.add(str(sentence))
    #     # input("next?")
    #
    # print(len(unique_sentences))
    #
    # with open("/home/weiyu/Research/intern/semantic-rearrangement/src/natural_sentences.txt", "w") as fh:
    #     for sentence in unique_natural_sentences:
    #         fh.write("{}\n".format(sentence))
    #
    # with open("/home/weiyu/Research/intern/semantic-rearrangement/src/sentences.txt", "w") as fh:
    #     for sentence in unique_sentences:
    #         fh.write("{}\n".format(sentence))

