import cv2
import h5py
import numpy as np
import os
import trimesh
import torch
from torch.utils.data import Dataset, DataLoader
import json

# Local imports
from StructDiffusion.utils.rearrangement import show_pcs, get_pts, array_to_tensor, random_move_obj_xyzs
from StructDiffusion.data.tokenizer import Tokenizer
from StructDiffusion.utils.pointnet import pc_normalize
from StructDiffusion.data.MeshSceneViewer import MeshSceneViewer

import StructDiffusion.utils.brain2.camera as cam
import StructDiffusion.utils.brain2.image as img
import StructDiffusion.utils.transformations as tra


"""
This dataset is built on top of dataset_v39. The differences are:
1. we allow random shape rotation
2. we allow pc normalization
3. the language only contains local structure parameters (e.g., radius, structure type) and 
   does not include global parameters (e.g., rotation, position)
"""


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


class SemanticArrangementDataset(torch.utils.data.Dataset):

    def __init__(self, data_roots, index_roots, split, tokenizer,
                 num_random_negative_examples,
                 min_translation, max_translation, min_rotation, max_rotation,
                 max_num_objects=11, max_num_shape_parameters=2,
                 num_pts=1024, num_scene_pts=2048, oversample_positive=False, perturbation_mode="6d",
                 random_structure_rotation=False, return_perturbation_score=False, num_objects_to_include=None,
                 debug=False, data_augmentation=True, old_object_ordering=False, include_env_pc=False, num_env_pts=None,
                 normalize_pc=False, return_current_obj_xyzs=False, return_structure_type_index=False, mesh_scene_viewer=None):

        print("data dirs:", data_roots)

        self.num_pts = num_pts
        self.num_scene_pts = num_scene_pts
        self.debug = debug
        self.old_object_ordering = old_object_ordering
        if self.old_object_ordering:
            print("WARNING: using incorrect object ordering!")
        self.include_env_pc = include_env_pc
        if self.include_env_pc:
            print("include environment point cloud")
            if num_env_pts is None:
                self.num_env_pts = num_pts
            else:
                self.num_env_pts = num_env_pts

        self.max_num_objects = max_num_objects

        self.tokenizer = tokenizer
        self.max_num_shape_parameters = max_num_shape_parameters

        # ToDo: 1. limit the number of objects. There are two ways to do this. One is limit to arrangements with the #objects.
        #          Second is to only use the first #objects in an arrangement.
        #       2. return obj pcl independently

        # whether we add a random rotation to the arrangement
        self.random_structure_rotation = random_structure_rotation
        # whether return a score between 0 and 1 to indicate how far the arrangement is from perfect
        self.return_perturbation_score = return_perturbation_score
        # only include these numbers of objects
        if num_objects_to_include is not None:
            assert type(num_objects_to_include) == int
        self.num_objects_to_include = num_objects_to_include
        self.normalize_pc = normalize_pc
        self.return_current_obj_xyzs = return_current_obj_xyzs
        self.return_structure_type_index = return_structure_type_index
        print("random structure rotation", random_structure_rotation)
        print("return perturbation score", return_perturbation_score)
        print("num objects to include", num_objects_to_include)
        print("normalize pc", normalize_pc)
        print("return_structure_type_index", return_structure_type_index)

        if self.return_structure_type_index:
            self.structure_type_2_index = {"circle": 0, "line": 1, "tower": 2, "dinner": 3}

        # parameters for generating negative examples
        self.perturbation_mode = perturbation_mode
        self.min_translation = min_translation
        self.max_translation = max_translation
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation

        self.mesh_scene_viewer = mesh_scene_viewer

        # each data point is a tuple of (filename, step_t)
        arrangement_steps = []
        for data_root, index_root in zip(data_roots, index_roots):
            arrangement_indices_file = os.path.join(data_root, index_root, "{}_arrangement_indices_file_all.txt".format(split))
            if os.path.exists(arrangement_indices_file):
                with open(arrangement_indices_file, "r") as fh:
                    arrangement_steps.extend([(os.path.join(data_root, f[0]), f[1]) for f in eval(fh.readline().strip())])
            else:
                print("{} does not exist".format(arrangement_indices_file))

        # each data point is a tuple of (filename, step_t, label), where label is True or False
        self.arrangement_data = []
        print("Adding {} random negative examples per arrangement step".format(num_random_negative_examples))
        for filename, step_t in arrangement_steps:

            # sample n negative examples from each step
            for _ in range(num_random_negative_examples):
                self.arrangement_data.append((filename, step_t, False))

            if not oversample_positive:
                self.arrangement_data.append((filename, step_t, True))
            else:
                # balance positive and negative examples
                for _ in range(num_random_negative_examples):
                    self.arrangement_data.append((filename, step_t, True))
        print("{} positive and negative examples in total".format(len(self.arrangement_data)))

        # Noise
        self.data_augmentation = data_augmentation
        # additive noise
        self.gp_rescale_factor_range = [12, 20]
        self.gaussian_scale_range = [0., 0.003]
        # multiplicative noise
        self.gamma_shape = 1000.
        self.gamma_scale = 0.001

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

    def get_random_negative_params(self):
        return self.min_translation, self.max_translation, self.min_rotation, self.max_rotation

    def set_random_negative_params(self, min_translation, max_translation, min_rotation, max_rotation):
        self.min_translation = min_translation
        self.max_translation = max_translation
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation

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

    def get_data_index(self, idx):
        filename, step_t, label = self.arrangement_data[idx]
        return filename, step_t, label

    def get_positive_ratio(self):
        num_pos = 0
        for d in self.arrangement_data:
            filename, step_t, label = d
            if label:
                num_pos += 1
        neg_pos_ratio = (len(self.arrangement_data) - num_pos) * 1.0 / num_pos
        print("negative/positive ratio", neg_pos_ratio)
        return neg_pos_ratio

    def get_raw_data(self, idx, inference_mode=False):

        filename, step_t, label = self.arrangement_data[idx]

        h5 = h5py.File(filename, 'r')
        ids = self._get_ids(h5)
        all_objs = sorted([o for o in ids.keys() if "object_" in o])
        goal_specification = json.loads(str(np.array(h5["goal_specification"])))
        num_rearrange_objs = len(goal_specification["rearrange"]["objects"])
        target_objs = all_objs[:num_rearrange_objs]

        structure_parameters = goal_specification["shape"]
        if not self.old_object_ordering:
            # Important: ensure the order is correct
            if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
                target_objs = target_objs[::-1]
            elif structure_parameters["type"] == "tower" or structure_parameters["type"] == "dinner":
                target_objs = target_objs
            else:
                raise KeyError("{} structure is not recognized".format(structure_parameters["type"]))
        else:
            target_objs = target_objs[::-1]

        # only include this many objects
        if self.num_objects_to_include is not None:
            assert len(target_objs) >= self.num_objects_to_include
            target_objs = target_objs[:self.num_objects_to_include]
        assert len(target_objs) <= self.max_num_objects

        ###################################
        # getting scene images and point clouds
        scene = self._get_images(h5, step_t, ee=True)
        rgb, depth, seg, valid, xyz = scene

        # getting object point clouds
        obj_xyzs = []
        obj_rgbs = []
        obj_pc_centers = []
        object_pad_mask = []
        for obj in target_objs:
            obj_mask = np.logical_and(seg == ids[obj], valid)
            if np.sum(obj_mask) <= 0:
                raise Exception
            ok, obj_xyz, obj_rgb, _ = get_pts(xyz, rgb, obj_mask, num_pts=self.num_pts, to_tensor=False)
            obj_xyzs.append(obj_xyz)
            obj_rgbs.append(obj_rgb)
            obj_pc_centers.append(np.mean(obj_xyz, axis=0))
            object_pad_mask.append(0)

        for _ in range(self.max_num_objects - len(target_objs)):
            object_pad_mask.append(1)

        if self.return_current_obj_xyzs:
            current_obj_xyzs = obj_xyzs

        if self.debug:
            print("current point cloud")
            show_pcs(obj_xyzs, obj_rgbs, add_coordinate_frame=True)

        ###################################
        if self.include_env_pc:
            # getting environment point clouds, e.g., table and distractor objects
            table_mask = np.logical_and(seg == ids["table"], valid)
            env_mask = table_mask
            # ToDo: uncomment code below to add points for distractor objects
            # for obj in all_objs:
            #     if obj not in target_objs:
            #         obj_mask = np.logical_and(seg == ids[obj], valid)
            #         env_mask = np.logical_or(env_mask, obj_mask)

            structure_pc_center = np.mean(obj_pc_centers, axis=0)
            ok, bg_xyz, bg_rgb, _ = get_pts(xyz, rgb, env_mask,
                                            num_pts=self.num_env_pts, center=structure_pc_center, radius=0.5)
            if self.debug:
                print("env point cloud")
                show_pcs([bg_xyz], [bg_rgb])
                show_pcs([bg_xyz] + obj_xyzs, [bg_rgb] + obj_rgbs)

        ###################################
        # find goal translation and rotation to move object point clouds in the (rotated) structure frame
        structure_pose = tra.euler_matrix(structure_parameters["rotation"][0], structure_parameters["rotation"][1],
                                          structure_parameters["rotation"][2])
        structure_pose[:3, 3] = [structure_parameters["position"][0], structure_parameters["position"][1],
                                 structure_parameters["position"][2]]
        structure_pose_inv = np.linalg.inv(structure_pose)

        if self.random_structure_rotation:
            random_rotation = np.random.uniform(low=0, high=2 * np.pi) * np.random.choice([-1, 1])
            struct_random_yaw = tra.euler_matrix(0, 0, random_rotation)

        current_obj_poses = []
        goal_obj_poses = []
        goal_pc_poses = []
        current_pc_poses = []
        for obj, obj_pc_center in zip(target_objs, obj_pc_centers):
            current_pc_pose = np.eye(4)
            current_pc_pose[:3, 3] = obj_pc_center[:3]
            current_pc_poses.append(current_pc_pose)

            goal_pose = h5[obj][0]
            current_pose = h5[obj][step_t]
            goal_obj_poses.append(goal_pose)
            current_obj_poses.append(current_pose)

            goal_pc_pose = goal_pose @ np.linalg.inv(current_pose) @ current_pc_pose
            if self.random_structure_rotation:
                goal_pc_pose = structure_pose @ struct_random_yaw @ structure_pose_inv @ goal_pc_pose
            goal_pc_poses.append(goal_pc_pose)

        # move object point clouds to goal poses in world frame
        goal_obj_xyzs = []
        for i, (obj, obj_xyz) in enumerate(zip(target_objs, obj_xyzs)):

            goal_pc_pose = goal_pc_poses[i]
            current_pc_pose = current_pc_poses[i]

            goal_pc_transform = goal_pc_pose @ np.linalg.inv(current_pc_pose)
            new_obj_xyz = trimesh.transform_points(obj_xyz, goal_pc_transform)
            goal_obj_xyzs.append(new_obj_xyz)

        obj_xyzs = goal_obj_xyzs

        # rotate env point if a random rotation has been added to objects
        if self.include_env_pc and self.random_structure_rotation:
            centered_bg_xyz = bg_xyz - structure_pose[:3, 3]
            centered_bg_xyz = trimesh.transform_points(centered_bg_xyz, struct_random_yaw, translate=False)
            bg_xyz = centered_bg_xyz + structure_pose[:3, 3]

        if self.debug:
            if self.include_env_pc:
                print("rotated goal object point cloud + env point cloud")
                show_pcs([bg_xyz] + obj_xyzs, [bg_rgb] + obj_rgbs, add_coordinate_frame=True)
            else:
                print("rotated goal object point cloud")
                show_pcs(obj_xyzs, obj_rgbs, add_coordinate_frame=True)

        ###################################
        # if negative example, perturb object point cloud
        total_perturbation_translation = 0
        total_perturbation_rotation = 0
        if not label:

            # debug:
            # obj, moved_obj_idxs, obj_perturbation_matrices = random_move_obj_xyzs(obj_xyzs,
            #                                 min_translation=self.min_translation, max_translation=self.max_translation,
            #                                 min_rotation=self.min_rotation, max_rotation=self.max_rotation,
            #                                 mode=self.perturbation_mode, return_perturbation=True, return_moved_obj_idxs=True)
            _, moved_obj_idxs, obj_perturbation_matrices = random_move_obj_xyzs(obj_xyzs,
                                            min_translation=self.min_translation, max_translation=self.max_translation,
                                            min_rotation=self.min_rotation, max_rotation=self.max_rotation,
                                            mode=self.perturbation_mode, return_perturbation=True, return_moved_obj_idxs=True,
                                            return_perturbed_obj_xyzs=False)

            perturbed_obj_xyzs = []
            for goal_pc_pose, obj_xyz, pm in zip(goal_pc_poses, obj_xyzs, obj_perturbation_matrices):
                # perturbation happens in the goal pc pose, therefore first move pc in the world frame back to the goal pc pose
                perturbed_obj_xyz = trimesh.transform_points(obj_xyz, goal_pc_pose @ pm @ np.linalg.inv(goal_pc_pose))
                perturbed_obj_xyzs.append(perturbed_obj_xyz)
            obj_xyzs = perturbed_obj_xyzs

            for pm in obj_perturbation_matrices:
                total_perturbation_translation += np.sum(np.abs(pm[:3, 3]))
                total_perturbation_rotation += np.sum(np.abs(tra.euler_from_matrix(pm)))
            total_perturbation_rotation = total_perturbation_rotation
            total_perturbation_translation = total_perturbation_translation

            if self.debug:
                print("perturbing {} objects".format(len(moved_obj_idxs)))
                print("total perturbation translation in m:", total_perturbation_translation)
                print("total perturbation rotation in radiance:", total_perturbation_rotation)
                print("perturbed point cloud")
                for pm in obj_perturbation_matrices:
                    print(pm)
                show_pcs(obj_xyzs, obj_rgbs, add_coordinate_frame=True)

        # ToDo: require the model to predict perturbation for each object

        ###################################
        num_indicator = self.max_num_objects
        if self.include_env_pc:
            # add env points after perturbing objects
            obj_xyzs = [bg_xyz] + obj_xyzs
            obj_rgbs = [bg_rgb] + obj_rgbs
            num_indicator += 1

        ###################################
        # add one hot
        new_obj_xyzs = []
        # ToDo: we can also shuffle the object point clouds
        for oi, obj_xyz in enumerate(obj_xyzs):
            obj_xyz = np.concatenate([obj_xyz, np.tile(np.eye(num_indicator)[oi], (obj_xyz.shape[0], 1))], axis=1)
            new_obj_xyzs.append(obj_xyz)
        scene_xyz = np.concatenate(new_obj_xyzs, axis=0)

        # fsp
        # scene_xyz = scene_xyz.unsqueeze(0)
        # fps_idx = farthest_point_sample(scene_xyz[:, :, 0:3], self.num_scene_pts)  # [B, npoint]
        # scene_xyz = index_points(scene_xyz, fps_idx).squeeze(0)
        # scene_xyz[:, 0:3] = pc_normalize(scene_xyz[:, 0:3])

        # subsampling and normalizing pc
        idx = np.random.randint(0, scene_xyz.shape[0], self.num_scene_pts)
        scene_xyz = scene_xyz[idx]
        if self.normalize_pc:
            scene_xyz[:, 0:3] = pc_normalize(scene_xyz[:, 0:3])

        ###################################
        # preparing sentence
        sentence = []
        sentence_pad_mask = []

        # structure parameters
        # 2 parameters
        structure_parameters = goal_specification["shape"]
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            sentence.append((structure_parameters["type"], "shape"))
            if structure_parameters["type"] == "circle":
                sentence.append((structure_parameters["radius"], "radius"))
            elif structure_parameters["type"] == "line":
                sentence.append((structure_parameters["length"] / 2.0, "radius"))
            for _ in range(2):
                sentence_pad_mask.append(0)
        else:
            sentence.append((structure_parameters["type"], "shape"))
            for _ in range(1):
                sentence_pad_mask.append(0)
            sentence.append(("PAD", None))
            sentence_pad_mask.append(1)
        ###################################

        scene_xyz = array_to_tensor(scene_xyz)

        if self.return_perturbation_score:
            # ToDo: find a better way to compute score
            obj_num = self.num_objects_to_include if self.num_objects_to_include is not None else self.max_num_objects
            # option 1:
            # # 8:2 because position is more important than rotation?
            # t_distance = 0.8 * total_perturbation_translation / (self.max_translation * 3 * obj_num)  # 3 is from xyz
            # r_distance = 0.2 * total_perturbation_rotation / (self.max_rotation * 3 * obj_num)  # 3 is from rpy
            # # print(t_distance + r_distance)
            # score = max(0, 1 - (t_distance + r_distance))
            # option 2: just count how many objects are perturbed
            if label:
                score = 1.0
            else:
                score = 1.0 - len(moved_obj_idxs) * 1.0 / len(obj_perturbation_matrices)
        else:
            score = int(label)

        position_index = list(range(self.max_num_shape_parameters))

        if self.return_current_obj_xyzs:
            for i in range(len(current_obj_xyzs)):
                if type(current_obj_xyzs[i]).__module__ == np.__name__:
                    current_obj_xyzs[i] = array_to_tensor(current_obj_xyzs[i])
            # pad data
            for i in range(self.max_num_objects - len(current_obj_xyzs)):
                current_obj_xyzs.append(torch.zeros([self.num_pts, 3], dtype=torch.float32))

        if self.debug:
            print("score:", score)
            print("all objects:", all_objs)
            print("sentence:", sentence)
            print("sentence_pad_mask:", sentence_pad_mask)
            print("position_index:", position_index)
            print("object_pad_mask:", object_pad_mask)
            show_pcs(obj_xyzs, obj_rgbs, add_coordinate_frame=True)
            show_pcs([scene_xyz[:, 0:3]], [np.tile(np.array([0, 1, 0], dtype=np.float), (scene_xyz.shape[0], 1))],
                     add_coordinate_frame=True)
            if self.return_current_obj_xyzs:
                show_pcs(current_obj_xyzs, obj_rgbs, add_coordinate_frame=True)

        datum = {
            "scene_xyz": scene_xyz,
            "is_circle": score,
            "object_pad_mask": object_pad_mask,
            "sentence": sentence,
            "sentence_pad_mask": sentence_pad_mask,
            "position_index": position_index,
        }
        if self.return_current_obj_xyzs:
            datum["current_obj_xyzs"] = current_obj_xyzs

        if self.return_structure_type_index:
            datum["structure_type_index"] = self.structure_type_2_index[structure_parameters["type"]]

        if inference_mode:
            datum["obj_xyzs"] = obj_xyzs
            datum["obj_rgbs"] = obj_rgbs
            datum["goal_specification"] = goal_specification

            datum["target_objs"] = target_objs
            datum["structure_pose"] = structure_pose
            datum["current_obj_poses"] = current_obj_poses
            datum["goal_obj_poses"] = goal_obj_poses
            datum["goal_pc_poses"] = goal_pc_poses
            datum["current_pc_poses"] = current_pc_poses
            if not label:
                datum["obj_perturbation_matrices"] = obj_perturbation_matrices
            if self.random_structure_rotation:
                datum["struct_random_yaw"] = struct_random_yaw

        if self.mesh_scene_viewer:
            if score:
                scene = mesh_scene_viewer.load_mesh_scene(target_objs, goal_specification, current_obj_poses,
                                                          current_pc_poses, goal_pc_poses, visualize=False)
                collision = mesh_scene_viewer.check_scene_collision(scene)
                print("scene has collision", collision)
            else:
                perturbed_goal_pc_poses = []
                for goal_pc_pose, pm in zip(goal_pc_poses, obj_perturbation_matrices):
                    pgpcp = goal_pc_pose @ pm
                    perturbed_goal_pc_poses.append(pgpcp)
                scene = mesh_scene_viewer.load_mesh_scene(target_objs, goal_specification, current_obj_poses,
                                                          current_pc_poses, perturbed_goal_pc_poses, visualize=False)
                collision = mesh_scene_viewer.check_scene_collision(scene)
                print("scene has collision", collision)


            pc_vis = [trimesh.PointCloud(obj_xyz) for obj_xyz in obj_xyzs]
            trimesh.Scene([pc_vis, scene]).show()

        return datum

    # ToDo: add language
    # ToDo: remove pc normalize
    def convert_to_scene_xyz(self, obj_xyzs, label, sentence, sentence_pad_mask, initial_scene=None, ids=None, structure_params=None, debug=False):

        # check if all pcs are non-zero
        valid_pcs = []
        for obj_xyz in obj_xyzs:
            if type(obj_xyz).__module__ == np.__name__:
                 valid_pcs.append(np.sum(obj_xyz) != 0)
            else:
                valid_pcs.append(torch.sum(obj_xyz) != 0)
        print(valid_pcs)
        assert sum(valid_pcs) == len(obj_xyzs)

        assert len(obj_xyzs) <= self.max_num_objects

        assert len(sentence) == self.max_num_shape_parameters
        assert len(sentence_pad_mask) == self.max_num_shape_parameters

        if self.include_env_pc:
            assert initial_scene is not None
            assert ids is not None

        ###################################
        # if object poses are in the structure frame, put them in the global frame
        if structure_params:
            raise NotImplementedError("does not support transposing bg xyz")
            # assert len(structure_params) == 6
            # new_obj_xyzs = []
            # for obj_xyz in obj_xyzs:
            #     T = tra.euler_matrix(0, 0, structure_params[5])
            #     T[:3, 3] = [structure_params[0], structure_params[1],
            #                 structure_params[2]]
            #     new_obj_xyz = trimesh.transform_points(obj_xyz, np.linalg.inv(T))
            #     new_obj_xyzs.append(new_obj_xyz)
            # obj_xyzs = new_obj_xyzs

        ###################################
        object_pad_mask = []
        for obj_xyz in obj_xyzs:
            object_pad_mask.append(0)
        for _ in range(self.max_num_objects - len(obj_xyzs)):
            object_pad_mask.append(1)

        if self.return_current_obj_xyzs:
            current_obj_xyzs = obj_xyzs

        ###################################
        if self.include_env_pc:
            rgb, depth, seg, valid, xyz = initial_scene

            if type(obj_xyzs[0]).__module__ == np.__name__:
                obj_pc_centers = [np.mean(obj_xyz, axis=0) for obj_xyz in obj_xyzs]
            else:
                obj_pc_centers = [torch.mean(obj_xyz, dim=0).numpy() for obj_xyz in obj_xyzs]

            # getting environment point clouds, e.g., table and distractor objects
            # should we get environment point cloud
            table_mask = np.logical_and(seg == ids["table"], valid)
            env_mask = table_mask
            # ToDo: uncomment code below to add points for distractor objects
            # for obj in all_objs:
            #     if obj not in target_objs:
            #         obj_mask = np.logical_and(seg == ids[obj], valid)
            #         env_mask = np.logical_or(env_mask, obj_mask)

            structure_pc_center = np.mean(obj_pc_centers, axis=0)
            ok, bg_xyz, bg_rgb, _ = get_pts(xyz, rgb, env_mask,
                                            num_pts=self.num_env_pts, center=structure_pc_center, radius=0.5)

            # if self.debug:
            #     print("env point cloud")
            #     show_pcs([bg_xyz], [bg_rgb])
            #     show_pcs([bg_xyz] + obj_xyzs, [bg_rgb] + obj_rgbs)

        ###################################
        num_indicator = self.max_num_objects
        if self.include_env_pc:
            obj_xyzs = [bg_xyz] + obj_xyzs
            num_indicator += 1

        # add one hot
        new_obj_xyzs = []
        for oi, obj_xyz in enumerate(obj_xyzs):
            obj_xyz = np.concatenate([obj_xyz, np.tile(np.eye(num_indicator)[oi], (obj_xyz.shape[0], 1))], axis=1)
            new_obj_xyzs.append(obj_xyz)
        scene_xyz = np.concatenate(new_obj_xyzs, axis=0)

        # subsampling and normalizing pc
        idx = np.random.randint(0, scene_xyz.shape[0], self.num_scene_pts)
        scene_xyz = scene_xyz[idx]
        if self.normalize_pc:
            scene_xyz[:, 0:3] = pc_normalize(scene_xyz[:, 0:3])

        ###################################
        scene_xyz = array_to_tensor(scene_xyz)

        position_index = list(range(self.max_num_shape_parameters))

        if self.return_current_obj_xyzs:
            for i in range(len(current_obj_xyzs)):
                if type(current_obj_xyzs[i]).__module__ == np.__name__:
                    current_obj_xyzs[i] = array_to_tensor(current_obj_xyzs[i])
            # pad data
            for i in range(self.max_num_objects - len(current_obj_xyzs)):
                current_obj_xyzs.append(torch.zeros([self.num_pts, 3], dtype=torch.float32))

        # convert to torch data
        is_circle = bool(label)

        if debug:
            print("is circle:", is_circle)
            print("sentence:", sentence)
            print("sentence_pad_mask:", sentence_pad_mask)
            print("position_index:", position_index)
            print("object_pad_mask:", object_pad_mask)
            show_pcs([scene_xyz[:, 0:3]], [np.tile(np.array([0, 1, 0], dtype=np.float), (scene_xyz.shape[0], 1))],
                     add_coordinate_frame=True)

        datum = {
            "scene_xyz": scene_xyz,
            "is_circle": is_circle,
            "object_pad_mask": object_pad_mask,
            "sentence": sentence,
            "sentence_pad_mask": sentence_pad_mask,
            "position_index": position_index,
        }
        if self.return_current_obj_xyzs:
            datum["current_obj_xyzs"] = current_obj_xyzs

        if self.structure_type_2_index:
            raise NotImplementedError("Output structure type index not implemented")

        return datum

    # def convert_to_scene_xyz_tensor(self, obj_xyzs, t, debug=False):
    #     """
    #     Tensor version, supports backpropagation
    #     """
    #     ##########################################
    #     # prepare
    #     new_obj_xyzs = []
    #     for oi, obj_xyz in enumerate(obj_xyzs):
    #         obj_xyz = torch.cat([obj_xyz, torch.eye(self.max_num_objects)[oi].repeat(obj_xyz.shape[0], 1)], dim=1)
    #         new_obj_xyzs.append(obj_xyz)
    #     scene_xyz = torch.cat(new_obj_xyzs, dim=0)
    #
    #     # fsp
    #     # scene_xyz = scene_xyz.unsqueeze(0)
    #     # fps_idx = farthest_point_sample(scene_xyz[:, :, 0:3], self.num_scene_pts)  # [B, npoint]
    #     # scene_xyz = index_points(scene_xyz, fps_idx).squeeze(0)
    #     # scene_xyz[:, 0:3] = pc_normalize(scene_xyz[:, 0:3])
    #
    #     # subsampling
    #     idx = torch.randint(0, scene_xyz.shape[0], (self.num_scene_pts,))   # np.random.randint(0, scene_xyz.shape[0], self.num_scene_pts)
    #     scene_xyz = scene_xyz[idx]
    #     scene_xyz[:, 0:3] = pc_normalize(scene_xyz[:, 0:3])
    #
    #     ###################################
    #
    #     # convert to torch data
    #     is_circle = t == 0
    #
    #     if debug:
    #         print("is circle:", is_circle)
    #         show_pcs([scene_xyz[:, 0:3]],
    #                  [np.tile(np.array([0, 1, 0], dtype=np.float), (scene_xyz.shape[0], 1))],
    #                  add_coordinate_frame=True)
    #
    #     datum = {
    #         "scene_xyz": scene_xyz,
    #         "is_circle": is_circle,
    #     }
    #
    #     return datum

    @staticmethod
    def convert_to_tensors(datum, tokenizer, return_current_obj_xyzs, return_structure_type_index):

        object_pad_mask = torch.LongTensor(datum["object_pad_mask"])
        sentence = torch.LongTensor([tokenizer.tokenize(*i) for i in datum["sentence"]])
        sentence_pad_mask = torch.LongTensor(datum["sentence_pad_mask"])
        position_index = torch.LongTensor(datum["position_index"])

        tensors = {
            "scene_xyz": datum["scene_xyz"],
            "is_circle": torch.FloatTensor([datum["is_circle"]]),
            "object_pad_mask": object_pad_mask,
            "sentence": sentence,
            "sentence_pad_mask": sentence_pad_mask,
            "position_index": position_index,
        }

        if return_current_obj_xyzs:
            tensors["current_obj_xyzs"] = torch.stack(datum["current_obj_xyzs"], dim=0)
        if return_structure_type_index:
            tensors["structure_type_index"] = torch.LongTensor([datum["structure_type_index"]])

        return tensors

    def tensorfy_sentence(self, raw_sentence, raw_sentence_pad_mask, raw_position_index):
        sentence = torch.LongTensor([self.tokenizer.tokenize(*i) for i in raw_sentence])
        sentence_pad_mask = torch.LongTensor(raw_sentence_pad_mask)
        position_index = torch.LongTensor(raw_position_index)
        return sentence, sentence_pad_mask, position_index

    def __getitem__(self, idx):

        datum = self.convert_to_tensors(self.get_raw_data(idx), self.tokenizer, self.return_current_obj_xyzs, self.return_structure_type_index)

        return datum

    @staticmethod
    def collate_fn(data):
        """
        :param data:
        :return:
        """

        batched_data_dict = {}
        for key in ["is_circle", "current_obj_xyzs"]:
            if key in data[0]:
                batched_data_dict[key] = torch.cat([dict[key] for dict in data], dim=0)
        for key in ["scene_xyz", "object_pad_mask", "sentence", "sentence_pad_mask", "position_index", "structure_type_index"]:
            if key in data[0]:
                batched_data_dict[key] = torch.stack([dict[key] for dict in data], dim=0)

        # structure_type_index: B, 1

        return batched_data_dict


if __name__ == "__main__":
    # dataset = SemanticArrangementDataset(data_roots=["/home/weiyu/data_drive/data_new_objects/examples_tower_new_objects/result"],
    #                                      index_roots=["index_13k"], split="train",
    #                                      num_random_negative_examples=1,
    #                                      min_translation=0.01, max_translation=0.08, min_rotation=0.1, max_rotation=1,
    #                                      max_num_objects=7, num_pts=1024, num_scene_pts=2048, oversample_positive=True,
    #                                      perturbation_mode="3d_planar", random_structure_rotation=True, return_perturbation_score=True,
    #                                      num_objects_to_include=2,
    #                                      debug=True, data_augmentation=False)

    tokenizer = Tokenizer("/home/weiyu/data_drive/data_new_objects/type_vocabs_coarse.json")
    mesh_scene_viewer = MeshSceneViewer(assets_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large", cache_mesh=True)

    dataset = SemanticArrangementDataset(
        data_roots=["/home/weiyu/data_drive/data_new_objects/examples_stacking_new_objects/result"],
        index_roots=["index_10k"], split="test", tokenizer=tokenizer,
        num_random_negative_examples=1,
        min_translation=0.01, max_translation=0.08, min_rotation=0.1, max_rotation=0.5,
        max_num_objects=7, max_num_shape_parameters=5,
        num_pts=1024, num_scene_pts=2048, oversample_positive=False,
        perturbation_mode="3d_planar", random_structure_rotation=True, return_perturbation_score=False,
        num_objects_to_include=None,
        debug=True, data_augmentation=False, include_env_pc=False, normalize_pc=True, return_current_obj_xyzs=True,
        return_structure_type_index=True, mesh_scene_viewer=mesh_scene_viewer)

    # assets_path = "/home/weiyu/Research/intern/brain_gym/assets/urdf"
    # object_model_dir = "/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large"
    #
    # with open("/home/weiyu/Research/intern/semantic-rearrangement/src/generative_models/check_line.txt", "a") as fh:
    #     for i in [227]: # list(range(len(dataset))):
    #         d = dataset.get_raw_data(i, inference_mode=True)
    #         label = d["is_circle"]
    #         check, check_dict = verify_datum_in_simulation(d, assets_path, object_model_dir, visualize=False)
    #         print("data point {}, label {}, check {}".format(i, bool(label), bool(check)))
    #         print(check_dict)
    #         fh.write("data point {}, label {}, check {}: {}\n".format(i, bool(label), bool(check), check_dict))


    for i in range(len(dataset)):
        d = dataset.get_raw_data(i, inference_mode=False)
        print(d["sentence"])
        input("next?")


    # dataset = SemanticArrangementDataset(
    #     data_roots=["/home/weiyu/data_drive/data_new_objects/examples_line_new_objects/result"],
    #     index_roots=["index_10k"], split="test", tokenizer=tokenizer,
    #     num_random_negative_examples=1,
    #     min_translation=0.01, max_translation=0.08, min_rotation=0.1, max_rotation=0.5,
    #     max_num_objects=3, max_num_shape_parameters=5,
    #     num_pts=1024, num_scene_pts=2048, oversample_positive=False,
    #     perturbation_mode="3d_planar", random_structure_rotation=False, return_perturbation_score=False,
    #     num_objects_to_include=3,
    #     debug=True, data_augmentation=False, include_env_pc=False, normalize_pc=False)

    # print(dataset.get_random_negative_params())
    # dataset.set_random_negative_params(min_translation, max_translation, min_rotation, max_rotation):

    # dataset.get_positive_ratio()

    # for i in range(49338, 50000):
    # for i in range(1000, 10000):
    #     print("-----------------------")
    #     print(i)
    #     dataset.get_raw_data(i)
    #     input("next?")
    #     # d = dataset.__getitem__(i)
    #     # for k in d:
    #     #     print("--size", k, d[k].shape)
    #     # for k in d:
    #     #     print(k, d[k])

    # start_time = time.time()
    # for i, d in enumerate(dataset):
    #     print(i)
    #     if i == 100:
    #         break
    # print(time.time() - start_time)

    # dataset.test()

    # for i, d in enumerate(dataset):
    #     print(i)
    #     for k in d:
    #         print("--size", k, d[k].shape)
    #     for k in d:
    #         print(k, d[k])
    #
    #     input("next?")

    # dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1,
    #                         collate_fn=SemanticArrangementDataset.collate_fn, pin_memory=True)
    # # T = 0
    # # batch_timer_start = time.time()
    # # for step, d in enumerate(dataloader):
    # #     batch_timer_stop = time.time()
    # #     elapsed = batch_timer_stop - batch_timer_start
    # #     T += elapsed
    # #     print("Load data time (step {}): {}".format(step, elapsed))
    # #     batch_timer_start = time.time()
    # # print("Total time: {}".format(T))
    #
    # for i, d in enumerate(dataloader):
    #     print(i)
    #     for k in d:
    #         print("--size", k, d[k].shape)
    #     for k in d:
    #         print(k, d[k])
    #
    #     input("next?")
