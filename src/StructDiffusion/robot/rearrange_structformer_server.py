#! /usr/bin/env python
from __future__ import print_function

import os.path
import sys
import rospy
import datetime
import trimesh
import numpy as np
import pytorch3d.transforms as tra3d
from geometry_msgs.msg import PoseArray, Transform, TransformStamped, Pose
from rail_manipulation_msgs.srv import RearrangeObjects, RearrangeObjectsResponse

import tf.transformations as tf_tra

import sensor_msgs.point_cloud2 as point_cloud2
import ctypes
import struct

import tqdm
import torch
from torch.utils.data import DataLoader

from StructDiffusion.utils.torch_data import default_collate
from StructDiffusion.data.dataset_v1_diffuser import SemanticArrangementDataset
import StructDiffusion.utils.transformations as tra

from StructDiffusion.utils.batch_inference import move_pc, move_pc_and_create_scene_new, visualize_batch_pcs


import StructDiffusion.data.dataset_v23_continuous_out_ar_6d_all_objects_all_shapes as prior_dataset
import StructDiffusion.training.train_prior_continuous_out_encoder_decoder_struct_pct_6d_dropout_all_objects_all_shapes as prior_model
from StructDiffusion.evaluation.infer_collision import CollisionInference
from StructDiffusion.evaluation.infer_discriminator import DiscriminatorInference


class PriorInference:

    def __init__(self, model_dir, data_split="test", override_data_dirs=None, override_index_dirs=None):
        # load prior
        cfg, tokenizer, model, optimizer, scheduler, epoch = prior_model.load_model(model_dir, ngc_vocab=False)

        data_cfg = cfg.dataset

        # data_cfg.dir = "/home/weiyu/data_drive/examples_local_200/result"
        # data_cfg.index_dir = "index_200"
        if override_data_dirs is None:
            override_data_dirs = data_cfg.dirs
        if override_index_dirs is None:
            override_index_dirs = data_cfg.index_dirs
        dataset = prior_dataset.SemanticArrangementDataset(override_data_dirs, override_index_dirs, data_split, tokenizer,
                                                           data_cfg.max_num_objects,
                                                           data_cfg.max_num_other_objects,
                                                           data_cfg.max_num_shape_parameters,
                                                           data_cfg.max_num_rearrange_features,
                                                           data_cfg.max_num_anchor_features,
                                                           data_cfg.num_pts)

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model
        self.dataset = dataset
        self.epoch = epoch

    def validate(self):
        data_cfg = self.cfg.dataset
        data_iter = DataLoader(self.dataset, batch_size=data_cfg.batch_size, shuffle=False,
                               collate_fn=prior_dataset.SemanticArrangementDataset.collate_fn,
                               pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)

        prior_model.validate(self.cfg, self.model, data_iter, self.epoch, self.cfg.device)

    def limited_batch_inference(self, data, verbose=False, convert_to_tensors=True, return_numpy=True):
        """
        This function makes the assumption that scenes in the batch have the same number of objects that need to be
        rearranged

        :param data:
        :param model:
        :param test_dataset:
        :param tokenizer:
        :param cfg:
        :param num_samples:
        :param verbose:
        :return:
        """

        data_size = len(data)
        batch_size = self.cfg.dataset.batch_size
        if verbose:
            print("data size:", data_size)
            print("batch size:", batch_size)

        num_batches = int(data_size / batch_size)
        if data_size % batch_size != 0:
            num_batches += 1

        all_obj_preds = []
        all_struct_preds = []
        for b in range(num_batches):
            if b + 1 == num_batches:
                # last batch
                batch = data[b * batch_size:]
            else:
                batch = data[b * batch_size: (b+1) * batch_size]
            if convert_to_tensors:
                data_tensors = [self.dataset.convert_to_tensors(d, self.tokenizer) for d in batch]
            else:
                data_tensors = batch
            data_tensors = self.dataset.collate_fn(data_tensors)
            predictions = prior_model.infer_once(self.cfg, self.model, data_tensors, self.cfg.device)

            obj_x_preds = torch.cat(predictions["obj_x_outputs"], dim=0)
            obj_y_preds = torch.cat(predictions["obj_y_outputs"], dim=0)
            obj_z_preds = torch.cat(predictions["obj_z_outputs"], dim=0)
            obj_theta_preds = torch.cat(predictions["obj_theta_outputs"], dim=0)
            obj_preds = torch.cat([obj_x_preds, obj_y_preds, obj_z_preds, obj_theta_preds], dim=1)  # batch_size * max num objects, output_dim

            struct_x_preds = torch.cat(predictions["struct_x_inputs"], dim=0)
            struct_y_preds = torch.cat(predictions["struct_y_inputs"], dim=0)
            struct_z_preds = torch.cat(predictions["struct_z_inputs"], dim=0)
            struct_theta_preds = torch.cat(predictions["struct_theta_inputs"], dim=0)
            struct_preds = torch.cat([struct_x_preds, struct_y_preds, struct_z_preds, struct_theta_preds], dim=1) # batch_size, output_dim

            all_obj_preds.append(obj_preds)
            all_struct_preds.append(struct_preds)

        obj_preds = torch.cat(all_obj_preds, dim=0)  # data_size * max num objects, output_dim
        struct_preds = torch.cat(all_struct_preds, dim=0)  # data_size, output_dim

        if return_numpy:
            obj_preds = obj_preds.detach().cpu().numpy()
            struct_preds = struct_preds.detach().cpu().numpy()

        obj_preds = obj_preds.reshape(data_size, -1, obj_preds.shape[-1])  # batch_size, max num objects, output_dim

        return struct_preds, obj_preds



def pc2_to_xyzrgb(pc2, skip_nans=True):
    gen = point_cloud2.read_points(pc2, skip_nans=skip_nans)
    pc_list = []
    for x in list(gen):
        test = x[3]
        # cast float32 to int so that bitwise operations are possible
        s = struct.pack('>f' ,test)
        i = struct.unpack('>l',s)[0]
        # you can get back the float value by the inverse operations
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000)>> 16
        g = (pack & 0x0000FF00)>> 8
        b = (pack & 0x000000FF)
        # print r,g,b # prints r,g,b values in the 0-255 range
        #             # x,y,z can be retrieved from the x[0],x[1],x[2]
        pc_list.append([x[0],x[1],x[2], r, g, b])
    return np.array(pc_list)


def transform_matrix_to_pose_msg(transform_matrix):
    pose = Pose()
    quat = tf_tra.quaternion_from_matrix(transform_matrix)
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    pose.position.x = transform_matrix[0, 3]
    pose.position.y = transform_matrix[1, 3]
    pose.position.z = transform_matrix[2, 3]
    return pose


def batch_object_transform_matrices_to_pose_array_list(batch_object_transform_matrices):
    # batch_object_transform_matrices: B, N, 4, 4
    pose_arrays = []
    for object_transform_matrices in batch_object_transform_matrices:
        # N, 4, 4
        poses = []
        for obj_t_m in object_transform_matrices:
            # 4, 4
            obj_pose = transform_matrix_to_pose_msg(obj_t_m)
            poses.append(obj_pose)
        pose_array = PoseArray()
        pose_array.poses = poses
        pose_arrays.append(pose_array)
    return pose_arrays


class RearrangementModelServer:

    def __init__(self, model_dir,
                 discriminator_model_dir=None, discriminator_score_weight=0.0,
                 collision_model_dir=None, collision_score_weight=0.0,
                 num_samples=50, num_elite=5, discriminator_inference_batch_size=10,
                 visualize=False, save_dir=None):

        self.prior_inference = PriorInference(model_dir)

        self.discriminator_score_weight = discriminator_score_weight
        if discriminator_score_weight > 0:
            self.discriminator_inference = DiscriminatorInference(discriminator_model_dir)
        self.collision_score_weight = collision_score_weight
        if collision_score_weight > 0:
            self.collision_inference = CollisionInference(collision_model_dir, empty_dataset=True)

        self.num_samples = num_samples
        self.num_elite = num_elite
        self.discriminator_inference_batch_size = discriminator_inference_batch_size
        self.visualize = visualize
        self.save_dir = save_dir

        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # TODO: figure out how to resolve this name
        self.server = rospy.Service('rearrangement_model', RearrangeObjects, self.run_model)
        print("Rearrangement model server started")


    def run_model(self, request):

        obj_pc2s = request.object_point_clouds
        xyzrgbs = [pc2_to_xyzrgb(pc2) for pc2 in obj_pc2s]

        shape = request.shape
        position_x = request.position_x
        position_y = request.position_y
        rotation = request.rotation
        radius = request.radius

        assert shape in ["line", "circle", "dinner", "tower"]
        assert -0.1 <= position_x <= 1.0
        assert -0.5 <= position_y <= 0.5
        assert -3.15 <= rotation <= 3.15
        assert 0.0 <= radius <= 0.5

        current_pc_poses, goal_pc_poses = self.predict(xyzrgbs, shape, position_x, position_y, rotation, radius)

        batch_current_pc_poses = batch_object_transform_matrices_to_pose_array_list(current_pc_poses)
        batch_goal_pc_poses = batch_object_transform_matrices_to_pose_array_list(goal_pc_poses)

        response = RearrangeObjectsResponse()
        response.batch_current_pc_poses = batch_current_pc_poses
        response.batch_goal_pc_poses = batch_goal_pc_poses

        return response

    def predict(self, xyzrgbs, shape, position_x, position_y, rotation, radius):

        num_samples = self.num_samples
        prior_inference = self.prior_inference
        device = self.prior_inference.cfg.device
        prior_dataset = self.prior_inference.dataset
        discriminator_score_weight = self.discriminator_score_weight
        collision_score_weight = self.collision_score_weight
        visualize = self.visualize
        num_elite = self.num_elite

        data_cfg = self.prior_inference.cfg.dataset

        if discriminator_score_weight > 0:
            discriminator_inference = self.discriminator_inference
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
            collision_inference = self.collision_inference
            collision_model = collision_inference.model
            collision_cfg = collision_inference.cfg
            collision_model.eval()
            collision_num_pair_pc_pts = collision_cfg.dataset.num_scene_pts
            collision_normalize_pc = collision_cfg.dataset.normalize_pc
        else:
            collision_num_pair_pc_pts = None
            collision_normalize_pc = False

        S = num_samples
        B = self.discriminator_inference_batch_size


        ###################
        # prepare the data

        num_pts = data_cfg.num_pts
        num_objects = len(xyzrgbs)

        obj_xyzs = []
        obj_rgbs = []
        for xyzrgb in xyzrgbs:
            if len(xyzrgb) <= num_pts:
                xyz_idxs = np.random.choice(range(len(xyzrgb)), num_pts, replace=True)
            else:
                xyz_idxs = np.random.choice(range(len(xyzrgb)), num_pts, replace=False)
            obj_xyz = xyzrgb[xyz_idxs, :3]
            obj_rgb = xyzrgb[xyz_idxs, 3:]
            obj_xyzs.append(torch.FloatTensor(obj_xyz))
            obj_rgbs.append(torch.FloatTensor(obj_rgb))

        structure_parameters = {"type": shape,
                                "rotation": [0, 0, rotation],
                                "position": [position_x, position_y]}
        if shape == "circle":
            structure_parameters["radius"] = radius
        if shape == "line":
            structure_parameters["length"] = radius * 2.0

        sample_raw_data = prior_dataset.prepare_test_data(obj_xyzs, obj_rgbs, other_obj_xyzs=[], other_obj_rgbs=[],
                                        structure_parameters=structure_parameters)




        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # sample_data = prior_dataset.convert_to_tensors(sample_raw_data, prior_dataset.tokenizer)
        # # struct_preds, obj_preds = prior_inference.limited_batch_inference([sample_raw_data], verbose=False,
        # #                                                                   return_numpy=False)
        # # struct_preds: B, 12
        # # obj_preds: B, N, 12
        #
        # beam_data = [sample_raw_data]
        # B = 1
        # num_target_objects = num_objects
        # # first predict structure pose
        # beam_goal_struct_pose, target_object_preds = prior_inference.limited_batch_inference(beam_data)
        # for b in range(B):
        #     datum = beam_data[b]
        #     datum["struct_x_inputs"] = [beam_goal_struct_pose[b][0]]
        #     datum["struct_y_inputs"] = [beam_goal_struct_pose[b][1]]
        #     datum["struct_z_inputs"] = [beam_goal_struct_pose[b][2]]
        #     datum["struct_theta_inputs"] = [beam_goal_struct_pose[b][3:]]
        #
        # # then iteratively predict pose of each object
        # beam_goal_obj_poses = []
        # for obj_idx in range(prior_dataset.max_num_objects):
        #     struct_preds, target_object_preds = prior_inference.limited_batch_inference(beam_data)
        #     beam_goal_obj_poses.append(target_object_preds[:, obj_idx])
        #     for b in range(B):
        #         datum = beam_data[b]
        #         datum["obj_x_inputs"][obj_idx] = target_object_preds[b][obj_idx][0]
        #         datum["obj_y_inputs"][obj_idx] = target_object_preds[b][obj_idx][1]
        #         datum["obj_z_inputs"][obj_idx] = target_object_preds[b][obj_idx][2]
        #         datum["obj_theta_inputs"][obj_idx] = target_object_preds[b][obj_idx][3:]
        # # concat in the object dim
        # beam_goal_obj_poses = np.stack(beam_goal_obj_poses, axis=0)
        # # swap axis
        # beam_goal_obj_poses = np.swapaxes(beam_goal_obj_poses, 1, 0)  # batch size, number of target objects, pose dim
        #
        # struct_preds = torch.FloatTensor(beam_goal_struct_pose)
        # obj_preds = torch.FloatTensor(beam_goal_obj_poses)
        #
        # print(struct_preds.shape)
        # print(obj_preds.shape)
        #
        #
        #
        #
        # # #*************************************************************
        # # # test obj poses
        # # #*************************************************************
        # # import trimesh
        # # def load_object_mesh_from_object_info(assets_dir, object_urdf):
        # #     mesh_path = os.path.join(assets_dir, "visual", object_urdf[:-5] + ".obj")
        # #     object_visual_mesh = trimesh.load(mesh_path)
        # #     return object_visual_mesh
        # #
        # # goal_specification = sample_raw_data["goal_specification"]
        # # obj_xyzs = sample_raw_data["xyzs"]
        # # current_obj_poses = sample_raw_data["current_obj_poses"]
        # # target_objs = sample_raw_data["target_objs"]
        # #
        # # target_obj_urdfs = [obj_spec["urdf"] for obj_spec in goal_specification["rearrange"]["objects"]]
        # # structure_parameters = goal_specification["shape"]
        # # if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
        # #     target_obj_urdfs = target_obj_urdfs[::-1]
        # # elif structure_parameters["type"] == "tower" or structure_parameters["type"] == "dinner":
        # #     target_obj_urdfs = target_obj_urdfs
        # # else:
        # #     raise KeyError("{} structure is not recognized".format(structure_parameters["type"]))
        # #
        # # target_obj_vis_meshes = [load_object_mesh_from_object_info(object_model_dir, u) for u in target_obj_urdfs]
        # # for i, obj_name in enumerate(target_objs):
        # #     current_obj_pose = current_obj_poses[i]
        # #     target_obj_vis_meshes[i].apply_transform(current_obj_pose)
        # #
        # # obj_pcs_vis = [trimesh.PointCloud(pc_obj[:, :3], colors=[255, 0, 0, 255]) for pc_obj in obj_xyzs]
        # # scene = trimesh.Scene()
        # # # add the coordinate frame first
        # # geom = trimesh.creation.axis(0.01)
        # # scene.add_geometry(geom)
        # # table = trimesh.creation.box(extents=[1.0, 1.0, 0.02])
        # # table.apply_translation([0.5, 0, -0.01])
        # # table.visual.vertex_colors = [150, 111, 87, 125]
        # # scene.add_geometry(table)
        # #
        # # for obj_vis_mesh in target_obj_vis_meshes:
        # #     obj_vis_mesh.visual.vertex_colors = [50, 50, 50, 100]
        # #
        # # scene.add_geometry(target_obj_vis_meshes)
        # # scene.add_geometry(obj_pcs_vis)
        # # scene.show()
        # # # *************************************************************
        #
        # ####################################################
        # # obj_xyzs: N, P, 3
        # # obj_params: B, N, 6
        # # struct_pose: B x N, 4, 4
        # # current_pc_pose: B x N, 4, 4
        # # target_object_inds: 1, N
        #
        # S = 1
        #
        # # N, P, 3
        # obj_xyzs = sample_data["xyzs"].to(device)
        # print("obj_xyzs shape", obj_xyzs.shape)
        # N, P, _ = obj_xyzs.shape
        # print("B, N, P: {}, {}, {}".format(S, N, P))
        # if visualize:
        #     visualize_batch_pcs(obj_xyzs, 1, N, P)
        #
        # struct_pose = torch.eye(4).repeat(S, 1, 1).to(device)  # S, 4, 4
        # struct_pose[:, :3, :3] = struct_preds[:, 3:].reshape(-1, 3, 3)
        # struct_pose[:, :3, 3] = struct_preds[:, :3]
        # struct_pose = struct_pose.repeat_interleave(N, dim=0)  # S x N, 4, 4
        #
        # current_pc_pose = torch.eye(4).repeat(N, 1, 1).to(device)  # N, 4, 4
        # current_pc_pose[:, :3, 3] = torch.mean(obj_xyzs, dim=1)  # N, 4, 4
        # current_pc_pose = current_pc_pose.repeat(S, 1, 1)  # S x N, 4, 4
        #
        # obj_params = torch.zeros((S, N, 6)).to(device)
        # obj_preds = obj_preds.reshape(S, N, -1)  # S, N, 12
        # obj_params[:, :, :3] = obj_preds[:, :, :3]
        # obj_params[:, :, 3:] = tra3d.matrix_to_euler_angles(obj_preds[:, :, 3:].reshape(S, N, 3, 3), "XYZ")
        #
        # best_new_obj_xyzs, best_goal_pc_pose = move_pc(obj_xyzs, obj_params, struct_pose, current_pc_pose, device)
        #
        # if visualize:
        #     visualize_batch_pcs(best_new_obj_xyzs, S, N, P)
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
















        ####################################################
        # sample S predictions
        samples_data = [prior_dataset.convert_to_tensors(sample_raw_data, prior_dataset.tokenizer)] * S

        # struct_preds, obj_preds = prior_inference.limited_batch_inference(samples_data, verbose=False, convert_to_tensors=False, return_numpy=False)

        # important: iterative sampling
        beam_data = [sample_raw_data] * S
        # first predict structure pose
        beam_goal_struct_pose, target_object_preds = prior_inference.limited_batch_inference(beam_data)
        for b in range(S):
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
            for b in range(S):
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

        # struct_preds: S, 12
        # obj_preds: S, N, 12
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
        obj_xyzs = samples_data[0]["xyzs"].to(device)
        print("obj_xyzs shape", obj_xyzs.shape)

        # 1, N
        # object_pad_mask: padding location has 1
        object_pad_mask = samples_data[0]["object_pad_mask"].to(device).unsqueeze(0)
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

        struct_pose = torch.eye(4).repeat(S, 1, 1).to(device)  # S, 4, 4
        struct_pose[:, :3, :3] = struct_preds[:, 3:].reshape(-1, 3, 3)
        struct_pose[:, :3, 3] = struct_preds[:, :3]
        struct_pose = struct_pose.repeat_interleave(N, dim=0)  # S x N, 4, 4

        new_obj_xyzs = obj_xyzs.repeat(S, 1, 1, 1)  # S, N, P, 3
        current_pc_pose = torch.eye(4).repeat(S, N, 1, 1).to(device)  # S, N, 4, 4
        # print(torch.mean(obj_xyzs, dim=2).shape)
        current_pc_pose[:, :, :3, 3] = torch.mean(new_obj_xyzs, dim=2)  # S, N, 4, 4
        current_pc_pose = current_pc_pose.reshape(S * N, 4, 4)  # S x N, 4, 4

        # optimize xyzrpy
        obj_params = torch.zeros((S, N, 6)).to(device)
        obj_preds = obj_preds.reshape(S, N, -1)
        obj_params[:, :, :3] = obj_preds[:, :, :3]
        obj_params[:, :, 3:] = tra3d.matrix_to_euler_angles(obj_preds[:, :, 3:].reshape(S, N, 3, 3), "XYZ")  # S, N, 6

        new_obj_xyzs_before_cem, goal_pc_pose_before_cem = move_pc(obj_xyzs, obj_params, struct_pose,
                                                                   current_pc_pose,
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
                    no_intersection_scores[cur_batch_idxs_start:cur_batch_idxs_end] = 1 - torch.mean(
                        collision_scores,
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
                                                        sentence_disc.unsqueeze(0).repeat(cur_batch_size, 1).to(
                                                            device),
                                                        sentence_pad_mask_disc.unsqueeze(0).repeat(cur_batch_size,
                                                                                                   1).to(device),
                                                        position_index_dic.unsqueeze(0).repeat(cur_batch_size,
                                                                                               1).to(
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
        # elite_struct_poses = elite_struct_poses.reshape(num_elite * N, 4, 4)  # num_elite x N, 4, 4
        elite_scores = scores[sort_idx]
        print("elite scores:", elite_scores)

        ####################################################
        # visualize best samples
        elite_struct_poses1 = elite_struct_poses.reshape(num_elite * N, 4, 4)  # num_elite x N, 4, 4
        num_scene_pts = 4096 if discriminator_num_scene_pts is None else discriminator_num_scene_pts
        batch_current_pc_pose = current_pc_pose[0: num_elite * N]
        best_new_obj_xyzs, best_goal_pc_pose, best_subsampled_scene_xyz, _, _ = \
            move_pc_and_create_scene_new(obj_xyzs, elite_obj_params, elite_struct_poses1, batch_current_pc_pose,
                                         target_object_inds, device,
                                         return_scene_pts=True, num_scene_pts=num_scene_pts, normalize_pc=True)
        # if visualize:
        #     print("visualizing elite rearrangements ranked by collision model/discriminator")
        #     visualize_batch_pcs(best_new_obj_xyzs, num_elite, N, P, limit_B=num_elite)

        ####################################################

        # xyzs = obj_xyzs.repeat(num_elite, 1, 1, 1)  # xyzs: num_elite, N, P, 3
        # print(xyzs.shape)
        # struct_pose = elite_struct_poses[:, 0]  # struct_pose: num_elite, 1, 4, 4
        # flat_obj_params = elite_obj_params.reshape(num_elite * N, -1)
        # goal_pc_pose_in_struct = torch.eye(4).repeat(num_elite * N, 1, 1).to(device)
        # goal_pc_pose_in_struct[:, :3, :3] = tra3d.euler_angles_to_matrix(flat_obj_params[:, 3:], "XYZ")
        # goal_pc_pose_in_struct[:, :3, 3] = flat_obj_params[:, :3]  # num_elite x N, 4, 4
        # pc_poses_in_struct = goal_pc_pose_in_struct.reshape(num_elite, N, 4, 4)


        if self.save_dir is not None:
            now = datetime.datetime.now()
            datastr = now.strftime("%Y-%m-%d-%H-%M")
            lang = str(prior_dataset.tokenizer.convert_structure_params_to_natural_language(sample_raw_data["sentence"]))
            save_file_temp = os.path.join(self.save_dir, datastr + "_" + lang + "_{}.png")
        if self.visualize or self.save_dir is not None:
            print("visualizing elite rearrangements ranked by collision model/discriminator")
            # new_obj_xyzs = move_pc_and_create_scene(xyzs, struct_pose, pc_poses_in_struct)
            # visualize_batch_pcs(new_obj_xyzs, B, N, P, verbose=False, limit_B=num_samples)

            batch_new_obj_xyzs = best_new_obj_xyzs.cpu().numpy()[:, :, :, :3]
            for bi, new_obj_xyzs in enumerate(batch_new_obj_xyzs):
                # new_obj_xyzs: num_target_objs, P, 3
                vis_pcs = [trimesh.PointCloud(obj_xyz, colors=np.concatenate([obj_rgb, np.ones([P, 1]) * 255], axis=-1))
                           for obj_xyz, obj_rgb in zip(new_obj_xyzs, obj_rgbs)]

                scene = trimesh.Scene()
                # add the coordinate frame first
                geom = trimesh.creation.axis(0.01)
                # scene.add_geometry(geom)
                table = trimesh.creation.box(extents=[1.0, 1.0, 0.02])
                table.apply_translation([0.5, 0, -0.01])
                table.visual.vertex_colors = [150, 111, 87, 125]
                scene.add_geometry(table)
                # bounds = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
                bounds = trimesh.creation.icosphere(subdivisions=3, radius=3.1)
                bounds.apply_translation([0, 0, 0])
                bounds.visual.vertex_colors = [30, 30, 30, 30]
                # scene.add_geometry(bounds)
                scene.add_geometry(vis_pcs)
                # RT_4x4 = np.array([[-0.7147778097036409, -0.6987369263935487, 0.02931536200292423, 0.3434544782290732],
                #                    [-0.47073865286968597, 0.4496990781074231, -0.7590624874434035, 0.10599949513304896],
                #                    [0.5172018981497449, -0.5563608962206371, -0.6503574015161744, 5.32058832987803],
                #                    [0.0, 0.0, 0.0, 1.0]])
                RT_4x4 = np.array([[-0.005378542346186285, 0.9998161380851034, -0.018405469481106367, -0.00822956735846642],
                                   [0.9144495496588564, -0.0025306059632757057, -0.40469200283941076, -0.5900283926985573],
                                   [-0.40466417238365116, -0.019007526352692254, -0.9142678062422447, 1.636231273015809],
                                   [0.0, 0.0, 0.0, 1.0]])
                RT_4x4 = np.linalg.inv(RT_4x4)
                RT_4x4 = RT_4x4 @ np.diag([1, -1, -1, 1])
                scene.camera_transform = RT_4x4
                if self.visualize:
                    scene.show()
                if self.save_dir is not None:
                    png = scene.save_image(resolution=[1000, 1000],
                                           visible=True)
                    with open(save_file_temp.format(bi), 'wb') as f:
                        f.write(png)
                        f.close()
                    rospy.sleep(1.0)

        # obj_xyzs: B, N, P, 3
        # struct_pose: B, 1, 4, 4
        # pc_poses_in_struct: B, N, 4, 4
        # current_pc_poses, goal_pc_poses = compute_current_and_goal_pc_poses(xyzs, struct_pose, pc_poses_in_struct)  # B, N, 4, 4

        current_pc_poses = batch_current_pc_pose.reshape(num_elite, N, 4, 4)  # num_elite * N
        goal_pc_poses = best_goal_pc_pose  # B, N, 4, 4

        current_pc_poses = current_pc_poses[:, :num_objects, :, :].cpu().numpy()  # B, num of objs, 4, 4
        goal_pc_poses = goal_pc_poses[:, :num_objects, :, :].cpu().numpy()  # B, num of objs, 4, 4
        return current_pc_poses, goal_pc_poses


if __name__ == '__main__':

    try:
        rospy.init_node('semantic_rearrangement')
        RearrangementModelServer(model_dir="/home/weiyu/data_drive/models_0914/transformer_line/best_model",
                                 # discriminator_model_dir="/home/weiyu/data_drive/models_0914/discriminator_lan_local_shape_param/best_model", discriminator_score_weight=1.0,
                                 collision_model_dir="/home/weiyu/data_drive/models_0914/collision/best_model", collision_score_weight=1.0,
                                 num_samples=50, num_elite=10, discriminator_inference_batch_size=5,
                                 visualize=True) #, save_dir="/home/weiyu/Desktop/robot_pc_rearrangement/structformer_rearrangements")
        rospy.spin()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)