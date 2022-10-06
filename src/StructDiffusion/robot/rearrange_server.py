#! /usr/bin/env python
from __future__ import print_function

import sys
import rospy

import trimesh
import numpy as np

from geometry_msgs.msg import PoseArray, Transform, TransformStamped, Pose
from rail_manipulation_msgs.srv import RearrangeObjects, RearrangeObjectsResponse

import tf.transformations as tf_tra

import sensor_msgs.point_cloud2 as point_cloud2
import ctypes
import struct

import tqdm
import torch


from StructDiffusion.utils.torch_data import default_collate
from StructDiffusion.training.train_diffuser_v3_lang import load_model, get_diffusion_variables, extract, get_struct_objs_poses, move_pc_and_create_scene, visualize_batch_pcs, compute_current_and_goal_pc_poses
from StructDiffusion.data.dataset_v1_diffuser import SemanticArrangementDataset
import StructDiffusion.utils.transformations as tra


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

    def __init__(self, model_dir, num_samples=30, visualize=False):

        self.model_dir = model_dir
        self.num_samples = num_samples
        self.visualize = visualize

        self.init_model()

        # TODO: figure out how to resolve this name
        self.server = rospy.Service('rearrangement_model', RearrangeObjects, self.run_model)
        print("Rearrangement model server started")

    def init_model(self):
        self.cfg, self.tokenizer, self.model, self.noise_schedule, _, _, _ = load_model(self.model_dir)
        self.model.eval()
        self.device = self.cfg.device
        self.data_cfg = self.cfg.dataset

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

        model = self.model
        device = self.device
        data_cfg = self.data_cfg
        tokenizer = self.tokenizer
        noise_schedule = self.noise_schedule
        num_samples = self.num_samples

        ###################
        # prepare the data

        num_pts = data_cfg.num_pts
        max_num_objects = data_cfg.max_num_objects
        max_num_shape_parameters = data_cfg.max_num_shape_parameters
        max_num_other_objects = data_cfg.max_num_other_objects

        num_objects = len(xyzrgbs)

        # -------------------------
        obj_xyzs = []
        obj_rgbs = []
        obj_pad_mask = []
        for xyzrgb in xyzrgbs:
            if len(xyzrgb) <= num_pts:
                xyz_idxs = np.random.choice(range(len(xyzrgb)), num_pts, replace=True)
            else:
                xyz_idxs = np.random.choice(range(len(xyzrgb)), num_pts, replace=False)
            obj_xyz = xyzrgb[xyz_idxs, :3]
            obj_rgb = xyzrgb[xyz_idxs, 3:]
            obj_xyzs.append(torch.FloatTensor(obj_xyz))
            obj_rgbs.append(obj_rgb)
            obj_pad_mask.append(0)
        for i in range(max_num_objects - len(obj_xyzs)):
            obj_xyzs.append(torch.zeros([1024, 3], dtype=torch.float32))
            obj_pad_mask.append(1)

        # -------------------------
        obj_xyztheta_inputs = []
        struct_xyztheta_inputs = []
        struct_xyz_theta = [0] * 12
        struct_xyztheta_inputs.append(struct_xyz_theta)
        # objects that need to be rearranged
        for i in range(len(obj_xyzs)):
            obj_xyztheta_inputs.append([0] * 12)
        # paddings
        for i in range(max_num_objects - len(obj_xyzs)):
            obj_xyztheta_inputs.append([0] * 12)

        # -------------------------
        sentence = []
        sentence_pad_mask = []

        # DEBUG: hardcode for now
        sentence.append((shape, "shape"))
        sentence.append((rotation, "rotation"))
        sentence.append((position_x, "position_x"))
        sentence.append((position_y, "position_y"))
        sentence.append((radius, "radius"))
        for _ in range(5):
            sentence_pad_mask.append(0)

        # sentence.append(("PAD", None))
        # for _ in range(4):
        #     sentence_pad_mask.append(0)
        # sentence_pad_mask.append(1)

        # # structure parameters
        # # 5 parameters
        # structure_parameters = goal_specification["shape"]
        # if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
        #     sentence.append((structure_parameters["type"], "shape"))
        #     sentence.append((structure_parameters["rotation"][2], "rotation"))
        #     sentence.append((structure_parameters["position"][0], "position_x"))
        #     sentence.append((structure_parameters["position"][1], "position_y"))
        #     if structure_parameters["type"] == "circle":
        #         sentence.append((structure_parameters["radius"], "radius"))
        #     elif structure_parameters["type"] == "line":
        #         sentence.append((structure_parameters["length"] / 2.0, "radius"))
        #     for _ in range(5):
        #         sentence_pad_mask.append(0)
        # else:
        #     sentence.append((structure_parameters["type"], "shape"))
        #     sentence.append((structure_parameters["rotation"][2], "rotation"))
        #     sentence.append((structure_parameters["position"][0], "position_x"))
        #     sentence.append((structure_parameters["position"][1], "position_y"))
        #     for _ in range(4):
        #         sentence_pad_mask.append(0)
        #     sentence.append(("PAD", None))
        #     sentence_pad_mask.append(1)

        # -------------------------
        token_type_index = [0] * (max_num_shape_parameters) + [1] * (max_num_other_objects) + [2] * max_num_objects
        position_index = list(range(max_num_shape_parameters)) + list(range(max_num_other_objects)) + list(
            range(max_num_objects))

        struct_position_index = [0]
        struct_token_type_index = [3]
        struct_pad_mask = [0]

        datum = {
            "xyzs": obj_xyzs,
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
        }

        batch = {
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

        with torch.no_grad():

            # input
            xyzs = batch["xyzs"].to(device, non_blocking=True).unsqueeze(0)
            B, N, P, _ = xyzs.shape
            obj_xyztheta_inputs = batch["obj_xyztheta_inputs"].to(device, non_blocking=True).unsqueeze(0)
            struct_xyztheta_inputs = batch["struct_xyztheta_inputs"].to(device, non_blocking=True).unsqueeze(0)
            position_index = batch["position_index"].to(device, non_blocking=True).unsqueeze(0)
            struct_position_index = batch["struct_position_index"].to(device, non_blocking=True).unsqueeze(0)
            start_token = torch.zeros((B, 1), dtype=torch.long).to(device, non_blocking=True)
            object_pad_mask = batch["obj_pad_mask"].to(device, non_blocking=True).unsqueeze(0)
            struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True).unsqueeze(0)
            sentence = batch["sentence"].to(device, non_blocking=True).unsqueeze(0)
            token_type_index = batch["token_type_index"].to(device, non_blocking=True).unsqueeze(0)
            struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True).unsqueeze(0)
            sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True).unsqueeze(0)

            # repeat num_samples times for this scene
            assert B == 1
            xyzs = xyzs.repeat(num_samples, 1, 1, 1)
            obj_xyztheta_inputs = obj_xyztheta_inputs.repeat(num_samples, 1, 1)
            struct_xyztheta_inputs = struct_xyztheta_inputs.repeat(num_samples, 1, 1)
            position_index = position_index.repeat(num_samples, 1)
            struct_position_index = struct_position_index.repeat(num_samples, 1)
            start_token = start_token.repeat(num_samples, 1)
            object_pad_mask = object_pad_mask.repeat(num_samples, 1)
            struct_pad_mask = struct_pad_mask.repeat(num_samples, 1)
            sentence = sentence.repeat(num_samples, 1)
            token_type_index = token_type_index.repeat(num_samples, 1)
            struct_token_type_index = struct_token_type_index.repeat(num_samples, 1)
            sentence_pad_mask = sentence_pad_mask.repeat(num_samples, 1)
            B = num_samples

            # start diffusion
            x_gt = get_diffusion_variables(struct_xyztheta_inputs, obj_xyztheta_inputs)
            x = torch.randn_like(x_gt, device=device)
            xs = []
            for t_index in tqdm.tqdm(reversed(range(0, noise_schedule.timesteps)), desc='sampling loop time step',
                                     total=noise_schedule.timesteps):
                # for t_index in tqdm.tqdm(reversed(range(0, 1)), desc='sampling loop time step',total=1):

                # get noise params
                t = torch.full((B,), t_index, device=device, dtype=torch.long)
                betas_t = extract(noise_schedule.betas, t, x.shape)
                sqrt_one_minus_alphas_cumprod_t = extract(noise_schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
                sqrt_recip_alphas_t = extract(noise_schedule.sqrt_recip_alphas, t, x.shape)

                # predict noise
                struct_xyztheta_inputs = x[:, 0, :].unsqueeze(1)  # B, 1, 3 + 6
                obj_xyztheta_inputs = x[:, 1:, :]  # B, N, 3 + 6
                struct_xyztheta_outputs, obj_xyztheta_outputs = model.forward(t, xyzs, obj_xyztheta_inputs,
                                                                              struct_xyztheta_inputs, sentence,
                                                                              position_index, struct_position_index,
                                                                              token_type_index, struct_token_type_index,
                                                                              start_token,
                                                                              object_pad_mask, struct_pad_mask,
                                                                              sentence_pad_mask)

                predicted_noise = torch.cat([struct_xyztheta_outputs, obj_xyztheta_outputs], dim=1)

                # compute noisy x at t
                model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
                if t_index == 0:
                    x = model_mean
                else:
                    posterior_variance_t = extract(noise_schedule.posterior_variance, t, x.shape)
                    noise = torch.randn_like(x)
                    # Algorithm 2 line 4:
                    x = model_mean + torch.sqrt(posterior_variance_t) * noise

                xs.append(x)

        xs = list(reversed(xs))

        struct_pose, pc_poses_in_struct = get_struct_objs_poses(xs[0])

        if self.visualize:
            new_obj_xyzs = move_pc_and_create_scene(xyzs, struct_pose, pc_poses_in_struct)
            # visualize_batch_pcs(new_obj_xyzs, B, N, P, verbose=False, limit_B=num_samples)

            batch_new_obj_xyzs = new_obj_xyzs.cpu().numpy()
            for new_obj_xyzs in batch_new_obj_xyzs:
                # new_obj_xyzs: num_target_objs, P, 3
                vis_pcs = [trimesh.PointCloud(obj_xyz, colors=np.concatenate([obj_rgb, np.ones([P, 1]) * 255], axis=-1))
                           for obj_xyz, obj_rgb in zip(new_obj_xyzs, obj_rgbs)]

                scene = trimesh.Scene()
                # add the coordinate frame first
                geom = trimesh.creation.axis(0.01)
                scene.add_geometry(geom)
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
                # RT_4x4 = np.linalg.inv(RT_4x4)
                # RT_4x4 = RT_4x4 @ np.diag([1, -1, -1, 1])
                # scene.camera_transform = RT_4x4
                scene.show()

        current_pc_poses, goal_pc_poses = compute_current_and_goal_pc_poses(xyzs, struct_pose, pc_poses_in_struct)  # B, N, 4, 4
        current_pc_poses = current_pc_poses[:, :num_objects, :, :].cpu().numpy()  # B, num of objs, 4, 4
        goal_pc_poses = goal_pc_poses[:, :num_objects, :, :].cpu().numpy()  # B, num of objs, 4, 4
        return current_pc_poses, goal_pc_poses


if __name__ == '__main__':

    model_dir = "/home/weiyu/data_drive/models_0914/diffuser/model"
    num_samples = 30

    try:
        rospy.init_node('semantic_rearrangement')
        RearrangementModelServer(model_dir, num_samples, visualize=False)
        rospy.spin()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)