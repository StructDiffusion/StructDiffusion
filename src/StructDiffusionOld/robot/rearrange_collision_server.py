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


from StructDiffusion.utils.torch_data import default_collate
from StructDiffusion.training.train_diffuser_v3_lang import load_model, get_diffusion_variables, extract, get_struct_objs_poses, move_pc_and_create_scene, visualize_batch_pcs, compute_current_and_goal_pc_poses
from StructDiffusion.data.dataset_v1_diffuser import SemanticArrangementDataset
import StructDiffusion.utils.transformations as tra

from StructDiffusion.evaluation.infer_diffuser_v3_lang import DiffuserInference
from StructDiffusion.evaluation.infer_collision import CollisionInference
from StructDiffusion.evaluation.infer_discriminator import DiscriminatorInference
from StructDiffusion.utils.batch_inference import move_pc, move_pc_and_create_scene_new


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

        self.prior_inference = DiffuserInference(model_dir)
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

        sample_raw_data = {
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


        ####################################################
        # sample S predictions
        sample_tensor_data = prior_dataset.convert_to_tensors(sample_raw_data, prior_dataset.tokenizer, robot_mode=True)

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
            print("visualizing initial scene")
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

    model_dir = "/home/weiyu/data_drive/models_0914/diffuser/model"

    try:
        rospy.init_node('semantic_rearrangement')
        RearrangementModelServer(model_dir="/home/weiyu/data_drive/models_0914/diffuser/model",
                                 # discriminator_model_dir="/home/weiyu/data_drive/models_0914/discriminator_lan_local_shape_param/best_model", discriminator_score_weight=1.0,
                                 collision_model_dir="/home/weiyu/data_drive/models_0914/collision/best_model", collision_score_weight=1.0,
                                 num_samples=50, num_elite=10, discriminator_inference_batch_size=5,
                                 visualize=True) #, save_dir="/home/weiyu/Desktop/robot_pc_rearrangement/pc_rearrangements")
        rospy.spin()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)