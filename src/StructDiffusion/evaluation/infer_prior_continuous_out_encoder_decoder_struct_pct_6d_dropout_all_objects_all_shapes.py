import torch
import numpy as np
import os
import copy
import argparse

from torch.utils.data import DataLoader

from StructDiffusion.utils.rearrangement import show_pcs, save_pcs
from StructDiffusion.data.tokenizer import Tokenizer

import StructDiffusion.data.dataset_v23_continuous_out_ar_6d_all_objects_all_shapes as prior_dataset
import StructDiffusion.training.train_prior_continuous_out_encoder_decoder_struct_pct_6d_dropout_all_objects_all_shapes as prior_model


def infer_model_batch(model_dir, override_data_dirs=None, override_index_dirs=None):
    prior_inference = PriorInference(model_dir, data_split="test",
                                     override_data_dirs=override_data_dirs, override_index_dirs=override_index_dirs)
    prior_inference.validate()


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


# class DiscriminatorInference:
#
#     def __init__(self, model_dir, data_split="test"):
#
#         cfg, model, _, _, _ = discriminator_model.load_model(model_dir)
#
#         data_cfg = cfg.dataset
#         dataset = discriminator_dataset.SemanticArrangementDataset(data_cfg.dir, data_cfg.index_dir,
#                                                                    data_split,
#                                                                    num_random_negative_examples=data_cfg.num_random_negative_examples,
#                                                                    min_translation=data_cfg.min_translation,
#                                                                    max_translation=data_cfg.max_translation,
#                                                                    min_rotation=data_cfg.min_rotation,
#                                                                    max_rotation=data_cfg.max_rotation,
#                                                                    max_num_objects=data_cfg.max_num_objects,
#                                                                    num_pts=data_cfg.num_pts,
#                                                                    num_scene_pts=data_cfg.num_scene_pts,
#                                                                    oversample_positive=data_cfg.oversample_positive,
#                                                                    pertubation_mode=data_cfg.pertubation_mode)
#
#         self.cfg = cfg
#         self.model = model
#         self.cfg = cfg
#         self.dataset = dataset
#
#     def single_inference(self, datum, num_target_objects, verbose=False):
#         """
#         This function makes the assumption that scenes in the batch have the same number of objects that need to be
#         rearranged
#
#         :param datum: datum from the prior model
#         :return:
#         """
#
#         datum = self.dataset.convert_to_scene_xyz(datum["xyzs"][:num_target_objects], 1, debug=verbose)
#         tensor = self.dataset.convert_to_tensors(datum)
#         tensor = self.dataset.collate_fn([tensor])
#
#         discriminator_scores = discriminator_model.infer_once(self.model, tensor, self.cfg.device, verbose)
#         discriminator_score = discriminator_scores["is_circle"][0].detach().cpu().numpy()
#
#         return discriminator_score
#
#     def limited_batch_inference(self, data, num_target_objects, verbose=False):
#         """
#         Can only run one data point at a time
#
#         :param datum: datum from the prior model
#         :return:
#         """
#
#         data_size = len(data)
#         batch_size = self.cfg.dataset.batch_size
#         if verbose:
#             print("data size:", data_size)
#             print("batch size:", batch_size)
#
#         num_batches = int(data_size / batch_size)
#         if data_size % batch_size != 0:
#             num_batches += 1
#
#         scores = []
#         for b in range(num_batches):
#             if b + 1 == num_batches:
#                 # last batch
#                 batch = data[b * batch_size:]
#             else:
#                 batch = data[b * batch_size: (b + 1) * batch_size]
#
#             batch_data = [self.dataset.convert_to_scene_xyz(d["xyzs"][:num_target_objects], 1, debug=verbose) for d in batch]
#             batch_tensors = [self.dataset.convert_to_tensors(d) for d in batch_data]
#             batch_tensors = self.dataset.collate_fn(batch_tensors)
#             batch_scores = discriminator_model.infer_once(self.model, batch_tensors, self.cfg.device, verbose)
#             scores.append(batch_scores["is_circle"])
#
#         scores = torch.cat(scores, dim=0)
#         scores = scores.detach().cpu().numpy()
#
#         return scores
#
#
# class Visualizer:
#
#     def __init__(self, inference_visualization_dir=None, cem_visualization_dir=None,
#                  visualize_action_sequence=False, max_beam_size=1000,):
#
#         self.inference_visualization_dir = inference_visualization_dir
#         self.cem_visualization_dir = cem_visualization_dir
#         self.visualize_action_sequence = visualize_action_sequence
#         self.max_beam_size = max_beam_size
#
#         for vdir in [inference_visualization_dir, cem_visualization_dir]:
#             if vdir:
#                 if not os.path.exists(vdir):
#                     os.makedirs(vdir)
#
#         self.images = {}
#
#     def _append_image(self, image, data_index, beam_index):
#
#         if data_index not in self.images:
#             self.images[data_index] = {}
#         if beam_index not in self.images[data_index]:
#             self.images[data_index][beam_index] = []
#
#         self.images[data_index][beam_index].append(image)
#
#     def append_datum_visualization(self, datum, data_index, beam_index, add_coordinate_frame=True, side_view=True):
#
#         if self.inference_visualization_dir is None:
#             return
#
#         if not self.visualize_action_sequence:
#             if data_index in self.images:
#                 if beam_index in self.images[data_index]:
#                     print("do not add more images since visualize_action_sequence is set to False")
#                     return
#
#         buffer = save_pcs(datum["xyzs"], datum["rgbs"],
#                           return_buffer=True, add_coordinate_frame=add_coordinate_frame, side_view=side_view)
#         # plt.imshow(np.asarray(buffer))
#         # convert to PIL format
#         image = np.uint8(np.asarray(buffer) * 255)
#         self._append_image(image, data_index, beam_index)
#
#     def append_beam_data_visualizations(self, beam_data, data_index, add_coordinate_frame=True, side_view=True):
#
#         if self.inference_visualization_dir is None:
#             return
#
#         if not self.visualize_action_sequence:
#             if data_index in self.images:
#                 return
#
#         for bi, datum in enumerate(beam_data):
#             buffer = save_pcs(datum["xyzs"], datum["rgbs"],
#                               return_buffer=True, add_coordinate_frame=add_coordinate_frame, side_view=side_view)
#             # plt.imshow(np.asarray(buffer))
#             # convert to PIL format
#             image = np.uint8(np.asarray(buffer) * 255)
#             self._append_image(image, data_index, bi)
#
#     def append_datum_to_beam_visualizations(self, datum, beam_data, data_index, add_coordinate_frame=True, side_view=True):
#
#         if self.inference_visualization_dir is None:
#             return
#
#         if not self.visualize_action_sequence:
#             if data_index in self.images:
#                 return
#
#         buffer = save_pcs(datum["xyzs"], datum["rgbs"],
#                           return_buffer=True, add_coordinate_frame=add_coordinate_frame, side_view=side_view)
#         # plt.imshow(np.asarray(buffer))
#         # convert to PIL format
#         image = np.uint8(np.asarray(buffer) * 255)
#         for bi in range(len(beam_data)):
#             self._append_image(image, data_index, bi)
#
#
# class PredictionWriter:
#
#     def __init__(self, prediction_dir=None):
#         self.predictions = {}
#         self.prediction_dir = prediction_dir
#
#         if prediction_dir is not None:
#             if not os.path.exists(prediction_dir):
#                 os.makedirs(prediction_dir)
#
#     def append_prediction(self, prediction, data_index, beam_index):
#
#         if data_index not in self.predictions:
#             self.predictions[data_index] = {}
#         if beam_index not in self.predictions[data_index]:
#             self.predictions[data_index][beam_index] = []
#
#         self.predictions[data_index][beam_index].append(prediction)
#
#     def convert_for_cem(self):
#         pass
#
#     def convert_from_pc_frame_to_object_frame(self):
#         pass
#
#     def delete_data_index(self, data_index):
#         # to save space
#         pass
#
#     def get_cem_predictions(self, data_index):
#         """
#         Prepare predictions for CEM.
#
#         :param data_index:
#         :return: predictions: a list of arrays, where each array has length (num_target_objects + 1) x num_gaussians.
#                               The number of gaussians is 6, 3 for xyz, 3 for rpy.
#         """
#
#         predictions = []
#         for beam_index in self.predictions[data_index]:
#             beam_prediction = []
#             for predicted_parameters in self.predictions[data_index][beam_index]:
#                 pos = predicted_parameters[:3]
#                 ax, ay, az = tra.euler_from_matrix(np.asarray(predicted_parameters[3:]).reshape(3, 3))
#                 beam_prediction.append(pos + [ax, ay, az])
#
#             beam_prediction = np.array(beam_prediction).flatten()
#             predictions.append(beam_prediction)
#         return predictions
#
#     def get_object_goal_pose_predictions(self, data_index, initial_scene_datum, num_target_objects,
#                                          euler_angles=False, debug=False, write_to_file=False):
#
#         # data necessary to compute predicted goal object pose
#         goal_obj_poses = initial_scene_datum["goal_obj_poses"][:num_target_objects]
#         current_obj_poses = initial_scene_datum["current_obj_poses"][:num_target_objects]
#         current_pc_poses = []
#         for obj_xyz in initial_scene_datum["xyzs"][:num_target_objects]:
#             current_pc_center = torch.mean(obj_xyz, dim=0).numpy()
#             current_pc_pose = np.eye(4)
#             current_pc_pose[:3, 3] = current_pc_center
#             current_pc_poses.append(current_pc_pose)
#
#         beam_predicted_goal_obj_poses = []
#         for beam_index in self.predictions[data_index]:
#
#             # the first position is struct, the remaining are objects
#             predicted_parameters = self.predictions[data_index][beam_index]
#
#             struct_params = predicted_parameters[0]
#             struct_params = np.asanyarray(struct_params)
#
#             if debug:
#                 # Important: in order to compare structure parameter predictions for debugging, need to not zero the data
#                 print("%%%%%%%%%%%%%%%%%%")
#                 print("beam", beam_index, "struct")
#                 print("predict", struct_params)
#                 print("gt", initial_scene_datum["struct_x_inputs"], initial_scene_datum["struct_y_inputs"], initial_scene_datum["struct_z_inputs"], initial_scene_datum["struct_theta_inputs"])
#
#             if not euler_angles:
#                 structure_pose = np.eye(4)
#                 structure_pose[:3, 3] = struct_params[:3]
#                 structure_pose[:3, :3] = struct_params[3:].reshape(3, 3)
#             else:
#                 raise NotImplementedError
#
#             predicted_goal_obj_poses = []
#             objects_params = predicted_parameters[1:]
#             for oi, object_params in enumerate(objects_params):
#
#                 object_params = np.asanyarray(object_params)
#                 if not euler_angles:
#                     goal_pc_pose = np.eye(4)
#                     goal_pc_pose[:3, 3] = object_params[:3]
#                     goal_pc_pose[:3, :3] = object_params[3:].reshape(3, 3)
#                 else:
#                     raise NotImplementedError
#
#                 predicted_goal_obj_pose = structure_pose @ goal_pc_pose @ np.linalg.inv(current_pc_poses[oi]) @ current_obj_poses[oi]
#                 if debug:
#                     print("%%%%%%%%%%%%%%%%%%")
#                     print("beam", beam_index, "object", oi)
#                     print("predict", predicted_goal_obj_pose)
#                     print("gt", goal_obj_poses[oi])
#
#                 predicted_goal_obj_poses.append(predicted_goal_obj_pose)
#             beam_predicted_goal_obj_poses.append(predicted_goal_obj_poses)
#
#         prediction_data = {"beam_predicted_goal_obj_poses": beam_predicted_goal_obj_poses,
#                            "filename": initial_scene_datum["filename"],
#                            "target_objs": initial_scene_datum["target_objs"]}
#         if self.prediction_dir is not None and write_to_file:
#             prediction_file = os.path.join(self.prediction_dir, "prediction_{}.pkl".format(data_index))
#             with open(prediction_file, "wb") as fh:
#                 pickle.dump(prediction_data, fh)
#
#         return beam_predicted_goal_obj_poses
#
#
# def inference_beam_decoding(model_dir, discriminator_model_dir, perform_cem=False, inference_visualization_dir=None,
#                              cem_visualization_dir=None, prediction_writer_dir=None,
#                              visualize=True, verbose=True, max_scene_decodes=30000, side_view=False, beam_size=100,
#                             reject_self_collision=False, visualize_action_sequence=False):
#     """
#     This function decodes a scene with a single forward pass
#
#     :param model_dir:
#     :param discriminator_model_dir:
#     :param inference_visualization_dir:
#     :param visualize:
#     :param num_samples: number of MDN samples drawn, in this case it's also the number of rearrangements
#     :param keep_steps:
#     :param initial_scenes_only:
#     :param verbose:
#     :return:
#     """
#
#     visualizer = Visualizer(inference_visualization_dir, cem_visualization_dir, visualize_action_sequence=False)
#     prediction_writer = PredictionWriter(prediction_writer_dir)
#
#     prior_inference = PriorInference(model_dir)
#     if discriminator_model_dir is not None:
#         discrminator_inference = DiscriminatorInference(discriminator_model_dir)
#     else:
#         discrminator_inference = None
#
#     test_dataset = prior_inference.dataset
#
#     decoded_scene_count = 0
#     all_rearrange_images = []
#     with tqdm.tqdm(total=len(test_dataset)) as pbar:
#         for idx in np.random.choice(range(len(test_dataset)), len(test_dataset), replace=False):
#         # for idx in range(len(test_dataset)):
#
#             if decoded_scene_count == max_scene_decodes:
#                 break
#
#             filename = test_dataset.get_data_index(idx)
#             scene_id = os.path.split(filename)[1][4:-3]
#
#             decoded_scene_count += 1
#
#             ############################################
#             # retrieve data
#             beam_data = []
#             num_target_objects = None
#             for b in range(beam_size):
#                 datum = test_dataset.get_raw_data(idx, inference_mode=True, shuffle_object_index=False)
#
#                 # not necessary, but just to ensure no test leakage
#                 datum["struct_x_inputs"] = [0]
#                 datum["struct_y_inputs"] = [0]
#                 datum["struct_y_inputs"] = [0]
#                 datum["struct_theta_inputs"] = [[0] * 9]
#                 for obj_idx in range(len(datum["obj_x_inputs"])):
#                     datum["obj_x_inputs"][obj_idx] = 0
#                     datum["obj_y_inputs"][obj_idx] = 0
#                     datum["obj_z_inputs"][obj_idx] = 0
#                     datum["obj_theta_inputs"][obj_idx] = [0] * 9
#
#                 beam_data.append(datum)
#
#                 if num_target_objects is None:
#                     num_target_objects = np.sum(np.array(datum["object_pad_mask"]) == 0)
#
#                 # Important: setting to smallest structure
#                 # datum["sentence"] = modify_language(datum["sentence"], radius=0.5)
#
#             # this is used for CEM
#             initial_scene_datum = copy.deepcopy(beam_data[0])
#
#             if visualize:
#                 datum = beam_data[0]
#                 print("#"*50)
#                 print("sentence", datum["sentence"])
#                 print("num target objects", num_target_objects)
#                 show_pcs(datum["xyzs"] + datum["other_xyzs"], datum["rgbs"] + datum["other_rgbs"], add_coordinate_frame=True, side_view=side_view)
#                 # plt.figure()
#                 # plt.imshow(datum["rgb"])
#                 # plt.show()
#                 # plt.figure()
#                 # plt.imshow(datum["goal_rgb"])
#                 # plt.show()
#
#             visualizer.append_datum_to_beam_visualizations(beam_data[0], beam_data, scene_id,
#                                                            add_coordinate_frame=True, side_view=side_view)
#
#             ############################################
#             # iteratively decoding
#             struct_preds, target_object_preds = prior_inference.limited_batch_inference(beam_data, verbose)
#
#             for b in range(beam_size):
#                 datum = beam_data[b]
#                 datum["struct_x_inputs"] = [struct_preds[b][0]]
#                 datum["struct_y_inputs"] = [struct_preds[b][1]]
#                 datum["struct_z_inputs"] = [struct_preds[b][2]]
#                 datum["struct_theta_inputs"] = [struct_preds[b][3:]]
#             for obj_idx in range(num_target_objects):
#                 struct_preds, target_object_preds = prior_inference.limited_batch_inference(beam_data, verbose)
#                 for b in range(beam_size):
#                     datum = beam_data[b]
#                     datum["obj_x_inputs"][obj_idx] = target_object_preds[b][obj_idx][0]
#                     datum["obj_y_inputs"][obj_idx] = target_object_preds[b][obj_idx][1]
#                     datum["obj_z_inputs"][obj_idx] = target_object_preds[b][obj_idx][2]
#                     datum["obj_theta_inputs"][obj_idx] = target_object_preds[b][obj_idx][3:]
#
#             ############################################
#             # move pc
#             for b in range(beam_size):
#                 datum = beam_data[b]
#                 structure_pos = [datum["struct_x_inputs"][0], datum["struct_y_inputs"][0], datum["struct_z_inputs"][0]]
#                 structure_rot = [v for v in datum["struct_theta_inputs"][0]]
#                 structure_params = structure_pos + structure_rot
#                 prediction_writer.append_prediction(structure_params, scene_id, b)
#
#                 for obj_idx in range(num_target_objects):
#                     object_pos = [datum["obj_x_inputs"][obj_idx], datum["obj_y_inputs"][obj_idx], datum["obj_z_inputs"][obj_idx]]
#                     object_rot = [v for v in datum["obj_theta_inputs"][obj_idx]]
#                     object_params = object_pos + object_rot
#                     prediction_writer.append_prediction(object_params, scene_id, b)
#
#                     imagined_obj_xyz, imagined_obj_rgb = move_one_object_pc(datum["xyzs"][obj_idx],
#                                                                             datum["rgbs"][obj_idx],
#                                                                             structure_params,
#                                                                             object_params)
#                     datum["xyzs"][obj_idx] = imagined_obj_xyz
#                     datum["rgbs"][obj_idx] = imagined_obj_rgb
#
#                     # visualize sequence
#                     if visualize_action_sequence:
#                         # show_pcs(datum["xyzs"] + datum["other_xyzs"], datum["rgbs"] + datum["other_rgbs"], add_coordinate_frame=True, side_view=side_view)
#                         # show rearranged object in green
#                         visualization_rgbs = copy.deepcopy(datum["rgbs"])
#                         visualization_rgbs[obj_idx] = np.tile(np.array([0, 1, 0], dtype=np.float), (imagined_obj_rgb.shape[0], 1))
#                         show_pcs(datum["xyzs"] + datum["other_xyzs"], visualization_rgbs + datum["other_rgbs"], add_coordinate_frame=True, side_view=side_view)
#
#                     visualizer.append_datum_visualization(datum, scene_id, b, add_coordinate_frame=True, side_view=True)
#
#                 # check collision
#                 # in_collision = check_pairwise_collision(datum["xyzs"], visualize=False)
#                 # randomly perturb the prediction
#                 # datum["xyzs"] = random_move_obj_xyzs(datum["xyzs"], min_translation=0.01, max_translation=0.03,
#                 #                                      min_rotation=0.10, max_rotation=0.20, mode="planar",
#                 #                                      visualize=False)
#
#             ###########################################
#             if visualize:
#                 for datum in beam_data[:10]:
#                     show_pcs(datum["xyzs"] + datum["other_xyzs"], datum["rgbs"] + datum["other_rgbs"], add_coordinate_frame=True)
#
#             # for i, datum in enumerate(beam_data[:10]):
#             #     visualization_rgbs = [np.tile(np.array([0, 1.0, 0], dtype=np.float), (d.shape[0], 1)) for d in datum["rgbs"]]
#             #     other_visualization_rgbs = [np.tile(np.array([0, 0, 1.0], dtype=np.float), (d.shape[0], 1)) for d in datum["other_rgbs"]]
#             #     # show_pcs(datum["xyzs"] + datum["other_xyzs"], visualization_rgbs + other_visualization_rgbs, add_coordinate_frame=True)
#             #     save_pcs(datum["xyzs"] + datum["other_xyzs"], visualization_rgbs + other_visualization_rgbs,
#             #              save_path="/home/weiyu/Research/intern/semantic-rearrangement/test/full/{}_{}.jpg".format(scene_id, i),
#             #              return_buffer=False,
#             #              add_coordinate_frame=False, side_view=False)
#
#             # we can (1) rank samples, (2) reject structure with self-collision, (3) perform cem, (4) perform metropolis
#
#             # # score samples
#             # scores = discrminator_inference.limited_batch_inference(beam_data, num_target_objects)
#
#             if perform_cem and discrminator_inference:
#                 cem_predictions = prediction_writer.get_cem_predictions(scene_id)
#                 perform_cross_entropy_refinement(initial_scene_datum, beam_data, cem_predictions,
#                                                  num_target_objects,
#                                                  discrminator_inference,
#                                                  scene_id, cem_visualization_dir)
#
#             if reject_self_collision:
#                 good_datum = None
#                 for datum in beam_data:
#                     in_collision = check_pairwise_collision(datum["xyzs"] + datum["other_xyzs"], visualize=False)
#                     if not in_collision:
#                         good_datum = datum
#                         break
#
#                 if good_datum is None:
#                     print("fail")
#                 else:
#                     show_pcs(good_datum["xyzs"] + good_datum["other_xyzs"], good_datum["rgbs"] + good_datum["other_rgbs"], add_coordinate_frame=True, side_view=True)
#
#
#             ###########################################
#             # after we have refined or rejected our rearrangement predictions, we want to make sure the action sequence
#             # if okay.
#             # collisions:
#             # if the object collides with a distractor object
#             # we want to move the distractor object to an empty space
#
#             # if the object collides with a rearranged object
#             # we want to skip this sample OR we want to refine?
#
#             # if the object collides with an object that will be rearranged
#             # we want to move that object but remember its position
#             # collisions = check_collision_with(datum["xyzs"][obj_idx],
#             #                                   [datum["xyzs"][ix] for ix in range(len(datum["xyzs"])) if ix != obj_idx],
#             #                                   visualize=True)
#             # # in_collision = check_pairwise_collision(datum["xyzs"], visualize=True)
#             # print("collisions:", collisions)
#
#             ###########################################
#             prediction_writer.get_object_goal_pose_predictions(scene_id, initial_scene_datum, num_target_objects,
#                                                                write_to_file=True)
#
#             pbar.update(1)
#
#
# def perform_cross_entropy_refinement(original_datum, data, predictions, num_target_objects,
#                                      discrminator_inference,
#                                      scene_id=None, inference_visualization_dir=None):
#
#     # ToDo: use visualizer
#
#     ###############################################
#     ce_num_iteration = 5
#     num_samples = 100
#     num_elite = 5
#     sigma_eps = 0.001 # noise
#     # pos_sigma_eps = 0.001
#     # rot_sigma_eps = 0.001
#     # sigma_eps = torch.FloatTensor(([pos_sigma_eps, pos_sigma_eps, pos_sigma_eps] + [rot_sigma_eps] * 3) * (num_target_objects + 1))
#     ###############################################
#
#     if inference_visualization_dir:
#         if not os.path.exists(inference_visualization_dir):
#             os.makedirs(inference_visualization_dir)
#
#     mus = None
#     sigmas = None
#
#     iteration_txts = []
#     iteration_imgs = []
#
#     for ce_iter in range(ce_num_iteration):
#         print("iteration ", ce_iter)
#
#         new_rearrangement_data = []
#         new_predictions = []
#         new_scores = []
#
#         if ce_iter == 0:
#             new_rearrangement_data = data
#             new_predictions = predictions  # each element in list: [num_target_objects + 1, number of individual gaussians]
#         else:
#             while len(new_rearrangement_data) < num_samples:
#
#                 datum = copy.deepcopy(original_datum)
#
#                 # predict x, y, theta based on mus and sigmas
#                 sample = sample_gaussians(mus, sigmas, sample_size=1)[0]
#                 sample = sample.detach().cpu().numpy()
#                 sample_correct_shape = sample.reshape(num_target_objects + 1, -1)  # num_target_objects + 1, number of individual gaussians
#
#                 structure_params = sample_correct_shape[0]
#                 for obj_idx in range(num_target_objects):
#                     object_params = sample_correct_shape[obj_idx + 1]
#                     imagined_obj_xyz, imagined_obj_rgb = move_one_object_pc(datum["xyzs"][obj_idx],
#                                                                             datum["rgbs"][obj_idx],
#                                                                             structure_params,
#                                                                             object_params,
#                                                                             euler_angles=True)
#                     datum["xyzs"][obj_idx] = imagined_obj_xyz
#                     datum["rgbs"][obj_idx] = imagined_obj_rgb
#
#                 new_rearrangement_data.append(datum)
#                 new_predictions.append(sample)
#
#         # compute cost
#         # for datum in new_rearrangement_data:
#         #     score = dicriminator_single_inference(datum, num_target_objects, ds_model, ds_test_dataset, ds_cfg)
#         #     new_scores.append(score)
#         new_scores = discrminator_inference.limited_batch_inference(new_rearrangement_data, num_target_objects)
#
#         # sort scored rearrangements
#         sort_idx = np.argsort(new_scores)[::-1][:num_elite]
#         new_predictions = [new_predictions[si] for si in sort_idx]
#         new_rearrangement_data = [new_rearrangement_data[si] for si in sort_idx]
#         new_scores = [new_scores[si] for si in sort_idx]
#
#         for datum, s in zip(new_rearrangement_data[::-1], new_scores[::-1]):
#             print("score:", s)
#             show_pcs(datum["xyzs"], datum["rgbs"], add_coordinate_frame=True)
#             if inference_visualization_dir:
#                 buffer = save_pcs(datum["xyzs"], datum["rgbs"],
#                                   return_buffer=True, add_coordinate_frame=True)
#                 iteration_imgs.append(np.uint8(np.asarray(buffer) * 255))
#                 iteration_txts.append("iteration {} score {}".format(ce_iter, s))
#
#         # use the top N to fit mus and sigmas
#         new_predictions = torch.FloatTensor(new_predictions)
#         # each dimension for each object as separate gaussians
#         mus, sigmas = fit_gaussians(new_predictions, sigma_eps)
#
#         print("mus", mus)
#         print("sigmas", sigmas)
#         # str_input = input("continue? y/n ")
#         # if str_input == "n":
#         #     break
#
#     if inference_visualization_dir:
#         make_gifs(iteration_imgs, os.path.join(inference_visualization_dir, "{}_cem.gif".format(scene_id)), texts=iteration_txts,
#                   numpy_img=True, duration=50)


def inference_beam_encoding_robot_simple(datum, beam_size, prior_inference,
                                         visualize=False, visualize_dir=None,
                                         bullet_interface=None, collision_distance=-0.01):
    # retrieve data
    beam_data = []
    beam_pc_rearrangements = []
    for b in range(beam_size):
        datum = copy.deepcopy(datum)

        # not necessary, but just to ensure no test leakage
        datum["struct_x_inputs"] = [0]
        datum["struct_y_inputs"] = [0]
        datum["struct_y_inputs"] = [0]
        datum["struct_theta_inputs"] = [[0] * 9]
        for obj_idx in range(len(datum["obj_x_inputs"])):
            datum["obj_x_inputs"][obj_idx] = 0
            datum["obj_y_inputs"][obj_idx] = 0
            datum["obj_z_inputs"][obj_idx] = 0
            datum["obj_theta_inputs"][obj_idx] = [0] * 9

        beam_data.append(datum)
        beam_pc_rearrangements.append(PointCloudRearrangement(datum))

    if visualize:
        datum = beam_data[0]
        print("#"*50)
        print("sentence", datum["sentence"])
        print("num target objects", beam_pc_rearrangements[0].num_target_objects)
        show_pcs(datum["xyzs"] + datum["other_xyzs"], datum["rgbs"] + datum["other_rgbs"], add_coordinate_frame=False, side_view=True, add_table=True)

    ############################################
    num_target_objects = beam_pc_rearrangements[0].num_target_objects
    # first predict structure pose
    beam_goal_struct_pose, target_object_preds = prior_inference.limited_batch_inference(beam_data)
    for b in range(beam_size):
        datum = beam_data[b]
        datum["struct_x_inputs"] = [beam_goal_struct_pose[b][0]]
        datum["struct_y_inputs"] = [beam_goal_struct_pose[b][1]]
        datum["struct_z_inputs"] = [beam_goal_struct_pose[b][2]]
        datum["struct_theta_inputs"] = [beam_goal_struct_pose[b][3:]]

    # then iteratively predict pose of each object
    beam_goal_obj_poses = []
    for obj_idx in range(num_target_objects):
        struct_preds, target_object_preds = prior_inference.limited_batch_inference(beam_data)
        beam_goal_obj_poses.append(target_object_preds[:, obj_idx])
        for b in range(beam_size):
            datum = beam_data[b]
            datum["obj_x_inputs"][obj_idx] = target_object_preds[b][obj_idx][0]
            datum["obj_y_inputs"][obj_idx] = target_object_preds[b][obj_idx][1]
            datum["obj_z_inputs"][obj_idx] = target_object_preds[b][obj_idx][2]
            datum["obj_theta_inputs"][obj_idx] = target_object_preds[b][obj_idx][3:]
    # concat in the object dim
    beam_goal_obj_poses = np.stack(beam_goal_obj_poses, axis=0)
    # swap axis
    beam_goal_obj_poses = np.swapaxes(beam_goal_obj_poses, 1, 0)  # batch size, number of target objects, pose dim

    ############################################
    # move pc
    for bi in range(beam_size):
        beam_pc_rearrangements[bi].set_goal_poses(beam_goal_struct_pose[bi], beam_goal_obj_poses[bi])
        beam_pc_rearrangements[bi].rearrange()

    ############################################
    if visualize:
        for pc_rearrangement in beam_pc_rearrangements[:10]:
            pc_rearrangement.visualize("goal", add_other_objects=False,
                                       add_coordinate_frame=True, side_view=False, add_table=False)
    if visualize_dir:
        for pci, pc_rearrangement in enumerate(beam_pc_rearrangements[:10]):
            pc_rearrangement.visualize("goal", add_other_objects=False,
                                       add_coordinate_frame=True, side_view=False, add_table=False,
                                       show_vis=False, save_vis=True,
                                       save_filename=os.path.join(visualize_dir, "{}_sample.png".format(pci)))

    found_pc_collision_free_arrangement, collision_free_beam_idxs = check_num_collisions(bullet_interface,
                                                                                         beam_pc_rearrangements,
                                                                                         collision_distance, visualize,
                                                                                         visualize_dir)

    # return
    perform_cem = False
    beam_scores = []

    beam_predicted_parameters = []
    for pcr in beam_pc_rearrangements:
        goal_struct_pose, goal_obj_poses = pcr.get_goal_poses(output_pose_format="xyz+3x3")
        beam_predicted_parameters.append([goal_struct_pose] + goal_obj_poses)
    output = {"beam_predicted_parameters": beam_predicted_parameters,
              "beam_pc_rearrangements": beam_pc_rearrangements,
              "performed_cem": perform_cem,
              "found_pc_collision_free_arrangement": found_pc_collision_free_arrangement,
              "collision_free_beam_idxs": collision_free_beam_idxs,
              "beam_scores": beam_scores}

    return output


def check_num_collisions(bullet_interface, beam_pc_rearrangements, collision_distance, visualize, visualize_dir, scores=None):
    found_pc_collision_free_arrangement = False
    collision_free_beam_idxs = []
    if bullet_interface is not None and len(beam_pc_rearrangements) > 0:
        beam_predicted_parameters = [pcr.get_goal_poses(output_pose_format="xyz+3x3", combine_struct_objs=True) for pcr in beam_pc_rearrangements]
        initial_obj_xyzs = beam_pc_rearrangements[0].initial_xyzs["xyzs"]

        collision_free_beam_idxs, beam_idx_to_collisions = bullet_interface.check_collision_for_beam_predictions(beam_predicted_parameters, initial_obj_xyzs,
                                                                                                                 collision_distance=collision_distance, use_structure=True)

        if collision_free_beam_idxs:
            print("found {} collision-free rearrangement out of {}".format(len(collision_free_beam_idxs), len(beam_pc_rearrangements)))
            found_pc_collision_free_arrangement = True

            # save bullet visualization
            if visualize_dir:
                for pci, pcr in enumerate([beam_pc_rearrangements[bi] for bi in collision_free_beam_idxs[:5]]):
                    pcr.bullet_visualize(bullet_interface, os.path.join(visualize_dir, "collision_free"), visualize_idx=pci)

            # visualization
            if visualize:
                if scores is None:
                    for count, bi in enumerate(collision_free_beam_idxs[:5]):
                        pcr = beam_pc_rearrangements[bi]
                        print("No.{} collision-free".format(count))
                        pcr.visualize("goal", add_other_objects=False,
                                      add_coordinate_frame=True, side_view=True, add_table=True,
                                      show_vis=True, save_vis=False)
                else:
                    collision_free_idx_score_tuples = [(bi, scores[bi]) for bi in collision_free_beam_idxs]
                    collision_free_idx_score_tuples = sorted(collision_free_idx_score_tuples, key=lambda x: x[1], reverse=True)

                    for count, (bi, score) in enumerate(collision_free_idx_score_tuples[:5]):
                        print("No.{} collision-free, score {}".format(count, score))
                        pcr = beam_pc_rearrangements[bi]
                        pcr.visualize("goal", add_other_objects=False,
                                      add_coordinate_frame=True, side_view=True, add_table=True,
                                      show_vis=True, save_vis=False)
        else:
            print("no collision free rearrangement found out of {}".format(len(beam_pc_rearrangements)))
            if visualize:
                for pci, pcr in enumerate(beam_pc_rearrangements[:5]):
                    if scores is not None:
                        print("No.{} collision, score {}".format(pci, scores[pci]))
                    else:
                        print("No.{} collision".format(pci))
                    pcr.visualize("goal", add_other_objects=False,
                                  add_coordinate_frame=True, side_view=True, add_table=True,
                                  show_vis=True, save_vis=False)

    return found_pc_collision_free_arrangement, collision_free_beam_idxs



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--model_dir", help='location for the saved model', type=str)
    args = parser.parse_args()

    infer_model_batch("/home/weiyu/data_drive/models_0914/transformer_circle/best_model",
                     override_data_dirs=["/home/weiyu/data_drive/data_test_objects/circle_data/result"], override_index_dirs=["index"])

    discriminator_model_dir = None
    # smaller margins larger range
    # discriminator_model_dir = "/home/weiyu/Research/intern/obj_pred/experiments/20210812-223451/best_model"
    # larger margins smaller range
    # discriminator_model_dir = "/home/weiyu/Research/intern/obj_pred/experiments/20210812-172916/best_model"

    # inference_beam_decoding(args.model_dir, discriminator_model_dir, perform_cem=False,
    #                         inference_visualization_dir=None,
    #                         prediction_writer_dir=None,
    #                         visualize=True, verbose=False, max_scene_decodes=30, beam_size=10, reject_self_collision=False)

    # inference_beam_decoding(args.model_dir, discriminator_model_dir, perform_cem=False,
    #                         inference_visualization_dir=None, prediction_writer_dir="/home/weiyu/Research/intern/semantic-rearrangement/inference_results/test_circle_new_objects",
    #                         visualize=True, verbose=False, max_scene_decodes=100, beam_size=100)

    # inference_batch_decoding(args.model_dir, discriminator_model_dir, inference_visualization_dir="/home/weiyu/Research/intern/obj_pred/inference_results_real_pct_6d_circle_dropout_shuffle_order",
    #                          visualize=False, verbose=False, max_scene_decodes=100)

    # inference_batch_decoding(args.model_dir, discriminator_model_dir, inference_visualization_dir=None,
    #                          cem_visualization_dir="/home/weiyu/Research/intern/obj_pred/inference_results_real_pct_6d_circle_cem",
    #                          visualize=False, verbose=False, max_scene_decodes=300)


