import cv2
import h5py
import numpy as np
import os
import trimesh
import torch
import json
from collections import defaultdict
import tqdm
import pickle
from random import shuffle

# Local imports
from StructDiffusion.utils.rearrangement import show_pcs, get_pts, array_to_tensor
from StructDiffusion.utils.pointnet import pc_normalize

import StructDiffusion.utils.brain2.camera as cam
import StructDiffusion.utils.brain2.image as img
import StructDiffusion.utils.transformations as tra


def load_pairwise_collision_data(h5_filename):

    fh = h5py.File(h5_filename, 'r')
    data_dict = {}
    data_dict["obj1_info"] = eval(fh["obj1_info"][()])
    data_dict["obj2_info"] = eval(fh["obj2_info"][()])
    data_dict["obj1_poses"] = fh["obj1_poses"][:]
    data_dict["obj2_poses"] = fh["obj2_poses"][:]
    data_dict["intersection_labels"] = fh["intersection_labels"][:]

    return data_dict


class PairwiseCollisionDataset(torch.utils.data.Dataset):

    def __init__(self, data_roots, index_roots, urdf_pc_idx_file, collision_data_dir, random_rotation=True, data_augmentation=False, num_pts=1024, debug=False, normalize_pc=True, num_scene_pts=2048):

        # load dictionary mapping from urdf to list of pc data, each sample is
        #   {"step_t": step_t, "obj": obj, "filename": filename}
        if urdf_pc_idx_file is not None:
            if not os.path.exists(urdf_pc_idx_file):
                self.urdf_to_pc_data = self.create_urdf_pc_idxs(urdf_pc_idx_file, data_roots, index_roots)
            else:
                with open(urdf_pc_idx_file, "rb") as fh:
                    self.urdf_to_pc_data = pickle.load(fh)
        else:
            print("WARNING: urdf_pc_idx_file is None")

        # build data index
        # each sample is a tuple of (collision filename, idx for the labels and poses)
        if collision_data_dir is not None:
            self.data_idxs = self.build_data_idxs(collision_data_dir)
        else:
            print("WARNING: collision_data_dir is None")

        self.num_pts = num_pts
        self.debug = debug
        self.normalize_pc = normalize_pc
        self.num_scene_pts = num_scene_pts
        self.random_rotation = random_rotation

        # Noise
        self.data_augmentation = data_augmentation
        # additive noise
        self.gp_rescale_factor_range = [12, 20]
        self.gaussian_scale_range = [0., 0.003]
        # multiplicative noise
        self.gamma_shape = 1000.
        self.gamma_scale = 0.001

    def build_data_idxs(self, collision_data_dir):
        print("Load collision data...")
        positive_data = []
        negative_data = []
        for filename in tqdm.tqdm(os.listdir(collision_data_dir)):
            if "h5" not in filename:
                continue
            h5_filename = os.path.join(collision_data_dir, filename)
            data_dict = load_pairwise_collision_data(h5_filename)
            obj1_urdf = data_dict["obj1_info"]["urdf"]
            obj2_urdf = data_dict["obj2_info"]["urdf"]
            if obj1_urdf not in self.urdf_to_pc_data:
                print("no pc data for urdf:", obj1_urdf)
                continue
            if obj2_urdf not in self.urdf_to_pc_data:
                print("no pc data for urdf:", obj2_urdf)
                continue
            for idx, l in enumerate(data_dict["intersection_labels"]):
                if l:
                    # intersection
                    positive_data.append((h5_filename, idx))
                else:
                    negative_data.append((h5_filename, idx))
        print("Num pairwise intersections:", len(positive_data))
        print("Num pairwise no intersections:", len(negative_data))

        if len(negative_data) != len(positive_data):
            min_len = min(len(negative_data), len(positive_data))
            positive_data = [positive_data[i] for i in np.random.permutation(len(positive_data))[:min_len]]
            negative_data = [negative_data[i] for i in np.random.permutation(len(negative_data))[:min_len]]
            print("after balancing")
            print("Num pairwise intersections:", len(positive_data))
            print("Num pairwise no intersections:", len(negative_data))

        return positive_data + negative_data

    def create_urdf_pc_idxs(self, urdf_pc_idx_file, data_roots, index_roots):
        print("Load pc data")
        # data_roots = []
        # index_roots = []
        # for shape, index in [("stacking", "index_10k"), ("circle", "index_10k"), ("line", "index_10k"),
        #                      ("dinner", "index_10k")]:
        #     data_roots.append("/home/weiyu/data_drive/data_new_objects/examples_{}_new_objects/result".format(shape))
        #     index_roots.append(index)

        arrangement_steps = []
        for split in ["train"]:
            for data_root, index_root in zip(data_roots, index_roots):
                arrangement_indices_file = os.path.join(data_root, index_root,"{}_arrangement_indices_file_all.txt".format(split))
                if os.path.exists(arrangement_indices_file):
                    with open(arrangement_indices_file, "r") as fh:
                        arrangement_steps.extend([(os.path.join(data_root, f[0]), f[1]) for f in eval(fh.readline().strip())])
                else:
                    print("{} does not exist".format(arrangement_indices_file))

        urdf_to_pc_data = defaultdict(list)
        for filename, step_t in tqdm.tqdm(arrangement_steps):
            h5 = h5py.File(filename, 'r')
            ids = self._get_ids(h5)
            # moved_objs = h5['moved_objs'][()].split(',')
            all_objs = sorted([o for o in ids.keys() if "object_" in o])
            goal_specification = json.loads(str(np.array(h5["goal_specification"])))
            obj_infos = goal_specification["rearrange"]["objects"] + goal_specification["anchor"]["objects"] + goal_specification["distract"]["objects"]
            for obj, obj_info in zip(all_objs, obj_infos):
                urdf_to_pc_data[obj_info["urdf"]].append({"step_t": step_t, "obj": obj, "filename": filename})

        with open(urdf_pc_idx_file, "wb") as fh:
            pickle.dump(urdf_to_pc_data, fh)

        return urdf_to_pc_data

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

    def get_obj_pc(self, h5, step_t, obj):
        scene = self._get_images(h5, step_t, ee=True)
        rgb, depth, seg, valid, xyz = scene

        # getting object point clouds
        ids = self._get_ids(h5)
        obj_mask = np.logical_and(seg == ids[obj], valid)
        if np.sum(obj_mask) <= 0:
            raise Exception
        ok, obj_xyz, obj_rgb, _ = get_pts(xyz, rgb, obj_mask, num_pts=self.num_pts, to_tensor=False)
        obj_pc_center = np.mean(obj_xyz, axis=0)
        obj_pose = h5[obj][step_t]

        obj_pc_pose = np.eye(4)
        obj_pc_pose[:3, 3] = obj_pc_center[:3]

        return obj_xyz, obj_rgb, obj_pc_pose, obj_pose

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        collision_filename, collision_idx = self.data_idxs[idx]
        collision_data_dict = load_pairwise_collision_data(collision_filename)

        obj1_urdf = collision_data_dict["obj1_info"]["urdf"]
        obj2_urdf = collision_data_dict["obj2_info"]["urdf"]

        # TODO: find a better way to sample pc data?
        obj1_pc_data = np.random.choice(self.urdf_to_pc_data[obj1_urdf])
        obj2_pc_data = np.random.choice(self.urdf_to_pc_data[obj2_urdf])

        obj1_xyz, obj1_rgb, obj1_pc_pose, obj1_pose = self.get_obj_pc(h5py.File(obj1_pc_data["filename"], "r"), obj1_pc_data["step_t"], obj1_pc_data["obj"])
        obj2_xyz, obj2_rgb, obj2_pc_pose, obj2_pose = self.get_obj_pc(h5py.File(obj2_pc_data["filename"], "r"), obj2_pc_data["step_t"], obj2_pc_data["obj"])

        obj1_c_pose = collision_data_dict["obj1_poses"][collision_idx]
        obj2_c_pose = collision_data_dict["obj2_poses"][collision_idx]
        label = collision_data_dict["intersection_labels"][collision_idx]

        obj1_transform = obj1_c_pose @ np.linalg.inv(obj1_pose)
        obj2_transform = obj2_c_pose @ np.linalg.inv(obj2_pose)
        obj1_c_xyz = trimesh.transform_points(obj1_xyz, obj1_transform)
        obj2_c_xyz = trimesh.transform_points(obj2_xyz, obj2_transform)

        if self.debug:
            show_pcs([obj1_c_xyz, obj2_c_xyz], [obj1_rgb, obj2_rgb], add_coordinate_frame=True)

        ###################################
        obj_xyzs = [obj1_c_xyz, obj2_c_xyz]
        shuffle(obj_xyzs)

        num_indicator = 2
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

        if self.random_rotation:
            scene_xyz[:, 0:3] = trimesh.transform_points(scene_xyz[:, 0:3], tra.euler_matrix(0, 0, np.random.uniform(low=0, high=2 * np.pi)))

        ###################################
        scene_xyz = array_to_tensor(scene_xyz)
        # convert to torch data
        label = bool(label)

        if self.debug:
            print("intersection:", label)
            show_pcs([scene_xyz[:, 0:3]], [np.tile(np.array([0, 1, 0], dtype=np.float), (scene_xyz.shape[0], 1))], add_coordinate_frame=True)

        datum = {
            "scene_xyz": scene_xyz,
            "is_circle": torch.FloatTensor([label]),
        }
        return datum

    @staticmethod
    def collate_fn(data):
        """
        :param data:
        :return:
        """

        batched_data_dict = {}
        for key in ["is_circle"]:
            batched_data_dict[key] = torch.cat([dict[key] for dict in data], dim=0)
        for key in ["scene_xyz"]:
            batched_data_dict[key] = torch.stack([dict[key] for dict in data], dim=0)

        return batched_data_dict

    # def create_pair_xyzs_from_obj_xyzs(self, new_obj_xyzs, debug=False):
    #
    #     new_obj_xyzs = [xyz.cpu().numpy() for xyz in new_obj_xyzs]
    #
    #     # compute pairwise collision
    #     scene_xyzs = []
    #     obj_xyz_pair_idxs = list(itertools.combinations(range(len(new_obj_xyzs)), 2))
    #
    #     for obj_xyz_pair_idx in obj_xyz_pair_idxs:
    #         obj_xyz_pair = [new_obj_xyzs[obj_xyz_pair_idx[0]], new_obj_xyzs[obj_xyz_pair_idx[1]]]
    #         num_indicator = 2
    #         obj_xyz_pair_ind = []
    #         for oi, obj_xyz in enumerate(obj_xyz_pair):
    #             obj_xyz = np.concatenate([obj_xyz, np.tile(np.eye(num_indicator)[oi], (obj_xyz.shape[0], 1))], axis=1)
    #             obj_xyz_pair_ind.append(obj_xyz)
    #         pair_scene_xyz = np.concatenate(obj_xyz_pair_ind, axis=0)
    #
    #         # subsampling and normalizing pc
    #         rand_idx = np.random.randint(0, pair_scene_xyz.shape[0], self.num_scene_pts)
    #         pair_scene_xyz = pair_scene_xyz[rand_idx]
    #         if self.normalize_pc:
    #             pair_scene_xyz[:, 0:3] = pc_normalize(pair_scene_xyz[:, 0:3])
    #
    #         scene_xyzs.append(array_to_tensor(pair_scene_xyz))
    #
    #     if debug:
    #         for scene_xyz in scene_xyzs:
    #             show_pcs([scene_xyz[:, 0:3]], [np.tile(np.array([0, 1, 0], dtype=np.float), (scene_xyz.shape[0], 1))],
    #                      add_coordinate_frame=True)
    #
    #     return scene_xyzs


if __name__ == "__main__":
    prep = PairwiseCollisionDataset(urdf_pc_idx_file="/home/weiyu/data_drive/pairwise_collision_data/urdf_pc_idx.pkl",
                      collision_data_dir="/home/weiyu/data_drive/pairwise_collision_data",
                      debug=True)

    for i in np.random.permutation(len(prep)):
        print(i)
        d = prep[i]
