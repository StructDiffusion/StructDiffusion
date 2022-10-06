# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
import open3d
import trimesh

import StructDiffusion.utils.transformations as tra
from StructDiffusion.utils.brain2.pose import make_pose


def get_camera_from_h5(h5):
    """ Simple reference to help make these """
    proj_near = h5['cam_near'][()]
    proj_far = h5['cam_far'][()]
    proj_fov = h5['cam_fov'][()]
    width = h5['cam_width'][()]
    height = h5['cam_height'][()]
    return GenericCameraReference(proj_near, proj_far, proj_fov, width, height)


class GenericCameraReference(object):
    """ Class storing camera information and providing easy image capture """

    def __init__(self, proj_near=0.01, proj_far=5., proj_fov=60., img_width=640,
            img_height=480):

        self.proj_near = proj_near
        self.proj_far = proj_far
        self.proj_fov = proj_fov
        self.img_width = img_width
        self.img_height = img_height
        self.x_offset = self.img_width / 2.
        self.y_offset = self.img_height / 2.

        # Compute focal params
        aspect_ratio = self.img_width / self.img_height
        e = 1 / (np.tan(np.radians(self.proj_fov/2.)))
        t = self.proj_near / e
        b = -t
        r = t * aspect_ratio
        l = -r
        # pixels per meter
        alpha = self.img_width / (r-l)
        self.focal_length = self.proj_near * alpha 
        self.fx = self.focal_length
        self.fy = self.focal_length
        self.pose = None
        self.inv_pose = None

    def set_pose(self, trans, rot):
        self.pose = make_pose(trans, rot)
        self.inv_pose = tra.inverse_matrix(self.pose)

    def set_pose_matrix(self, matrix):
        self.pose = matrix
        self.inv_pose = tra.inverse_matrix(matrix)

    def transform_to_world_coords(self, xyz):
        """ transform xyz into world coordinates """
        #cam_pose = tra.inverse_matrix(self.pose).dot(tra.euler_matrix(np.pi, 0, 0))
        #xyz = trimesh.transform_points(xyz, self.inv_pose)
        #xyz = trimesh.transform_points(xyz, cam_pose)
        #pose = tra.euler_matrix(np.pi, 0, 0) @ self.pose
        pose = self.pose
        xyz = trimesh.transform_points(xyz, pose)
        return xyz

def get_camera_presets():
    return [
            "n/a",
            "azure_depth_nfov",
            "realsense",
            "azure_720p",
            "simple256",
            "simple512",
            ]


def get_camera_preset(name):

    if name == "azure_depth_nfov":
        # Setting for depth camera is pretty different from RGB
        height, width, fov = 576, 640, 75
    if name == "azure_720p":
        # This is actually the 720p RGB setting
        # Used for our color camera most of the time
        #height, width, fov = 720, 1280, 90
        height, width, fov = 720, 1280, 60
    elif name == "realsense":
        height, width, fov = 480, 640, 60
    elif name == "simple256":
        height, width, fov = 256, 256, 60
    elif name == "simple512":
        height, width, fov = 512, 512, 60
    else:
        raise RuntimeError(('camera "%s" not supported, choose from: ' +
            str(get_camera_presets())) % str(name))
    return height, width, fov


def get_generic_camera(name):
    h, w, fov = get_camera_preset(name)
    return GenericCameraReference(img_height=h, img_width=w, proj_fov=fov)


def get_matrix_of_indices(height, width):
    """ Get indices """
    return np.indices((height, width), dtype=np.float32).transpose(1,2,0)

# --------------------------------------------------------
# NOTE: this code taken from Arsalan and modified
def compute_xyz(depth_img, camera, visualize_xyz=False,
        xmap=None, ymap=None, max_clip_depth=5):
    """ Compute xyz image from depth for a camera """

    # We need thes eparameters
    height = camera.img_height
    width = camera.img_width
    assert depth_img.shape[0] == camera.img_height
    assert depth_img.shape[1] == camera.img_width
    fx = camera.fx
    fy = camera.fy
    cx = camera.x_offset
    cy = camera.y_offset

    """
    # Create the matrix of parameters
    indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    # indices[..., 0] = np.flipud(indices[..., 0])
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    """

    height = depth_img.shape[0]
    width = depth_img.shape[1]
    input_x = np.arange(width)
    input_y = np.arange(height)
    input_x, input_y = np.meshgrid(input_x, input_y)
    input_x = input_x.flatten()
    input_y = input_y.flatten()
    input_z = depth_img.flatten()
    # clip points that are farther than max distance
    input_z[input_z > max_clip_depth] = 0
    output_x = (input_x * input_z - cx * input_z) / fx
    output_y = (input_y * input_z - cy * input_z) / fy
    raw_pc = np.stack([output_x, output_y, input_z], -1).reshape(
        height, width, 3
    )
    return raw_pc

    if visualize_xyz:
        unordered_pc = xyz_img.reshape(-1, 3)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(unordered_pc) 
        pcd.transform([[1,0,0,0], [0,1,0,0], [0,0,-1,0], [0,0,0,1]]) # Transform it so it's not upside down
        open3d.visualization.draw_geometries([pcd])
        
    return xyz_img

def show_pcs(xyz, rgb):
    """ Display point clouds """
    if len(xyz.shape) > 2:
        unordered_pc = xyz.reshape(-1, 3)
        unordered_rgb = rgb.reshape(-1, 3) / 255.
    assert(unordered_rgb.shape[0] == unordered_pc.shape[0])
    assert(unordered_pc.shape[1] == 3)
    assert(unordered_rgb.shape[1] == 3)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(unordered_pc)
    pcd.colors = open3d.utility.Vector3dVector(unordered_rgb)
    pcd.transform([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]]) # Transform it so it's not upside down
    open3d.visualization.draw_geometries([pcd])
