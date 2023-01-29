import copy
import os
import torch
import trimesh
import numpy as np
import open3d
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools
import matplotlib

import StructDiffusion.utils.transformations as tra
from StructDiffusion.utils.rotation_continuity import compute_geodesic_distance_from_two_matrices

# from pointnet_utils import farthest_point_sample, index_points


def flatten1d(img):
    return img.reshape(-1)


def flatten3d(img):
    hw = img.shape[0] * img.shape[1]
    return img.reshape(hw, -1)


def array_to_tensor(array):
    """ Assume arrays are in numpy (channels-last) format and put them into the right one """
    if array.ndim == 4: # NHWC
        tensor = torch.from_numpy(array).permute(0,3,1,2).float()
    elif array.ndim == 3: # HWC
        tensor = torch.from_numpy(array).permute(2,0,1).float()
    else: # everything else - just keep it as-is
        tensor = torch.from_numpy(array).float()
    return tensor


def get_pts(xyz_in, rgb_in, mask, bg_mask=None, num_pts=1024, center=None,
            radius=0.5, filename=None, to_tensor=True):

    # Get the XYZ and RGB
    mask = flatten1d(mask)
    assert(np.sum(mask) > 0)
    xyz = flatten3d(xyz_in)[mask > 0]
    if rgb_in is not None:
        rgb = flatten3d(rgb_in)[mask > 0]

    if xyz.shape[0] == 0:
        raise RuntimeError('this should not happen')
        ok = False
        xyz =  flatten3d(xyz_in)
        if rgb_in is not None:
            rgb = flatten3d(rgb_in)
    else:
        ok = True

    # prune to this region
    if center is not None:
        # numpy matrix
        # use the full xyz point cloud to determine what is close enough
        # now that we have the closest background point we can place the object on it
        # Just center on the point
        center = center.numpy()
        center = center[None].repeat(xyz.shape[0], axis=0)
        dists = np.linalg.norm(xyz - center, axis=-1)
        idx = dists < radius
        xyz = xyz[idx]
        if rgb_in is not None:
            rgb = rgb[idx]
        center = center[0]
    else:
        center = None

    # Compute number of points we are using
    if num_pts is not None:
        if xyz.shape[0] < 1:
            print("!!!! bad shape:", xyz.shape, filename, "!!!!")
            return (None, None, None, None)
        idx = np.random.randint(0, xyz.shape[0], num_pts)
        xyz = xyz[idx]
        if rgb_in is not None:
            rgb = rgb[idx]

    # Shuffle the points
    if rgb_in is not None:
        rgb = array_to_tensor(rgb) if to_tensor else rgb
    else:
        rgb = None
    xyz = array_to_tensor(xyz) if to_tensor else xyz
    return (ok, xyz, rgb, center)


def align(y_true, y_pred):
    """ Add or remove 2*pi to predicted angle to minimize difference from GT"""
    y_pred = y_pred.copy()
    y_pred[y_true - y_pred > np.pi] += np.pi * 2
    y_pred[y_true - y_pred < -np.pi] -= np.pi * 2
    return y_pred


def random_move_obj_xyz(obj_xyz,
                       min_translation, max_translation,
                       min_rotation, max_rotation, mode,
                       visualize=False, return_perturbed_obj_xyzs=True):

    assert mode in ["planar", "6d", "3d_planar"]

    if mode == "planar":
        random_translation = np.random.uniform(low=min_translation, high=max_translation, size=2) * np.random.choice(
            [-1, 1], size=2)
        random_rotation = np.random.uniform(low=min_rotation, high=max_rotation) * np.random.choice([-1, 1])
        random_rotation = tra.euler_matrix(0, 0, random_rotation)
    elif mode == "6d":
        random_rotation = np.random.uniform(low=min_rotation, high=max_rotation, size=3) * np.random.choice([-1, 1], size=3)
        random_rotation = tra.euler_matrix(*random_rotation)
        random_translation = np.random.uniform(low=min_translation, high=max_translation, size=3) * np.random.choice([-1, 1], size=3)
    elif mode == "3d_planar":
        random_translation = np.random.uniform(low=min_translation, high=max_translation, size=3) * np.random.choice(
            [-1, 1], size=3)
        random_rotation = np.random.uniform(low=min_rotation, high=max_rotation) * np.random.choice([-1, 1])
        random_rotation = tra.euler_matrix(0, 0, random_rotation)

    if return_perturbed_obj_xyzs:
        raise Exception("return_perturbed_obj_xyzs=True is no longer supported")
        # xyz_mean = np.mean(obj_xyz, axis=0)
        # new_obj_xyz = obj_xyz - xyz_mean
        # new_obj_xyz = trimesh.transform_points(new_obj_xyz, random_rotation, translate=False)
        # new_obj_xyz = new_obj_xyz + xyz_mean + random_translation
    else:
        new_obj_xyz = obj_xyz

    # test moving the perturbed obj pc back
    # new_xyz_mean = np.mean(new_obj_xyz, axis=0)
    # old_obj_xyz = new_obj_xyz - new_xyz_mean
    # old_obj_xyz = trimesh.transform_points(old_obj_xyz, np.linalg.inv(random_rotation), translate=False)
    # old_obj_xyz = old_obj_xyz + new_xyz_mean - random_translation

    # even though we are putting perturbation rotation and translation in the same matrix, they should be applied
    # independently. More specifically, rotate the object pc in place and then translate it.
    perturbation_matrix = random_rotation
    perturbation_matrix[:3, 3] = random_translation

    if visualize:
        show_pcs([new_obj_xyz, obj_xyz],
                 [np.tile(np.array([1, 0, 0], dtype=np.float), (obj_xyz.shape[0], 1)),
                  np.tile(np.array([0, 1, 0], dtype=np.float), (obj_xyz.shape[0], 1))], add_coordinate_frame=True)

    return new_obj_xyz, perturbation_matrix


def random_move_obj_xyzs(obj_xyzs,
                         min_translation, max_translation,
                         min_rotation, max_rotation, mode, move_obj_idxs=None, visualize=False, return_moved_obj_idxs=False,
                         return_perturbation=False, return_perturbed_obj_xyzs=True):
    """

    :param obj_xyzs:
    :param min_translation:
    :param max_translation:
    :param min_rotation:
    :param max_rotation:
    :param mode:
    :param move_obj_idxs:
    :param visualize:
    :param return_moved_obj_idxs:
    :param return_perturbation:
    :param return_perturbed_obj_xyzs:
    :return:
    """

    new_obj_xyzs = []
    new_obj_rgbs = []
    old_obj_rgbs = []
    perturbation_matrices = []

    if move_obj_idxs is None:
        move_obj_idxs = list(range(len(obj_xyzs)))

    # this many objects will not be randomly moved
    stationary_obj_idxs = np.random.choice(move_obj_idxs, np.random.randint(0, len(move_obj_idxs)), replace=False).tolist()

    moved_obj_idxs = []
    for obj_idx, obj_xyz in enumerate(obj_xyzs):

        if obj_idx in stationary_obj_idxs:
            new_obj_xyzs.append(obj_xyz)
            perturbation_matrices.append(np.eye(4))
            if visualize:
                new_obj_rgbs.append(np.tile(np.array([1, 0, 0], dtype=np.float), (obj_xyz.shape[0], 1)))
                old_obj_rgbs.append(np.tile(np.array([0, 0, 1], dtype=np.float), (obj_xyz.shape[0], 1)))
        else:
            new_obj_xyz, perturbation_matrix = random_move_obj_xyz(obj_xyz,
                                              min_translation=min_translation, max_translation=max_translation,
                                              min_rotation=min_rotation, max_rotation=max_rotation, mode=mode,
                                              return_perturbed_obj_xyzs=return_perturbed_obj_xyzs)
            new_obj_xyzs.append(new_obj_xyz)
            moved_obj_idxs.append(obj_idx)
            perturbation_matrices.append(perturbation_matrix)
            if visualize:
                new_obj_rgbs.append(np.tile(np.array([1, 0, 0], dtype=np.float), (obj_xyz.shape[0], 1)))
                old_obj_rgbs.append(np.tile(np.array([0, 1, 0], dtype=np.float), (obj_xyz.shape[0], 1)))
    if visualize:
        show_pcs(new_obj_xyzs + obj_xyzs,
                 new_obj_rgbs + old_obj_rgbs, add_coordinate_frame=True)

    if return_moved_obj_idxs:
        if return_perturbation:
            return new_obj_xyzs, moved_obj_idxs, perturbation_matrices
        else:
            return new_obj_xyzs, moved_obj_idxs
    else:
        if return_perturbation:
            return new_obj_xyzs, perturbation_matrices
        else:
            return new_obj_xyzs


def check_pairwise_collision(pcs, visualize=False):

    voxel_extents = [0.005] * 3

    collision_managers = []
    collision_objects = []

    for pc in pcs:

        # farthest point sample
        pc = pc.unsqueeze(0)
        fps_idx = farthest_point_sample(pc, 100)  # [B, npoint]
        pc = index_points(pc, fps_idx).squeeze(0)

        pc = np.asanyarray(pc)
        # ignore empty pc
        if np.all(pc == 0):
            continue

        n_points = pc.shape[0]
        collision_object = []
        collision_manager = trimesh.collision.CollisionManager()

        # Construct collision objects
        for i in range(n_points):
            extents = voxel_extents
            transform = np.eye(4)
            transform[:3, 3] = pc[i, :3]
            voxel = trimesh.primitives.Box(extents=extents, transform=transform)
            collision_object.append((voxel, extents, transform))

        # Add to collision manager
        for i, (voxel, _, _) in enumerate(collision_object):
            collision_manager.add_object("voxel_{}".format(i), voxel)

        collision_managers.append(collision_manager)
        collision_objects.append(collision_object)

    in_collision = False
    for i, cm_i in enumerate(collision_managers):
        for j, cm_j in enumerate(collision_managers):
            if i == j:
                continue
            if cm_i.in_collision_other(cm_j):
                in_collision = True

                if visualize:
                    visualize_collision_objects(collision_objects[i] + collision_objects[j])

                break

        if in_collision:
            break

    return in_collision


def check_collision_with(this_pc, other_pcs, visualize=False):

    voxel_extents = [0.005] * 3

    this_collision_manager = None
    this_collision_object = None
    other_collision_managers = []
    other_collision_objects = []

    for oi, pc in enumerate([this_pc] + other_pcs):

        # farthest point sample
        pc = pc.unsqueeze(0)
        fps_idx = farthest_point_sample(pc, 100)  # [B, npoint]
        pc = index_points(pc, fps_idx).squeeze(0)

        pc = np.asanyarray(pc)
        # ignore empty pc
        if np.all(pc == 0):
            continue

        n_points = pc.shape[0]
        collision_object = []
        collision_manager = trimesh.collision.CollisionManager()

        # Construct collision objects
        for i in range(n_points):
            extents = voxel_extents
            transform = np.eye(4)
            transform[:3, 3] = pc[i, :3]
            voxel = trimesh.primitives.Box(extents=extents, transform=transform)
            collision_object.append((voxel, extents, transform))

        # Add to collision manager
        for i, (voxel, _, _) in enumerate(collision_object):
            collision_manager.add_object("voxel_{}".format(i), voxel)

        if oi == 0:
            this_collision_manager = collision_manager
            this_collision_object = collision_object
        else:
            other_collision_managers.append(collision_manager)
            other_collision_objects.append(collision_object)

    collisions = []
    for i, cm_i in enumerate(other_collision_managers):
        if this_collision_manager.in_collision_other(cm_i):
            collisions.append(i)

            if visualize:
                visualize_collision_objects(this_collision_object + other_collision_objects[i])

    return collisions


def visualize_collision_objects(collision_objects):

    # Convert from trimesh to open3d
    meshes_o3d = []
    for elem in collision_objects:
        (voxel, extents, transform) = elem
        voxel_o3d = open3d.geometry.TriangleMesh.create_box(width=extents[0], height=extents[1],
                                                            depth=extents[2])
        voxel_o3d.compute_vertex_normals()
        voxel_o3d.paint_uniform_color([0.8, 0.2, 0])
        voxel_o3d.transform(transform)
        meshes_o3d.append(voxel_o3d)
    meshes = meshes_o3d

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    for mesh in meshes:
        vis.add_geometry(mesh)

    vis.run()
    vis.destroy_window()


# def test_collision(pc):
#     n_points = pc.shape[0]
#     voxel_extents = [0.005] * 3
#     collision_objects = []
#     collision_manager = trimesh.collision.CollisionManager()
#
#     # Construct collision objects
#     for i in range(n_points):
#         extents = voxel_extents
#         transform = np.eye(4)
#         transform[:3, 3] = pc[i, :3]
#         voxel = trimesh.primitives.Box(extents=extents, transform=transform)
#         collision_objects.append((voxel, extents, transform))
#
#     # Add to collision manager
#     for i, (voxel, _, _) in enumerate(collision_objects):
#         collision_manager.add_object("voxel_{}".format(i), voxel)
#
#     for i, (voxel, _, _) in enumerate(collision_objects):
#         c, names = collision_manager.in_collision_single(voxel, return_names=True)
#         if c:
#             print(i, names)
#
#     # Convert from trimesh to open3d
#     meshes_o3d = []
#     for elem in collision_objects:
#         (voxel, extents, transform) = elem
#         voxel_o3d = open3d.geometry.TriangleMesh.create_box(width=extents[0], height=extents[1],
#                                                             depth=extents[2])
#         voxel_o3d.compute_vertex_normals()
#         voxel_o3d.paint_uniform_color([0.8, 0.2, 0])
#         voxel_o3d.transform(transform)
#         meshes_o3d.append(voxel_o3d)
#     meshes = meshes_o3d
#
#     vis = open3d.visualization.Visualizer()
#     vis.create_window()
#
#     for mesh in meshes:
#         vis.add_geometry(mesh)
#
#     vis.run()
#     vis.destroy_window()
#
#
# def test_collision2(pc):
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pc)
#     pcd.estimate_normals()
#     open3d.visualization.draw_geometries([pcd])
#
#     # poisson_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
#     # bbox = pcd.get_axis_aligned_bounding_box()
#     # p_mesh_crop = poisson_mesh.crop(bbox)
#     # open3d.visualization.draw_geometries([p_mesh_crop, pcd])
#
#     distances = pcd.compute_nearest_neighbor_distance()
#     avg_dist = np.mean(distances)
#     radius = 3 * avg_dist
#     bpa_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, open3d.utility.DoubleVector(
#         [radius, radius * 2]))
#     dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)
#     dec_mesh.remove_degenerate_triangles()
#     dec_mesh.remove_duplicated_triangles()
#     dec_mesh.remove_duplicated_vertices()
#     dec_mesh.remove_non_manifold_edges()
#     open3d.visualization.draw_geometries([dec_mesh, pcd])
#     open3d.visualization.draw_geometries([dec_mesh])


def make_gifs(imgs, save_path, texts=None, numpy_img=True, duration=10):
    gif_filename = os.path.join(save_path)
    pil_imgs = []
    for i, img in enumerate(imgs):
        if numpy_img:
            img = Image.fromarray(img)
        if texts:
            text = texts[i]
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("FreeMono.ttf", 40)
            draw.text((0, 0), text, (120, 120, 120), font=font)
        pil_imgs.append(img)

    pil_imgs[0].save(gif_filename, save_all=True,
                     append_images=pil_imgs[1:], optimize=True,
                     duration=duration*len(pil_imgs), loop=0)


def save_img(img, save_path, text=None, numpy_img=True):
    if numpy_img:
        img = Image.fromarray(img)
    if text:
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("FreeMono.ttf", 40)
        draw.text((0, 0), text, (120, 120, 120), font=font)
    img.save(save_path)


def move_one_object_pc(obj_xyz, obj_rgb, struct_params, object_params, euler_angles=False):
    struct_params = np.asanyarray(struct_params)
    object_params = np.asanyarray(object_params)

    R_struct = np.eye(4)
    if not euler_angles:
        R_struct[:3, :3] = struct_params[3:].reshape(3, 3)
    else:
        R_struct[:3, :3] = tra.euler_matrix(*struct_params[3:])[:3, :3]
    R_obj = np.eye(4)
    if not euler_angles:
        R_obj[:3, :3] = object_params[3:].reshape(3, 3)
    else:
        R_obj[:3, :3] = tra.euler_matrix(*object_params[3:])[:3, :3]

    T_struct = R_struct
    T_struct[:3, 3] = [struct_params[0], struct_params[1], struct_params[2]]

    # translate to structure frame
    t = np.eye(4)
    obj_center = torch.mean(obj_xyz, dim=0)
    t[:3, 3] = [object_params[0] - obj_center[0], object_params[1] - obj_center[1], object_params[2] - obj_center[2]]
    new_obj_xyz = trimesh.transform_points(obj_xyz, t)

    # rotate in place
    R = R_obj
    obj_center = np.mean(new_obj_xyz, axis=0)
    centered_obj_xyz = new_obj_xyz - obj_center
    new_centered_obj_xyz = trimesh.transform_points(centered_obj_xyz, R, translate=True)
    new_obj_xyz = new_centered_obj_xyz + obj_center

    # transform to the global frame from the structure frame
    new_obj_xyz = trimesh.transform_points(new_obj_xyz, T_struct)

    # convert back to torch
    new_obj_xyz = torch.tensor(new_obj_xyz, dtype=obj_xyz.dtype)

    return new_obj_xyz, obj_rgb


def move_one_object_pc_no_struct(obj_xyz, obj_rgb, object_params, euler_angles=False):
    object_params = np.asanyarray(object_params)

    R_obj = np.eye(4)
    if not euler_angles:
        R_obj[:3, :3] = object_params[3:].reshape(3, 3)
    else:
        R_obj[:3, :3] = tra.euler_matrix(*object_params[3:])[:3, :3]

    t = np.eye(4)
    obj_center = torch.mean(obj_xyz, dim=0)
    t[:3, 3] = [object_params[0] - obj_center[0], object_params[1] - obj_center[1], object_params[2] - obj_center[2]]
    new_obj_xyz = trimesh.transform_points(obj_xyz, t)

    # rotate in place
    R = R_obj
    obj_center = np.mean(new_obj_xyz, axis=0)
    centered_obj_xyz = new_obj_xyz - obj_center
    new_centered_obj_xyz = trimesh.transform_points(centered_obj_xyz, R, translate=True)
    new_obj_xyz = new_centered_obj_xyz + obj_center

    # convert back to torch
    new_obj_xyz = torch.tensor(new_obj_xyz, dtype=obj_xyz.dtype)

    return new_obj_xyz, obj_rgb


def modify_language(sentence, radius=None, position_x=None, position_y=None, rotation=None, shape=None):
    # "radius": [0.0, 0.5, 3], "position_x": [-0.1, 1.0, 3], "position_y": [-0.5, 0.5, 3], "rotation": [-3.15, 3.15, 4]

    sentence = copy.deepcopy(sentence)
    for pi, pair in enumerate(sentence):
        if radius is not None and len(pair) == 2 and pair[1] == "radius":
            sentence[pi] = (radius, 'radius')
        if position_y is not None and len(pair) == 2 and pair[1] == "position_y":
            sentence[pi] = (position_y, 'position_y')
        if position_x is not None and len(pair) == 2 and pair[1] == "position_x":
            sentence[pi] = (position_x, 'position_x')
        if rotation is not None and len(pair) == 2 and pair[1] == "rotation":
            sentence[pi] = (rotation, 'rotation')
        if shape is not None and len(pair) == 2 and pair[1] == "shape":
            sentence[pi] = (shape, 'shape')

    return sentence


def sample_gaussians(mus, sigmas, sample_size):
    # mus: [number of individual gaussians]
    # sigmas: [number of individual gaussians]
    normal = torch.distributions.Normal(mus, sigmas)
    samples = normal.sample((sample_size,))
    # samples: [sample_size, number of individual gaussians]
    return samples


def fit_gaussians(samples, sigma_eps=0.01):
    # samples: [sample_size, number of individual gaussians]
    num_gs = samples.shape[1]
    mus = torch.mean(samples, dim=0)
    sigmas = torch.std(samples, dim=0) + sigma_eps * torch.ones(num_gs)
    # mus: [number of individual gaussians]
    # sigmas: [number of individual gaussians]
    return mus, sigmas


def show_pcs_with_predictions(xyz, rgb, gts, predictions, add_coordinate_frame=False, return_buffer=False, add_table=True, side_view=True):
    """ Display point clouds """

    assert len(gts) == len(predictions) == len(xyz) == len(rgb)

    unordered_pc = np.concatenate(xyz, axis=0)
    unordered_rgb = np.concatenate(rgb, axis=0)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(unordered_pc)
    pcd.colors = open3d.utility.Vector3dVector(unordered_rgb)

    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    if add_table:
        table_color = [0.7, 0.7, 0.7]
        origin = [0, -0.5, -0.05]
        table = open3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.02)
        table.paint_uniform_color(table_color)
        table.translate(origin)
        vis.add_geometry(table)

    if add_coordinate_frame:
        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(mesh_frame)

    for i in range(len(xyz)):
        pred_color = [0.0, 1.0, 0] if predictions[i] else [1.0, 0.0, 0]
        gt_color = [0.0, 1.0, 0] if gts[i] else [1.0, 0.0, 0]
        origin = torch.mean(xyz[i], dim=0)
        origin[2] += 0.02
        pred_vis = open3d.geometry.TriangleMesh.create_torus(torus_radius=0.02, tube_radius=0.01)
        pred_vis.paint_uniform_color(pred_color)
        pred_vis.translate(origin)
        gt_vis = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        gt_vis.paint_uniform_color(gt_color)
        gt_vis.translate(origin)
        vis.add_geometry(pred_vis)
        vis.add_geometry(gt_vis)

    if side_view:
        open3d_set_side_view(vis)

    if return_buffer:
        vis.poll_events()
        vis.update_renderer()
        buffer = vis.capture_screen_float_buffer(False)
        vis.destroy_window()
        return buffer
    else:
        vis.run()
        vis.destroy_window()


def show_pcs_with_only_predictions(xyz, rgb, gts, predictions, add_coordinate_frame=False, return_buffer=False, add_table=True, side_view=True):
    """ Display point clouds """

    assert len(gts) == len(predictions) == len(xyz) == len(rgb)

    unordered_pc = np.concatenate(xyz, axis=0)
    unordered_rgb = np.concatenate(rgb, axis=0)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(unordered_pc)
    pcd.colors = open3d.utility.Vector3dVector(unordered_rgb)

    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    if add_table:
        table_color = [0.7, 0.7, 0.7]
        origin = [0, -0.5, -0.05]
        table = open3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.02)
        table.paint_uniform_color(table_color)
        table.translate(origin)
        vis.add_geometry(table)

    if add_coordinate_frame:
        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(mesh_frame)

    for i in range(len(xyz)):
        pred_color = [0.0, 1.0, 0] if predictions[i] else [1.0, 0.0, 0]
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(xyz[i])
        pcd.colors = open3d.utility.Vector3dVector(np.tile(np.array(pred_color, dtype=np.float), (xyz[i].shape[0], 1)))
        # pcd = pcd.uniform_down_sample(10)
        # vis.add_geometry(pcd)

        obb = pcd.get_axis_aligned_bounding_box()
        obb.color = pred_color
        vis.add_geometry(obb)


        # origin = torch.mean(xyz[i], dim=0)
        # origin[2] += 0.02
        # pred_vis = open3d.geometry.TriangleMesh.create_torus(torus_radius=0.02, tube_radius=0.01)
        # pred_vis.paint_uniform_color(pred_color)
        # pred_vis.translate(origin)
        # gt_vis = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # gt_vis.paint_uniform_color(gt_color)
        # gt_vis.translate(origin)
        # vis.add_geometry(pred_vis)
        # vis.add_geometry(gt_vis)

    if side_view:
        open3d_set_side_view(vis)

    if return_buffer:
        vis.poll_events()
        vis.update_renderer()
        buffer = vis.capture_screen_float_buffer(False)
        vis.destroy_window()
        return buffer
    else:
        vis.run()
        vis.destroy_window()


def test_new_vis(xyz, rgb):
    pass
#     unordered_pc = np.concatenate(xyz, axis=0)
#     unordered_rgb = np.concatenate(rgb, axis=0)
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(unordered_pc)
#     pcd.colors = open3d.utility.Vector3dVector(unordered_rgb)
#
#     # Some platforms do not require OpenGL implementations to support wide lines,
#     # so the renderer requires a custom shader to implement this: "unlitLine".
#     # The line_width field is only used by this shader; all other shaders ignore
#     # it.
#     # mat = o3d.visualization.rendering.Material()
#     # mat.shader = "unlitLine"
#     # mat.line_width = 10  # note that this is scaled with respect to pixels,
#     # # so will give different results depending on the
#     # # scaling values of your system
#     # mat.transmission = 0.5
#     open3d.visualization.draw({
#         "name": "pcd",
#         "geometry": pcd,
#         # "material": mat
#     })
#
#     for i in range(len(xyz)):
#         pred_color = [0.0, 1.0, 0] if predictions[i] else [1.0, 0.0, 0]
#         pcd = open3d.geometry.PointCloud()
#         pcd.points = open3d.utility.Vector3dVector(xyz[i])
#         pcd.colors = open3d.utility.Vector3dVector(np.tile(np.array(pred_color, dtype=np.float), (xyz[i].shape[0], 1)))
#         # pcd = pcd.uniform_down_sample(10)
#         # vis.add_geometry(pcd)
#
#         obb = pcd.get_axis_aligned_bounding_box()
#         obb.color = pred_color
#         vis.add_geometry(obb)


def show_pcs(xyz, rgb, add_coordinate_frame=False, side_view=False, add_table=True):
    """ Display point clouds """

    unordered_pc = np.concatenate(xyz, axis=0)
    unordered_rgb = np.concatenate(rgb, axis=0)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(unordered_pc)
    pcd.colors = open3d.utility.Vector3dVector(unordered_rgb)

    if add_table:
        table_color = [0.78, 0.64, 0.44]
        origin = [0, -0.5, -0.02]
        table = open3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.001)
        table.paint_uniform_color(table_color)
        table.translate(origin)

    if not add_coordinate_frame:
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        if add_table:
            vis.add_geometry(table)
        if side_view:
            open3d_set_side_view(vis)
        vis.run()
        vis.destroy_window()
    else:
        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # open3d.visualization.draw_geometries([pcd, mesh_frame])
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.add_geometry(mesh_frame)
        if add_table:
            vis.add_geometry(table)
        if side_view:
            open3d_set_side_view(vis)
        vis.run()
        vis.destroy_window()


def show_pcs_color_order(xyzs, rgbs, add_coordinate_frame=False, side_view=False, add_table=True, save_path=None, texts=None, visualize=False):

    rgb_colors = get_rgb_colors()

    order_rgbs = []
    for i, xyz in enumerate(xyzs):
        order_rgbs.append(np.tile(np.array(rgb_colors[i][1], dtype=np.float), (xyz.shape[0], 1)))

    if visualize:
        show_pcs(xyzs, order_rgbs, add_coordinate_frame=add_coordinate_frame, side_view=side_view, add_table=add_table)
    if save_path:
        if not texts:
            save_pcs(xyzs, order_rgbs, save_path=save_path, add_coordinate_frame=add_coordinate_frame, side_view=side_view, add_table=add_table)
        if texts:
            buffer = save_pcs(xyzs, order_rgbs, add_coordinate_frame=add_coordinate_frame,
                     side_view=side_view, add_table=add_table, return_buffer=True)
            img = np.uint8(np.asarray(buffer) * 255)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("FreeMono.ttf", 20)
            for it, text in enumerate(texts):
                draw.text((0, it*20), text, (120, 120, 120), font=font)
            img.save(save_path)


def get_rgb_colors():
    rgb_colors = []
    # each color is a tuple of (name, (r,g,b))
    for name, hex in matplotlib.colors.cnames.items():
        rgb_colors.append((name, matplotlib.colors.to_rgb(hex)))

    rgb_colors = sorted(rgb_colors, key=lambda x: x[0])

    priority_colors = [('red', (1.0, 0.0, 0.0)),  ('green', (0.0, 1.0, 0.0)), ('blue', (0.0, 0.0, 1.0)),  ('orange', (1.0, 0.6470588235294118, 0.0)),  ('purple', (0.5019607843137255, 0.0, 0.5019607843137255)),  ('magenta', (1.0, 0.0, 1.0)),]
    rgb_colors = priority_colors + rgb_colors

    return rgb_colors


def open3d_set_side_view(vis):
    ctr = vis.get_view_control()
    # ctr.set_front([-0.61959040621518757, 0.46765094085676973, 0.63040489055992976])
    # ctr.set_lookat([0.28810001969337462, 0.10746435821056366, 0.23499999999999999])
    # ctr.set_up([0.64188154672853504, -0.16037991603449936, 0.74984422549096852])
    # ctr.set_zoom(0.7)
    # ctr.rotate(10.0, 0.0)

    # ctr.set_front([ -0.51720189814974493, 0.55636089622063711, 0.65035740151617438 ])
    # ctr.set_lookat([ 0.23103321183824999, 0.26154772406860449, 0.15131956132592411 ])
    # ctr.set_up([ 0.47073865286968591, -0.44969907810742304, 0.75906248744340343 ])
    # ctr.set_zoom(3)

    # ctr.set_front([-0.86019269757539152, 0.40355968763418076, 0.31178213796587784])
    # ctr.set_lookat([0.28810001969337462, 0.10746435821056366, 0.23499999999999999])
    # ctr.set_up([0.30587875107201218, -0.080905438599338214, 0.94862663869811026])
    # ctr.set_zoom(0.69999999999999996)

    ctr.set_front([0.40466417238365116, 0.019007526352692254, 0.91426780624224468])
    ctr.set_lookat([0.61287602731590907, 0.010181152776318789, -0.073166629933366326])
    ctr.set_up([-0.91444954965885639, 0.0025306059632757057, 0.40469200283941076])
    ctr.set_zoom(0.84000000000000008)

    init_param = ctr.convert_to_pinhole_camera_parameters()
    print("camera extrinsic", init_param.extrinsic.tolist())


def save_pcs(xyz, rgb, save_path=None, return_buffer=False, add_coordinate_frame=False, side_view=False, add_table=True):

    assert save_path or return_buffer, "provide path to save or set return_buffer to true"

    unordered_pc = np.concatenate(xyz, axis=0)
    unordered_rgb = np.concatenate(rgb, axis=0)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(unordered_pc)
    pcd.colors = open3d.utility.Vector3dVector(unordered_rgb)

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)
    vis.update_geometry(pcd)

    if add_table:
        table_color = [0.7, 0.7, 0.7]
        origin = [0, -0.5, -0.03]
        table = open3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.02)
        table.paint_uniform_color(table_color)
        table.translate(origin)
        vis.add_geometry(table)

    if add_coordinate_frame:
        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(mesh_frame)
        vis.update_geometry(mesh_frame)

    if side_view:
        open3d_set_side_view(vis)

    vis.poll_events()
    vis.update_renderer()
    if save_path:
        vis.capture_screen_image(save_path)
    elif return_buffer:
        buffer = vis.capture_screen_float_buffer(False)

    vis.destroy_window()

    if return_buffer:
        return buffer
    else:
        return None


def get_initial_scene_idxs(dataset):
    """
    This function finds initial scenes from the dataset
    :param dataset:
    :return:
    """

    initial_scene2idx_t = {}
    for idx in range(len(dataset)):
        filename, t = dataset.get_data_index(idx)
        if filename not in initial_scene2idx_t:
            initial_scene2idx_t[filename] = (idx, t)
        else:
            if t > initial_scene2idx_t[filename][1]:
                initial_scene2idx_t[filename] = (idx, t)
    initial_scene_idxs = [initial_scene2idx_t[f][0] for f in initial_scene2idx_t]
    return initial_scene_idxs


def get_initial_scene_idxs_raw_data(data):
    """
    This function finds initial scenes from the dataset
    :param dataset:
    :return:
    """

    initial_scene2idx_t = {}
    for idx in range(len(data)):
        filename, t = data[idx]
        if filename not in initial_scene2idx_t:
            initial_scene2idx_t[filename] = (idx, t)
        else:
            if t > initial_scene2idx_t[filename][1]:
                initial_scene2idx_t[filename] = (idx, t)
    initial_scene_idxs = [initial_scene2idx_t[f][0] for f in initial_scene2idx_t]
    return initial_scene_idxs


def evaluate_target_object_predictions(all_gts, all_predictions, all_sentences, initial_scene_idxs, tokenizer):
    """
    This function evaluates target object predictions

    :param all_gts: a list of predictions for scenes. Each element is a list of booleans for objects in the scene
    :param all_predictions:
    :param all_sentences: a list of descriptions for scenes
    :param initial_scene_idxs:
    :param tokenizer:
    :return:
    """

    # overall accuracy
    print("\noverall accuracy")
    report = classification_report(list(itertools.chain(*all_gts)), list(itertools.chain(*all_predictions)),
                                   output_dict=True)
    print(report)

    # scene average
    print("\naccuracy per scene")
    acc_per_scene = []
    for gts, preds in zip(all_gts, all_predictions):
        acc_per_scene.append(sum(np.array(gts) == np.array(preds)) * 1.0 / len(gts))
    print(np.mean(acc_per_scene))
    plt.hist(acc_per_scene, 10, range=(0, 1), facecolor='g', alpha=0.75)
    plt.xlabel('Accuracy')
    plt.ylabel('# Scene')
    plt.title('Predicting objects to be rearranged')
    plt.xticks(np.linspace(0, 1, 11), np.linspace(0, 1, 11).round(1))
    plt.grid(True)
    plt.show()

    # initial scene accuracy
    print("\noverall accuracy for initial scenes")
    tested_initial_scene_idxs = [i for i in initial_scene_idxs if i < len(all_gts)]
    initial_gts = [all_gts[i] for i in tested_initial_scene_idxs]
    initial_predictions = [all_predictions[i] for i in tested_initial_scene_idxs]
    report = classification_report(list(itertools.chain(*initial_gts)), list(itertools.chain(*initial_predictions)),
                                   output_dict=True)
    print(report)

    # break down by the number of objects
    print("\naccuracy for # objects in scene")
    num_objects_in_scenes = np.array([len(gts) for gts in all_gts])
    unique_num_objects = np.unique(num_objects_in_scenes)
    acc_per_scene = np.array(acc_per_scene)
    assert len(acc_per_scene) == len(num_objects_in_scenes)
    for num_objects in unique_num_objects:
        this_scene_idxs = [i for i in range(len(all_gts)) if len(all_gts[i]) == num_objects]
        this_num_obj_gts = [all_gts[i] for i in this_scene_idxs]
        this_num_obj_predictions = [all_predictions[i] for i in this_scene_idxs]
        report = classification_report(list(itertools.chain(*this_num_obj_gts)), list(itertools.chain(*this_num_obj_predictions)),
                                       output_dict=True)
        print("{} objects".format(num_objects))
        print(report)

    # reference
    print("\noverall accuracy break down")
    direct_gts_by_type = defaultdict(list)
    direct_preds_by_type = defaultdict(list)
    d_anchor_gts_by_type = defaultdict(list)
    d_anchor_preds_by_type = defaultdict(list)
    c_anchor_gts_by_type = defaultdict(list)
    c_anchor_preds_by_type = defaultdict(list)

    for i, s in enumerate(all_sentences):
        v, t = s[0]
        if t[-2:] == "_c" or t[-2:] == "_d":
            t = t[:-2]
        if v != "MASK" and t in tokenizer.discrete_types:
            # direct reference
            direct_gts_by_type[t].extend(all_gts[i])
            direct_preds_by_type[t].extend(all_predictions[i])
        else:
            if v == "MASK":
                # discrete anchor
                d_anchor_gts_by_type[t].extend(all_gts[i])
                d_anchor_preds_by_type[t].extend(all_predictions[i])
            else:
                c_anchor_gts_by_type[t].extend(all_gts[i])
                c_anchor_preds_by_type[t].extend(all_predictions[i])

    print("direct")
    for t in direct_gts_by_type:
        report = classification_report(direct_gts_by_type[t], direct_preds_by_type[t], output_dict=True)
        print(t, report)

    print("discrete anchor")
    for t in d_anchor_gts_by_type:
        report = classification_report(d_anchor_gts_by_type[t], d_anchor_preds_by_type[t], output_dict=True)
        print(t, report)

    print("continuous anchor")
    for t in c_anchor_gts_by_type:
        report = classification_report(c_anchor_gts_by_type[t], c_anchor_preds_by_type[t], output_dict=True)
        print(t, report)

    # break down by object class


def combine_and_sample_xyzs(xyzs, rgbs, center=None, radius=0.5, num_pts=1024):
    xyz = torch.cat(xyzs, dim=0)
    rgb = torch.cat(rgbs, dim=0)

    if center is not None:
        center = center.repeat(xyz.shape[0], 1)
        dists = torch.linalg.norm(xyz - center, dim=-1)
        idx = dists < radius
        xyz = xyz[idx]
        rgb = rgb[idx]

    idx = np.random.randint(0, xyz.shape[0], num_pts)
    xyz = xyz[idx]
    rgb = rgb[idx]

    return xyz, rgb


def evaluate_prior_prediction(gts, predictions, keys, debug=False):
    """
    :param gts: expect a list of tensors
    :param predictions: expect a list of tensor
    :return:
    """

    total_mses = 0
    obj_dists = []
    struct_dists = []
    for key in keys:
        # predictions[key][0]: [batch_size * number_of_objects, dim]
        predictions_for_key = torch.cat(predictions[key], dim=0)
        # gts[key][0]: [batch_size * number_of_objects, dim]
        gts_for_key = torch.cat(gts[key], dim=0)

        assert gts_for_key.shape == predictions_for_key.shape

        target_indices = gts_for_key != -100
        gts_for_key = gts_for_key[target_indices]
        predictions_for_key = predictions_for_key[target_indices]
        num_objects = len(predictions_for_key)

        distances = predictions_for_key - gts_for_key

        me = torch.mean(torch.abs(distances))
        mse = torch.mean(distances ** 2)
        med = torch.median(torch.abs(distances))

        if "obj_x" in key or "obj_y" in key or "obj_z" in key:
            obj_dists.append(distances)
        if "struct_x" in key or "struct_y" in key or "struct_z" in key:
            struct_dists.append(distances)

        if debug:
            print("Groundtruths:")
            print(gts_for_key[:100])
            print("Predictions")
            print(predictions_for_key[:100])

        print("{} ME for {} objects: {}".format(key, num_objects, me))
        print("{} MSE for {} objects: {}".format(key, num_objects, mse))
        print("{} MEDIAN for {} objects: {}".format(key, num_objects, med))
        total_mses += mse

        if "theta" in key:
            predictions_for_key = predictions_for_key.reshape(-1, 3, 3)
            gts_for_key = gts_for_key.reshape(-1, 3, 3)
            geodesic_distance = compute_geodesic_distance_from_two_matrices(predictions_for_key, gts_for_key)
            geodesic_distance = torch.rad2deg(geodesic_distance)
            mgd = torch.mean(geodesic_distance)
            stdgd = torch.std(geodesic_distance)
            megd = torch.median(geodesic_distance)
            print("{} Mean and std Geodesic Distance for {} objects: {} +- {}".format(key, num_objects, mgd, stdgd))
            print("{} Median Geodesic Distance for {} objects: {}".format(key, num_objects, megd))

    if obj_dists:
        euclidean_dists = torch.sqrt(obj_dists[0]**2 + obj_dists[1]**2 + obj_dists[2]**2)
        me = torch.mean(euclidean_dists)
        stde = torch.std(euclidean_dists)
        med = torch.median(euclidean_dists)
        print("Mean and std euclidean dist for {} objects: {} +- {}".format(len(euclidean_dists), me, stde))
        print("Median euclidean dist for {} objects: {}".format(len(euclidean_dists), med))
    if struct_dists:
        euclidean_dists = torch.sqrt(struct_dists[0] ** 2 + struct_dists[1] ** 2 + struct_dists[2] ** 2)
        me = torch.mean(euclidean_dists)
        stde = torch.std(euclidean_dists)
        med = torch.median(euclidean_dists)
        print("Mean euclidean dist for {} structures: {} +- {}".format(len(euclidean_dists), me, stde))
        print("Median euclidean dist for {} structures: {}".format(len(euclidean_dists), med))

    return -total_mses


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def visualize_occ(points, occupancies, in_num_pts=1000, out_num_pts=1000, visualize=False, threshold=0.5):

    rix = np.random.permutation(points.shape[0])
    vis_points = points[rix]
    vis_occupancies = occupancies[rix]
    in_pc = vis_points[vis_occupancies.squeeze() > threshold, :][:in_num_pts]
    out_pc = vis_points[vis_occupancies.squeeze() < threshold, :][:out_num_pts]

    if len(in_pc) == 0:
        print("no in points")
    if len(out_pc) == 0:
        print("no out points")

    in_pc = trimesh.PointCloud(in_pc)
    out_pc = trimesh.PointCloud(out_pc)
    in_pc.colors = np.tile((255, 0, 0, 255), (in_pc.vertices.shape[0], 1))
    out_pc.colors = np.tile((255, 255, 0, 120), (out_pc.vertices.shape[0], 1))

    if visualize:
        scene = trimesh.Scene([in_pc, out_pc])
        scene.show()

    return in_pc, out_pc