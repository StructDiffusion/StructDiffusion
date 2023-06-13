import sys
import os
import h5py
import torch
import pytorch3d.transforms as tra3d

from StructDiffusion.utils.rearrangement import show_pcs_color_order
from StructDiffusion.utils.pointnet import random_point_sample, index_points


def switch_stdout(stdout_filename=None):
    if stdout_filename:
        print("setting stdout to {}".format(stdout_filename))
        if os.path.exists(stdout_filename):
            sys.stdout = open(stdout_filename, 'a')
        else:
            sys.stdout = open(stdout_filename, 'w')
    else:
        sys.stdout = sys.__stdout__


def visualize_batch_pcs(obj_xyzs, B, N, P, verbose=True, limit_B=None):
    if limit_B is None:
        limit_B = B

    vis_obj_xyzs = obj_xyzs.reshape(B, N, P, -1)
    vis_obj_xyzs = vis_obj_xyzs[:limit_B]

    if type(vis_obj_xyzs).__module__ == torch.__name__:
        if vis_obj_xyzs.is_cuda:
            vis_obj_xyzs = vis_obj_xyzs.detach().cpu()
        vis_obj_xyzs = vis_obj_xyzs.numpy()

    for bi, vis_obj_xyz in enumerate(vis_obj_xyzs):
        if verbose:
            print("example {}".format(bi))
            print(vis_obj_xyz.shape)
        show_pcs_color_order([xyz[:, :3] for xyz in vis_obj_xyz], None, visualize=True, add_coordinate_frame=True, add_table=False)


def convert_bool(d):
    for k in d:
        if type(d[k]) == list:
            d[k] = [bool(i) for i in d[k]]
        else:
            d[k] = bool(d[k])
    return d


def save_dict_to_h5(dict_data, filename):
    fh = h5py.File(filename, 'w')
    for k in dict_data:
        key_data = dict_data[k]
        if key_data is None:
            raise RuntimeError('data was not properly populated')
        # if type(key_data) is dict:
        #     key_data = json.dumps(key_data, sort_keys=True)
        try:
            fh.create_dataset(k, data=key_data)
        except TypeError as e:
            print("Failure on key", k)
            print(key_data)
            print(e)
            raise e
    fh.close()


def move_pc_and_create_scene_new(obj_xyzs, obj_params, struct_pose, current_pc_pose, target_object_inds, device,
                                 return_scene_pts=False, return_scene_pts_and_pc_idxs=False, num_scene_pts=None, normalize_pc=False,
                                 return_pair_pc=False, num_pair_pc_pts=None, normalize_pair_pc=False):

    # obj_xyzs: N, P, 3
    # obj_params: B, N, 6
    # struct_pose: B x N, 4, 4
    # current_pc_pose: B x N, 4, 4
    # target_object_inds: 1, N

    B, N, _ = obj_params.shape
    _, P, _ = obj_xyzs.shape

    # B, N, 6
    flat_obj_params = obj_params.reshape(B * N, -1)
    goal_pc_pose_in_struct = torch.eye(4).repeat(B * N, 1, 1).to(device)
    goal_pc_pose_in_struct[:, :3, :3] = tra3d.euler_angles_to_matrix(flat_obj_params[:, 3:], "XYZ")
    goal_pc_pose_in_struct[:, :3, 3] = flat_obj_params[:, :3]  # B x N, 4, 4

    goal_pc_pose = struct_pose @ goal_pc_pose_in_struct
    goal_pc_transform = goal_pc_pose @ torch.inverse(current_pc_pose)  # cur_batch_size x N, 4, 4

    # important: pytorch3d uses row-major ordering, need to transpose each transformation matrix
    transpose = tra3d.Transform3d(matrix=goal_pc_transform.transpose(1, 2))

    # obj_xyzs: N, P, 3
    new_obj_xyzs = obj_xyzs.repeat(B, 1, 1)
    new_obj_xyzs = transpose.transform_points(new_obj_xyzs)

    # put it back to B, N, P, 3
    new_obj_xyzs = new_obj_xyzs.reshape(B, N, P, -1)
    # visualize_batch_pcs(new_obj_xyzs, S, N, P)


    # initialize the additional outputs
    subsampled_scene_xyz = None
    subsampled_pc_idxs = None
    obj_pair_xyzs = None

    # ===================================
    # Pass to discriminator
    if return_scene_pts:

        num_indicator = N

        # add one hot
        indicator_variables = torch.eye(num_indicator).repeat(B, 1, 1, P).reshape(B, num_indicator, P, num_indicator).to(device)  # B, N, P, N
        # print(indicator_variables.shape)
        # print(new_obj_xyzs.shape)
        new_obj_xyzs = torch.cat([new_obj_xyzs, indicator_variables], dim=-1)  # B, N, P, 3 + N

        # combine pcs in each scene
        scene_xyzs = new_obj_xyzs.reshape(B, N * P, 3 + N)

        # ToDo: maybe convert this to a batch operation
        subsampled_scene_xyz = torch.FloatTensor(B, num_scene_pts, 3 + N).to(device)
        for si, scene_xyz in enumerate(scene_xyzs):
            # scene_xyz: N*P, 3+N
            # target_object_inds: 1, N
            subsample_idx = torch.randint(0, torch.sum(target_object_inds[0]) * P, (num_scene_pts,)).to(device)
            subsampled_scene_xyz[si] = scene_xyz[subsample_idx]

            # # debug:
            # print("-"*50)
            # if si < 10:
            #     trimesh.PointCloud(scene_xyz[:, :3].cpu().numpy(), colors=[255, 0, 0, 255]).show()
            #     trimesh.PointCloud(subsampled_scene_xyz[si, :, :3].cpu().numpy(), colors=[0, 255, 0, 255]).show()

        # subsampled_scene_xyz: B, num_scene_pts, 3+N
        # new_obj_xyzs: B, N, P, 3
        # goal_pc_pose: B, N, 4, 4

        # important:
        if normalize_pc:
            subsampled_scene_xyz[:, :, 0:3] = pc_normalize_batch(subsampled_scene_xyz[:, :, 0:3])

            # # debug:
            # for si in range(10):
            #     trimesh.PointCloud(subsampled_scene_xyz[si, :, :3].cpu().numpy(), colors=[0, 0, 255, 255]).show()

    if return_scene_pts_and_pc_idxs:
        num_indicator = N
        pc_idxs = torch.arange(0, num_indicator)[:, None].repeat(B, 1, P).reshape(B, num_indicator, P).to(device)  # B, N, P
        # new_obj_xyzs: B, N, P, 3 + 1

        # combine pcs in each scene
        scene_xyzs = new_obj_xyzs.reshape(B, N * P, 3)
        pc_idxs = pc_idxs.reshape(B, N*P)

        subsampled_scene_xyz = torch.FloatTensor(B, num_scene_pts, 3).to(device)
        subsampled_pc_idxs = torch.LongTensor(B, num_scene_pts).to(device)
        for si, (scene_xyz, pc_idx) in enumerate(zip(scene_xyzs, pc_idxs)):
            # scene_xyz: N*P, 3+1
            # target_object_inds: 1, N
            subsample_idx = torch.randint(0, torch.sum(target_object_inds[0]) * P, (num_scene_pts,)).to(device)
            subsampled_scene_xyz[si] = scene_xyz[subsample_idx]
            subsampled_pc_idxs[si] = pc_idx[subsample_idx]

        # subsampled_scene_xyz: B, num_scene_pts, 3
        # subsampled_pc_idxs: B, num_scene_pts
        # new_obj_xyzs: B, N, P, 3
        # goal_pc_pose: B, N, 4, 4

        # important:
        if normalize_pc:
            subsampled_scene_xyz[:, :, 0:3] = pc_normalize_batch(subsampled_scene_xyz[:, :, 0:3])

        # TODO: visualize each individual object
        # debug
        # print(subsampled_scene_xyz.shape)
        # print(subsampled_pc_idxs.shape)
        # print("visualize subsampled scene")
        # for si in range(5):
        #     trimesh.PointCloud(subsampled_scene_xyz[si, :, :3].cpu().numpy(), colors=[0, 0, 255, 255]).show()

    ###############################################
    # Create input for pairwise collision detector
    if return_pair_pc:

        assert num_pair_pc_pts is not None

        # new_obj_xyzs: B, N, P, 3 + N
        # target_object_inds: 1, N
        # ignore paddings
        num_objs = torch.sum(target_object_inds[0])
        obj_pair_idxs = torch.combinations(torch.arange(num_objs), r=2)  # num_comb, 2

        # use [:, :, :, :3] to get obj_xyzs without object-wise indicator
        obj_pair_xyzs = new_obj_xyzs[:, :, :, :3][:, obj_pair_idxs]  # B, num_comb, 2 (obj 1 and obj 2), P, 3
        num_comb = obj_pair_xyzs.shape[1]
        pair_indicator_variables = torch.eye(2).repeat(B, num_comb, 1, 1, P).reshape(B, num_comb, 2, P, 2).to(device)  # B, num_comb, 2, P, 2
        obj_pair_xyzs = torch.cat([obj_pair_xyzs, pair_indicator_variables], dim=-1)  # B, num_comb, 2, P, 3 (pc channels) + 2 (indicator for obj 1 and obj 2)
        obj_pair_xyzs = obj_pair_xyzs.reshape(B, num_comb, P * 2, 5)

        # random sample: idx = np.random.randint(0, scene_xyz.shape[0], self.num_scene_pts)
        obj_pair_xyzs = obj_pair_xyzs.reshape(B * num_comb, P * 2, 5)
        # random_point_sample() input dim: B, N, C
        rand_idxs = random_point_sample(obj_pair_xyzs, num_pair_pc_pts)  # B * num_comb, num_pair_pc_pts
        obj_pair_xyzs = index_points(obj_pair_xyzs, rand_idxs)  # B * num_comb, num_pair_pc_pts, 5

        if normalize_pair_pc:
            # pc_normalize_batch() input dim: pc: B, num_scene_pts, 3
            # obj_pair_xyzs = obj_pair_xyzs.reshape(B * num_comb, num_pair_pc_pts, 5)
            obj_pair_xyzs[:, :, 0:3] = pc_normalize_batch(obj_pair_xyzs[:, :, 0:3])
            obj_pair_xyzs = obj_pair_xyzs.reshape(B, num_comb, num_pair_pc_pts, 5)

            # # debug
            # for bi, this_obj_pair_xyzs in enumerate(obj_pair_xyzs):
            #     print("batch id", bi)
            #     for pi, obj_pair_xyz in enumerate(this_obj_pair_xyzs):
            #         print("pair", pi)
            #         # obj_pair_xyzs: 2 * P, 5
            #         print(obj_pair_xyz[:, :3].shape)
            #         trimesh.PointCloud(obj_pair_xyz[:, :3].cpu()).show()

    # obj_pair_xyzs: B, num_comb, num_pair_pc_pts, 3 + 2
    goal_pc_pose = goal_pc_pose.reshape(B, N, 4, 4)

    return new_obj_xyzs, goal_pc_pose, subsampled_scene_xyz, subsampled_pc_idxs, obj_pair_xyzs



def move_pc(obj_xyzs, obj_params, struct_pose, current_pc_pose, device):

    # obj_xyzs: N, P, 3
    # obj_params: B, N, 6
    # struct_pose: B x N, 4, 4
    # current_pc_pose: B x N, 4, 4
    # target_object_inds: 1, N

    B, N, _ = obj_params.shape
    _, P, _ = obj_xyzs.shape

    # B, N, 6
    flat_obj_params = obj_params.reshape(B * N, -1)
    goal_pc_pose_in_struct = torch.eye(4).repeat(B * N, 1, 1).to(device)
    goal_pc_pose_in_struct[:, :3, :3] = tra3d.euler_angles_to_matrix(flat_obj_params[:, 3:], "XYZ")
    goal_pc_pose_in_struct[:, :3, 3] = flat_obj_params[:, :3]  # B x N, 4, 4

    goal_pc_pose = struct_pose @ goal_pc_pose_in_struct
    goal_pc_transform = goal_pc_pose @ torch.inverse(current_pc_pose)  # cur_batch_size x N, 4, 4

    # important: pytorch3d uses row-major ordering, need to transpose each transformation matrix
    transpose = tra3d.Transform3d(matrix=goal_pc_transform.transpose(1, 2))

    # obj_xyzs: N, P, 3
    new_obj_xyzs = obj_xyzs.repeat(B, 1, 1)
    new_obj_xyzs = transpose.transform_points(new_obj_xyzs)

    # put it back to B, N, P, 3
    new_obj_xyzs = new_obj_xyzs.reshape(B, N, P, -1)
    # visualize_batch_pcs(new_obj_xyzs, S, N, P)

    # subsampled_scene_xyz: B, num_scene_pts, 3+N
    # new_obj_xyzs: B, N, P, 3
    # goal_pc_pose: B, N, 4, 4

    goal_pc_pose = goal_pc_pose.reshape(B, N, 4, 4)
    return new_obj_xyzs, goal_pc_pose


def sample_gaussians(mus, sigmas, sample_size):
    # mus: [number of individual gaussians]
    # sigmas: [number of individual gaussians]
    normal = torch.distributions.Normal(mus, sigmas)
    samples = normal.sample((sample_size,))
    # samples: [sample_size, number of individual gaussians]
    return samples

def fit_gaussians(samples, sigma_eps=0.01):
    device = samples.device

    # samples: [sample_size, number of individual gaussians]
    num_gs = samples.shape[1]
    mus = torch.mean(samples, dim=0).to(device)
    sigmas = torch.std(samples, dim=0).to(device) + sigma_eps * torch.ones(num_gs).to(device)
    # mus: [number of individual gaussians]
    # sigmas: [number of individual gaussians]
    return mus, sigmas


def pc_normalize_batch(pc):
    # pc: B, num_scene_pts, 3
    centroid = torch.mean(pc, dim=1)  # B, 3
    pc = pc - centroid[:, None, :]
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=2)), dim=1)[0]
    pc = pc / m[:, None, None]
    return pc
