import os
import torch
import pytorch3d.transforms as tra3d

from StructDiffusion.utils.rotation_continuity import compute_rotation_matrix_from_ortho6d


def get_diffusion_variables_from_9D_actions(struct_xyztheta_inputs, obj_xyztheta_inputs):

    # important: we need to get the first two columns, not first two rows
    # array([[ 3,  4,  5],
    #   [ 6,  7,  8],
    #   [ 9, 10, 11]])
    xyz_6d_idxs = [0, 1, 2, 3, 6, 9, 4, 7, 10]

    # print(batch_data["obj_xyztheta_inputs"].shape)
    # print(batch_data["struct_xyztheta_inputs"].shape)

    # only get the first and second columns of rotation
    obj_xyztheta_inputs = obj_xyztheta_inputs[:, :, xyz_6d_idxs]  # B, N, 9
    struct_xyztheta_inputs = struct_xyztheta_inputs[:, :, xyz_6d_idxs]  # B, 1, 9

    x = torch.cat([struct_xyztheta_inputs, obj_xyztheta_inputs], dim=1)  # B, 1 + N, 9

    # print(x.shape)

    return x


def get_diffusion_variables_from_H(poses):
    """
    [[0,1,2,3],
    [4,5,6,7],
    [8,9,10,11],
    [12,13,14,15]
    :param obj_xyztheta_inputs: B, N, 4, 4
    :return:
    """

    xyz_6d_idxs = [3, 7, 11, 0, 4, 8, 1, 5, 9]

    B, N, _, _ = poses.shape
    x = poses.reshape(B, N, 16)[:, :, xyz_6d_idxs]  # B, N, 9
    return x


def get_struct_objs_poses(x):

    on_gpu = x.is_cuda
    if not on_gpu:
        x = x.cuda()

    # assert x.is_cuda, "compute_rotation_matrix_from_ortho6d requires input to be on gpu"
    device = x.device

    # important: the noisy x can go out of bounds
    x = torch.clamp(x, min=-1, max=1)

    # x: B, 1 + N, 9
    B = x.shape[0]
    N = x.shape[1] - 1

    # compute_rotation_matrix_from_ortho6d takes in [B, 6], outputs [B, 3, 3]
    x_6d = x[:, :, 3:].reshape(-1, 6)
    x_rot = compute_rotation_matrix_from_ortho6d(x_6d).reshape(B, N+1, 3, 3)  # B, 1 + N, 3, 3

    x_trans = x[:, :, :3] # B, 1 + N, 3

    x_full = torch.eye(4).repeat(B, 1 + N, 1, 1).to(device)
    x_full[:, :, :3, :3] = x_rot
    x_full[:, :, :3, 3] = x_trans

    struct_pose = x_full[:, 0].unsqueeze(1) # B, 1, 4, 4
    pc_poses_in_struct = x_full[:, 1:] # B, N, 4, 4

    if not on_gpu:
        struct_pose = struct_pose.cpu()
        pc_poses_in_struct = pc_poses_in_struct.cpu()

    return struct_pose, pc_poses_in_struct


def compute_current_and_goal_pc_poses(obj_xyzs, struct_pose, pc_poses_in_struct):

    device = obj_xyzs.device

    # obj_xyzs: B, N, P, 3
    # struct_pose: B, 1, 4, 4
    # pc_poses_in_struct: B, N, 4, 4
    B, N, _, _ = pc_poses_in_struct.shape
    _, _, P, _ = obj_xyzs.shape

    current_pc_poses = torch.eye(4).repeat(B, N, 1, 1).to(device)  # B, N, 4, 4
    # print(torch.mean(obj_xyzs, dim=2).shape)
    current_pc_poses[:, :, :3, 3] = torch.mean(obj_xyzs, dim=2)  # B, N, 4, 4

    struct_pose = struct_pose.repeat(1, N, 1, 1)  # B, N, 4, 4
    struct_pose = struct_pose.reshape(B * N, 4, 4)  # B x 1, 4, 4
    pc_poses_in_struct = pc_poses_in_struct.reshape(B * N, 4, 4)  # B x N, 4, 4

    goal_pc_poses = struct_pose @ pc_poses_in_struct  # B x N, 4, 4
    goal_pc_poses = goal_pc_poses.reshape(B, N, 4, 4)  # B, N, 4, 4
    return current_pc_poses, goal_pc_poses