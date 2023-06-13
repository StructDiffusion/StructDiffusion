from __future__ import print_function

import StructDiffusion.utils.transformations as tra


def make_pose(trans, rot):
    """Make 4x4 matrix from (trans, rot)"""
    pose = tra.quaternion_matrix(rot)
    pose[:3, 3] = trans
    return pose
