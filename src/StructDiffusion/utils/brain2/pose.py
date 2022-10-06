# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import StructDiffusion.utils.transformations as tra


def make_pose(trans, rot):
    """Make 4x4 matrix from (trans, rot)"""
    pose = tra.quaternion_matrix(rot)
    pose[:3, 3] = trans
    return pose
