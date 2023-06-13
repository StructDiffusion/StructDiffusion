import os
import trimesh
import json
import h5py
import numpy as np


class MeshSceneViewer:

    def __init__(self, assets_dir, cache_mesh=True, mesh_type="meshes"):

        # load visual mesh or collision mesh (after convex decomposition)
        assert mesh_type in ["meshes", "visual"]
        self.mesh_type = mesh_type
        self.cache_mesh = cache_mesh

        self.assets_dir = assets_dir
        if cache_mesh:
            self.urdf_to_mesh = {}
            for filename in os.listdir(os.path.join(self.assets_dir, mesh_type)):
                urdf = filename[:-4]
                # TODO: can also enforce scene or single mesh here
                self.urdf_to_mesh[urdf] = trimesh.load(os.path.join(self.assets_dir, mesh_type, filename))

    def load_object_mesh_from_urdf(self, urdf):
        if self.cache_mesh:
            return self.urdf_to_mesh[urdf[:-5]].copy()
        else:
            mesh_path = os.path.join(self.assets_dir, self.mesh_type, urdf[:-5] + ".obj")
            mesh = trimesh.load(mesh_path)
            return mesh

    def load_mesh_scene(self, target_objs, goal_specification, current_obj_poses, current_pc_poses, goal_pc_poses, visualize=False):

        structure_parameters = goal_specification["shape"]
        target_obj_urdfs = [obj_spec["urdf"] for obj_spec in goal_specification["rearrange"]["objects"]]
        # Important: ensure the order is correct
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            target_obj_urdfs = target_obj_urdfs[::-1]
        elif structure_parameters["type"] == "tower" or structure_parameters["type"] == "dinner":
            target_obj_urdfs = target_obj_urdfs
        else:
            raise KeyError("{} structure is not recognized".format(structure_parameters["type"]))

        target_obj_meshes = [self.load_object_mesh_from_urdf(u) for u in target_obj_urdfs]

        for i, obj_name in enumerate(target_objs):
            current_obj_pose = current_obj_poses[i]
            current_pc_pose = current_pc_poses[i]
            goal_pc_pose = goal_pc_poses[i]
            goal_obj_pose = goal_pc_pose @ np.linalg.inv(current_pc_pose) @ current_obj_pose

            target_obj_meshes[i].apply_transform(goal_obj_pose)

        scene = trimesh.Scene(target_obj_meshes)
        if visualize:
            scene.show()
        return scene

    def check_scene_collision(self, scene):
        collision_manager, _ = trimesh.collision.scene_to_collision(scene)
        in_collision_internal = collision_manager.in_collision_internal()
        return in_collision_internal


def run_example():

    import torch
    from StructDiffusion.data.dataset_v1_diffuser import SemanticArrangementDataset
    from StructDiffusion.data.tokenizer import Tokenizer
    from StructDiffusion.training.train_diffuser_v3_lang import get_diffusion_variables, get_struct_objs_poses

    # dataloader
    tokenizer = Tokenizer("/home/weiyu/data_drive/data_new_objects/type_vocabs_coarse.json")
    data_roots = []
    index_roots = []
    for shape, index in [("circle", "index_10k")]:
        data_roots.append("/home/weiyu/data_drive/data_new_objects/examples_{}_new_objects/result".format(shape))
        index_roots.append(index)
    dataset = SemanticArrangementDataset(data_roots=data_roots,
                                         index_roots=index_roots,
                                         split="train", tokenizer=tokenizer,
                                         max_num_objects=7,
                                         max_num_other_objects=5,
                                         max_num_shape_parameters=5,
                                         max_num_rearrange_features=0,
                                         max_num_anchor_features=0,
                                         num_pts=1024,
                                         filter_num_moved_objects_range=None,  # [5, 5]
                                         data_augmentation=False,
                                         shuffle_object_index=False,
                                         debug=False)

    # mesh scene loader
    msv = MeshSceneViewer(assets_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large", cache_mesh=True)

    print(len(dataset))
    for i in range(len(dataset)):
        d = dataset.get_raw_data(i, inference_mode=True)

        # compute goal pc poses
        obj_xyztheta_inputs = torch.FloatTensor(d["obj_xyztheta_inputs"])[None, :]  # B, N, 4, 4
        struct_xyztheta_inputs = torch.FloatTensor(d["struct_xyztheta_inputs"])[None, :]  # B,
        x_gt = get_diffusion_variables(struct_xyztheta_inputs, obj_xyztheta_inputs)
        struct_pose, pc_poses_in_struct = get_struct_objs_poses(x_gt)  # B, 1, 4, 4; B, N, 4, 4

        B, N, _, _ = pc_poses_in_struct.shape
        struct_pose = struct_pose.repeat(1, N, 1, 1)  # B, N, 4, 4
        struct_pose = struct_pose.reshape(B * N, 4, 4)  # B x N, 4, 4
        pc_poses_in_struct = pc_poses_in_struct.reshape(B * N, 4, 4)  # B x N, 4, 4
        goal_pc_pose = struct_pose @ pc_poses_in_struct  # B x N, 4, 4
        goal_pc_poses = goal_pc_pose.reshape(B, N, 4, 4)[0].numpy()

        scene = msv.load_mesh_scene(d["target_objs"], d["goal_specification"], d["current_obj_poses"], d["current_pc_poses"], goal_pc_poses, visualize=True)
        collision = msv.check_scene_collision(scene)
        print("scene has collision", collision)


if __name__ == "__main__":
    run_example()