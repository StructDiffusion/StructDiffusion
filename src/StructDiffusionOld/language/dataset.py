from __future__ import print_function

import argparse
import copy
import csv
import cv2
import h5py
import numpy as np
import open3d
import os
import PIL
import scipy
import scipy.io
import sys
import trimesh

import torch
import pytorch3d.transforms as tra3d
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# Local imports
from predictor_v2 import show_pcs, get_pts
from StructDiffuser.tokenizer import Tokenizer

from brain2.utils.info import logwarn
import brain2.utils.image as img
import brain2.utils.transformations as tra
import brain2.utils.camera as cam


class SemanticArrangementDataset:

    def __init__(self, data_roots, index_roots, splits, tokenizer):

        self.data_roots = data_roots
        print("data dirs:", self.data_roots)

        self.tokenizer = tokenizer

        self.arrangement_data = []
        arrangement_steps = []
        for split in splits:
            for data_root, index_root in zip(data_roots, index_roots):
                arrangement_indices_file = os.path.join(data_root, index_root, "{}_arrangement_indices_file_all.txt".format(split))
                if os.path.exists(arrangement_indices_file):
                    with open(arrangement_indices_file, "r") as fh:
                        arrangement_steps.extend([(os.path.join(data_root, f[0]), f[1]) for f in eval(fh.readline().strip())])
                else:
                    print("{} does not exist".format(arrangement_indices_file))

        # only keep one dummy step for each rearrangement
        for filename, step_t in arrangement_steps:
            if step_t == 0:
                self.arrangement_data.append(filename)
        print("{} valid sequences".format(len(self.arrangement_data)))

    def __len__(self):
        return len(self.arrangement_data)

    def get_raw_data(self, idx):

        filename = self.arrangement_data[idx]
        h5 = h5py.File(filename, 'r')
        goal_specification = json.loads(str(np.array(h5["goal_specification"])))

        ###################################
        # preparing sentence
        sentence = []

        # structure parameters
        # 5 parameters
        structure_parameters = goal_specification["shape"]
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))
            if structure_parameters["type"] == "circle":
                sentence.append((structure_parameters["radius"], "radius"))
            elif structure_parameters["type"] == "line":
                sentence.append((structure_parameters["length"] / 2.0, "radius"))
        else:
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))

        # print(sentence)

        # token_idxs = np.random.permutation(len(sentence))
        # token_idxs = token_idxs[:np.random.randint(1, len(sentence) + 1)]
        # token_idxs = sorted(token_idxs)
        # incomplete_sentence = [sentence[ti] for ti in token_idxs]

        # print(incomplete_sentence)

        # print(self.tokenizer.convert_structure_params_to_natural_language(sentence))

        return sentence




if __name__ == "__main__":

    tokenizer = Tokenizer("/home/weiyu/data_drive/data_new_objects/type_vocabs_coarse.json")

    data_roots = []
    index_roots = []
    for shape, index in [("circle", "index_10k"), ("line", "index_10k"), ("tower", "index_10k"), ("dinner", "index_10k")]:
        data_roots.append("/home/weiyu/data_drive/data_new_objects/examples_{}_new_objects/result".format(shape))
        index_roots.append(index)

    dataset = SemanticArrangementDataset(data_roots=data_roots,
                                         index_roots=index_roots,
                                         splits=["train", "valid", "test"], tokenizer=tokenizer)

    print(len(dataset))
    idxs = np.random.permutation(len(dataset))
    for i in idxs:
        dataset.get_raw_data(i)

        input("next?")

    # dataloader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=8)
    # for i, d in enumerate(dataloader):
    #     print(i)
    #     for k in d:
    #         if isinstance(d[k], torch.Tensor):
    #             print("--size", k, d[k].shape)
    #     for k in d:
    #         print(k, d[k])
    #
    #     input("next?")

    # tokenizer = Tokenizer("/home/weiyu/data_drive/data_new_objects/type_vocabs.json")
    # for shape, index in [("circle", "index_34k"), ("line", "index_42k"), ("tower", "index_13k"), ("dinner", "index_24k")]:
    #     for split in ["train", "valid", "test"]:
    #         dataset = SemanticArrangementDataset(data_root="/home/weiyu/data_drive/data_new_objects/examples_{}_new_objects/result".format(shape),
    #                                              index_root=index,
    #                                              split=split, tokenizer=tokenizer,
    #                                              max_num_objects=7,
    #                                              max_num_other_objects=5,
    #                                              max_num_shape_parameters=5,
    #                                              max_num_rearrange_features=0,
    #                                              max_num_anchor_features=0,
    #                                              num_pts=1024,
    #                                              debug=True)
    #
    #         for i in range(0, 1):
    #             d = dataset.get_raw_data(i)
    #             d = dataset.convert_to_tensors(d, dataset.tokenizer)
    #             for k in d:
    #                 if torch.is_tensor(d[k]):
    #                     print("--size", k, d[k].shape)
    #             for k in d:
    #                 print(k, d[k])
    #             input("next?")

            # dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8,
            #                         collate_fn=SemanticArrangementDataset.collate_fn)
            # for d in tqdm(dataloader):
            #     pass