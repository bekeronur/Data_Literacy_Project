import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.spatial import cKDTree
import cv2
import os
import sys
import json
import glob
from os.path import isfile, join, splitext, dirname, basename
from warnings import warn

        



def extract_negative_cls(backbone, negative_dataset, device):
    cls_token_list = []
    with torch.no_grad():
        for rgb in negative_dataset:
            cls_token_list.append(backbone.model(backbone.preprocess(rgb).to(device)))
    negative_cls_tokens = torch.cat(cls_token_list)
    return negative_cls_tokens


class NegativeDataset(Dataset):
    def __init__(self, root_dir='/home/user/fewshot_data/fewshot_data'):
        self.img_paths = sorted(glob.glob(os.path.join('/home/user/fewshot_data/fewshot_data', '**/*.jpg')))
        self.mask_paths = sorted(glob.glob(os.path.join('/home/user/fewshot_data/fewshot_data', '**/*.png')))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb = np.asarray(Image.open(self.img_paths[idx]))
        mask = np.asarray(Image.open(self.mask_paths[idx])) / 255
        if mask.shape[-1] == 4: mask = mask[..., :-1]
        if rgb.shape == mask.shape == (224, 224, 3):
            return Image.fromarray((rgb * mask).astype(np.uint8))
        else:
            return Image.fromarray(np.zeros([224, 224, 3]).astype(np.uint8))


def find_antipodal_point(point_coord, point_normal, pcd, voxel_size=0.01):
    point_vec = np.asarray(pcd.points) - point_coord
    perpendicular_dist = np.linalg.norm(np.cross(point_vec, point_normal), axis=-1)
    line_dist = np.linalg.norm(point_vec, axis=-1)

    intersection_idx = (perpendicular_dist < voxel_size).nonzero()[0]
    intersection_line_dist = line_dist[intersection_idx]
    antipodal_id = intersection_idx[intersection_line_dist.argmax()]
    antipodal_distance = intersection_line_dist.max()
    
    return antipodal_distance, antipodal_id