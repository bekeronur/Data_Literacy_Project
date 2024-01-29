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

    
def show_anns(anns):
    '''
    anns: output struct of mask_generator.generate(), with mask_generator = SamAutomaticMaskGenerator(sam),
    '''
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: ndarray [n1, 2], pixel coordinates
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: ndarray [n2, 2], pixel coordinates
        matches: ndarray [n_match, 2], idx in kp1 and kp2
        color: The color of the circles and connecting lines drawn on the images. If None, randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
            c = ( int (c[ 0 ]), int(c[ 1 ]), int (c[ 2 ]))
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m[0]]).astype(int))
        end2 = tuple(np.round(kp2[m[1]]).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
        
    return new_img


def show_descriptor_data(descriptor_data, img_size, dilation_kernel_size=5, figsize = (10, 10)):
    spatial_descriptor_mask = np.zeros(img_size)
    spatial_descriptor_mask[descriptor_data['local_descriptor_px_coords_HW'][:, 0].cpu().numpy().astype(int), descriptor_data['local_descriptor_px_coords_HW'][:, 1].cpu().numpy().astype(int)] = True
    dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    spatial_descriptor_mask = cv2.dilate(spatial_descriptor_mask.astype(np.uint8), dilation_kernel, iterations=1).astype(bool)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 3, 1)
    ax.title.set_text('Image')
    plt.imshow(descriptor_data['img_crop_pad'])
    plt.axis('off')
    ax = fig.add_subplot(1, 3, 2)
    ax.title.set_text('Mask')
    plt.imshow(descriptor_data['mask_crop_pad'])
    plt.axis('off')
    ax = fig.add_subplot(1, 3, 3)
    ax.title.set_text('Spatial Descriptors')
    plt.imshow(spatial_descriptor_mask)
    plt.axis('off')
    plt.show()