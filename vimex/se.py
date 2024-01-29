import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation

from .dpc import DescriptorPointCloud
from . import utils


############################################################################################################################################ 
# Utilities related to coordinate frames
############################################################################################################################################ 

def create_lookat_frame(z_camera_in_wf, up_in_wf, origin_in_wf):
    '''
    Creates a 3D camera frame in world coordinates given z axis and up direction.
    The convention is that z axis looks at the object, y axis looks up.

    Inputs
    --------------------
    - z_camera_in_wf: np.array() of shape(3,), the z axis of the camera frame (aligned with the principal ray) in world coordinates.
      Does not need to be normalized.
    - up_wf: np.array() of shape(3,), the y axis of the world frame (looking up) in world coordinates.
      Does not need to be normalized.
    - origin_wf: np.array() of shape(3,), the origin of the camera frame in world coordinates.

    Outputs
    --------------------
    - camera_pose_in_wf: np.array of shape(4, 4), the rigid-body transformation of the camera frame as seen from the world frame.
    '''
    camera_pose_in_wf = np.eye(4)
    x_camera_in_wf = np.cross(up_in_wf, z_camera_in_wf)
    y_camera_in_wf = np.cross(z_camera_in_wf, x_camera_in_wf)
    camera_pose_in_wf[:-1, 0] = x_camera_in_wf / np.linalg.norm(x_camera_in_wf)
    camera_pose_in_wf[:-1, 1] = y_camera_in_wf / np.linalg.norm(y_camera_in_wf)
    camera_pose_in_wf[:-1, 2] = z_camera_in_wf / np.linalg.norm(z_camera_in_wf)
    camera_pose_in_wf[:-1, 3] = origin_in_wf

    return camera_pose_in_wf


############################################################################################################################################ 
# Utilities to create SO(3), SE(3) transformations
############################################################################################################################################ 

def make_so3_from_axis(w):
    '''
    Creates an so(3) transformation (i.e., infinitesimal generator for the lie group SO(3) of rotations) given a rotation axis.

    Inputs
    --------------------
    w: np.array of shape(N, 3). The direction vector for the rotation axis.

    Outputs
    --------------------
    w_so3: np.array of shape(N, 3, 3) with dtype float interpreted as a rotational displacement (i.e., rather than a change of coordinate frames).
    '''
    
    # Normalize the rotation axis
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    
    # Convert the axis to so(3)
    w_so3 = np.zeros([*w.shape, 3])
    w_so3[:, 0, 1] = -w[:, 2]
    w_so3[:, 0, 2] = w[:, 1]
    w_so3[:, 1, 0] = w[:, 2]
    w_so3[:, 1, 2] = -w[:, 0]
    w_so3[:, 2, 0] = -w[:, 1]
    w_so3[:, 2, 1] = -w[:, 0]

    return w_so3


def make_SO3_from_angle_axis(angle, w):
    '''
    Creates an SO(3) transformation (i.e., rotational displacement in 3D) given a rotation axis.
    Applies the exponential map so(3)->SO(3). 

    Inputs
    --------------------
    angle: np.array of shape(N). The angular displacement in radian units.
    w: np.array of shape(N, 3). The direction vector for the rotation axis.

    Outputs
    --------------------
    R: np.array of shape(N, 3, 3) with dtype float interpreted as a rotational displacement (i.e., rather than a change of coordinate frames).
    '''

    # Normalize the rotation axis
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    N = w.shape[0]
    
    # Convert to so(3) 
    w_so3 = make_so3_from_axis(w)

    # Exponential map so(3)->SO(3) using Rodrigues' formula
    angle = angle[:, np.newaxis, np.newaxis]
    R = np.repeat(np.eye(3)[np.newaxis, ...], N, axis=0) + np.sin(angle)*w_so3 + (1 - np.cos(angle)) * (w_so3 @ w_so3)
    
    return R
    

def make_se3_from_axis(q, w, h):
    '''
    Creates an se(3) transformation (i.e., infinitesimal generator for the lie group SE(3) of rigid body displacements) given a rotation axis.

    Inputs
    --------------------
    q: np.array of shape(N, 3). Any point on the screw axis
    w: np.array of shape(N, 3). The direction vector for the screw axis.
    h: np.array of shape(N). Pitch along the screw axis (i.e., the ratio of linear displacement to angular displacement along the screw axis)
    
    Outputs
    --------------------
    w_so3: np.array of shape(N, 3, 3) with dtype float interpreted as a rotational displacement (i.e., rather than a change of coordinate frames).
    v: np.array of shape(N, 3) with dtype float interpreted as a linear velocity
    '''

    # Normalize the rotation axis
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)

    # Create the so(3) part of the se(3) twist
    w_so3 = make_so3_from_axis(w)

    # Create the linear part of the se(3) twist
    v = np.cross(q, w) + h[:, np.newaxis] * w
    
    return w_so3, v


def make_SE3_from_screw_axis(angle, q, w, h):
    '''
    Creates an SE(3) transformation (i.e., rigid body displacement in 3D) given a screw axis.
    First converts the screw axis to a twist, then applies the exponential map se(3)->SE(3). 

    Inputs
    --------------------
    angle: np.array of shape(N). The angular displacement in radian units.
    q: np.array of shape(N, 3). Any point on the screw axis
    w: np.array of shape(N, 3). The direction vector for the screw axis.
    h: np.array of shape(N). Pitch along the screw axis (i.e., the ratio of linear displacement to angular displacement along the screw axis)

    Outputs
    --------------------
    T: np.array of shape(N, 4, 4) with dtype float interpreted as a rigid body displacement (i.e., rather than a change of coordinate frames).
    '''

    # Normalize the rotation axis
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    N = w.shape[0]

    # Create the se(3) twist
    w_so3, v = make_se3_from_axis(q, w, h)

    # Exponential map se(3)->SE(3) 
    R = make_SO3_from_angle_axis(angle, w)
    angle = angle[:, np.newaxis, np.newaxis]
    t = (angle * np.repeat(np.eye(3)[np.newaxis, ...], N, axis=0)
         + (1 - np.cos(angle)) * w_so3 
         + (angle - np.sin(angle)) * (w_so3 @ w_so3)) @ v[..., np.newaxis] # The translational part of the SE(3) transform
    T = np.repeat(np.eye(4)[np.newaxis, ...], N, axis=0)
    T[:, 0:3, 0:3] = R
    T[:, 0:3, 3] = t[:, :, 0]
    
    return T