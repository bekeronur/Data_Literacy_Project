import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import cv2
from matplotlib import pyplot as plt
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation
from PIL import Image

from .dpc import DescriptorPointCloud
from . import utils


############################################################################################################################################ 
# Main RGBD Class
############################################################################################################################################ 

class RGBDImage:
    """ 
    Abstraction for an rgbd image.
    """

    def __init__(self, rgb, depth, camera_K, 
                 camera_pose_in_world_frame=None,
                 device=torch.device('cuda'),
                 descriptor_data=None,
                 part_masks=None,
                 part_data=None):
        '''
        rgb: np.array of shape(H, W, 3), 0-255 with uint8 dtype.
        depth: np.array of shape(H, W), in meters with float dtype.
        camera_K: np.array of shape(3, 3).
        camera_pose_in_world_frame: np.array of shape(4, 4), the rigid-body transformation of the camera frame as seen from the world frame.
        mask: np.array of shape(H, W), bool dtype.
        descriptor_data: dict with keys ['img_crop_pad', 'mask_crop_pad', 'global_descriptor', 'local_descriptors', 'local_descriptor_mask', 'local_descriptors_flat', 'local_descriptor_px_coords_HW'].
            img_crop_pad: PIL image.
            mask_crop_pad: PIL image.
            global_descriptor: torch.Size([1, 384]).
            local_descriptors: torch.Size([1, 28, 28, 384]).
            local_descriptor_mask: torch.Size([1, 28, 28]).
            local_descriptors_flat: torch.Size([285, 384]).
            local_descriptor_px_coords_HW: torch.Size([285, 2]).
        part_data: list of dicts, each with keys ['mask', 'descriptor_data'].
            mask: np.array of shape(H, W), bool dtype.
            descriptor_data: dict with keys ['img_crop_pad', 'mask_crop_pad', 'global_descriptor', 'local_descriptors', 'local_descriptor_mask', 'local_descriptors_flat', 'local_descriptor_px_coords_HW'].
                img_crop_pad: PIL image.
                mask_crop_pad: PIL image.
                global_descriptor: torch.Size([1, 384]).
                local_descriptors: torch.Size([1, 28, 28, 384]).
                local_descriptor_mask: torch.Size([1, 28, 28]).
                local_descriptors_flat: torch.Size([285, 384]).
                local_descriptor_px_coords_HW: torch.Size([285, 2]).
        '''

        # Bare minimum information necessary
        self.rgb, self.depth, self.camera_K, self.device = rgb, depth, camera_K, device 
        
        # Can be computed additionally as needed
        self.camera_pose_in_world_frame, self.descriptor_data, self.part_masks, self.part_data = camera_pose_in_world_frame, descriptor_data, part_masks, part_data

        # Move the provided descriptor data to device
        if descriptor_data is not None: move_descriptor_data_to_device(self.descriptor_data, device)
        if part_data is not None: 
            for part_datum in self.part_data:
                move_descriptor_data_to_device(part_datum['descriptor_data'], device)
            
        

############################################################################################################################################ 
# Utilities for reading RGBD 
############################################################################################################################################ 

def read_mid_rgbd_from_bag(bag_path):
    """
    Read the middle rgbd image in a bag file.
    """
    rgbd_reader = o3d.t.io.RSBagReader()
    rgbd_reader.open(bag_path)
    
    depth_scale = rgbd_reader.metadata.depth_scale
    camera_K = rgbd_reader.metadata.intrinsics.intrinsic_matrix
    stream_length = rgbd_reader.metadata.stream_length_usec
    
    rgbd_reader.seek_timestamp(stream_length//2)
    rgbd_frame = rgbd_reader.next_frame()
    
    rgb = np.asarray(rgbd_frame.color) # 0-255 uint8
    depth = np.asarray(rgbd_frame.depth).squeeze() / depth_scale # float, in meters
    rgbd = RGBDImage(rgb, depth, camera_K)

    return rgbd


def extract_all_rgbd_from_bag(rgbd_video_file, frames_folder=None):
    """
    Extract color and aligned depth frames and intrinsic calibration from an
    RGBD video file (currently only RealSense bag files supported). Folder
    structure is:
        <directory of rgbd_video_file/<rgbd_video_file name without extension>/
            {depth/00000.jpg,color/00000.png,intrinsic.json}
    """
    if frames_folder is None: 
        frames_folder = join(dirname(rgbd_video_file), asename(splitext(rgbd_video_file)[0]))
        
    path_intrinsic = join(frames_folder, "intrinsic.json")
    if isfile(path_intrinsic):
        warn(f"Skipping frame extraction for {rgbd_video_file} since files are"
             " present.")
    else:
        rgbd_video = o3d.t.io.RGBDVideoReader.create(rgbd_video_file)
        rgbd_video.save_frames(frames_folder)
    with open(path_intrinsic) as intr_file:
        intr = json.load(intr_file)
    depth_scale = intr["depth_scale"]
    return frames_folder, path_intrinsic, depth_scale
    
    
def extract_mid_rgbd_from_bag_list(SCAN_PATH, OUT_PATH):
    IMG_DIR = OUT_PATH + 'images/'
    DEPTH_DIR = OUT_PATH + 'depth/'
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)
    BAG_PATHS = glob.glob(SCAN_PATH + '*.bag')

    intrinsic = {}
    img_count = 0
    for BAG_PATH in bag_paths:
        rgbd_reader = o3d.t.io.RSBagReader()
        rgbd_reader.open(bag_path)
        
        depth_scale = rgbd_reader.metadata.depth_scale
        K = rgbd_reader.metadata.intrinsics.intrinsic_matrix
        stream_length = rgbd_reader.metadata.stream_length_usec
        intrinsic.update({'depth_scale': depth_scale, 'K': K})
        
        rgbd_reader.seek_timestamp(stream_length//2)
        rgbd_frame = rgbd_reader.next_frame()
        
        rgb = Image.fromarray(np.asarray(rgbd_frame.color))
        depth = Image.fromarray(np.asarray(rgbd_frame.depth).squeeze())
        rgb.save(IMG_DIR + '{}.jpg'.format(img_count))
        depth.save(DEPTH_DIR + '{}.png'.format(img_count))
        img_count = img_count + 1
    
    with open(OUT_PATH + 'intrinsic.json', 'w') as fp:
        json.dump(intrinsic, fp)


############################################################################################################################################ 
# Utilities to manage camera intrinsics/extrinsics
############################################################################################################################################

def camera_pose_to_opengl(camera_pose_in_wf):
    '''
    Convets a camera pose matrix from Vimex convention (z points towards the scene, y is down, x is right) to
    OpenGL convention (z points backwards from the scene, y is up, x is right)

    Inputs
    --------------------
    - camera_pose_in_wf: np.array of shape(4, 4), the rigid-body transformation of the camera frame as seen from the world frame.

    Outputs
    --------------------
    - camera_pose_in_wf_opengl: np.array of shape(4, 4), z and x axes of camera_pose_in_wf are flipped backwards
    '''
    camera_pose_in_wf_opengl = np.copy(camera_pose_in_wf)
    camera_pose_in_wf_opengl[:-1, 1] = -camera_pose_in_wf[:-1, 1]
    camera_pose_in_wf_opengl[:-1, 2] = -camera_pose_in_wf[:-1, 2]
    
    return camera_pose_in_wf_opengl


def create_K_from_opengl(fovy, H, W):
    '''
    Convets the fov and resolution parameters into the standard intrinsic matrix form (with a convention f = 1)

    Inputs
    --------------------
    - fovy: field of view along H axis, tan(fovy/2) = d/(2*fy)
    - H,W: image resolution

    Outputs
    --------------------
    - camera_K: np.array of shape(3, 3), the intrinsic matrix
    '''
    camera_K = np.eye(3)
    dy = (2 * np.tan(fovy/2))
    f_pixel = H / dy
    cx, cy = W // 2, H // 2
    camera_K[0,0], camera_K[1,1], camera_K[0, -1], camera_K[1, -1] = f_pixel, f_pixel, cx, cy
    return camera_K


############################################################################################################################################ 
# Utilities for extracting descriptors and masks
############################################################################################################################################ 

def extract_part_masks_(rgbd, mask_generator, verbose=False):
    '''
    mask_generator: segment_anything.SamAutomaticMaskGenerator object.

    Note: This function modifies the fields of the input rgbd image.
    '''
    
    # Extract part masks
    if verbose: print('Extracting part masks...')
    rgbd.part_masks = mask_generator.generate(rgbd.rgb)


def crop_around_mask(img, mask, padding_ratio=0.05, return_bb=True, resize=224):
    '''
    img: np.array of shape(H, W, 3), 0-255 with uint8 dtype
    mask: np.array of shape(H, W), bool dtype 

    Crops a square patch around a given mask.
    '''
    # Compute crop dimensions
    mask_pix_coords = mask.nonzero()
    H_mask_min, H_mask_max = mask_pix_coords[0].min(), mask_pix_coords[0].max()
    W_mask_min, W_mask_max = mask_pix_coords[1].min(), mask_pix_coords[1].max()
    H_mask_delta = H_mask_max - H_mask_min
    W_mask_delta = W_mask_max - W_mask_min

    # Crop image and mask
    img_crop = np.zeros_like(img)
    img_crop[mask] = img[mask]
    img_crop = img_crop[H_mask_min:H_mask_max+1, W_mask_min:W_mask_max+1, :]
    mask_crop = mask[H_mask_min:H_mask_max+1, W_mask_min:W_mask_max+1]

    # Pad to square
    if H_mask_delta > W_mask_delta:
        H_pad = int(H_mask_delta * 0.05)
        W_pad_before = (H_mask_delta + (2 * H_pad) - W_mask_delta) // 2
        W_pad_after = H_mask_delta + (2 * H_pad) - W_mask_delta - W_pad_before 
        img_crop_pad = np.pad(img_crop, ((H_pad, H_pad), (W_pad_before, W_pad_after), (0, 0)), 'constant')
        mask_crop_pad = np.pad(mask_crop, ((H_pad, H_pad), (W_pad_before, W_pad_after)), 'constant')
        upper_left_corner_H_W = [H_mask_min - H_pad, W_mask_min - W_pad_before]
        lower_right_corner_H_W = [ H_mask_max + H_pad, W_mask_max + W_pad_after]

    else:
        W_pad = int(W_mask_delta * 0.05)
        H_pad_before = (W_mask_delta + (2 * W_pad) - H_mask_delta) // 2
        H_pad_after = W_mask_delta + (2 * W_pad) - H_mask_delta - H_pad_before 
        img_crop_pad = np.pad(img_crop, ((H_pad_before, H_pad_after), (W_pad, W_pad), (0, 0)), 'constant')
        mask_crop_pad = np.pad(mask_crop, ((H_pad_before, H_pad_after), (W_pad, W_pad)), 'constant')
        upper_left_corner_H_W = [H_mask_min - H_pad_before, W_mask_min - W_pad]
        lower_right_corner_H_W = [ H_mask_max + H_pad_after, W_mask_max + W_pad]

    if resize is not None:
        img_crop_pad = F.resize(Image.fromarray(img_crop_pad), resize, interpolation=transforms.InterpolationMode.LANCZOS)
        mask_crop_pad = F.resize(Image.fromarray(mask_crop_pad), resize, interpolation=transforms.InterpolationMode.NEAREST_EXACT)
        
    if return_bb:
        return img_crop_pad, mask_crop_pad, upper_left_corner_H_W, lower_right_corner_H_W
    else:
        return img_crop_pad, mask_crop_pad


def extract_descriptor_data(img, mask, backbone, out_device=torch.device('cuda'), layer = 23, facet = 'value'):
    '''
    img: np.array of shape(H, W, 3), 0-255 with uint8 dtype
    mask: np.array of shape(H, W), bool dtype 
    '''
    H, W, _ = img.shape
    img_crop_pad, mask_crop_pad, upper_left_corner_H_W, lower_right_corner_H_W = crop_around_mask(img, mask)
    device = backbone.device
    with torch.no_grad():
        # Extract descriptors
        rgb = backbone.preprocess(img_crop_pad)
        global_descriptor, local_descriptors = backbone.extract_descriptors(rgb.to(device), layer, facet) # shape(B, D), shape(B, H, W, D)
        local_descriptor_mask = F.to_tensor(F.resize(mask_crop_pad, backbone.num_patches, interpolation=transforms.InterpolationMode.NEAREST_EXACT)).type(torch.bool).to(device)

    # Pixel coord map computation
    _, descriptor_H, descriptor_W = local_descriptor_mask.shape
    descriptor_pixel_coords_H_W = torch.stack(torch.meshgrid(torch.arange(descriptor_H), torch.arange(descriptor_W)), dim=-1).unsqueeze(0).float().to(local_descriptor_mask.device)
    H_min, H_max, W_min, W_max = upper_left_corner_H_W[0], lower_right_corner_H_W[0], upper_left_corner_H_W[1], lower_right_corner_H_W[1]
    descriptor_pixel_coords_H_W [..., 0] = (descriptor_pixel_coords_H_W[..., 0] / (descriptor_H - 1)) * (H_max - H_min) + H_min
    descriptor_pixel_coords_H_W [..., 1] = (descriptor_pixel_coords_H_W[..., 1] / (descriptor_W - 1)) * (W_max - W_min) + W_min
    
    local_descriptors_flat = local_descriptors[local_descriptor_mask]
    local_descriptor_px_coords_HW = descriptor_pixel_coords_H_W[local_descriptor_mask].type(torch.int).cpu()

    valid_px_index = (local_descriptor_px_coords_HW[:, 0] < H) \
                     * (local_descriptor_px_coords_HW[:, 1] < W) \
                     * (local_descriptor_px_coords_HW[:, 0] > 0) \
                     * (local_descriptor_px_coords_HW[:, 1] > 0) 
    
    local_descriptor_px_coords_HW = local_descriptor_px_coords_HW[valid_px_index]
    local_descriptors_flat = local_descriptors_flat[valid_px_index]
    
    filter_mask = mask[local_descriptor_px_coords_HW[:, 0], local_descriptor_px_coords_HW[:, 1]]
    local_descriptors_flat = local_descriptors_flat[filter_mask]
    local_descriptor_px_coords_HW = local_descriptor_px_coords_HW[filter_mask]

    descriptor_data = {'img_crop_pad': img_crop_pad, 
                       'mask_crop_pad': mask_crop_pad, 
                       'global_descriptor': global_descriptor.to(out_device), 
                       'local_descriptors': local_descriptors.to(out_device), 
                       'local_descriptor_mask': local_descriptor_mask.to(out_device), 
                       'local_descriptors_flat': local_descriptors_flat.to(out_device), 
                       'local_descriptor_px_coords_HW': local_descriptor_px_coords_HW.to(out_device)}
        
    return descriptor_data


def extract_descriptor_data_(rgbd, descriptor_model, mask=None, layer=23, facet='value'):
    '''
    descriptor_model: vimex.descriptor_backbones.ViTBackbone object.
    layer=23, facet='value': parameters directly passed to vimex.utils.extract_descriptors().

    Note: This function modifies the fields of the input rgbd image.
    '''
    if mask is None: mask = np.ones_like(rgbd.depth).astype(bool)
    rgbd.descriptor_data = extract_descriptor_data(rgbd.rgb, mask, descriptor_model, out_device=rgbd.device, layer=layer, facet=facet)
    

def extract_part_data_(rgbd, mask_generator, descriptor_model, layer=23, facet='value', verbose=False):
    '''
    mask_generator: segment_anything.SamAutomaticMaskGenerator object.
    descriptor_model: vimex.descriptor_backbones.ViTBackbone object.
    layer=23, facet='value': parameters directly passed to vimex.utils.extract_descriptors().

    Note: This function modifies the fields of the input rgbd image.
    '''

    if rgbd.part_masks is None:
        print('rgbd.part_masks was not found. Please call extract_part_masks_() first')
        
    for mask_data in part_masks:
        part_data.append({'mask': mask_data['segmentation']})

    # Extract part descriptors
    if verbose: print('Extracting part descriptors...')
    for part in part_data:
        descriptor_data_part = utils.extract_descriptors(rgbd.rgb, part['mask'], descriptor_model, out_device=rgbd.device, layer=layer, facet=facet)
        part.update({'descriptor_data':descriptor_data_part})
        
    rgbd.part_data = part_data


def move_descriptor_data_to_device(descriptor_data, device):
    descriptor_data['global_descriptor'].to(out_device) 
    descriptor_data['local_descriptors'].to(out_device) 
    descriptor_data['local_descriptor_mask'].to(out_device) 
    descriptor_data['local_descriptors_flat'].to(out_device) 
    descriptor_data['local_descriptor_px_coords_HW'].to(out_device) 


############################################################################################################################################ 
# Functions related to RGBD backprojection and visibility queries
############################################################################################################################################ 

def backproject_rgbd_to_pcd(rgbd, mask=None, max_depth=np.inf, visualize_camera=False):
    '''
    rgb: np.array of shape(H, W, 3), 0-255 with uint8 dtype
    depth: np.array of shape(H, W), in meters with float dtype
    K: np.array of shape(3, 3)
    camera_pose_from_world: np.array of shape(4, 4), the rigid-body transformation of the camera frame as seen from the world frame
    mask: np.array of shape(H, W), bool dtype
    '''

    if rgbd.camera_pose_in_world_frame is None: 
        camera_pose_in_world_frame = np.eye(4)
    else:
        camera_pose_in_world_frame = rgbd.camera_pose_in_world_frame
        
    # Get pixel coordinates
    camera_H, camera_W, _ = rgbd.rgb.shape
    xs, ys = np.meshgrid(np.arange(camera_W), np.arange(camera_H))
    xs = xs.reshape(1, camera_H, camera_W)
    ys = ys.reshape(1, camera_H, camera_W)

    # Backproject
    depth = rgbd.depth.reshape(1, camera_H, camera_W)
    rgb = rgbd.rgb
    xys = np.vstack((xs * depth , ys * depth, depth)) # shape (3, H, W)
    # Apply seg_mask
    if mask is not None: 
        xys = xys[:, mask]
        rgb = rgb[mask, :]

    rgb = rgb.reshape(-1, 3) / 255
    xys = xys.reshape(3, -1) # shape (3, H*W)
    
    # Filter inf depth
    valid_idx = (np.isfinite(xys.sum(0))) * (xys[-1, ...] != 0) * (xys[-1, ...] <= max_depth)
    xys = xys[:, valid_idx]
    rgb = rgb[valid_idx, :]
    
    # Convert from pix to camera coordinates
    xy_camera_coord = np.matmul(np.linalg.inv(rgbd.camera_K), xys) # shape (3, H*W)
    
    # Add points for camera visualization
    if visualize_camera:
        camera_x = np.array([[1], [0], [0]]) * 0.05
        camera_y = np.array([[0], [1], [0]]) * 0.05
        camera_z = np.array([[0], [0], [1]]) * 0.05
        camera_o = np.array([[0], [0], [0]])

        camera_x_color = np.array([[255, 0, 0]])
        camera_y_color = np.array([[66, 239, 245]])
        camera_z_color = np.array([[0, 0, 255]])
        camera_o_color = np.array([[255, 255, 255]])

        xy_camera_coord = np.concatenate([xy_camera_coord, camera_x, camera_y, camera_z, camera_o], axis=-1)
        rgb = np.concatenate([rgb, camera_x_color, camera_y_color, camera_z_color, camera_o_color], axis=0)

    # This part inverts the coordinate transformation
    xy_camera_coord_hom = np.concatenate([xy_camera_coord, np.ones([1, xy_camera_coord.shape[-1]])], axis=0)
    xy_world_coord = np.matmul(camera_pose_in_world_frame, xy_camera_coord_hom)[:-1, ...]  # shape (3, H*W)
    vertices = xy_world_coord.transpose(1, 0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd


def backproject_rgbd_to_dpc(rgbd, local_descriptor_px_coords_HW=None, local_descriptors_flat=None, max_depth=np.inf, filter_inf_depth=True):
    if local_descriptor_px_coords_HW is None or local_descriptors_flat is None:
        local_descriptor_px_coords_HW = rgbd.descriptor_data['local_descriptor_px_coords_HW']
        local_descriptors_flat = rgbd.descriptor_data['local_descriptors_flat']

    if rgbd.camera_pose_in_world_frame is None: 
        camera_pose_in_world_frame = np.eye(4)
    else:
        camera_pose_in_world_frame = rgbd.camera_pose_in_world_frame
        
    # Get pixel coordinates
    local_descriptor_px_coords_HW = local_descriptor_px_coords_HW.type(torch.int).cpu() # shape(2, N)
    rgb = rgbd.rgb[local_descriptor_px_coords_HW[:, 0], local_descriptor_px_coords_HW[:, 1]] / 255 # shape(N, 3)
    depth = rgbd.depth[local_descriptor_px_coords_HW[:, 0], local_descriptor_px_coords_HW[:, 1]] # shape(N)

    # Backproject
    xys = np.vstack((local_descriptor_px_coords_HW[:, 1] * depth, local_descriptor_px_coords_HW[:, 0] * depth, depth)) # shape (3, N)
    
    # Filter inf depth
    if filter_inf_depth:
        valid_idx = (np.isfinite(xys.sum(0))) * (xys[-1, ...] != 0) * (xys[-1, ...] <= max_depth)
        xys = xys[:, valid_idx]
        rgb = rgb[valid_idx, :]
        local_descriptors_flat = local_descriptors_flat[valid_idx, :]
    
    # Convert from pix to camera coordinates
    xy_camera_coord = np.matmul(np.linalg.inv(rgbd.camera_K), xys) # shape (3, H*W)

    # This part inverts the coordinate transformation
    xy_camera_coord_hom = np.concatenate([xy_camera_coord, np.ones([1, xy_camera_coord.shape[-1]])], axis=0)
    xy_world_coord = np.matmul(camera_pose_in_world_frame, xy_camera_coord_hom)[:-1, ...]  # shape (3, H*W)
    vertices = xy_world_coord.transpose(1, 0)

    # Create pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return DescriptorPointCloud(pcd, local_descriptors_flat, device=rgbd.device)


def get_occluded_point_mask(point_coords_world, rgbd, tolerance=0.01):
    '''
    point_coords_world: np.array of shape(N, 3)
    tolerance: the length of the vector that pushes visible surface points towards the camera origin before depth comparison.
    '''
    # Project onto the depth image and classify grid points based on visibility (0=> occluded, 1 =>empty) 
    world_pose_from_camera_frame = np.linalg.inv(rgbd.camera_pose_in_world_frame)
    point_coords_world_hom = np.concatenate([point_coords_world, np.ones([point_coords_world.shape[0], 1])], axis=-1)
    point_coords_camera = np.matmul(world_pose_from_camera_frame, point_coords_world_hom.transpose())[:-1, ...].transpose()  # shape (N, 3)
    
    xyd = np.matmul(rgbd.camera_K, point_coords_camera.transpose()).transpose() 
    point_depth = xyd[:,-1]
    px_coords_HW = np.flip((xyd // (point_depth.reshape(-1,1) + 1e-6))[:, :-1], axis=-1).astype(int)
    
    out_of_bounds_px_index = (px_coords_HW[:, 0] >= rgbd.depth.shape[0]) \
                             + (px_coords_HW[:, 1] >= rgbd.depth.shape[1]) \
                             + (px_coords_HW[:, 0] < 0) \
                             + (px_coords_HW[:, 1] < 0) \
    
    px_coords_HW[out_of_bounds_px_index, :] = np.array([0, 0])
    image_depth = rgbd.depth[px_coords_HW[:, 0], px_coords_HW[:, 1]]
    sensor_error_idx = (image_depth == 0.0)
    point_visibility = (point_depth <= image_depth - tolerance)
    point_visibility[out_of_bounds_px_index] = True
    point_visibility[sensor_error_idx] = True
    occluded_point_mask = ~point_visibility
    
    return occluded_point_mask
    

############################################################################################################################################ 
# Utilities to compute global/spatial descriptor similarities between RGBD images and their parts
############################################################################################################################################ 

def compute_global_descriptor_similarity(rgbd_img1, rgbd_img2):
    '''
    Inputs
    ------
    rgbd_img1, rgb_imgd2: vimex.rgbd.RGBDImage objects.

    Outputs
    -------
    global_descriptor_similarity = type int.
    '''

    if rgbd_img1.descriptor_data is None or rgbd_img2.descriptor_data is None:
        print('Descriptor data were not found. Please make sure to call extract_descriptor_data_() on both images first.')
    else:
        return (rgbd_img1.descriptor_data['global_descriptor'] @ rgbd_img2.descriptor_data['global_descriptor'].T).squeeze().item()


def compute_partwise_global_descriptor_similarity(rgbd_img1, rgbd_imgd2, return_similarity_matrix=False):
    '''
    Inputs
    ------
    rgbd_img1, rgb_imgd2: vimex.rgbd.RGBDImage objects.

    Outputs
    -------
    dict with keys:
        'num_bb_pairs': type int, number of parts whose global descriptors were bb pairs between two rgbd images.
        'bb_pairs_from_2_in_1': torch.tensor of type bool, with shape torch.Size([num_parts in rgbd_imgd2]). Each entry denotes whether the nearest neighbor for that part forms to a bb pair.
        'nn_from_2_in_1': torch.tensor of ints, with shape torch.Size([num_parts in rgbd_imgd2]). Each entry denotes the id of its nearest neighbor part in rgbd_img1.
        'similarity_matrix': torch.tensor of floats, with shape torch.Size([num_parts in rgbd_imgd1, num_parts in rgbd_imgd2]). The matrix of global descriptor inner-products.
    '''

    if rgbd_img1.part_data is None or rgbd_img2.part_data is None:
        print('Part data were not found. Please make sure to call extract_part_data_() on both images first.')
    
    global_descriptors_1 = torch.cat([part['descriptor_data']['global_descriptor'] for part in rgbd_img1.part_data])
    global_descriptors_2 = torch.cat([part['descriptor_data']['global_descriptor'] for part in rgbd_imgd2.part_data])
    similarity_matrix = global_descriptors_1 @ global_descriptors_2.T
    nn_from_1_in_2 = similarity_matrix.argmax(axis=-1)
    nn_from_2_in_1 = similarity_matrix.argmax(axis=0) 
    bb_pairs = torch.eq(nn_from_1_in_2[nn_from_2_in_1], torch.arange(nn_from_2_in_1.shape[0]).cuda())
    num_bb_pairs = bb_pairs.sum().item()

    output = {'similarity_matrix': similarity_matrix,
              'num_bb_pairs': num_bb_pairs,
              'bb_pairs_from_2_in_1': bb_pairs,
              'nn_from_2_in_1': nn_from_2_in_1}

    return output


def compute_partwise_spatial_descriptor_matches(rgbd_img1, rgbd_img2, partwise_global_descriptor_similarity, topK=5, return_descriptors=True):
    '''
    Inputs
    ------
    rgbd_img1, rgb_imgd2: vimex.rgbd.RGBDImage objects.
    partwise_global_descriptor_similarity: output of vimex.rgbd.compute_partwise_global_descriptor_similarity()

    Outputs
    -------
    dict with keys:
        'rgbd_imgX_local_descriptors_flat': torch.Size([num_matches, 1024]), spatial descriptor vectors for matches.
        'rgbd_imgX_local_descriptor_px_coords_HW': torch.Size([num_matches, 2]), pixel coordinates of the matches, [num_matches, 0] is H and [num_matches, 1] is W.
    '''
    
    v, i = torch.topk(partwise_global_descriptor_similarity['similarity_matrix'].flatten(), topK)
    top_part_idx = np.array(np.unravel_index(i.cpu().numpy(), partwise_global_descriptor_similarity['similarity_matrix'].shape)).T
    
    part2_id_list = []
    for part2_id in top_part_idx[:, -1]:
        if partwise_global_descriptor_similarity['bb_pairs_from_2_in_1'][part2_id].item() == True:
            part2_id_list.append(part2_id)
    
    part2_id_list = sorted(set(part2_id_list))

    rgbd_img1_local_descriptors_flat_list = []
    rgbd_img1_local_descriptor_px_coords_HW_list = []
    rgbd_img2_local_descriptors_flat_list = []
    rgbd_img2_local_descriptor_px_coords_HW_list = []
    for part2_id in part2_id_list:
        part2 = rgbd_img2.part_data[part2_id]
        if partwise_global_descriptor_similarity['bb_pairs_from_2_in_1'][part2_id].item() is not True: continue
        part2_local_descriptors_flat = part2['descriptor_data']['local_descriptors_flat']
        part2_local_descriptor_px_coords_HW = part2['descriptor_data']['local_descriptor_px_coords_HW']
    
        part1_id = partwise_global_descriptor_similarity['nn_from_2_in_1'][part2_id].item()
        part1 = rgbd_img1.part_data[part1_id]
        part1_local_descriptors_flat = part1['descriptor_data']['local_descriptors_flat']
        part1_local_descriptor_px_coords_HW = part1['descriptor_data']['local_descriptor_px_coords_HW']
    
        part_similarity_matrix = part1_local_descriptors_flat @ part2_local_descriptors_flat.T
        part_nn_from_1_in_2 = part_similarity_matrix.argmax(axis=-1)
        part_nn_from_2_in_1 = part_similarity_matrix.argmax(axis=0) 
        part_bb_pairs = torch.eq(part_nn_from_1_in_2[part_nn_from_2_in_1], torch.arange(part_nn_from_2_in_1.shape[0]).cuda())
        for part2_spatial_descriptor_id, part1_spatial_descriptor_id in enumerate(part_nn_from_2_in_1.tolist()):
            if part_bb_pairs[part2_spatial_descriptor_id].item() is not True: continue
            rgbd_img1_local_descriptors_flat_list.append(part1_local_descriptors_flat[part1_spatial_descriptor_id])
            rgbd_img1_local_descriptor_px_coords_HW_list.append(part1_local_descriptor_px_coords_HW[part1_spatial_descriptor_id])
            rgbd_img2_local_descriptors_flat_list.append(part2_local_descriptors_flat[part2_spatial_descriptor_id])
            rgbd_img2_local_descriptor_px_coords_HW_list.append(part2_local_descriptor_px_coords_HW[part2_spatial_descriptor_id])

    rgbd_img1_local_descriptors_flat = torch.stack(rgbd_img1_local_descriptors_flat_list)
    rgbd_img1_local_descriptor_px_coords_HW = torch.stack(rgbd_img1_local_descriptor_px_coords_HW_list)
    rgbd_img2_local_descriptors_flat = torch.stack(rgbd_img2_local_descriptors_flat_list)
    rgbd_img2_local_descriptor_px_coords_HW = torch.stack(rgbd_img2_local_descriptor_px_coords_HW_list)

    if return_descriptors:
        output = {'rgbd_img1_local_descriptors_flat': rgbd_img1_local_descriptors_flat,
                  'rgbd_img1_local_descriptor_px_coords_HW': rgbd_img1_local_descriptor_px_coords_HW,
                  'rgbd_img2_local_descriptors_flat': rgbd_img2_local_descriptors_flat,
                  'rgbd_img2_local_descriptor_px_coords_HW': rgbd_img2_local_descriptor_px_coords_HW}
    else:
        output = {'rgbd_img1_local_descriptor_px_coords_HW': rgbd_img1_local_descriptor_px_coords_HW,
                  'rgbd_img2_local_descriptor_px_coords_HW': rgbd_img2_local_descriptor_px_coords_HW}

    return output