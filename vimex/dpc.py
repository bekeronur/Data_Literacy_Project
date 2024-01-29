import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation
import pdb

from . import utils


############################################################################################################################################ 
# Main DPC (descriptor pointcloud) Class
############################################################################################################################################ 

class DescriptorPointCloud:
    """ 
    Abstraction for a descriptor pointcloud.
    """

    def __init__(self, pcd, descriptors, kdtree=None, device=torch.device('cuda')):
        '''
        visible_pcd: o3d.geometry.PointCloud() object, with N points. Corresponds to visible surface points.
        descriptors: torch.Size([N, D]). D dimensional descriptors of the N points in pcd. 
        
        empty_volume_pcd: o3d.geometry.PointCloud() object, with K points. Corresponds to empty space points.
        occluded_volume_pcd: o3d.geometry.PointCloud() object, with L points. Corresponds to occluded space points.
        '''

        # Bare minimum information necessary
        self.pcd, self.descriptors = pcd, descriptors.to(device)
        self.points = torch.from_numpy(np.asarray(pcd.points)).to(device)
        self.device = device
        
        # Additional information
        self.kdtree = kdtree

    def to(device):
        self.descriptors.to(device)
        self.points.to(device)

    def create_kdtree(overwrite=False):
        if self.kdtree is None or overwrite:
            self.kdtree = o3d.geometry.KDTreeFlann(self.pcd) 


############################################################################################################################################ 
# Basic operations
############################################################################################################################################ 

def average_array_using_trace(array, trace):
    N, D = array.shape
    M = len(trace)
    if torch.is_tensor(array):
        new_array = torch.zeros([M, D]).to(array.device)
    else:
        new_array = np.zeros([M, D])

    for voxel_id, point_ids in enumerate(trace):
        new_array[voxel_id, :] = array[point_ids].mean(axis=0)
        
    return new_array


def voxel_downsample(dpc, voxel_size):
    pcd_down, _, trace = dpc.pcd.voxel_down_sample_and_trace(voxel_size, dpc.pcd.get_min_bound(), dpc.pcd.get_max_bound())
    descriptors_down = average_array_using_trace(dpc.descriptors, trace)
    return DescriptorPointCloud(pcd_down, descriptors_down, device=dpc.device)


def filter(dpc, outlier_std_ratio, nb_neighbors=20):
    pcd_filtered, ind = dpc.pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=outlier_std_ratio)
    descriptors_filtered = dpc.descriptors[ind]
    return DescriptorPointCloud(pcd_filtered, descriptors_filtered, device=dpc.device)


def join_descriptor_pointclouds(dpcs):
    # Join pcds
    pcd_joint = o3d.geometry.PointCloud()
    for dpc in dpcs: pcd_joint += dpc.pcd
    descriptors_joint = torch.concatenate([dpc.descriptors for dpc in dpcs])
    return DescriptorPointCloud(pcd_joint, descriptors_joint, device=dpcs[0].device)


def calculate_surface_curvature(pcd, radius=0.1, max_nn=30):
    pcd_n = copy.deepcopy(pcd)
    pcd_n.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    covs = np.asarray(pcd_n.covariances)
    vals, vecs = np.linalg.eig(covs)
    curvature = np.min(vals, axis=1)/np.sum(vals, axis=1)
    return curvature


############################################################################################################################################ 
# Voxel grid visibility analysis
############################################################################################################################################ 

def create_voxel_grid_from_center(center, bb_edges, voxel_size=0.01):
    bb_max_bound = center + bb_edges/2 + 2 * voxel_size
    bb_min_bound = center - bb_edges/2 - 2 * voxel_size
    grid_coords_world = np.stack(np.meshgrid(np.arange(bb_min_bound[0], bb_max_bound[0], voxel_size),
                                             np.arange(bb_min_bound[1], bb_max_bound[1], voxel_size), 
                                             np.arange(bb_min_bound[2], bb_max_bound[2], voxel_size)), axis=-1).transpose(1,0,2,3).reshape(-1, 3) # shape(N, 3)

    pcd_grid = o3d.geometry.PointCloud()
    pcd_grid.points = o3d.utility.Vector3dVector(grid_coords_world)

    return pcd_grid


def create_visibility_data(pcd_grid, rgbd_img):
    occluded_point_mask = get_occluded_point_mask(np.array(pcd_grid.points), rgbd_img)
    empty_point_mask = ~occluded_point_mask 
    empty_volume_pcd = pcd_grid.select_by_index(empty_point_mask.nonzero()[0])
    empty_volume_pcd.paint_uniform_color([0,0,1])
    occluded_volume_pcd = pcd_grid.select_by_index(occluded_point_mask.nonzero()[0])
    occluded_volume_pcd.paint_uniform_color([0,1,0])
    return empty_volume_pcd, occluded_volume_pcd


############################################################################################################################################ 
# Visualization functions
############################################################################################################################################

def visualize_dpc_similarities(obj_dpc_1, obj_dpc_2, 
                               translation = None, 
                               select_color = [0, 0, 1],
                               point_size=0.05):
    if translation is None:
        translation = np.array([obj_dpc_1.pcd.get_axis_aligned_bounding_box().get_extent()[0]*1.5, 0, 0])
        
    obj_dpc_1_pcd_vis_color = copy.deepcopy(obj_dpc_1.pcd)
    obj_dpc_2_pcd_vis_color = copy.deepcopy(obj_dpc_2.pcd)
    obj_dpc_2_pcd_vis_sim = copy.deepcopy(obj_dpc_2.pcd)
    while True:
        # Pick points on the first object
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(obj_dpc_1_pcd_vis_color)
        vis.run()  # user picks points
        vis.destroy_window()
        picked_points = vis.get_picked_points()
        if len(picked_points) < 1: break
            
        # Show similarities on the second object
        selection_indicator = o3d.geometry.TriangleMesh.create_sphere(radius = point_size, resolution=5)
        selection_indicator.translate(obj_dpc_1_pcd_vis_color.points[picked_points[0]])
        selection_indicator.paint_uniform_color(select_color)
        
        obj_1_selected_descriptor = obj_dpc_1.descriptors[picked_points[0]]
        obj_2_similarity_weights = obj_dpc_2.descriptors @ obj_1_selected_descriptor
        
        obj_2_similarity_weights = (obj_2_similarity_weights - obj_2_similarity_weights.min()) / (obj_2_similarity_weights.max() - obj_2_similarity_weights.min())
        obj_dpc_2_pcd_vis_sim.colors = o3d.utility.Vector3dVector(obj_2_similarity_weights.unsqueeze(-1).repeat(1, 3).cpu().numpy())
    
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(obj_dpc_1_pcd_vis_color + obj_dpc_2_pcd_vis_sim.translate(translation) + obj_dpc_2_pcd_vis_color.translate(translation*2))
        vis.add_geometry(selection_indicator)
        vis.get_render_option().point_size = 15
        vis.run()
        vis.destroy_window()


############################################################################################################################################ 
# Descriptor Matching
############################################################################################################################################

def bbp_matching(similarity_matrix):
    '''
    similarity_matrix: torch.Size([N, M]). Descriptor similarities between DPCs with N and M points each. 
    '''
    nn_from_1_in_2 = similarity_matrix.argmax(axis=-1)
    nn_from_2_in_1 = similarity_matrix.argmax(axis=0) 
    bb_pairs_from_2_in_1 = torch.eq(nn_from_1_in_2[nn_from_2_in_1], torch.arange(nn_from_2_in_1.shape[0]).cuda())
    num_bb_pairs = bb_pairs_from_2_in_1.sum().item()
    bb_pair_ids_1_and_2 = torch.stack([nn_from_2_in_1[bb_pairs_from_2_in_1], bb_pairs_from_2_in_1.nonzero().squeeze()], dim=-1).cpu().numpy()
        
    return bb_pair_ids_1_and_2 

def visualize_bb_pairs(pcd_1, pcd_2, bb_pair_ids_1_and_2, translation = None, num_pairs_to_show=20, point_size=0.02):
    if translation is None:
        translation = np.array([pcd_1.get_axis_aligned_bounding_box().get_extent()[0]*1.5, 0, 0])

    pcd_2 = copy.deepcopy(pcd_2)
    pcd_2.translate(translation)
    
    bbp_markers_1 = []
    bbp_markers_2 = []
    points_1 = []
    points_2 = []
    colors = []
    for pair_id, bb_pair in enumerate(bb_pair_ids_1_and_2):
        # Create bbp markers
        bbp_color = np.random.rand(3)
        bbp_marker_1 = o3d.geometry.TriangleMesh.create_sphere(radius = point_size, resolution=5)
        bbp_marker_1.translate(pcd_1.points[bb_pair[0]])
        bbp_marker_1.paint_uniform_color(bbp_color)
        bbp_marker_2 = o3d.geometry.TriangleMesh.create_sphere(radius = point_size, resolution=5)
        bbp_marker_2.translate(pcd_2.points[bb_pair[1]])
        bbp_marker_2.paint_uniform_color(bbp_color)
        
        colors.append(bbp_color)
        bbp_markers_1.append(bbp_marker_1)
        bbp_markers_2.append(bbp_marker_2)
        points_1.append(pcd_1.points[bb_pair[0]])
        points_2.append(pcd_2.points[bb_pair[1]])

    for i in range(0, bb_pair_ids_1_and_2.shape[0] - num_pairs_to_show, num_pairs_to_show):
        points = [val for pair in zip(points_1[i:i+num_pairs_to_show], points_2[i:i+num_pairs_to_show]) for val in pair]
        lines = [[2*pair_id, 2*pair_id+1] for pair_id in range(num_pairs_to_show)]
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                        lines=o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(colors[i:i+num_pairs_to_show])
        o3d.visualization.draw_geometries([line_set] + bbp_markers_1[i:i+num_pairs_to_show] + bbp_markers_2[i:i+num_pairs_to_show] + [pcd_1, pcd_2])