# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
import sys
import scipy.io as sio
import trimesh
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os, time
import argparse
import mano

from grabnet.tools.utils import euler
from grabnet.tools.cfg_parser import Config
from grabnet.tests.tester import Tester
from psbody.mesh.colors import name_to_rgb
from grabnet.tools.train_tools import point2point_signed
from grabnet.tools.utils import aa2rotmat
from grabnet.tools.utils import makepath
from grabnet.tools.utils import to_cpu
from grabnet.tools.vis_tools import points_to_spheres
from grabnet.tools.meshviewer import Mesh, MeshViewer, points2sphere
from bps_torch.bps import bps_torch


def create_rotation_matrices(rotation_angles):
    """
    Create rotation matrices based on a list of rotation angle sets.

    Args:
        rotation_angles (list): List of lists where each inner list represents a set of x, y, and z rotation angles in degrees.

    Returns:
        list: List of rotation matrices.
    """
    rotation_matrices = []

    for angles in rotation_angles:
        if len(angles) != 3:
            raise ValueError("Each inner list should contain exactly three rotation angles (x, y, z) in degrees.")

        x_degrees, y_degrees, z_degrees = angles

        # Convert degrees to radians
        x_radians = np.radians(x_degrees)
        y_radians = np.radians(y_degrees)
        z_radians = np.radians(z_degrees)

        # Create the rotation matrix for each axis
        x_rotation_matrix = np.array([[1, 0, 0],
                                      [0, np.cos(x_radians), -np.sin(x_radians)],
                                      [0, np.sin(x_radians), np.cos(x_radians)]])

        y_rotation_matrix = np.array([[np.cos(y_radians), 0, np.sin(y_radians)],
                                      [0, 1, 0],
                                      [-np.sin(y_radians), 0, np.cos(y_radians)]])

        z_rotation_matrix = np.array([[np.cos(z_radians), -np.sin(z_radians), 0],
                                      [np.sin(z_radians), np.cos(z_radians), 0],
                                      [0, 0, 1]])

        # Combine the rotations into a single rotation matrix
        rand_rotmat = x_rotation_matrix.dot(y_rotation_matrix).dot(z_rotation_matrix)
        
        rotation_matrices.append(rand_rotmat)

    return rotation_matrices

def get_meshes(object_data, coarse_net, refine_net, rh_model, save=False, save_dir=None,mat_name =None):
    """
    Generates hand and object meshes using CoarseNet and RefineNet, and optionally saves the generated meshes.

    Args:
        object_data (dict): Dictionary containing BPS encoded objects, vertices, mesh objects, and rotation matrices.
        coarse_net (torch.nn.Module): CoarseNet for initial pose estimation.
        refine_net (torch.nn.Module): RefineNet for pose refinement.
        rh_model (torch.nn.Module): The MANO right hand model.
        save (bool, optional): Whether to save the generated meshes. Defaults to False.
        save_dir (str, optional): Directory to save the generated meshes. Defaults to None.
        mat_name (str, optional): Name for the saved .mat file. Defaults to None.

    Returns:
        list: List of generated meshes.
    """

    # Disable gradient computation for efficiency
    with torch.no_grad():

        # Determine the device (GPU if available, otherwise CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Sample poses using CoarseNet
        coarse_net_output, zgen = coarse_net.sample_poses(object_data['bps_object'])

        # Generate hand vertices of the MANO right hand model using the sampled poses
        verts_rh_gen_cnet = rh_model(**coarse_net_output).vertices
        zgen = zgen.cpu().float().numpy()

         # Compute point-to-point signed distances
        _, h2o, _ = point2point_signed(verts_rh_gen_cnet, object_data['verts_object'].to(device))

        # Prepare data for RefineNet
        coarse_net_output['trans_rhand_f'] = coarse_net_output['transl']
        coarse_net_output['global_orient_rhand_rotmat_f'] = aa2rotmat(coarse_net_output['global_orient']).view(-1, 3, 3)
        coarse_net_output['fpose_rhand_rotmat_f'] = aa2rotmat(coarse_net_output['hand_pose']).view(-1, 15, 3, 3)
        coarse_net_output['verts_object'] = object_data['verts_object'].to(device)
        coarse_net_output['h2o_dist'] = h2o.abs()

        # Refine the poses using RefineNet
        refine_net_output = refine_net(**coarse_net_output)

        # Save generated data to a dictionary
        save_dict = {'zgen': zgen, 'rotmat': object_data['rotmat']}
        for this_key in refine_net_output.keys():
            save_dict[this_key] = refine_net_output[this_key].detach().cpu().numpy()

        # Generate final hand vertices and joints using the refined poses  
        out_1= rh_model(**refine_net_output)
        verts_rh_gen_rnet = out_1.vertices
        save_dict['joints'] = out_1.joints.cpu().numpy()
        save_dict['vert'] = out_1.vertices.cpu().numpy()

        # Create output directory if it doesn't exist
        os.makedirs('./OUT', exist_ok=True)
        sio.savemat('./OUT/generate.mat', save_dict)
        
        gen_meshes = []
        for cId in range(0, len(object_data['bps_object'])):
            try:
                # Attempt to retrieve the original object mesh
                obj_mesh = object_data['mesh_object'][cId]
            except:
                # If not available, create a sphere mesh from the object vertices
                obj_mesh = points2sphere(points=to_cpu(object_data['verts_object'][cId]), radius=0.002, vc=name_to_rgb['yellow'])
                print('points ')

            # Create the hand mesh from the refined vertices
            hand_mesh_gen_rnet = Mesh(vertices=to_cpu(verts_rh_gen_rnet[cId]), faces=rh_model.faces, vc=[245, 191, 177])

            # Rotate the meshes if rotation matrix is available
            if 'rotmat' in object_data:
                rotmat = object_data['rotmat'][cId].T
                obj_mesh = obj_mesh.rotate_vertices(rotmat)
                hand_mesh_gen_rnet.rotate_vertices(rotmat)
                combined = trimesh.util.concatenate( [hand_mesh_gen_rnet, obj_mesh] )

            gen_meshes.append([obj_mesh, hand_mesh_gen_rnet])
            if save:
                # Create the save directory if it doesn't exist
                save_path = os.path.join(save_dir)
                makepath(save_path)
                # Save the combined mesh, hand mesh, and object mesh
                combined.export(os.path.join(save_path,str(cId).zfill(6)+'_Combined.ply'))
                hand_mesh_gen_rnet.export(os.path.join(save_path,str(cId).zfill(6)+'_Hand.ply'))
                obj_mesh.export(os.path.join(save_path,str(cId).zfill(6)+'_Object.ply'))

        return gen_meshes


def grab_new_objs(grabnet, objs_path, rotation_angles, rot=True, n_samples=10, scale=1.,save_name = 'meshes'):
    """
    Grabs new objects and generates corresponding hand meshes using the GrabNet model.

    Args:
        grabnet (Tester): An instance of the GrabNet Tester class, containing the trained models and configurations.
        objs_path (str or list): Path or list of paths to the 3D object meshes.
        rotation_angles (list): List of lists where each inner list represents a set of x, y, and z rotation angles in degrees.
        rot (bool, optional): Whether to apply random rotations to the objects. Defaults to True.
        n_samples (int, optional): Number of grasp samples to generate. Defaults to 10.
        scale (float, optional): Scaling factor for the 3D objects. Defaults to 1.0.
        save_name (str, optional): Name to use when saving the generated meshes. Defaults to 'meshes'.

    Returns:
        None: The function saves the generated meshes to the specified directory and file.
    """

    # Set CoareNet and RefineNet to evaluation mode
    grabnet.coarse_net.eval()
    grabnet.refine_net.eval()

    # Load the MANO right hand model
    rh_model = mano.load(model_path=grabnet.cfg.rhm_path,
                         model_type='mano',
                         num_pca_comps=45,
                         batch_size=n_samples,
                         flat_hand_mean=True).to(grabnet.device)

    # Assign the hand model to the refinement network
    grabnet.refine_net.rhm_train = rh_model

    # Log the start of the object grabbing process
    grabnet.logger(f'################# \n'
                   f'Grabbing the object!'
                   )

    # Initialize the BPS encoder
    bps = bps_torch(custom_basis=grabnet.bps)

    # Ensure objs_path is a list
    if not isinstance(objs_path, list):
        objs_path = [objs_path]

    # Process each sample
    for new_obj in objs_path:
        # Generate rotation matrices from the provided rotation angles
        rotations_in_deg = create_rotation_matrices(rotation_angles)
        rand_rotmat = euler(rotations_in_deg)

        # Initialize the dictionary to store object data
        object_data = {'bps_object': [], 'verts_object': [], 'mesh_object': [], 'rotmat': []}

        # Process each sample
        for samples in range(n_samples):
            # Load the object vertices, mesh, and rotation matrix
            verts_obj, mesh_obj, rotmat = load_obj_verts(new_obj, rand_rotmat[samples], rndrotate=rot, scale=scale)

            # Encode the object using BPS and add it to the dictionary
            bps_object = bps.encode( torch.tensor(verts_obj.astype(np.float32)), feature_type='dists')['dists']
            object_data['bps_object'].append(bps_object.to(grabnet.device))
            object_data['verts_object'].append(torch.from_numpy(verts_obj.astype(np.float32)).unsqueeze(0))
            object_data['mesh_object'].append(mesh_obj)
            object_data['rotmat'].append(rotmat)
            obj_name = os.path.basename(new_obj)

        # Concatenate the lists of tensors into single tensors
        object_data['bps_object'] = torch.cat(object_data['bps_object'])
        object_data['verts_object'] = torch.cat(object_data['verts_object'])

         # Create the directory to save the results
        save_dir = os.path.join('./OUT', 'generated_hand_grasp_meshes')
        
        # Log the saving process
        grabnet.logger(f'#################\n'
                       f'                   \n'
                       f'Saving results for the {obj_name.upper()}'
                       f'                      \n')

        # Generate meshes and save them
        gen_meshes = get_meshes(object_data=object_data,
                                coarse_net=grabnet.coarse_net,
                                refine_net=grabnet.refine_net,
                                rh_model=rh_model,
                                save=True,
                                save_dir=save_dir,
                                mat_name = save_name
                                )

        # Save the generated meshes to a file
        torch.save(gen_meshes, 'OUT/' +save_name+'.pt')


def load_obj_verts(mesh_path, rand_rotmat, rndrotate=True, scale=1., n_sample_verts=3000):
    """
    Loads and processes the vertices of an object mesh.

    Args:
        mesh_path (str): Path to the mesh file.
        rand_rotmat (numpy.ndarray): Rotation matrix to apply to the vertices.
        rndrotate (bool, optional): Whether to apply the random rotation. Defaults to True.
        scale (float, optional): Scaling factor for the mesh. Defaults to 1.0.
        n_sample_verts (int, optional): Number of vertices to sample from the mesh. Defaults to 3000.

    Returns:
        tuple: Sampled vertices, processed mesh object, and applied rotation matrix.
    """
    # Set a random seed for reproducibility
    np.random.seed(100)
    # Load the mesh from the given file path and scale it
    obj_mesh = Mesh(filename=mesh_path, vscale=scale)

    # if the object has no texture, make it yellow

    # Center and scale the object
    max_length = np.linalg.norm(obj_mesh.vertices, axis=1).max()
    if max_length > 1:
        re_scale = max_length / .08
        print(f'The object is very large, down-scaling by {re_scale} factor')
        obj_mesh.vertices[:] = obj_mesh.vertices / re_scale

    # Get the vertices of the object
    object_fullpts = obj_mesh.vertices
    maximum = object_fullpts.max(0, keepdims=True)
    minimum = object_fullpts.min(0, keepdims=True)

    # Calculate the offset to center the object
    offset = (maximum + minimum) / 2
    verts_obj = object_fullpts - offset
    obj_mesh.vertices[:] = verts_obj

    # Apply the rotation if rndrotate is True
    if rndrotate:
        obj_mesh.rotate_vertices(rand_rotmat)
    else:
        rand_rotmat = np.eye(3)

    # Subdivide the mesh if the number of vertices is less than n_sample_verts
    while (obj_mesh.vertices.shape[0]<n_sample_verts):
        new_mesh = obj_mesh.subdivide()
        obj_mesh = Mesh(vertices=new_mesh.vertices,
                        faces = new_mesh.faces,
                        visual = new_mesh.visual)

    # Get the vertices of the object and sample n_sample_verts vertices
    verts_obj = obj_mesh.vertices
    verts_sampled,_ = trimesh.sample.sample_surface_even(obj_mesh, n_sample_verts, radius=None)

    return verts_sampled, obj_mesh, rand_rotmat


if __name__ == '__main__':
    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser(description='GrabNet-Testing')

    # Add argument for the path to the 3D object mesh or point cloud
    parser.add_argument('--obj-path', required=True, type=str,
                        help='The path to the 3D object Mesh or Pointcloud')

    # Add argument for the path to the folder containing the MANO_RIGHT model
    parser.add_argument('--rhm-path', required=True, type=str,
                        help='The path to the folder containing MANO_RIGHT model')
    
    # Add argument for the scaling factor for the 3D object
    parser.add_argument('--scale', default=1., type=float,
                        help='The scaling for the 3D object')
    
    # Add argument for the number of grasps to generate
    parser.add_argument('--n-samples', default=100, type=int,
                        help='number of grasps to generate')

    # Add argument for the name to use when saving the generated meshes
    parser.add_argument('--save_name', default='meshes', type=str,
                        help='name to save')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Assign the parsed arguments to variables
    obj_path = args.obj_path
    rhm_path = args.rhm_path
    scale = args.scale
    n_samples = args.n_samples
    save_name = args.save_name

    # Get the current working directory and set it
    cwd = os.getcwd()
    work_dir = cwd + '/logs'

    # Define the paths to CoarseNet and RefineNet models and BPS configuration
    best_cnet = 'grabnet/models/coarsenet.pt'
    best_rnet = 'grabnet/models/refinenet.pt'
    bps_dir = 'grabnet/configs/bps.npz'

    config = {
        'work_dir': work_dir,
        'best_cnet': best_cnet,
        'best_rnet': best_rnet,
        'bps_dir': bps_dir,
        'rhm_path': rhm_path

    }

    # Initialize the configuration object
    cfg = Config(**config)

    # Set scale and number of grasps
    scale = 1
    nr_grasps = 100 # This number should be adjusted based on individual requirements.

    # Below, we create default rotation angles of [0, 0, 0] for the ultrasound probe mesh which influence the orientation of the generated GrabNet-based grasps. 
    # Different rotation angles can be tried here. 
    rotation_angles = [[0, 0, 0] for _ in range(nr_grasps)] 

    # Initialize the GrabNet Tester with the configuration
    grabnet = Tester(cfg=cfg)

    # Generate and save the grasps for the object
    grab_new_objs(grabnet, obj_path, rotation_angles, rot=True, n_samples=n_samples, scale=scale, save_name = save_name)