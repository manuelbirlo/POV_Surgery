import sys
sys.path.append('.')
sys.path.append('..')
import torch
import numpy as np
from psbody.mesh.colors import name_to_rgb
import os
import time
from grabnet.tools.meshviewer import Mesh
import trimesh
import re

save_name = 'test_meshes'
mesh_folder_path = './OUT'
out_dir = os.path.join(mesh_folder_path, save_name)

out_dir_combined_hand_and_obj = os.path.join(out_dir, 'Combined')
os.makedirs(out_dir_combined_hand_and_obj, exist_ok=True)

out_dir_hand = os.path.join(out_dir, 'Hand')
os.makedirs(out_dir_hand, exist_ok=True)

out_dir_obj = os.path.join(out_dir, 'Object')
os.makedirs(out_dir_obj, exist_ok=True)

mesh_nr = -1
for mesh_file in os.listdir(mesh_folder_path):
    if mesh_file.endswith(".pt"):

        meshes = torch.load(os.path.join(mesh_folder_path, mesh_file))
        
        
        os.makedirs(out_dir, exist_ok=True)
        N = len(meshes)
        
        obj = meshes[0]
        hand = meshes[1]

        hand.set_vertex_colors(vc=[245, 191, 177])

        def_color = np.array([[102, 102, 102, 255]])
        if hasattr(obj.visual, 'vertex_colors') and (obj.visual.vertex_colors == np.repeat(def_color, obj.vertices.shape[0], axis=0)).all():
            obj.set_vertex_colors(vc=name_to_rgb['yellow'])
        combined = trimesh.util.concatenate( [hand, obj] )

        # Use a regular expression to find the mesh number in the file name.
        mesh_nr_regex_result = re.search(r"_(\d+)\.", mesh_file)
        if mesh_nr_regex_result:
            mesh_nr = int(mesh_nr_regex_result.group(1))
        else:
            mesh_nr = mesh_nr + 1
        
        # Ensure the directory exists
        temp = combined.export(os.path.join(out_dir_combined_hand_and_obj,str(mesh_nr).zfill(6)+'_Combined.ply'))
        temp = hand.export(os.path.join(out_dir_hand,str(mesh_nr).zfill(6)+'_Hand.ply'))
        temp = obj.export(os.path.join(out_dir_obj,str(mesh_nr).zfill(6)+'_Object.ply'))
