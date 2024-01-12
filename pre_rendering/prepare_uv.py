"""
Prepare UV map for body model. This script needs a 00000.ply full body model. It can be converted from the 00000.obj in MeshLab and copied into 
/root/POV_Surgery/assets/transfer_surgical_Source/smplx_fitted_ply via docker cp.
Output is an updated body .obj file (example: 00000.obj) that is saved to .../transfer_surgical_Source/texture_rotate/default_smplx_male.obj
"""

import open3d as o3d
import numpy as np
import os
import shutil
from tqdm import tqdm

#ROOT_DIR = '/home/ray/Downloads/zju-ls-feng/output/smplx'
ROOT_DIR  = '/root/POV_Surgery/assets/transfer_surgical_Source'
OUT_dir = os.path.join(ROOT_DIR, 'texture_rotate')
TMP_dir = os.path.join(ROOT_DIR, 'tmp')
os.makedirs(TMP_dir,exist_ok=True)
os.makedirs(OUT_dir,exist_ok=True)
#mesh = o3d.io.read_triangle_mesh('/home/ray/code_release/hand_texture/transfer_surgical_Source/test1am.obj',enable_post_processing=True)
mesh = o3d.io.read_triangle_mesh('/root/POV_Surgery/assets/transfer_surgical_Source/test1am.obj',enable_post_processing=True)

mesh_save = mesh

# This is the folder that contains the rotated body model .ply that was generated with 'python transfer_pose.py'.
BASE_mesh = os.path.join(ROOT_DIR, 'rotated_body_ply')

tqdm.write("________________________________________________________________________________________________________________")

all_file_list = []
#for i in (range(0, 10000)):
for i in (range(0, 1)): # I use only 1 body .ply file (00000.obj) at the moment.
    
    this_mesh = os.path.join(BASE_mesh,str(i).zfill(5)+'.ply') # This has to be a body mesh, for example 00000.obj.
    
    if not os.path.exists(this_mesh):
         print("continued!  {}".format(this_mesh))
         continue
    all_file_list.append(this_mesh)
    tqdm.write("_____________ appended {}".format(this_mesh))


for i in tqdm(range(0, len(all_file_list)+1)):
    this_mesh = os.path.join(BASE_mesh,str(i).zfill(5)+'.ply')
    if not os.path.exists(this_mesh):
        continue

    # Reads in a body .ply file but updates the respective body .obj file with vertices and triangles. Why is a .ply file being used here and not an .obj file as well?
    temp = o3d.io.read_triangle_mesh(this_mesh)
    # mesh.triangle_uvs
    mesh = mesh_save
    mesh.vertices = temp.vertices
    mesh.triangles = temp.triangles
    # os.rename(os.path.join(OUT_dir, 'default_smplx_male.obj'), os.path.join(OUT_dir, str(i).zfill(5) + '.obj'))
    # o3d.io.write_triangle_mesh(os.path.join(OUT_dir,'default_smplx_male.obj'),mesh)
    o3d.io.write_triangle_mesh(os.path.join(TMP_dir, 'default_smplx_male.obj'), mesh)
    tqdm.write("____ write____ {}".format(os.path.join(TMP_dir, 'default_smplx_male.obj')))

    shutil.move(os.path.join(TMP_dir, 'default_smplx_male.obj'),os.path.join(OUT_dir, str(i).zfill(5) + '.obj'))
    if i == 0 :
        tqdm.write("__________i = 0")
        shutil.move(os.path.join(TMP_dir, 'default_smplx_male.mtl'), os.path.join(OUT_dir, 'default_smplx_male.mtl'))
        shutil.move(os.path.join(TMP_dir, 'default_smplx_male_0.png'), os.path.join(OUT_dir, 'default_smplx_male_0.png'))

