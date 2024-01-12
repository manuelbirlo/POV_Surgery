import os
import pickle
from scipy.spatial.transform import Rotation as R
# original_dict = pickle.load(open('/Users/wangrui/Downloads/debug_SMPL/00261.pkl','rb'))
import open3d as o3d
from tqdm import tqdm
from slerp_utils import quaternion_from_matrix, quaternion_slerp, quaternion_matrix


######################################3
#base_dir = '/home/ray/Downloads/zju-ls-feng/output/smplx'
# Assumption: The body model from /SMPLX_texture/SMPLX_texture/transfer_surgical_Source has been copied into /root/POV_Surgery/assets/transfer_surgical_Source.
# The model model can be downloaded from https://drive.google.com/drive/folders/1nSDig2cEHscCPgG10-VcSW3Q1zKge4tP (SMPLX_texture.zip)
base_dir = '/root/POV_Surgery/assets/transfer_surgical_Source'
######################################3

# The following two directories should exist upfront and contain objct .ply files and a whole body .ply file. 
rotated_body_ply = os.path.join(base_dir,'rotated_body_ply')
rotated_object_ply = os.path.join(base_dir,'rotated_object_ply')
os.makedirs(rotated_object_ply,exist_ok=True)
os.makedirs(rotated_body_ply,exist_ok=True)

# The following two directories are being created in this script and don't need to exist upfront.
object_to_be_transfered = os.path.join(base_dir,'smplx_fitted_ply_Object')
body_to_be_transfered = os.path.join(base_dir, 'smplx_fitted_ply')
os.makedirs(object_to_be_transfered,exist_ok=True)
os.makedirs(body_to_be_transfered,exist_ok=True)

#list_of_ply = os.listdir(body_to_be_transfered)
list_of_ply = os.listdir(object_to_be_transfered)

for i in tqdm(range(len(list_of_ply))):

    mesh_body = o3d.io.read_triangle_mesh(os.path.join(body_to_be_transfered, list_of_ply[i]))
    rr = R.from_euler('xyz',[90,0,0],degrees=True).as_matrix()
    mesh_body.rotate(rr, center=(0, 0, 0))
    print("___ WRITING body : {}".format(os.path.join(rotated_body_ply,list_of_ply[i])))
    o3d.io.write_triangle_mesh(os.path.join(rotated_body_ply,list_of_ply[i]), mesh_body)


    mesh_object = o3d.io.read_triangle_mesh(os.path.join(object_to_be_transfered, list_of_ply[i]))
    rr = R.from_euler('xyz',[90,0,0],degrees=True).as_matrix()
    mesh_object.rotate(rr, center=(0, 0, 0))

    o3d.io.write_triangle_mesh(os.path.join(rotated_object_ply,list_of_ply[i]), mesh_object)
    print("___ WRITING object: {}".format(os.path.join(rotated_object_ply,list_of_ply[i])))

