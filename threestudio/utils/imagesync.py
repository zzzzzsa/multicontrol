import torch
import torchvision.transforms as transforms

import numpy as np
import trimesh
import pyrender
from PIL import Image
import platform
import os
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import math

from threestudio.utils.typing import *

def get_camera_pose(position, azimuth, elevation):
    # 确保输入是Tensor并且在GPU上
    device = azimuth.device
    azimuth_rad = azimuth * torch.pi / 180.0
    elevation_rad = elevation * torch.pi / 180.0

    # 计算旋转矩阵
    R_azimuth = torch.tensor([
        [torch.cos(azimuth_rad), -torch.sin(azimuth_rad), 0],
        [torch.sin(azimuth_rad),  torch.cos(azimuth_rad), 0],
        [0,                      0,                       1]
    ], device=device)

    R_elevation = torch.tensor([
        [1, 0,                     0                     ],
        [0, torch.cos(elevation_rad), -torch.sin(elevation_rad)],
        [0, torch.sin(elevation_rad),  torch.cos(elevation_rad)]
    ], device=device)

    R = torch.mm(R_elevation, R_azimuth)

    # 创建4x4变换矩阵
    transform_matrix = torch.eye(4, device=device)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = position

    return transform_matrix

def rotation_matrix_y(angle):
    device = angle.device
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.tensor([
        [c,  torch.tensor(0.0, device=device), s],
        [torch.tensor(0.0, device=device), torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)],
        [-s, torch.tensor(0.0, device=device), c]
    ], device=device)

def rotation_matrix_x(angle):
    device = angle.device
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.tensor([
        [torch.tensor(1.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)],
        [torch.tensor(0.0, device=device), c, -s],
        [torch.tensor(0.0, device=device), s, c]
    ], device=device)

def get_camera_pose_cpu(position, azimuth, elevation):
    # 将角度从度转换为弧度
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)

    # 计算旋转矩阵
    R_azimuth = np.array([
        [np.cos(azimuth_rad), -np.sin(azimuth_rad), 0],
        [np.sin(azimuth_rad),  np.cos(azimuth_rad), 0],
        [0,                    0,                   1]
    ])

    R_elevation = np.array([
        [1, 0,                    0                   ],
        [0, np.cos(elevation_rad), -np.sin(elevation_rad)],
        [0, np.sin(elevation_rad),  np.cos(elevation_rad)]
    ])

    R = R_elevation @ R_azimuth

    # 创建4x4变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = position

    return transform_matrix

def rotation_matrix_y_cpu(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ])

def rotation_matrix_x_cpu(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ])

def sample_image(
    mesh_path, 
    height, 
    width, 
    camera_positions, 
    elevation: Float[Tensor, "B"],
    azimuth: Float[Tensor, "B"],
    light_positions, 
    fovy,
    **kwargs,
):
    device = camera_positions.device
    rot_matrix_azimuth = rotation_matrix_y(azimuth)
    rot_matrix_elevation = rotation_matrix_x(elevation)
    combined_rot_matrix = rot_matrix_elevation @ rot_matrix_azimuth

    #transform_matrix = np.eye(4)
    transform_matrix = torch.eye(4, device=device)
    transform_matrix[:3, :3] = combined_rot_matrix
    transform_matrix[:3, 3] = camera_positions

    light_matrix = torch.eye(4, device=device)
    light_matrix[:3, 3] = light_positions

    mesh = trimesh.load(mesh_path)
    scene = pyrender.Scene.from_trimesh_scene(mesh)
    camera = pyrender.PerspectiveCamera(yfov=fovy * 180 / torch.pi)
    directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    #scene.add(directional_light, pose=light_positions)

    transform_matrix_cpu = transform_matrix.to('cpu')
    transform_matrix_cpu = transform_matrix_cpu.clone().numpy()
    light_matrix_cpu = light_matrix.to('cpu')
    light_matrix_cpu = light_matrix_cpu.clone().numpy()

    camera_node = pyrender.Node(camera=camera, matrix=transform_matrix_cpu)
    light_node = pyrender.Node(light=directional_light, matrix=light_matrix_cpu)
    scene.add_node(camera_node)
    scene.add_node(light_node)

    r = pyrender.OffscreenRenderer(width, height)

    color, depth = r.render(scene)
    img = Image.fromarray(color)
    img.save('/nvme/yyh/threestudio/shape_sampled_img.png')
    transform = transforms.ToTensor()

    shape_img = transform(img)

    shape_img = shape_img.to(device)
    r.delete() 

    return shape_img

def sample_image_4x4(    
    mesh_path, 
    height, 
    width, 
    camera_positions, 
    elevation,
    azimuth,
    light_positions, 
    fovy,
    **kwargs,
):
    camera_pose = get_camera_pose(camera_positions,azimuth,elevation)
    device = camera_positions.device
    #print(camera_pose)
    #light_matrix = np.eye(4)
    light_matrix = torch.eye(4, device=device)
    light_matrix[:3, 3] = light_positions

    mesh = trimesh.load(mesh_path)
    centroid = mesh.bounding_box.centroid
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = -centroid
    #scene = pyrender.Scene.from_trimesh_scene(mesh)
    scene = pyrender.Scene()

    for m in mesh.geometry.values():
        r_mesh = pyrender.Mesh.from_trimesh(m)
        scene.add(r_mesh, pose=transformation_matrix)
    camera = pyrender.PerspectiveCamera(yfov=fovy, aspectRatio = 1.0)
    directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    #scene.add(directional_light, pose=light_positions)

    camera_pose_cpu = camera_pose.to('cpu').clone().numpy()
    light_matrix_cpu = light_matrix.to('cpu')
    light_matrix_cpu = light_matrix_cpu.clone().numpy()

    camera_node = scene.add(camera, pose=camera_pose_cpu)
    light_node = pyrender.Node(light=directional_light, matrix=light_matrix_cpu)
    #scene.add_node(camera_node)
    scene.add_node(light_node)
    projection_matrix = camera.get_projection_matrix()

    r = pyrender.OffscreenRenderer(width, height)

    color, depth = r.render(scene)
    img = Image.fromarray(color)
    img.save('/nvme/yyh/threestudio/shape_sampled_4x4matrix_img.png')
    transform = transforms.ToTensor()

    shape_img = transform(img)

    shape_img = shape_img.to(device)
    r.delete() 

    return shape_img, projection_matrix
