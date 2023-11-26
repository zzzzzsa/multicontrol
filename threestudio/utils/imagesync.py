import torch
import numpy as np
import trimesh
import pyrender
from PIL import Image
import platform
import os
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
from threestudio.utils.typing import *

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
    camera = pyrender.PerspectiveCamera(yfov=fovy)
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

    r.delete() 

    return img


