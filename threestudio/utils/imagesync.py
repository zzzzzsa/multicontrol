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

from threestudio.utils.typing import *

def convert_coordinates_threestudio_to_opengl(position):
    """Convert coordinates from ThreeStudio format to OpenGL format."""
    return torch.tensor([[position[0, 1], position[0, 2], -position[0, 0]]], device=position.device)

def calculate_rotation_matrix(azimuth, elevation, device):
    """Calculate combined rotation matrix based on azimuth and elevation."""
    azimuth_rad = azimuth * torch.pi / 180.0
    elevation_rad = elevation * torch.pi / 180.0

    R_azimuth = torch.tensor([
        [torch.cos(azimuth_rad), -torch.sin(azimuth_rad), 0],
        [torch.sin(azimuth_rad),  torch.cos(azimuth_rad), 0],
        [0, 0, 1]
    ], device=device)

    R_elevation = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(elevation_rad), -torch.sin(elevation_rad)],
        [0, torch.sin(elevation_rad),  torch.cos(elevation_rad)]
    ], device=device)

    return R_elevation @ R_azimuth

def get_camera_pose(position, azimuth, elevation, distance, camera_pose_front):
    """Generate the camera pose matrix."""
    device = azimuth.device
    rotation_matrix = calculate_rotation_matrix(azimuth, elevation, device)
    transform_matrix = torch.eye(4, device=device)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix = transform_matrix @ camera_pose_front.to(torch.float16)
    transform_matrix[:3, 3] = position
    return transform_matrix

def rotation_matrix_y(angle):
    """Create a rotation matrix for a rotation around the Y-axis."""
    device = angle.device
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.tensor([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], device=device)

def rotation_matrix_x(angle):
    """Create a rotation matrix for a rotation around the X-axis."""
    device = angle.device
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.tensor([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], device=device)

# CPU versions of the functions
# [Include get_camera_pose_cpu, rotation_matrix_y_cpu, and rotation_matrix_x_cpu functions here with numpy]

def setup_scene(mesh_path, camera_pose, light_position, width, height, fovy):
    """Set up the scene with the mesh, camera, and lighting."""
    mesh = trimesh.load(mesh_path)
    scene = pyrender.Scene.from_trimesh_scene(mesh)

    camera = pyrender.PerspectiveCamera(yfov=fovy)
    directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    light_node = pyrender.Node(light=directional_light, matrix=light_position)

    scene.add_node(camera_node)
    scene.add_node(light_node)

    return scene, camera_node, light_node

def render_scene(scene, camera_node, light_node, width, height):
    """Render the scene and return the image."""
    renderer = pyrender.OffscreenRenderer(width, height)
    color, _ = renderer.render(scene)
    renderer.delete()
    scene.remove_node(camera_node)
    scene.remove_node(light_node)
    return Image.fromarray(color)

""" 
def sample_image(mesh_path, height, width, camera_positions, elevation, azimuth, light_positions, fovy, **kwargs):
    device = camera_positions.device
    camera_pose = get_camera_pose(camera_positions, azimuth, elevation, None, np.eye(4))
    light_position = torch.eye(4, device=device)
    light_position[:3, 3] = light_positions

    scene, camera_node, light_node = setup_scene(mesh_path, camera_pose, light_position, width, height, fovy)
    img = render_scene(scene, camera_node, light_node, width, height)

    img.save('/nvme/yyh/threestudio/shape_sampled_img.png')
    transform = transforms.ToTensor()
    shape_img = transform(img)
    shape_img = shape_img.to(device)
    return shape_img
"""

def sample_image_4x4(mesh_path, height, width, camera_positions, camera_distances, elevation, azimuth, light_positions, fovy, **kwargs):
    # Functionality of sample_image_4x4 remains as per your request
    device = camera_positions.device
    print(camera_positions)
    camera_positions = convert_coordinates_threestudio_to_opengl(camera_positions)

    light_matrix = torch.eye(4, device=device)
    light_matrix[:3, 3] = light_positions

    mesh = trimesh.load(mesh_path)
    centroid = mesh.bounding_box.centroid
    centroid_matrix = np.eye(4)
    centroid_matrix[:3, 3] = -centroid

    camera_pose_front = np.array([
        [1, 0,  0,  0],
        [0, 1,  0,  0],
        [0, 0,  1,  3],
        [0, 0,  0,  1]
    ]) 
    
    camera_pose = get_camera_pose(camera_positions, azimuth, elevation, camera_distances, camera_pose_front)
    scene = pyrender.Scene()

    for m in mesh.geometry.values():
        r_mesh = pyrender.Mesh.from_trimesh(m)
        scene.add(r_mesh, pose=centroid_matrix)

    camera = pyrender.PerspectiveCamera(yfov=fovy, aspectRatio=1.0)
    directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    camera_pose_cpu = camera_pose.to('cpu').numpy()
    light_matrix_cpu = light_matrix.to('cpu').numpy()

    camera_node = scene.add(camera, pose=camera_pose_cpu)
    light_node = pyrender.Node(light=directional_light, matrix=light_matrix_cpu)
    scene.add_node(light_node)

    r = pyrender.OffscreenRenderer(width, height)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color[..., :3] 

    img = Image.fromarray(color)
    img.save('/nvme/lhr/dev/shape_sampled_4x4matrix_img.png')
    transform = transforms.ToTensor()

    shape_img = transform(img)
    shape_img = shape_img.to(device)

    scene.remove_node(camera_node)
    scene.remove_node(light_node)
    r.delete() 

    return shape_img, camera.get_projection_matrix()

