import matplotlib.pyplot as plt
from skimage.io import imread
import torch
import numpy as np
# Util function for loading meshes
from pytorch3d.io import load_obj
from PIL import Image
import math

import threestudio
from threestudio.utils.typing import *
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    PointLights, 
    DirectionalLights,
    FoVPerspectiveCameras,
    camera_position_from_spherical_angles, 
    FoVOrthographicCameras, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    HardFlatShader,
    TexturesVertex,
    TexturesAtlas,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    AmbientLights,
    BlendParams,
    rotate_on_spot,
)
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.utils import ico_sphere
from pytorch3d.vis.texture_vis import texturesuv_image_PIL

def _render(
    mesh: Meshes,
    name: str,
    dist: float = 3.0,
    elev: float = 10.0,
    azim: float = 0,
    image_size: int = 256,
    fov: float = 70.0,
    pan=None,
    RT=None,
    use_ambient=False,
):
    device = mesh.device
    if RT is not None:
        R, T = RT
    else:
        R, T = look_at_view_transform(dist, elev, azim)
        if pan is not None:
            R, T = rotate_on_spot(R, T, pan)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)
    blur_radius = 0.0
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=blur_radius, faces_per_pixel=1
    )

    # Init shader settings
    if use_ambient:
        lights = AmbientLights(device=device)
    else:
        lights = PointLights(device=device)
        lights.location = torch.tensor([0.0, 0.0, 2.0], device=device)[None]

    blend_params = BlendParams(
        sigma=1e-1,
        gamma=1e-4,
        background_color=torch.tensor([1.0, 1.0, 1.0], device=device),
    )
    # Init renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(
            device=device, lights=lights, cameras=cameras, blend_params=blend_params
        ),
    )

    output = renderer(mesh)

    image = (output[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

    Image.fromarray(image).save(f"glb_{name}_.png")

    return output

def sample_image(
    mesh_path, 
    height, 
    width, 
    camera_positions, 
    camera_distances,
    elevation: Float[Tensor, "B"],
    azimuth: Float[Tensor, "B"],
    light_positions, 
    fovy,
    **kwargs,
):
    device = azimuth.device
    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    mesh = io.load_mesh(mesh_path, device=device)
    mesh.textures = TexturesVertex(0.5 * torch.ones_like(mesh.verts_padded()))
    #threestudio.info('We have {0} vertices and {1} faces.'.format(mesh._verts_list[0].shape[0], mesh._faces_list[0].shape[0]))
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
    # R, T = look_at_view_transform(2.7, 0, 180) 
    distance = camera_distances * 2
    elevation_deg = elevation.float()
    azimuth_deg = azimuth.float()
    fov = fovy.float()
    R, T = look_at_view_transform(dist=distance, elev=elevation_deg, azim=azimuth_deg)
    # print(R)
    # print(T)
    #image = _render(mesh=mesh, name="sample_image_grey", dist=distance, elev=elevation_deg, azim=azimuth_deg, image_size=height * width, fov=fov * 180 / torch.pi)

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov * 180 / torch.pi)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        perspective_correct=False,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
    # shader=HardFlatShader(
    #     device=device, 
    #     cameras=cameras,
    #     lights=lights
    # )
    
)
    image = renderer(mesh)
    image = (image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

    Image.fromarray(image).save(f"glb_grey_.png")

    return image


