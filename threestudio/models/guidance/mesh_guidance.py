from dataclasses import dataclass, field

import threestudio
import torch.nn.functional as F
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *
from PIL import Image


@threestudio.register("mesh-fitting-guidance")
class MeshGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        geometry_type: str = ""
        geometry: dict = field(default_factory=dict)
        renderer_type: str = ""
        renderer: dict = field(default_factory=dict)
        mesh_renderer_type: str = ""
        mesh_renderer: dict = field(default_factory=dict)
        material_type: str = ""
        material: dict = field(default_factory=dict)
        background_type: str = ""
        background: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading obj")
        geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)
        material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        background = threestudio.find(self.cfg.background_type)(self.cfg.background)
        # self.renderer = threestudio.find(self.cfg.renderer_type)(
        #     self.cfg.renderer,
        #     geometry=geometry,
        #     material=material,
        #     background=background,
        # )
        self.mesh_renderer = threestudio.find(self.cfg.mesh_renderer_type)(
            self.cfg.mesh_renderer,
            geometry=geometry,
            material=material,
            background=background,
        )
        threestudio.info(f"Loaded mesh!")

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        guide_rgb = self.mesh_renderer(**kwargs)

        comp_rgb_img = guide_rgb["comp_rgb"] * 255
        comp_rgb_img = comp_rgb_img.byte()
        #print(comp_rgb_img)
        comp_rgb_img = Image.fromarray(comp_rgb_img.cpu().squeeze(0).numpy())
        comp_rgb_img.save('meshrenderer_meshfitting_img.png')

        guidance_out = {"loss_l1": F.l1_loss(rgb, guide_rgb["comp_rgb"])}
        return guidance_out
