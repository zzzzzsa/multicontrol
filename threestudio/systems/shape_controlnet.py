from dataclasses import dataclass, field

import torch
from torchvision.transforms import ToTensor
import torch.nn.functional as F

import cv2
from PIL import Image
import numpy as np

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.imagesync import *
from threestudio.models.geometry.base import BaseExplicitGeometry
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
import trimesh
import os

    
@threestudio.register("shape-controlnet-system")
class DreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        mesh_path: str = ""
        mesh_fitting_geometry_type: str = ""
        mesh_fitting_geometry: dict = field(default_factory=dict)
        mesh_fitting_renderer_type: str = ""
        mesh_fitting_renderer: dict = field(default_factory=dict)
        mesh_fitting_material_type: str = ""
        mesh_fitting_material: dict = field(default_factory=dict)
        mesh_fitting_background_type: str = ""
        mesh_fitting_background: dict = field(default_factory=dict)

        nvdiff_renderer_type: str = ""
        nvdiff_renderer: dict = field(default_factory=dict)
        mesh_geometry_type: str = ""
        mesh_geometry: dict = field(default_factory=dict)


    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        geometry = threestudio.find(self.cfg.mesh_fitting_geometry_type)(self.cfg.mesh_fitting_geometry)
        material = threestudio.find(self.cfg.mesh_fitting_material_type)(self.cfg.mesh_fitting_material)
        background = threestudio.find(self.cfg.mesh_fitting_background_type)(self.cfg.mesh_fitting_background)
        self.mesh_renderer = threestudio.find(self.cfg.mesh_fitting_renderer_type)(
            self.cfg.mesh_fitting_renderer,
            geometry=geometry,
            material=material,
            background=background,
        )
        mesh_geometry = threestudio.find(self.cfg.mesh_geometry_type)(self.cfg.mesh_geometry)
        self.nvdiff_renderer = threestudio.find(self.cfg.nvdiff_renderer_type)(
            self.cfg.nvdiff_renderer,
            geometry=mesh_geometry,
            material=material,
            background=background,
        )



    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):
        # if self.true_global_step < 1000:
        #     if self.true_global_step == 1:
        #         threestudio.info("Using L1 loss simulating implicit representatoin of given mesh...")
        #     guide_rgb = self.mesh_renderer(**batch)
        #     guide_rgb_save = guide_rgb["comp_rgb"] * 255
            
        #     guide_rgb_save = guide_rgb_save.byte()
        #     #print(comp_rgb_img)
        #     guide_rgb_save = Image.fromarray(guide_rgb_save.cpu().squeeze(0).numpy())
        #     guide_rgb_save.save('guide_rgb_save.png')
            

        #     rgb = self.nvdiff_renderer(**batch)
        #     rgb_save = rgb["comp_rgb"] * 255
            
        #     rgb_save = rgb_save.byte()
        #     #print(comp_rgb_img)
        #     rgb_save = Image.fromarray(rgb_save.cpu().squeeze(0).numpy())
        #     rgb_save.save('rgb_save.png')

        #     guidance_out = {"loss_l1": F.l1_loss(rgb["comp_rgb"], guide_rgb["comp_rgb"])}
        #     loss = 0.0
        #     for name, value in guidance_out.items():
        #         self.log(f"train/{name}", value)
        #         if name.startswith("loss_"):
        #             loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        #     for name, value in self.cfg.loss.items():
        #         self.log(f"train_params/{name}", self.C(value))

        #     return {"loss": loss}
        
        # else:
        out = self(batch)
        prompt_utils = self.prompt_processor()
        #print(self.cfg.cond_img_path)
        # rgb_image = cv2.imread(self.cfg.cond_img_path)[:, :, ::-1].copy() / 255
        # rgb_image = torch.FloatTensor(rgb_image).unsqueeze(0)
        #scaled_image = 2 * rgb_image - 1
        
        # print(out["comp_rgb"])
        # print(scaled_image[0])
        comp_rgb_img = out["comp_rgb"] * 255
        comp_rgb_img = comp_rgb_img.byte()
        #print(comp_rgb_img)
        comp_rgb_img = Image.fromarray(comp_rgb_img.cpu().squeeze(0).numpy())
        comp_rgb_img.save('comp_rgb_img.png')

        # scaled_image = (scaled_image[0] + 1) * 0.5 * 255
        # scaled_image = scaled_image.byte()
        # print(scaled_image)
        # scaled_image = Image.fromarray(scaled_image.cpu().squeeze(0).numpy())
        # scaled_image.save('scaled_image.png')
        cond_inp = out["comp_rgb"]
        cond_inp = cond_inp.permute(0, 3, 1, 2)
        cond_inp_resized = F.interpolate(cond_inp, (512, 512), mode="bilinear", align_corners=False)
        cond_inp_resized = cond_inp_resized.permute(0, 2, 3, 1)

        render_batch = batch.copy()
        render_batch['height'] = 512
        render_batch['width'] = 512
        guide_rgb = self.mesh_renderer(**render_batch)
        guide_rgb_save = guide_rgb["comp_rgb"] * 255
        
        guide_rgb_save = guide_rgb_save.byte()
        #print(comp_rgb_img)
        guide_rgb_save = Image.fromarray(guide_rgb_save.cpu().squeeze(0).numpy())
        guide_rgb_save.save('guide_rgb_save.png')

        # control_inp = sample_image(self.cfg.mesh_path,**batch)
        # control_inp = torch.from_numpy(control_inp)
        # control_inp = control_inp.unsqueeze(0)

        # print(type(prompt_utils))
        # print(list(batch.keys()))
        # print(list(out.keys()))
        



        guidance_out = self.guidance(
            cond_inp_resized, guide_rgb["comp_rgb"], prompt_utils, rgb_as_latents=False, **batch
        )

        loss = 0.0

        # guidance_eval = guidance_out.get("eval")
        # imgs_final = guidance_eval.get("imgs_final")
        # imgs_1step = guidance_eval.get("imgs_1step")
        # imgs_1orig = guidance_eval.get("imgs_1orig")
        # imgs_noisy = guidance_eval.get("imgs_noisy")
        #print(imgs_final.shape)

        # imgs_final = (imgs_final + 1) * 0.5 * 255
        # imgs_final = imgs_final.byte()
        # imgs_final = Image.fromarray(imgs_final.cpu().squeeze(0).numpy())
        # imgs_final.save('imgs_final.png')

        # imgs_1step = (imgs_1step + 1) * 0.5 * 255
        # imgs_1step = imgs_1step.byte()
        # imgs_1step = Image.fromarray(imgs_1step.cpu().squeeze(0).numpy())
        # imgs_1step.save('imgs_1step.png')

        # imgs_1orig = (imgs_1orig + 1) * 0.5 * 255
        # imgs_1orig = imgs_1orig.byte()
        # imgs_1orig = Image.fromarray(imgs_1orig.cpu().squeeze(0).numpy())
        # imgs_1orig.save('imgs_1orig.png')

        # imgs_noisy = (imgs_noisy + 1) * 0.5 * 255
        # imgs_noisy = imgs_noisy.byte()
        # imgs_noisy = Image.fromarray(imgs_noisy.cpu().squeeze(0).numpy())
        # imgs_noisy.save('imgs_noisy.png')



        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
