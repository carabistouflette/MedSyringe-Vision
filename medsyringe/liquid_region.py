"""Frozen DINOv2 patch features for localization of visible colored liquid."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import normalize,pil_to_tensor

from .data import digest,read_json

BACKBONE=Path("data/models/dinov2-small")
CHECKPOINT=Path("artifacts/auxiliary_crop_backbone.json")
HEAD=Path("artifacts/liquid_region_head.json")


class RegionFeatures:
    def __init__(self,device="cpu"):
        from transformers import Dinov2Model

        record=read_json(CHECKPOINT)
        if any(digest(BACKBONE/name)!=expected for name,expected in record["files"].items()):
            raise ValueError("The liquid region encoder files changed.")
        self.model=Dinov2Model.from_pretrained(BACKBONE,local_files_only=True).eval().to(device)
        self.device=device

    @torch.inference_mode()
    def extract(self,image):
        resized=image.convert("RGB").resize((224,224),Image.Resampling.BILINEAR)
        inputs=normalize(pil_to_tensor(resized).float()/255,(.485,.456,.406),(.229,.224,.225))
        grid=self.model(pixel_values=inputs.unsqueeze(0).to(self.device)).last_hidden_state[0,1:]
        return torch.nn.functional.normalize(grid,dim=-1).reshape(16,16,-1).cpu().numpy()


class LiquidRegion:
    def __init__(self,device):
        self.head=read_json(HEAD)
        self.encoder=RegionFeatures(device)
        self.weights=np.asarray(self.head["weights"])

    def mask(self,image):
        logits=self.encoder.extract(image)@self.weights+self.head["bias"]
        logits=cv2.resize(logits,image.size,interpolation=cv2.INTER_LINEAR)
        return logits>=self.head["cutoff"]
