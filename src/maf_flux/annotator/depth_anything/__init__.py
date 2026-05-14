import numpy as np
import torch

from pathlib import Path
from .metric_depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2Model

_CKPT_DIR = Path(__file__).parents[4] / "checkpoints"


class DepthAnythingV2:
    def __init__(self, joint_min_max: bool = False):
        encoder = "vitl"
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        dataset, max_depth = "hypersim", 20
        depth_anything = DepthAnythingV2Model(**{**model_configs[encoder], 'max_depth': max_depth})
        depth_anything.load_state_dict(
            torch.load(_CKPT_DIR / f"depth_anything_v2_metric_{dataset}_{encoder}.pth", map_location='cpu'))
        self.model = depth_anything.to("cuda").eval()
        self.input_size = 518
        self.joint_min_max = joint_min_max

    def __call__(self, input_images: list[np.ndarray]) -> np.ndarray:
        raw_depths = [self.model.infer_image(img, self.input_size) for img in input_images]
        raw_depths = np.stack(raw_depths)

        vmin = np.percentile(raw_depths, 2, axis=(1, 2), keepdims=True)
        vmax = np.percentile(raw_depths, 95, axis=(1, 2), keepdims=True)
        if self.joint_min_max:
            vmin, vmax = vmin.min(), vmax.max()

        depths = 1 - (raw_depths - vmin) / (vmax - vmin)
        depths = (depths.clip(0, 1) * 255).astype(np.uint8)
        return depths  # shape: [N, H, W]

