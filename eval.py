"""
Evaluate multi-view consistency of a set of edited images.

Loads input_*.png (originals) and output_*.png (edits) from a single folder.
DINOv3 correspondences are always extracted from the input images and used to
score both the inputs (baseline) and the outputs.

Usage:
    python eval.py --dir results/my_run/

    # Also compute T2I score (auto-detected from captions.json in --dir)
    python eval.py --dir results/my_run/ --captions_path results/my_run/captions.json

    # Custom output location for visualizations and metrics
    python eval.py --dir results/my_run/ --save_viz results/my_run/eval/
"""
import argparse
import json
import os
import re
from glob import glob
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.multimodal.clip_score import CLIPScore

from dino_matchsim import DinoMatchSimCfg, BgCfg, dino_matchsim_score


def _numeric_key(path: str) -> int:
    digits = re.findall(r"\d+", os.path.basename(path))
    return int(digits[-1]) if digits else 0


def load_image_set(folder: str, pattern: str) -> tuple[list, list[str]]:
    paths = sorted(glob(os.path.join(folder, pattern)), key=_numeric_key)
    return [Image.open(p).convert("RGB") for p in paths], paths


def load_t2i_prompts(image_paths: list[str], captions_path: str) -> list[str]:
    """
    Extract per-image edit prompts from a captions JSON (same format as main.py).
    Falls back to positional order when stems don't match (e.g. output_0 vs original stems).
    """
    with open(captions_path) as f:
        caption = json.load(f)
    edit = caption["edit"]
    per_image = caption.get("per_image_non_shared_edit", {})
    if not per_image:
        return [edit] * len(image_paths)
    per_image_values = list(per_image.values())
    prompts = []
    for i, p in enumerate(image_paths):
        stem = Path(p).stem
        if stem in per_image:
            prompts.append(f"{edit} {per_image[stem]}")
        elif i < len(per_image_values):
            prompts.append(f"{edit} {per_image_values[i]}")
        else:
            prompts.append(edit)
    return prompts


@torch.no_grad()
def compute_t2i_score(images: list, prompts: list[str], device: torch.device) -> float:
    metric = CLIPScore(model_name_or_path="zer0int/LongCLIP-L-Diffusers").to(device)
    score = 0.0
    for img, prompt in zip(images, prompts):
        t = pil_to_tensor(img).unsqueeze(0).to(device)  # uint8 [0, 255]
        score += metric(t, [prompt]).item()
    return (score / len(prompts) / 100) * 2.5  # CLIPScore [0,100] → CLIPScore paper w=2.5 coefficient


def create_argparser():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-view consistency of edited images in a results folder."
    )
    parser.add_argument(
        "--dir", type=str, required=True,
        help="Folder containing input_*.png (originals) and output_*.png (edits).",
    )
    parser.add_argument(
        "--captions_path", type=str, default=None,
        help="Path to captions JSON for T2I scoring. Defaults to <dir>/captions.json if it exists.",
    )
    parser.add_argument(
        "--save_viz", type=str, default=None, metavar="DIR",
        help="Directory for match overlay images and metrics JSON. Defaults to <dir>/metrics/.",
    )
    _cfg = DinoMatchSimCfg()
    parser.add_argument("--matchsim_model", type=str, default=_cfg.model_name)
    parser.add_argument("--matchsim_size",  type=int, default=_cfg.image_size)
    parser.add_argument("--matchsim_patch", type=int, default=_cfg.patch)
    parser.add_argument(
        "--no_remove_bg", action="store_true",
        help="Disable foreground segmentation before building correspondences.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main():
    args = create_argparser().parse_args()

    device_str = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    out_dir = args.save_viz or os.path.join(args.dir, "metrics")

    input_images, input_paths = load_image_set(args.dir, "input_*.png")
    output_images, output_paths = load_image_set(args.dir, "output_*.png")

    if len(input_images) < 2 or len(output_images) < 2:
        raise SystemExit(f"Need at least 2 input_*.png and 2 output_*.png in {args.dir!r}.")
    if len(input_images) != len(output_images):
        raise SystemExit(f"Count mismatch: {len(input_images)} inputs vs {len(output_images)} outputs.")

    # --- DINO-MatchSim ---
    results = dino_matchsim_score(
        input_images, output_images,
        cfg=DinoMatchSimCfg(
            model_name=args.matchsim_model,
            image_size=args.matchsim_size,
            patch=args.matchsim_patch,
            device=device_str,
        ),
        bg_cfg=BgCfg(remove_bg=not args.no_remove_bg),
        viz_dir=os.path.join(out_dir, "dino_matches_overlays"),
    )

    # --- T2I (LongCLIP-L) ---
    captions_path = args.captions_path
    if captions_path is None:
        auto = os.path.join(args.dir, "captions.json")
        if os.path.exists(auto):
            captions_path = auto

    if captions_path is not None:
        prompts = load_t2i_prompts(output_paths, captions_path)
        results["t2i_score"] = compute_t2i_score(output_images, prompts, device)

    # --- Print & Save ---

    print(f"\nDINO-MatchSim  output: {results['dino_matchsim_output']:.4f}"
          f"   input: {results['dino_matchsim_input']:.4f}"
          f"   ({results['pair_count']} pairs, {results['total_matches']} matches)")

    if "t2i_score" in results:
        print(f"T2I (LongCLIP-L):    {results['t2i_score']:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
