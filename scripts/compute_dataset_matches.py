"""
Compute RoMa pairwise matches for all scenes under img_root; save to match_root.

img_root can be either:
  - a root containing dataset dirs (data/images/)
  - a single dataset dir          (data/images/dreambooth/)

Usage:
    python scripts/compute_dataset_matches.py --img_root <path> --match_root <path> [--device <device>]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image as _PIL
from src.maf_flux.compute_matches import compute_matches
from src.maf_flux.utils.image_utils import IMG_EXTS


def has_images(d: Path) -> bool:
    return any(p.suffix.lower() in {e.lower() for e in IMG_EXTS} for p in d.iterdir())


def process_scene(scene_dir: Path, out_dir: Path, device: str):
    images = sorted(p for p in scene_dir.iterdir() if p.suffix.lower() in {e.lower() for e in IMG_EXTS})
    if len(images) < 2:
        return
    w, h = _PIL.open(images[0]).size
    print(f"  {scene_dir.name}: {len(images)} images ({w}x{h})")
    compute_matches(images, out_dir, w, h, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", required=True, type=Path)
    parser.add_argument("--match_root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    for child in sorted(args.img_root.iterdir()):
        if not child.is_dir():
            continue
        if has_images(child):
            # img_root is a single dataset dir; child is a scene
            process_scene(child, args.match_root / child.name, args.device)
        else:
            # img_root is a datasets root; child is a dataset dir
            for scene_dir in sorted(child.iterdir()):
                if scene_dir.is_dir():
                    process_scene(scene_dir, args.match_root / child.name / scene_dir.name, args.device)


if __name__ == "__main__":
    main()
