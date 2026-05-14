"""
Center-crop and resize all scenes under src_root into dst_root.

Resolution is the most common compute_target_resolution across all scene images.

Usage:
    python scripts/prepare_images.py <src_root> <dst_root>
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.maf_flux.utils.image_utils import prepare_scene_images, IMG_EXTS
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Center-crop and resize all scenes under src_root into dst_root.")
    parser.add_argument("src_root", type=Path, help="Root directory containing scene subdirectories.")
    parser.add_argument("dst_root", type=Path, help="Root directory to write processed images into.")
    args = parser.parse_args()

    src_root, dst_root = args.src_root, args.dst_root
    dst_root.mkdir(parents=True, exist_ok=True)

    for scene_dir in sorted(d for d in src_root.iterdir() if d.is_dir()):
        name = scene_dir.name
        imgs = [p for p in scene_dir.iterdir() if p.suffix.lower() in {e.lower() for e in IMG_EXTS}]
        if len(imgs) < 2:
            continue
        dst = dst_root / name
        written = prepare_scene_images(scene_dir, dst)
        if written:
            w, h = Image.open(written[0]).size
            print(f"  {name}: {len(written)} images -> {w}x{h}")
        else:
            print(f"  skip {name} (already prepared)")


if __name__ == "__main__":
    main()
