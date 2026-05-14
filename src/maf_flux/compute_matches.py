"""
Pre-compute RoMa dense matches between all pairs of input images.
Matches are saved as .pth files loadable by main.py via --matches.

Usage (pre-compute):
    python compute_matches.py \
        --images img0.jpg img1.jpg img2.jpg \
        --save_path matches/ \
        --width 512 --height 512

Requires: pip install romatch
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from romatch import roma_indoor

# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _occurrence_matching(h, w, matches_im0, matches_im1):
    matches_grid = -1 * np.ones((2, h, w, 2))
    for idx, matches0, matches1 in [(0, matches_im0, matches_im1), (1, matches_im1, matches_im0)]:
        unique_im0, inverse_im0 = np.unique(matches0, axis=0, return_inverse=True)
        grouped_im1 = [matches1[inverse_im0 == i] for i in range(len(unique_im0))]
        most_frequent_im1 = np.array([
            Counter(map(tuple, group)).most_common(1)[0][0] if len(group) > 0 else (-1, -1)
            for group in grouped_im1
        ])
        matches_grid[idx, unique_im0[:, 1], unique_im0[:, 0]] = most_frequent_im1
    return matches_grid


def _create_grid(kpts0, kpts1, width, height, patch_size, match_mode='occurrence'):
    """Convert pixel keypoints to a [2, H/patch, W/patch, 2] match grid.

    match_mode:
        'occurrence' — most-frequent target patch wins (default)
        'random'     — last-write-wins (mirrors original patch_size==1 behaviour)
    """
    valid = (
        (kpts0[:, 0] >= 0) & (kpts0[:, 0] < width) &
        (kpts0[:, 1] >= 0) & (kpts0[:, 1] < height) &
        (kpts1[:, 0] >= 0) & (kpts1[:, 0] < width) &
        (kpts1[:, 1] >= 0) & (kpts1[:, 1] < height)
    )
    kpts0, kpts1 = kpts0[valid], kpts1[valid]

    q_w, q_h = width // patch_size, height // patch_size
    scale_w, scale_h = width / q_w, height / q_h

    def to_patch_coords(kpts):
        attn = (kpts.astype(float) / np.array([[scale_w, scale_h]])).astype(int)
        attn[:, 0] = attn[:, 0].clip(0, q_w - 1)
        attn[:, 1] = attn[:, 1].clip(0, q_h - 1)
        return attn

    p0, p1 = to_patch_coords(kpts0), to_patch_coords(kpts1)
    if match_mode == 'random':
        matches_grid = -1 * np.ones((2, q_h, q_w, 2))
        matches_grid[0, p0[:, 1], p0[:, 0]] = p1
        matches_grid[1, p1[:, 1], p1[:, 0]] = p0
    else:
        matches_grid = _occurrence_matching(q_h, q_w, p0, p1)
    matches_grid[:, 0, 0] = -1  # fix border artifact
    return torch.tensor(matches_grid).to(torch.long)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _create_color_wheel():
    color_wheel = plt.get_cmap('hsv')(np.linspace(0, 1, 256))[:, :3]
    return (color_wheel * 255).astype(np.uint8)


def _assign_colors(h, w, color_wheel):
    y, x = np.indices((h, w))
    cx, cy = w // 2, h // 2
    angle = np.arctan2(y - cy, x - cx)
    normalized_angle = (angle + np.pi) / (2 * np.pi)
    indices = (normalized_angle * (len(color_wheel) - 1)).astype(int)
    return color_wheel[indices]


def _hstack(img0, img1):
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.ones((max(h0, h1), w0 + w1, 3), dtype=np.uint8) * 255
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0 + w1] = img1
    return canvas


def _visualize_matches(matches_flat, self_img, other_img, patch_size, img_alpha=0.3):
    self_img = np.array(self_img)
    other_img = np.array(other_img)
    self_h, self_w = self_img.shape[0] // patch_size, self_img.shape[1] // patch_size
    other_h, other_w = other_img.shape[0] // patch_size, other_img.shape[1] // patch_size

    color_wheel = _create_color_wheel()
    other_colored = _assign_colors(other_h, other_w, color_wheel)

    matches = matches_flat.cpu().numpy()
    no_match_mask = matches < 0
    matches[no_match_mask] = 0
    unassigned_mask = np.ones(len(matches), dtype=bool)
    unassigned_mask[np.unique(matches[matches >= 0])] = False

    self_colored = other_colored.reshape(-1, 3)[matches].reshape(self_h, self_w, 3)
    no_match_mask = no_match_mask.reshape(self_h, self_w).astype(np.uint8) * 255
    unassigned_mask = unassigned_mask.reshape(other_h, other_w).astype(np.uint8) * 255

    H0, W0 = self_img.shape[:2]
    H1, W1 = other_img.shape[:2]
    r_self = cv2.resize(self_colored, (W0, H0), interpolation=cv2.INTER_NEAREST)
    r_no_match = cv2.resize(no_match_mask, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
    r_other = cv2.resize(other_colored, (W1, H1), interpolation=cv2.INTER_NEAREST)
    r_unassigned = cv2.resize(unassigned_mask, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)

    self_plot = cv2.addWeighted(self_img, img_alpha, r_self, 1 - img_alpha, 0)
    self_plot[r_no_match] = self_img[r_no_match]
    other_plot = cv2.addWeighted(other_img, img_alpha, r_other, 1 - img_alpha, 0)
    other_plot[r_unassigned] = other_img[r_unassigned]

    return Image.fromarray(_hstack(self_plot, other_plot))


def _plot_warp(im0_path, im1_path, grid, patch_size, width, height, save_path, stem0, stem1):
    def open_resize(p):
        img = Image.open(p).convert("RGB")
        w = patch_size * (width // patch_size)
        h = patch_size * (height // patch_size)
        return img.resize((w, h))

    img0, img1 = open_resize(im0_path), open_resize(im1_path)
    q_h, q_w = grid.shape[1], grid.shape[2]
    matches_flat = grid[..., 1] * q_w + grid[..., 0]  # y * width + x
    matches_flat = matches_flat.flatten(1)  # [2, seq_len]

    _visualize_matches(matches_flat[0], img0, img1, patch_size).save(
        save_path / f"{stem0}_{stem1}_matches_warp.png"
    )
    _visualize_matches(matches_flat[1], img1, img0, patch_size).save(
        save_path / f"{stem1}_{stem0}_matches_warp.png"
    )


# ---------------------------------------------------------------------------
# Main matching function
# ---------------------------------------------------------------------------

def compute_matches(image_paths, save_path, width, height, patch_size=16,
                    conf_thresh=0.05, device='cuda', match_mode='occurrence'):
    """
    Run RoMa indoor matching for all pairs in image_paths and save .pth grids.

    Args:
        image_paths: list of image file paths
        save_path:   directory to write {stem_a}_{stem_b}_matches.pth files
        width, height: image resolution used during matching
        patch_size:  attention patch size (default 16; must match FLUX resolution)
        conf_thresh: RoMa confidence threshold for sampling matches
        device:      'cuda' or 'cpu'

    Returns:
        Path to save_path
    """

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    roma_model = roma_indoor(device=device)

    num_images = len(image_paths)
    edges = [(i, j) for i in range(num_images) for j in range(i + 1, num_images)]

    for e0, e1 in edges:
        im0_path, im1_path = image_paths[e0], image_paths[e1]
        stem0, stem1 = Path(im0_path).stem, Path(im1_path).stem
        out_path = save_path / f'{stem0}_{stem1}_matches.pth'

        if out_path.exists():
            print(f"  Skipping {stem0}_{stem1} (exists)")
            continue

        print(f"  Matching {stem0} <-> {stem1}")
        warp, certainty = roma_model.match(im0_path, im1_path, device=device)
        matches, _ = roma_model.sample_by_conf(warp, certainty, conf_thresh=conf_thresh)
        kpts0, kpts1 = roma_model.to_pixel_coordinates(matches, height, width, height, width)
        kpts0, kpts1 = kpts0.cpu().numpy(), kpts1.cpu().numpy()

        grid = _create_grid(kpts0, kpts1, width, height, patch_size, match_mode)
        torch.save(grid, out_path)
        _plot_warp(im0_path, im1_path, grid, patch_size, width, height, save_path, stem0, stem1)
        print(f"    -> {out_path}")

    return save_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', nargs='+', required=True, help='Input image paths')
    parser.add_argument('--save_path', required=True, help='Output directory for .pth match files')
    parser.add_argument('--width', type=int, default=None,
                        help='Image width for matching. If omitted, inferred to best fit the images.')
    parser.add_argument('--height', type=int, default=None,
                        help='Image height for matching. If omitted, inferred to best fit the images.')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Attention patch size (must match the resolution used in main.py)')
    parser.add_argument('--conf_thresh', type=float, default=0.05,
                        help='RoMa confidence threshold')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--match_mode', default='occurrence', choices=['occurrence', 'random'],
                        help='occurrence: most-frequent target patch wins; random: last-write-wins')
    args = parser.parse_args()

    if args.width is None or args.height is None:
        from PIL import Image as _PIL
        from src.maf_flux.utils.image_utils import compute_target_resolution
        _w, _h = _PIL.open(args.images[0]).size
        args.width, args.height = compute_target_resolution(_w, _h)
        print(f"Auto-detected resolution: {args.width}x{args.height}")

    print(f"Computing matches for {len(args.images)} images -> {args.save_path}")
    compute_matches(
        args.images, args.save_path,
        args.width, args.height,
        args.patch_size, args.conf_thresh, args.device, args.match_mode,
    )
    print("Done.")
