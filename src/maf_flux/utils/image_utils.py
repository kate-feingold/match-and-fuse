import shutil
from collections import Counter
from pathlib import Path
from PIL import Image


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.JPG', '.JPEG', '.PNG'}


def compute_target_resolution(orig_w: int, orig_h: int) -> tuple[int, int]:
    """Compute (width, height) such that the pairwise grid's longest side is 1024,
    both dimensions are divisible by 64, and the aspect ratio is preserved.

    Grid orientation:
        landscape / square (W >= H) → vertical stack:   W × 2H
        portrait           (H >  W) → horizontal stack: 2W × H
    """
    def round64(x: float) -> int:
        return max(64, round(x / 64) * 64)

    if orig_w >= orig_h:          # vertical stack: max(W, 2H) = 1024
        r = orig_w / orig_h
        if r < 2:
            h, w = 512, round64(512 * r)
        else:
            w, h = 1024, round64(1024 / r)
    else:                          # horizontal stack: max(2W, H) = 1024
        s = orig_h / orig_w
        if s < 2:
            w, h = 512, round64(512 * s)
        else:
            h, w = 1024, round64(1024 / s)

    return w, h


def compute_scene_resolution(image_paths: list) -> tuple[int, int]:
    """Return the most common target resolution across all images in the scene.

    Applies compute_target_resolution to every image and returns the mode,
    so one outlier image with an unusual crop doesn't dictate the resolution.
    """
    counts = Counter(
        compute_target_resolution(*Image.open(p).size)
        for p in image_paths
    )
    return counts.most_common(1)[0][0]


def center_crop_and_resize(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Center-crop img to the target aspect ratio, then resize to (target_w, target_h)."""
    orig_w, orig_h = img.size
    target_ratio = target_w / target_h
    orig_ratio = orig_w / orig_h

    if orig_ratio > target_ratio + 1e-6:        # original is wider → crop width
        new_w = round(orig_h * target_ratio)
        left = (orig_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, orig_h))
    elif orig_ratio < target_ratio - 1e-6:      # original is taller → crop height
        new_h = round(orig_w / target_ratio)
        top = (orig_h - new_h) // 2
        img = img.crop((0, top, orig_w, top + new_h))

    return img.resize((target_w, target_h))


def load_images_from_folder(folder) -> list[str]:
    """Return sorted list of image paths in folder."""
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {e.lower() for e in IMG_EXTS})
    if not paths:
        raise ValueError(f"No images found in {folder}")
    return [str(p) for p in paths]


def prepare_scene_images(src_dir, dst_dir) -> list[str]:
    """Center-crop and resize all images in src_dir; write results to dst_dir.

    Resolution is the mode of compute_target_resolution across all images.
    If dst_dir already has the correct count at the correct resolution, returns []
    (nothing written). Clears and reprocesses if the resolution has changed.
    Returns list of newly written file paths.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    src_paths = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in {e.lower() for e in IMG_EXTS})
    if not src_paths:
        return []

    target_w, target_h = compute_scene_resolution(src_paths)
    dst_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(p for p in dst_dir.iterdir() if p.suffix.lower() in {e.lower() for e in IMG_EXTS})
    if len(existing) >= len(src_paths):
        w, h = Image.open(existing[0]).size
        if (w, h) == (target_w, target_h):
            return []  # already up-to-date
        shutil.rmtree(dst_dir)  # resolution changed: clear and reprocess
        dst_dir.mkdir()

    out_paths = []
    for src_path in src_paths:
        dst_path = dst_dir / (src_path.stem + ".jpg")
        img = Image.open(src_path).convert("RGB")
        img = center_crop_and_resize(img, target_w, target_h)
        img.save(dst_path, quality=95)
        out_paths.append(str(dst_path))

    return out_paths
