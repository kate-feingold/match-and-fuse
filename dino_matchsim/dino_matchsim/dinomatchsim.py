# dinomatchsim.py
# DINO-MatchSim multi-view consistency metric
import os
import math
import colorsys
from dataclasses import dataclass
from typing import List, Optional
from itertools import combinations

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from transformers import AutoModel

try:
    from carvekit.ml.wrap.fba_matting import FBAMatting
    from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
    from carvekit.pipelines.postprocessing import MattingMethod
    from carvekit.pipelines.preprocessing import PreprocessingStub
    from carvekit.trimap.generator import TrimapGenerator
    from carvekit.api.interface import Interface as CK_Interface
    HAVE_CARVEKIT = True
except Exception:
    HAVE_CARVEKIT = False

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False
from scipy import ndimage as ndi

JPEG_SAVE_KW = {"format": "JPEG", "quality": 85, "optimize": True, "subsampling": 2}

# ----------------------------
# Constants
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DINO_MODEL_VITS  = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINO_MODEL_VITSP = "facebook/dinov3-vitsp16-dinov2-style"
DINO_MODEL_VITB  = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_MODEL_VITL  = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_MODEL_VITHP = "facebook/dinov3-vith14-pretrain-lvd1689m"
DINO_MODEL_VIT7B = "facebook/dinov3-vit7b14-pretrain-lvd1689m"


# ----------------------------
# Morphology helpers
# ----------------------------
def _morph_kernel(r: int):
    r = max(int(r), 0)
    if r <= 0:
        return None
    if HAVE_CV2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    return np.ones((2*r+1, 2*r+1), dtype=bool)


def _connected_components_bool(mask_bool: np.ndarray):
    """Return (n, labels, stats) with either cv2 or scipy.ndimage."""
    if HAVE_CV2:
        mask_u8 = mask_bool.astype(np.uint8) * 255
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        return n, labels, stats
    labels, n = ndi.label(mask_bool)
    stats = []
    for lab in range(n + 1):
        if lab == 0:
            stats.append([0, 0, 0, 0, 0, int((labels == 0).sum())])
            continue
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            stats.append([lab, 0, 0, 0, 0, 0])
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        stats.append([lab, x0, y0, x1-x0+1, y1-y0+1, ys.size])
    return n, labels, np.array(stats, dtype=np.int64)


def _noise_like(img_np: np.ndarray, std: float = 30.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=127.5, scale=std, size=img_np.shape).astype(np.float32)
    return np.clip(noise, 0, 255).astype(np.uint8)


# ----------------------------
# Background remover
# ----------------------------
class BackgroundRemover:
    def __init__(self,
                 device: str = "cuda",
                 bg_mode: str = "noise",
                 bg_noise_std: float = 30.0,
                 crop_bbox: bool = False,
                 seed: int = 0,
                 fg_threshold: int = 200,
                 keep_largest: bool = True,
                 min_area_ratio: float = 0.003,
                 erode_px: int = 8,
                 open_px: int = 2,
                 close_px: int = 0,
                 crop_margin: float = 0.02,
                 crop_shrink_px: int = 6):
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.bg_mode = bg_mode
        self.std = float(bg_noise_std)
        self.crop_bbox = bool(crop_bbox)
        self.seed = int(seed)
        self.fg_threshold = int(np.clip(fg_threshold, 0, 255))
        self.keep_largest = bool(keep_largest)
        self.min_area_ratio = float(min_area_ratio)
        self.erode_px = int(erode_px)
        self.open_px = int(open_px)
        self.close_px = int(close_px)
        self.crop_margin = float(crop_margin)
        self.crop_shrink_px = int(crop_shrink_px)

        if HAVE_CARVEKIT:
            seg = TracerUniversalB7(device=self.device, batch_size=1)
            fba = FBAMatting(device=self.device, input_tensor_size=2048, batch_size=1)
            trimap = TrimapGenerator()
            post = MattingMethod(matting_module=fba, trimap_generator=trimap, device=self.device)
            self.interface = CK_Interface(pre_pipe=PreprocessingStub(), post_pipe=post, seg_pipe=seg)
            self.seg = seg
        else:
            self.interface = None
            self.seg = None

    def _refine_mask(self, raw_mask_bool: np.ndarray) -> np.ndarray:
        H, W = raw_mask_bool.shape
        mask = raw_mask_bool.astype(bool) if raw_mask_bool.dtype != np.bool_ else raw_mask_bool.copy()

        if raw_mask_bool.dtype != np.bool_ and raw_mask_bool.max() > 1:
            mask = (raw_mask_bool >= self.fg_threshold)

        min_area = int(max(1, self.min_area_ratio * H * W))
        n, labels, stats = _connected_components_bool(mask)
        if n > 1:
            for lab in range(1, n):
                if int(stats[lab, -1]) < min_area:
                    mask[labels == lab] = False

        if self.keep_largest and mask.any():
            n2, labels2, stats2 = _connected_components_bool(mask)
            if n2 > 1:
                areas = [(lab, int(stats2[lab, -1])) for lab in range(1, n2)]
                if areas:
                    largest_lab = max(areas, key=lambda t: t[1])[0]
                    mask = (labels2 == largest_lab)

        if self.open_px > 0:
            k = _morph_kernel(self.open_px)
            if HAVE_CV2:
                mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, k) > 0
            else:
                mask = ndi.binary_opening(mask, structure=k)
        if self.erode_px > 0:
            k = _morph_kernel(self.erode_px)
            if HAVE_CV2:
                mask = cv2.erode(mask.astype(np.uint8) * 255, k) > 0
            else:
                mask = ndi.binary_erosion(mask, structure=k)
        if self.close_px > 0:
            k = _morph_kernel(self.close_px)
            if HAVE_CV2:
                mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k) > 0
            else:
                mask = ndi.binary_closing(mask, structure=k)

        if not mask.any():
            base = raw_mask_bool.astype(bool)
            return base if base.any() else np.ones((H, W), dtype=bool)
        return mask

    def get_mask(self, img: Image.Image) -> np.ndarray:
        if HAVE_CARVEKIT and self.seg is not None:
            m = np.array(self.seg([img]))[0]
            raw_bool = (m > 0) if m.dtype == np.uint8 else m.astype(bool)
            return self._refine_mask(raw_bool)
        return np.ones((img.height, img.width), dtype=bool)

    def apply(self, img: Image.Image):
        mask_bool = self.get_mask(img)
        img_np = np.array(img.convert("RGB"))

        if self.bg_mode == "gray":
            bg = np.full_like(img_np, 127, dtype=np.uint8)
        elif self.bg_mode == "white":
            bg = np.full_like(img_np, 255, dtype=np.uint8)
        elif self.bg_mode == "black":
            bg = np.zeros_like(img_np, dtype=np.uint8)
        else:  # noise (default)
            bg = _noise_like(img_np, self.std, seed=self.seed)

        out = img_np.copy()
        out[~mask_bool] = bg[~mask_bool]

        if self.crop_bbox:
            ys, xs = np.where(mask_bool)
            if ys.size > 0:
                x0, x1 = xs.min(), xs.max()
                y0, y1 = ys.min(), ys.max()
                bw, bh = x1 - x0 + 1, y1 - y0 + 1
                mx = int(round(self.crop_margin * bw))
                my = int(round(self.crop_margin * bh))
                x0 = max(0, x0 - mx + self.crop_shrink_px)
                y0 = max(0, y0 - my + self.crop_shrink_px)
                x1 = min(out.shape[1], x1 + 1 + mx - self.crop_shrink_px)
                y1 = min(out.shape[0], y1 + 1 + my - self.crop_shrink_px)
                out = out[y0:y1, x0:x1]
                mask_bool = mask_bool[y0:y1, x0:x1]

        return Image.fromarray(out), mask_bool


# ----------------------------
# Resize helper
# ----------------------------
def resize_transform_patches(mask_or_image: Image.Image,
                              image_size: int,
                              patch_size: int) -> torch.Tensor:
    w, h = mask_or_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    H = h_patches * patch_size
    W = w_patches * patch_size
    return TF.to_tensor(TF.resize(mask_or_image, (H, W)))


# ----------------------------
# Configs
# ----------------------------
@dataclass
class DinoMatchSimCfg:
    model_name: str = DINO_MODEL_VITB  # HuggingFace DINOv3 model ID
    image_size: int = 768              # target height in pixels; width is aspect-ratio-scaled, both rounded to multiples of patch
    patch: int = 16                    # patch size in pixels — must match the model variant (e.g. 16 for ViT-*16)
    mask_fg_threshold: float = 0.5    # avg patch coverage above which a patch is considered foreground
    device: str = "cuda"
    # --- correspondence filtering ---
    mutual: bool = True        # keep only mutual nearest-neighbours (A→B and B→A agree)
    ratio: float = 1.00        # Lowe's ratio test on sharpened similarities; disabled when ≤ 1.0
    temperature: float = 0.12  # sharpening divisor applied to similarity matrix before ratio/prob filtering
    sim_thresh: float = 0.5    # minimum raw cosine similarity to keep a correspondence
    prob_thresh: float = 0.00  # minimum row-wise softmax probability after sharpening; 0 disables
    # --- scoring ---
    # exp(-mean_distance / tau): compresses negative cosine similarities into small scores (≈0–0.2)
    # and expands positive similarities into a wider interval (≈0.2–1).
    tau: float = 0.6


@dataclass
class BgCfg:
    remove_bg: bool = True       # run CarveKit segmentation and replace background before feature extraction
    bg_mode: str = "noise"       # background fill: "noise" (random Gaussian), "gray", "white", "black"
    bg_noise_std: float = 30.0   # std of Gaussian noise when bg_mode="noise"
    crop_bbox: bool = False      # crop to foreground bounding box after background replacement
    seed: int = 0                # RNG seed for noise generation


# ----------------------------
# Matcher
# ----------------------------
class DinoMatchSim:
    """
    DINO-MatchSim multi-view consistency metric.

    Builds foreground-filtered mutual nearest-neighbour correspondences from a
    set of reference images, then measures how well those correspondences are
    preserved in a second set of images (e.g. before/after an edit).

    Typical usage::

        matcher = DinoMatchSim(DinoMatchSimCfg(), BgCfg())
        cache   = matcher.build_fg_nn_field(input_images)
        result  = matcher.score_fixed_matches_with_details(output_images, cache)
        print(result["score"])  # exp(-mean_cosine_distance / tau)

    Or use the module-level convenience wrapper::

        result = dino_matchsim_score(input_images, output_images)
    """

    def __init__(self, cfg: DinoMatchSimCfg, bg: BgCfg):
        self.cfg = cfg
        self.bg = bg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.model.eval().to(self.device)

        p = cfg.patch
        self.patch_quant_filter = torch.nn.Conv2d(1, 1, p, stride=p, bias=False)
        self.patch_quant_filter.weight.data.fill_(1.0 / (p * p))
        self.patch_quant_filter.to(self.device)

        self.bg_remover = BackgroundRemover(
            device=cfg.device, bg_mode=bg.bg_mode, bg_noise_std=bg.bg_noise_std,
            crop_bbox=bg.crop_bbox, seed=bg.seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prep_image_and_mask(self, img: Image.Image, mask: Optional[Image.Image]):
        """Apply BG removal (if enabled) and return (processed_img, mask_bool)."""
        if self.bg.remove_bg:
            return self.bg_remover.apply(img)
        if mask is not None:
            return img, (np.array(mask) > 0)
        if img.mode in ("LA", "RGBA"):
            return img, (np.array(img.split()[-1]) > 0)
        return img, np.ones((img.height, img.width), dtype=bool)

    @torch.inference_mode()
    def _extract_patch_features(self, image_pil: Image.Image):
        """Returns (feats [C,Hp,Wp], h_patches, w_patches) on CPU, L2-normalised."""
        img_t = resize_transform_patches(image_pil.convert("RGB"), self.cfg.image_size, self.cfg.patch)
        img_t = TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(self.device)

        out = self.model(pixel_values=img_t, output_hidden_states=True, return_dict=True)
        last = out.hidden_states[-1].squeeze(0)  # [1+reg+N, C]
        C = last.shape[-1]
        num_reg = getattr(self.model.config, "num_register_tokens", 0)
        patch_tokens = last[1 + num_reg:, :]

        _, _, H, W = img_t.shape
        h_patches, w_patches = H // self.cfg.patch, W // self.cfg.patch
        assert patch_tokens.shape[0] == h_patches * w_patches, "token count mismatch"

        feats = patch_tokens.T.contiguous().view(C, h_patches, w_patches)
        feats = F.normalize(feats, p=2, dim=0)
        return feats.detach().cpu(), h_patches, w_patches

    @torch.inference_mode()
    def _quantize_mask(self, mask_bool: np.ndarray, h_patches: int, w_patches: int,
                       image_pil: Image.Image) -> torch.Tensor:
        """Average-pool mask to patch grid; returns [Hp, Wp] float tensor in [0,1]."""
        m_pil = Image.fromarray((mask_bool.astype(np.uint8) * 255))
        m_t = resize_transform_patches(m_pil, self.cfg.image_size, self.cfg.patch)
        if m_t.shape[0] > 1:
            m_t = m_t[-1:].contiguous()
        q = self.patch_quant_filter(m_t.unsqueeze(0).to(self.device)).squeeze(0).squeeze(0).detach().cpu()
        if q.shape != (h_patches, w_patches):
            q = TF.resize(q.unsqueeze(0), (h_patches, w_patches),
                          interpolation=TF.InterpolationMode.BILINEAR).squeeze(0)
        return q

    @torch.no_grad()
    def extract_flat_tokens_and_fg(self, pil_img: Image.Image):
        """Return (tokens [K,C], Hp, Wp, fg_mask [K]) after BG removal."""
        proc_img, mask_bool = self._prep_image_and_mask(pil_img, None)
        feats, Hp, Wp = self._extract_patch_features(proc_img)
        qmask = self._quantize_mask(mask_bool, Hp, Wp, proc_img)
        toks = feats.view(feats.shape[0], -1).permute(1, 0).contiguous()
        fg = (qmask.view(-1) > float(self.cfg.mask_fg_threshold))
        return toks, Hp, Wp, fg

    @torch.no_grad()
    def _extract_tokens_preprocessed(self, pil_img: Image.Image):
        """Extract tokens with BG removal (matching cache preprocessing)."""
        proc_img, _ = self._prep_image_and_mask(pil_img, None)
        feats, Hp, Wp = self._extract_patch_features(proc_img)
        toks = feats.view(feats.shape[0], -1).permute(1, 0).contiguous()
        return toks, Hp, Wp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_fg_nn_field(self, pil_images: List[Image.Image]) -> dict:
        """
        Build foreground-filtered mutual NN correspondences for a set of images.

        Returns a cache dict to pass to score_fixed_matches_with_details().
        The cache is independent of which images will be scored later.
        """
        per_img = []
        for img in pil_images:
            toks, Hp, Wp, fg = self.extract_flat_tokens_and_fg(img)
            per_img.append({"toks": toks, "Hp": Hp, "Wp": Wp, "fg": fg})
        shapes = [(p["Hp"], p["Wp"]) for p in per_img]

        cfg = self.cfg
        pairs = {}
        for (i, Ai), (j, Bj) in combinations(list(enumerate(per_img)), 2):
            A, B = Ai["toks"], Bj["toks"]
            fgA, fgB = Ai["fg"], Bj["fg"]
            S = A @ B.t()
            T = max(float(cfg.temperature), 1e-6)
            S_sharp = S / T

            # L->R
            top2A = S_sharp.topk(k=min(2, S_sharp.shape[1]), dim=1)
            idx_ab = top2A.indices[:, 0]
            keepA = torch.ones_like(idx_ab, dtype=torch.bool)
            if S_sharp.shape[1] > 1 and cfg.ratio > 1.0:
                keepA &= (top2A.values[:, 0] / (top2A.values[:, 1] + 1e-12)) >= cfg.ratio
            sim_ab = S.gather(1, idx_ab.unsqueeze(1)).squeeze(1)
            keepA &= (sim_ab >= float(cfg.sim_thresh))
            if cfg.prob_thresh > 0.0:
                pa = torch.softmax(S_sharp, dim=1)[torch.arange(S.shape[0]), idx_ab]
                keepA &= (pa >= cfg.prob_thresh)

            # R->L
            top2B = S_sharp.topk(k=min(2, S_sharp.shape[0]), dim=0)
            idx_ba = top2B.indices[0]
            keepB = torch.ones_like(idx_ba, dtype=torch.bool)
            if S_sharp.shape[0] > 1 and cfg.ratio > 1.0:
                keepB &= (top2B.values[0] / (top2B.values[1] + 1e-12)) >= cfg.ratio
            sim_ba = S[idx_ba, torch.arange(S.shape[1])]
            keepB &= (sim_ba >= float(cfg.sim_thresh))
            if cfg.prob_thresh > 0.0:
                pb = torch.softmax(S_sharp, dim=0)[idx_ba, torch.arange(S.shape[1])]
                keepB &= (pb >= cfg.prob_thresh)

            a_ids = torch.arange(S.shape[0])
            bi = (keepA & keepB[idx_ab] & (idx_ba[idx_ab] == a_ids)) if cfg.mutual else keepA
            fgB_at = fgB[idx_ab.cpu()].to(bi.device)
            keep = bi & fgA.to(bi.device) & fgB_at
            pairs[(i, j)] = (torch.nonzero(keep).squeeze(1).cpu(), idx_ab[keep].cpu())

        return {"pairs": pairs, "shapes": shapes}

    @torch.no_grad()
    def score_fixed_matches_with_details(self,
                                         pil_images: List[Image.Image],
                                         cache: dict,
                                         use_preprocessed: bool = True) -> dict:
        """
        Score images using correspondences from build_fg_nn_field().

        Args:
            pil_images: images to score (same count as used to build cache)
            cache: dict returned by build_fg_nn_field()
            use_preprocessed: True  — applies BG removal (use for input/reference images)
                              False — skips BG removal (use for output/edited images)

        Returns dict with keys:
            score         — exp(-mean_cosine_distance / tau), in (0, 1]
            mean_distance — raw average 1-cos over all pairs
            pair_count    — number of valid image pairs scored
            total_matches — total correspondences used
            match_pairs   — per-pair {"mean_distance": float, "matches": int}
        """
        toks_list = []
        shapes_now = []
        for im in pil_images:
            if use_preprocessed:
                toks, Hp, Wp = self._extract_tokens_preprocessed(im)
            else:
                feats, Hp, Wp = self._extract_patch_features(im)
                toks = feats.view(feats.shape[0], -1).permute(1, 0).contiguous()
            toks_list.append(toks)
            shapes_now.append((Hp, Wp))

        pair_details = {}
        dists = []
        match_counts = []
        for (i, j), (a_idx, b_idx) in cache["pairs"].items():
            if cache["shapes"][i] != shapes_now[i] or cache["shapes"][j] != shapes_now[j]:
                continue
            A, B = toks_list[i], toks_list[j]
            valid = (a_idx < A.shape[0]) & (b_idx < B.shape[0])
            if not valid.any():
                continue
            sims = (A[a_idx[valid]] * B[b_idx[valid]]).sum(dim=1).clamp(-1, 1)
            dist = (1 - sims).mean().item()
            dists.append(dist)
            mc = int(valid.sum().item())
            match_counts.append(mc)
            pair_details[f"{i},{j}"] = {"mean_distance": float(dist), "matches": mc}

        if not dists:
            return {"score": 1.0, "mean_distance": None, "pair_count": 0,
                    "total_matches": 0, "match_pairs": pair_details}
        mean_d = float(np.mean(dists))
        return {
            "score":         float(math.exp(-mean_d / float(self.cfg.tau))),
            "mean_distance": mean_d,
            "pair_count":    len(dists),
            "total_matches": int(sum(match_counts)),
            "match_pairs":   pair_details,
        }

    @torch.no_grad()
    def visualize_cache_overlays_matches_style(self,
                                               pil_images: List[Image.Image],
                                               cache: dict,
                                               out_dir: str,
                                               tag_prefix: str = "matchsim",
                                               remove_bg: Optional[bool] = None):
        """
        Render side-by-side match overlays for each cached pair drawn on pil_images.

        Args:
            remove_bg: override BG-removal for preprocessing (None = use self.bg.remove_bg)
        """
        apply_bg = self.bg.remove_bg if remove_bg is None else remove_bg
        os.makedirs(out_dir, exist_ok=True)

        proc_tensors = []
        for im in pil_images:
            proc_img = self.bg_remover.apply(im)[0] if apply_bg else im
            t = resize_transform_patches(proc_img.convert("RGB"), self.cfg.image_size, self.cfg.patch)
            proc_tensors.append(t)

        patch = self.cfg.patch
        n = len(cache["shapes"])
        default_indices = list(range(n))

        for (i, j), (a_idx, b_idx) in cache["pairs"].items():
            if i >= len(proc_tensors) or j >= len(proc_tensors):
                continue
            t_i, t_j = proc_tensors[i], proc_tensors[j]
            if (t_i.shape[1] // patch, t_i.shape[2] // patch) != cache["shapes"][i]:
                continue
            if (t_j.shape[1] // patch, t_j.shape[2] // patch) != cache["shapes"][j]:
                continue

            left_tag  = cache.get("orig_indices", default_indices)[i]
            right_tag = cache.get("orig_indices", default_indices)[j]

            left_img  = (t_i.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            right_img = (t_j.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            left_overlay  = Image.fromarray(left_img).convert("RGBA")
            right_overlay = Image.fromarray(right_img).convert("RGBA")
            dL = ImageDraw.Draw(left_overlay)
            dR = ImageDraw.Draw(right_overlay)

            total = max(1, len(a_idx))
            for idx_pos, (ka, kb) in enumerate(zip(a_idx.tolist(), b_idx.tolist())):
                h = idx_pos / total
                r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
                col = (int(r * 255), int(g * 255), int(b * 255))
                rA, cA = divmod(int(ka), cache["shapes"][i][1])
                rB, cB = divmod(int(kb), cache["shapes"][j][1])
                for draw_obj, rP, cP in ((dL, rA, cA), (dR, rB, cB)):
                    x0, y0 = cP * patch, rP * patch
                    draw_obj.rectangle([x0, y0, x0 + patch, y0 + patch],
                                       fill=(*col, 120), outline=(*col, 255), width=1)

            L_rgb = left_overlay.convert("RGB")
            R_rgb = right_overlay.convert("RGB")
            W, H = L_rgb.size
            combined = Image.new("RGB", (W * 2, H), (255, 255, 255))
            combined.paste(L_rgb, (0, 0))
            combined.paste(R_rgb, (W, 0))
            out_path = os.path.join(out_dir, f"img{left_tag}_img{right_tag}_{tag_prefix}_matches_overlay.jpg")
            combined.save(out_path, **JPEG_SAVE_KW)


# ----------------------------
# Convenience wrapper
# ----------------------------
def dino_matchsim_score(
    input_images: List[Image.Image],
    output_images: List[Image.Image],
    cfg: Optional[DinoMatchSimCfg] = None,
    bg_cfg: Optional[BgCfg] = None,
    viz_dir: Optional[str] = None,
) -> dict:
    """
    Compute DINO-MatchSim multi-view consistency score.

    Builds patch correspondences from input_images (before edit), then measures
    how well those correspondences are preserved in output_images (after edit).
    Score = exp(-mean_cosine_distance / tau), in (0, 1].

    Args:
        input_images:  original/reference images (used to build correspondences)
        output_images: edited/generated images to evaluate
        cfg:    DinoMatchSimCfg (default: ViT-B/16, image_size=768, tau=0.6)
        bg_cfg: BgCfg (default: foreground segmentation enabled via CarveKit)
        viz_dir: if given, save match overlay images to this directory

    Returns dict with:
        dino_matchsim_output — DINO-MatchSim for outputs (higher = more consistent)
        dino_matchsim_input  — baseline score on inputs (practical upper bound)
        pair_count           — number of valid image pairs scored
        total_matches        — total correspondences used
    """
    matcher = DinoMatchSim(cfg or DinoMatchSimCfg(), bg_cfg or BgCfg())
    cache = matcher.build_fg_nn_field(input_images)
    out = matcher.score_fixed_matches_with_details(output_images, cache, use_preprocessed=False)
    inp = matcher.score_fixed_matches_with_details(input_images,  cache, use_preprocessed=True)
    if viz_dir is not None:
        matcher.visualize_cache_overlays_matches_style(
            input_images, cache, viz_dir, tag_prefix="input"
        )
        matcher.visualize_cache_overlays_matches_style(
            output_images, cache, viz_dir, tag_prefix="output", remove_bg=False
        )
    return {
        "dino_matchsim_output": out["score"],
        "dino_matchsim_input":  inp["score"],
        "pair_count":           out["pair_count"],
        "total_matches":        out["total_matches"],
    }
