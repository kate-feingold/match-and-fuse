from collections import defaultdict
import torch
from einops import rearrange
from torch import Tensor
from src.maf_flux.utils.method_utils import image_inds_in_grid, define_graph, get_num_vertices
from .constants import MFF_DOUBLE_BLOCK_STEP_LIMIT, MFF_SINGLE_BLOCK_STEP_LIMIT, MFF_SINGLE_BLOCK_IDX_START


def apply_mff(
    k: Tensor, v: Tensor, text_len: int, attn_matches: dict,
    max_adjacent_nodes: int | None, img_wh_align: tuple[int, int, bool],
    step: int, block_name: str,
    mff_double_step_limit: int,
    mff_single_step_limit: int,
) -> tuple[Tensor, Tensor]:
    """Average K/V features across graph-matched image tokens (Multi-view Feature Fusion).

    Returns (k, v) unchanged if the current block/step is outside the MFF active schedule,
    otherwise returns the fused (k, v).
    """
    is_double = step is not None and step < mff_double_step_limit and block_name.startswith("B")
    is_single = (step is not None and step < mff_single_step_limit
                 and block_name.startswith("SB")
                 and int(block_name.replace("SB", "")) >= MFF_SINGLE_BLOCK_IDX_START)
    if not (is_double or is_single):
        return k, v

    assert attn_matches is not None, "Need attn_matches for MFF"

    if list(attn_matches.values())[0].ndim == 2:
        # Grid mode: images arrange canvases in pairs; matches shape [1, seq_len], batch dim = number of edges
        return _apply_mff_grid(k, v, text_len, attn_matches, max_adjacent_nodes, img_wh_align)
    else:
        # No-grid mode: images processed separately; matches shape [1, 2, seq_len], batch dim = number of images
        return _apply_mff_no_grid(k, v, text_len, attn_matches, img_wh_align)


def _apply_mff_grid(
    k: Tensor, v: Tensor, text_len: int, attn_matches: dict,
    max_adjacent_nodes: int | None, img_wh_align: tuple[int, int, bool],
) -> tuple[Tensor, Tensor]:
    """MFF for grid mode: images are arranged in canvases in pairs; batch dim is number of edges."""
    num_images = get_num_vertices(k.shape[0], max_adjacent_nodes=max_adjacent_nodes)
    graph, edge_presence = define_graph(num_images, max_adjacent_nodes=max_adjacent_nodes)
    token_inds_left_right = image_inds_in_grid(
        width=img_wh_align[0], height=img_wh_align[1], vertical=img_wh_align[2]
    )

    matched_feat = {img_id: defaultdict(list) for img_id in range(num_images)}
    matched_mask = {img_id: [] for img_id in range(num_images)}

    # Gather K/V features across the graph
    for img_id in range(num_images):
        # Own features: always valid, contribute 1 to the averaging mask
        for edge_i, position_self in edge_presence[img_id]:
            token_ids_self = token_inds_left_right[position_self]
            matched_mask[img_id].append(torch.ones_like(token_ids_self).to(k.device))
            for feat, feat_name in [(k, "k"), (v, "v")]:
                matched_feat[img_id][feat_name].append(feat[edge_i, :, text_len + token_ids_self])

        # Matched features from other images: contribute 0 where unmatched
        for img_id_other in range(num_images):
            if img_id_other == img_id:
                continue
            for edge_i, position_other in edge_presence[img_id_other]:
                match_edge = (img_id, img_id_other) if position_other == 1 else (img_id_other, img_id)
                matched_ids = attn_matches[match_edge][0, token_inds_left_right[1 - position_other]]
                valid_mask = matched_ids >= 0
                matched_mask[img_id].append(valid_mask)
                for feat, feat_name in [(k, "k"), (v, "v")]:
                    gathered = feat[edge_i, :, text_len + matched_ids]
                    gathered[:, ~valid_mask] = 0.
                    matched_feat[img_id][feat_name].append(gathered)

    # Write averaged features back into k/v for each edge position
    for img_id in range(num_images):
        feat_avg = {
            feat_name: sum(matched_feat[img_id][feat_name]) / sum(matched_mask[img_id]).view(1, -1, 1)
            for feat_name in matched_feat[img_id]
        }
        for edge_i, position in edge_presence[img_id]:
            k[edge_i, :, text_len + token_inds_left_right[position]] = feat_avg["k"]
            v[edge_i, :, text_len + token_inds_left_right[position]] = feat_avg["v"]

    return k, v


def _apply_mff_no_grid(
    k: Tensor, v: Tensor, text_len: int, attn_matches: dict,
    img_wh_align: tuple[int, int, bool],
) -> tuple[Tensor, Tensor]:
    """MFF for no-grid mode: images are processed separately; batch dim is number of images.

    Matches shape: [1, 2, seq_len] — direction 0 is img_id→img_id_other.
    """
    img_tokens_len = img_wh_align[0] * img_wh_align[1]
    num_images = k.shape[0]

    matched_feat = {img_id: defaultdict(list) for img_id in range(num_images)}
    matched_mask = {img_id: [] for img_id in range(num_images)}

    for img_id in range(num_images):
        # Own features: always valid
        matched_mask[img_id].append(torch.ones(img_tokens_len, device=k.device))
        for feat, feat_name in [(k, "k"), (v, "v")]:
            matched_feat[img_id][feat_name].append(feat[img_id, :, text_len:text_len + img_tokens_len])

        # Matched features from each other image
        for img_id_other in range(num_images):
            if img_id_other == img_id:
                continue
            # Direction 0: for each token of img_id, which token in img_id_other it matches
            matched_ids = attn_matches[(img_id, img_id_other)][0, 0]  # [seq_len]
            valid_mask = matched_ids >= 0
            matched_mask[img_id].append(valid_mask)
            for feat, feat_name in [(k, "k"), (v, "v")]:
                gathered = feat[img_id_other, :, text_len + matched_ids]
                gathered[:, ~valid_mask] = 0.
                matched_feat[img_id][feat_name].append(gathered)

    for img_id in range(num_images):
        feat_avg = {
            feat_name: sum(matched_feat[img_id][feat_name]) / sum(matched_mask[img_id]).view(1, -1, 1)
            for feat_name in matched_feat[img_id]
        }
        k[img_id, :, text_len:text_len + img_tokens_len] = feat_avg["k"]
        v[img_id, :, text_len:text_len + img_tokens_len] = feat_avg["v"]

    return k, v


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, *, step: int,
              text_len: int = None, attn_matches=None, mff: bool = False,
              max_adjacent_nodes: int | None = None, block_name: str = None,
              img_wh_align: tuple[int, int, bool] = None,
              mff_double_step_limit: int = MFF_DOUBLE_BLOCK_STEP_LIMIT,
              mff_single_step_limit: int = MFF_SINGLE_BLOCK_STEP_LIMIT) -> Tensor:
    if mff:
        k, v = apply_mff(k, v, text_len, attn_matches, max_adjacent_nodes, img_wh_align, step, block_name,
                         mff_double_step_limit=mff_double_step_limit,
                         mff_single_step_limit=mff_single_step_limit)

    k_unroped = k
    k = apply_rope(k, pe)
    q = apply_rope(q, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
    x = rearrange(x, "B H L D -> B L (H D)")

    if not mff:
        # Hooked for guidance
        return x, (k_unroped, v)
    else:
        return x, ()


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    x_out = freqs_cis[..., 0] * x_[..., 0] + freqs_cis[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape).type_as(x)
