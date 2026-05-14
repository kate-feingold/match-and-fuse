import math
from tqdm import tqdm
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder
from .constants import (
    CONTROLNET_GS_START, CONTROLNET_GS_END,
    GUIDANCE_STEP_LIMIT, GUIDANCE_LR_START, GUIDANCE_LR_END,
    GUIDANCE_OPT_STEPS, GUIDANCE_ACCUM_BATCH,
    FLOWEDIT_MFF_DOUBLE_BLOCK_STEP_LIMIT, FLOWEDIT_MFF_SINGLE_BLOCK_STEP_LIMIT,
    FLOWEDIT_CONTROLNET_GS_START, FLOWEDIT_START_STEP, FLOWEDIT_STOP_STEP, FLOWEDIT_SRC_GUIDANCE,
    MFF_DOUBLE_BLOCK_STEP_LIMIT, MFF_SINGLE_BLOCK_STEP_LIMIT,
)
from src.maf_flux.utils.method_utils import image_inds_in_grid, define_graph


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    def reshape_img(x):
        x = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if x.shape[0] == 1 and bs > 1:
            x = repeat(x, "1 ... -> bs ...", bs=bs)
        return x

    img = reshape_img(img)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def register_feat_hook(model, text_len):
    # All blocks
    all_block_names = [f"B{b}" for b in range(19)] + [f"SB{b}" for b in range(38)]
    block_heads_to_save = {b: list(range(24)) for b in all_block_names}

    def get_feat(name, unhooked_out_len: int, hooked_heads):
        def hook(model, input, output):
            hooked_output = output[unhooked_out_len]
            hooked_output = [rearrange(f[:, :, text_len:], "B H L D -> B L (H D)") for f in hooked_output]
            feat_dict[f"{name}_k"] = hooked_output[0]
            feat_dict[f"{name}_v"] = hooked_output[1]
        return hook

    feat_dict = {}
    handles = []
    for block_i, m in enumerate(model.double_blocks):
        if f"B{block_i}" in block_heads_to_save:
            handle = m.processor.register_forward_hook(get_feat(f"B{block_i}", unhooked_out_len=2, hooked_heads=block_heads_to_save[f"B{block_i}"]))
            handles.append(handle)

    for block_i, m in enumerate(model.single_blocks):
        if f"SB{block_i}" in block_heads_to_save:
            m.processor.text_len = text_len
            handle = m.processor.register_forward_hook(get_feat(f"SB{block_i}", unhooked_out_len=1, hooked_heads=block_heads_to_save[f"SB{block_i}"]))
            handles.append(handle)

    return feat_dict, handles


def disable_hooks(handles):
    for handle in handles:
        handle.remove()


def configure_mff_processors(mff, model, controlnet, matches, text_len, img_wh, max_adjacent_nodes: int | None = None, flowedit: bool = False):
    """Set static MFF (Multi-view Feature Fusion) config on all attention processors."""
    double_limit = FLOWEDIT_MFF_DOUBLE_BLOCK_STEP_LIMIT if flowedit else MFF_DOUBLE_BLOCK_STEP_LIMIT
    single_limit = FLOWEDIT_MFF_SINGLE_BLOCK_STEP_LIMIT if flowedit else MFF_SINGLE_BLOCK_STEP_LIMIT

    for b_i, m in enumerate(model.double_blocks):
        m.processor.mff = mff
        m.processor.attn_matches = matches
        m.processor.block_name = f"B{b_i}"
        m.processor.img_wh = img_wh
        m.processor.max_adjacent_nodes = max_adjacent_nodes
        m.processor.mff_double_step_limit = double_limit
        m.processor.mff_single_step_limit = single_limit

    for b_i, m in enumerate(model.single_blocks):
        m.processor.mff = mff
        m.processor.attn_matches = matches
        m.processor.text_len = text_len
        m.processor.block_name = f"SB{b_i}"
        m.processor.img_wh = img_wh
        m.processor.max_adjacent_nodes = max_adjacent_nodes
        m.processor.mff_double_step_limit = double_limit
        m.processor.mff_single_step_limit = single_limit

    for controlnet_ in controlnet.controlnets:
        for b_i, m in enumerate(controlnet_.double_blocks):
            m.processor.mff = mff
            m.processor.attn_matches = matches
            m.processor.block_name = f"controlnet_B{b_i}"
            m.processor.img_wh = img_wh
            m.processor.mff_double_step_limit = double_limit
            m.processor.mff_single_step_limit = single_limit


def toggle_mff(model, mff: bool):
    """Temporarily enable or disable MFF on all processors (used during guidance steps)."""
    for m in model.double_blocks:
        m.processor.mff = mff
    for m in model.single_blocks:
        m.processor.mff = mff


def lin_anneal(value_start, value_end, steps_total, step):
    return value_start + (value_end - value_start) * (step / (steps_total - 1))


def set_denoising_step(model, controlnet, step: int, steps_total: int):
    """Update the current denoising step index on all processors (controls MFF schedule thresholds)."""
    assert 0 <= step < steps_total, "Step should be in the range [0, steps_total)"
    assert steps_total > 0, "steps_total must be greater than 0."

    for m in model.double_blocks:
        m.processor.step = step
    for m in model.single_blocks:
        m.processor.step = step
    for controlnet_ in controlnet.controlnets:
        for b_i, m in enumerate(controlnet_.double_blocks):
            m.processor.step = step


def batched_inference(model, controlnet, img, img_ids, txt, txt_ids, vec, t_vec, guidance_vec, controlnet_cond, controlnet_gs, inf_batch=70):
    preds = []
    for batch_i in range(0, img.shape[0], inf_batch):
        if controlnet_gs != 0:
            block_res_samples = controlnet(
                img=img[batch_i:batch_i + inf_batch],
                img_ids=img_ids[batch_i:batch_i + inf_batch],
                controlnet_cond=[c[batch_i:batch_i + inf_batch] for c in controlnet_cond],
                txt=txt[batch_i:batch_i + inf_batch],
                txt_ids=txt_ids[batch_i:batch_i + inf_batch],
                y=vec[batch_i:batch_i + inf_batch],
                timesteps=t_vec[batch_i:batch_i + inf_batch],
                guidance=guidance_vec[batch_i:batch_i + inf_batch],
            )
            block_controlnet_hidden_states = [i * controlnet_gs for i in block_res_samples]
        else:
            block_controlnet_hidden_states = None
        pred_batch = model(
            img=img[batch_i:batch_i + inf_batch],
            img_ids=img_ids[batch_i:batch_i + inf_batch],
            txt=txt[batch_i:batch_i + inf_batch],
            txt_ids=txt_ids[batch_i:batch_i + inf_batch],
            y=vec[batch_i:batch_i + inf_batch],
            timesteps=t_vec[batch_i:batch_i + inf_batch],
            guidance=guidance_vec[batch_i:batch_i + inf_batch],
            block_controlnet_hidden_states=block_controlnet_hidden_states,
        )
        preds.append(pred_batch)
    preds = torch.cat(preds, dim=0)
    return preds


def decode(ae, x, height, width):
    from PIL import Image
    x = unpack(x.float(), height * 16, width * 16)
    bs = 4
    x = torch.concatenate([ae.decode(x[i:i + bs]) for i in range(0, x.shape[0], bs)], dim=0)
    x1 = x.clamp(-1, 1)
    output_imgs = []
    for x1_ in x1:
        x1_ = rearrange(x1_, "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1_ + 1.0)).cpu().byte().numpy())
        output_imgs.append(output_img)
    return output_imgs



def _run_guidance_opt(model, controlnet, img, guidance_graph, step, mff, txt, compute_loss_fn):
    """Run match-guidance gradient descent: pulls matched feature pairs closer together.

    compute_loss_fn(pair_i_batch, feat_dict, criterion) -> scalar loss Tensor.
    The callback runs inference (populating feat_dict via forward hooks) and computes
    the feature proximity loss for the given batch of graph edges.
    """
    toggle_mff(model, mff=False)
    feat_dict, handles = register_feat_hook(model, text_len=txt.shape[1])
    criterion = torch.nn.L1Loss()
    pair_ids = list(range(len(guidance_graph)))
    with torch.set_grad_enabled(True):
        img.requires_grad = True
        lr = lin_anneal(GUIDANCE_LR_START, GUIDANCE_LR_END, GUIDANCE_STEP_LIMIT, step)
        optimizer = torch.optim.Adam([img], lr=lr)
        for _ in range(GUIDANCE_OPT_STEPS):
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            for pair_i_batch in [pair_ids[b:b + GUIDANCE_ACCUM_BATCH] for b in range(0, len(guidance_graph), GUIDANCE_ACCUM_BATCH)]:
                torch.cuda.empty_cache()
                loss = compute_loss_fn(pair_i_batch, feat_dict, criterion)
                loss.backward()
            optimizer.step()
        toggle_mff(model, mff=mff)
        disable_hooks(handles)
        img.requires_grad = False


def denoise_controlnet_no_grid(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    timestep_to_start_cfg=0,
    mff: bool = False,
    matches: Tensor = None,
    img_wh=None,
    feat_guide: bool = False,
    # flowedit parameters
    original_img: Tensor = None,
    original_txt: Tensor = None,
    original_txt_ids: Tensor = None,
    original_vec: Tensor = None,
    flowedit: bool = False,
    flowedit_start_step: int = FLOWEDIT_START_STEP,
    flowedit_stop_step: int = FLOWEDIT_STOP_STEP,
):
    img_wh_align = (img_wh[0], img_wh[1], img_wh[0] > img_wh[1])
    configure_mff_processors(mff, model, controlnet, matches, text_len=txt.shape[1], img_wh=img_wh_align, flowedit=flowedit)

    i = 0
    if flowedit:
        i = flowedit_start_step
        img = original_img
        guidance_vec_src = torch.full((img.shape[0],), FLOWEDIT_SRC_GUIDANCE, device=img.device, dtype=img.dtype)
        # Never doing guidance with flowedit
        feat_guide = False

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    controlnet_gs_start = FLOWEDIT_CONTROLNET_GS_START if flowedit else CONTROLNET_GS_START
    pbar = tqdm(total=len(timesteps) - 1)
    while i < len(timesteps) - 1:
        pbar.update(i - pbar.n)
        t_curr, t_prev = timesteps[i], timesteps[i + 1]
        set_denoising_step(model, controlnet, i, len(timesteps) - 1)

        # Transition out of flowedit at the stop step
        if flowedit and i == flowedit_stop_step:
            eps = torch.randn_like(original_img[[0]]).repeat(original_img.shape[0], 1, 1)
            original_noised = (1 - t_curr) * original_img + t_curr * eps
            img = img + original_noised - original_img
            flowedit = False

        controlnet_gs = lin_anneal(controlnet_gs_start, CONTROLNET_GS_END, len(timesteps) - 1, i)
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        if flowedit:
            eps = torch.randn_like(original_img[[0]]).repeat(original_img.shape[0], 1, 1)
            original_noised = (1 - t_curr) * original_img + t_curr * eps
            img_target = img + original_noised - original_img
            pred_src = batched_inference(model, controlnet, original_noised, img_ids, original_txt, original_txt_ids, original_vec, t_vec, guidance_vec_src, controlnet_cond, controlnet_gs)
            pred_target = batched_inference(model, controlnet, img_target, img_ids, txt, txt_ids, vec, t_vec, guidance_vec, controlnet_cond, controlnet_gs)
            pred = pred_target - pred_src
        else:
            pred = batched_inference(model, controlnet, img, img_ids, txt, txt_ids, vec, t_vec, guidance_vec, controlnet_cond, controlnet_gs)
            if i >= timestep_to_start_cfg:
                neg_pred = batched_inference(model, controlnet, img, img_ids, neg_txt, neg_txt_ids, neg_vec, t_vec,
                                             guidance_vec, controlnet_cond, controlnet_gs)
                pred = neg_pred + true_gs * (pred - neg_pred)

        img = img + (t_prev - t_curr) * pred

        if feat_guide and i < GUIDANCE_STEP_LIMIT:
            guidance_graph, _ = define_graph(img.shape[0])
            t_vec_next = torch.full((len(img),), t_prev, dtype=img.dtype, device=img.device)

            def compute_loss_fn(pair_i_batch, feat_dict, criterion):
                edges_batch = [tuple(guidance_graph[p]) for p in pair_i_batch]
                # Unique image nodes appearing in this batch of edges
                img_ids_batch = list(set([img_id for edge in edges_batch for img_id in edge]))
                [feat_dict.pop(k) for k in list(feat_dict.keys())]
                # Run model to populate feat_dict via forward hooks (output discarded)
                _ = batched_inference(
                    model, controlnet, img[img_ids_batch],
                    img_ids[img_ids_batch], txt[img_ids_batch], txt_ids[img_ids_batch], vec[img_ids_batch],
                    t_vec_next[img_ids_batch], guidance_vec[img_ids_batch],
                    [c[img_ids_batch] for c in controlnet_cond], controlnet_gs)
                # Feature proximity loss over matched token pairs
                loss = 0.
                for edge_a, edge_b in edges_batch:
                    img_id_a = img_ids_batch.index(edge_a)
                    img_id_b = img_ids_batch.index(edge_b)
                    matches_batch_edge = matches[(edge_a, edge_b)][0]
                    valid_mask = matches_batch_edge >= 0
                    valid_inds = valid_mask.nonzero(as_tuple=True)
                    matched_inds = (1 - valid_inds[0], matches_batch_edge[valid_inds])
                    # Remap batch-local index (0/1) to img_ids_batch positions
                    valid_inds = (torch.where(valid_inds[0] == 0, img_id_a, img_id_b), valid_inds[1])
                    matched_inds = (torch.where(matched_inds[0] == 0, img_id_a, img_id_b), matched_inds[1])
                    for fmap in feat_dict.values():
                        loss += criterion(fmap[valid_inds], fmap[matched_inds]) / len(feat_dict) / len(edges_batch)
                return loss

            _run_guidance_opt(model, controlnet, img, guidance_graph, i, mff, txt, compute_loss_fn)

        i += 1

    pbar.update(i - pbar.n)
    pbar.close()

    return img


def denoise_controlnet_grid(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    timestep_to_start_cfg=0,
    mff: bool = False,
    matches: Tensor = None,
    img_wh=None,
    feat_guide: bool = False,
    max_adjacent_nodes: int | None = None,
    # flowedit parameters
    original_img: Tensor = None,
    original_txt: Tensor = None,
    original_txt_ids: Tensor = None,
    original_vec: Tensor = None,
    flowedit: bool = False,
    flowedit_start_step: int = FLOWEDIT_START_STEP,
    flowedit_stop_step: int = FLOWEDIT_STOP_STEP,
):
    def grid_feats(feats):
        img_unflatten = feats.reshape((2, img_wh[1], img_wh[0], -1))
        grid_dim = 0 if img_wh[0] > img_wh[1] else 1
        img_concat = torch.concatenate([img_unflatten[0], img_unflatten[1]], dim=grid_dim)
        return img_concat.flatten(0, 1).unsqueeze(0)

    def grid_imgs(imgs):
        grid_dim = 1 if img_wh[0] > img_wh[1] else 2
        return torch.concatenate([imgs[0], imgs[1]], dim=grid_dim).unsqueeze(0)

    num_images = img.shape[0]
    graph, edge_presence = define_graph(num_images, max_adjacent_nodes=max_adjacent_nodes)
    grid_h, grid_w = img_wh[1], img_wh[0]
    grid_h = grid_h * 2 if img_wh[0] > img_wh[1] else grid_h
    grid_w = grid_w * 2 if img_wh[0] <= img_wh[1] else grid_w
    img_ids_grid = torch.zeros(grid_h, grid_w, 3)
    img_ids_grid[..., 1] = img_ids_grid[..., 1] + torch.arange(grid_h)[:, None]
    img_ids_grid[..., 2] = img_ids_grid[..., 2] + torch.arange(grid_w)[None, :]
    img_ids = repeat(img_ids_grid, "h w c -> b (h w) c", b=len(graph)).to(img.device)
    controlnet_cond = [torch.concatenate([grid_imgs(c[pair]) for pair in graph], dim=0) for c in controlnet_cond]
    repeat_for_edges = lambda x: repeat(x[[0]], "1 ... -> b ...", b=len(graph)) if x.shape[0] != len(graph) else x
    txt, neg_txt = repeat_for_edges(txt), repeat_for_edges(neg_txt)
    txt_ids, neg_txt_ids = repeat_for_edges(txt_ids), repeat_for_edges(neg_txt_ids)
    vec, neg_vec = repeat_for_edges(vec), repeat_for_edges(neg_vec)
    guidance_vec = torch.full((len(graph),), guidance, device=img.device, dtype=img.dtype)
    token_inds_left_right = image_inds_in_grid(width=grid_w, height=grid_h, vertical=img_wh[0] > img_wh[1])
    grid_wh = (grid_w, grid_h)

    i = 0
    img_wh_align = (grid_wh[0], grid_wh[1], img_wh[0] > img_wh[1])
    configure_mff_processors(mff, model, controlnet, matches, text_len=txt.shape[1], img_wh=img_wh_align, max_adjacent_nodes=max_adjacent_nodes, flowedit=flowedit)

    if flowedit:
        i = flowedit_start_step
        img = original_img
        # Grid original_img per edge
        original_img = torch.concat([grid_feats(original_img[pair]) for pair in graph], dim=0)
        original_txt = repeat_for_edges(original_txt)
        original_txt_ids = repeat_for_edges(original_txt_ids)
        original_vec = repeat_for_edges(original_vec)
        guidance_vec_src = torch.full((len(graph),), FLOWEDIT_SRC_GUIDANCE, device=img.device, dtype=img.dtype)
        feat_guide = False

    controlnet_gs_start = FLOWEDIT_CONTROLNET_GS_START if flowedit else CONTROLNET_GS_START
    pbar = tqdm(total=len(timesteps) - 1)
    while i < len(timesteps) - 1:
        pbar.update(i - pbar.n)
        t_curr, t_prev = timesteps[i], timesteps[i + 1]

        # Generate noise before stacking (per-image, shared across images for consistency)
        if flowedit:
            eps = torch.randn_like(img)
            eps = torch.concat([grid_feats(eps[pair]) for pair in graph], dim=0)

        # Arrange images into grid pairs along graph edges
        img = torch.concat([grid_feats(img[pair]) for pair in graph], dim=0)

        # Transition out of flowedit at the stop step
        if flowedit and i == flowedit_stop_step:
            original_noised = (1 - t_curr) * original_img + t_curr * eps
            img = img + original_noised - original_img
            flowedit = False

        set_denoising_step(model, controlnet, i, len(timesteps) - 1)

        controlnet_gs = lin_anneal(controlnet_gs_start, CONTROLNET_GS_END, len(timesteps) - 1, i)
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        if flowedit:
            original_noised = (1 - t_curr) * original_img + t_curr * eps
            img_target = img + original_noised - original_img
            pred_src = batched_inference(model, controlnet, original_noised, img_ids, original_txt, original_txt_ids, original_vec, t_vec, guidance_vec_src, controlnet_cond, controlnet_gs)
            pred_target = batched_inference(model, controlnet, img_target, img_ids, txt, txt_ids, vec, t_vec, guidance_vec, controlnet_cond, controlnet_gs)
            pred = pred_target - pred_src
        else:
            pred = batched_inference(model, controlnet, img, img_ids, txt, txt_ids, vec, t_vec, guidance_vec, controlnet_cond, controlnet_gs)
            if i >= timestep_to_start_cfg:
                neg_pred = batched_inference(
                    model, controlnet, img, img_ids, neg_txt, neg_txt_ids, neg_vec, t_vec, guidance_vec,
                                         controlnet_cond, controlnet_gs)
                pred = neg_pred + true_gs * (pred - neg_pred)

        img = img + (t_prev - t_curr) * pred

        # Average and cut image
        img_avgs = []
        for img_id, presence in edge_presence.items():
            img_i_avg = sum([img[edge_i, token_inds_left_right[position]] for edge_i, position in presence]) / len(presence)
            img_avgs.append(img_i_avg)
        img = torch.stack(img_avgs, dim=0)

        # Ensure model params don't accumulate gradients during feature guidance
        for p in model.parameters():
            p.requires_grad = False
        if feat_guide and i < GUIDANCE_STEP_LIMIT:
            t_vec_next = torch.full((len(graph),), t_prev, dtype=img.dtype, device=img.device)

            def compute_loss_fn(pair_i_batch, feat_dict, criterion):
                matches_batch = torch.concatenate([matches[tuple(graph[p])] for p in pair_i_batch], 0)
                valid_mask = matches_batch >= 0
                valid_inds = valid_mask.nonzero(as_tuple=True)
                matched_inds = (valid_inds[0], matches_batch[valid_inds])
                guide_in_grid = torch.concat([grid_feats(img[graph[pair_i]]) for pair_i in pair_i_batch], dim=0)
                [feat_dict.pop(k) for k in list(feat_dict.keys())]
                # Run model to populate feat_dict via forward hooks (output discarded)
                _ = batched_inference(
                    model, controlnet, guide_in_grid,
                    img_ids[pair_i_batch], txt[pair_i_batch], txt_ids[pair_i_batch], vec[pair_i_batch],
                    t_vec_next[pair_i_batch], guidance_vec[pair_i_batch],
                    [c[pair_i_batch] for c in controlnet_cond], controlnet_gs)
                # Feature proximity loss (averaged over feature maps)
                loss = 0.
                for fmap in feat_dict.values():
                    loss += criterion(fmap[valid_inds], fmap[matched_inds]) / len(feat_dict)
                return loss

            _run_guidance_opt(model, controlnet, img, graph, i, mff, txt, compute_loss_fn)

        i += 1

    pbar.update(i - pbar.n)
    pbar.close()
    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
