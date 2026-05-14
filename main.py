import argparse
import json
import os
from pathlib import Path

from PIL import Image

from src.maf_flux.maf_pipeline import MatchAndFusePipeline
from src.maf_flux.utils.method_utils import vgrid_pil, hgrid_pil, define_graph, load_matches, image_path_to_prompt
from src.maf_flux.utils.image_utils import compute_scene_resolution, center_crop_and_resize, load_images_from_folder
from src.maf_flux.constants import GUIDANCE, FLOWEDIT_GUIDANCE, TRUE_GS, TIMESTEP_TO_START_CFG
from src.maf_flux.compute_matches import compute_matches as do_compute_matches
from src.maf_flux.captioning import caption_data


def create_argparser():
    parser = argparse.ArgumentParser()

    # --- Input ---
    parser.add_argument("--images", type=str, required=True, default=None, nargs='+',
                        help="Image paths, or a single folder path to use all images inside it")
    parser.add_argument("--captions_path", type=str, default=None, help="Path to the captions json")
    # Both --prompt_shared and --prompt_theme are required when --captions_path is not provided.
    parser.add_argument("--prompt_shared", type=str, default=None, help="Target shared content prompt (P^{shared})")
    parser.add_argument("--prompt_theme", type=str, default=None, help="Target theme prompt (P^{theme})")
    parser.add_argument("--caption_model", type=str, default="gpt-4o",
                        help="Vision LLM for auto-captioning when --captions_path is not provided. "
                             "OpenAI models (OPENAI_API_KEY), claude-* (ANTHROPIC_API_KEY), gemini-* (GOOGLE_API_KEY).")
    parser.add_argument("--neg_prompt", type=str, default="bad photo", help="Negative prompt")
    parser.add_argument("--matches_dir", type=str, default=None,
                        help="Path to folder with precomputed match .pth files. "
                             "Optional: auto-discovered from data/ if omitted, or computed on the fly.")

    # --- Method (all components are on by default; use --independent to disable all) ---
    parser.add_argument("--independent", action="store_true", help="Disable pairwise graph, MFF, and guidance (all off)")
    parser.add_argument("--no_pair_graph", action="store_true", help="Disable Pairwise Consistency Graph")
    parser.add_argument("--no_mff", action="store_true", help="Disable Multi-view Feature Fusion")
    parser.add_argument("--no_feat_guide", action="store_true", help="Disable Feature Guidance")
    parser.add_argument("--max_adjacent_nodes", type=int, default=None,
                        help="Max graph edges per node (must be even). Defaults to 4 for >5 images, None (fully connected) otherwise.")

    # --- Diffusion ---
    parser.add_argument("--seed", type=int, default=2, help="Seed for reproducibility")
    parser.add_argument("--width", type=int, default=None,
                        help="Output width in pixels (divisible by 64). Inferred to best fit the images if omitted.")
    parser.add_argument("--height", type=int, default=None,
                        help="Output height in pixels (divisible by 64). Inferred to best fit the images if omitted.")

    # --- ControlNet ---
    parser.add_argument("--control_type", default=["depth", "hed"], nargs='+', choices=("hed", "depth"),
                        help="ControlNet condition type(s)")
    parser.add_argument("--local_path", type=str, default=None, help="Local path to ControlNet checkpoint")
    parser.add_argument("--repo_id", default=["XLabs-AI/flux-controlnet-depth-v3", "XLabs-AI/flux-controlnet-hed-v3"],
                        nargs='+', help="HuggingFace repo id for ControlNet")
    parser.add_argument("--name", default=["flux-depth-controlnet-v3.safetensors", "flux-hed-controlnet-v3.safetensors"],
                        nargs='+', help="Filename to download from HuggingFace")

    # --- FlowEdit ---
    parser.add_argument("--flowedit", action="store_true",
                        help="Use FlowEdit: edit from original images instead of pure noise. "
                             "Requires --captions_path with src/per_image_non_shared_src fields.")

    # --- Infrastructure ---
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--offload", action='store_true', help="Offload model to CPU when not in use")
    parser.add_argument("--save_path", type=str, default='results', help="Path to save results")

    return parser


def _resolve_image_paths(images_arg: list[str]) -> list[str]:
    """Expand a single directory argument to a sorted list of image files."""
    if len(images_arg) == 1 and Path(images_arg[0]).is_dir():
        return load_images_from_folder(images_arg[0])
    return images_arg


def _resolve_resolution(args, image_paths):
    if args.width is None or args.height is None:
        w, h = compute_scene_resolution(image_paths)
        print(f"Auto-detected resolution: {w}x{h}")
    else:
        w, h = args.width, args.height
    return 64 * (w // 64), 64 * (h // 64)


def _resolve_matches(args, image_paths, mff, feat_guide, use_pair_graph, width, height):
    if not (mff or feat_guide):
        return None

    matches_folder = args.matches_dir
    if matches_folder is None:
        # Auto-discover: if images live under data/images/…, use parallel data/matches/…
        folder = Path(image_paths[0]).parent.resolve()
        data_images = Path(__file__).parent / "data" / "images"
        try:
            rel = folder.relative_to(data_images)
            candidate = Path(__file__).parent / "data" / "matches" / rel
            if candidate.exists() and any(candidate.glob("*_matches.pth")):
                matches_folder = str(candidate)
                print(f"Auto-discovered matches: {matches_folder}")
        except ValueError:
            pass

    if matches_folder is None:
        computed_path = Path(args.save_path) / "matches"
        print(f"Computing matches -> {computed_path}")
        do_compute_matches(image_paths, computed_path, width, height, device=args.device)
        matches_folder = str(computed_path)

    graph, _ = define_graph(len(image_paths))
    match_path = Path(matches_folder)
    matches = {}
    for edge in graph:
        for ef, es in [edge, edge[::-1]]:
            m = load_matches(match_path, image_paths, ef, es,
                             grid_matches=use_pair_graph,
                             grid_horizontal=width <= height, flatten=True)
            matches[(ef, es)] = m.to(args.device)
    return matches


def _load_caption(args, image_paths, width, height):
    assert (args.prompt_shared is None) == (args.prompt_theme is None), \
        "Must provide both --prompt_shared and --prompt_theme or neither"
    if args.captions_path is not None:
        with open(args.captions_path) as f:
            caption = json.load(f)
    elif args.prompt_shared is not None:
        caption = caption_data(image_paths, img_wh=(width, height),
                               prompt_shared=args.prompt_shared, prompt_theme=args.prompt_theme,
                               model=args.caption_model)
    else:
        raise ValueError("Must give either --captions_path or both --prompt_shared and --prompt_theme")
    with open(os.path.join(args.save_path, "captions.json"), 'w') as f:
        json.dump(caption, f, indent=4)
    return caption


def _build_prompts(image_paths, width, height, caption, use_pair_graph, max_adjacent_nodes, flowedit):
    prompts = image_path_to_prompt(
        image_paths, img_wh=(width, height),
        use_pair_graph=use_pair_graph, caption=caption,
        max_adjacent_nodes=max_adjacent_nodes, flowedit=flowedit,
    )
    src_prompts = image_path_to_prompt(
        image_paths, img_wh=(width, height),
        use_pair_graph=use_pair_graph, caption=caption,
        max_adjacent_nodes=max_adjacent_nodes, mode="src",
    ) if flowedit else None
    return prompts, src_prompts


def _save_results(outputs, images, controlnet_images, args):
    save = args.save_path
    for i in range(len(outputs)):
        outputs[i].save(os.path.join(save, f"output_{i}.png"))
        images[i].save(os.path.join(save, f"input_{i}.png"))
        for ctrl_imgs, ctrl_name in zip(controlnet_images, args.control_type):
            ctrl_imgs[i].save(os.path.join(save, f"control_{ctrl_name}_{i}.png"))
    vgrid_pil(hgrid_pil(*images), hgrid_pil(*outputs)).save(os.path.join(save, "all_inputs_outputs.png"))
    vgrid_pil(*[hgrid_pil(*ct) for ct in controlnet_images]).save(os.path.join(save, "all_controls.png"))


def main(args):
    use_pair_graph = not (args.independent or args.no_pair_graph)
    mff = not (args.independent or args.no_mff)
    feat_guide = not (args.independent or args.no_feat_guide)

    os.makedirs(args.save_path, exist_ok=True)
    Path(args.save_path, "args.txt").write_text(str(args))

    image_paths = _resolve_image_paths(args.images)
    width, height = _resolve_resolution(args, image_paths)
    images = [center_crop_and_resize(Image.open(p).convert("RGB"), width, height) for p in image_paths]
    max_adjacent_nodes = args.max_adjacent_nodes or (4 if len(image_paths) > 5 else None)

    matches = _resolve_matches(args, image_paths, mff, feat_guide, use_pair_graph, width, height)
    caption = _load_caption(args, image_paths, width, height)
    prompts, src_prompts = _build_prompts(image_paths, width, height, caption, use_pair_graph, max_adjacent_nodes, args.flowedit)

    maf_pipeline = MatchAndFusePipeline(
        "flux-dev", args.device, args.offload,
        mff=mff, matches=matches, use_pair_graph=use_pair_graph,
        feat_guide=feat_guide, max_adjacent_nodes=max_adjacent_nodes, flowedit=args.flowedit,
    )
    print('Loading controlnets:', args.local_path, args.repo_id, args.name)
    maf_pipeline.set_controlnet(args.control_type, args.local_path, args.repo_id, args.name)

    outputs, controlnet_images = maf_pipeline(
        prompts=prompts, src_prompts=src_prompts,
        controlnet_image=images, width=width, height=height,
        guidance=FLOWEDIT_GUIDANCE if args.flowedit else GUIDANCE,
        seed=args.seed, true_gs=TRUE_GS,
        neg_prompt=args.neg_prompt, timestep_to_start_cfg=TIMESTEP_TO_START_CFG,
    )
    _save_results(outputs, images, controlnet_images, args)


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
