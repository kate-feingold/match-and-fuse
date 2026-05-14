import os
import json
import base64
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image

import torch
import gradio as gr

from src.maf_flux.maf_pipeline import MatchAndFusePipeline
from src.maf_flux.utils.method_utils import vgrid_pil, hgrid_pil, define_graph, load_matches, image_path_to_prompt
from src.maf_flux.utils.image_utils import compute_scene_resolution, center_crop_and_resize, load_images_from_folder
from src.maf_flux.constants import GUIDANCE, FLOWEDIT_GUIDANCE, TRUE_GS, TIMESTEP_TO_START_CFG
from src.maf_flux.compute_matches import compute_matches
from src.maf_flux.captioning import caption_data

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DATA_ROOT = Path(__file__).parent / "data"
IMAGES_ROOT = DATA_ROOT / "images"
MATCHES_ROOT = DATA_ROOT / "matches"
CAPTIONS_ROOT = DATA_ROOT / "benchmark_captions"

_CAPTION_FORMAT_HINT = """\
```json
{
  "src":  "source object description",
  "edit": "edited object description",
  "per_image_non_shared_src":  {"img_name_stem": "pose+bg desc", ...},
  "per_image_non_shared_edit": {"img_name_stem": "pose+bg desc", ...}
}
```
Omit `per_image_non_shared_*` keys for a shared prompt across all images (similar backgrounds).
When **FlowEdit** is enabled, `*src` keys must be specified, otherwise can be omitted. 
"""


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def _discover_scenes() -> dict:
    """Return {scene_name: {dataset, image_dir, match_dir, prompts: [str]}}."""
    registry = {}
    if not IMAGES_ROOT.exists():
        return registry
    for dataset_dir in sorted(IMAGES_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for scene_dir in sorted(dataset_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene = scene_dir.name
            dataset = dataset_dir.name
            has_images = any(f.suffix.lower() in _IMG_EXTS for f in scene_dir.iterdir())
            match_dir = MATCHES_ROOT / dataset / scene
            has_matches = match_dir.exists() and any(match_dir.glob("*_matches.pth"))
            prefix = f"{scene}_p"
            pnums = sorted(
                f.stem[len(prefix):]
                for f in CAPTIONS_ROOT.glob(f"{prefix}*.json")
                if f.stem.startswith(prefix)
            )
            if has_images and has_matches and pnums:
                registry[scene] = {
                    "dataset": dataset,
                    "image_dir": str(scene_dir),
                    "match_dir": str(match_dir),
                    "prompts": pnums,
                }
    return registry


SCENE_REGISTRY = _discover_scenes()
SCENE_NAMES = list(SCENE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _subsample(paths: list, max_img: int) -> list:
    n = len(paths)
    if max_img >= n:
        return paths
    step = (n - 1) / (max_img - 1) if max_img > 1 else 0
    return [paths[round(i * step)] for i in range(max_img)]


def load_dataset_scene(scene_name, max_img):
    if not scene_name or scene_name not in SCENE_REGISTRY:
        return None, None, None, None
    info = SCENE_REGISTRY[scene_name]
    paths = _subsample(load_images_from_folder(info["image_dir"]), int(max_img))
    w, h = compute_scene_resolution(paths)
    w, h = 64 * (w // 64), 64 * (h // 64)
    imgs = [center_crop_and_resize(Image.open(p).convert("RGB"), w, h) for p in paths]
    return hgrid_pil(*imgs), paths, (w, h), info["match_dir"]


def load_custom_images(files, max_img):
    if not files:
        return None, None, None, None
    paths = _subsample(sorted(f if isinstance(f, str) else f.name for f in files), int(max_img))
    w, h = compute_scene_resolution(paths)
    w, h = 64 * (w // 64), 64 * (h // 64)
    imgs = [center_crop_and_resize(Image.open(p).convert("RGB"), w, h) for p in paths]
    return hgrid_pil(*imgs), paths, (w, h), None  # matches_dir unknown until run time


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def update_preset_choices(scene_name):
    if not scene_name or scene_name not in SCENE_REGISTRY:
        return gr.update(choices=[], value=None)
    pnums = SCENE_REGISTRY[scene_name]["prompts"]
    choices = [f"p{p}" for p in pnums]
    return gr.update(choices=choices, value=choices[0] if choices else None)


def load_preset_caption(scene_name, prompt_id):
    if not scene_name or not prompt_id:
        return ""
    pnum = prompt_id.lstrip("p")
    path = CAPTIONS_ROOT / f"{scene_name}_p{pnum}.json"
    if not path.exists():
        return ""
    with open(path) as f:
        return json.dumps(json.load(f), indent=2)


# ---------------------------------------------------------------------------
# Auto-captioning
# ---------------------------------------------------------------------------

_API_PROVIDERS = {
    "OpenAI":    {"default_model": "gpt-4o",                 "env_var": "OPENAI_API_KEY"},
    "Anthropic": {"default_model": "claude-haiku-4-5",       "env_var": "ANTHROPIC_API_KEY"},
    "Google":    {"default_model": "gemini-2.5-flash-lite",  "env_var": "GOOGLE_API_KEY"},
}


def run_autocaption(image_paths, wh, obj_prompt, bg_prompt, provider, model):
    if not image_paths:
        raise gr.Error("Load images first.")
    if not obj_prompt:
        raise gr.Error("Edit description is required.")

    env_var = _API_PROVIDERS[provider]["env_var"]
    if not os.environ.get(env_var):
        raise gr.Error(f"{env_var} is not set. Export it in the shell before launching the demo.")

    try:
        cap = caption_data(image_paths, wh, obj_prompt, bg_prompt, model)
    except ImportError as e:
        raise gr.Error(f"SDK not installed: {e}")
    return json.dumps(cap, indent=2)


def key_status_md(provider):
    env_var = _API_PROVIDERS[provider]["env_var"]
    if env_var in os.environ:
        return f"✅ `{env_var}` is set."
    return f"⚠️ `{env_var}` is not set — export it before captioning."


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_maf_pipeline = None


def _load_pipeline():
    global _maf_pipeline
    if _maf_pipeline is None:
        print("Loading MatchAndFuse pipeline…")
        _maf_pipeline = MatchAndFusePipeline(
            "flux-dev", "cuda", offload=False,
            mff=True, matches=None,
            use_pair_graph=True, feat_guide=False,
        )
        _maf_pipeline.set_controlnet(
            ["depth", "hed"], None,
            ["XLabs-AI/flux-controlnet-depth-v3", "XLabs-AI/flux-controlnet-hed-v3"],
            ["flux-depth-controlnet-v3.safetensors", "flux-hed-controlnet-v3.safetensors"],
        )
        print("Pipeline ready.")
    return _maf_pipeline, gr.update(interactive=True), gr.update(value="")


def run_pipeline(pipeline, save_path, image_paths, wh, caption_json,
                 use_prior, use_fusion, use_guide, flowedit, matches_dir, seed):
    if pipeline is None:
        raise gr.Error("Pipeline not loaded — wait for startup to finish.")
    if not image_paths:
        raise gr.Error("Select or upload images first.")
    if not caption_json or not caption_json.strip():
        raise gr.Error("Set a prompt in Step 2 before running.")
    try:
        caption = json.loads(caption_json)
    except json.JSONDecodeError as e:
        raise gr.Error(f"Invalid caption JSON: {e}")
    if "edit" not in caption:
        raise gr.Error('Caption must have an "edit" key.')

    w, h = wh
    images = [center_crop_and_resize(Image.open(p).convert("RGB"), w, h) for p in image_paths]
    max_adj = 4 if len(image_paths) > 5 else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(save_path) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-compute matches on the fly if needed and not pre-computed
    if (use_fusion or use_guide) and not matches_dir:
        matches_dir = str(out_dir / "matches")
        print(f"Computing matches on the fly → {matches_dir}")
        compute_matches(image_paths, matches_dir, w, h)
        print("Matches computed.")

    matches = None
    if (use_fusion or use_guide) and matches_dir:
        graph, _ = define_graph(len(image_paths))
        mp = Path(matches_dir)
        matches = {}
        for edge in graph:
            for ef, es in [edge, edge[::-1]]:
                m = load_matches(mp, image_paths, ef, es,
                                 grid_matches=use_prior,
                                 grid_horizontal=w <= h, flatten=True)
                matches[(ef, es)] = m.to(pipeline.device)

    pipeline.use_pair_graph = use_prior
    pipeline.mff = use_fusion
    pipeline.feat_guide = use_guide
    pipeline.flowedit = flowedit
    pipeline.matches = matches
    pipeline.max_adjacent_nodes = max_adj

    prompts = image_path_to_prompt(image_paths, (w, h), use_prior, caption, max_adj, flowedit=flowedit)
    src_prompts = image_path_to_prompt(image_paths, (w, h), use_prior, caption, max_adj, mode="src") if flowedit else None
    with open(out_dir / "captions.json", "w") as f:
        json.dump(caption, f, indent=2)
    with open(out_dir / "args.json", "w") as f:
        json.dump({"image_paths": image_paths, "width": w, "height": h,
                   "use_pair_graph": use_prior, "mff": use_fusion, "feat_guide": use_guide,
                   "flowedit": flowedit, "seed": int(seed), "matches_dir": matches_dir}, f, indent=2)

    outputs, controlnet_images = pipeline(
        prompts=prompts,
        src_prompts=src_prompts,
        controlnet_image=images,
        width=w, height=h,
        guidance=FLOWEDIT_GUIDANCE if flowedit else GUIDANCE,
        seed=int(seed),
        true_gs=TRUE_GS,
        neg_prompt="bad photo",
        timestep_to_start_cfg=TIMESTEP_TO_START_CFG,
    )

    for i, (inp, out) in enumerate(zip(images, outputs)):
        inp.save(out_dir / f"input_{i}.png")
        out.save(out_dir / f"output_{i}.png")
        for ctrl_imgs, ctrl_name in zip(controlnet_images, pipeline.control_type):
            ctrl_imgs[i].save(out_dir / f"control_{ctrl_name}_{i}.png")

    vgrid_pil(hgrid_pil(*images), hgrid_pil(*outputs)).save(out_dir / "all_inputs_outputs.png")
    vgrid_pil(*[hgrid_pil(*ct) for ct in controlnet_images]).save(out_dir / "all_controls.png")

    result = hgrid_pil(*outputs)
    torch.cuda.empty_cache()
    return result, str(out_dir), matches_dir


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def create_demo(save_path: str):
    Path(save_path).mkdir(parents=True, exist_ok=True)

    _favicon = Path(__file__).parent / "assets" / "favicon.png"
    _icon_html = ""
    if _favicon.exists():
        _b64 = base64.b64encode(_favicon.read_bytes()).decode()
        _icon_html = f'<img src="data:image/png;base64,{_b64}" style="height:52px;vertical-align:middle;margin-right:12px">'

    with gr.Blocks(title="Match-and-Fuse", theme=gr.themes.Soft()) as demo:

        pipeline_st      = gr.State()
        image_paths_st   = gr.State()
        wh_st            = gr.State()
        caption_json_st  = gr.State(value="")
        matches_dir_st   = gr.State()
        save_path_st     = gr.State(value=save_path)

        gr.HTML(
            f'<div style="display:flex;align-items:center;margin-bottom:4px">'
            f'{_icon_html}'
            f'<div><h1 style="margin:0">Match-and-Fuse</h1>'
            f'<p style="margin:0;color:gray">A zero-shot, training-free controlled set-to-set generation for unstructured image sets</p>'
            f'</div></div>'
        )

        # ── Step 1: Input Images ───────────────────────────────────────────
        gr.Markdown("## Step 1 — Input Images")
        with gr.Tabs():

            with gr.Tab("Dataset Scene"):
                gr.Markdown(
                    "Choose a pre-loaded scene from `data/`. "
                    "Only scenes with images, pre-computed matches, and captions are listed."
                )
                with gr.Row():
                    scene_dd    = gr.Dropdown(choices=SCENE_NAMES, label="Scene", scale=3)
                    max_img_ds  = gr.Slider(2, 10, value=5, step=1, label="Max images", scale=1)
                preview_ds = gr.Image(label="Input images", interactive=False)

                max_img_ds.change(
                    fn=load_dataset_scene,
                    inputs=[scene_dd, max_img_ds],
                    outputs=[preview_ds, image_paths_st, wh_st, matches_dir_st],
                )

            with gr.Tab("Custom Images"):
                gr.Markdown("Upload your own images. Matches are auto-computed on the fly when needed.")
                with gr.Row():
                    upload      = gr.Files(label="Upload images", file_types=["image"], scale=3)
                    max_img_cu  = gr.Slider(2, 10, value=5, step=1, label="Max images", scale=1)
                preview_cu = gr.Image(label="Input images", interactive=False)

                upload.change(
                    fn=load_custom_images,
                    inputs=[upload, max_img_cu],
                    outputs=[preview_cu, image_paths_st, wh_st, matches_dir_st],
                )
                max_img_cu.change(
                    fn=load_custom_images,
                    inputs=[upload, max_img_cu],
                    outputs=[preview_cu, image_paths_st, wh_st, matches_dir_st],
                )

        # ── Step 2: Prompts ────────────────────────────────────────────────
        gr.Markdown("## Step 2 — Prompts")
        with gr.Tabs():

            # ── Preset (dataset scenes only) ──────────────────────────────
            with gr.Tab("Preset"):
                preset_unavailable = gr.Markdown("*Preset unavailable — presets are only available for dataset scenes.*", visible=False)
                preset_info = gr.Markdown(
                    "Load a pre-computed caption for the selected dataset scene. "
                    "Select a scene in Step 1 to populate the dropdown."
                )
                preset_dd = gr.Dropdown(choices=[], label="Prompt variant", value=None)
                preset_accordion = gr.Accordion("Caption JSON", open=True)
                with preset_accordion:
                    preset_preview = gr.Textbox(
                        label="Caption JSON (editable)", lines=10, interactive=True,
                        placeholder="Select a scene and prompt variant above.",
                    )

                # Wire scene dropdown → images/matches + preset choices
                scene_dd.change(
                    fn=load_dataset_scene,
                    inputs=[scene_dd, max_img_ds],
                    outputs=[preview_ds, image_paths_st, wh_st, matches_dir_st],
                )
                scene_dd.change(fn=update_preset_choices, inputs=scene_dd, outputs=preset_dd)

                # Preset variant chosen → populate textbox; textbox edits → update state
                preset_dd.change(
                    fn=load_preset_caption,
                    inputs=[scene_dd, preset_dd],
                    outputs=preset_preview,
                ).then(fn=lambda x: x, inputs=preset_preview, outputs=caption_json_st)
                preset_preview.change(fn=lambda x: x, inputs=preset_preview, outputs=caption_json_st)

                # Show/hide "Preset unavailable" based on image source
                _preset_controls = [preset_info, preset_dd, preset_accordion]
                upload.change(fn=lambda f: [gr.update(visible=bool(f))] + [gr.update(visible=not bool(f))] * 3,
                              inputs=upload, outputs=[preset_unavailable] + _preset_controls)
                scene_dd.change(fn=lambda _: [gr.update(visible=False)] + [gr.update(visible=True)] * 3,
                                inputs=scene_dd, outputs=[preset_unavailable] + _preset_controls)

            # ── Auto-caption ──────────────────────────────────────────────
            with gr.Tab("Auto-captioning"):
                gr.Markdown("Use a vision LLM to generate per-image captions from the loaded images.")
                with gr.Row():
                    obj_prompt = gr.Textbox(label="Edit description", placeholder="e.g. a cyborg", scale=2)
                    bg_prompt  = gr.Textbox(label="Background theme",  placeholder="e.g. cyberpunk", scale=2)
                with gr.Row():
                    api_provider = gr.Radio(
                        list(_API_PROVIDERS.keys()), value="OpenAI",
                        label="API provider", scale=2,
                    )
                    api_model = gr.Textbox(
                        value=_API_PROVIDERS["OpenAI"]["default_model"],
                        label="Model", scale=2,
                    )
                key_status = gr.Markdown(value=key_status_md("OpenAI"))

                def _on_provider_change(provider):
                    return gr.update(value=_API_PROVIDERS[provider]["default_model"]), key_status_md(provider)

                api_provider.change(fn=_on_provider_change, inputs=api_provider, outputs=[api_model, key_status])

                caption_btn = gr.Button("Auto-caption images", variant="secondary")

                with gr.Accordion("Caption JSON", open=True):
                    auto_preview = gr.Textbox(
                        label="Caption JSON (editable)", lines=14, interactive=True,
                    )

                caption_btn.click(
                    fn=run_autocaption,
                    inputs=[image_paths_st, wh_st, obj_prompt, bg_prompt, api_provider, api_model],
                    outputs=auto_preview,
                ).then(fn=lambda x: x, inputs=auto_preview, outputs=caption_json_st)
                auto_preview.change(fn=lambda x: x, inputs=auto_preview, outputs=caption_json_st)

            # ── Manual JSON ───────────────────────────────────────────────
            with gr.Tab("Manual captioning"):
                gr.Markdown(_CAPTION_FORMAT_HINT)
                with gr.Accordion("Caption JSON", open=True):
                    manual_json = gr.Textbox(
                        label="Caption JSON",
                        lines=14,
                        placeholder='{\n  "edit": "a marble statue",\n  "src": "a cat"\n}',
                    )
                manual_json.change(fn=lambda x: x, inputs=manual_json, outputs=caption_json_st)

        # ── Step 3: Advanced Options ───────────────────────────────────────
        gr.Markdown("## Step 3 — Advanced Options")
        with gr.Accordion("Expand to adjust", open=False):
            gr.Markdown(
                "All method components are **on** by default.\n\n"
                "**FlowEdit** edits from the original images instead of noise — requires `src` fields in the caption.\n\n"
                "Feature Guidance can be omitted to trade fine-grained consistency in favor of lower runtime."
            )
            with gr.Row():
                use_prior  = gr.Checkbox(value=True,  label="Pairwise Consistency Graph")
                use_fusion = gr.Checkbox(value=True,  label="Multi-view Feature Fusion")
                use_guide  = gr.Checkbox(value=True,  label="Feature Guidance")
                flowedit   = gr.Checkbox(value=False, label="FlowEdit")
            seed = gr.Number(value=2, label="Seed", precision=0)

        # ── Step 4: Generate ───────────────────────────────────────────────
        gr.Markdown("## Step 4 — Generate")
        pipeline_status = gr.Markdown(
            '<p style="color:orange;font-weight:bold">⏳ Loading model weights — the Run button will enable when ready.</p>'
        )
        run_btn    = gr.Button("Run Match-and-Fuse ✨", variant="primary", size="lg", interactive=False)
        result_img = gr.Image(label="Result", interactive=False)
        save_dir_display = gr.Textbox(label="Saved to", interactive=False, visible=False)

        demo.load(fn=_load_pipeline, outputs=[pipeline_st, run_btn, pipeline_status])

        run_btn.click(
            fn=run_pipeline,
            inputs=[pipeline_st, save_path_st, image_paths_st, wh_st, caption_json_st,
                    use_prior, use_fusion, use_guide, flowedit, matches_dir_st, seed],
            outputs=[result_img, save_dir_display, matches_dir_st],
        ).then(fn=lambda p: gr.update(value=p, visible=True), inputs=save_dir_display, outputs=save_dir_display)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match-and-Fuse Gradio Demo")
    parser.add_argument("--save_path", type=str, default="gradio_demo_out")
    args = parser.parse_args()

    demo = create_demo(args.save_path)
    demo.launch(server_name="0.0.0.0", server_port=5001)
