"""
Auto-captioning for multi-view editing via vision LLMs.

EXPERIMENTAL: only tested with OpenAI gpt-4o (now deprecated). Output quality
and JSON format compliance with other models/providers has not been validated.
Contributions and bug reports welcome.
"""

import base64
import json
import os
import tempfile
from mimetypes import guess_type
from pathlib import Path

from PIL import Image


# NOTE: This prompt was written and tested with gpt-4o (deprecated).
# Prompting behavior and output reliability may differ with other models.
_SYSTEM_PROMPT = """
You will receive:
    * A set of images of the source scene with a shared object;
    * A short Object Edit description (e.g., "a cyborg");
    * A short Background Edit description (e.g., "autumn", "cyberpunk");
Your task is to provide:
    * Source Object Prompt;
    * Edited Object Prompt;
    * Per Image Source Pose and Background Prompts;
    * Per Image Edit Pose and Background Prompts;
To create the prompts, follow the rules below.

**In both Source Object Prompt and Edited Object Prompt:**
Focus on the appearance description (colors of elements, shape, texture, details and their position on the object), include as many visual details that fit into the prompt length.
Make sure not to include pose or background descriptions.

**Source Object Prompt:**
Provide a detailed description of the shared source object without considering the edit.

**Edited Object Prompt:**
Provide a detailed description of the shared edited object that maintains structure and shape but changes the appearance according to the Object Edit description.
Be creative: specify new elements, their colors, textures, and other visual details that fit the requested edit.
Do not reference the original appearance, make the description stand-alone.

**Per Image Source Pose and Background Prompts (rules that apply to each image):**
Refer to the common visual element by its specific type (e.g. "the object", "the person", "etc") and use that exact name consistently across all per-image prompts.
Provide a detailed description of the pose of the object (if it's located at the side, rotated, close-up, etc.), especially pose details that are not standard or different than in other images.
Provide a detailed description of the background: list all the surrounding objects and their locations, their appearance, and the environment.

**Per Image Edit Pose and Background Prompts (rules that apply to each image):**
Provide prompts with a similar sentence pattern to the corresponding Per Image Source Pose and Background Prompts, but focus on editing the background according to the Background Edit description.
Don't change the types of any objects and don't add any new objects, just change the appearance description of the existing ones according to the style/setting, make sure to mention the style name at least once.
Don't say "the scene transformed into", instead, make the prompt standalone and just describe the new appearance of the background.
Keep the same pose description and the same specific name for the common visual element as in the corresponding Per Image Source Pose and Background Prompts.

**Rules to follow strictly for all prompts:**
    * Each prompt must have between 20-50 words, not more than that!
    * Don't start with - "The scene now.." or "Added to the scene" "Scene has transformed", each prompt must be standalone and understandable to anyone who doesn't have access to other prompts.
    * Don't include any emotion-related details, things that cannot be visualized, and meaningless phrases like:
      "cozy charm", "playful look", "evoking a sense of ...", "emphasizing the setting typical of ...", "sitting *dramatically*", "accentuating its ... look", "warm setting", etc (don't use any of such phrases!);
    * All listed rules are equally important and must be followed strictly.

**Output format** - a dictionary with keys:
    * "source_object_prompt" (str, Source Object Caption),
    * "edited_object_prompt" (str, Edited Object Caption),
    * "per_image_pose_background" (list[str], Per Image Source Pose and Background Prompts, the number of items corresponds to the number of provided images).
    * "per_image_edit_pose_background" (list[str], Per Image Edit Pose and Background Prompts, the number of items corresponds to the number of provided images).
"""

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _image_to_base64(image_path: str, width_height: tuple = None) -> tuple[str, str]:
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"

    if width_height is not None:
        with Image.open(image_path) as img:
            img = img.resize(width_height)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp, format="PNG")
            tmp.flush()
            tmp_path = tmp.name
            tmp.close()
        with open(tmp_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        os.remove(tmp_path)
        return b64, "image/png"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime_type


def _parse_json(text: str) -> dict:
    """Parse JSON from model response, stripping markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


def _call_openai_sdk(image_paths, img_wh, prompt_shared, prompt_theme, model, api_key, base_url=None):
    from openai import OpenAI

    client = OpenAI(api_key=api_key, **{"base_url": base_url} if base_url else {})

    user_content = []
    for path in image_paths:
        b64, mime = _image_to_base64(path, img_wh)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
        })
    if prompt_shared:
        user_content.append({"type": "text", "text": f"Object Edit: {prompt_shared}"})
    user_content.append({"type": "text", "text": f"Background Edit: {prompt_theme}"})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    return _parse_json(response.choices[0].message.content)


def _call_anthropic(image_paths, img_wh, prompt_shared, prompt_theme, model):
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable required for Claude models")
    client = anthropic.Anthropic(api_key=api_key)

    user_content = []
    for path in image_paths:
        b64, mime = _image_to_base64(path, img_wh)
        user_content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64},
        })
    if prompt_shared:
        user_content.append({"type": "text", "text": f"Object Edit: {prompt_shared}"})
    user_content.append({"type": "text", "text": f"Background Edit: {prompt_theme}"})

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return _parse_json(response.content[0].text)


def caption_images(
    image_paths: list[str],
    img_wh: tuple[int, int],
    prompt_shared: str,
    prompt_theme: str,
    model: str = "gpt-4o",
) -> dict:
    """
    Caption images using a vision LLM. Returns a dict with keys:
        source_object_prompt, edited_object_prompt,
        per_image_pose_background, per_image_edit_pose_background.

    EXPERIMENTAL: only tested with OpenAI gpt-4o. Other providers/models
    may produce unexpected output format or quality. Contributions welcome.

    Model routing and required env vars:
        - OpenAI  (gpt-*, o1-*, ...):  OPENAI_API_KEY
        - Anthropic (claude-*):         ANTHROPIC_API_KEY
        - Gemini  (gemini-*):           GOOGLE_API_KEY
    """
    print(f"Captioning {len(image_paths)} images with {model}...")

    if model.startswith("claude-"):
        result = _call_anthropic(image_paths, img_wh, prompt_shared, prompt_theme, model)
    elif model.startswith("gemini-"):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable required for Gemini models")
        result = _call_openai_sdk(image_paths, img_wh, prompt_shared, prompt_theme, model,
                                  api_key=api_key, base_url=_GEMINI_BASE_URL)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required for OpenAI models")
        result = _call_openai_sdk(image_paths, img_wh, prompt_shared, prompt_theme, model, api_key=api_key)

    required = {"source_object_prompt", "edited_object_prompt",
                "per_image_pose_background", "per_image_edit_pose_background"}
    if missing := required - result.keys():
        raise ValueError(f"Captioning response missing expected keys: {missing}")

    print(f"Captioning complete.")
    return result


def caption_data(
    image_paths: list[str],
    img_wh: tuple[int, int],
    prompt_shared: str,
    prompt_theme: str,
    model: str = "gpt-4o",
) -> dict:
    """Call caption_images and reformat into the caption dict used by main/gradio."""
    image_names = [Path(p).stem for p in image_paths]
    response = caption_images(image_paths, img_wh, prompt_shared, prompt_theme, model)
    return {
        "src": response["source_object_prompt"],
        "edit": response["edited_object_prompt"],
        "per_image_non_shared_src": dict(zip(image_names, response["per_image_pose_background"])),
        "per_image_non_shared_edit": dict(zip(image_names, response["per_image_edit_pose_background"])),
        "prompt_shared": prompt_shared,
        "prompt_theme": prompt_theme,
    }
