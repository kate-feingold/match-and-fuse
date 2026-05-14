from PIL import Image
import numpy as np
import torch
from torch import Tensor

from einops import rearrange

from src.maf_flux.constants import NUM_STEPS, FLOWEDIT_START_STEP, FLOWEDIT_STOP_STEP
from src.maf_flux.controlnet import MultiControlNetFlux
from src.maf_flux.sampling import denoise_controlnet_grid, denoise_controlnet_no_grid, get_noise, get_schedule, prepare, unpack
from src.maf_flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    load_controlnet,
    load_flow_model_quintized,
    Annotator,
    load_checkpoint
)



class MatchAndFusePipeline:
    def __init__(self, model_type, device, offload: bool = False,
                 mff: bool = False, matches: Tensor = None,
                 use_pair_graph: bool = False, feat_guide: bool = False, max_adjacent_nodes: int | None = None,
                 flowedit: bool = False, flowedit_start_step: int = FLOWEDIT_START_STEP, flowedit_stop_step: int = FLOWEDIT_STOP_STEP):
        self.device = torch.device(device)
        self.offload = offload
        self.model_type = model_type

        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device)
        if "fp8" in model_type:
            self.model = load_flow_model_quintized(model_type, device="cpu" if offload else self.device)
        else:
            self.model = load_flow_model(model_type, device="cpu" if offload else self.device)

        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False

        self.mff = mff
        self.matches = matches
        self.use_pair_graph = use_pair_graph
        self.feat_guide = feat_guide
        self.max_adjacent_nodes = max_adjacent_nodes
        self.flowedit = flowedit
        self.flowedit_start_step = flowedit_start_step
        self.flowedit_stop_step = flowedit_stop_step

    def set_controlnet(self, control_types: list[str], local_path: str = None, repo_ids: list[str] = None, names: list[str] = None, control_weights: list[float] = None):
        self.model.to(self.device)

        controlnets, annotators = [], []
        for control_type, repo_id, name in zip(control_types, repo_ids, names):
            controlnet = load_controlnet(self.model_type, self.device).to(torch.bfloat16)
            checkpoint = load_checkpoint(local_path, repo_id, name)
            controlnet.load_state_dict(checkpoint, strict=False)
            annotator = Annotator(control_type, self.device, joint_min_max=False)
            controlnets.append(controlnet)
            annotators.append(annotator)

        if control_weights is None:
            control_weights = [1.0 / len(control_types)] * len(control_types)
        self.controlnet = MultiControlNetFlux(controlnets, weights=control_weights)
        self.annotators = annotators
        self.controlnet_loaded = True
        self.control_type = control_types

    def __call__(self,
                 prompts: list[str],
                 src_prompts: list[str] = None,
                 controlnet_image: list[Image.Image] = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = NUM_STEPS,
                 seed: int = 2,
                 true_gs: float = 3,
                 neg_prompt: str = '',
                 timestep_to_start_cfg: int = 0,
                 ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)

        controlnet_images = [controlnet_image] if isinstance(controlnet_image, Image.Image) else controlnet_image
        num_samples = 1
        controlnet_images_np_list, controlnet_image_list = [], []
        if self.controlnet_loaded:
            num_samples = len(controlnet_images)
            for annotator in self.annotators:
                controlnet_images_np = annotator(controlnet_images, width, height)
                controlnet_image = torch.cat([
                    torch.from_numpy((np.array(im) / 127.5) - 1)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(torch.bfloat16)
                    .to(self.device)
                    for im in controlnet_images_np
                ])
                controlnet_image_list.append(controlnet_image)
                controlnet_images_np = [Image.fromarray(im) for im in controlnet_images_np]
                controlnet_images_np_list.append(controlnet_images_np)

        return self.forward(
            prompts,
            width,
            height,
            guidance,
            num_steps,
            seed,
            controlnet_image_list,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            neg_prompt=neg_prompt,
            num_samples=num_samples,
            original_pil_images=controlnet_images if self.flowedit else None,
            src_prompts=src_prompts,
        ), controlnet_images_np_list

    def forward(
        self,
        prompts: list[str],
        width: int,
        height: int,
        guidance: float,
        num_steps: int,
        seed: int,
        controlnet_image=None,
        timestep_to_start_cfg: int = 0,
        true_gs: float = 3.5,
        neg_prompt: str = "",
        num_samples: int = 1,
        original_pil_images=None,
        src_prompts: list[str] = None,
    ):
        if self.use_pair_graph:
            # unique noise for each image
            x = get_noise(
                num_samples, height, width, device=self.device,
                dtype=torch.bfloat16, seed=seed
            )
        else:
            # same seed for all images, more fair
            x = get_noise(
                1, height, width, device=self.device,
                dtype=torch.bfloat16, seed=seed
            )
            x = torch.concatenate([x] * num_samples, dim=0)

        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )

        original_latents = None
        if self.flowedit and original_pil_images is not None:
            with torch.no_grad():
                if self.offload:
                    self.ae.encoder.to(self.device)
                imgs_tensor = torch.cat([
                    (torch.from_numpy(np.array(im)).float() / 127.5 - 1) .permute(2, 0, 1).unsqueeze(0).to(self.device)
                    for im in original_pil_images
                ])
                original_latents = self.ae.encode(imgs_tensor).to(torch.bfloat16)
                if self.offload:
                    self.ae.encoder.cpu()

        torch.manual_seed(seed)

        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompts)
            neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)
            orig_inp_cond = prepare(t5=self.t5, clip=self.clip, img=original_latents, prompt=src_prompts) if src_prompts is not None else None

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            denoise_controlnet = denoise_controlnet_grid if self.use_pair_graph else denoise_controlnet_no_grid
            x = denoise_controlnet(
                self.model,
                **inp_cond,
                controlnet=self.controlnet,
                timesteps=timesteps,
                guidance=guidance,
                controlnet_cond=controlnet_image,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'],
                neg_txt_ids=neg_inp_cond['txt_ids'],
                neg_vec=neg_inp_cond['vec'],
                true_gs=true_gs,
                mff=self.mff,
                matches=self.matches,
                img_wh=(width // 16, height // 16),
                feat_guide=self.feat_guide,
                original_img=orig_inp_cond['img'] if orig_inp_cond is not None else None,
                original_txt=orig_inp_cond['txt'] if orig_inp_cond is not None else None,
                original_txt_ids=orig_inp_cond['txt_ids'] if orig_inp_cond is not None else None,
                original_vec=orig_inp_cond['vec'] if orig_inp_cond is not None else None,
                flowedit=self.flowedit,
                flowedit_start_step=self.flowedit_start_step,
                flowedit_stop_step=self.flowedit_stop_step,
                **({"max_adjacent_nodes": self.max_adjacent_nodes} if self.use_pair_graph else {})
            )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.float(), height, width)
            bs = 4
            x = torch.concatenate([self.ae.decode(x[i:i + bs]) for i in range(0, x.shape[0], bs)], dim=0)
            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        output_imgs = []
        for x1_ in x1:
            x1_ = rearrange(x1_, "c h w -> h w c")
            output_img = Image.fromarray((127.5 * (x1_ + 1.0)).cpu().byte().numpy())
            output_imgs.append(output_img)
        return output_imgs

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()


