# Method constants

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

GRID_PREFIX: str = "Stacked image grid with nearby views of"
QUALITY: str = "Best quality, hyperrealistic, extremely detailed."

# ---------------------------------------------------------------------------
# Diffusion  sampling parameters
# ---------------------------------------------------------------------------

# All step-index constants below are calibrated for this many denoising steps.
# If you change NUM_STEPS, revisit every constant that carries a "step" comment.
NUM_STEPS: int = 25

# Classifier-free guidance scale
GUIDANCE: float = 4.0

# True CFG guidance scale and when to start applying it
TRUE_GS: float = 3.5
TIMESTEP_TO_START_CFG: int = 1


# ---------------------------------------------------------------------------
# Multiview Feature Fusion (MFF) schedule
# ---------------------------------------------------------------------------

# MFF is active in double-stream blocks only for the first N denoising steps
# (22 / 25 steps ≈ first 88% of the schedule)
MFF_DOUBLE_BLOCK_STEP_LIMIT: int = 22

# MFF is active in single-stream blocks only for the first N denoising steps
# (10 / 25 steps = first 40% of the schedule)
MFF_SINGLE_BLOCK_STEP_LIMIT: int = 10

# MFF is applied only to the tail of single-stream blocks (those with index >= this).
# 38 total single blocks; 15/16 fraction → last 1/16 (~2 blocks).
MFF_SINGLE_BLOCK_IDX_START: int = 38 * 15 // 16  # = 35

# ---------------------------------------------------------------------------
# ControlNet weight annealing
# ---------------------------------------------------------------------------

# Starting controlnet guidance scale (annealed to CONTROLNET_GS_END over the denoising loop)
CONTROLNET_GS_START: float = 1.0

# ControlNet guidance is linearly annealed to this value over the denoising loop
CONTROLNET_GS_END: float = 0.0

# ---------------------------------------------------------------------------
# Feature guidance optimization
# ---------------------------------------------------------------------------

# Guidance runs only for the first N denoising steps
# (16 / 25 steps = first 64% of the schedule)
GUIDANCE_STEP_LIMIT: int = 16

# Learning rate linearly annealed from start → end over GUIDANCE_STEP_LIMIT steps
GUIDANCE_LR_START: float = 0.016
GUIDANCE_LR_END: float = 0.002

# Adam optimizer steps taken per denoising step
GUIDANCE_OPT_STEPS: int = 1

# Number of graph edges processed per optimizer step (gradient accumulation)
GUIDANCE_ACCUM_BATCH: int = 1

# ---------------------------------------------------------------------------
# FlowEdit mode parameter overrides
# ---------------------------------------------------------------------------

# FlowEdit denoising window (relative to NUM_STEPS = 25):
#   start: skip the first N steps (start from the original image latent, not pure noise)
#   stop:  transition to normal sampling at this step index
FLOWEDIT_START_STEP: int = 1
FLOWEDIT_STOP_STEP: int = 15

# Classifier-free guidance scale for the target branch in FlowEdit
FLOWEDIT_GUIDANCE: float = 5.5

# Guidance scale for the source branch in FlowEdit (kept low — src is near-clean)
FLOWEDIT_SRC_GUIDANCE: float = 1.5

# In FlowEdit mode, MFF is applied much more conservatively
# (5 / 25 steps for double blocks; 0 for single blocks — single-block MFF is off)
FLOWEDIT_MFF_DOUBLE_BLOCK_STEP_LIMIT: int = 5
FLOWEDIT_MFF_SINGLE_BLOCK_STEP_LIMIT: int = 0

# ControlNet guidance starting scale in FlowEdit mode
FLOWEDIT_CONTROLNET_GS_START: float = 0.5