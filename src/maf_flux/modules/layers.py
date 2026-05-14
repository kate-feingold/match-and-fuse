import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from ..math import attention, rope
from ..constants import MFF_DOUBLE_BLOCK_STEP_LIMIT, MFF_SINGLE_BLOCK_STEP_LIMIT


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self):
        pass


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class MFFDoubleStreamBlockProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mff = False
        self.max_adjacent_nodes = None
        self.attn_matches = None
        self.step = -1
        self.block_name = None
        self.img_wh = None
        self.mff_double_step_limit = MFF_DOUBLE_BLOCK_STEP_LIMIT
        self.mff_single_step_limit = MFF_SINGLE_BLOCK_STEP_LIMIT

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1, aux_result = attention(
            q, k, v, pe=pe, text_len=txt.shape[1],
            attn_matches=self.attn_matches, mff=self.mff,
            max_adjacent_nodes=self.max_adjacent_nodes, step=self.step,
            block_name=self.block_name, img_wh_align=self.img_wh,
            mff_double_step_limit=self.mff_double_step_limit,
            mff_single_step_limit=self.mff_single_step_limit)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt, aux_result


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        processor = MFFDoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        return self.processor(self, img, txt, vec, pe)


class MFFSingleStreamBlockProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mff = False
        self.max_adjacent_nodes = None
        self.attn_matches = None
        self.text_len = None
        self.step = -1
        self.block_name = None
        self.img_wh = None
        self.mff_double_step_limit = MFF_DOUBLE_BLOCK_STEP_LIMIT
        self.mff_single_step_limit = MFF_SINGLE_BLOCK_STEP_LIMIT

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor):

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1, aux_result = attention(
            q, k, v, pe=pe, mff=self.mff, text_len=self.text_len,
            max_adjacent_nodes=self.max_adjacent_nodes, attn_matches=self.attn_matches,
            step=self.step, block_name=self.block_name, img_wh_align=self.img_wh,
            mff_double_step_limit=self.mff_double_step_limit,
            mff_single_step_limit=self.mff_single_step_limit)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = x + mod.gate * output
        return output, aux_result


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        processor = MFFSingleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        return self.processor(self, x, vec, pe)


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
