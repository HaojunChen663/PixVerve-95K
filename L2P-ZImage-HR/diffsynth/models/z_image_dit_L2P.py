import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch.nn import RMSNorm
from ..core.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                mid_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                mid_size,
                out_size,
                bias=True,
            ),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(torch.bfloat16))
        return t_emb


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class Attention(torch.nn.Module):

    def __init__(self, q_dim, num_heads, head_dim, kv_dim=None, bias_q=False, bias_kv=False, bias_out=False):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = torch.nn.Linear(q_dim, dim_inner, bias=bias_q)
        self.to_k = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_v = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(dim_inner, q_dim, bias=bias_out)])

        self.norm_q = RMSNorm(head_dim, eps=1e-5)
        self.norm_k = RMSNorm(head_dim, eps=1e-5)
    
    def forward(self, hidden_states, freqs_cis):
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.num_heads, -1))
        key = key.unflatten(-1, (self.num_heads, -1))
        value = value.unflatten(-1, (self.num_heads, -1))

        # Apply Norms
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)  # todo

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # Compute joint attention
        hidden_states = attention_forward(
            query,
            key,
            value,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = self.to_out[0](hidden_states)
        if len(self.to_out) > 1:  # dropout
            output = self.to_out[1](output)

        return output


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        # Refactored to use diffusers Attention with custom processor
        # Original Z-Image params: dim, n_heads, n_kv_heads, qk_norm
        self.attention = Attention(
            q_dim=dim,
            num_heads=n_heads,
            head_dim=dim // n_heads,
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x) * scale_mlp,
                )
            )
        else:
            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x),
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)

            # FFN block
            x = x + self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x),
                )
            )

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)  # complex64
                freqs_cis.append(freqs_cis_i)

            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


class MicroDiffusionModel(nn.Module):
    """
    U-Net local decoder for pixel-space diffusion (resolution-aware).
    Takes full-resolution noisy image and Transformer feature map, produces denoised output.

    A single class supports the 4K / 8K / 10K variants, selected by `patch_size`.
    Every variant is byte-for-byte identical (layer names, shapes, construction
    order and forward computation) to the original standalone code, so existing
    checkpoints load with `strict=True` and produce identical results.

      - patch_size=64  -> 4K  (4096x4096):  feat_map 64x64, 6 pools (enc0..enc4),
                          bottleneck 512+Dim, decoder up4..up0.
      - patch_size=128 -> 8K  (8192x8192):  4K subnet + outer wrapper
                          (enc_8k/pool_8k, up_8k/dec_8k); feat_map stays 64x64.
      - patch_size=160 -> 10K (10240x10240): no enc4 level; bottleneck 256+Dim
                          at 160x160 (feat_map upsampled 64->160); outer wrapper
                          enc_10k_a/up_10k_a/dec_10k_a; per-stage grad-checkpoint.
    """
    # patch_size -> internal resolution variant
    _SUPPORTED_PATCH_SIZES = (64, 128, 160)

    def __init__(self, in_channels, si_t_hidden_size, patch_size=64):
        super().__init__()
        if patch_size not in self._SUPPORTED_PATCH_SIZES:
            raise ValueError(
                f"Unsupported patch_size={patch_size}. "
                f"Expected one of {self._SUPPORTED_PATCH_SIZES} (4K/8K/10K)."
            )
        self.patch_size = patch_size

        if patch_size in (64, 128):
            # ========================================================
            # 4K subnet (shared by 4K & 8K).
            # 8K wraps it with an extra outermost scale (enc_8k / up_8k / dec_8k).
            # ========================================================
            if patch_size == 128:
                # NEW (8K wrapper) encoder: 8192 -> 4096
                # Keep channel count = in_channels so that `enc0` below can still be
                # Conv2d(in_channels, 16) and stay weight-compatible with 4K.
                self.enc_8k = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.SiLU(),
                )
                self.pool_8k = nn.MaxPool2d(2, stride=2)  # 8192 -> 4096

            # U-Net Encoder: 4096 -> 2048 -> 1024 -> 512 -> 256 -> 128 -> 64
            self.enc0 = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.pool0 = nn.MaxPool2d(2, stride=2)

            self.enc0b = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.pool0b = nn.MaxPool2d(2, stride=2)

            self.enc1 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.pool1 = nn.MaxPool2d(2, stride=2)

            self.enc2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.pool2 = nn.MaxPool2d(2, stride=2)

            self.enc3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.pool3 = nn.MaxPool2d(2, stride=2)

            self.enc4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.pool4 = nn.MaxPool2d(2, stride=2)

            # Bottleneck at 64x64: inject Transformer features
            self.bottleneck = nn.Sequential(
                nn.Conv2d(512 + si_t_hidden_size, 512, kernel_size=1),
                nn.SiLU(),
            )

            # U-Net Decoder: 64 -> 128 -> 256 -> 512 -> 1024 -> 2048 -> 4096
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 512, kernel_size=3, padding=1)
            )
            self.dec4 = nn.Sequential(
                nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1),
                nn.SiLU(),
            )

            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(256, 256, kernel_size=3, padding=1)
            )
            self.dec3 = nn.Sequential(
                nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
                nn.SiLU(),
            )

            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
            self.dec2 = nn.Sequential(
                nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
                nn.SiLU(),
            )

            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
            self.dec1 = nn.Sequential(
                nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),
                nn.SiLU(),
            )

            self.up0b = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
            self.dec0b = nn.Sequential(
                nn.Conv2d(32 + 32, 16, kernel_size=3, padding=1),
                nn.SiLU(),
            )

            self.up0 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(16, 16, kernel_size=3, padding=1)
            )
            self.dec0 = nn.Sequential(
                nn.Conv2d(16 + 16, 16, kernel_size=3, padding=1),
                nn.SiLU(),
            )

            if patch_size == 128:
                # NEW (8K wrapper) decoder: 4096 -> 8192
                self.up_8k = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(16, 16, kernel_size=3, padding=1),
                )
                self.dec_8k = nn.Sequential(
                    nn.Conv2d(16 + in_channels, 16, kernel_size=3, padding=1),
                    nn.SiLU(),
                )

            # Output Layer
            self.out_conv = nn.Conv2d(16, in_channels, kernel_size=1)

        else:
            # ========================================================
            # 10K variant (patch_size=160). 10240 is not a power-of-two
            # multiple of 64, so the bottleneck sits at 160x160 and the
            # Transformer feat_map is upsampled 64->160 before injection.
            # enc0..enc3 / dec3..dec0 / up0..up2 / out_conv keep 4K/8K
            # channel layout for weight transfer; enc4/up4/dec4 are absent.
            # ========================================================
            # NEW (10K wrapper) encoder: 10240 -> 5120
            self.enc_10k_a = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.pool_10k_a = nn.MaxPool2d(2, stride=2)  # 10240 -> 5120

            # U-Net Encoder: 5120 -> 2560 -> 1280 -> 640 -> 320 -> 160
            self.enc0 = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.pool0 = nn.MaxPool2d(2, stride=2)

            self.enc0b = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.pool0b = nn.MaxPool2d(2, stride=2)

            self.enc1 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.pool1 = nn.MaxPool2d(2, stride=2)

            self.enc2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.pool2 = nn.MaxPool2d(2, stride=2)

            self.enc3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.pool3 = nn.MaxPool2d(2, stride=2)

            # Bottleneck at 160x160 (10K-specific: 256+Dim instead of 512+Dim)
            self.bottleneck = nn.Sequential(
                nn.Conv2d(256 + si_t_hidden_size, 512, kernel_size=1),
                nn.SiLU(),
            )

            # U-Net Decoder: 160 -> 320 -> 640 -> 1280 -> 2560 -> 5120
            # 10K-specific up3 takes 512 (bottleneck output) instead of 4K's 256.
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 256, kernel_size=3, padding=1)
            )
            self.dec3 = nn.Sequential(
                nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
                nn.SiLU(),
            )

            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
            self.dec2 = nn.Sequential(
                nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
                nn.SiLU(),
            )

            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
            self.dec1 = nn.Sequential(
                nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),
                nn.SiLU(),
            )

            self.up0b = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
            self.dec0b = nn.Sequential(
                nn.Conv2d(32 + 32, 16, kernel_size=3, padding=1),
                nn.SiLU(),
            )

            self.up0 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(16, 16, kernel_size=3, padding=1)
            )
            self.dec0 = nn.Sequential(
                nn.Conv2d(16 + 16, 16, kernel_size=3, padding=1),
                nn.SiLU(),
            )

            # NEW (10K wrapper) decoder: 5120 -> 10240
            self.up_10k_a = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
            )
            self.dec_10k_a = nn.Sequential(
                nn.Conv2d(16 + in_channels, 16, kernel_size=3, padding=1),
                nn.SiLU(),
            )

            # Output Layer
            self.out_conv = nn.Conv2d(16, in_channels, kernel_size=1)

    def forward(self, x, c, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False):
        """
        x: [B, C, R, R] full-resolution noisy image (R = 4096 / 8192 / 10240)
        c: [B, Dim, 64, 64] Transformer feature map
        The gradient-checkpointing flags only take effect for the 10K variant,
        whose shallow activations are large; 4K/8K ignore them (matching the
        original implementations exactly).
        """
        if self.patch_size in (64, 128):
            return self._forward_4k8k(x, c)
        return self._forward_10k(x, c, use_gradient_checkpointing, use_gradient_checkpointing_offload)

    def _forward_4k8k(self, x, c):
        # ---- 8K wrapper (encoder side): 8192 -> 4096 ----
        if self.patch_size == 128:
            enc_8k_out = self.enc_8k(x)
            x = self.pool_8k(enc_8k_out)

        # ---- U-Net Encoder: 4096 -> 64 ----
        enc0_out = self.enc0(x)
        p0_out = self.pool0(enc0_out)

        enc0b_out = self.enc0b(p0_out)
        p0b_out = self.pool0b(enc0b_out)

        enc1_out = self.enc1(p0b_out)
        p1_out = self.pool1(enc1_out)

        enc2_out = self.enc2(p1_out)
        p2_out = self.pool2(enc2_out)

        enc3_out = self.enc3(p2_out)
        p3_out = self.pool3(enc3_out)

        enc4_out = self.enc4(p3_out)
        p4_out = self.pool4(enc4_out)

        # Inject Transformer feature at 64x64 bottleneck
        bottleneck_input = torch.cat([p4_out, c], dim=1)
        bottleneck_out = self.bottleneck(bottleneck_input)

        # ---- U-Net Decoder: 64 -> 4096 ----
        dec4_out = self.up4(bottleneck_out)
        dec4_out = torch.cat([dec4_out, enc4_out], dim=1)
        dec4_out = self.dec4(dec4_out)

        dec3_out = self.up3(dec4_out)
        dec3_out = torch.cat([dec3_out, enc3_out], dim=1)
        dec3_out = self.dec3(dec3_out)

        dec2_out = self.up2(dec3_out)
        dec2_out = torch.cat([dec2_out, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_out)

        dec1_out = self.up1(dec2_out)
        dec1_out = torch.cat([dec1_out, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_out)

        dec0b_out = self.up0b(dec1_out)
        dec0b_out = torch.cat([dec0b_out, enc0b_out], dim=1)
        dec0b_out = self.dec0b(dec0b_out)

        dec0_out = self.up0(dec0b_out)
        dec0_out = torch.cat([dec0_out, enc0_out], dim=1)
        dec0_out = self.dec0(dec0_out)

        # ---- 8K wrapper (decoder side): 4096 -> 8192 ----
        if self.patch_size == 128:
            dec_8k_out = self.up_8k(dec0_out)
            dec_8k_out = torch.cat([dec_8k_out, enc_8k_out], dim=1)
            dec_8k_out = self.dec_8k(dec_8k_out)
            x_out = self.out_conv(dec_8k_out)
        else:
            x_out = self.out_conv(dec0_out)
        return x_out

    def _forward_10k(self, x, c, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False):
        # Each resolution stage is wrapped by gradient_checkpoint_forward so that
        # large shallow activations are recomputed on backward instead of stored.

        def _enc_stage_10k_a(x_in):
            enc_10k_a_out = self.enc_10k_a(x_in)
            p_10k_a_out = self.pool_10k_a(enc_10k_a_out)
            return enc_10k_a_out, p_10k_a_out

        def _enc_stage_0(p_in):
            enc0_out = self.enc0(p_in)
            p0_out = self.pool0(enc0_out)
            return enc0_out, p0_out

        def _enc_stage_0b(p_in):
            enc0b_out = self.enc0b(p_in)
            p0b_out = self.pool0b(enc0b_out)
            return enc0b_out, p0b_out

        def _enc_stage_1(p_in):
            enc1_out = self.enc1(p_in)
            p1_out = self.pool1(enc1_out)
            return enc1_out, p1_out

        def _enc_stage_2(p_in):
            enc2_out = self.enc2(p_in)
            p2_out = self.pool2(enc2_out)
            return enc2_out, p2_out

        def _enc_stage_3_and_bottleneck(p_in, c_in):
            enc3_out = self.enc3(p_in)
            p3_out = self.pool3(enc3_out)
            c_upsampled = F.interpolate(c_in, size=(160, 160), mode='nearest')
            bottleneck_input = torch.cat([p3_out, c_upsampled], dim=1)
            bottleneck_out = self.bottleneck(bottleneck_input)
            return enc3_out, bottleneck_out

        def _dec_stage_3(bottleneck_in, enc3_in):
            dec3_out = self.up3(bottleneck_in)
            dec3_out = torch.cat([dec3_out, enc3_in], dim=1)
            dec3_out = self.dec3(dec3_out)
            return dec3_out

        def _dec_stage_2(dec_in, enc2_in):
            dec2_out = self.up2(dec_in)
            dec2_out = torch.cat([dec2_out, enc2_in], dim=1)
            dec2_out = self.dec2(dec2_out)
            return dec2_out

        def _dec_stage_1(dec_in, enc1_in):
            dec1_out = self.up1(dec_in)
            dec1_out = torch.cat([dec1_out, enc1_in], dim=1)
            dec1_out = self.dec1(dec1_out)
            return dec1_out

        def _dec_stage_0b(dec_in, enc0b_in):
            dec0b_out = self.up0b(dec_in)
            dec0b_out = torch.cat([dec0b_out, enc0b_in], dim=1)
            dec0b_out = self.dec0b(dec0b_out)
            return dec0b_out

        def _dec_stage_0(dec_in, enc0_in):
            dec0_out = self.up0(dec_in)
            dec0_out = torch.cat([dec0_out, enc0_in], dim=1)
            dec0_out = self.dec0(dec0_out)
            return dec0_out

        def _dec_stage_10k_a_and_out(dec_in, enc_10k_a_in):
            dec_10k_a_out = self.up_10k_a(dec_in)
            dec_10k_a_out = torch.cat([dec_10k_a_out, enc_10k_a_in], dim=1)
            dec_10k_a_out = self.dec_10k_a(dec_10k_a_out)
            x_out = self.out_conv(dec_10k_a_out)
            return x_out

        _gc = lambda fn, *a: gradient_checkpoint_forward(
            fn, use_gradient_checkpointing, use_gradient_checkpointing_offload, *a,
        )

        # Encoder
        enc_10k_a_out, p_10k_a_out = _gc(_enc_stage_10k_a, x)
        enc0_out,      p0_out      = _gc(_enc_stage_0,     p_10k_a_out)
        enc0b_out,     p0b_out     = _gc(_enc_stage_0b,    p0_out)
        enc1_out,      p1_out      = _gc(_enc_stage_1,     p0b_out)
        enc2_out,      p2_out      = _gc(_enc_stage_2,     p1_out)
        enc3_out, bottleneck_out   = _gc(_enc_stage_3_and_bottleneck, p2_out, c)

        # Decoder
        dec3_out  = _gc(_dec_stage_3,  bottleneck_out, enc3_out)
        dec2_out  = _gc(_dec_stage_2,  dec3_out,       enc2_out)
        dec1_out  = _gc(_dec_stage_1,  dec2_out,       enc1_out)
        dec0b_out = _gc(_dec_stage_0b, dec1_out,       enc0b_out)
        dec0_out  = _gc(_dec_stage_0,  dec0b_out,      enc0_out)
        x_out     = _gc(_dec_stage_10k_a_and_out, dec0_out, enc_10k_a_out)

        return x_out

class ZImageDiT(nn.Module):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]

    def __init__(
        self,
        all_patch_size=(64,),
        all_f_patch_size=(1,),
        in_channels=3,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        # all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            # final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            # all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        # self.all_final_layer = nn.ModuleDict(all_final_layer)

        # The U-Net local decoder architecture is selected by the (single) patch
        # size: 64 -> 4K, 128 -> 8K, 160 -> 10K. This keeps the checkpoint of each
        # resolution byte-for-byte compatible with its original standalone code.
        self.local_decoder = MicroDiffusionModel(
            in_channels=in_channels,
            si_t_hidden_size=dim,
            patch_size=all_patch_size[0],
        )


        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            ### Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            all_cap_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            cap_padded_feat = torch.cat(
                [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
                dim=0,
            )
            all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=16,
        f_patch_size=1,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        adaln_input = t

        (
            x_patches_flat_list,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x_patches_flat_list]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x_embed = torch.cat(x_patches_flat_list, dim=0)
        x_embed = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_embed)
        x_embed[torch.cat(x_inner_pad_mask)] = self.x_pad_token.to(dtype=x_embed.dtype, device=x_embed.device)
        x_embed = list(x_embed.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        x_embed = pad_sequence(x_embed, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        for layer in self.noise_refiner:
            x_embed = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=x_embed,
                attn_mask=x_attn_mask,
                freqs_cis=x_freqs_cis,
                adaln_input=adaln_input,
            )

        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token.to(dtype=x_embed.dtype, device=x_embed.device)
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        for layer in self.context_refiner:
            cap_feats = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=cap_feats,
                attn_mask=cap_attn_mask,
                freqs_cis=cap_freqs_cis,
            )

        # unified
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x_embed[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        for layer in self.layers:
            unified = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=unified,
                attn_mask=unified_attn_mask,
                freqs_cis=unified_freqs_cis,
                adaln_input=adaln_input,
            )


        # ============================================================
        # 6. Pixel-space decoding using MicroDiffusionModel
        # ============================================================

        # 1. Prepare the feature map (condition).
        #    unified: [B, Max_Seq_Len, Dim]; image tokens (possibly padded)
        #    come first, caption tokens afterwards.
        #    Compute the feature-map spatial size (original token count, no padding).
        F_ori, H_ori, W_ori = x_size[0] 
        feat_H = H_ori // patch_size
        feat_W = W_ori // patch_size
        img_ori_token_len = feat_H * feat_W  # number of real image tokens (excluding SEQ_MULTI_OF padding)

        # Take the image tokens only (first img_ori_token_len, skip padding tokens)
        img_features = unified[:, :img_ori_token_len, :] 

        # Reshape: [B, H*W, Dim] -> [B, Dim, H, W]
        feat_map = img_features.view(bsz, feat_H, feat_W, self.dim).permute(0, 3, 1, 2)
        
        # 2. Prepare the noisy image (input).
        #    x is a list of [C, F, H, W] (F=1); stack into [B, C, F, H, W].
        noisy_images = torch.stack(x, dim=0)
        
        # Drop the F dimension (for a video DiT with F>1 this would become (B*F, C, H, W))
        if noisy_images.dim() == 5:
            noisy_images = noisy_images.squeeze(2) # [B, C, H, W]
            
        # 3. Batch decoding (single pass through the U-Net).
        #    Input: [B, C, R, R] (R = 4096/8192/10240)
        #    Cond:  [B, Dim, 64, 64] (for 10K this is up-sampled to 160x160 before injection)
        #    The gradient-checkpointing flag only affects the shallow U-Net layers
        #    for 10K; 4K/8K ignore it internally, so the results stay numerically
        #    identical to the three original implementations.
        decoded_batch = self.local_decoder(
            noisy_images,
            feat_map,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        
        # 4. Restore dimensions and format.
        #    The last step of unpatchify is .reshape(self.out_channels, F, H, W),
        #    so add the F=1 dimension back to get [B, C, 1, H, W].
        decoded_batch = decoded_batch.unsqueeze(2) 

        # 5. Split into a list: [[C, 1, H, W], [C, 1, H, W], ...]
        x_final = list(decoded_batch.unbind(0))

        return x_final, {}
