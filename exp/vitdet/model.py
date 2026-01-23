"""
1. Backbone: Plain ViT + Window Attention + Global Propagation Block
2. Neck: Simple Feature Pyramid
3. Head: FCN -> Segmentation
"""
import math
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


# ------------------------------
# 1. Backbone - Plain ViT (Patch Embedding + Transformer Blocks)
# ------------------------------

def get_1d_sincos_pos_embed(embed_dim: int, t_len: int) -> torch.Tensor:
    """Standard 1D sine-cos positional embedding. Returns (1, T, C)."""
    assert embed_dim % 2 == 0, "embed_dim must be even for sin/cos."
    position = torch.arange(t_len).float()
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(t_len, embed_dim)
    pe[:, 0::2] = torch.sin(position[:, None] * div_term[None, :])
    pe[:, 1::2] = torch.cos(position[:, None] * div_term[None, :])
    return pe.unsqueeze(0)

class PatchEmbedding(nn.Module):
    def __init__(
            self,
            data_size: Tuple[int, int],
            patch_size: Tuple[int, int],
            embed_dim: int,
            channels: int = 1):
        super().__init__()

        self.patch_size = patch_size
        self.grid_size = (
            data_size[0] // patch_size[0],
            data_size[1] // patch_size[1]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)  # [128, 256, 1, 300] -> [128, 300, 256]
        return x


# Blocks for Transformer
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        drop_path: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class FeedForward(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 dropout: float,
                 out_dim: int = None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)  # q, k, v 동시에 생성 -> self-attention 전용 구조
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # attn = softmax 후의 가중치
        attn = self.attn_drop(attn)
        self.attn = attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


# ViT (Transformer encoder)
class VisionTransformer(nn.Module):
    def __init__(
        self,
        data_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        n_layers: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_cls: int,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        channels: int = 1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(data_size, patch_size, d_model, channels)
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # torch.Size([1, 1, 128])
        x = torch.cat((cls_tokens, x), dim=1)  # torch.Size([1024, 151, 128])

        x_len = x.shape[1]
        pe = get_1d_sincos_pos_embed(self.d_model, x_len).to(x.device)
        x = x + pe
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_features:
            return x

        x = x[:, 0]
        x = self.head(x)
        return x


# ------------------------------
# 2. Neck - Simple Feature Pyramid
# ------------------------------

class SimpleFeaturePyramid(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            scale_factors=(4.0, 2.0, 1.0, 0.5)
    ):
        super().__init__()
        self.stages = nn.ModuleList()

        for scale in scale_factors:
            stages = []

            # Step 1: Resizing
            if scale == 0.5:  # 1/2 Downsample
                stages.append(nn.MaxPool1d(kernel_size=2, stride=2))

            elif scale == 1.0:  # Keep
                pass

            elif scale == 2.0:  # x2 Upsample
                stages.append(
                    nn.ConvTranspose1d(in_dim, in_dim, kernel_size=2, stride=2)
                )

            elif scale == 4.0:  # x4 Upsample
                stages.extend([
                    nn.ConvTranspose1d(in_dim, in_dim, kernel_size=2, stride=2),
                    nn.GroupNorm(1, in_dim),
                    nn.GELU(),
                    nn.ConvTranspose1d(in_dim, in_dim, kernel_size=2, stride=2)
                ])

            # Step 2: Projection & Mixing
            stages.extend([
                # 1x1 Conv
                nn.Conv1d(in_dim, out_dim, kernel_size=1),
                nn.GroupNorm(1, out_dim),

                # 3x3 Conv
                nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.GroupNorm(1, out_dim)
            ])

            self.stages.append(nn.Sequential(*stages))

    def forward(self, x):
        results = []
        for stage in self.stages:
            results.append(stage(x))

        return results



# ------------------------------
# 3. Head - FCN for Segmentation
# ------------------------------

class FCNHead(nn.Module):
    def __init__(
            self,
            in_channels: int,
            classes: int
    ):
        super().__init__()

        # 1. Fusion Layer
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(in_channels * 4, in_channels, kernel_size=1),
            nn.GroupNorm(1, in_channels),
            nn.ReLU(inplace=True)
        )

        # 2. Classification Layer
        self.cls_conv = nn.Conv1d(in_channels, classes, kernel_size=1)

    def forward(self, features: List):
        p2, p3, p4, p5 = features

        target_len = p2.shape[-1]

        # Step 1: Upsampling
        p3_up = F.interpolate(p3, size=target_len, mode='linear', align_corners=False)
        p4_up = F.interpolate(p4, size=target_len, mode='linear', align_corners=False)
        p5_up = F.interpolate(p5, size=target_len, mode='linear', align_corners=False)

        # Step 2: Concatenation
        x = torch.cat([p2, p3_up, p4_up, p5_up], dim=1)

        # Step 3: Prediction
        x = self.fusion_conv(x)
        logits = self.cls_conv(x)

        return logits


# ------------------------------
# ViTDet model
# ------------------------------

class ViTDet(nn.Module):
    def __init__(
        self,
        n_cls: int,
        data_size: Tuple[int, int] = (1, 3000),
        patch_size: Tuple[int, int] = (1, 10),
        channels: int = 2,
        enc_d_model: int = 768,
        enc_d_ff: int = 3072,
        enc_n_heads: int = 12,
        enc_n_layers: int = 12,
        enc_dropout: float = 0.1,
        enc_drop_path_rate: float = 0.0,
        dec_d_model: int = 256,
    ):

        super().__init__()
        self.n_cls = n_cls
        self.enc_d_model = enc_d_model
        self.data_size = data_size
        self.patch_size = patch_size
        self.norm = nn.BatchNorm1d(channels, affine=True)

        # 1. Backbone (Plain ViT)
        self.encoder = VisionTransformer(
            data_size=data_size,
            patch_size=patch_size,
            n_layers=enc_n_layers,
            d_model=enc_d_model,
            d_ff=enc_d_ff,
            n_heads=enc_n_heads,
            n_cls=n_cls,
            dropout=enc_dropout,
            drop_path_rate=enc_drop_path_rate,
            channels=channels,
        )

        # 2. Neck (Simple Feature Pyramid)
        self.neck = SimpleFeaturePyramid(in_dim=enc_d_model, out_dim=dec_d_model)

        # 3. Head (FCN for Segmentation)
        self.head = FCNHead(in_channels=dec_d_model, classes=n_cls)

    def forward(self, x):
        x0 = x.shape[-1]
        x = self.norm(x)
        x = torch.unsqueeze(x, 2)  # (2, 3, 1, 3000)

        # 1. Encoder (Backbone)
        features = self.encoder(x, return_features=True)
        features = features[:, 1:]

        # 2. Neck
        features = features.transpose(1, 2)  # (B, L, C) -> (B, C, L)
        pyramid_features = self.neck(features)  # Input: Last feature map, Output: [p2, p3, p4, p5]

        # 3. Head
        logits = self.head(pyramid_features)  # (B, n_cls, L/4)

        # 4. Upsample to original length
        output = F.interpolate(logits, size=x0, mode='linear', align_corners=False)

        return output


if __name__ == "__main__":
    model = ViTDet(
        n_cls=5,
        channels=1,
        data_size=(1, 3000),
        patch_size=(1, 20),
        enc_d_model=768,
        enc_n_layers=12
    )

    x = torch.randn(2, 1, 3000)
    y = model(x)
    print(f"Output: {y.shape}")
