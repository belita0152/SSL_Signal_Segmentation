import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange

# -----------------------------------------------------------------------------
# ViT(1D): Patch Embedding + Transformer Blocks
#         return (train_dataset, eval_dataset), (channel_num, class_num)
# -----------------------------------------------------------------------------

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


class PatchEmbed1D(nn.Module):
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
        self.attn = Attention(dim, heads, dropout)
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


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
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
        attn = attn.softmax(dim=-1)
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
        self.patch_embed = PatchEmbed1D(data_size, patch_size, d_model, channels)
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, d_model)
        )  # learnable position embeddings

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

        trunc_normal_(self.pos_embed, std=0.02)
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


# Mask Transformer (Transformer decoder)
class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls: int,
        patch_size: Tuple[int, int],
        d_encoder: int,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_ff: int,
        drop_path_rate: float,
        dropout: float,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x, data_size):
        H, W = data_size
        GS_H = H // self.patch_size[0]
        GS_W = W // self.patch_size[1]

        x = self.proj_dec(x)  # (B, N, enc_dim) -> (B, N, dec_dim)

        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)  # (1, n_cls, model_dim) -> (B, n_cls, model_dim)
        x = torch.cat((x, cls_emb), 1)  # add cls tokens -> (B, N + n_cls, model_dim)

        for blk in self.blocks:
            x = blk(x)  # transformer block 통과. patch <-> token 상호작용 학습
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]  # patches (B, N, model_dim) // cls_seg_feat (B, n_cls, model_dim)

        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes
        patches = F.normalize(patches, dim=-1, p=2)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=-1, p=2)

        masks = patches @ cls_seg_feat.transpose(1, 2)  # scalar product (cos sim)
        masks = self.mask_norm(masks)  # torch.Size([1, 1200, 6])

        masks = rearrange(masks, "b (h w) n -> b n h w", h=GS_H, w=GS_W)
        return masks


# Segmenter
class Segmenter(nn.Module):
    def __init__(
        self,
        n_cls: int,
        data_size: Tuple[int, int] = (1, 3000),
        patch_size: Tuple[int, int] = (1, 10),
        channels: int = 3,
        enc_d_model: int = 128,
        enc_d_ff: int = 64,
        enc_n_heads: int = 8,
        enc_n_layers: int = 1,
        enc_dropout: float = 0.1,
        enc_drop_path_rate: float = 0.0,
        dec_d_model: int = 128,
        dec_d_ff: int = 64,
        dec_n_heads: int = 8,
        dec_n_layers: int = 1,
        dec_dropout: float = 0.1,
        dec_drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.data_size = data_size
        self.patch_size = patch_size
        self.norm = nn.BatchNorm1d(channels, affine=True)

        self.encoder = VisionTransformer(  # heartbeat -> 625 / ahi -> 3000
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

        self.decoder = MaskTransformer(
            n_cls=n_cls,
            patch_size=patch_size,
            d_encoder=enc_d_model,
            n_layers=dec_n_layers,
            n_heads=dec_n_heads,
            d_model=dec_d_model,
            d_ff=dec_d_ff,
            drop_path_rate=dec_drop_path_rate,
            dropout=dec_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape  # [176, 4, 3000]
        x = self.norm(x)
        x = torch.unsqueeze(x, 2)  # (176, 4, 1, 3000)

        x = self.encoder(x, return_features=True)
        x = x[:, 1:]  # remove CLS tokens for decoding

        masks = self.decoder(x, (1, T))
        masks = F.interpolate(
            masks, size=(1, T), mode="bilinear", align_corners=False
        )

        return masks.squeeze(2)


if __name__ == "__main__":
    model = Segmenter(
        n_cls=3,
        data_size=(1, 3000),
        patch_size=(1, 20),
        channels=3,
        enc_d_model=128,
        enc_d_ff=64,
        enc_n_heads=4,
        enc_n_layers=1,
        dec_d_model=256,
        dec_d_ff=128,
        dec_n_heads=8,
        dec_n_layers=2,
    )

    dummy_input = torch.randn(2, 3, 3000)

    output = model(dummy_input)
    print(dummy_input.shape)
    print(output.shape)
