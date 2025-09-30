import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange

# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, data_size: Tuple[int, int], patch_size: Tuple[int, int], embed_dim, channels=2):
        super().__init__()

        self.data_size = data_size
        self.grid_size = data_size[0] // patch_size[0], data_size[1] // patch_size[1]
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # [128, 256, 1, 300] -> [128, 300, 256]
        return x


# Blocks for Transformer
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
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
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
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

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


# ViT (Transformer encoder)
class VisionTransformer(nn.Module):
    def __init__(
        self,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        patch_size,
        num_patches,
        dropout=0.1,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model)
        )

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

        self.pre_logits = nn.Identity()


    def forward(self, x, return_features=False):
        B, C, T = x.shape  # [176, 4, 3000]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # torch.Size([1, 1, 128])

        x = torch.cat((cls_tokens, x), dim=1)  # torch.Size([1, 1201, 128])

        pos_embed = self.pos_embed
        print(x.shape)
        print(pos_embed.shape)

        x = x + pos_embed

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
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
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
        C, T = data_size
        GS = T // self.patch_size[1]

        x = self.proj_dec(x)  # (B, N, enc_dim) -> (B, N, dec_dim)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)  # (1, n_cls, model_dim) -> (B, n_cls, model_dim)
        x = torch.cat((x, cls_emb), 1)  # add cls tokens -> (B, N + n_cls, model_dim)

        for blk in self.blocks:
            x = blk(x)  # transformer block 통과. patch <-> token 상호작용 학습
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]  # patches (B, N, model_dim) // cls_seg_feat (B, n_cls, model_dim)
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)  # L2 norm
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)  # L2 norm

        masks = patches @ cls_seg_feat.transpose(1, 2)  # scalar product (cos sim)
        masks = self.mask_norm(masks)  # torch.Size([1, 1200, 6])
        masks = rearrange(masks, "b (c t) n -> b n c t", t=int(GS))
        return masks


# Segmenter
class Segmenter(nn.Module):
    def __init__(
        self,
        n_cls,
        columns_names: List[str],
        data_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        flag = False
    ):
        super().__init__()
        self.flag = flag
        self.n_cls = n_cls
        self.patch_size = patch_size

        self.columns_names = columns_names

        self.patch_embed1 = PatchEmbedding(data_size, patch_size, embed_dim=128, channels=2)
        self.patch_embed2 = nn.ModuleDict({
            column: PatchEmbedding(data_size, patch_size, embed_dim=128, channels=2)
            for column in columns_names
        })

        num_patches = (data_size[0] // patch_size[0]) * (data_size[1] // patch_size[1])
        self.encoder = VisionTransformer(  # heartbeat -> 625 / ahi -> 3000
                            n_layers=12,
                            d_model=128,
                            d_ff=64,
                            n_heads=8,
                            n_cls=n_cls,
                            patch_size=patch_size,
                            num_patches=num_patches,
                            dropout=0.1,
                            drop_path_rate=0.0)
        self.patch_size = self.encoder.patch_size
        self.decoder = MaskTransformer(n_cls=n_cls,
                              patch_size=patch_size,
                              d_encoder=128,
                              n_layers=1,
                              n_heads=8,
                              d_model=128,
                              d_ff=64,
                              drop_path_rate=0.0,
                              dropout=0.1)
        self.gap_layer = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x: Dict):
        if self.flag:  # True
            total_x = []
            for column_name, data in x.items():
                x = self.patch_embed2[column_name](data)
                total_x.append(x)
            total_x = torch.cat(total_x, dim=1)
            B, C, T = total_x.shape  # [176, 4, 3000]
            x = torch.unsqueeze(total_x, 2)  # (176, 4, 1, 3000)

        else:
            B, C, T = x.shape  # [176, 4, 3000]
            x = torch.unsqueeze(x, 2)  # (176, 4, 1, 3000)

        x = self.patch_embed1(x)  # [176, 300, 128]
        x = self.encoder(x, return_features=True)
        x = x[:, 1:]  # remove CLS tokens for decoding

        masks = self.decoder(x, (C, T))
        masks = F.interpolate(masks, size=(C, T), mode="bilinear")  # torch.Size([176, 6, 4, 3000])

        masks = masks.permute(0, 1, 3, 2)
        masks = self.gap_layer(masks).squeeze()
        return masks


if __name__ == "__main__":
    model = Segmenter(n_cls=2, patch_size=(1, 10), data_size=(1, 1000), columns_names=['AIRFLOW', 'THOR RES', 'ABDO RES', 'SaO2'])

    x_ = model(
        torch.randn(4, 2, 1250),
    )
    print(x_.shape)