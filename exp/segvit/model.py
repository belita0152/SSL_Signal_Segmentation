import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
import math
from timm.models.layers import DropPath, trunc_normal_


# ------------------------------
# Encoder - Plain ViT (Patch Embedding + Transformer Blocks)
#           return (train_dataset, eval_dataset), (channel_num, class_num)
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
        # self.pos_embed = nn.Parameter(
        #     torch.randn(1, self.patch_embed.num_patches + 1, d_model)
        # )

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
# Decoder - Mask-to-Attention (ATM) modules
#           return (mask, updated_class_tokens)
# ------------------------------

class CrossAttention1D(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)  # q 따로
        self.k = nn.Linear(dim, dim, bias=True)  # k 따로
        self.v = nn.Linear(dim, dim, bias=True)  # v 따로
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.shape
        Nk = xk.shape[1]
        Nv = xv.shape[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()  # softmax 전의 sim map 저장 -> sigmoid/aggregation 해서 segmentation mask로 활용
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_save.sum(dim=1) / self.num_heads  # 반환되는 attn은 평균된 raw sim => mask prediction


class ATMModule(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 dropout,
                 mlp_ratio=4.0,):
        super().__init__()

        self.self_attn = SelfAttention(dim, num_heads, dropout)
        self.cross_attn = CrossAttention1D(dim, num_heads, attn_drop=0.1, proj_drop=0.0)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # FFN
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)


    def forward(
        self,
        q: Tensor,
        enc_features: Tensor,
    ) -> Tuple:

        # ATM Module
        # (1) masked self-attention (decoder 내부 q 토큰끼리)
        q = self.norm(q)
        q1, _ = self.self_attn(q)  # q=k=v : tgt (class tokens, decoder의 입력 토큰)
        q = q + self.dropout(q1)
        q = self.norm(q)

        # (2) cross-attention (q=class tokens, k/v=encoder features)
        q2, attn2 = self.cross_attn(q, enc_features, enc_features)
        q = q + self.dropout(q2)
        q = self.norm(q)

        # (3) FFN
        q2 = self.fc2(self.dropout(self.activation(self.fc1(q))))
        q = q + self.dropout(q2)
        q = self.norm(q)

        return q, attn2


class ATMHead(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            embed_dim: int,
            num_layers: int = 3,
            num_heads: int = 8,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Encoder output dim -> Decoder input dim
        self.input_proj = nn.Linear(in_channels, embed_dim)
        self.proj_norm = nn.LayerNorm(embed_dim)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            ATMModule(dim=self.embed_dim, num_heads=self.num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])

        self.class_prediction_head = nn.Linear(embed_dim, self.num_classes+1)  # github는 nn.Linear(dim, self.num_classes + 1)

        # Final classification (class embedding)
        self.class_embed = nn.Linear(embed_dim, num_classes + 1)

    def forward(self, queries, enc_features: torch.Tensor):
        B, L, C = enc_features.shape  # torch.Size([2, 150, 128])

        # (1) projection
        enc_features = self.input_proj(enc_features)
        enc_features = self.proj_norm(enc_features)

        # (2) transformer decoder
        qs, masks = [], []
        for layer in self.layers:
            q, mask = layer(queries, enc_features)
            qs.append(q.transpose(0, 1))
            masks.append(mask)

        qs = torch.stack(qs, dim=0)

        # (4) class prediction (FC in diagram)
        output_class = self.class_prediction_head(qs).squeeze(-1)
        class_logits = output_class[-1]

        # (5) generate masks
        norm_encoder_features = F.normalize(enc_features, p=2, dim=-1)
        norm_class_queries = F.normalize(q, p=2, dim=-1)  # final output from ATM module

        mask_logits = norm_encoder_features @ norm_class_queries.transpose(1, 2)

        return class_logits, mask_logits


# ------------------------------
# Model - SegViT
# ------------------------------

class SegViT(nn.Module):
    def __init__(
        self,
        n_cls: int,
        data_size: Tuple[int, int] = (1, 3000),
        patch_size: Tuple[int, int] = (1, 10),
        channels: int = 2,
        enc_d_model: int = 128,
        enc_d_ff: int = 64,
        enc_n_heads: int = 8,
        enc_n_layers: int = 5,
        enc_dropout: float = 0.1,
        enc_drop_path_rate: float = 0.0,
        dec_d_model: int = 128,
        dec_n_heads: int = 8,
        dec_n_layers: int = 2,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.enc_d_model = enc_d_model
        self.data_size = data_size
        self.patch_size = patch_size
        self.norm = nn.BatchNorm1d(channels, affine=True)

        # 1. Backbone (ViT Encoder)
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

        # Learnable class queries
        self.class_queries = nn.Parameter(torch.randn(self.n_cls, self.enc_d_model))

        # 2. ATM Decoder Head
        self.decoder = ATMHead(
            in_channels=enc_d_model,
            num_classes=n_cls,
            embed_dim=dec_d_model,
            num_layers=dec_n_layers,
            num_heads=dec_n_heads,
        )

    def forward(self, x: Tensor) -> torch.Tensor:
        x0 = x.shape[-1]

        x = self.norm(x)
        x = torch.unsqueeze(x, 2)  # (2, 3, 1, 3000)

        enc_features = self.encoder(x, return_features=True)
        enc_features = enc_features[:, 1:]  # remove CLS tokens

        batch_class_queries = self.class_queries.unsqueeze(0).repeat(x.shape[0], 1, 1)

        _, mask_logits = self.decoder(batch_class_queries, enc_features)
        mask_logits = mask_logits.transpose(1, 2)

        if mask_logits.shape[-1] != x.shape[-1]:  # Use padded length for interpolation
            mask_logits = F.interpolate(mask_logits, size=x0, mode="linear", align_corners=False)

        return mask_logits


if __name__ == "__main__":
    model = SegViT(
        n_cls=3,
        data_size=(1, 3000),
        patch_size=(1, 20),
        channels=3,
        enc_d_model=128,
        enc_d_ff=64,
        enc_n_heads=4,
        enc_n_layers=5,
        dec_d_model=128,
        dec_n_heads=8,
        dec_n_layers=2,
    )

    x = torch.randn(2, 3, 3000)
    output = model(x)  # dict_keys(['pred_logits', 'pred_masks', 'pred'])
    print(output.shape)
