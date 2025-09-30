import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Tuple, Optional
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from timm.models.layers import DropPath, trunc_normal_


# ------------------------------
# Encoder - Plain ViT (Patch Embedding + Transformer Blocks)
#           return (train_dataset, eval_dataset), (channel_num, class_num)
# ------------------------------


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
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, d_model)
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

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # torch.Size([1, 1, 128])
        x = torch.cat((cls_tokens, x), dim=1)  # torch.Size([1024, 151, 128])

        x = x + self.pos_embed  # torch.Size([1, 51, 128])
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

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # q 따로
        self.k = nn.Linear(dim, dim, bias=qkv_bias)  # k 따로
        self.v = nn.Linear(dim, dim, bias=qkv_bias)  # v 따로 => cross-attention 구조

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size()
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()  # softmax 전의 sim map 저장 -> sigmoid/aggregation 해서 segmentation mask로 활용
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads  # 반환되는 attn은 평균된 raw sim => mask prediction


class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = CrossAttention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1
        )  # 기본 Transformer Decoder의 Attention 모듈을 SegViT 모델에 맞게 교체

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tuple:

        # (1) masked self-attention (decoder 내부 q 토큰끼리)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,  # q=k=v : tgt (class tokens, decoder의 입력 토큰)
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # (2) cross-attention (q=class tokens, key/values=encoder memory)
        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # (3) FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn2


class TPN_Decoder(TransformerDecoder):
    def forward(
        self,
        tgt: Tensor,  # query (class tokens)
        memory: Tensor,  # key/value (encoder에서 나온 patch-level feature tokens)
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> Tuple:

        output = tgt
        # attns = []
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)  # Decoder layers
            # attns.append(attn)  # 각 layer별 raw sim map

        if self.norm is not None:
            output = self.norm(output)

        return output, attn  # 실제 구현에서는 마지막 decoder layer의 sim map을 사용


# Decoder
class ATMHead(nn.Module):
    def __init__(
            self,
            data_size,
            in_channels,
            num_classes,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=3,
            use_proj=True,
            CE_loss=False,
            crop_train=False,
            shrink_ratio=None,
            **kwargs,
    ):

        super().__init__()

        self.data_size = data_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_stages = use_stages
        self.crop_train = crop_train

        nhead = num_heads
        dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []

        # stage별 projection + norm + decoder
        for i in range(self.use_stages):
            # 1. projection layer (FC layer to change ch)
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_(proj.weight, std=.02)
            else:
                proj = nn.Identity()
            input_proj.append(proj)

            # 2. normalization layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            proj_norm.append(norm)

            # 3. decoder layer
            decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
            decoder = TPN_Decoder(decoder_layer, num_layers)
            atm_decoders.append(decoder)

        self.input_proj = nn.ModuleList(input_proj)
        self.proj_norm = nn.ModuleList(proj_norm)
        self.decoder = nn.ModuleList(atm_decoders)
        self.q = nn.Embedding(self.num_classes, dim)

        self.class_embed = nn.Linear(dim, self.num_classes + 1)
        self.CE_loss = CE_loss

    def forward(self, inputs):
        x = []
        for stage_ in inputs[:self.use_stages]:
            x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        x.reverse()

        bs = x[0].size()[0]
        laterals = []
        attns = []
        maps_size = []
        qs = []

        # init class tokens (query)
        q = self.q.weight.repeat(bs, 1, 1).transpose(0, 1)

        for idx, (x_, proj_, norm_, decoder_) in enumerate(zip(x, self.input_proj, self.proj_norm, self.decoder)):
            lateral = norm_(proj_(x_))
            laterals.append(lateral)

            q, attn = decoder_(q, lateral.transpose(0, 1))
            attn = attn.transpose(-1, -2)

            attn = attn.permute(0, 2, 1)  #  (B, L, C) -> (B, C, L) for 1D signals
            maps_size.append(attn.size()[-2:])
            qs.append(q.transpose(0, 1))
            attns.append(attn)

        qs = torch.stack(qs, dim=0)
        outputs_class = self.class_embed(qs)
        out = {"pred_logits": outputs_class[-1]}

        outputs_seg_masks = []
        size = maps_size[-1][-1]  # torch.Size([3, 150]) -> 150
        print(size)

        for i_attn, attn in enumerate(attns):
            if i_attn == 0:
                outputs_seg_masks.append(F.interpolate(attn, size=size, mode='linear', align_corners=False))
            else:
                outputs_seg_masks.append(outputs_seg_masks[i_attn - 1] +
                                         F.interpolate(attn, size=size, mode='linear', align_corners=False))

        out["pred_masks"] = F.interpolate(outputs_seg_masks[-1],
                                          size=self.data_size[1],
                                          mode='linear', align_corners=False)

        out["pred"] = self.semantic_inference(out["pred_logits"], out["pred_masks"])

        return out

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bql->bcl", mask_cls, mask_pred)
        return semseg


# ------------------------------
# Model - SegViT
# ------------------------------

class SegViT(nn.Module):
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
        dec_n_heads: int = 8,
        dec_n_layers: int = 1,
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

        self.decoder = ATMHead(
            data_size=data_size,
            in_channels=enc_d_model,
            num_classes=n_cls,
            embed_dims=dec_d_model,
            num_layers=dec_n_layers,
            num_heads=dec_n_heads,
            use_stages=1,
        )

    def forward(self, x: Tensor) -> torch.Tensor:
        print(x.shape)
        x = self.norm(x)
        x = torch.unsqueeze(x, 2)  # (2, 3, 1, 3000)

        x = self.encoder(x, return_features=True)
        x = x[:, 1:]  # remove CLS tokens

        out = self.decoder([x])

        return out


if __name__ == "__main__":
    model = SegViT(
        n_cls=3,
        data_size=(1, 3000),
        patch_size=(1, 20),
        channels=3,
        enc_d_model=128,
        enc_d_ff=64,
        enc_n_heads=4,
        enc_n_layers=1,
        dec_d_model=256,
        dec_n_heads=8,
        dec_n_layers=2,
    )

    x = torch.randn(2, 3, 3000)
    output = model(x)  # dict_keys(['pred_logits', 'pred_masks', 'pred'])

    print(output['pred_logits'].size())
    print(output['pred_masks'].size())
    print(output['pred'].shape)