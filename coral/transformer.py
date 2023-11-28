import math
from audioop import bias
from calendar import c
from functools import lru_cache, partial

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import (Block, PatchEmbed, _cfg,
                                            trunc_normal_)
from torchvision import transforms


class UnPatchToken(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        out_chans=1,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[0]
        self.flatten = flatten

        self.proj = nn.ConvTranspose2d(
            embed_dim, out_chans, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        B, N, L = x.shape
        grid_size = int(math.sqrt(N))
        patch_side = int(math.sqrt(L // self.out_chans))

        x = x.permute(0, 2, 1)
        x = x.reshape(B, L, grid_size, grid_size)
        # x = x.reshape(B, self.out_chans, L // self.out_chans, grid_size, grid_size)
        # x = x.reshape(B, self.out_chans, patch_side*grid_size, patch_side*grid_size)
        x = self.proj(x)  # !!!!! WARNING: commented out beacuse of the head
        return x


class SimpleTokenizer(nn.Module):
    def __init__(self, input_size, patch_size, num_frames, dim, padding=0):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Conv1d(
            input_size, dim, 1, stride=1
        )  # nn.Linear(input_size, dim)
        self.n_patches = num_frames
        self.patch_size = patch_size
        # self.norm = nn.LayerNorm(input_size)
        # self.posemb = nn.Parameter(torch.randn(1, num_frames, dim))

    def forward(self, x):
        t = x.shape[1]
        # expect x to be of shape (b t z)
        p = self.patch_size
        # x = self.norm(x)
        # print('toto x1', x.shape, self.prefc(x).shape, self.posemb.shape)
        x = x.permute(0, 2, 1)
        x = self.prefc(x)  # + self.posemb
        x = x.permute(0, 2, 1)

        return x


class CoderecTokenizer(nn.Module):
    def __init__(self, input_size, patch_size, num_frames, dim, padding=0):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Linear(patch_size, dim)
        # self.norm = nn.LayerNorm(input_size)
        n_patches = (input_size + padding * 2) // patch_size * num_frames
        self.n_patches = n_patches
        self.posemb = nn.Parameter(torch.randn(1, n_patches, dim))

    def forward(self, x):
        t = x.shape[1]
        # expect x to be of shape (b t z)
        p = self.patch_size
        # x = self.norm(x)
        x = x.unfold(2, p, p)  # (b, t, n p)
        x = rearrange(x, "b t n p -> b (t n) p").contiguous()
        # print('toto x1', x.shape, self.prefc(x).shape, self.posemb.shape)
        x = self.prefc(x) + self.posemb

        return x


class CodeSeqrecTokenizer(nn.Module):
    def __init__(self, input_size, patch_size, num_frames, dim, padding=0):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Linear(patch_size, dim)
        n_patches = (input_size + padding * 2) // patch_size
        self.n_patches = n_patches
        self.posemb = nn.Parameter(torch.randn(1, num_frames, n_patches, dim))
        self.mu = 100
        self.M = 256
        self.denominator = torch.tensor(self.M * self.mu + 1)

    def forward(self, x):
        t = x.shape[1]
        # expect x to be of shape (b t z)
        x = (
            torch.sign(x)
            * torch.log(x.abs() * self.mu + 1)
            / torch.log(self.denominator)
        )
        p = self.patch_size
        x = x.unfold(2, p, p).contiguous()  # (b, t, n, p)
        # print('toto x1', x.shape, self.prefc(x).shape, self.posemb.shape)
        x = self.prefc(x) + self.posemb

        return x


class AttentionOld(nn.Module):
    def __init__(self, dim, n_head, head_dim, dropout=0):
        super().__init__()
        self.n_head = n_head
        inner_dim = n_head * head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.scale = head_dim**-0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, fr, to=None):
        if to is None:
            to = fr
        q = self.to_q(fr)
        k, v = self.to_kv(to).chunk(2, dim=-1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.n_head),
            [q, k, v],
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)  # b h n n
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for norm_attn, norm_ff in self.layers:
            x = x + norm_attn(x)
            x = x + norm_ff(x)
        return self.norm(x)  # x


class TransformerEncoderTime(nn.Module):
    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)
        for l in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, n_head, head_dim, dropout=l / depth * dropout
                            ),
                        ),
                        PreNorm(
                            dim, FeedForward(dim, ff_dim, dropout=l / depth * dropout)
                        ),
                    ]
                )
            )

    def forward(self, x):
        for norm_attn, norm_ff in self.layers:
            x = x + norm_attn(x)
            x = x + norm_ff(x)
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(
        self,
        dim=192,
        depth=4,
        n_head=3,
        pool="cls",
        head_dim=64,
        dropout=0.0,
        ff_dim=768,
        emb_dropout=0.0,
        scale_dim=4,
    ):
        super().__init__()

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = TransformerEncoderTime(
            dim, depth, n_head, head_dim, ff_dim, dropout
        )

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = TransformerEncoderTime(
            dim, 3, n_head, head_dim, ff_dim, dropout
        )

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, "() n d -> b t n d", b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x = self.dropout(x)

        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.space_transformer(x)
        x = rearrange(x, "(b t) ... -> b t ...", b=b)

        cls_temporal_tokens = repeat(
            self.temporal_token, "() n d -> b n d", b=b * (n + 1)
        )  # b
        x = rearrange(x, "b t n d -> (b n) t d")

        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0, ...]

        x = rearrange(x, "(b n) d -> b n d", b=b)
        return self.mlp_head(x)


class TransForecast(nn.Module):
    def __init__(self, tokenizer, transformer_encoder):
        super().__init__()

        self.dim = transformer_encoder["args"]["dim"]
        self.num_frames = tokenizer["args"]["num_frames"]
        tokenizer["args"]["dim"] = self.dim
        # self.tokenizer = CoderecTokenizer(**dict(tokenizer['args']))
        self.tokenizer = SimpleTokenizer(**dict(tokenizer["args"]))
        self.transformer_encoder = TransformerEncoder(
            **dict(transformer_encoder["args"])
        )
        self.n_patches = self.tokenizer.n_patches

        # self.temporal_linear = nn.Linear(self.n_patches, int(self.n_patches/self.num_frames))
        # self.project_token = nn.Linear(self.dim, self.tokenizer.patch_size)
        # self.project_code = nn.Linear(int(self.n_patches/self.num_frames)*self.tokenizer.patch_size, self.tokenizer.input_size)

        # self.dropout = nn.Dropout(0.01)
        # self.project_forecast = nn.Linear(self.n_patches*self.dim,  self.tokenizer.input_size)
        self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # self.project_map = nn.Linear(self.dim, self.dim, bias=False)
        self.project_map = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Dropout(0.01),
            nn.Linear(self.dim, self.dim, bias=False),
        )

    def forward(self, data):
        dtokens = self.tokenizer(data)
        B = dtokens.shape[0]
        input_tokens = torch.cat((self.temporal_token.repeat(B, 1, 1), dtokens), 1)
        trans_out = self.transformer_encoder(input_tokens)

        x = trans_out[:, 0, :]
        # x = self.dropout(x)
        # out = self.mlp_head(x)
        out = self.project_map(x)

        # x = einops.rearrange(trans_out, 'b n l -> b (n l)')
        # x = self.dropout(x)
        # out = self.project_forecast(x)
        # trans_out = einops.rearrange(trans_out, 'b n l -> b l n')
        # x = self.temporal_linear(trans_out)
        # x = einops.rearrange(x, 'b l n -> b n l')
        # x = self.project_token(x)
        # x = einops.rearrange(x, 'b n l -> b (n l)')
        # out = self.project_code(x)

        return out


class TransForecastTime(nn.Module):
    def __init__(self, tokenizer, transformer_encoder):
        super().__init__()

        self.dim = transformer_encoder["args"]["dim"]
        self.num_frames = tokenizer["args"]["num_frames"]
        tokenizer["args"]["dim"] = self.dim
        self.tokenizer = CodeSeqrecTokenizer(**dict(tokenizer["args"]))
        self.transformer_encoder = ViViT(**dict(transformer_encoder["args"]))
        self.n_patches = self.tokenizer.n_patches
        # self.temporal_linear = nn.Linear(self.n_patches, int(self.n_patches/self.num_frames))
        # self.project_token = nn.Linear(self.dim, self.tokenizer.patch_size)
        # self.project_code = nn.Linear(self.dim, self.tokenizer.input_size)
        self.dropout = nn.Dropout(0.01)
        self.project_forecast = nn.Linear(
            (self.n_patches + 1) * self.dim, self.tokenizer.input_size
        )

    def forward(self, data):
        dtokens = self.tokenizer(data)
        # print('dtokens', dtokens.shape)
        B = dtokens.shape[0]
        x = self.transformer_encoder(dtokens)
        x = rearrange(x, "b n l -> b (n l)")
        x = self.dropout(x)
        out = self.project_forecast(x)

        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.id = nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        # print('shape', B, N, C)
        # x = x.reshape(B, N, self.num_heads, self.head_dim)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = self.id(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = bias
        drop_probs = drop

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Layer_scale_init_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class vit_models(nn.Module):
    """Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(
        self,
        input_size=256,
        num_frames=10,
        patch_size=16,
        in_chans=1,
        num_classes=0,
        embed_dim=256,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        global_pool=None,
        block_layers=Layer_scale_init_Block,
        Patch_layer=PatchEmbed,
        act_layer=nn.GELU,
        Attention_block=Attention,
        Mlp_block=Mlp,
        dpr_constant=True,
        init_scale=1e-4,
        mlp_ratio_clstk=4.0,
    ):
        super().__init__()

        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # self.patch_embed = Patch_layer(
        #        img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches
        self.patch_embed = SimpleTokenizer(
            input_size, patch_size, num_frames, embed_dim
        ).cuda()
        num_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                block_layers(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block,
                    Mlp_block=Mlp_block,
                    init_values=init_scale,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module="head")]
        self.head = nn.Linear(
            embed_dim, input_size
        )  # if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        # print('pos', self.pos_embed.device)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = x + self.pos_embed
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)

        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)

        return x


class FrameToFrameTF(nn.Module):
    """Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(
        self,
        input_size=64,
        patch_size=16,
        in_chans=1,
        num_classes=0,
        embed_dim=256,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        global_pool=None,
        block_layers=Layer_scale_init_Block,
        Patch_layer=PatchEmbed,
        act_layer=nn.GELU,
        Attention_block=Attention,
        Mlp_block=Mlp,
        dpr_constant=True,
        init_scale=1e-4,
        mlp_ratio_clstk=4.0,
    ):
        super().__init__()

        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.unpatch = UnPatchToken(
            img_size=input_size,
            patch_size=patch_size,
            out_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=None,
            flatten=True,
        )

        self.pred_token = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                block_layers(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block,
                    Mlp_block=Mlp_block,
                    init_values=init_scale,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module="head")]
        self.head = HeadMLP(embed_dim, embed_dim, in_chans, 2)

        # nn.Linear(
        #    embed_dim, input_size
        # )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.pred_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "pred_token"}

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        # print('pos', self.pos_embed.device)

        pred_token = self.pred_token.expand(B, -1, -1)
        x = x + self.pos_embed
        x = torch.cat((pred_token, x), dim=1)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, : self.num_patches]

    def forward(self, x):
        x = self.forward_features(x)

        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)

        # x = self.head(x) # to comment if not working
        x = self.unpatch(x)

        return x


class HeadMLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ClimaX(nn.Module):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343
    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
    """

    def __init__(
        self,
        img_size=[16, 16],
        patch_size=1,
        in_channels=8,
        embed_dim=768,
        depth=4,
        decoder_depth=2,
        num_heads=12,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
    ):
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_size

        # variable tokenization: separate embedding layer for each input variable
        # self.token_embeds = nn.ModuleList(
        #    [PatchEmbed(img_size, patch_size, 1, embed_dim) for i in range(len(default_vars))]
        # )
        self.in_channels = in_channels
        self.token_embeds = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.token_embeds.num_patches

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        # self.var_embed, self.var_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding and lead time embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=True
        )
        self.lead_time_embed = nn.Linear(1, embed_dim)

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # --------------------------------------------------------------------------

        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(
            nn.Linear(embed_dim, self.in_channels * patch_size**2)
        )  # len(self.default_vars) * patch_size**2)
        self.head = nn.Sequential(*self.head)

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        v = self.pos_embed.data
        trunc_normal_(v, std=0.02)

        w = self.token_embeds.proj.weight.data
        trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
        # self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        # token embedding layer
        # for i in range(len(self.token_embeds)):
        #    w = self.token_embeds[i].proj.weight.data
        #    trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        pass
        # var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        # TODO: create a mapping from var --> idx
        # var_map = {}
        # idx = 0
        # for var in self.default_vars:
        #    var_map[var] = idx
        #    idx += 1
        # return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: torch.Tensor):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """

        p = self.patch_size
        c = self.in_channels
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward_encoder(self, x: torch.Tensor):
        # x: `[B, V, H, W]` shape.

        # tokenize each variable separately
        # embeds = []
        # var_ids = self.get_var_ids(variables, x.device)
        # for i in range(len(var_ids)):
        #    id = var_ids[i]
        #    embeds.append(self.token_embeds[id](x[:, i : i + 1]))
        # x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        # var_embed = self.get_var_emb(self.var_embed, variables)
        # x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        # x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding

        x = self.token_embeds(x) + self.pos_embed

        # add lead time embedding
        # lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        # lead_time_emb = lead_time_emb.unsqueeze(1)
        # x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, x):
        """Forward pass through the model.
        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.
        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        out_transformers = self.forward_encoder(x)  # B, L, D
        preds = self.head(out_transformers)  # B, L, V*p*p

        preds = self.unpatchify(preds)

        return preds

    def evaluate(
        self,
        x,
        y,
        lead_times,
        variables,
        out_variables,
        transform,
        metrics,
        lat,
        clim,
        log_postfix,
    ):
        _, preds = self.forward(
            x, y, lead_times, variables, out_variables, metric=None, lat=lat
        )
        return [
            m(preds, y, transform, out_variables, lat, clim, log_postfix)
            for m in metrics
        ]
