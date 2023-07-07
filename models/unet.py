# Source from guided-diffusion (https://github.com/openai/guided-diffusion/tree/main)

from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from diffusion.utils.fp16_util import convert_module_to_f16, convert_module_to_f32
from diffusion.utils.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class Upsample(nn.Module):
    """
    an upsampling layer
    """
    def __init__(self, in_ch:int, out_ch:int, use_conv:bool, stride:int=2):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.use_conv = use_conv
        if use_conv:
            self.layer = nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1)
    
    def forward(self, x:th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.in_ch, f'x and upsampling layer({self.in_ch}->{self.out_ch}) doesn\'t match.'
        x = F.interpolate(x, scale_factor = self.stride, mode = "nearest")
        if self.use_conv:
            x = self.layer(x)
        return x


class Downsample(nn.Module):
    """
    a downsampling layer
    """
    def __init__(self, in_ch:int, out_ch:int, use_conv:bool, stride:int=2):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        if use_conv:
            self.layer = nn.Conv2d(self.in_ch, self.out_ch, kernel_size = 3, stride = stride, padding = 1)
        else:
            assert self.in_ch == self.out_ch
            self.layer = nn.AvgPool2d(kernel_size = stride, stride = stride)
            
    def forward(self, x:th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.in_ch, f'x and upsampling layer({self.in_ch}->{self.out_ch}) doesn\'t match.'
        return self.layer(x)


class EmbedBlock(nn.Module):
    """
    abstract class
    """
    @abstractmethod
    def forward(self, x, temb, cemb):
        """
        abstract method
        """
        
        
class EmbedSequential(nn.Sequential, EmbedBlock):
    def forward(self, x:th.Tensor, temb:th.Tensor, cemb:th.Tensor) -> th.Tensor:
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, temb, cemb)
            else:
                x = layer(x)
        return x
    
    
# Source from (https://github.com/coderpiaobozhe/classifier-free-diffusion-guidance-Pytorch/blob/master/unet.py#75)
class ResBlock(EmbedBlock):
    def __init__(self, 
            in_ch, 
            out_ch, 
            tdim, 
            zdim, 
            droprate,
            use_conv=False,
            use_scale_shift_norm=False,
            use_checkpoint=False,
            up=False, 
            down=False
        ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.tdim = tdim
        self.zdim = zdim
        self.droprate = droprate
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.updown = up or down

        if up:
            self.h_upd = Upsample(in_ch, in_ch, False)
            self.x_upd = Upsample(in_ch, in_ch, False)
        elif down:
            self.h_upd = Downsample(in_ch, in_ch, False)
            self.x_upd = Downsample(in_ch, in_ch, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.in_layers = nn.Sequential(
            normalization(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1),
        )

        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            linear(
                tdim,
                2 * out_ch if use_scale_shift_norm else out_ch,
            ),
        )
        
        self.zemb_proj = nn.Sequential(
            nn.SiLU(),
            linear(
                zdim,
                2 * out_ch if use_scale_shift_norm else out_ch,
            ),
        )
        
        self.out_layers = nn.Sequential(
            normalization(out_ch),
            nn.SiLU(),
            nn.Dropout(p = self.droprate),
            zero_module(
                nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1)
            ),
            
        )
        if in_ch != out_ch:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size = 1, stride = 1, padding = 0)
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1)
        else:
            self.skip_connection = nn.Identity()
         
    def forward(self, x, temb, zemb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param temb: an [N x tdim] Tensor of timestep embeddings.
        :param zemb: an [N x zdim] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, temb, zemb), self.parameters(), self.use_checkpoint
        )
                
    def _forward(self, x:th.Tensor, temb:th.Tensor, zemb:th.Tensor) -> th.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
            
        emb_out = self.temb_proj(temb)[:, :, None, None]
        emb_out += self.zemb_proj(zemb)[:, :, None, None]
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h
    

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        latent_dim,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dims = dims
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        self.z_emb = nn.Sequential(
            linear(latent_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [EmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        in_ch=ch,
                        out_ch=int(mult * model_channels),
                        tdim=time_embed_dim,
                        zdim=time_embed_dim,
                        droprate=dropout,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbedSequential(
                        ResBlock(
                            in_ch=ch,
                            out_ch=out_ch,
                            tdim=time_embed_dim,
                            zdim=time_embed_dim,
                            droprate=dropout,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, out_ch, conv_resample
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = EmbedSequential(
            ResBlock(
                in_ch=ch,
                out_ch=ch,
                tdim=time_embed_dim,
                zdim=time_embed_dim,
                droprate=dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                in_ch=ch,
                out_ch=ch,
                tdim=time_embed_dim,
                zdim=time_embed_dim,
                droprate=dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        in_ch=ch + ich,
                        tdim=time_embed_dim,
                        zdim=time_embed_dim,
                        droprate=dropout,
                        out_ch=int(model_channels * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            in_ch=ch,
                            tdim=time_embed_dim,
                            zdim=time_embed_dim,
                            droprate=dropout,
                            out_ch=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, out_ch, conv_resample)
                    )
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        
    
    def forward(self, x, timesteps, z):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param z: an [N, d] Tensor of latent, if latent-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        temb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        zemb = self.z_emb(z)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, temb, zemb)
            hs.append(h)
        h = self.middle_block(h, temb, zemb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, temb, zemb)
        h = h.type(x.dtype)
        return self.out(h)



class ThreedAwareConv(nn.Module):
    """
    Roll out triplane features & apply 3d-aware conv
    
    :param input_size: input feature resolution
    :param channels: channels in the inputs and outputs
    :
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        # conv input tensor should be channel-wise concated triplanes
        assert in_ch % 3 == 0 and out_ch % 3 == 0

        self.in_ch = in_ch
        self.out_ch = out_ch
        # 3d-aware conv input channel is expanded 3 times in batch axis (roll-out 3D representation per feature)
        # then shrink to 1/3 channels through channel axis, and reshaped to original size.
        self.conv = nn.Conv2d(in_ch, out_ch // 3, kernel_size, stride, padding)
        
    def forward(self, y):
        # y : [B, 3C, H, W]
        B, C, H, W = y.size()
        assert self.in_ch == C # input tensor channel size must match conv weight size
        
        # y = B x [y_uv, y_vw, y_wu]
        # row_pool = B x [y_(.)v, y_(.)w, y_(.)u]
        # col_pool = B x [y_u(.), y_v(.), y_w(.)]
        row_pool = th.mean(y, dim=-2, keepdim=True).expand_as(y) # [B, 3C, 1, W] -> [B, 3C, H, W]
        col_pool = th.mean(y, dim=-1, keepdim=True).expand_as(y) # [B, 3C, H, 1] -> [B, 3C, H, W]
        
        roll_out = th.stack([
            y, 
            th.roll(row_pool, self.in_ch//3, dims=1), 
            th.roll(col_pool, self.in_ch//3, dims=-1)
        ], dim=1).view(B, 3, 3, self.in_ch//3, H, W).transpose(1, 2).reshape(3*B, self.in_ch, H, W)
        return self.conv(roll_out).view(B, self.out_ch, H, W)
        


class ThreedAwareResBlock(ResBlock):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.in_layers = nn.Sequential(
            normalization(self.in_ch),
            nn.SiLU(),
            ThreedAwareConv(self.in_ch, self.out_ch, 3, padding=1),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_ch),
            nn.SiLU(),
            nn.Dropout(p=self.droprate),
            zero_module(
                ThreedAwareConv(self.out_ch, self.out_ch, 3, padding=1)
            ),
        )

        if self.out_ch == self.in_ch:
            self.skip_connection = nn.Identity()
        elif self.use_conv:
            self.skip_connection = ThreedAwareConv(self.in_ch, self.out_ch, 3, padding=1)
        else:
            self.skip_connection = ThreedAwareConv(self.in_ch, self.out_ch, 1, padding=0)



class TriplaneUNetModel(UNetModel):
    """
    Triplane UNet model containing 3d-aware convolution
    """

    def __init__(
        self,
        threedaware_resolutions,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        if self.num_heads_upsample == -1:
            self.num_heads_upsample = self.num_heads

        self.threedaware_resolutions = threedaware_resolutions

        time_embed_dim = self.model_channels * 4
        

        ch = input_ch = 3 * int(self.channel_mult[0] * self.model_channels) # Triplane! -> Triple channels!!
        self.input_blocks = nn.ModuleList(
            [EmbedSequential(conv_nd(2, 3 * self.model_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    self.resblock(
                        ds,
                        in_ch=ch,
                        tdim=time_embed_dim,
                        zdim=time_embed_dim,
                        droprate=self.dropout,
                        out_ch=3 * int(mult * self.model_channels), # Triplane! -> Triple channels!!
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = 3 * int(mult * self.model_channels) # Triplane! -> Triple channels!!
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbedSequential(
                        self.resblock(
                            ds,
                            in_ch=ch,
                            tdim=time_embed_dim,
                            zdim=time_embed_dim,
                            droprate=self.dropout,
                            out_ch=out_ch,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                        )
                        if self.resblock_updown
                        else Downsample(
                            ch, out_ch, self.conv_resample
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = EmbedSequential(
            self.resblock(
                ds,
                in_ch=ch,
                out_ch=ch,
                tdim=time_embed_dim,
                zdim=time_embed_dim,
                droprate=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=self.use_checkpoint,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
                use_new_attention_order=self.use_new_attention_order,
            ),
            self.resblock(
                ds,
                in_ch=ch,
                out_ch=ch,
                tdim=time_embed_dim,
                zdim=time_embed_dim,
                droprate=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    self.resblock(
                        ds,
                        in_ch=ch + ich,
                        out_ch=3 * int(self.model_channels * mult),
                        tdim=time_embed_dim,
                        zdim=time_embed_dim,
                        droprate=self.dropout,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = 3 * int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        self.resblock(
                            ds,
                            in_ch=ch,
                            out_ch=out_ch,
                            tdim=time_embed_dim,
                            zdim=time_embed_dim,
                            droprate=self.dropout,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                        )
                        if self.resblock_updown
                        else Upsample(ch, out_ch, self.conv_resample)
                    )
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            #conv_nd(2, input_ch, self.out_channels, 3, padding=1)
            zero_module(conv_nd(2, input_ch, self.out_channels, 3, padding=1)),
        )

    def resblock(self, ds, *args, **kwargs):
        if ds in self.threedaware_resolutions:
            return ThreedAwareResBlock(*args, **kwargs) 
        else:
            return ResBlock(*args, **kwargs)
    
    

class SuperResModel(nn.Module):
    """
    A Tri-plane Upsampler that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, opt):
        super().__init__()
        
        assert opt.large_size % opt.small_size == 0 # Downsample from HR -> LR 
        
        self.large_size = opt.large_size 
        self.small_size = opt.small_size 
        
        from torchvision.transforms import Resize, InterpolationMode
        self.resizer = Resize(self.small_size, interpolation=InterpolationMode.BICUBIC)
        
        time_embed_dim = opt.num_channels * 4
        
        self.time_embed = nn.Sequential(
            linear(opt.num_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        self.z_emb = nn.Sequential(
            linear(opt.latent_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        upsample_ratio = downsample_ratio = opt.large_size // opt.small_size
                
        self.in_channels = 3 * opt.num_channels  # Triplane! -> Triple channels!!
        self.num_res_blocks = opt.num_res_blocks
        self.num_conv_per_resblock = opt.num_conv_per_resblock
        self.use_scale_shift_norm = opt.use_scale_shift_norm
        
        self.down_block = EmbedSequential(Downsample(self.in_channels, self.in_channels, 3, stride=downsample_ratio))
        
        ch = 2 * self.in_channels # Twice for low res image concat!
        
        # Residual blocks
        self.conv_blocks = nn.ModuleList([])
        self.skip_connections = nn.ModuleList([])
        
        self.temb_proj = nn.ModuleList([])
        self.zemb_proj = nn.ModuleList([])
        
        res_layers = []
        for _ in range(opt.num_res_blocks):
            for _ in range(opt.num_conv_per_resblock):
                res_layers.append(nn.Sequential(
                        normalization(ch),
                        nn.SiLU(),
                        ThreedAwareConv(ch, ch, 3, padding=1),
                    )
                )
            self.conv_blocks.append(EmbedSequential(*res_layers))
            self.temb_proj.append(nn.Sequential(
                nn.SiLU(),
                linear(time_embed_dim, 2 * ch if opt.use_scale_shift_norm else ch)
            ))
            self.zemb_proj.append(nn.Sequential(
                nn.SiLU(),
                linear(time_embed_dim, 2 * ch if opt.use_scale_shift_norm else ch)
            ))
            self.skip_connections.append(nn.Identity())
        
        self.up_block = EmbedSequential(Upsample(ch, self.in_channels, 3, stride=upsample_ratio))
        

    def forward(self, x, timesteps, z, low_res=None, **kwargs):
        if low_res is None:
            low_res = self.resizer(x)
            
        temb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        zemb = self.z_emb(z)
        assert len(temb.shape) == len(zemb.shape)
        
        x = self.down_block(x)
        
        
        x = th.cat([x, low_res], dim=1)
        
        for i in range(self.num_res_blocks):
            h = self.conv_blocks[i][:-1](x, temb, zemb)
            temb_out = self.temb_proj[i](temb).type(h.dtype)
            zemb_out = self.zemb_proj[i](temb).type(h.dtype)
            while len(temb_out.shape) < len(h.shape):
                temb_out = temb_out[..., None]
                zemb_out = zemb_out[..., None]
            emb_out = temb_out + zemb_out
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.conv_blocks[i][-1][0], self.conv_blocks[i][-1][1:]
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.conv_blocks[i][-1](h) 
            x = h + self.skip_connections[i](x)
        
        return self.up_block(x)
    
'''
class SuperResModel(TriplaneUNetModel):
"""
A UNetModel that performs super-resolution.

Expects an extra kwarg `low_res` to condition on a low-resolution image.
"""

def __init__(self, threedaware_resolutions, image_size, in_channels, *args, **kwargs):
    super().__init__(threedaware_resolutions, image_size, in_channels * 2, *args, **kwargs)

def forward(self, x, timesteps, z, low_res=None, **kwargs):
    _, _, new_height, new_width = x.shape
    upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
    x = th.cat([x, upsampled], dim=1)
    return super().forward(x, timesteps, z, **kwargs)
'''