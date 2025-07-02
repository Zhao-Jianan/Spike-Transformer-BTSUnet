import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer, base
from timm.layers import trunc_normal_, DropPath
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

# class BNAndPad3DLayer(nn.Module):
#     def __init__(self, pad_voxels, num_features, step_mode='m', **bn_kwargs):
#         """
#         3D BN+ padding组合，支持单步和多步模式。

#         :param pad_voxels: int，前后上下左右各方向填充的体素数
#         :param num_features: 通道数
#         :param bn_kwargs: 传给 BatchNorm3d 的额外参数，如 eps、momentum 等
#         """
#         super().__init__()
#         self.pad_voxels = pad_voxels
#         self.bn = layer.BatchNorm3d(num_features, step_mode=step_mode, **bn_kwargs)
#         self.step_mode = step_mode

#     def _compute_pad_value(self):
#         if self.bn.affine:
#             pad_value = (
#                 self.bn.bias.detach()
#                 - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps))
#         else:
#             pad_value = -self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
#         return pad_value.view(1, -1, 1, 1, 1)  # reshape to [1, C, 1, 1, 1]

#     def _pad_tensor(self, x, pad_value):
#         pad = [self.pad_voxels] * 6  # [W_left, W_right, H_top, H_bottom, D_front, D_back]
#         x = F.pad(x, pad)  # 使用0填充
#         # 再替换成 pad_value
#         x[:, :, :self.pad_voxels, :, :] = pad_value
#         x[:, :, -self.pad_voxels:, :, :] = pad_value
#         x[:, :, :, :self.pad_voxels, :] = pad_value
#         x[:, :, :, -self.pad_voxels:, :] = pad_value
#         x[:, :, :, :, :self.pad_voxels] = pad_value
#         x[:, :, :, :, -self.pad_voxels:] = pad_value
#         return x

#     def forward(self, x):
#         if self.step_mode == 's':
#             x = self.bn(x)  # shape: [N, C, D, H, W]
#             if self.pad_voxels > 0:
#                 pad_value = self._compute_pad_value()
#                 x = self._pad_tensor(x, pad_value)
#             return x

#         elif self.step_mode == 'm':
#             if x.dim() != 6:
#                 raise ValueError(f"Expected input shape [T, N, C, D, H, W], but got {x.shape}")
#             x = self.bn(x)
#             if self.pad_voxels > 0:
#                 pad_value = self._compute_pad_value()
#                 # 对每个时间步进行 padding
#                 padded = []
#                 for t in range(x.shape[0]):
#                     padded.append(self._pad_tensor(x[t], pad_value))
#                 x = torch.stack(padded, dim=0)  # [T, N, C, D+2p, H+2p, W+2p]
#             return x

#     @property
#     def weight(self):
#         return self.bn.weight

#     @property
#     def bias(self):
#         return self.bn.bias

#     @property
#     def running_mean(self):
#         return self.bn.running_mean

#     @property
#     def running_var(self):
#         return self.bn.running_var

#     @property
#     def eps(self):
#         return self.bn.eps



# class RepConv3D(nn.Module):
#     def __init__(self, in_channel, out_channel, pad_voxels=1, bias=False, step_mode='m'):
#         super().__init__()

#         # 1x1 projection conv
#         self.proj_conv = layer.Conv3d(in_channels=in_channel, out_channels=in_channel,
#                                       kernel_size=1, stride=1, padding=0, bias=bias, step_mode=step_mode)

#         # BN + Padding
#         self.bn_pad = BNAndPad3DLayer(pad_voxels=pad_voxels, num_features=in_channel, step_mode=step_mode)

#         # Depthwise 3x3 conv
#         self.dw_conv3x3 = layer.Conv3d(in_channels=in_channel, out_channels=in_channel,
#                                        kernel_size=3, stride=1, padding=0, bias=bias,
#                                        groups=in_channel, step_mode=step_mode)

#         # Pointwise 1x1 conv
#         self.pw_conv1x1 = layer.Conv3d(in_channels=in_channel, out_channels=out_channel,
#                                        kernel_size=1, stride=1, padding=0, bias=bias,
#                                        step_mode=step_mode)

#         # Output BatchNorm
#         self.out_bn = layer.BatchNorm3d(num_features=out_channel, step_mode=step_mode)

#     def forward(self, x):  
#         x = self.proj_conv(x)          # 1×1 conv
#         x = self.bn_pad(x)             # BN + padding
#         x = self.dw_conv3x3(x)         # depthwise 3×3 conv
#         x = self.pw_conv1x1(x)         # pointwise 1×1 conv
#         x = self.out_bn(x)             # output BN
#         return x


# class SepConv3D(nn.Module):
#     """
#     Spiking 3D version of inverted separable convolution from MobileNetV2.
#     Input: [T, B, C, D, H, W]
#     """

#     def __init__(
#         self,
#         dim,
#         expansion_ratio=2,
#         kernel_size=7,
#         padding=3,
#         tau=2.0,
#         step_mode='m',
#         bias=False):
#         super().__init__()
#         med_channels = int(expansion_ratio * dim)

#         # spike layer 1
#         self.lif1 = neuron.ParametricLIFNode(
#             init_tau=tau,
#             decay_input=True,
#             detach_reset=True,
#             v_threshold=1.0,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode,
#             backend='cupy')

#         # pointwise conv 1
#         self.pwconv1 = layer.Conv3d(dim, med_channels, kernel_size=1, stride=1,
#                                     bias=bias, step_mode=step_mode)
#         self.bn1 = layer.BatchNorm3d(med_channels, step_mode=step_mode)

#         # spike layer 2
#         self.lif2 = neuron.ParametricLIFNode(
#             init_tau=tau,
#             decay_input=True,
#             detach_reset=True,
#             v_threshold=1.0,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode,
#             backend='cupy')

#         # depthwise conv
#         self.dwconv = layer.Conv3d(med_channels, med_channels, kernel_size=kernel_size,
#                                    padding=padding, groups=med_channels,
#                                    bias=bias, step_mode=step_mode)

#         # pointwise conv 2
#         self.pwconv2 = layer.Conv3d(med_channels, dim, kernel_size=1, stride=1,
#                                     bias=bias, step_mode=step_mode)
#         self.bn2 = layer.BatchNorm3d(dim, step_mode=step_mode)

#     def forward(self, x):
#         # x: [T, B, C, D, H, W]
#         x = self.lif1(x)
#         x = self.pwconv1(x)
#         x = self.bn1(x)
#         x = self.lif2(x)
#         x = self.dwconv(x)
#         x = self.pwconv2(x)
#         x = self.bn2(x)
#         return x


# class MS_ConvBlock3D(nn.Module):
#     def __init__(
#         self,
#         dim,
#         mlp_ratio=4.0,
#         tau=2.0,
#         step_mode='m'):
#         super().__init__()
#         hidden_dim = int(dim * mlp_ratio)

#         self.conv = SepConv3D(dim=dim, step_mode=step_mode)

#         # Spike + Conv + BN block1
#         self.lif1 = neuron.ParametricLIFNode(
#             init_tau=tau,
#             decay_input=True,
#             detach_reset=True,
#             v_threshold=1.0,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode,
#             backend='cupy')
        
#         self.conv1 = layer.Conv3d(in_channels=dim, out_channels=hidden_dim,
#                                   kernel_size=3, padding=1, bias=False,
#                                   step_mode=step_mode)
#         self.bn1 = layer.BatchNorm3d(num_features=hidden_dim, step_mode=step_mode)

#         # Spike + Conv + BN block2
#         self.lif2 = neuron.ParametricLIFNode(
#             init_tau=tau,
#             decay_input=True,
#             detach_reset=True,
#             v_threshold=1.0,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode,
#             backend='cupy')
        
#         self.conv2 = layer.Conv3d(in_channels=hidden_dim, out_channels=dim,
#                                   kernel_size=3, padding=1, bias=False,
#                                   step_mode=step_mode)
#         self.bn2 = layer.BatchNorm3d(num_features=dim, step_mode=step_mode)

#     def forward(self, x):
#         # x: [T, B, C, D, H, W]

#         # Branch 1: Lightweight convolution block + residual
#         x = self.conv(x) + x
#         x_feat = x

#         # Branch 2: MLP-like spike conv
#         x = self.lif1(x)
#         x = self.conv1(x)
#         x = self.bn1(x)

#         x = self.lif2(x)
#         x = self.conv2(x)
#         x = self.bn2(x)

#         # Final residual
#         x = x_feat + x

#         return x


# class MS_MLP3D(nn.Module):
#     def __init__(
#         self,
#         in_features,
#         hidden_features=None,
#         out_features=None,
#         tau=2.0,
#         step_mode='m'):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features

#         # 用1x1x1卷积实现“全连接”操作
#         self.fc1_conv = layer.Conv3d(in_features, hidden_features, kernel_size=1,
#                                     stride=1, padding=0, bias=False, step_mode=step_mode)
#         self.fc1_bn = layer.BatchNorm3d(hidden_features, step_mode=step_mode)
#         self.fc1_lif = neuron.ParametricLIFNode(
#             init_tau=tau,
#             decay_input=True,
#             detach_reset=True,
#             v_threshold=1.0,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode,
#             backend='cupy')

#         self.fc2_conv = layer.Conv3d(hidden_features, out_features, kernel_size=1,
#                                     stride=1, padding=0, bias=False, step_mode=step_mode)
#         self.fc2_bn = layer.BatchNorm3d(out_features, step_mode=step_mode)
#         self.fc2_lif = neuron.ParametricLIFNode(
#             init_tau=tau,
#             decay_input=True,
#             detach_reset=True,
#             v_threshold=1.0,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode,
#             backend='cupy')

#     def forward(self, x):
#         # x: [T, B, C, D, H, W]
#         x = self.fc1_lif(x)
#         x = self.fc1_conv(x)
#         x = self.fc1_bn(x)

#         x = self.fc2_lif(x)
#         x = self.fc2_conv(x)
#         x = self.fc2_bn(x)
#         return x


# class MS_Attention_RepConv3D_qkv_id(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads=8,
#         qkv_bias=False,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         sr_ratio=1,
#         tau=2.0,
#         step_mode='m'):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
#         self.dim = dim
#         self.num_heads = num_heads
#         self.scale = 0.125 if qk_scale is None else qk_scale

#         self.head_lif = neuron.ParametricLIFNode(
#             init_tau=tau,
#             decay_input=True,
#             detach_reset=True,
#             v_threshold=1.0,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode,
#             backend='cupy')

#         self.q_conv = nn.Sequential(
#             RepConv3D(dim, dim, bias=False, step_mode=step_mode),
#             layer.BatchNorm3d(dim, step_mode=step_mode))
#         self.k_conv = nn.Sequential(
#             RepConv3D(dim, dim, bias=False, step_mode=step_mode),
#             layer.BatchNorm3d(dim, step_mode=step_mode))
#         self.v_conv = nn.Sequential(
#             RepConv3D(dim, dim, bias=False, step_mode=step_mode),
#             layer.BatchNorm3d(dim, step_mode=step_mode))

#         self.q_lif = neuron.ParametricLIFNode(
#             init_tau=tau,
#             decay_input=True,
#             detach_reset=True,
#             v_threshold=1.0,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode,
#             backend='cupy')
        
#         self.k_lif = neuron.ParametricLIFNode(
#             init_tau=tau,
#             decay_input=True,
#             detach_reset=True,
#             v_threshold=1.0,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode,
#             backend='cupy')
        
#         self.v_lif = neuron.ParametricLIFNode(
#             init_tau=tau,
#             decay_input=True,
#             detach_reset=True,
#             v_threshold=1.0,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode,
#             backend='cupy')

#         self.attn_lif = neuron.ParametricLIFNode(
#             init_tau=tau,
#             decay_input=True,
#             detach_reset=True,
#             v_threshold=0.5,
#             v_reset=0.0,
#             surrogate_function=surrogate.ATan(), 
#             step_mode=step_mode,
#             backend='cupy')

#         self.proj_conv = nn.Sequential(
#             RepConv3D(dim, dim, bias=False, step_mode=step_mode),
#             layer.BatchNorm3d(dim, step_mode=step_mode))

#     def forward(self, x):
#         # x: [T, B, C, D, H, W]
#         T, B, C, D, H, W = x.shape
#         N = D * H * W

#         x = self.head_lif(x)
#         q = self.q_conv(x)
#         k = self.k_conv(x)
#         v = self.v_conv(x)

#         # LIF激活 + 拉平空间维到 N
#         q = self.q_lif(q).flatten(3)  # [T, B, C, N]
#         q = q.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
#         q = q.permute(0, 1, 3, 2, 4).contiguous()  # [T, B, heads, N, head_dim]

#         k = self.k_lif(k).flatten(3)
#         k = k.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
#         k = k.permute(0, 1, 3, 2, 4).contiguous()

#         v = self.v_lif(v).flatten(3)
#         v = v.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
#         v = v.permute(0, 1, 3, 2, 4).contiguous()

#         # 注意力计算：k^T @ v
#         attn_kv = torch.matmul(k.transpose(-2, -1), v)  # [T, B, heads, head_dim, head_dim]
#         x = torch.matmul(q, attn_kv) * self.scale  # [T, B, heads, N, head_dim]

#         # 恢复空间维度
#         x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
#         x = self.attn_lif(x).reshape(T, B, C, D, H, W)

#         # 投影卷积
#         x = self.proj_conv(x)

#         return x


# class MS_Block3D(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads,
#         mlp_ratio=4.0,
#         qkv_bias=False,
#         qk_scale=None,
#         drop=0.0,
#         attn_drop=0.0,
#         drop_path=0.0,
#         sr_ratio=1,
#         tau=2.0,
#         step_mode='m'):
#         super().__init__()

#         self.attn = MS_Attention_RepConv3D_qkv_id(
#             dim=dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop,
#             sr_ratio=sr_ratio,
#             tau=tau,
#             step_mode=step_mode)

#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = MS_MLP3D(
#             in_features=dim,
#             hidden_features=mlp_hidden_dim,
#             out_features=dim,
#             tau=tau,
#             step_mode=step_mode
#         )

#     def forward(self, x):
#         # x: [T, B, C, D, H, W]
#         # x = x + self.drop_path(self.attn(x))
#         # x = x + self.drop_path(self.mlp(x))
#         x = x + self.attn(x)
#         x = x + self.mlp(x)
#         return x


class MS_DownSampling3D(nn.Module):
    def __init__(
        self,
        in_channels=4,
        embed_dims=96,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        tau=2.0,
        step_mode='m'):
        super().__init__()

        self.encode_conv = layer.Conv3d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            step_mode=step_mode
        )

        self.encode_bn = layer.GroupNorm(num_groups=8, num_channels=embed_dims, step_mode=step_mode)

        self.use_lif = not first_layer
        if self.use_lif:
            self.encode_lif = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)
            # self.encode_lif = neuron.ParametricLIFNode(
            #     init_tau=tau,
            #     decay_input=True,
            #     #detach_reset=True,
            #     v_threshold=1.0,
            #     v_reset=0.0,
            #     surrogate_function=surrogate.ATan(), 
            #     step_mode=step_mode,
            #     backend='torch'
            #     )
            
    def forward(self, x):
        # x: [T, B, C, D, H, W]
        if self.use_lif:
            x = self.encode_lif(x)
        x = self.encode_conv(x)
        x = self.encode_bn(x)
        return x
    
    
class MS_UpSampling3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        last_layer=False,
        tau=2.0,
        step_mode='m'
    ):
        super().__init__()

        self.decode_conv = layer.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True,
            step_mode=step_mode
        )

        self.decode_bn = layer.GroupNorm(num_groups=8, num_channels=out_channels, step_mode=step_mode)

        self.use_lif = not last_layer
        if self.use_lif:
            self.decode_lif = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=step_mode)
            # self.decode_lif = neuron.ParametricLIFNode(
            #     init_tau=tau,
            #     decay_input=True,
            #     #detach_reset=True,
            #     v_threshold=1.0,
            #     v_reset=0.0,
            #     surrogate_function=surrogate.ATan(),
            #     step_mode=step_mode,
            #     backend='torch'
            # )

    def forward(self, x):
        # x: [T, B, C, D, H, W]
        x = self.decode_conv(x)
        x = self.decode_bn(x)
        if self.use_lif:
            x = self.decode_lif(x)
        return x    
    
 
class AddConverge3D(base.MemoryModule):
    def __init__(self, channels, step_mode='m'):
        super().__init__()
        self.norm = layer.GroupNorm(num_groups=8, num_channels=channels, step_mode=step_mode)

    def forward(self, x1, x2):
        x = x1 + x2  # skip connection by addition
        x = self.norm(x)
        return x 
 
    

class Spike_Former_Unet3D(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_classes=3,
        embed_dim=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        depths=[8, 8, 8, 8],
        layers=[2, 2, 6, 2],
        sr_ratios=[8, 4, 2, 1],
        T=4,
        step_mode='m'):
        super().__init__()
        self.T = T

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Encode-Stage 1
        self.downsample1_a = MS_DownSampling3D(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=False,
            step_mode=step_mode)
        
        # self.encode_block1_a = nn.ModuleList([
        #     MS_ConvBlock3D(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios[0], step_mode=step_mode)])

        self.downsample1_b = MS_DownSampling3D(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            step_mode=step_mode)
        
        # self.encode_block1_b = nn.ModuleList([
        #     MS_ConvBlock3D(dim=embed_dim[0], mlp_ratio=mlp_ratios[0], step_mode=step_mode)])

        # Encode-Stage 2
        self.downsample2 = MS_DownSampling3D(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            step_mode=step_mode)
                
        # self.encode_block2_a = nn.ModuleList([
        #     MS_ConvBlock3D(dim=embed_dim[1], mlp_ratio=mlp_ratios[1], step_mode=step_mode)])
       
        # self.encode_block2_b = nn.ModuleList([
        #     MS_ConvBlock3D(dim=embed_dim[1], mlp_ratio=mlp_ratios[1], step_mode=step_mode)])

        # Encode-Stage 3
        self.downsample3 = MS_DownSampling3D(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            step_mode=step_mode)
        
        # self.encode_block3 = nn.ModuleList([
        #     MS_Block3D(
        #         dim=embed_dim[2],
        #         num_heads=num_heads[2],
        #         mlp_ratio=mlp_ratios[2],
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[i],
        #         sr_ratio=sr_ratios[2],
        #         step_mode=step_mode
        #     ) for i in range(layers[2])])

        # feature-Stage
        self.feature_downsample = MS_DownSampling3D(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
            step_mode=step_mode)
        
        # self.feature_block = nn.ModuleList([
        #     MS_Block3D(
        #         dim=embed_dim[3],
        #         num_heads=num_heads[3],
        #         mlp_ratio=mlp_ratios[3],
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[i],
        #         sr_ratio=sr_ratios[3],
        #         step_mode=step_mode
        #     ) for i in range(layers[3])
        # ])
        
        # Decode-Stage 3
        self.upsample3 = MS_UpSampling3D(
            in_channels=embed_dim[3],
            out_channels=embed_dim[2],
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0,
            last_layer=False,
            step_mode=step_mode)
        
        # self.decode_block3 = nn.ModuleList([
        #     MS_Block3D(
        #         dim=embed_dim[2],
        #         num_heads=num_heads[2],
        #         mlp_ratio=mlp_ratios[2],
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[i],
        #         sr_ratio=sr_ratios[2],
        #         step_mode=step_mode
        #     ) for i in range(layers[2])])

        
        self.converge3 = AddConverge3D(channels=embed_dim[2], step_mode=step_mode)        
        
        # Decode-Stage 2
        self.upsample2 = MS_UpSampling3D(
            in_channels=embed_dim[2],
            out_channels=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            last_layer=False,
            step_mode=step_mode)
                
        # self.decode_block2_a = nn.ModuleList([
        #     MS_ConvBlock3D(dim=embed_dim[1], mlp_ratio=mlp_ratios[1], step_mode=step_mode)])
       
        # self.decode_block2_b = nn.ModuleList([
        #     MS_ConvBlock3D(dim=embed_dim[1], mlp_ratio=mlp_ratios[1], step_mode=step_mode)])
        
        self.converge2 = AddConverge3D(channels=embed_dim[1], step_mode=step_mode)  
                   
        # Decode-Stage 1
        self.upsample1_b = MS_UpSampling3D(
            in_channels=embed_dim[1],
            out_channels= embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            last_layer=False,
            step_mode=step_mode)
        
        # self.decode_block1_b = nn.ModuleList([
        #     MS_ConvBlock3D(dim=embed_dim[0], mlp_ratio=mlp_ratios[0], step_mode=step_mode)])
        
        self.upsample1_a = MS_UpSampling3D(
            in_channels=embed_dim[0],
            out_channels=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            last_layer=False,
            step_mode=step_mode)
        
        # self.decode_block1_a = nn.ModuleList([
        #     MS_ConvBlock3D(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios[0], step_mode=step_mode)])


        self.converge1 = AddConverge3D(channels=embed_dim[0], step_mode=step_mode)  
        
        self.final_upsample = MS_UpSampling3D(
            in_channels=embed_dim[0] // 2,
            out_channels=embed_dim[0] // 4,
            kernel_size=3,
            stride=2,
            padding=1,
            last_layer=True,
            step_mode=step_mode)
        
        
        self.lif = neuron.ParametricLIFNode(
            init_tau=2.0,
            decay_input=True,
            detach_reset=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), 
            step_mode=step_mode,
            backend='torch')
        
        self.readout = layer.Conv3d(embed_dim[0] // 4, num_classes, kernel_size=1, step_mode=step_mode)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def print_mem(self, name, x):
        size_MB = x.numel() * x.element_size() / 1024 ** 2
        print(f"[{name}] shape: {x.shape}, mem: {size_MB:.2f} MB")
        return x


    def forward_encoder_decoder(self, x):         # input shape: [T, B, 4, 128, 128, 128]
        # Encode-stage 1
        e1 = self.downsample1_a(x)          # Downsample1_a output shape: [T, B, 48, 64, 64, 64]
        # for blk in self.encode_block1_a:
        #     e1 = blk(e1)                     # shape: [T, B, 48, 64, 64, 64]
        
        e1 = self.downsample1_b(e1)          # Downsample1_b output shape: [T, B, 96, 32, 32, 32]
        # for blk in self.encode_block1_b:
        #     e1 = blk(e1)
        #self.print_mem("After-Encode-Stage 1", e1) 
        skip1 = e1                 # Skip2 shape: [T, B, 96, 32, 32, 32]
        #self.print_mem("Skip1", skip1)
        # Encode-stage 2
        e2 = self.downsample2(e1)            # Downsample2 output shape: [T, B, 192, 16, 16, 16]
        # for blk in self.encode_block2_a:
        #     e2 = blk(e2)
        # for blk in self.encode_block2_b:
        #     e2 = blk(e2)
        #self.print_mem("After-Encode-Stage 2", e2)
        skip2 = e2                  # Skip3 shape: [T, B, 192, 16, 16, 16]
        # self.print_mem("Skip2", skip2)
        # # Encode-stage 3
        # e3 = self.downsample3(e2)            # Downsample3 output shape: [T, B, 384, 8, 8, 8]
        # # for blk in self.encode_block3:
        # #     e3 = blk(e3)
        # self.print_mem("After-Encode-Stage 3", e3)
        # skip3 = e3                  # Skip4 shape: [T, B, 384, 8, 8, 8]
        # self.print_mem("Skip3", skip3)
        # # Encode-stage 4
        # e4 = self.feature_downsample(e3)     # Downsample4 output shape: [T, B, 480, 8, 8, 8]
        # # for blk in self.feature_block:
        # #     e4 = blk(e4)                     # After Encode-Stage 4: [T, B, 480, 8, 8, 8]
        # self.print_mem("After-Encode-Stage 4", e4)
        
        # # Decode-Stage 3
        # d3 = self.upsample3(e4)              # Upsample3 output shape: [T, B, 384, 8, 8, 8]
        # d3 = self.converge3(d3, skip3)       # converge3 output shape: [T, B, 384, 8, 8, 8]
        # # for blk in self.decode_block3:
        # #     d3 = blk(d3)                     # After Decode-Stage3: [T, B, 384, 8, 8, 8]
        # self.print_mem("After-Decode-Stage 3", d3)
        
        # # Decode-Stage 2
        # d2 = self.upsample2(d3)              # Upsample2 output shape: [T, B, 192, 16, 16, 16]
        # d2 = self.converge2(d2, skip2)       # Converge2 output shape: [T, B, 192, 16, 16, 16]
        # # for blk in self.decode_block2_a:
        # #     d2 = blk(d2)
        # # for blk in self.decode_block2_b:
        # #     d2 = blk(d2)                     # After Decode-Stage2: [T, B, 192, 16, 16, 16]
        # self.print_mem("After-Decode-Stage 2", d2)

        # Decode-Stage 1
        d1 = self.upsample1_b(e2)            # Upsample1_b output shape: [T, B, 96, 32, 32, 32]
        d1 = self.converge1(d1, skip1)       # Converge1 output shape: [T, B, 96, 32, 32, 32]
        # for blk in self.decode_block1_b:
        #     d1 = blk(d1)
        
        d1 = self.upsample1_a(d1)            # Upsample1_a output shape: [T, B, 48, 64, 64, 64]
        # for blk in self.decode_block1_a:
        #     d1 = blk(d1)
            
        out =self.final_upsample(d1)          # Final Upsample output shape: [T, B, 24, 128, 128, 128]
        #self.print_mem("After-Final-Upsample", out)

        return out

    def forward(self, x):
        out = self.forward_encoder_decoder(x)

        # Readout
        # out_lif = self.lif(out) 
        output = self.readout(out).mean(0)  # [3, 128, 128, 128]

        return output


    
def spike_former_unet3D_8_384(in_channels=4, num_classes=3, T=4, step_mode='m',**kwargs):
    model = Spike_Former_Unet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=[96, 192, 384, 480],
        num_heads=[8, 8, 8, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        depths=[8, 8, 8, 8],
        sr_ratios=[1, 1, 1, 1],
        T=T,
        step_mode=step_mode,
        **kwargs
    )
    return model


def spike_former_unet3D_8_512(in_channels=4, num_classes=3, T=4, step_mode='m',**kwargs):
    model = Spike_Former_Unet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=[128, 256, 512, 640],
        num_heads=[8, 8, 8, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        depths=[8, 8, 8, 8],
        sr_ratios=[1, 1, 1, 1],
        T=T,
        step_mode=step_mode,
        **kwargs,
    )
    return model


def spike_former_unet3D_8_768(in_channels=4, num_classes=3, T=4, step_mode='m',**kwargs):
    model = Spike_Former_Unet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=[192, 384, 768, 960],
        num_heads=[8, 8, 8, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        depths=[8, 8, 8, 8],
        sr_ratios=[1, 1, 1, 1],
        T=T,
        step_mode=step_mode,
        **kwargs,
    )
    return model    
    
    
        
def main():
    # 测试模型
    device = torch.device("cuda:1")
    model = spike_former_unet3D_8_384().to(device)
    x = torch.randn(2, 4, 128, 128, 128)  # 假设输入是一个 batch 的数据
    x = x.to(device)
    output = model(x)
    print(output.shape)  # 输出形状应该是 [1, 3, 128, 128, 128]
    
if __name__ == "__main__":
    main()