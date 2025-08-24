import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from spikingjelly.activation_based import neuron, functional, surrogate, layer, base
from timm.layers import trunc_normal_, DropPath

class GeneralParametricLIFNode(neuron.BaseNode):
    def __init__(self,
                 init_tau: float = 2.0,
                 init_threshold: float = 1.0,
                 learnable_tau: bool = True,
                 learnable_threshold: bool = False,
                 decay_input: bool = True,
                 v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False,
                 step_mode='s',
                 backend='cupy',
                 store_v_seq: bool = False):

        super().__init__(v_threshold=0., 
                         v_reset=v_reset,
                         surrogate_function=surrogate_function,
                         detach_reset=detach_reset,
                         step_mode=step_mode,
                         backend=backend,
                         store_v_seq=store_v_seq)

        self.decay_input = decay_input

        # τ = 1 / sigmoid(w)
        init_w = -math.log(init_tau - 1.)
        w_tensor = torch.tensor(init_w, dtype=torch.float)
        if learnable_tau:
            self.w = nn.Parameter(w_tensor)
        else:
            self.register_buffer('w', w_tensor)

        # 可学习 or 固定 threshold
        threshold_tensor = torch.tensor(init_threshold, dtype=torch.float)
        if learnable_threshold:
            self.v_threshold = nn.Parameter(threshold_tensor)
        else:
            self.register_buffer('v_threshold', threshold_tensor)

    @property
    def supported_backends(self):
        return ('torch',) if self.step_mode == 's' else ('torch', 'cupy')

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()
        return super().extra_repr() + \
               f', tau={tau.item():.4f}, learnable_tau={isinstance(self.w, nn.Parameter)}, ' \
               f'threshold={self.v_threshold.item():.4f}, learnable_threshold={isinstance(self.v_threshold, nn.Parameter)}'

    def neuronal_charge(self, x: torch.Tensor):
        tau_inv = self.w.sigmoid()

        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) * tau_inv
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * tau_inv
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - tau_inv) + x
            else:
                self.v = self.v - (self.v - self.v_reset) * tau_inv + x

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)
    
    

class NormAndPad3DLayer(nn.Module):
    def __init__(self, pad_voxels, num_features, norm_type='group', step_mode='m', num_groups=8, **norm_kwargs):
        """
        支持 BatchNorm3d 和 GroupNorm 的 3D 归一化 + Padding 组合模块。

        :param pad_voxels: int，六个方向填充体素数
        :param num_features: 通道数
        :param norm_type: 'batch' 或 'group'
        :param num_groups: GroupNorm 的组数（norm_type='group' 时生效）
        :param norm_kwargs: BatchNorm3d 的参数（如 eps、momentum）
        """
        super().__init__()
        self.pad_voxels = pad_voxels
        self.step_mode = step_mode
        self.norm_type = norm_type

        if norm_type == 'batch':
            self.norm = layer.BatchNorm3d(num_features, step_mode=step_mode, **norm_kwargs)
        elif norm_type == 'group':
            self.norm = layer.GroupNorm(num_groups=num_groups, num_channels=num_features, step_mode=step_mode)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def _compute_pad_value(self):
        if self.norm_type == 'batch':
            if self.norm.affine:
                pad_value = (
                    self.norm.bias.detach()
                    - self.norm.running_mean * self.norm.weight.detach()
                      / torch.sqrt(self.norm.running_var + self.norm.eps)
                )
            else:
                pad_value = -self.norm.running_mean / torch.sqrt(self.norm.running_var + self.norm.eps)
        elif self.norm_type == 'group':
            if self.norm.affine:
                pad_value = self.norm.bias.detach()
            else:
                pad_value = torch.zeros(self.norm.num_channels, device=self.norm.weight.device)
        else:
            raise NotImplementedError

        return pad_value.view(1, -1, 1, 1, 1)  # [1, C, 1, 1, 1]

    def _pad_tensor(self, x, pad_value):
        pad = [self.pad_voxels] * 6  # [W_l, W_r, H_t, H_b, D_f, D_b]
        x = F.pad(x, pad)  # 先 0 填充
        # 替换成 pad_value
        x[:, :, :self.pad_voxels, :, :] = pad_value
        x[:, :, -self.pad_voxels:, :, :] = pad_value
        x[:, :, :, :self.pad_voxels, :] = pad_value
        x[:, :, :, -self.pad_voxels:, :] = pad_value
        x[:, :, :, :, :self.pad_voxels] = pad_value
        x[:, :, :, :, -self.pad_voxels:] = pad_value
        return x
    
    def _pad_tensor_batch(self, x, pad_value):
        # 输入x形状: [T, N, C, D, H, W]
        T, N, C, D, H, W = x.shape

        # 将时间和批量维度合并，方便一次pad
        x = x.view(T * N, C, D, H, W)

        pad = [self.pad_voxels] * 6  # pad格式：左右、上下、前后

        x = F.pad(x, pad)  # 先用0填充

        # 用计算的pad_value替换边缘voxels
        x[:, :, :self.pad_voxels, :, :] = pad_value  # D轴前面
        x[:, :, -self.pad_voxels:, :, :] = pad_value  # D轴后面
        x[:, :, :, :self.pad_voxels, :] = pad_value  # H轴上面
        x[:, :, :, -self.pad_voxels:, :] = pad_value  # H轴下面
        x[:, :, :, :, :self.pad_voxels] = pad_value  # W轴左边
        x[:, :, :, :, -self.pad_voxels:] = pad_value  # W轴右边

        # 恢复成原始6维形状，包含pad后尺寸
        x = x.view(T, N, C, D + 2 * self.pad_voxels, H + 2 * self.pad_voxels, W + 2 * self.pad_voxels)
        return x

    def forward(self, x):
        if self.step_mode == 's':
            x = self.norm(x)  # shape: [N, C, D, H, W]
            if self.pad_voxels > 0:
                pad_value = self._compute_pad_value()
                x = self._pad_tensor(x, pad_value)
            return x

        elif self.step_mode == 'm':
            if x.dim() != 6:
                raise ValueError(f"Expected input shape [T, N, C, D, H, W], but got {x.shape}")
            x = self.norm(x)
            if self.pad_voxels > 0:
                pad_value = self._compute_pad_value()
                x = self._pad_tensor_batch(x, pad_value)
            return x

    @property
    def weight(self):
        return self.norm.weight

    @property
    def bias(self):
        return self.norm.bias

    @property
    def eps(self):
        return self.norm.eps if hasattr(self.norm, 'eps') else 1e-5



class RepConv3D(nn.Module):
    def __init__(self, in_channel, out_channel, pad_voxels=1, norm_type='group', bias=False, step_mode='m'):
        super().__init__()

        # 1x1 projection conv
        self.proj_conv = layer.Conv3d(in_channels=in_channel, out_channels=in_channel,
                                      kernel_size=1, stride=1, padding=0, bias=bias, step_mode=step_mode)

        # Norm + Padding
        self.norm_pad = NormAndPad3DLayer(pad_voxels=pad_voxels, num_features=in_channel, step_mode=step_mode, norm_type=norm_type)

        # Depthwise 3x3 conv
        self.dw_conv3x3 = layer.Conv3d(in_channels=in_channel, out_channels=in_channel,
                                       kernel_size=3, stride=1, padding=0, bias=bias,
                                       groups=in_channel, step_mode=step_mode)

        # Pointwise 1x1 conv
        self.pw_conv1x1 = layer.Conv3d(in_channels=in_channel, out_channels=out_channel,
                                       kernel_size=1, stride=1, padding=0, bias=bias,
                                       step_mode=step_mode)

        # Output Norm
        if norm_type == 'batch':
            self.out_norm = layer.BatchNorm3d(num_features=out_channel, step_mode=step_mode)
        elif norm_type == 'group':
            self.out_norm = layer.GroupNorm(num_groups=8, num_channels=out_channel, step_mode=step_mode)

    def forward(self, x):  
        x = self.proj_conv(x)          # 1×1 conv
        x = self.norm_pad(x)             # Norm + padding
        x = self.dw_conv3x3(x)         # depthwise 3×3 conv
        x = self.pw_conv1x1(x)         # pointwise 1×1 conv
        x = self.out_norm(x)             # output Norm
        return x


class SepConv3D(nn.Module):
    """
    Spiking 3D version of inverted separable convolution from MobileNetV2.
    Input: [T, B, C, D, H, W]
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        kernel_size=7,
        padding=3,
        tau=2.0,
        lif_type='para_lif',
        step_mode='m',
        norm_type='group',
        bias=False):
        super().__init__()
        med_channels = int(expansion_ratio * dim)

        # spike layer 1
        if lif_type == 'lif':
            self.lif1 = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
        elif lif_type == 'para_lif':
            self.lif1 = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode,
                backend='cupy'
                )
        elif lif_type == 'general_para_lif':
            self.lif1 = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )

        # pointwise conv 1
        self.pwconv1 = layer.Conv3d(dim, med_channels, kernel_size=1, stride=1,
                                    bias=bias, step_mode=step_mode)
        
        # norm layer 1
        if norm_type == 'batch':
            self.norm1 = layer.BatchNorm3d(med_channels, step_mode=step_mode)
        elif norm_type == 'group':
            self.norm1 = layer.GroupNorm(num_groups=8, num_channels=med_channels, step_mode=step_mode)
        

        # spike layer 2        
        if lif_type == 'lif':
            self.lif2 = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
        elif lif_type == 'para_lif':
            self.lif2 = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode,
                backend='cupy'
                )
        elif lif_type == 'general_para_lif':
            self.lif2 = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )

        # depthwise conv
        self.dwconv = layer.Conv3d(med_channels, med_channels, kernel_size=kernel_size,
                                   padding=padding, groups=med_channels,
                                   bias=bias, step_mode=step_mode)

        # pointwise conv 2
        self.pwconv2 = layer.Conv3d(med_channels, dim, kernel_size=1, stride=1,
                                    bias=bias, step_mode=step_mode)
        
        # norm layer 2
        if norm_type == 'batch':
            self.norm2 = layer.BatchNorm3d(dim, step_mode=step_mode)
        elif norm_type == 'group':
            self.norm2 = layer.GroupNorm(num_groups=8, num_channels=dim, step_mode=step_mode)

    def forward(self, x):
        # x: [T, B, C, D, H, W]
        x = self.lif1(x)
        x = self.pwconv1(x)
        x = self.norm1(x)
        x = self.lif2(x)
        x = self.dwconv(x)
        x = self.pwconv2(x)
        x = self.norm2(x)
        return x


class MS_SpikeConvBlock3D(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        tau=2.0,
        lif_type='para_lif',
        norm_type='group',
        step_mode='m'):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.sep_conv = SepConv3D(dim=dim, step_mode=step_mode)

        if lif_type == 'lif':
            self.lif1 = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
        elif lif_type == 'para_lif':
            self.lif1 = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode,
                backend='cupy'
                )
        elif lif_type == 'general_para_lif':
            self.lif1 = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )
        
        self.conv1 = layer.Conv3d(in_channels=dim, out_channels=hidden_dim,
                                  kernel_size=3, padding=1, bias=False,
                                  step_mode=step_mode)
        
        if norm_type == 'batch':
            self.norm1 = layer.BatchNorm3d(num_features=hidden_dim, step_mode=step_mode)
        elif norm_type == 'group':
            self.norm1 = layer.GroupNorm(num_groups=8, num_channels=hidden_dim, step_mode=step_mode)

        # Spike + Conv + Norm block2
        if lif_type == 'lif':
            self.lif2 = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
        elif lif_type == 'para_lif':
            self.lif2 = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode,
                backend='cupy'
                )
        elif lif_type == 'general_para_lif':
            self.lif2 = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )
        
        self.conv2 = layer.Conv3d(in_channels=hidden_dim, out_channels=dim,
                                  kernel_size=3, padding=1, bias=False,
                                  step_mode=step_mode)
        
        if norm_type == 'batch':
            self.norm2 = layer.BatchNorm3d(num_features=dim, step_mode=step_mode)
        elif norm_type == 'group':
            self.norm2 = layer.GroupNorm(num_groups=8, num_channels=dim, step_mode=step_mode)

    def forward(self, x):
        # x: [T, B, C, D, H, W]

        # Branch 1: Lightweight convolution block + residual
        x = self.sep_conv(x) + x
        x_feat = x

        # Branch 2: MLP-like spike conv
        x = self.lif1(x)
        x = self.conv1(x)
        x = self.norm1(x)

        x = self.lif2(x)
        x = self.conv2(x)
        x = self.norm2(x)

        # Final residual
        x = x_feat + x

        return x


class MS_SpikeMLP3D(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        tau=2.0,
        lif_type='para_lif',
        norm_type='group',
        step_mode='m'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 用1x1x1卷积实现“全连接”操作
        self.fc1_conv = layer.Conv3d(in_features, hidden_features, kernel_size=1,
                                    stride=1, padding=0, bias=False, step_mode=step_mode)
        
        if norm_type == 'batch':
            self.fc1_norm = layer.BatchNorm3d(hidden_features, step_mode=step_mode)
        elif norm_type == 'group':
            self.fc1_norm = layer.GroupNorm(num_groups=8, num_channels=hidden_features, step_mode=step_mode)
            
        if lif_type == 'lif':    
            self.fc1_lif = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
        elif lif_type == 'para_lif':
            self.fc1_lif = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode,
                backend='cupy'
                )
        elif lif_type == 'general_para_lif':
            self.fc1_lif = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )

        self.fc2_conv = layer.Conv3d(hidden_features, out_features, kernel_size=1,
                                    stride=1, padding=0, bias=False, step_mode=step_mode)
        if norm_type == 'batch':
            self.fc2_norm = layer.BatchNorm3d(out_features, step_mode=step_mode)
        elif norm_type == 'group':
            self.fc2_norm = layer.GroupNorm(num_groups=8, num_channels=out_features, step_mode=step_mode)
 
        if lif_type == 'lif':           
            self.fc2_lif = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
        elif lif_type == 'para_lif':
            self.fc2_lif = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode,
                backend='cupy'
                )
        elif lif_type == 'general_para_lif':
            self.fc2_lif = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )

    def forward(self, x):
        # x: [T, B, C, D, H, W]
        x = self.fc1_lif(x)
        x = self.fc1_conv(x)
        x = self.fc1_norm(x)

        x = self.fc2_lif(x)
        x = self.fc2_conv(x)
        x = self.fc2_norm(x)
        return x


class MS_SpikeAttention_RepConv3D_qkv_id(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        tau=2.0,
        lif_type='para_lif',
        norm_type='group',
        step_mode='m'):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125 if qk_scale is None else qk_scale

        if lif_type == 'lif':
            self.head_lif = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
        elif lif_type == 'para_lif':
            self.head_lif = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode,
                backend='cupy'
                )
        elif lif_type == 'general_para_lif':
            self.head_lif = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )
        
        if norm_type == 'batch':
            q_norm = layer.BatchNorm3d(dim, step_mode=step_mode)
            k_norm = layer.BatchNorm3d(dim, step_mode=step_mode)
            v_norm = layer.BatchNorm3d(dim, step_mode=step_mode)
            proj_norm = layer.BatchNorm3d(dim, step_mode=step_mode)
        elif norm_type == 'group':
            q_norm = layer.GroupNorm(num_groups=8, num_channels=dim, step_mode=step_mode)
            k_norm = layer.GroupNorm(num_groups=8, num_channels=dim, step_mode=step_mode)
            v_norm = layer.GroupNorm(num_groups=8, num_channels=dim, step_mode=step_mode)
            proj_norm = layer.GroupNorm(num_groups=8, num_channels=dim, step_mode=step_mode)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.q_conv = nn.Sequential(
            RepConv3D(dim, dim, bias=False, step_mode=step_mode),
            q_norm)
        self.k_conv = nn.Sequential(
            RepConv3D(dim, dim, bias=False, step_mode=step_mode),
            k_norm)
        self.v_conv = nn.Sequential(
            RepConv3D(dim, dim, bias=False, step_mode=step_mode),
            v_norm)

        if lif_type == 'lif':
            self.q_lif = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
            
            self.k_lif = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )           
            
            self.v_lif = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
            
            self.attn_lif = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
                                   
        elif lif_type == 'para_lif':
            self.q_lif = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode,
                backend='cupy'
                )
        
            self.k_lif = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode,
                backend='cupy'
                )
        
            self.v_lif = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode,
                backend='cupy'
                )

            self.attn_lif = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=0.5,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode,
                backend='cupy'
                )
        elif lif_type == 'general_para_lif':
            self.q_lif = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )
            self.k_lif = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=False,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )
            self.v_lif = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=False,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )
            self.attn_lif = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=0.5,
                learnable_tau=False,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )

        self.proj_conv = nn.Sequential(
            RepConv3D(dim, dim, bias=False, step_mode=step_mode),
            proj_norm)

    def forward(self, x):
        # x: [T, B, C, D, H, W]
        T, B, C, D, H, W = x.shape
        N = D * H * W

        x = self.head_lif(x)
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        # LIF激活 + 拉平空间维到 N
        q = self.q_lif(q).flatten(3)  # [T, B, C, N]
        q = q.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 1, 3, 2, 4).contiguous()  # [T, B, heads, N, head_dim]

        k = self.k_lif(k).flatten(3)
        k = k.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
        k = k.permute(0, 1, 3, 2, 4).contiguous()

        v = self.v_lif(v).flatten(3)
        v = v.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
        v = v.permute(0, 1, 3, 2, 4).contiguous()

        # 注意力计算：k^T @ v
        attn_kv = torch.matmul(k.transpose(-2, -1), v)  # [T, B, heads, head_dim, head_dim]
        x = torch.matmul(q, attn_kv) * self.scale  # [T, B, heads, N, head_dim]

        # 恢复空间维度
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, D, H, W)

        # 投影卷积
        x = self.proj_conv(x)

        return x


class MS_SpikeTransformerBlock3D(nn.Module):
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
        sr_ratio=1,
        tau=2.0,
        lif_type='para_lif',
        norm_type='group',
        step_mode='m'):
        super().__init__()

        self.attn = MS_SpikeAttention_RepConv3D_qkv_id(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            tau=tau,
            lif_type=lif_type,
            norm_type=norm_type,
            step_mode=step_mode)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_SpikeMLP3D(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            tau=tau,
            lif_type=lif_type,
            norm_type=norm_type,
            step_mode=step_mode
        )

    def forward(self, x):
        # x: [T, B, C, D, H, W]
        # x = x + self.drop_path(self.attn(x))
        # x = x + self.drop_path(self.mlp(x))
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x



class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):  # [T, B, ...]
        T, B = x.shape[:2]
        x = x.view(T * B, *x.shape[2:])
        x = self.module(x)
        x = x.view(T, B, *x.shape[1:])
        return x

class MS_SpikeDownSampling3D(nn.Module):
    def __init__(
        self,
        in_channels=4,
        embed_dims=96,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        tau=2.0,
        lif_type='para_lif',
        norm_type='group',
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

        if norm_type == 'batch':
            self.encode_norm = layer.BatchNorm3d(num_features=embed_dims, step_mode=step_mode)
        elif norm_type == 'group':
            self.encode_norm = layer.GroupNorm(num_groups=8, num_channels=embed_dims, step_mode=step_mode)

        # self.relu = TimeDistributed(nn.ReLU())
        self.use_lif = not first_layer
        if self.use_lif:
            if lif_type == 'lif':
                self.encode_lif = neuron.LIFNode(
                    tau=tau,
                    decay_input=True,
                    detach_reset=True,
                    v_threshold=1.0,
                    v_reset=0.0,
                    surrogate_function=surrogate.ATan(), 
                    step_mode=step_mode
                )
            elif lif_type == 'para_lif':
                self.encode_lif = neuron.ParametricLIFNode(
                    init_tau=tau,
                    decay_input=True,
                    detach_reset=True,
                    v_threshold=1.0,
                    v_reset=0.0,
                    surrogate_function=surrogate.ATan(), 
                    step_mode=step_mode,
                    backend='cupy'
                    )
            elif lif_type == 'general_para_lif':
                self.encode_lif = GeneralParametricLIFNode(
                    init_tau=tau,
                    init_threshold=1.0,
                    learnable_tau=True,
                    learnable_threshold=True,
                    decay_input=True,
                    detach_reset=True,                
                    v_reset=0.0,
                    surrogate_function=surrogate.ATan(), # surrogate.ATan()
                    step_mode=step_mode,
                    backend='cupy'
                )            
            
    def forward(self, x):
        # x: [T, B, C, D, H, W]
        if self.use_lif:
            x = self.encode_lif(x)
        x = self.encode_conv(x)
        x = self.encode_norm(x)
        return x
    
    
class MS_SpikeUpSampling3D(nn.Module):
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
        lif_type='para_lif',
        norm_type='group',
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

        if norm_type == 'batch':
            self.decode_norm = layer.BatchNorm3d(num_features=out_channels, step_mode=step_mode)
        elif norm_type == 'group':
            self.decode_norm = layer.GroupNorm(num_groups=8, num_channels=out_channels, step_mode=step_mode)

        self.use_lif = not last_layer
        if self.use_lif:
            if lif_type == 'lif':
                self.decode_lif = neuron.LIFNode(
                    tau=tau,
                    decay_input=True,
                    detach_reset=True,
                    v_threshold=1.0,
                    v_reset=0.0,
                    surrogate_function=surrogate.ATan(), 
                    step_mode=step_mode
                )
            elif lif_type == 'para_lif':
                self.decode_lif = neuron.ParametricLIFNode(
                    init_tau=tau,
                    decay_input=True,
                    detach_reset=True,
                    v_threshold=1.0,
                    v_reset=0.0,
                    surrogate_function=surrogate.ATan(),
                    step_mode=step_mode,
                    backend='cupy'
                )
            elif lif_type == 'general_para_lif':
                self.decode_lif = GeneralParametricLIFNode(
                    init_tau=tau,
                    init_threshold=1.0,
                    learnable_tau=True,
                    learnable_threshold=True,
                    decay_input=True,
                    detach_reset=True,                
                    v_reset=0.0,
                    surrogate_function=surrogate.ATan(), # surrogate.ATan()
                    step_mode=step_mode,
                    backend='cupy'
                )     

    def forward(self, x):
        # x: [T, B, C, D, H, W]
        if self.use_lif:
            x = self.decode_lif(x)
        x = self.decode_conv(x)
        x = self.decode_norm(x)
        return x    
    
 
# class AddConverge3D(base.MemoryModule):
#     def __init__(self, channels, norm_type='group', tau=2.0, lif_type='para_lif', step_mode='m'):
#         super().__init__()
#         if lif_type == 'lif':
#             self.lif = neuron.LIFNode(
#                 tau=tau,
#                 decay_input=True,
#                 detach_reset=True,
#                 v_threshold=1.0,
#                 v_reset=0.0,
#                 surrogate_function=surrogate.ATan(), 
#                 step_mode=step_mode
#             )
#         elif lif_type == 'para_lif':
#             self.lif = neuron.ParametricLIFNode(
#                 init_tau=tau,
#                 decay_input=True,
#                 detach_reset=True,
#                 v_threshold=1.0,
#                 v_reset=0.0,
#                 surrogate_function=surrogate.ATan(),
#                 step_mode=step_mode,
#                 backend='cupy'
#             )
#         elif lif_type == 'general_para_lif':
#             self.lif = GeneralParametricLIFNode(
#                 init_tau=tau,
#                 init_threshold=1.0,
#                 learnable_tau=True,
#                 learnable_threshold=True,
#                 decay_input=True,
#                 detach_reset=True,                
#                 v_reset=0.0,
#                 surrogate_function=surrogate.ATan(), # surrogate.ATan()
#                 step_mode=step_mode,
#                 backend='cupy'
#             )

#         if norm_type == 'batch':
#             self.norm = layer.BatchNorm3d(num_features=channels, step_mode=step_mode)
#         elif norm_type == 'group':
#             self.norm = layer.GroupNorm(num_groups=8, num_channels=channels, step_mode=step_mode)

#     def forward(self, x1, x2):
#         x = x1 + x2  # skip connection by addition
#         x = self.lif(x)
#         x = self.norm(x)
#         return x 
    
class AddConverge3D(base.MemoryModule):
    def __init__(self, channels, norm_type='group', tau=2.0, lif_type='para_lif', step_mode='m'):
        super().__init__()
        if lif_type == 'lif':
            self.lif1 = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
            self.lif2 = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
        elif lif_type == 'para_lif':
            self.lif1 = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode,
                backend='cupy'
            )
            self.lif2 = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode,
                backend='cupy'
            )
        elif lif_type == 'general_para_lif':
            self.lif1 = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )
            self.lif2 = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )      

        if norm_type == 'batch':
            self.norm = layer.BatchNorm3d(num_features=channels, step_mode=step_mode)
        elif norm_type == 'group':
            self.norm = layer.GroupNorm(num_groups=8, num_channels=channels, step_mode=step_mode)

    def forward(self, x1, x2):
        x1 = self.lif1(x1)
        x2 = self.lif2(x2)
        x = x1 + x2  # skip connection by addition
        x = self.norm(x)
        return x 
    
    
class MS_SpikeCatConverge3D(base.MemoryModule):
    def __init__(self, channels, norm_type='group', tau=2.0, lif_type='para_lif', step_mode='m'):
        super().__init__()
        if lif_type == 'lif':
            self.lif1 = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
            self.lif2 = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
        elif lif_type == 'para_lif':
            self.lif1 = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode,
                backend='cupy'
            )
            self.lif2 = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode,
                backend='cupy'
            )
        elif lif_type == 'general_para_lif':
            self.lif1 = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )
            self.lif2 = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )            
            
        self.conv = layer.Conv3d(channels*2, channels, kernel_size=1, step_mode=step_mode)
        
        if norm_type == 'batch':
            self.norm = layer.BatchNorm3d(num_features=channels, step_mode=step_mode)
        elif norm_type == 'group':
            self.norm = layer.GroupNorm(num_groups=8, num_channels=channels, step_mode=step_mode)

    def forward(self, x1, x2):
        x1 = self.lif1(x1)
        x2 = self.lif2(x2)
        x = torch.cat((x1, x2), dim=2)
        x = self.conv(x)
        x = self.norm(x)
        return x 
    
    
class Gated_SpikeConverge3D(base.MemoryModule):
    def __init__(self, channels, norm_type='group', tau=2.0, lif_type='para_lif', step_mode='m'):
        super().__init__()

        # ----------------- LIF 单元 -----------------
        if lif_type == 'lif':
            self.lif1 = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
            self.lif2 = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), 
                step_mode=step_mode
            )
        elif lif_type == 'para_lif':
            self.lif1 = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode,
                backend='cupy'
            )
            self.lif2 = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode,
                backend='cupy'
            )
        elif lif_type == 'general_para_lif':
            self.lif1 = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )
            self.lif2 = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )    

        # ----------------- 门控网络 -----------------
        # 输入: cat(x1,x2)，通道=2C；输出: 通道=C 的 gate
        self.gate_conv = layer.Conv3d(channels * 2, channels, kernel_size=1, step_mode=step_mode)

            
        if norm_type == 'batch':
            self.norm = layer.BatchNorm3d(num_features=channels, step_mode=step_mode)
        elif norm_type == 'group':
            self.norm = layer.GroupNorm(num_groups=8, num_channels=channels, step_mode=step_mode)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x1, x2):
        # spike 激活
        x1 = self.lif1(x1)
        x2 = self.lif2(x2)

        # 计算 gate
        gate_input = torch.cat((x1, x2), dim=2)     # [B, 2C, D, H, W]
        g = self.gate_conv(gate_input)              # [B, C, D, H, W]
        g = self.norm(g)
        g = self.sigmoid(g)                         # [0,1] 门控系数

        # 门控融合
        out = g * x1 + (1 - g) * x2
        return out


class MS_SpikeAttentionConverge3D(base.MemoryModule):
    def __init__(self, channels, norm_type='group', tau=2.0, lif_type='para_lif', step_mode='m'):
        super().__init__()

        # ----------------- LIF 激活 -----------------
        if lif_type == 'lif':
            self.lif1 = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode
            )
            self.lif2 = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode
            )
            self.att_lif = neuron.LIFNode(
                tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode
            )
        elif lif_type == 'para_lif':
            self.lif1 = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode,
                backend='cupy'
            )
            self.lif2 = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode,
                backend='cupy'
            )
            self.att_lif = neuron.ParametricLIFNode(
                init_tau=tau,
                decay_input=True,
                detach_reset=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode,
                backend='cupy'
            )
        elif lif_type == 'general_para_lif':
            self.lif1 = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )
            self.lif2 = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )
            self.att_lif = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )    

        # ----------------- 融合卷积 -----------------
        self.conv = layer.Conv3d(channels, channels, kernel_size=1, step_mode=step_mode)

        # ----------------- 通道注意力 -----------------
        self.ca_fc1 = layer.Conv3d(channels * 2, channels, kernel_size=1, step_mode=step_mode)
        self.ca_fc2 = layer.Conv3d(channels, channels, kernel_size=1, step_mode=step_mode)

        # ----------------- 空间注意力 -----------------
        self.sa_conv = layer.Conv3d(1, 1, kernel_size=7, padding=3, step_mode=step_mode)  # 输出1通道用于广播

        # ----------------- 归一化 -----------------
        if norm_type == 'batch':
            self.norm = layer.BatchNorm3d(num_features=channels, step_mode=step_mode)
        elif norm_type == 'group':
            self.norm = layer.GroupNorm(num_groups=8, num_channels=channels, step_mode=step_mode)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x1, x2):
        """
        x1, x2: [T, B, C, D, H, W]
        """
        # ----------------- LIF 激活 -----------------        
        x1 = self.lif1(x1)
        x2 = self.lif2(x2)

        # ----------------- 通道注意力 -----------------
        ca = torch.cat([x1, x2], dim=2)      # [T,B,2C,D,H,W]
        ca = self.ca_fc1(ca)                 # [T,B,C//2,D,H,W]
        ca = self.att_lif(ca)
        ca = self.ca_fc2(ca)                 # [T,B,C,D,H,W]
        ca = self.sigmoid(ca)                # [0,1] 门控系数
        out_ca = ca * x1 + (1 - ca) * x2        # 通道融合

        # ----------------- 空间注意力（广播版本） -----------------
        sa = out_ca.mean(2, keepdim=True)  # [T,B,1,D,H,W]
        sa = self.sa_conv(sa)               # [T,B,1,D,H,W]
        sa = self.att_lif(sa)
        sa = self.sigmoid(sa)
        out = sa * out_ca + (1 - sa) * out_ca    # 广播到所有通道

        # ----------------- 融合卷积 + 归一化 -----------------
        out = self.conv(out)
        out = self.norm(out)
        return out

 
    
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
        skip_connection='cat',
        T=4,
        lif_type='para_lif',
        norm_type='group',
        step_mode='m'):
        super().__init__()
        self.T = T

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Encode-Stage 1
        self.downsample1_a = MS_SpikeDownSampling3D(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
            lif_type=lif_type,
            norm_type=norm_type,
            step_mode=step_mode)
        
        self.encode_block1_a = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios[0], lif_type=lif_type, norm_type=norm_type, step_mode=step_mode)])

        self.downsample1_b = MS_SpikeDownSampling3D(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            lif_type=lif_type,
            norm_type=norm_type,
            step_mode=step_mode)
        
        self.encode_block1_b = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim[0], mlp_ratio=mlp_ratios[0], lif_type=lif_type, norm_type=norm_type, step_mode=step_mode)])

        # Encode-Stage 2
        self.downsample2 = MS_SpikeDownSampling3D(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            lif_type=lif_type,
            norm_type=norm_type,
            step_mode=step_mode)
                
        self.encode_block2_a = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim[1], mlp_ratio=mlp_ratios[1], lif_type=lif_type, norm_type=norm_type, step_mode=step_mode)])
       
        self.encode_block2_b = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim[1], mlp_ratio=mlp_ratios[1], lif_type=lif_type, norm_type=norm_type, step_mode=step_mode)])

        # Encode-Stage 3
        self.downsample3 = MS_SpikeDownSampling3D(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            lif_type=lif_type,
            norm_type=norm_type,
            step_mode=step_mode)
        
        self.encode_block3 = nn.ModuleList([
            MS_SpikeTransformerBlock3D(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                sr_ratio=sr_ratios[2],
                lif_type=lif_type,
                norm_type=norm_type,
                step_mode=step_mode
            ) for i in range(layers[2])])

        # feature-Stage
        self.feature_downsample = MS_SpikeDownSampling3D(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
            lif_type=lif_type,
            norm_type=norm_type,
            step_mode=step_mode)
        
        self.feature_block = nn.ModuleList([
            MS_SpikeTransformerBlock3D(
                dim=embed_dim[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                sr_ratio=sr_ratios[3],
                lif_type=lif_type,
                norm_type=norm_type,
                step_mode=step_mode
            ) for i in range(layers[3])
        ])
        
        # Decode-Stage 3
        self.upsample3 = MS_SpikeUpSampling3D(
            in_channels=embed_dim[3],
            out_channels=embed_dim[2],
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0,
            last_layer=False,
            lif_type=lif_type,
            norm_type=norm_type,
            step_mode=step_mode)
        
        self.decode_block3 = nn.ModuleList([
            MS_SpikeTransformerBlock3D(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                sr_ratio=sr_ratios[2],
                lif_type=lif_type,
                norm_type=norm_type,
                step_mode=step_mode
            ) for i in range(layers[2])])

        if skip_connection == 'add':
            self.converge3 = AddConverge3D(
                channels=embed_dim[2], norm_type=norm_type, step_mode=step_mode)
        elif skip_connection == 'cat':
            self.converge3 = MS_SpikeCatConverge3D(
                channels=embed_dim[2], norm_type=norm_type, lif_type=lif_type, step_mode=step_mode) 
        elif skip_connection == 'gate':
            self.converge3 = Gated_SpikeConverge3D(
                channels=embed_dim[2], norm_type=norm_type, lif_type=lif_type, step_mode=step_mode)
        elif skip_connection == 'attention':
            self.converge3 = MS_SpikeAttentionConverge3D(
                channels=embed_dim[2], norm_type=norm_type, lif_type=lif_type, step_mode=step_mode)

        # Decode-Stage 2
        self.upsample2 = MS_SpikeUpSampling3D(
            in_channels=embed_dim[2],
            out_channels=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            last_layer=False,
            lif_type=lif_type,
            norm_type=norm_type,
            step_mode=step_mode)
                
        self.decode_block2_a = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim[1], mlp_ratio=mlp_ratios[1], lif_type=lif_type, norm_type=norm_type, step_mode=step_mode)])
       
        self.decode_block2_b = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim[1], mlp_ratio=mlp_ratios[1], lif_type=lif_type, norm_type=norm_type, step_mode=step_mode)])
        

        if skip_connection == 'add':
            self.converge2 = AddConverge3D(
                channels=embed_dim[1], norm_type=norm_type, step_mode=step_mode)
        elif skip_connection == 'cat':
            self.converge2 = MS_SpikeCatConverge3D(
                channels=embed_dim[1], norm_type=norm_type, lif_type=lif_type, step_mode=step_mode)                     
        elif skip_connection == 'gate':
            self.converge2 = Gated_SpikeConverge3D(
                channels=embed_dim[1], norm_type=norm_type, lif_type=lif_type, step_mode=step_mode)
        elif skip_connection == 'attention':
            self.converge2 = MS_SpikeAttentionConverge3D(
                channels=embed_dim[1], norm_type=norm_type, lif_type=lif_type, step_mode=step_mode)

                   
        # Decode-Stage 1
        self.upsample1_b = MS_SpikeUpSampling3D(
            in_channels=embed_dim[1],
            out_channels= embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            last_layer=False,
            lif_type=lif_type,
            norm_type=norm_type,
            step_mode=step_mode)
        
        self.decode_block1_b = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim[0], mlp_ratio=mlp_ratios[0], lif_type=lif_type, norm_type=norm_type, step_mode=step_mode)])
        
        self.upsample1_a = MS_SpikeUpSampling3D(
            in_channels=embed_dim[0],
            out_channels=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            last_layer=False,
            lif_type=lif_type,
            norm_type=norm_type,
            step_mode=step_mode)
        
        self.decode_block1_a = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios[0], lif_type=lif_type, norm_type=norm_type, step_mode=step_mode)])


        if skip_connection == 'add':
            self.converge1 = AddConverge3D(
                channels=embed_dim[0], norm_type=norm_type, step_mode=step_mode)
        elif skip_connection == 'cat':
            self.converge1 = MS_SpikeCatConverge3D(
                channels=embed_dim[0], norm_type=norm_type, lif_type=lif_type, step_mode=step_mode)   
        elif skip_connection == 'gate':
            self.converge1 = Gated_SpikeConverge3D(
                channels=embed_dim[0], norm_type=norm_type, lif_type=lif_type, step_mode=step_mode)
        elif skip_connection == 'attention':
            self.converge1 = MS_SpikeAttentionConverge3D(
                channels=embed_dim[0], norm_type=norm_type, lif_type=lif_type, step_mode=step_mode)

        
        self.final_upsample = MS_SpikeUpSampling3D(
            in_channels=embed_dim[0] // 2,
            out_channels=embed_dim[0] // 4,
            kernel_size=3,
            stride=2,
            padding=1,
            last_layer=True,
            lif_type=lif_type,
            norm_type=norm_type,
            step_mode=step_mode)
        
        
        # self.lif = neuron.ParametricLIFNode(
        #     init_tau=2.0,
        #     decay_input=True,
        #     detach_reset=True,
        #     v_threshold=1.0,
        #     v_reset=0.0,
        #     surrogate_function=surrogate.ATan(), 
        #     step_mode=step_mode,
        #     backend='cupy')
        
        self.readout = layer.Conv3d(embed_dim[0] // 4, num_classes, kernel_size=1, step_mode=step_mode)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder_decoder(self, x):         # input shape: [T, B, 4, 64, 64, 64]
        # Encode-stage 1
        e1 = self.downsample1_a(x)          # Downsample1_a output shape: [T, B, 48, 32, 32, 32]
        for blk in self.encode_block1_a:
            e1 = blk(e1)                     # shape: [T, B, 48, 32, 32, 32]
        
        e1 = self.downsample1_b(e1)          # Downsample1_b output shape: [T, B, 96, 16, 16, 16]
        for blk in self.encode_block1_b:
            e1 = blk(e1)
        skip1 = e1                 # Skip2 shape: [T, B, 96, 16, 16, 16]

        # Encode-stage 2
        e2 = self.downsample2(e1)            # Downsample2 output shape: [T, B, 192, 8, 8, 8]
        for blk in self.encode_block2_a:
            e2 = blk(e2)
        for blk in self.encode_block2_b:
            e2 = blk(e2)
        skip2 = e2                  # Skip3 shape: [T, B, 192, 8, 8, 8]

        # Encode-stage 3
        e3 = self.downsample3(e2)            # Downsample3 output shape: [T, B, 384, 4, 4, 4]
        for blk in self.encode_block3:
            e3 = blk(e3)
        skip3 = e3                  # Skip4 shape: [T, B, 384, 4, 4, 4]

        # Encode-stage 4
        e4 = self.feature_downsample(e3)     # Downsample4 output shape: [T, B, 480, 4, 4, 4]
        for blk in self.feature_block:
            e4 = blk(e4)                     # After Encode-Stage 4: [T, B, 480, 4, 4, 4]
        
        # Decode-Stage 3
        d3 = self.upsample3(e4)              # Upsample3 output shape: [T, B, 384, 4, 4, 4]
        d3 = self.converge3(d3, skip3)       # converge3 output shape: [T, B, 384, 4, 4, 4]
        for blk in self.decode_block3:
            d3 = blk(d3)                     # After Decode-Stage3: [T, B, 384, 4, 4, 4]
        
        # Decode-Stage 2
        d2 = self.upsample2(d3)              # Upsample2 output shape: [T, B, 192, 8, 8, 8]
        
        d2 = self.converge2(d2, skip2)       # Converge2 output shape: [T, B, 192, 8, 8, 8]
        for blk in self.decode_block2_a:
            d2 = blk(d2)
        for blk in self.decode_block2_b:
            d2 = blk(d2)                     # After Decode-Stage2: [T, B, 192, 8, 8, 8]

        # Decode-Stage 1
        d1 = self.upsample1_b(d2)            # Upsample1_b output shape: [T, B, 96, 16, 16, 16]
        d1 = self.converge1(d1, skip1)       # Converge1 output shape: [T, B, 96, 16, 16, 16]
        for blk in self.decode_block1_b:
            d1 = blk(d1)
        
        d1 = self.upsample1_a(d1)            # Upsample1_a output shape: [T, B, 48, 32, 32, 32]
        for blk in self.decode_block1_a:
            d1 = blk(d1)
            
        out =self.final_upsample(d1)          # Final Upsample output shape: [T, B, 24, 64, 64, 64]

        return out

    def forward(self, x):
        functional.reset_net(self)
        out = self.forward_encoder_decoder(x)

        # Readout
        # out = self.lif(out) 
        output = self.readout(out).mean(0)  # [3, 64, 64, 64]

        return output


    
def spike_former_unet3D_8_384(in_channels=4, num_classes=3, T=4, norm_type='group', step_mode='m',**kwargs):
    model = Spike_Former_Unet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=[96, 192, 384, 480],
        num_heads=[8, 8, 8, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        depths=[8, 8, 8, 8],
        layers=[2, 2, 6, 2],
        sr_ratios=[1, 1, 1, 1],
        skip_connection='cat', #  add, cat, gate, attention
        T=T,
        lif_type='general_para_lif', # lif, para_lif, general_para_lif
        norm_type=norm_type,
        step_mode=step_mode,
        **kwargs
    )
    return model


def spike_former_unet3D_8_512(in_channels=4, num_classes=3, T=4, norm_type='group', step_mode='m',**kwargs):
    model = Spike_Former_Unet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=[128, 256, 512, 640],
        num_heads=[8, 8, 8, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        depths=[8, 8, 8, 8],
        sr_ratios=[1, 1, 1, 1],
        skip_connection='cat',
        T=T,
        lif_type='para_lif',
        norm_type=norm_type,
        step_mode=step_mode,
        **kwargs,
    )
    return model


def spike_former_unet3D_8_768(in_channels=4, num_classes=3, T=4, norm_type='group', step_mode='m',**kwargs):
    model = Spike_Former_Unet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=[192, 384, 768, 960],
        num_heads=[8, 8, 8, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        depths=[8, 8, 8, 8],
        sr_ratios=[1, 1, 1, 1],
        skip_connection='cat',
        T=T,
        lif_type='para_lif',
        norm_type=norm_type,
        step_mode=step_mode,
        **kwargs,
    )
    return model    
    


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params


       
def main():
    # 测试模型
    import os
    os.chdir(os.path.dirname(__file__))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = spike_former_unet3D_8_384().to(device)
    x = torch.randn(2, 2, 4, 64, 64, 64)  # 假设输入是一个 batch 的数据
    x = x.to(device)
    output = model(x)
    print(output.shape)  # 输出形状应该是 [1, 3, 128, 128, 128]
    # from config import config as cfg  
    # model = spike_former_unet3D_8_384(
    #     num_classes=cfg.num_classes,
    #     T=cfg.T,
    #     norm_type=cfg.norm_type,
    #     step_mode=cfg.step_mode)  # 模型
    # print("Model parameters for spike_former_unet3D_8_384:")
    # count_parameters(model)  # 91.852 M 
    # model = spike_former_unet3D_8_512(
    #     num_classes=cfg.num_classes,
    #     T=cfg.T,
    #     norm_type=cfg.norm_type,
    #     step_mode=cfg.step_mode)
    # print("Model parameters for spike_former_unet3D_8_512:")
    # count_parameters(model)  # 162.579 M
    # model = spike_former_unet3D_8_768(
    #     num_classes=cfg.num_classes,
    #     T=cfg.T,
    #     norm_type=cfg.norm_type,
    #     step_mode=cfg.step_mode)
    # print("Model parameters for spike_former_unet3D_8_768:")
    # count_parameters(model) # 364.198 M
    
if __name__ == "__main__":
    main()