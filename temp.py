class BNAndPad3DLayer(nn.Module):
    def __init__(self, pad_voxels, num_features, step_mode='m', **bn_kwargs):
        """
        3D BN+ padding组合，支持单步和多步模式。

        :param pad_voxels: int，前后上下左右各方向填充的体素数
        :param num_features: 通道数
        :param bn_kwargs: 传给 BatchNorm3d 的额外参数，如 eps、momentum 等
        """
        super().__init__()
        self.pad_voxels = pad_voxels
        # self.bn = layer.BatchNorm3d(num_features, step_mode=step_mode, **bn_kwargs)
        self.bn = layer.GroupNorm(num_groups=8, num_channels=num_features, step_mode=step_mode)
        
        self.step_mode = step_mode

    def _compute_pad_value(self):
        if self.bn.affine:
            pad_value = (
                self.bn.bias.detach()
                - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps))
        else:
            pad_value = -self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
        return pad_value.view(1, -1, 1, 1, 1)  # reshape to [1, C, 1, 1, 1]

    def _pad_tensor(self, x, pad_value):
        pad = [self.pad_voxels] * 6  # [W_left, W_right, H_top, H_bottom, D_front, D_back]
        x = F.pad(x, pad)  # 使用0填充
        # 再替换成 pad_value
        x[:, :, :self.pad_voxels, :, :] = pad_value
        x[:, :, -self.pad_voxels:, :, :] = pad_value
        x[:, :, :, :self.pad_voxels, :] = pad_value
        x[:, :, :, -self.pad_voxels:, :] = pad_value
        x[:, :, :, :, :self.pad_voxels] = pad_value
        x[:, :, :, :, -self.pad_voxels:] = pad_value
        return x

    def forward(self, x):
        if self.step_mode == 's':
            x = self.bn(x)  # shape: [N, C, D, H, W]
            if self.pad_voxels > 0:
                pad_value = self._compute_pad_value()
                x = self._pad_tensor(x, pad_value)
            return x

        elif self.step_mode == 'm':
            if x.dim() != 6:
                raise ValueError(f"Expected input shape [T, N, C, D, H, W], but got {x.shape}")
            x = self.bn(x)
            if self.pad_voxels > 0:
                pad_value = self._compute_pad_value()
                # 对每个时间步进行 padding
                padded = []
                for t in range(x.shape[0]):
                    padded.append(self._pad_tensor(x[t], pad_value))
                x = torch.stack(padded, dim=0)  # [T, N, C, D+2p, H+2p, W+2p]
            return x

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps