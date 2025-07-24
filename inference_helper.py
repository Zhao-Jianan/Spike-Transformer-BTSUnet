import torch
import numpy as np
from monai.transforms import NormalizeIntensity
from spikingjelly.clock_driven.encoding import PoissonEncoder, LatencyEncoder, WeightedPhaseEncoder
from monai.inferers import sliding_window_inference
import torch.nn.functional as F

class TemporalSlidingWindowInference:
    def __init__(
        self,
        patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="gaussian",
        encode_method='none',
        T=4,
        num_classes=3
    ):
        self.patch_size = patch_size
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.mode = mode
        self.T = T
        self.encode_method = encode_method
        self.poisson_encoder = PoissonEncoder()
        self.latency_encoder = LatencyEncoder(self.T)
        self.weighted_phase_encoder = WeightedPhaseEncoder(self.T)
        self.num_classes = num_classes
        
        
    def encode_spike_input(self, img_rescale: torch.Tensor) -> torch.Tensor:
        """
        对归一化图像进行脉冲编码，支持 poisson / latency / weighted_phase。
        输入:
            img_rescale: torch.Tensor, shape (B, C, D, H, W)
        输出:
            spike_tensor: torch.Tensor, shape (T, B, C, D, H, W)
        """
        # 维度判断
        if img_rescale.dim() != 5:
            raise ValueError(f"Unexpected input shape {img_rescale.shape}, expected 5D tensor.")
        

        if self.encode_method == 'poisson':
            # [T, B, C, D, H, W]
            spike = torch.stack([self.poisson_encoder(img_rescale) for _ in range(self.T)], dim=0)

        elif self.encode_method == 'latency':
            # Latency 编码只需 encode 一次
            self.latency_encoder.encode(img_rescale)   # 输入 (B, C, D, H, W)
            spike = self.latency_encoder.spike         # (T, B, C, D, H, W)

        elif self.encode_method == 'weighted_phase':
            # Weighted Phase 编码前需调整强度
            x = img_rescale * (1 - 2 ** (-self.T))      # 重要：训练中也做了
            self.weighted_phase_encoder.encode(x)      # 输入 (B, C, D, H, W)
            spike = self.weighted_phase_encoder.spike.float()  # (T, B, C, D, H, W)
        elif self.encode_method == 'none':
            # 直接使用归一化后的图像作为输入
            spike = img_rescale.unsqueeze(0).repeat(self.T, 1, 1, 1, 1, 1)
        else:
            raise NotImplementedError(f"Encoding method '{self.encode_method}' is not implemented.")

        return spike

    def _get_weight_window(self, device):
        if self.mode == "gaussian":
            coords = [torch.arange(s, dtype=torch.float32, device=device) for s in self.patch_size]
            zz, yy, xx = torch.meshgrid(*coords, indexing="ij")
            zz = zz - (self.patch_size[0] - 1) / 2
            yy = yy - (self.patch_size[1] - 1) / 2
            xx = xx - (self.patch_size[2] - 1) / 2
            sigmas = [s / 8 for s in self.patch_size]  # 分别计算 σ_D, σ_H, σ_W
            gaussian = torch.exp(
                -(zz ** 2 / (2 * sigmas[0] ** 2) +
                yy ** 2 / (2 * sigmas[1] ** 2) +
                xx ** 2 / (2 * sigmas[2] ** 2))
            )
            gaussian = gaussian.unsqueeze(0).unsqueeze(0)  # 形状变成 [1, 1, pd, ph, pw]
            return gaussian
        elif self.mode == "constant":
            weight = torch.ones(self.patch_size, device=device)
            weight = weight.unsqueeze(0).unsqueeze(0)
            return weight
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def __call__(self, inputs: torch.Tensor, predictor: callable) -> torch.Tensor:
        B, C, D, H, W = inputs.shape
        device = inputs.device
        pd, ph, pw = self.patch_size
        stride = [int(r * (1 - self.overlap)) for r in self.patch_size]

        # ---------------------------
        # Step 0: Global rescale before sliding window
        # ---------------------------
        def global_rescale_0_1(img: torch.Tensor) -> torch.Tensor:
            # img: [B, C, D, H, W]
            min_val = img.amin(dim=[2, 3, 4], keepdim=True)
            max_val = img.amax(dim=[2, 3, 4], keepdim=True)
            scale = (max_val - min_val).clamp(min=1e-5)
            return (img - min_val) / scale

        # 使用第0帧的图像进行归一化（不影响时间维度）
        inputs_rescaled = inputs.clone()
        if self.encode_method != 'none':
            inputs_rescaled = global_rescale_0_1(inputs)

        # ---------------------------
        # Step 1: Padding if needed
        # ---------------------------
        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        pad = [
            pad_w // 2, pad_w - pad_w // 2,  # W
            pad_h // 2, pad_h - pad_h // 2,  # H
            pad_d // 2, pad_d - pad_d // 2   # D
        ]

        inputs_rescaled = inputs_rescaled.view(-1, D, H, W)
        inputs_rescaled = F.pad(inputs_rescaled, pad=pad, mode="constant", value=0.0)
        inputs_rescaled = inputs_rescaled.view(B, C, D + pad_d, H + pad_h, W + pad_w)
        D_pad, H_pad, W_pad = inputs_rescaled.shape[-3:]
        padded = any([pad_d, pad_h, pad_w])
        pad_info = (pad_d, pad_h, pad_w, D, H, W)

        # ---------------------------
        # Step 2: Prepare output tensors
        # ---------------------------
        weight_window = self._get_weight_window(device)
        output = torch.zeros((B, self.num_classes, D_pad, H_pad, W_pad), device=device)
        weight_map = torch.zeros((1, 1, D_pad, H_pad, W_pad), device=device)

        # ---------------------------
        # Step 3: Sliding window
        # ---------------------------
        def get_starts(dim, patch_size, stride):
            starts = list(range(0, dim - patch_size + 1, stride))
            if starts[-1] != dim - patch_size:
                starts.append(dim - patch_size)
            return starts

        z_starts = get_starts(D_pad, pd, stride[0])
        y_starts = get_starts(H_pad, ph, stride[1])
        x_starts = get_starts(W_pad, pw, stride[2])

        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch = inputs_rescaled[:, :, z:z+pd, y:y+ph, x:x+pw]  # [B, C, pd, ph, pw]
                    for b_start in range(0, B, self.sw_batch_size):
                        b_end = min(b_start + self.sw_batch_size, B)
                        patch_img = patch[b_start:b_end, :, :, :, :]  # [b, C, pd, ph, pw]

                        # Step 3.1: Encode directly using already rescaled intensity
                        patch_encoded = self.encode_spike_input(patch_img)  # [T, b, C, pd, ph, pw]

                        # Step 3.2: Model inference
                        pred = predictor(patch_encoded)  # [b, C_out, pd, ph, pw]

                        output[b_start:b_end, :, z:z+pd, y:y+ph, x:x+pw] += pred * weight_window
                        weight_map[:, :, z:z+pd, y:y+ph, x:x+pw] += weight_window

        # ---------------------------
        # Step 4: Normalize & remove padding
        # ---------------------------
        weight_map = weight_map.clamp(min=1e-5)
        output = output / weight_map

        if padded:
            pad_d, pad_h, pad_w, D, H, W = pad_info
            d_start = pad_d // 2
            h_start = pad_h // 2
            w_start = pad_w // 2
            output = output[:, :, d_start:d_start + D, h_start:h_start + H, w_start:w_start + W]

        return output




class TemporalSlidingWindowInferenceWithROI:
    def __init__(
        self,
        patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="constant", 
        encode_method='none',
        T=4,
        num_classes=3
    ):
        self.patch_size = patch_size
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.mode = mode
        self.T = T
        self.encode_method = encode_method
        self.poisson_encoder = PoissonEncoder()
        self.latency_encoder = LatencyEncoder(self.T)
        self.weighted_phase_encoder = WeightedPhaseEncoder(self.T)
        self.num_classes = num_classes

    def encode_spike_input(self, img_rescale: torch.Tensor) -> torch.Tensor:
        if img_rescale.dim() != 5:
            raise ValueError(f"Unexpected input shape {img_rescale.shape}, expected 5D tensor.")
        if self.encode_method == 'poisson':
            spike = torch.stack([self.poisson_encoder(img_rescale) for _ in range(self.T)], dim=0)
        elif self.encode_method == 'latency':
            self.latency_encoder.encode(img_rescale)
            spike = self.latency_encoder.spike
        elif self.encode_method == 'weighted_phase':
            x = img_rescale * (1 - 2 ** (-self.T))
            self.weighted_phase_encoder.encode(x)
            spike = self.weighted_phase_encoder.spike.float()
        elif self.encode_method == 'none':
            spike = img_rescale.unsqueeze(0).repeat(self.T, 1, 1, 1, 1, 1)
        else:
            raise NotImplementedError(f"Encoding method '{self.encode_method}' not implemented.")
        return spike

    def global_rescale_0_1(self, img: torch.Tensor) -> torch.Tensor:
        min_val = img.amin(dim=[2, 3, 4], keepdim=True)
        max_val = img.amax(dim=[2, 3, 4], keepdim=True)
        scale = (max_val - min_val).clamp(min=1e-5)
        return (img - min_val) / scale

    def compute_patch_indices(self, img_shape, patch_size, stride):
        """
        计算patch的起点索引列表，类似别人代码compute_patch_indices_for_prediction
        """
        indices = []
        for z in range(0, img_shape[0] - patch_size[0] + 1, stride[0]):
            for y in range(0, img_shape[1] - patch_size[1] + 1, stride[1]):
                for x in range(0, img_shape[2] - patch_size[2] + 1, stride[2]):
                    indices.append((z, y, x))
        # 如果最后的patch没有覆盖末尾，追加覆盖末尾的patch索引
        last_z = img_shape[0] - patch_size[0]
        last_y = img_shape[1] - patch_size[1]
        last_x = img_shape[2] - patch_size[2]
        if indices[-1][0] != last_z:
            for y in range(0, img_shape[1] - patch_size[1] + 1, stride[1]):
                for x in range(0, img_shape[2] - patch_size[2] + 1, stride[2]):
                    indices.append((last_z, y, x))
        if indices[-1][1] != last_y:
            for z in range(0, img_shape[0] - patch_size[0] + 1, stride[0]):
                for x in range(0, img_shape[2] - patch_size[2] + 1, stride[2]):
                    indices.append((z, last_y, x))
        if indices[-1][2] != last_x:
            for z in range(0, img_shape[0] - patch_size[0] + 1, stride[0]):
                for y in range(0, img_shape[1] - patch_size[1] + 1, stride[1]):
                    indices.append((z, y, last_x))
        # 去重
        indices = list(set(indices))
        indices.sort()
        return indices

    def __call__(self, inputs: torch.Tensor, brain_width: np.ndarray, predictor: callable) -> torch.Tensor:
        """
        inputs: torch.Tensor, shape (B, C, D, H, W)
        brain_width: np.ndarray, shape (2,3), 两个坐标点表示ROI的开始和结束坐标（含）
        predictor: 预测函数，输入脉冲编码数据，输出对应patch预测概率
        """
        B, C, D, H, W = inputs.shape
        device = inputs.device
        pd, ph, pw = self.patch_size
        stride = [int(r * (1 - self.overlap)) for r in self.patch_size]

        # Step 0: ROI裁剪
        # 保证brain_width在范围内
        brain_width = np.clip(brain_width, a_min=[0,0,0], a_max=[D-1, H-1, W-1])
        d_start, h_start, w_start = brain_width[0]
        d_end, h_end, w_end = brain_width[1]
        # 注意slice索引，python slice不含结束索引，需要+1
        roi_data = inputs[:, :, d_start:d_end+1, h_start:h_end+1, w_start:w_end+1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        roi_shape = roi_data.shape[-3:]  # 裁剪后的ROI尺寸

        # Step 1: 全局归一化(ROI内)
        if self.encode_method != 'none':
            roi_data_rescaled = self.global_rescale_0_1(roi_data)
        else:
            roi_data_rescaled = roi_data.clone()

        # Step 2: Padding ROI到patch大小
        pad_d = max(0, pd - roi_shape[0])
        pad_h = max(0, ph - roi_shape[1])
        pad_w = max(0, pw - roi_shape[2])
        pad = [
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2
        ]
        roi_data_rescaled = roi_data_rescaled.view(-1, *roi_shape)
        roi_data_rescaled = F.pad(roi_data_rescaled, pad=pad, mode='constant', value=0.0)
        roi_data_rescaled = roi_data_rescaled.view(B, C, roi_shape[0] + pad_d, roi_shape[1] + pad_h, roi_shape[2] + pad_w)
        roi_shape_pad = roi_data_rescaled.shape[-3:]

        # Step 3: 计算patch索引（起点）
        indices = self.compute_patch_indices(roi_shape_pad, self.patch_size, stride)

        # Step 4: 按batch批量预测patch
        predictions = []
        for b in range(B):
            patches = []
            patch_indices = []
            for idx, (z, y, x) in enumerate(indices):
                patch = roi_data_rescaled[b:b+1, :, z:z+pd, y:y+ph, x:x+pw]  # [1, C, pd, ph, pw]
                patches.append(patch)
                patch_indices.append((z, y, x))

                # 满batch或者最后一个patch时预测
                if len(patches) == self.sw_batch_size or idx == len(indices) -1:
                    batch_patch = torch.cat(patches, dim=0)  # [batch_size, C, pd, ph, pw]
                    # 编码
                    batch_patch_encoded = self.encode_spike_input(batch_patch)  # [T, batch, C, pd, ph, pw]
                    batch_patch_encoded = batch_patch_encoded.to(device)
                    # 预测，返回 [batch, num_classes, pd, ph, pw]
                    pred = predictor(batch_patch_encoded)
                    predictions.extend([(p, patch_indices[i]) for i, p in enumerate(pred)])
                    patches = []
                    patch_indices = []

        # Step 5: 重组ROI预测输出（无权重加权）
        output_roi = torch.zeros((B, self.num_classes, *roi_shape_pad), device=device)
        count_map = torch.zeros((B, 1, *roi_shape_pad), device=device)

        for b in range(B):
            for pred_patch, (z, y, x) in filter(lambda x: x[0].device == device, predictions):
                output_roi[b:b+1, :, z:z+pd, y:y+ph, x:x+pw] += pred_patch.unsqueeze(0)
                count_map[b:b+1, :, z:z+pd, y:y+ph, x:x+pw] += 1

        # 防止除零
        count_map[count_map == 0] = 1
        output_roi = output_roi / count_map

        # Step 6: 去padding，还原ROI大小
        if pad_d + pad_h + pad_w > 0:
            d_start_pad = pad_d // 2
            h_start_pad = pad_h // 2
            w_start_pad = pad_w // 2
            output_roi = output_roi[:, :, d_start_pad:d_start_pad+roi_shape[0], h_start_pad:h_start_pad+roi_shape[1], w_start_pad:w_start_pad+roi_shape[2]]

        # Step 7: 将ROI结果拼回原图大小，空白处填0
        output_full = torch.zeros((B, self.num_classes, D, H, W), device=device)
        output_full[:, :, d_start:d_end+1, h_start:h_end+1, w_start:w_end+1] = output_roi

        return output_full