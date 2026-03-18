import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGuidedAttentionMultiClip(nn.Module):
    def __init__(self, enable_temporal_smoothing=True, clip_length=16, clip_stride=10):
        super().__init__()
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.clip_length = clip_length
        self.clip_stride = clip_stride

    def _temporal_smoothing(self, masks):
        B, T, H, W = masks.shape
        if T < 2:
            return masks
        num_pairs = T // 2
        even_frames = masks[:, 0:num_pairs*2:2, :, :]
        odd_frames = masks[:, 1:num_pairs*2:2, :, :]
        summed = even_frames + odd_frames
        smoothed = F.avg_pool2d(summed, kernel_size=2, stride=2, padding=0)
        smoothed = smoothed / 2.0
        return smoothed

    def _split_masks_for_clips(self, masks, num_clips):
        B, T, H, W = masks.shape
        clip_masks = []
        for i in range(num_clips):
            start_idx = i * self.clip_stride
            end_idx = start_idx + self.clip_length
            if end_idx > T:
                clip_mask = masks[:, start_idx:T, :, :]
                padding_len = self.clip_length - (T - start_idx)
                last_frame = masks[:, -1:, :].unsqueeze(2)
                padding = last_frame.expand(B, padding_len, H, W)
                clip_mask = torch.cat([clip_mask, padding], dim=1)
            else:
                clip_mask = masks[:, start_idx:end_idx, :, :]
            clip_masks.append(clip_mask)
        return clip_masks

    def forward(self, dynamic_features_per_clip, masks, num_clips=None):
        B, num_clips, C, T_feat, H_feat, W_feat = dynamic_features_per_clip.shape
        if num_clips is None:
            num_clips = dynamic_features_per_clip.shape[1]
        if masks is None:
            clip_masks = [torch.ones(B, self.clip_length, H_feat, W_feat,
                                       device=dynamic_features_per_clip.device)
                       for _ in range(num_clips)]
        else:
            if self.enable_temporal_smoothing:
                masks = self._temporal_smoothing(masks)
            clip_masks = self._split_masks_for_clips(masks, num_clips)
        masked_features_list = []
        attention_maps_list = []
        for clip_idx in range(num_clips):
            clip_feat_map = dynamic_features_per_clip[:, clip_idx, :, :, :]
            clip_mask = clip_masks[clip_idx]
            mask_3d = clip_mask.unsqueeze(1)
            mask_aligned = F.interpolate(mask_3d, size=(T_feat, H_feat, W_feat),
                                       mode="bilinear", align_corners=False)
            attention_map = mask_aligned.squeeze(1)
            masked_feat_map = clip_feat_map * (attention_map.unsqueeze(1) + 1.0)
            pooled_feat = F.adaptive_avg_pool3d(masked_feat_map, (1, 1, 1))
            pooled_feat = pooled_feat.flatten(1)
            masked_features_list.append(pooled_feat)
            attention_maps_list.append(attention_map)
        masked_features_per_clip = torch.stack(masked_features_list, dim=1)
        attention_maps = torch.stack(attention_maps_list, dim=1)
        return masked_features_per_clip, attention_maps
