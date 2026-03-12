"""
Mask-Guided Attention Module
Based on paper1231: Mask-guided Attention Module (Section C.2)

This module implements mask-guided attention to focus on instrument manipulation regions.

Reference from paper1231:
1. Generates mean and max spatial attention maps from mask features
2. Aggregates attention maps using learnable weights
3. Constrains attention with ground-truth masks via pixel-level loss
4. Multiplies attention with features to enhance instrument regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGuidedAttention(nn.Module):
    """
    Mask-Guided Attention Module from paper1231.

    Formula from paper:
    A_mean = (1/C) * sum(F_ms)  # Average pooling over channels
    A_max = max(F_ms)              # Max pooling over channels
    A = sigmoid(f_agg(A_mean, A_max))  # Aggregate and activate
    F_dy = F_clip * (A + I)           # Apply to features

    Where:
    - F_ms: Instrument mask features from segmentation network
    - f_agg: Learnable aggregation function
    - I: Identity matrix
    - F_clip: Short-term spatiotemporal features from I3D
    """
    def __init__(self,
                 feature_dim=832,
                 use_mask_loss=True,
                 enable_temporal_smoothing=True):
        """
        Args:
            feature_dim: Input feature channel dimension
            use_mask_loss: Whether to use mask supervision loss
            enable_temporal_smoothing: Smooth masks across adjacent frames
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.use_mask_loss = use_mask_loss
        self.enable_temporal_smoothing = enable_temporal_smoothing

        # Aggregation function f_agg: combines mean and max attention maps
        self.aggregation = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Additional attention generation from dynamic features
        # This allows the network to learn attention beyond mask guidance
        self.attention_conv = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Mask projection for matching feature dimensions
        self.mask_proj = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Fusion weight (learnable balance between mask and learned attention)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _temporal_smoothing(self, masks):
        """
        Apply temporal smoothing to masks as in paper1231.

        Formula: M_gt[w,h] = f_2D(sum(M^{2t-1,w,h} + M^{2t,w,h}))
        This reduces jitter and local instability in single-frame masks.

        Args:
            masks: (B, T, H, W)
        Returns:
            smoothed_masks: (B, T, H, W)
        """
        if masks.size(1) < 2:
            return masks

        # Sum adjacent frames
        even_frames = masks[:, 0::2, :, :]  # M^0, M^2, ...
        odd_frames = masks[:, 1::2, :, :]  # M^1, M^3, ...

        # Pad if odd number of frames
        if even_frames.size(1) > odd_frames.size(1):
            odd_frames = F.pad(odd_frames, (0, 0, 0, 0, 0, 1), 'replicate')

        # Sum adjacent frames
        smoothed = even_frames + odd_frames

        # Average to normalize
        smoothed = smoothed / 2.0

        return smoothed

    def _generate_attention_from_masks(self, masks, target_size):
        """
        Generate attention map from mask features.

        Args:
            masks: (B, T, H, W) - binary mask values [0,1]
            target_size: (H', W') - target spatial size
        Returns:
            attention: (B, T, H', W') - attention map [0,1]
        """
        B, T, H, W = masks.shape

        # Prepare mask features
        # Stack along channel dimension for mean/max operations
        masks_expanded = masks.unsqueeze(1)  # (B, 1, T, H, W)

        # Mean attention (average over spatial dimensions)
        A_mean = F.avg_pool2d(masks_expanded, kernel_size=H, stride=1, padding=H//2)
        A_mean = F.avg_pool2d(A_mean, kernel_size=W, stride=1, padding=W//2)
        # This becomes (B, 1, T, 1, 1) -> squeeze -> (B, T)

        # Max attention
        A_max = F.max_pool2d(masks_expanded, kernel_size=H, stride=1, padding=H//2)
        A_max = F.max_pool2d(A_max, kernel_size=W, stride=1, padding=W//2)
        # This becomes (B, 1, T, 1, 1) -> squeeze -> (B, T)

        # Stack mean and max for aggregation
        A_stacked = torch.cat([A_mean, A_max], dim=1)  # (B, 2, T, 1, 1)

        # Aggregate through learnable function
        # Reshape for 2D conv: (B*T, 2, 1, 1)
        B_T = A_stacked.size(0) * A_stacked.size(2)
        A_reshaped = A_stacked.permute(0, 2, 1, 3, 4).reshape(B_T, 2, 1, 1)
        A_agg = self.aggregation(A_reshaped)  # (B*T, 1, 1, 1)
        A_agg = A_agg.reshape(B, T, 1, 1).squeeze(-1)  # (B, T, 1, 1)

        # Resize to target size
        A_agg = F.interpolate(A_agg, size=target_size, mode='bilinear', align_corners=False)

        return A_agg.squeeze(1)  # (B, T, H', W')

    def forward(self, dynamic_features, masks, return_attention_map=False):
        """
        Apply mask-guided attention to dynamic features.

        Args:
            dynamic_features: (B, C, T, H, W) - spatiotemporal features from I3D
            masks: (B, T, H_mask, W_mask) - instrument masks [0,1]
            return_attention_map: Return final attention map for visualization

        Returns:
            masked_features: (B, C) - attention-applied features
            attention_map: (B, T, H, W) - optional
            mask_loss: Tensor - optional mask supervision loss
        """
        B, C, T, H, W = dynamic_features.shape
        device = dynamic_features.device

        # 1. Temporal smoothing of masks (as in paper1231)
        if self.enable_temporal_smoothing:
            masks = self._temporal_smoothing(masks)

        # 2. Generate attention from mask features
        target_size = (H, W)
        mask_attention = self._generate_attention_from_masks(masks, target_size)  # (B, T, H, W)

        # 3. Project mask to match feature dimensions
        # Mask is (B, T, H, W), convert to (B, 1, T, H, W)
        masks_3d = masks.unsqueeze(1)
        masks_3d = F.interpolate(masks_3d, size=(T, H, W), mode='trilinear', align_corners=False)

        # Project through conv layers
        projected_mask = self.mask_proj(masks_3d.squeeze(1))  # (B, 1, T, H, W)

        # 4. Generate learned attention from features
        feature_attention = self.attention_conv(dynamic_features)  # (B, 1, T, H, W)

        # 5. Combine mask attention and learned attention
        # Learnable fusion weight
        combined_attention = (
            self.alpha * mask_attention + (1 - self.alpha) * feature_attention.squeeze(1)
        ).unsqueeze(1)  # (B, 1, T, H, W)

        # 6. Apply attention to features (element-wise multiplication)
        # F_dy = F_clip * (A + I) where I is identity
        masked_features = dynamic_features * (combined_attention + 1.0)  # (B, C, T, H, W)

        # 7. Global pooling to get compact representation
        pooled_features = F.adaptive_avg_pool3d(masked_features, (1, 1, 1))  # (B, C, 1, 1, 1)
        pooled_features = pooled_features.flatten(1)  # (B, C)

        # 8. Optional mask supervision loss
        mask_loss = None
        if self.use_mask_loss and self.training:
            # L2 loss between attention and ground-truth mask
            # Loss_mask = (1/(T*W*H)) * sum(||A - M_gt||^2)
            target_mask_combined = masks_3d.squeeze(1)  # (B, T, H, W)
            pred_attention = combined_attention.squeeze(1)  # (B, T, H, W)

            loss = F.mse_loss(pred_attention, target_mask_combined)
            mask_loss = loss

        if return_attention_map:
            return pooled_features, combined_attention.squeeze(1), mask_loss
        else:
            return pooled_features, None, mask_loss


class AttentionVisualization(nn.Module):
    """
    Utility module for attention map visualization.
    """
    @staticmethod
    def visualize_attention(original_video, attention_map, save_path):
        """
        Visualize attention map overlayed on original frames.

        Args:
            original: (T, H, W, C) or (B, C, T, H, W)
            attention_map: (T, H, W) or (B, T, H, W)
            save_path: Path to save visualization
        """
        import cv2
        import numpy as np
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Handle different input formats
        if original.dim() == 4:  # (B, C, T, H, W)
            original = original[0]
            original = original.permute(1, 2, 3, 0)  # (T, H, W, C)
        elif original.dim() == 5:  # (B, C, T, H, W)
            original = original[0]
            original = original.permute(1, 2, 3, 0)  # (T, H, W, C)

        if attention_map.dim() == 3:  # (B, T, H, W)
            attention_map = attention_map[0]

        T, H, W, C = original.shape

        # Save each frame with attention overlay
        for t in range(T):
            frame = original[t].cpu().numpy()
            frame = (frame * 255).astype(np.uint8)

            attention = attention_map[t].cpu().numpy()
            attention = (attention * 255).astype(np.uint8)

            # Convert to color
            if C == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame

            # Create heatmap
            heatmap = cv2.applyColorMap(attention, cv2.COLORMAP_JR)

            # Blend original and heatmap
            blended = cv2.addWeighted(frame_bgr, 0.7, heatmap, 0.3, 0)

            cv2.imwrite(f"{save_path}/frame_{t:04d}.png", blended)

        print(f"Attention visualization saved to {save_path}")


if __name__ == '__main__':
    # Test module
    model = MaskGuidedAttention(feature_dim=832)

    # Create dummy inputs
    dynamic_features = torch.randn(2, 832, 8, 28, 28)  # (B, C, T, H, W)
    masks = torch.randint(0, 2, (2, 8, 224, 224)).float()  # (B, T, H, W)

    # Forward pass
    features, attention, loss = model(dynamic_features, masks, return_attention_map=True)

    print(f"Input features shape: {dynamic_features.shape}")
    print(f"Input masks shape: {masks.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Attention map shape: {attention.shape}")
    if loss is not None:
        print(f"Mask loss: {loss.item()}")
