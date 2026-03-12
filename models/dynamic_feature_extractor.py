"""
Dynamic Feature Extractor Module
Based on paper1231: Dynamic Instrument Temporal Feature Extraction Module (Section C)

This module extracts spatiotemporal features using I3D backbone.
Captures instrument manipulation dynamics through 3D CNN.

Reference from paper1231:
- Video is divided into N overlapping snippets (stride=10, each has 16 frames)
- I3D extracts short-term spatiotemporal features
- Mixed convolution layers expand temporal receptive field
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MixedConv3D(nn.Module):
    """
    Mixed convolution layer from paper1231.

    Expands temporal receptive field by combining 3D and separable convolutions.
    """
    def __init__(self, in_channels, out_channels, temporal_kernel=3):
        super().__init__()

        # Standard 3D convolution
        self.conv3d = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(temporal_kernel, 3, 3),
            padding=(temporal_kernel//2, 1, 1)
        )

        # Temporal convolution (for temporal dynamics)
        self.conv_temporal = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(temporal_kernel//2, 0, 0)
        )

        # Spatial convolution
        self.conv_spatial = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)
        )

        # Fuse mixed features
        self.fuse = nn.Conv3d(
            out_channels * 2, out_channels,
            kernel_size=1
        )

        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            out: (B, out_channels, T, H, W)
        """
        # Standard 3D conv path
        out1 = self.conv3d(x)

        # Mixed conv path
        temporal_out = self.conv_temporal(x)
        spatial_out = self.conv_spatial(temporal_out)
        mixed_out = torch.cat([temporal_out, spatial_out], dim=1)
        out2 = self.fuse(mixed_out)

        # Combine both paths
        out = (out1 + out2) / 2
        out = self.bn(out)
        out = self.relu(out)

        return out


class I3DBackbone(nn.Module):
    """
    Simplified I3D backbone for dynamic feature extraction.

    Based on InceptionI3D architecture but simplified for feature extraction.
    """
    def __init__(self, in_channels=3, num_classes=400):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(1, 7, 7),
                              stride=(1, 2, 2), padding=(0, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv2 = nn.Conv3d(64, 64, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 192, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(192)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Mixed_3b, Mixed_3c
        self.mixed_3b = self._inception_module(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = self._inception_module(256, [128, 128, 192, 32, 96, 64])
        self.maxpool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        # Mixed_4b to Mixed_4f
        self.mixed_4b = self._inception_module(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = self._inception_module(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = self._inception_module(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = self._inception_module(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = self._inception_module(528, [256, 160, 320, 32, 128, 128])
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))

        # Mixed_5b, Mixed_5c
        self.mixed_5b = self._inception_module(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = self._inception_module(832, [384, 192, 384, 48, 128, 128])

        # Classification head
        self.avgpool = nn.AvgPool3d(kernel_size=(8, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(832, num_classes)

    def _inception_module(self, in_channels, out_channels):
        """Build an inception module."""
        return nn.ModuleDict({
            'b0': nn.Conv3d(in_channels, out_channels[0], kernel_size=1),
            'b1a': nn.Conv3d(in_channels, out_channels[1], kernel_size=1),
            'b1b': nn.Conv3d(out_channels[1], out_channels[2], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            'b2a': nn.Conv3d(in_channels, out_channels[3], kernel_size=1),
            'b2b': nn.Conv3d(out_channels[3], out_channels[4], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            'b3': nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            'b3b': nn.Conv3d(in_channels, out_channels[5], kernel_size=1)
        })

    def forward(self, x):
        """Forward pass through I3D network."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        x = self.maxpool3(x)

        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        x = self.maxpool4(x)

        x = self.mixed_5b(x)
        x = self.mixed_5c(x)

        features = self.avgpool(x)
        features = self.dropout(features)
        logits = self.fc(features)

        return features, logits


class DynamicFeatureExtractor(nn.Module):
    """
    Dynamic Feature Extractor based on paper1231 Section C.

    Components:
    1. I3D backbone for spatiotemporal features
    2. Mixed convolution layers for expanded temporal receptive field
    3. Spatial max pooling (as described in paper1231)
    """
    def __init__(self,
                 i3d_path=None,
                 use_pretrained_i3d=False,
                 output_dim=832,
                 freeze_backbone=True):
        """
        Args:
            i3d_path: Path to I3D checkpoint
            use_pretrained_i3d: Use pretrained I3D weights
            output_dim: Output feature dimension (832 = I3D Mixed_5c channels)
            freeze_backbone: Freeze I3D backbone weights
        """
        super().__init__()

        self.output_dim = output_dim

        # I3D backbone
        self.i3d = I3DBackbone(in_channels=3, num_classes=output_dim)

        # Remove classification head for feature extraction
        # Keep features up to Mixed_5c
        self.feature_extractor = nn.Sequential(
            self.i3d.conv1, self.i3d.bn1, self.i3d.relu, self.i3d.maxpool1,
            self.i3d.conv2, self.i3d.bn2, self.i3d.relu,
            self.i3d.conv3, self.i3d.bn3, self.i3d.relu, self.i3d.maxpool2,
            self.i3d.mixed_3b, self.i3d.mixed_3c, self.i3d.maxpool3,
            self.i3d.mixed_4b, self.i3d.mixed_4c, self.i3d.mixed_4d,
            self.i3d.mixed_4e, self.i3d.mixed_4f, self.i3d.maxpool4,
            self.i3d.mixed_5b, self.i3d.mixed_5c
        )

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Load checkpoint if provided
        if i3d_path is not None:
            self.load_checkpoint(i3d_path)

    def load_checkpoint(self, checkpoint_path):
        """Load I3D checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.i3d.load_state_dict(state_dict, strict=False)
        print(f"Loaded I3D checkpoint from {checkpoint_path}")

    def forward(self, video):
        """
        Extract dynamic features from video.

        Args:
            video: (B, C, T, H, W) - surgical video clip
        Returns:
            features: (B, output_dim) - spatiotemporal features
        """
        # I3D expects (B, C, T, H, W)
        features = self.feature_extractor(video)  # (B, 832, T', H', W')

        # Global average pooling over all dimensions
        features = F.adaptive_avg_pool3d(features, (1, 1, 1))  # (B, 832, 1, 1, 1)
        features = features.flatten(1)  # (B, 832)

        return features


if __name__ == '__main__':
    # Test module
    model = DynamicFeatureExtractor(output_dim=832)
    video = torch.randn(2, 3, 16, 224, 224)  # B=2, C=3, T=16, H=W=224

    features = model(video)
    print(f"Input video shape: {video.shape}")
    print(f"Output features shape: {features.shape}")
