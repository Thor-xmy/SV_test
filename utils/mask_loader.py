"""
Mask Loader Module

Handles loading pre-computed surgical instrument masks from disk.

Since SAM3 segmentation is done offline, masks are stored as files
and loaded during training/inference.
"""

import os
import torch
import cv2
import numpy as np
from typing import List, Tuple, Optional


class MaskLoader:
    """
    Load pre-computed surgical instrument masks.

    Masks can be stored as:
    - Individual images: video_id/frame_0001_mask.png
    - Video files: video_id_masks.mp4
    - Numpy arrays: video_id_masks.npy
    """
    def __init__(self,
                 mask_dir,
                 mask_format='png',
                 mask_size=None,
                 normalize=True):
        """
        Args:
            mask_dir: Root directory containing mask files
            mask_format: Format of mask files ('png', 'npy', 'video')
            mask_size: Target size (H, W) - None for original size
            normalize: Normalize masks to [0,1]
        """
        self.mask_dir = mask_dir
        self.mask_format = mask_format
        self.mask_size = mask_size
        self.normalize = normalize

        print(f"MaskLoader initialized:")
        print(f"  Directory: {mask_dir}")
        print(f"  Format: {mask_format}")
        print(f"  Target size: {mask_size}")

    def load_mask(self, video_id, frame_indices=None):
        """
        Load masks for a specific video.

        Args:
            video_id: Video identifier
            frame_indices: List of frame indices to load
                           None -> load all frames

        Returns:
            masks: (T, H, W) or (B, T, H, W)
        """
        mask_path = self._get_mask_path(video_id)

        if self.mask_format == 'npy':
            masks = self._load_npy_mask(mask_path)
        elif self.mask_format == 'video':
            masks = self._load_video_mask(mask_path, frame_indices)
        else:  # png, jpg, etc.
            masks = self._load_image_masks(mask_path, frame_indices)

        # Normalize
        if self.normalize:
            masks = self._normalize_masks(masks)

        # Resize if needed
        if self.mask_size is not None:
            masks = self._resize_masks(masks, self.mask_size)

        return masks

    def _get_mask_path(self, video_id):
        """Get path to mask file(s)."""
        if self.mask_format == 'video':
            return os.path.join(self.mask_dir, f"{video_id}_masks.mp4")
        elif self.mask_format == 'npy':
            return os.path.join(self.mask_dir, f"{video_id}_masks.npy")
        else:
            return os.path.join(self.mask_dir, video_id)

    def _load_npy_mask(self, path):
        """Load masks from numpy array."""
        masks = np.load(path)
        return torch.from_numpy(masks).float()

    def _load_video_mask(self, path, frame_indices):
        """Load masks from video file."""
        cap = cv2.VideoCapture(path)
        frames = []

        if frame_indices is None:
            # Load all frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                masks.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if frame_indices is not None and len(frames) >= len(frame_indices):
                    break
        else:
            # Load specific frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    masks.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        cap.release()

        masks = np.stack(masks, axis=0) / 255.0
        return torch.from_numpy(masks).float()

    def _load_image_masks(self, path, frame_indices):
        """Load masks from individual image files."""
        mask_files = sorted([f for f in os.listdir(path) if f.endswith(self.mask_format)])

        if frame_indices is None:
            frame_indices = list(range(len(mask_files)))

        masks = []
        for idx in frame_indices:
            if idx < len(mask_files):
                mask_path = os.path.join(path, mask_files[idx])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = mask / 255.0
                masks.append(mask)

        masks = np.stack(masks, axis=0)
        return torch.from_numpy(masks).float()

    def _normalize_masks(self, masks):
        """Normalize masks to [0,1]."""
        if masks.max() > 1.0:
            masks = masks / 255.0
        return masks

    def _resize_masks(self, masks, target_size):
        """Resize masks to target size."""
        if len(masks.shape) == 3:  # (T, H, W)
            T, H, W = masks.shape
            target_H, target_W = target_size

            masks_np = masks.numpy()
            resized = np.zeros((T, target_H, target_W), dtype=np.float32)

            for t in range(T):
                resized[t] = cv2.resize(masks_np[t], (target_W, target_H))

            return torch.from_numpy(resized)

        return masks

    def load_batch_masks(self, video_ids, frame_indices=None):
        """
        Load masks for a batch of videos.

        Args:
            video_ids: List of video IDs
            frame_indices: Frame indices to load

        Returns:
            masks: (B, T, H, W) where B = len(video_ids)
        """
        masks_list = []
        max_frames = 0

        # Load each video's masks
        for video_id in video_ids:
            masks = self.load_mask(video_id, frame_indices)
            masks_list.append(masks)
            max_frames = max(max_frames, masks.size(0))

        # Pad to max frames if needed
        padded_masks = []
        for masks in masks_list:
            if masks.size(0) < max_frames:
                pad_size = max_frames - masks.size(0)
                padding = torch.zeros(pad_size, masks.size(1), masks.size(2))
                padded = torch.cat([masks, padding], dim=0)
                padded_masks.append(padded)
            else:
                padded_masks.append(masks)

        return torch.stack(padded_masks, dim=0)


class TemporalMaskSmoother:
    """
    Apply temporal smoothing to masks (as in paper1231).

    Reduces jitter and local instability by smoothing across adjacent frames.
    """
    def __init__(self, window_size=3, method='gaussian'):
        """
        Args:
            window_size: Size of smoothing window
            method: 'gaussian', 'average', or 'median'
        """
        self.window_size = window_size
        self.method = method

        # Precompute Gaussian kernel
        if method == 'gaussian':
            sigma = window_size / 3.0
            x = np.arange(-(window_size//2), window_size//2 + 1)
            kernel = np.exp(-0.5 * (x / sigma) ** 2)
            kernel = kernel / kernel.sum()
            self.kernel = torch.from_numpy(kernel).float()

    def smooth(self, masks):
        """
        Apply temporal smoothing to masks.

        Args:
            masks: (B, T, H, W) or (T, H, W)

        Returns:
            smoothed: Same shape as input
        """
        original_shape = masks.shape

        if masks.dim() == 3:  # (T, H, W)
            masks = masks.unsqueeze(0)  # (1, T, H, W)
            squeeze_output = True
        else:
            squeeze_output = False

        B, T, H, W = masks.shape
        smoothed = torch.zeros_like(masks)

        for b in range(B):
            for h in range(H):
                for w in range(W):
                    sequence = masks[b, :, h, w]  # (T,)

                    if self.method == 'average':
                        smoothed_seq = self._smooth_average(sequence)
                    elif self.method == 'median':
                        smoothed_seq = self._smooth_median(sequence)
                    else:  # gaussian
                        smoothed_seq = self._smooth_gaussian(sequence)

                    smoothed[b, :, h, w] = smoothed_seq

        if squeeze_output:
            return smoothed.squeeze(0)
        return smoothed

    def _smooth_average(self, sequence):
        """Average smoothing."""
        T = sequence.size(0)
        half_window = self.window_size // 2
        smoothed = torch.zeros_like(sequence)

        for t in range(T):
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            smoothed[t] = sequence[start:end].mean()

        return smoothed

    def _smooth_median(self, sequence):
        """Median smoothing."""
        T = sequence.size(0)
        half_window = self.window_size // 2
        smoothed = torch.zeros_like(sequence)

        for t in range(T):
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            smoothed[t] = torch.median(sequence[start:end])

        return smoothed

    def _smooth_gaussian(self, sequence):
        """Gaussian smoothing using 1D convolution."""
        T = sequence.size(0)
        padded = torch.nn.functional.pad(
            sequence.unsqueeze(0).unsqueeze(0),
            (self.window_size//2, self.window_size//2),
            mode='reflect'
        )
        kernel = self.kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(sequence.device)
        smoothed = torch.nn.functional.conv1d(padded, kernel)
        return smoothed.squeeze().squeeze()


if __name__ == '__main__':
    # Test mask loader
    import tempfile

    # Create dummy mask directory
    tmp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_dir, 'video_001'), exist_ok=True)

    # Create dummy masks
    for i in range(10):
        mask = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        cv2.imwrite(os.path.join(tmp_dir, 'video_001', f'frame_{i:04d}.png'), mask)

    # Test loader
    loader = MaskLoader(tmp_dir, mask_format='png', mask_size=(224, 224))
    masks = loader.load_mask('video_001')

    print(f"Loaded masks shape: {masks.shape}")
    print(f"Mask value range: [{masks.min():.2f}, {masks.max():.2f}]")

    # Test smoother
    smoother = TemporalMaskSmoother(window_size=3, method='gaussian')
    smoothed_masks = smoother.smooth(masks)
    print(f"Smoothed masks shape: {smoothed_masks.shape}")
