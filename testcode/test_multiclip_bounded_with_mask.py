"""
Complete Test for Multi-Clip Bounded Pipeline with Mask-Guided Attention

测试完整的pipeline，包括：
1. Multi-clip静态特征提取
2. Multi-clip动态特征提取
3. Mask引导attention（每个clip独立）
4. Per-clip融合
5. 时序扁平化
6. 有界回归（Sigmoid）
"""

import sys
import torch
import numpy as np

sys.path.insert(0, '/home/thor/surgical_qa_model')

print("="*70)
print("MULTI-CLIP BOUNDED PIPELINE WITH MASK-GUIDED ATTENTION")
print("="*70)

# Test 1: Test mask-guided attention module
print("\n[1/6] Testing MaskGuidedAttentionMultiClip...")
from models.mask_guided_attention_multiclip import MaskGuidedAttentionMultiClip

mask_attention = MaskGuidedAttentionMultiClip(
    enable_temporal_smoothing=True,
    clip_length=16,
    clip_stride=10
)

# Create test inputs
B, num_clips, C, T_feat, H_feat, W_feat = 2, 10, 1024, 2, 4, 4
dynamic_features_per_clip = torch.randn(B, num_clips, C, T_feat, H_feat, W_feat)

# Original masks (corresponding to full video)
# Assume original video has 100 frames with clip_length=16, stride=10
T_original = 100
masks = torch.randint(0, 2, (B, T_original, 224, 224)).float()

# Forward pass
masked_features, attention_maps = mask_attention(
    dynamic_features_per_clip,
    masks,
    num_clips=num_clips
)

print(f"  Input dynamic features: {dynamic_features_per_clip.shape}")
print(f"  Original masks: {masks.shape}")
print(f"  Masked features: {masked_features.shape}")
print(f"  Attention maps: {attention_maps.shape}")

assert masked_features.shape == (B, num_clips, C)
assert attention_maps.shape == (B, num_clips, T_feat, H_feat, W_feat)
print("  ✓ Test 1 PASSED")

# Test 2: Test dynamic extractor with mask-guided attention
print("\n[2/6] Testing DynamicFeatureMultiClipWithMask...")
from models.dynamic_feature_extractor_multiclip_with_mask import DynamicFeatureMultiClipWithMask

dynamic_model = DynamicFeatureMultiClipWithMask(
    output_dim=1024,
    clip_length=16,
    clip_stride=10,
    max_clips=None,
    use_mask_guided_attention=True,
    enable_temporal_smoothing=True
)

# Test with video
video = torch.randn(2, 3, 100, 112, 112)

# Test with mask-guided attention
per_clip_features, num_clips, attention_maps = dynamic_model.extract_multiclip_features(
    video,
    masks=masks
)

print(f"  Input video: {video.shape}")
print(f"  Original masks: {masks.shape}")
print(f"  Per-clip features: {per_clip_features.shape}")
print(f"  Number of clips: {num_clips}")
print(f"  Attention maps: {attention_maps.shape}")

assert per_clip_features.shape == (2, num_clips, 1024)
assert attention_maps.shape == (2, num_clips, 2, 4, 4)  # I3D feature map size
print("  ✓ Test 2 PASSED")

# Test 3: Test without mask-guided attention
print("\n[3/6] Testing DynamicFeatureMultiClipWithMask (without mask)...")
per_clip_features_no_mask, num_clips_no_mask, attention_maps_no_mask = \
    dynamic_model.extract_multiclip_features(
        video,
        masks=None  # No mask
    )

print(f"  Per-clip features (no mask): {per_clip_features_no_mask.shape}")
print(f"  Number of clips: {num_clips_no_mask}")
print(f"  Attention maps: {attention_maps_no_mask}")

assert per_clip_features_no_mask.shape == (2, num_clips_no_mask, 1024)
assert attention_maps_no_mask is None
print("  ✓ Test 3 PASSED")

# Test 4: Test complete model with mask-guided attention
print("\n[4/6] Testing Complete Model with Mask-Guided Attention...")
from models.surgical_qa_model_multiclip_bounded_with_mask import SurgicalQAModelMultiClipBoundedWithMask

config = {
    'static_dim': 512,
    'dynamic_dim': 1024,
    'clip_length': 16,
    'clip_stride': 10,
    'max_clips': None,
    'use_pretrained': False,
    'freeze_backbone': True,
    'use_mask_guided_attention': True,
    'enable_temporal_smoothing': True,
    'score_min': 6.0,
    'score_max': 30.0,
    'keyframe_strategy': 'middle'
}

model = SurgicalQAModelMultiClipBoundedWithMask(config)

# Test forward pass with mask
score, features = model(
    video,
    masks=masks,
    return_features=True,
    return_attention=True
)

print(f"  Input video: {video.shape}")
print(f"  Input masks: {masks.shape}")
print(f"  Output score: {score.shape}")
print(f"  Output score range: [{score.min().item():.4f}, {score.max().item():.4f}]")
print(f"\nFeatures:")
print(f"  Static per-clip: {features['static_per_clip'].shape}")
print(f"  Dynamic per-clip: {features['dynamic_per_clip'].shape}")
print(f"  Fused per-clip: {features['fused_per_clip'].shape}")
print(f"  Temporal features: {features['temporal_features'].shape}")
print(f"  Number of clips: {features['num_clips']}")
print(f"  Attention maps: {features['attention_maps'].shape}")

assert score.shape == (2, 1)
assert score.min() >= 0.0 and score.max() <= 1.0
assert features['attention_maps'] is not None
print("  ✓ Test 4 PASSED")

# Test 5: Test without masks (should still work)
print("\n[5/6] Testing Complete Model (without masks)...")
score_no_mask = model(video)

print(f"  Output score (no mask): {score_no_mask.shape}")
print(f"  Output score range: [{score_no_mask.min().item():.4f}, {score_no_mask.max().item():.4f}]")

assert score_no_mask.shape == (2, 1)
assert score_no_mask.min() >= 0.0 and score_no_mask.max() <= 1.0
print("  ✓ Test 5 PASSED")

# Test 6: Test score denormalization
print("\n[6/6] Testing Score Denormalization...")
score_6to30 = model.denormalize_score(score, target_min=6.0, target_max=30.0)
score_1to10 = model.denormalize_score(score, target_min=1.0, target_max=10.0)

print(f"  Input normalized scores: {score.squeeze().tolist()}")
print(f"  Denormalized [6,30]: {score_6to30.squeeze().tolist()}")
print(f"  Denormalized [1, 10]: {score_1to10.squeeze().tolist()}")

assert all(6.0 <= s <= 30.0 for s in score_6to30.squeeze().tolist())
assert all(1.0 <= s <= 10.0 for s in score_1to10.squeeze().tolist())
print("  ✓ Test 6 PASSED")

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)

print("\nSummary:")
print("  ✓ Mask-Guided Attention Module (Multi-Clip)")
print("  ✓ Dynamic Extractor with Mask-Guided Attention")
print("  ✓ Complete Model with Mask-Guided Attention")
print("  ✓ Mask handling (with and without)")
print("  ✓ Per-clip fusion")
print("  ✓ Temporal flattening")
print("  ✓ Bounded regression with Sigmoid")
print("  ✓ Score denormalization")

print("\nThe complete multi-clip bounded pipeline with mask-guided attention is ready!")
print("="*70)
