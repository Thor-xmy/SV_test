"""
Test the modified Multi-Clip Bounded Pipeline with proper data splitting.

This test verifies:
1. Data loader correctly loads frame sequences
2. Data splitting (train/val/test) works correctly
3. Model forward pass works
4. Loss computation works
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

# Import the new data loader
from utils.data_loader_video_level_frames import create_dataloader_with_split
from models.surgical_qa_model_multiclip_bounded import SurgicalQAModelMultiClipBounded


def test_data_loading():
    """Test data loading and splitting."""
    print("="*70)
    print("TEST 1: Data Loading and Splitting")
    print("="*70)

    data_root = '/root/autodl-tmp/SV_test/data'

    # Test all subsets
    subsets = ['train', 'val', 'test']
    dataloaders = {}

    for subset in subsets:
        print(f"\nCreating {subset} dataloader...")
        dataloader = create_dataloader_with_split(
            data_root=data_root,
            batch_size=2,
            num_workers=0,
            spatial_size=224,
            subset=subset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            split_seed=42,
            is_train=(subset == 'train')
        )
        dataloaders[subset] = dataloader

        # Load first batch
        print(f"Loading first batch from {subset}...")
        for batch in dataloader:
            print(f"  Batch keys: {list(batch.keys())}")
            print(f"  Video shape: {batch['video'].shape}")
            print(f"  Mask shape: {batch['masks'].shape if batch['masks'] is not None else 'None'}")
            print(f"  Score shape: {batch['score'].shape}")
            print(f"  Score range: [{batch['score'].min().item():.4f}, {batch['score'].max().item():.4f}]")
            print(f"  Video IDs: {batch['video_id']}")
            break

    return dataloaders


def test_model_forward():
    """Test model forward pass."""
    print("\n" + "="*70)
    print("TEST 2: Model Forward Pass")
    print("="*70)

    data_root = '/root/autodl-tmp/SV_test/data'

    # Create dataloader
    dataloader = create_dataloader_with_split(
        data_root=data_root,
        batch_size=2,
        num_workers=0,
        spatial_size=224,
        subset='train',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        split_seed=42,
        is_train=True
    )

    # Create model
    config = {
        'static_dim': 512,
        'dynamic_dim': 1024,
        'clip_length': 16,
        'clip_stride': 10,
        'max_clips': 5,  # Limit to 5 clips for testing
        'expected_clips': 5,
        'use_pretrained': False,
        'freeze_backbone': True,
        'use_mixed_conv': True,
        'score_min': 6.0,
        'score_max': 30.0,
        'keyframe_strategy': 'middle',
        'regressor_hidden_dims': [1024, 512, 256, 128]
    }

    print("\nCreating model...")
    model = SurgicalQAModelMultiClipBounded(config)

    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    print("\nTesting forward pass on first batch...")
    for batch in dataloader:
        video = batch['video'].to(device)
        masks = batch['masks']
        if masks is not None:
            masks = masks.to(device)
        score_gt = batch['score'].to(device)

        print(f"Input video shape: {video.shape}")
        print(f"Mask shape: {masks.shape if masks is not None else 'None'}")
        print(f"Ground truth score shape: {score_gt.shape}")

        with torch.no_grad():
            score_pred = model(video, masks)

        print(f"Predicted score shape: {score_pred.shape}")
        print(f"Predicted score range: [{score_pred.min().item():.4f}, {score_pred.max().item():.4f}]")
        print(f"Predicted score values: {score_pred.squeeze(-1).cpu().numpy()}")

        # Test loss computation
        loss, loss_dict = model.compute_loss(score_pred, score_gt)
        print(f"\nLoss: {loss.item():.4f}")
        print(f"Loss dict: {loss_dict}")

        # Test denormalization
        score_6to30 = model.denormalize_score(score_pred)
        score_1to10 = model.denormalize_score(score_pred, target_min=1.0, target_max=10.0)

        print(f"\nDenormalized to [6,30]: {score_6to30.squeeze(-1).cpu().numpy()}")
        print(f"Denormalized to [1,10]: {score_1to10.squeeze(-1).cpu().numpy()}")

        break

    print("\n✓ Model forward pass test passed!")
    return True


def test_data_splitting_correctness():
    """Verify that train/val/test splits are correct."""
    print("\n" + "="*70)
    print("TEST 3: Data Splitting Correctness")
    print("="*70)

    data_root = '/root/autodl-tmp/SV_test/data'

    # Create all dataloaders
    all_video_ids = set()

    for subset in ['train', 'val', 'test']:
        dataloader = create_dataloader_with_split(
            data_root=data_root,
            batch_size=48,  # Load all at once
            num_workers=0,
            spatial_size=224,
            subset=subset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            split_seed=42,
            is_train=False
        )

        # Get all video IDs
        for batch in dataloader:
            video_ids = batch['video_id']
            if isinstance(video_ids, list):
                all_video_ids.update(video_ids)
            else:
                all_video_ids.update([video_ids])

        print(f"Total unique videos in {subset}: {len(all_video_ids)}")

    print(f"\nTotal unique videos across all subsets: {len(all_video_ids)}")
    print(f"Expected total: 48")

    if len(all_video_ids) == 48:
        print("✓ All 48 videos are covered by the splits!")
    else:
        print(f"⚠ Warning: Expected 48 videos, found {len(all_video_ids)}")
        missing = 48 - len(all_video_ids)
        print(f"  Missing: {missing} videos")

    return len(all_video_ids) == 48


def test_consistency_across_epochs():
    """Test that data loader returns consistent splits across multiple epochs."""
    print("\n" + "="*70)
    print("TEST 4: Consistency Across Epochs")
    print("="*70)

    data_root = '/root/autodl-tmp/SV_test/data'

    # Create train dataloader
    dataloader = create_dataloader_with_split(
        data_root=data_root,
        batch_size=48,
        num_workers=0,
        spatial_size=224,
        subset='train',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        split_seed=42,
        is_train=True
    )

    # Collect video IDs from first epoch
    first_epoch_ids = []
    for batch in dataloader:
        video_ids = batch['video_id']
        if isinstance(video_ids, list):
            first_epoch_ids.extend(video_ids)
        else:
            first_epoch_ids.append(video_ids)

    print(f"First epoch: {len(first_epoch_ids)} videos")

    # Collect video IDs from second epoch (reshuffle)
    second_epoch_ids = []
    for batch in dataloader:
        video_ids = batch['video_id']
        if isinstance(video_ids, list):
            second_epoch_ids.extend(video_ids)
        else:
            second_epoch_ids.append(video_ids)

    print(f"Second epoch: {len(second_epoch_ids)} videos")

    # Check that both epochs have the same videos (different order)
    if set(first_epoch_ids) == set(second_epoch_ids):
        print("✓ Both epochs have the same video set (different order expected)")
    else:
        print("⚠ Warning: Video sets differ across epochs")
        print(f"  First only: {set(first_epoch_ids) - set(second_epoch_ids)}")
        print(f"  Second only: {set(second_epoch_ids) - set(first_epoch_ids)}")

    return set(first_epoch_ids) == set(second_epoch_ids)


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MODIFIED MULTI-CLIP BOUNDED PIPELINE TEST")
    print("="*70)

    results = {}

    # Test 1: Data loading
    try:
        test_data_loading()
        results['data_loading'] = 'PASS'
    except Exception as e:
        print(f"⚠ Test 1 failed: {e}")
        results['data_loading'] = f'FAIL: {e}'

    # Test 2: Model forward pass
    try:
        test_model_forward()
        results['model_forward'] = 'PASS'
    except Exception as e:
        print(f"⚠ Test 2 failed: {e}")
        results['model_forward'] = f'FAIL: {e}'

    # Test 3: Data splitting correctness
    try:
        passed = test_data_splitting_correctness()
        results['data_splitting'] = 'PASS' if passed else 'FAIL'
    except Exception as e:
        print(f"⚠ Test 3 failed: {e}")
        results['data_splitting'] = f'FAIL: {e}'

    # Test 4: Consistency across epochs
    try:
        passed = test_consistency_across_epochs()
        results['consistency'] = 'PASS' if passed else 'FAIL'
    except Exception as e:
        print(f"⚠ Test 4 failed: {e}")
        results['consistency'] = f'FAIL: {e}'

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, result in results.items():
        status = "✓ PASS" if result == 'PASS' else "✗ FAIL"
        print(f"{test_name:25} {status}")
        if result != 'PASS':
            print(f"  Details: {result}")

    all_passed = all(v == 'PASS' for v in results.values())
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED!")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
