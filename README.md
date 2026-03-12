# Surgical Video Quality Assessment Model

A PyTorch implementation for surgical video skill assessment based on paper1231 framework.

## Overview

This project implements a surgical video quality assessment model that evaluates surgical skill levels from video clips. The model architecture is inspired by paper1231 but adapted for single-video scoring (without contrastive learning).

### Key Differences from paper1231

| Aspect | paper1231 | This Model |
|---------|-----------|-------------|
| Framework | Contrastive learning with query/reference video pairs | Single video direct regression |
| Cross-attention | Yes (between query and reference) | No |
| Score prediction | `y_q = y_r + CR(E(V_q), E(V_r))` | `y = R(Concat(F_static, F_dynamic))` |
| Training objective | Learn relative score differences | Learn absolute scores |

## Architecture

```
Input Video (B, C, T, H, W)
    ├─→ ResNet-34 (Static Feature Extractor) ──→ Static Features A
    │   ├─ ResNet Backbone
    │   └─ Multi-scale Downsampling (s=2,4,8)
    │
    ├─→ I3D (Dynamic Feature Extractor) ──→ Dynamic Features C
    │   └─ Mixed 3D Convolutions
    │
    └─→ SAM3 Masks (Offline) ──→ Instrument Masks B
        └─→ Read from pre-computed files

Mask-Guided Attention Module
    ├─ Input: B (masks) and C (dynamic features)
    ├─ Generate attention from mask features
    ├─ Fuse with learned attention
    └─ Apply to features → Masked Dynamic Features D

Feature Fusion & Regression
    ├─ Concat(A, D) ──→ Fused Features
    └─ FC Layers ──→ Score y
```

## Project Structure

```
surgical_qa_model/
├── models/                          # Model architectures
│   ├── static_feature_extractor.py   # ResNet-34 based static features
│   ├── dynamic_feature_extractor.py  # I3D based dynamic features
│   ├── mask_guided_attention.py     # Mask-guided attention module
│   ├── surgical_qa_model.py        # Main model class
│   └── __init__.py
│
├── utils/                           # Utility functions
│   ├── data_loader.py              # Dataset and dataloader
│   ├── mask_loader.py              # Mask loading utilities
│   ├── metrics.py                  # Evaluation metrics (MAE, SRCC, etc.)
│   ├── training.py                 # Training loops and utilities
│   └── __init__.py
│
├── configs/                         # Configuration files
│   └── default.yaml
│
├── datasets/                        # Dataset storage (create this)
│   ├── videos/
│   │   ├── video_001.mp4
│   │   └── ...
│   ├── masks/
│   │   ├── video_001/
│   │   │   ├── frame_0000_mask.png
│   │   │   └── ...
│   │   └── ...
│   └── annotations
│       └── annotations.json
│
├── checkpoints/                     # Saved model checkpoints
├── logs/                           # Training logs and TensorBoard
└── train.py                        # Main training script
```

## Dataset Preparation

### 1. Video Files

Place your surgical videos in `datasets/videos/`:
```
datasets/videos/
├── video_001.mp4
├── video_002.mp4
└── ...
```

### 2. Mask Files (Offline Generation using SAM3)

Generate surgical instrument masks using SAM3 and place them in `datasets/masks/`:

```python
# Example: Generate masks using SAM3
from sam3 import build_sam3_video_model

model = build_sam3_video_model(device='cuda')
inference_state = model.init_state('path/to/video.mp4')

# Add point prompts for instrument detection
model.add_points_prompts(
    inference_state,
    frame_idx=0,
    point_coords=[[x, y]],  # Positive point on instrument
    point_labels=[1]
)

# Propagate masks through video
masks = model.propagate_in_video(inference_state)

# Save masks
for frame_idx, mask in enumerate(masks):
    cv2.imwrite(f'datasets/masks/video_001/frame_{frame_idx:04d}_mask.png', mask)
```

**Mask File Structure**:
```
datasets/masks/
├── video_001/
│   ├── frame_0000_mask.png
│   ├── frame_0001_mask.png
│   └── ...
├── video_002/
│   └── ...
└── ...
```

### 3. Annotation File

Create `datasets/annotations.json` with ground truth scores:

```json
{
    "video_001": {
        "score": 8.5,
        "duration": 120,
        "surgeon_id": "S001"
    },
    "video_002": {
        "score": 7.2,
        "duration": 115,
        "surgeon_id": "S002"
    }
}
```

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy opencv-python
pip install pyyaml tqdm
pip install scipy tensorboard
pip install huggingface-hub  # For SAM3 if needed
```

### SAM3 Installation

Follow the official SAM3 installation instructions from `/root/autodl-tmp/sam3`.

## Usage

### Training

```bash
# Basic training
python train.py --config configs/default.yaml --data_root /path/to/dataset

# With custom settings
python train.py \
    --config configs/default.yaml \
    --data_root /path/to/dataset \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 0.0001 \
    --gpus 0

# Resume training
python train.py --resume checkpoints/checkpoint_epoch_50.pth --resume_optim

# Evaluation only
python train.py --config configs/default.yaml --evaluate --gpus 0
```

### Key Command Line Arguments

- `--config`: Path to config file (default: `configs/default.yaml`)
- `--data_root`: Root directory of dataset
- `--batch_size`: Batch size for training (default: 8)
- `--epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 0.0001)
- `--optimizer`: Optimizer type: sgd, adam, adamw (default: adam)
- `--freeze_backbone`: Freeze backbone weights (default: True)
- `--use_amp`: Use automatic mixed precision (default: False)
- `--resume`: Path to checkpoint to resume from
- `--evaluate`: Evaluation mode only
- `--gpus`: GPU IDs to use (default: "0")

## Model Components

### 1. Static Feature Extractor (ResNet-34)

Captures high-level global features and tissue state variations.

**Features**:
- ResNet-34 backbone with 4 layers
- Multi-scale downsampling (strides: 2, 4, 8)
- Frame sampling strategy: middle, average, first, last

### 2. Dynamic Feature Extractor (I3D)

Extracts spatiotemporal features for instrument manipulation dynamics.

**Features**:
- Inception-3D backbone
- Mixed 3D convolutions for expanded temporal receptive field
- Short-term clip processing with overlapping windows

### 3. Mask-Guided Attention

Focuses model attention on surgical instrument regions.

**Implementation**:
- Generates mean and max attention from mask features
- Learnable aggregation function
- Temporal smoothing to reduce mask jitter
- Element-wise multiplication: `F_dy = F_clip * (A + I)`

### 4. Fusion Regressor

Concatenates static and dynamic features, regresses to score.

**Architecture**:
- Input: `[static_dim, dynamic_dim] = [512, 832]`
- Hidden layers: `[1024, 512, 256, 128]`
- Output: `[1]` (quality score)

## Evaluation Metrics

Following paper1231:

1. **Mean Absolute Error (MAE)**:
   ```
   MAE = (1/n) * sum(|y_pred - y_gt|)
   ```

2. **Spearman Rank Correlation Coefficient (SRCC)**:
   ```
   SRCC = covariance / (std_pred * std_gt)
   ```

3. **Pearson Correlation Coefficient (PCC)**:
   ```
   PCC = Pearson correlation between predictions and ground truth
   ```

4. **Normalized MAE (NMAE)**:
   ```
   NMAE = MAE / (score_range) * 100
   ```

## Advanced Features

### Gradient Accumulation

For larger effective batch sizes:
```bash
python train.py --config configs/default.yaml --accumulation_steps 4
```

### Automatic Mixed Precision (AMP)

For faster training with reduced memory:
```bash
python train.py --config configs/default.yaml --use_amp
```

### Multi-GPU Training

For training on multiple GPUs:
```bash
python train.py --config configs/default.yaml --gpus 0,1,2,3
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `--batch_size 4`
2. Enable AMP: `--use_amp`
3. Use gradient accumulation
4. Freeze backbones: `--freeze_backbone`

### Mask Loading Issues

1. Verify mask files exist in `datasets/masks/video_XXX/`
2. Check mask dimensions match video frames
3. Ensure mask values are in [0, 1] range

### Poor Convergence

1. Check learning rate (try lower: 1e-4 or 1e-5)
2. Enable gradient clipping: `--clip_grad_norm 1.0`
3. Add early stopping: `--early_stopping --patience 15`

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_name_202X_surgical,
  title={Surgical Video Quality Assessment via Static and Dynamic Feature Fusion},
  author={Your Name},
  journal={ArXiv preprint},
  year={202X}
}
```

Also cite the original paper1231:

```bibtex
@article{paper1231_202X_fine_grained,
  title={Fine-Grained Contrastive Learning for Robotic Surgery Skill Assessment Through Joint Instrument-Tissue Modeling},
  author={Paper1231 Authors},
  journal={Original Paper1231},
  year={202X}
}
```

## License

This project is for research and educational purposes.

## Contact

For questions or issues, please open an issue on the project repository.
