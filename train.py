"""
Training Script for Surgical Quality Assessment Model

Usage:
    python train.py --config configs/default.yaml --gpus 0,1
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SurgicalQAModel, build_model
from utils import (
    SurgicalQADataLoader,
    AverageMeter,
    train_epoch,
    validate,
    compute_metrics,
    TrainingLogger,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Surgical QA Model')

    # Config and paths
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--data_root', type=str, default=None,
                       help='Root directory of dataset')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for logs and checkpoints')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default for backbone)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay (paper requires 0)')
    parser.add_argument('--static_lr', type=float, default=1e-3,
                       help='Learning rate for static module (paper: 1e-3)')
    parser.add_argument('--backbone_lr', type=float, default=1e-4,
                       help='Learning rate for backbone (paper: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'adam', 'adamw'], help='Optimizer')

    # Model parameters
    parser.add_argument('--static_dim', type=int, default=512,
                       help='Static feature dimension')
    parser.add_argument('--dynamic_dim', type=int, default=1024,  # Standard I3D outputs 1024 channels
                       help='Dynamic feature dimension (1024 for standard I3D)')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                       help='Freeze backbone weights')
    parser.add_argument('--use_mask_loss', action='store_true', default=True,
                       help='Use mask supervision loss')

    # AMP and mixed precision
    parser.add_argument('--use_amp', action='store_true', default=False,
                       help='Use automatic mixed precision')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                       help='Gradient clipping norm (0 to disable)')

    # Checkpoint and resuming
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--resume_optim', action='store_true', default=False,
                       help='Resume optimizer state')
    parser.add_argument('--evaluate', action='store_true', default=False,
                       help='Evaluate only')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained backbones')

    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', default=True,
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (paper requires 10)')
    parser.add_argument('--min_delta', type=float, default=0.001,
                       help='Minimum delta for improvement')

    # Device
    parser.add_argument('--gpus', type=str, default='0',
                       help='GPU IDs to use (comma-separated)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Logging
    parser.add_argument('--print_freq', type=int, default=10,
                       help='Print frequency (steps)')
    parser.add_argument('--save_freq', type=int, default=1,
                       help='Save checkpoint frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=10,
                       help='TensorBoard log frequency (steps)')

    return parser.parse_args()


def load_config(config_path, args):
    """Load and merge configuration."""
    # Load yaml config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config file not found: {config_path}")
        config = {}

    # Override with command line args
    config_updates = {
        'data_root': args.data_root,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'optimizer': args.optimizer,
        'static_dim': args.static_dim,
        'dynamic_dim': args.dynamic_dim,
        'freeze_backbone': args.freeze_backbone,
        'use_mask_loss': args.use_mask_loss,
        'use_amp': args.use_amp,
        'clip_grad_norm': args.clip_grad_norm,
        'early_stopping': args.early_stopping,
        'patience': args.patience,
        'min_delta': args.min_delta,
        'pretrained': args.pretrained,
        'epochs': args.epochs,
        'print_freq': args.print_freq,
        'save_freq': args.save_freq,
        'log_freq': args.log_freq,
        'backbone_lr': args.backbone_lr,  # NEW
        'static_lr': args.static_lr          # NEW
    }

    # Update config (command line overrides yaml)
    for key, value in config_updates.items():
        if value is not None:  # Don't override with None
            config[key] = value

    return config


def setup_device(gpus):
    """Setup device and multi-GPU."""
    gpu_list = [int(x) for x in gpus.split(',')]

    if torch.cuda.is_available():
        if len(gpu_list) == 1:
            device = torch.device(f'cuda:{gpu_list[0]}')
            print(f"Using single GPU: {gpu_list[0]}")
        else:
            device = torch.device(f'cuda:{gpu_list[0]}')
            print(f"Using multi-GPU: {gpu_list}")
            print("Note: Multi-GPU not fully implemented yet")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def build_optimizer(model, config):
    """
    Build optimizer with differential learning rates (paper setup).

    Paper requirements:
    - Backbone (I3D + ResNet): 1e-4
    - Static module (fusion regressor): 1e-3
    - Weight decay: 0
    """
    # 论文要求的不同学习率
    # Paper: "I3D backbone network" → 1e-4, "static feature extraction module (ResNet)" → 1e-3
    backbone_lr = config.get('backbone_lr', 1e-4)      # 0.0001 (for I3D dynamic backbone)
    static_lr = config.get('static_lr', 1e-3)          # 0.001 (for ResNet static module)
    weight_decay = config.get('weight_decay', 0.0)    # 论文要求0

    # 分组参数：
    # - backbone_params: I3D网络 (dynamic_extractor) → backbone_lr (1e-4)
    # - other_params: ResNet (static_extractor) + fusion_regressor → static_lr (1e-3)
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            # I3D动态骨干网络 → backbone_lr (1e-4)
            if 'dynamic_extractor' in name:
                backbone_params.append(param)
            else:
                # ResNet静态提取器 + 回归模块 → static_lr (1e-3)
                other_params.append(param)

    # 构建参数组
    param_groups = []

    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'name': 'backbone'
        })

    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': static_lr,
            'name': 'static_module'
        })

    # 如果没有可训练参数
    if len(param_groups) == 0:
        raise ValueError("No trainable parameters found!")

    # 构建优化器
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            param_groups,
            weight_decay=weight_decay
        )
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    print(f"Optimizer: {config['optimizer']}")
    print(f"  Backbone LR: {backbone_lr}")
    print(f"  Static Module LR: {static_lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Backbone params: {len(backbone_params)}")
    print(f"  Other params: {len(other_params)}")

    return optimizer


def build_scheduler(optimizer, config):
    """Build learning rate scheduler."""
    # Cosine annealing
    T_max = config.get('epochs', 100)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=1e-6
    )

    return scheduler


def train(model, dataloaders, optimizer, scheduler, criterion, device,
          config, logger, tensorboard_writer=None):
    """
    Main training loop.

    Args:
        model: Neural network model
        dataloaders: Dict of dataloaders
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Training device
        config: Configuration dict
        logger: TrainingLogger instance
        tensorboard_writer: TensorBoard writer

    Returns:
        best_metrics: Best validation metrics
    """
    # Setup
    start_epoch = 0
    best_val_loss = float('inf')
    grad_scaler = GradScaler() if config['use_amp'] else None

    # Early stopping
    if config['early_stopping']:
        early_stopping = EarlyStopping(
            patience=config.get('patience', 15),
            min_delta=config.get('min_delta', 0.001),
            mode='min'
        )

    # Checkpoint saving
    checkpoint_dir = os.path.join(config['output_dir'], 'checkpoints')
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print("="*60)

    for epoch in range(start_epoch, config['epochs']):
        # Train for one epoch
        train_loss, train_metrics = train_epoch(
            model=model,
            dataloader=dataloaders['train'],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            grad_scaler=grad_scaler,
            use_amp=config['use_amp'],
            log_frequency=config['print_freq'],
            verbose=True
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Validate
        val_loss, val_metrics = validate(
            model=model,
            dataloader=dataloaders['val'],
            criterion=criterion,
            device=device,
            epoch=epoch,
            verbose=True,
            return_predictions=True
        )

        # Compute metrics if we have predictions
        if val_metrics['predictions'] is not None:
            predictions = val_metrics['predictions']
            targets = val_metrics['targets']

            eval_metrics = compute_metrics(predictions, targets, verbose=False)

            # Log to TensorBoard
            if tensorboard_writer is not None:
                global_step = epoch * len(dataloaders['train'])

                # Training metrics
                tensorboard_writer.add_scalar('train/loss', train_metrics['train_loss'], global_step)
                tensorboard_writer.add_scalar('train/score_loss', train_metrics['train_score_loss'], global_step)
                tensorboard_writer.add_scalar('train/learning_rate', current_lr, global_step)

                # Validation metrics
                tensorboard_writer.add_scalar('val/loss', val_metrics['val_loss'], epoch)
                tensorboard_writer.add_scalar('val/score_loss', val_metrics['val_score_loss'], epoch)
                tensorboard_writer.add_scalar('val/mae', eval_metrics['mae'], epoch)
                tensorboard_writer.add_scalar('val/srcc', eval_metrics['srcc'], epoch)
                tensorboard_writer.add_scalar('val/pcc', eval_metrics['pcc'], epoch)
                tensorboard_writer.add_scalar('val/nmae', eval_metrics['nmae'], epoch)

        # Prepare log entry
        train_log = {
            'loss': train_metrics['train_loss'],
            'score_loss': train_metrics['train_score_loss'],
            'mask_loss': train_metrics.get('train_mask_loss', 0.0),
            'epoch_time': train_metrics['epoch_time'],
            'learning_rate': current_lr
        }

        val_log = {
            'loss': val_metrics['val_loss'],
            'score_loss': val_metrics['val_score_loss'],
            'mask_loss': val_metrics.get('val_mask_loss', 0.0)
        }

        if val_metrics['predictions'] is not None:
            val_log.update({
                'mae': eval_metrics['mae'],
                'srcc': eval_metrics['srcc'],
                'pcc': eval_metrics['pcc'],
                'nmae': eval_metrics['nmae'],
                'rmse': eval_metrics['rmse']
            })

        # Log to file
        logger.log_epoch(epoch, train_log, val_log)

        # Save checkpoint
        if (epoch + 1) % config['save_freq'] == 0:
            checkpoint_metrics = {
                'train': train_log,
                'val': val_log
            }
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=checkpoint_metrics,
                save_dir=checkpoint_dir,
                filename=f'checkpoint_epoch_{epoch}.pth',
                is_best=(val_metrics['val_loss'] < best_val_loss)
            )

        # Update best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            print(f"  *** New best validation loss: {best_val_loss:.4f} ***")

        # Early stopping
        if config['early_stopping']:
            if early_stopping(val_metrics['val_loss']):
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

    # Save training history
    logger.save_history()

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)

    return {
        'best_val_loss': best_val_loss,
        'final_epoch': epoch
    }


def evaluate(model, dataloader, criterion, device, config):
    """
    Evaluate model on test set.

    Args:
        model: Neural network model
        dataloader: Test data loader
        criterion: Loss function
        device: Evaluation device
        config: Configuration dict
    """
    print("\n" + "="*60)
    print("EVALUATION MODE")
    print("="*60)

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            video = batch['frames'].to(device)
            score_gt = batch['score'].to(device)

            # Load masks
            if 'masks' in batch and batch['masks'] is not None:
                masks = batch['masks'].to(device)
            else:
                masks = None

            # Forward pass
            score_pred, _ = model(video, masks)

            # Store predictions
            all_predictions.append(score_pred.detach().cpu())
            all_targets.append(score_gt.detach().cpu())

    # Combine predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = compute_metrics(
        all_predictions.numpy(),
        all_targets.numpy(),
        verbose=True
    )

    return metrics


def main():
    """Main entry point."""
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Random seed set to: {args.seed}")

    # Load config
    config = load_config(args.config, args)
    config['output_dir'] = args.output_dir
    config['resume'] = args.resume
    config['resume_optim'] = args.resume_optim

    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'logs'), exist_ok=True)

    # Setup device
    device = setup_device(args.gpus)

    # Build model
    print("\nBuilding model...")
    model = build_model(config)
    model = model.to(device)
    model.count_parameters()

    # Load checkpoint if resuming
    if config['resume'] is not None:
        start_epoch, _ = load_checkpoint(
            config['resume'],
            model,
            optimizer=None,
            device=device
        )
        config['start_epoch'] = start_epoch + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        config['start_epoch'] = 0

    # Build optimizer
    optimizer = build_optimizer(model, config)

    # Load optimizer state if resuming
    if config['resume'] is not None and config['resume_optim']:
        checkpoint = torch.load(config['resume'], map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Resumed optimizer state")

    # Build scheduler
    scheduler = build_scheduler(optimizer, config)

    # Build dataloaders
    print("\nLoading datasets...")
    dataloaders = SurgicalQADataLoader(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        dataset_kwargs={
            'clip_length': 16,
            'clip_stride': 10,
            'spatial_size': 112,  # Paper requires 112x112
            'normalize': True,
            'use_mask': True,
            # Data augmentation (paper requirements)
            'horizontal_flip_prob': 0.5,  # Random horizontal flipping
            'enable_rotation': True,          # Random rotation
            'is_train': True  # Enable augmentation during training
        }
    )

    # Setup logging
    logger = TrainingLogger(
        log_dir=os.path.join(config['output_dir'], 'logs'),
        log_file=f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

    tensorboard_writer = SummaryWriter(os.path.join(config['output_dir'], 'logs'))

    # Evaluation only
    if args.evaluate:
        metrics = evaluate(
            model=model,
            dataloader=dataloaders.get_loader('test'),
            criterion=None,
            device=device,
            config=config
        )
        return

    # Training
    criterion = nn.MSELoss()

    train(
        model=model,
        dataloaders={
            'train': dataloaders.get_loader('train'),
            'val': dataloaders.get_loader('val'),
            'test': dataloaders.get_loader('test')
        },
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config,
        logger=logger,
        tensorboard_writer=tensorboard_writer
    )


if __name__ == '__main__':
    main()
