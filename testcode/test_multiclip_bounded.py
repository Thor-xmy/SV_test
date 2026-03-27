import os
import argparse
import yaml
import torch
import numpy as np
import csv
from tqdm import tqdm

from models.surgical_qa_model_multiclip_bounded import SurgicalQAModelMultiClipBounded
from utils.data_loader_video_level_frames import create_dataloader_with_split
from utils.metrics import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Test Multi-Clip Bounded Surgical QA Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file (e.g. configs/bounded.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--subset', type=str, default='test', choices=['train', 'val', 'test'], help='Which dataset split to evaluate on')
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs to use')
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 加载和合并配置 (YAML)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 3. 初始化模型
    print("\n" + "="*70)
    print("Building Model...")
    print("="*70)
    model = SurgicalQAModelMultiClipBounded(config)
    
    # 4. 加载训练权重
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device,weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded epoch {checkpoint.get('epoch', 'unknown')} successfully.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded state_dict successfully.")
        
    model = model.to(device)
    model.eval()

    # 5. 构建 DataLoader
    print("\n" + "="*70)
    print(f"Building DataLoader for subset: {args.subset.upper()}")
    print("="*70)
    
    score_min = config.get('score_min', 5.0)
    score_max = config.get('score_max', 25.0)
    target_min = config.get('target_min', 0.0)
    target_max = config.get('target_max', 1.0)
    
    dataloader = create_dataloader_with_split(
        data_root=config['data_root'],
        batch_size=config.get('batch_size', 2),
        num_workers=config.get('num_workers', 4),
        spatial_size=config.get('spatial_size', 112),
        subset=args.subset,
        train_ratio=config.get('split_ratio', [0.7, 0.15, 0.15])[0],
        val_ratio=config.get('split_ratio', [0.7, 0.15, 0.15])[1],
        test_ratio=config.get('split_ratio', [0.7, 0.15, 0.15])[2],
        split_seed=config.get('split_seed', 42),
        is_train=False,
        use_mask=config.get('use_mask', True),
        score_min=score_min,
        score_max=score_max,
        target_min=target_min,
        target_max=target_max
    )

    # 6. 开始推理
    print("\n" + "="*70)
    print("Starting Inference...")
    print("="*70)

    all_preds_denorm = []
    all_gts_denorm = []
    video_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating on {args.subset}"):
            videos = batch['video'].to(device)
            scores_gt_norm = batch['score'].to(device) 
            masks = batch['masks'].to(device) if 'masks' in batch else None
            video_ids = batch['video_id']

            scores_pred_norm = model(videos, masks=masks)

            # 反归一化
            scores_pred_denorm = model.denormalize_score(
                scores_pred_norm, target_min=score_min, target_max=score_max
            )
            scores_gt_denorm = model.denormalize_score(
                scores_gt_norm, target_min=score_min, target_max=score_max
            )

            preds = scores_pred_denorm.cpu().numpy().flatten()
            gts = scores_gt_denorm.cpu().numpy().flatten()
            
            all_preds_denorm.extend(preds)
            all_gts_denorm.extend(gts)

            for vid, p, g in zip(video_ids, preds, gts):
                video_results.append({
                    'video_id': vid,
                    'gt': float(g),
                    'pred': float(p),
                    'error': float(abs(p - g))
                })

    # 7. 计算评价指标
    print("\nComputing Metrics on Original Score Scale...")
    metrics = compute_metrics(
        y_pred=np.array(all_preds_denorm),
        y_gt=np.array(all_gts_denorm),
        verbose=True
    )

    # 8. 打印 Error 最大的几个视频
    print("\n" + "="*70)
    print("TOP 5 PREDICTION ERRORS (Bad Cases)")
    print("="*70)
    video_results.sort(key=lambda x: x['error'], reverse=True)
    for i, res in enumerate(video_results[:5]):
        print(f"{i+1}. {res['video_id']} | GT: {res['gt']:.2f} | Pred: {res['pred']:.2f} | Error: {res['error']:.2f}")

    # ==========================================
    # 9. 🌟 新增：保存测试结果到文件
    # ==========================================
    output_dir = os.path.dirname(args.checkpoint)
    if not output_dir:
        output_dir = '.'
        
    csv_path = os.path.join(output_dir, f'test_predictions_{args.subset}.csv')
    metrics_path = os.path.join(output_dir, f'test_metrics_{args.subset}.txt')

    print("\n" + "="*70)
    print(f"Saving Results to: {output_dir}")
    print("="*70)

    # 写入详细预测列表 (CSV)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'gt', 'pred', 'error'])
        writer.writeheader()
        writer.writerows(video_results)
    print(f"  [✓] Predictions saved to : {csv_path}")

    # 写入总体指标 (TXT)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("============================================================\n")
        f.write(f"EVALUATION METRICS ({args.subset.upper()} SET)\n")
        f.write("============================================================\n")
        f.write(f"  MAE: {metrics['mae']:.4f}\n")
        f.write(f"  NMAE: {metrics['nmae']:.2f}% (normalized)\n")
        f.write(f"  MSE: {metrics['mse']:.4f}\n")
        f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"  SRCC: {metrics['srcc']:.4f} (p={metrics['srcc_pvalue']:.4f})\n")
        f.write(f"  PCC:  {metrics['pcc']:.4f} (p={metrics['pcc_pvalue']:.4f})\n")
        f.write("============================================================\n")
        f.write(f"  Mean Pred: {metrics['mean_pred']:.4f}, Mean GT: {metrics['mean_gt']:.4f}\n")
        f.write(f"  Std Pred:  {metrics['std_pred']:.4f}, Std GT:  {metrics['std_gt']:.4f}\n")
    print(f"  [✓] Metrics saved to     : {metrics_path}")

    print("\nTest completed successfully!")

if __name__ == '__main__':
    main()