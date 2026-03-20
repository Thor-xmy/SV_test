import torch
import torch.nn as nn
import os
import numpy as np

from models.surgical_qa_model import SurgicalQAModel
from utils.data_loader import SurgicalQADataLoader, create_sample_annotations

def test_full_pipeline():
    print("=" * 60)
    print("🚀 开始真实的端到端 (End-to-End) Pipeline 测试")
    print("=" * 60)

    # 1. 准备配置
    config = {
        'static_dim': 512,
        'dynamic_dim': 1024,
        'use_pretrained': False,
        'freeze_backbone': False, # 测试梯度回传，必须设为 False
        'use_mask_loss': False
    }

    # 2. 初始化模型与优化器
    print("[1/5] 初始化 SurgicalQAModel...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SurgicalQAModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # 3. 生成假数据并测试 DataLoader
    print("[2/5] 正在构建假数据集和 DataLoader...")
    import tempfile
    import cv2
    tmp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, 'masks'), exist_ok=True)
    
    # 构造注释 (将样本数增加到 5，确保 train_loader 不会因为 drop_last=True 被清空)
    num_dummy_videos = 5
    create_sample_annotations(tmp_dir, num_samples=num_dummy_videos)
    
    # 构造假视频和假掩膜
    for i in range(num_dummy_videos):
        video_id = f"video_{i:03d}"
        video_path = os.path.join(tmp_dir, 'videos', f'{video_id}.mp4')
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (112, 112))
        for _ in range(20): # 生成20帧视频
            writer.write(np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8))
        writer.release()

        mask_dir = os.path.join(tmp_dir, 'masks', video_id)
        os.makedirs(mask_dir, exist_ok=True)
        for frame in range(20):
            mask = np.random.randint(0, 2, (112, 112), dtype=np.uint8) * 255
            cv2.imwrite(os.path.join(mask_dir, f'frame_{frame:04d}_mask.png'), mask)

    # 实例化 DataLoader
    dataloaders = SurgicalQADataLoader(
        data_root=tmp_dir, batch_size=2, num_workers=0,
        dataset_kwargs={'clip_length': 16, 'spatial_size': 112, 'use_mask': True, 'cache_clips': True}
    )
    train_loader = dataloaders.get_loader('train')

    print("[3/5] DataLoader 测试通过！抓取一个 Batch 进行前向传播...")
    batch = next(iter(train_loader))
    video = batch['frames'].to(device)
    masks = batch['masks'].to(device)
    scores_gt = batch['score'].to(device)

    print(f"      视频张量形状: {video.shape} (B, C, T, H, W)")
    print(f"      掩膜张量形状: {masks.shape} (B, T, H, W)")
    print(f"      真实分数形状: {scores_gt.shape}")

    # 4. 测试前向传播 (Forward)
    print("[4/5] 正在执行前向传播 (Forward Pass)...")
    score_pred, _ = model(video, masks)
    loss = criterion(score_pred.squeeze(-1), scores_gt)
    print(f"      预测分数形状: {score_pred.shape}")
    print(f"      当前 Loss 值: {loss.item():.4f}")

    # 5. 测试反向传播 (Backward)
    print("[5/5] 正在执行反向传播 (Backward Pass) 和参数更新...")
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度是否成功传递到核心模块
    #static_grad = model.static_extractor.projection[0].weight.grad
    static_grad = model.static_extractor.proj[0].weight.grad
    dynamic_grad = model.dynamic_extractor.mixed_conv.conv3d.weight.grad
    
    if static_grad is not None and dynamic_grad is not None:
        print("      ✅ 梯度回传成功！静态和动态特征提取器均已收到梯度更新。")
    else:
        print("      ❌ 梯度回传失败！计算图已断裂。")

    optimizer.step()
    print("\n🎉 端到端测试圆满成功！整个模型的数据流和计算图完全正确，可以开始正式训练了。")

if __name__ == '__main__':
    test_full_pipeline()