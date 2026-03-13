# 手术视频质量评估模型

基于 paper1231 框架的 PyTorch 实现，用于手术视频技能评估。

## 📋 目录

- [快速开始](#快速开始)
- [项目概述](#项目概述)
- [架构设计](#架构设计)
- [安装指南](#安装指南)
- [数据集准备](#数据集准备)
- [训练模型](#训练模型)
- [评估模型](#评估模型)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [故障排查](#故障排查)

---

## 🚀 快速开始

```bash
# 1. 进入项目目录
cd /root/autodl-tmp/surgical_qa_model

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备数据集（参见"数据集准备"章节）
#    - 将视频文件放入 datasets/videos/
#    - 将掩膜文件放入 datasets/masks/
#    - 创建 datasets/annotations.json

# 4. 更新配置文件：编辑 configs/default.yaml
#    将 data_root 设置为你的数据集路径，例如：data_root: /root/autodl-tmp/datasets

# 5. 开始训练
python train.py --config configs/default.yaml --gpus 0

# 6. 使用 TensorBoard 监控训练
tensorboard --logdir=logs
```

---

## 📖 项目概述

本项目实现了一个手术视频质量评估模型，能够从视频片段中评估手术技能水平。模型架构基于 paper1231，但适配为单视频直接评分（无需对比学习）。

### 与 paper1231 的主要区别

| 方面 | paper1231 | 本模型 |
|---------|-----------|-------------|
| 框架 | 使用查询/参考视频对的对比学习 | 单视频直接回归 |
| 交叉注意力 | 是（查询和参考之间） | 否 |
| 分数预测 | `y_q = y_r + CR(E(V_q), E(V_r))` | `y = R(Concat(F_static, F_dynamic))` |
| 训练目标 | 学习相对分数差异 | 学习绝对分数 |

---

## 🏗️ 架构设计

```
输入视频 (B, C, T, H, W)
    ├─→ ResNet-34 (静态特征提取器) ──→ 静态特征 A
    │   ├─ ResNet 主干网络
    │   └─ 多尺度下采样 (s=2,4,8)
    │
    ├─→ I3D (动态特征提取器) ──→ 动态特征 C
    │   └─ 混合 3D 卷积
    │
    └─→ SAM3 掩膜 (离线) ──→ 器械掩膜 B
        └─→ 从预计算文件读取

掩膜引导注意力模块
    ├─ 输入：B (掩膜) 和 C (动态特征)
    ├─ 从掩膜特征生成注意力
    ├─ 与学习到的注意力融合
    └─ 应用于特征 → 掩膜动态特征 D

特征融合与回归
    ├─ Concat(A, D) ──→ 融合特征
    └─ 全连接层 ──→ 分数 y
```

### 模型组件详解

**1. 静态特征提取器 (ResNet-34)**
- ResNet-34 主干网络，包含 4 层
- 多尺度下采样（步长：2, 4, 8）
- 帧采样策略：中间帧、平均、首帧、尾帧
- 捕获组织状态和手术视野清晰度

**2. 动态特征提取器 (I3D)**
- Inception-3D 主干网络
- 混合 3D 卷积扩展时间感受野
- 短期片段处理，使用重叠窗口
- 提取器械操作的时空特征

**3. 掩膜引导注意力**
- 从掩膜特征生成均值和最大值注意力
- 可学习的聚合函数
- 时间平滑以减少掩膜抖动
- 逐元素相乘：`F_dy = F_clip * (A + I)`

**4. 融合回归器**
- 输入：`[static_dim, dynamic_dim] = [512, 832]`
- 隐藏层：`[1024, 512, 256, 128]`
- 输出：`[1]`（质量分数）

---

## 💻 安装指南

### 前置要求

- Python 3.7+
- CUDA 10.2+（支持 GPU）
- PyTorch 1.9+

### 步骤 1：安装依赖

```bash
# 安装所有必需的包
pip install -r requirements.txt

# 或手动安装
pip install torch torchvision
pip install numpy opencv-python
pip install pyyaml tqdm
pip install scipy tensorboard
```

### 步骤 2：（可选）安装 SAM3

如果需要使用 SAM3 在线生成掩膜：

```bash
# 按照官方 SAM3 安装说明
# 位置：/root/autodl-tmp/sam3
```

---

## 📁 数据集准备

### 必需的目录结构

```
data_root/              # 在 configs/default.yaml 中设置的路径
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
├── masks/
│   ├── video_001/
│   │   ├── frame_0000_mask.png
│   │   ├── frame_0001_mask.png
│   │   └── ...
│   ├── video_002/
│   │   └── ...
│   └── ...
└── annotations.json
```

### 1. 视频文件

将手术视频放入 `data_root/videos/`：
```bash
data_root/videos/
├── video_001.mp4
├── video_002.mp4
└── ...
```

### 2. 掩膜文件

#### 选项 A：使用预计算掩膜（推荐）

使用 SAM3 生成手术器械掩膜并保存为 PNG 文件：

```python
# 示例：使用 SAM3 生成掩膜
from sam3 import build_sam3_video_model
import cv2
import os

model = build_sam3_video_model(device='cuda')
inference_state = model.init_state('path/to/video.mp4')

# 添加点提示以检测器械
model.add_points_prompts(
    inference_state,
    frame_idx=0,
    point_coords=[[x, y]],  # 器械上的正点
    point_labels=[1]
)

# 在视频中传播掩膜
masks = model.propagate_in_video(inference_state)

# 保存掩膜
output_dir = f'masks/video_001'
os.makedirs(output_dir, exist_ok=True)
for frame_idx, mask in enumerate(masks):
    cv2.imwrite(f'{output_dir}/frame_{frame_idx:04d}_mask.png', mask)
```

#### 选项 B：使用 NumPy 掩膜

或者，将掩膜保存为单个 NumPy 数组：
```python
import numpy as np
# masks 形状：(num_frames, H, W)
np.save('data_root/masks/video_001_masks.npy', masks)
```

### 3. 标注文件

创建 `data_root/annotations.json` 文件，包含真实分数：

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

**注意**：每个视频至少需要 `score` 字段。其他字段是可选的。

### 4. 创建示例数据集（用于测试）

代码包含创建虚拟数据集的辅助函数：

```python
from utils.data_loader import create_sample_annotations
create_sample_annotations('/path/to/dataset', num_samples=100)
```

---

## 🏋️ 训练模型

### 基础训练

```bash
# 使用默认配置
python train.py --config configs/default.yaml --gpus 0

# 或覆盖特定参数
python train.py \
    --config configs/default.yaml \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 0.0001
```

### 从断点恢复训练

```bash
# 从检查点恢复
python train.py \
    --config configs/default.yaml \
    --resume checkpoints/checkpoint_epoch_50.pth \
    --gpus 0
```

### 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir=logs

# 在浏览器中打开：http://localhost:6006
```

### 训练命令行参数

| 参数 | 描述 | 默认值 |
|----------|-------------|----------|
| `--config` | 配置文件路径 | `configs/default.yaml` |
| `--data_root` | 覆盖数据集路径 | 来自配置文件 |
| `--output_dir` | 日志输出目录 | `output` |
| `--batch_size` | 批次大小 | 8 |
| `--epochs` | 训练轮数 | 100 |
| `--learning_rate` | 学习率 | 0.0001 |
| `--optimizer` | 优化器 (sgd, adam, adamw) | adam |
| `--freeze_backbone` | 冻结主干网络权重 | True |
| `--use_amp` | 使用混合精度训练 | False |
| `--gpus` | GPU ID（逗号分隔） | 0 |
| `--resume` | 检查点路径 | None |
| `--evaluate` | 仅评估模式 | False |

---

## 📊 评估模型

```bash
# 在测试集上评估
python train.py \
    --config configs/default.yaml \
    --evaluate \
    --gpus 0
```

### 评估指标

遵循 paper1231：

1. **平均绝对误差 (MAE)**
   ```
   MAE = (1/n) * sum(|y_pred - y_gt|)
   ```

2. **斯皮尔曼等级相关系数 (SRCC)**
   ```
   SRCC = 预测值与真实值之间的等级相关系数
   ```

3. **皮尔逊相关系数 (PCC)**
   ```
   PCC = 预测值与真实值之间的线性相关系数
   ```

4. **归一化 MAE (NMAE)**
   ```
   NMAE = MAE / (score_range) * 100
   ```

---

## 📂 项目结构

```
surgical_qa_model/
├── models/                          # 模型架构
│   ├── __init__.py
│   ├── static_feature_extractor.py   # 基于 ResNet-34 的静态特征
│   ├── dynamic_feature_extractor.py  # 基于 I3D 的动态特征
│   ├── mask_guided_attention.py     # 掩膜引导注意力模块
│   └── surgical_qa_model.py        # 主模型类
│
├── utils/                           # 工具函数
│   ├── __init__.py
│   ├── data_loader.py              # 数据集和数据加载器
│   ├── mask_loader.py              # 掩膜加载工具
│   ├── metrics.py                  # 评估指标
│   └── training.py                 # 训练循环和工具
│
├── configs/                         # 配置文件
│   └── default.yaml                # 默认配置
│
├── datasets/                        # 数据集存储（需创建）
│   ├── videos/                    # 手术视频文件
│   ├── masks/                     # 预计算掩膜文件
│   └── annotations.json           # 真实分数
│
├── checkpoints/                     # 保存的模型检查点
├── logs/                           # 训练日志和 TensorBoard
├── requirements.txt                 # Python 依赖
├── README.md                      # 英文文档
├── README_ch.md                   # 本文档（中文）
├── ARCHITECTURE.md               # 详细架构文档
└── train.py                        # 主训练脚本
```

---

## ⚙️ 配置说明

`configs/default.yaml` 文件包含所有超参数：

### 数据设置
```yaml
data_root: /path/to/dataset    # 必须更新！
batch_size: 8
num_workers: 4
```

### 视频预处理
```yaml
clip_length: 16    # 每个片段的帧数
clip_stride: 10    # 片段提取步长
spatial_size: 224    # 调整大小为 (H, W)
```

### 模型设置
```yaml
static_dim: 512    # ResNet 特征维度
dynamic_dim: 832    # I3D 特征维度
freeze_backbone: true    # 冻结主干网络权重
use_pretrained: true    # 使用 ImageNet 预训练权重
use_mask_loss: true    # 使用掩膜监督损失
```

### 训练设置
```yaml
epochs: 100
learning_rate: 0.0001
weight_decay: 0.0001
optimizer: adam
use_amp: false    # 混合精度训练
clip_grad_norm: 1.0    # 梯度裁剪
```

### 学习率调度
```yaml
lr_scheduler: cosine    # 选项：cosine, step, plateau
lr_milestones: [30, 60, 90]    # 用于 step 调度器
lr_gamma: 0.1    # 学习率衰减因子
```

---

## 🔧 故障排查

### 常见问题

#### 1. 内存溢出 (OOM)

**症状**：`CUDA out of memory` 错误

**解决方案**：
```bash
# 减小批次大小
python train.py --config configs/default.yaml --batch_size 4

# 启用混合精度
python train.py --config configs/default.yaml --use_amp

# 冻结主干网络
# 编辑 configs/default.yaml：freeze_backbone: true
```

#### 2. 数据加载问题

**症状**：`FileNotFoundError` 或掩膜加载错误

**解决方案**：
- 验证 `data_root` 在 `configs/default.yaml` 中设置正确
- 检查 `videos/` 中的所有视频在 `annotations.json` 中都有对应条目
- 确保每个视频的掩膜文件存在于 `masks/video_XXX/` 中
- 验证掩膜尺寸与视频帧尺寸匹配

#### 3. SAM3 掩膜生成问题

**症状**：掩膜生成失败或产生不正确的掩膜

**解决方案**：
- 确保 SAM3 从正确路径安装：`/root/autodl-tmp/sam3`
- 检查 CUDA 可用性：`torch.cuda.is_available()`
- 验证点提示正确定位在器械区域上

#### 4. 模型收敛不佳

**症状**：训练损失不下降或剧烈波动

**解决方案**：
```bash
# 降低学习率
python train.py --config configs/default.yaml --learning_rate 1e-5

# 启用梯度裁剪
python train.py --config configs/default.yaml --clip_grad_norm 1.0

# 使用不同的优化器
python train.py --config configs/default.yaml --optimizer adamw
```

#### 5. 验证性能问题

**症状**：验证损失远高于训练损失

**解决方案**：
- 检查训练/验证/测试的数据集划分
- 确保 `is_train` 标志正确设置（已在最新版本修复）
- 验证划分之间没有数据泄漏

---

## 📚 引用

如果您在研究中使用此代码，请引用：

```bibtex
@article{your_name_202X_surgical,
  title={Surgical Video Quality Assessment via Static and Dynamic Feature Fusion},
  author={Your Name},
  journal={ArXiv preprint},
  year={202X}
}
```

同时引用原始 paper1231：

```bibtex
@article{paper1231_202X_fine_grained,
  title={Fine-Grained Contrastive Learning for Robotic Surgery Skill Assessment Through Joint Instrument-Tissue Modeling},
  author={Paper1231 Authors},
  journal={Original Paper1231},
  year={202X}
}
```

---

## 📄 许可证

本项目仅供研究和教育使用。

## 📧 支持

如有问题或疑虑，请查阅[故障排查](#故障排查)章节或在项目仓库上提交 issue。
