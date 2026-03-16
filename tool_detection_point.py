#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM 3 Surgical Tool Detection Script
使用 SAM 3 检测和跟踪视频中的手术工具 (支持无头服务器环境的终端坐标输入)
"""

import os
import sys
import glob
import argparse
import warnings
import time

# 修复 libgomp: Invalid value for environment variable OMP_NUM_THREADS 警告
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 添加项目路径
sam3_path = os.path.join(os.path.dirname(__file__))
if sam3_path not in sys.path:
    sys.path.insert(0, sam3_path)

from sam3.model_builder import build_sam3_video_model
from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
    save_masklet_video,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM 3 手术工具检测脚本"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/root/autodl-tmp/sam3/videoframes/Knot_Tying_B001_capture1_stride3_fps10",
        help="视频文件夹路径 (包含 JPEG 帧) 或 MP4 文件",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="surgical tool",
        help="用于检测的文本提示 (当不启用点提示时使用)",
    )
    parser.add_argument(
        "--use_point_prompt",
        action="store_true",
        help="是否启用点提示 (开启后会在终端提示您输入像素坐标)",
    )
    parser.add_argument(
        "--prompt_frame_idx",
        type=int,
        default=5,
        help="施加初始提示(文本或点)的帧索引",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/sam3/test_outputs",
        help="输出目录",
    )
    parser.add_argument(
        "--vis_frame_stride",
        type=int,
        default=10,
        help="可视化帧间隔",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=True,
        help="是否保存输出视频",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="是否使用 GPU",
    )
    parser.add_argument(
        "--mask_output_dir",
        type=str,
        default="/root/autodl-tmp/sam3/maskoutput",
        help="用于训练的纯净 .npy 掩膜张量的保存目录",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=None,
        help="输出视频的帧率 (默认自动检测输入视频帧率，如果是文件夹则默认为 10)",
    )
    return parser.parse_args()


def print_step(step_num, total_steps, message):
    """打印进度信息"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{step_num}/{total_steps}] {message}")
    sys.stdout.flush()


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """将绝对坐标转换为相对坐标 (0-1 范围)，SAM3 点提示需要此格式"""
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")


def get_points_from_terminal(frame_idx, W, H):
    """
    无头服务器环境使用的终端交互函数，让用户手动输入绝对像素坐标。
    """
    points = []
    labels = []
    
    print("\n" + "="*60)
    print(f"【点提示交互模式 - 第 {frame_idx} 帧】")
    print(f"当前图像分辨率: 宽(W)={W}, 高(H)={H}")
    print("由于服务器无图形界面，请在本地看图软件中查看该帧，找到目标的坐标(X, Y)。")
    print("="*60)
    
    print("\n--- 第一步: 添加正向点 (您想要追踪的手术器械) ---")
    while True:
        user_input = input("请输入正向点的 X Y 坐标 (用空格分隔，如 '250 380')，或者输入 'n' 结束输入: ")
        user_input = user_input.strip()
        if user_input.lower() == 'n' or user_input == '':
            break
        try:
            x_str, y_str = user_input.split()
            x, y = float(x_str), float(y_str)
            if 0 <= x <= W and 0 <= y <= H:
                points.append([x, y])
                labels.append(1)  # 1 表示正向点
                print(f"  -> 已成功记录【正向】点: ({x}, {y})")
            else:
                print(f"  -> 警告: 坐标 ({x}, {y}) 超出图片尺寸范围！")
        except ValueError:
            print("  -> 格式错误！请确保输入两个数字并用空格隔开。")

    print("\n--- 第二步: 添加负向点 (不想包含的背景/线/阴影，如果没有直接输入 n) ---")
    while True:
        user_input = input("请输入负向点的 X Y 坐标 (用空格分隔)，或者输入 'n' 结束输入: ")
        user_input = user_input.strip()
        if user_input.lower() == 'n' or user_input == '':
            break
        try:
            x_str, y_str = user_input.split()
            x, y = float(x_str), float(y_str)
            if 0 <= x <= W and 0 <= y <= H:
                points.append([x, y])
                labels.append(0)  # 0 表示负向点
                print(f"  -> 已成功记录【负向】点: ({x}, {y})")
            else:
                print(f"  -> 警告: 坐标 ({x}, {y}) 超出图片尺寸范围！")
        except ValueError:
            print("  -> 格式错误！请确保输入两个数字并用空格隔开。")

    print("="*60 + "\n")
    return points, labels


def load_video_frames(video_path):
    """加载视频帧，返回 (video_frames, original_fps)"""
    print_step(1, 9, f"正在加载视频: {video_path}")

    start_time = time.time()
    original_fps = None
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        video_frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
            if frame_count % 100 == 0:
                print_step(1, 9, f"已加载 {frame_count} 帧")
        cap.release()
    else:
        # 加载 JPEG 帧
        video_frames = glob.glob(os.path.join(video_path, "*.jpg"))
        video_frames.extend(glob.glob(os.path.join(video_path, "*.png")))

        try:
            # 按帧序号排序
            video_frames.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            # 如果文件名格式不匹配，回退到字典序
            print_step(1, 9, "警告: 帧文件名格式不是 '<frame_index>.jpg'，使用字典序")
            video_frames.sort()

    duration = time.time() - start_time
    if original_fps and original_fps > 0:
        print_step(1, 9, f"成功加载 {len(video_frames)} 帧 (原始帧率: {original_fps:.2f} fps, 耗时: {duration:.2f} 秒)")
    else:
        print_step(1, 9, f"成功加载 {len(video_frames)} 帧 ({duration:.2f} 秒)")
    return video_frames, original_fps


def propagate_in_video(predictor, session_id):
    """在整个视频中传播分割结果"""
    print_step(5, 9, "正在视频中传播分割结果...")
    outputs_per_frame = {}
    frame_count = 0
    start_time = time.time()
    try:
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]
            frame_count += 1
            if frame_count % 50 == 0:
                avg_time = (time.time() - start_time) / frame_count
                print_step(5, 9, f"已处理 {frame_count} 帧, 平均每帧 {avg_time:.3f} 秒")
    except Exception as e:
        print_step(5, 9, f"传播过程中出错: {e}")
        raise

    print_step(5, 9, f"完成传播，共处理 {len(outputs_per_frame)} 帧")
    return outputs_per_frame


def save_sample_frames(video_frames, outputs_formatted, output_dir, vis_stride=30):
    """保存样本帧的可视化结果"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print_step(6, 9, f"正在保存样本帧可视化 (间隔 {vis_stride} 帧)...")
    saved_count = 0
    start_time = time.time()

    for frame_idx in range(0, len(video_frames), vis_stride):
        if frame_idx in outputs_formatted:
            try:
                plt.close("all")
                visualize_formatted_frame_output(
                    frame_idx,
                    video_frames,
                    outputs_list=[outputs_formatted],
                    titles=["SAM 3 手术工具检测"],
                    figsize=(12, 8),
                )

                output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
                plt.savefig(output_path, bbox_inches="tight", dpi=100)
                saved_count += 1
                if saved_count % 10 == 0:
                    print_step(6, 9, f"已保存 {saved_count} 个样本帧")
                plt.close()
            except Exception as e:
                print_step(6, 9, f"保存帧 {frame_idx} 失败: {e}")

    duration = time.time() - start_time
    print_step(6, 9, f"完成样本帧保存，共保存 {saved_count} 个帧 ({duration:.2f} 秒)")


def convert_outputs_for_video(outputs_per_frame):
    """将输出转换为视频保存格式"""
    video_outputs = {}
    print_step(7, 9, "正在转换输出格式...")
    valid_frames = 0
    total_frames = len(outputs_per_frame)

    for frame_idx, out in outputs_per_frame.items():
        if all(key in out for key in ["out_boxes_xywh", "out_probs", "out_obj_ids", "out_binary_masks"]):
            video_outputs[frame_idx] = {
                "out_boxes_xywh": out["out_boxes_xywh"],
                "out_probs": out["out_probs"],
                "out_obj_ids": out["out_obj_ids"],
                "out_binary_masks": out["out_binary_masks"],
            }
            valid_frames += 1
        else:
            print_step(7, 9, f"警告: 帧 {frame_idx} 输出格式不正确，缺少键: {list(out.keys())}")
            print_step(7, 9, f"提示: 确保使用原始输出而不是可视化输出")

    print_step(7, 9, f"格式转换完成，有效帧: {valid_frames}/{total_frames}")
    return video_outputs


def main():
    args = parse_args()
    start_total = time.time()

    try:
        # 设置设备
        print_step(0, 9, "设置设备")
        if args.use_gpu and torch.cuda.is_available():
            gpus_to_use = range(torch.cuda.device_count())
            print_step(0, 9, f"使用 GPU: {list(gpus_to_use)}, 显存: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        else:
            gpus_to_use = None
            print_step(0, 9, "使用 CPU (可能会很慢)")

        os.makedirs(args.output_dir, exist_ok=True)

        # 1. 加载视频帧
        video_frames, original_fps = load_video_frames(args.video_path)
        if len(video_frames) == 0:
            print_step(1, 9, "错误: 未找到视频帧！")
            return

        if args.output_fps is not None:
            output_fps = args.output_fps
            print_step(1, 9, f"使用用户指定的输出帧率: {output_fps} fps")
        elif original_fps and original_fps > 0:
            output_fps = original_fps
            print_step(1, 9, f"使用输入视频的原始帧率: {output_fps:.2f} fps")
        else:
            output_fps = 10
            print_step(1, 9, f"无法检测原始帧率，使用默认值: {output_fps} fps")

        # 2. 构建 SAM 3 视频预测器
        print_step(2, 9, "正在加载 SAM 3 模型...")
        start_time = time.time()

        try:
            if args.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                print_step(2, 9, f"当前显存使用: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print_step(2, 9, f"剩余显存: {torch.cuda.memory_reserved()/1e9:.2f} GB")

            local_model_dir = "/root/autodl-tmp/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7"
            local_checkpoint_path = os.path.join(local_model_dir, "sam3.pt")
            
            predictor = Sam3VideoPredictorMultiGPU(
                gpus_to_use=gpus_to_use,
                checkpoint_path=local_checkpoint_path,
            )

            if args.use_gpu and torch.cuda.is_available():
                print_step(2, 9, f"模型加载后显存: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        except Exception as e:
            print_step(2, 9, f"模型加载失败: {e}")
            if args.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        duration = time.time() - start_time
        print_step(2, 9, f"模型加载完成 ({duration:.2f} 秒)")

        # 3. 启动推理会话
        print_step(3, 9, "正在启动推理会话...")
        start_time = time.time()
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=args.video_path,
            )
        )
        session_id = response["session_id"]
        duration = time.time() - start_time
        print_step(3, 9, f"会话 ID: {session_id} ({duration:.2f} 秒)")

        # 4. 添加初始提示 (根据参数选择使用点提示还是文本提示)
        start_time = time.time()
        frame_idx = args.prompt_frame_idx
        
        # 确保 frame_idx 不越界
        if frame_idx >= len(video_frames):
            frame_idx = 0
            print_step(4, 9, f"警告: 指定的提示帧超出范围，已重置为第 {frame_idx} 帧")
            
        if args.use_point_prompt:
            # 开启服务器终端的点提示流程
            print_step(4, 9, f"准备在第 {frame_idx} 帧使用【点提示 (Point Prompt)】")
            
            # 读取当前帧尺寸
            if isinstance(video_frames[frame_idx], str):
                temp_img = cv2.imread(video_frames[frame_idx])
                H, W = temp_img.shape[:2]
            else:
                H, W = video_frames[frame_idx].shape[:2]
                
            # 通过终端输入获取绝对坐标
            points_abs, labels = get_points_from_terminal(frame_idx, W, H)
            
            if len(points_abs) == 0:
                print_step(4, 9, "未输入任何点，程序将尝试仅追踪背景并退出。")
                return True
                
            # 将绝对像素坐标转化为相对坐标 (0~1 之间)
            points_rel = abs_to_rel_coords(points_abs, W, H, coord_type="point")
            
            # 转化为模型需要的 tensor 格式
            points_tensor = torch.tensor(points_rel, dtype=torch.float32)
            points_labels_tensor = torch.tensor(labels, dtype=torch.int32)
            
            try:
                response = predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=frame_idx,
                        points=points_tensor,
                        point_labels=points_labels_tensor,
                    )
                )
                initial_output = response["outputs"]
            except Exception as e:
                print_step(4, 9, f"添加点提示失败: {e}")
                raise
                
        else:
            # 默认的文本提示流程
            print_step(4, 9, f"在第 {frame_idx} 帧添加文本提示: '{args.text_prompt}'")
            try:
                response = predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=frame_idx,
                        text=args.text_prompt,
                    )
                )
                initial_output = response["outputs"]
            except Exception as e:
                print_step(4, 9, f"添加文本提示失败: {e}")
                raise
                
        duration = time.time() - start_time
        print_step(4, 9, f"提示添加完成 ({duration:.2f} 秒)")

        # 显示初始检测结果
        num_objects = len(initial_output.get("out_obj_ids", []))
        print_step(4, 9, f"初始帧检测到 {num_objects} 个对象")
        for i, obj_id in enumerate(initial_output.get("out_obj_ids", [])):
            print_step(4, 9, f"  - 对象 ID: {obj_id}, 置信度: {initial_output['out_probs'][i]:.3f}")

        # 5. 在整个视频中传播
        start_time = time.time()
        outputs_per_frame = propagate_in_video(predictor, session_id)
        duration = time.time() - start_time
        avg_time = duration / len(outputs_per_frame) if outputs_per_frame else 0
        print_step(5, 9, f"传播完成，处理 {len(outputs_per_frame)} 帧，平均每帧 {avg_time:.3f} 秒")

        total_objects = 0
        for out_frame_idx, out in outputs_per_frame.items():
            if "out_obj_ids" in out:
                total_objects += len(out["out_obj_ids"])

        print_step(5, 9, f"总检测到 {total_objects} 个对象")

        import copy
        outputs_original = copy.deepcopy(outputs_per_frame)

        # 6. 保存可视化样本帧
        outputs_for_vis = prepare_masks_for_visualization(outputs_per_frame)
        save_sample_frames(
            video_frames,
            outputs_for_vis,
            args.output_dir,
            vis_stride=args.vis_frame_stride,
        )
        
        # 6.5 生成并保存 .npy
        print_step(6.5, 9, "正在生成并保存网络训练所需的 .npy 掩膜文件...")
        try:
            os.makedirs(args.mask_output_dir, exist_ok=True)
            all_frames_masks = []
            
            if isinstance(video_frames[0], str):
                temp_img = cv2.imread(video_frames[0])
                H, W = temp_img.shape[:2]
            else:
                H, W = video_frames[0].shape[:2]
                
            total_frames_count = len(video_frames)

            for mask_frame_idx in range(total_frames_count):
                if mask_frame_idx in outputs_original and "out_binary_masks" in outputs_original[mask_frame_idx]:
                    frame_masks_tensor = outputs_original[mask_frame_idx]["out_binary_masks"]
                    if isinstance(frame_masks_tensor, torch.Tensor):
                        if len(frame_masks_tensor) == 0:
                            merged_mask = np.zeros((H, W), dtype=np.float32)
                        else:
                            if frame_masks_tensor.ndim == 4:
                                frame_masks_tensor = frame_masks_tensor.squeeze(1)
                            merged_mask = frame_masks_tensor.max(dim=0)[0].cpu().numpy()
                    elif isinstance(frame_masks_tensor, np.ndarray):
                        if len(frame_masks_tensor) == 0:
                            merged_mask = np.zeros((H, W), dtype=np.float32)
                        else:
                            if frame_masks_tensor.ndim == 4:
                                frame_masks_tensor = np.squeeze(frame_masks_tensor, axis=1)
                            merged_mask = np.max(frame_masks_tensor, axis=0)
                    else:
                        merged_mask = np.zeros((H, W), dtype=np.float32)
                else:
                    merged_mask = np.zeros((H, W), dtype=np.float32)
                
                merged_mask = (merged_mask > 0).astype(np.float32)
                all_frames_masks.append(merged_mask)

            final_mask_array = np.stack(all_frames_masks, axis=0)
            
            video_name = os.path.basename(os.path.normpath(args.video_path))
            video_name = os.path.splitext(video_name)[0]
            npy_save_path = os.path.join(args.mask_output_dir, f"{video_name}_masks.npy")
            np.save(npy_save_path, final_mask_array)
            
            print_step(6.5, 9, f"成功保存掩膜张量至: {npy_save_path}")

        except Exception as e:
            print_step(6.5, 9, f"保存 .npy 掩膜文件失败: {e}")
        
        # 7. 保存输出视频
        if args.save_video:
            start_time = time.time()
            print_step(7, 9, "正在生成输出视频...")
            video_outputs = convert_outputs_for_video(outputs_original)

            if len(video_outputs) > 0:
                output_video_path = os.path.join(args.output_dir, "surgical_tool_detection.mp4")
                try:
                    save_masklet_video(
                        video_frames,
                        video_outputs,
                        output_video_path,
                        alpha=0.5,
                        fps=output_fps,
                    )
                    duration = time.time() - start_time
                    print_step(7, 9, f"视频已保存至: {output_video_path} (耗时: {duration:.2f} 秒)")
                except Exception as e:
                    print_step(7, 9, f"视频保存失败: {e}")

        # 8. 关闭会话
        print_step(8, 9, "正在关闭会话...")
        start_time = time.time()
        try:
            _ = predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
        except Exception as e:
            pass
            
        # 9. 关闭预测器
        print_step(9, 9, "正在关闭预测器...")
        predictor.shutdown()

        total_duration = time.time() - start_total
        print("\n" + "="*50)
        print(f"处理完成！总耗时: {total_duration:.2f} 秒")
        print("="*50)

    except Exception as e:
        print_step(0, 9, f"错误: {e}")
        import traceback
        print(traceback.format_exc())
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)