"""
批量生成 SAM 初始标注

自动为文件夹中的所有图像生成初始 mask，使用 SAM + 颜色提示。
生成的 mask 保存到指定目录，供后续人工审核修正。

使用方法:
    python scripts/generate_sam_labels.py --input D:/AI/Test/ --output data/labels/
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from tqdm import tqdm

from src.sam_annotator import SAMAnnotator
from src.utils import load_image


def generate_labels(input_dir, output_dir, model_path="models/sam_vit_b_01ec64.pth",
                    extensions=None, device=None):
    """批量生成 SAM 初始标注

    Args:
        input_dir: 输入图像目录
        output_dir: 输出 mask 目录
        model_path: SAM 模型路径
        extensions: 支持的图像格式列表
        device: 'cuda' 或 'cpu'

    Returns:
        list: 生成的 mask 文件路径列表
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # 收集图像
    image_files = [f for f in input_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in extensions]

    if not image_files:
        print(f"未找到图像文件: {input_dir}")
        return []

    print(f"找到 {len(image_files)} 张图像")
    print(f"SAM 模型: {model_path}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)

    # 初始化 SAM 标注器
    try:
        annotator = SAMAnnotator(model_path=model_path, device=device)
    except Exception as e:
        print(f"SAM 初始化失败: {e}")
        print("\n请确保：")
        print("  1. 已安装 segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")
        print("  2. 已下载模型权重到 models/ 目录")
        return []

    generated = []
    failed = []

    for img_path in tqdm(image_files, desc="生成标注"):
        try:
            # 生成 mask
            mask = annotator.generate_mask(str(img_path))

            # 保存
            mask_path = output_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), mask)

            # 同时保存一个可视化叠加图供快速检查
            vis_path = output_dir / f"{img_path.stem}_vis.jpg"
            img = load_image(str(img_path))
            overlay = img.copy()
            overlay[mask > 0] = [0, 255, 0]  # 绿色标注
            vis = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
            cv2.imwrite(str(vis_path), vis)

            generated.append(mask_path)

        except Exception as e:
            print(f"\n✗ {img_path.name}: {e}")
            failed.append(img_path)

    # 输出统计
    print("\n" + "=" * 50)
    print(f"生成完成: {len(generated)}/{len(image_files)} 张")
    print(f"失败: {len(failed)} 张")
    if failed:
        print("失败文件:")
        for f in failed:
            print(f"  - {f.name}")
    print(f"\n输出目录: {output_dir}")
    print("=" * 50)

    return generated


def generate_weak_labels_from_threshold(input_dir, output_dir, config_path="config.yaml",
                                         extensions=None):
    """使用现有阈值方法生成弱标签（无需SAM）

    当 SAM 未安装时，使用传统阈值法生成初始 mask。
    质量低于 SAM，但无需额外依赖。

    Args:
        input_dir: 输入图像目录
        output_dir: 输出 mask 目录
        config_path: 配置文件路径
        extensions: 支持的图像格式列表
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    image_files = [f for f in input_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in extensions]

    if not image_files:
        print(f"未找到图像文件: {input_dir}")
        return []

    print(f"找到 {len(image_files)} 张图像")
    print("使用阈值法生成弱标签（无需SAM）")
    print("-" * 50)

    from src.utils import load_config
    from src.preprocessing import ImagePreprocessor
    from src.threshold_segment import ThresholdSegmenter

    config = load_config(config_path)
    preprocessor = ImagePreprocessor(config)
    segmenter = ThresholdSegmenter(config)

    generated = []

    for img_path in tqdm(image_files, desc="生成弱标签"):
        try:
            img = load_image(str(img_path))
            processed = preprocessor.process(img)
            result = segmenter.segment(processed)
            mask = result['mask']

            mask_path = output_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), mask)

            generated.append(mask_path)

        except Exception as e:
            print(f"\n✗ {img_path.name}: {e}")

    print(f"\n完成: {len(generated)}/{len(image_files)} 张")
    return generated


def main():
    parser = argparse.ArgumentParser(description='批量生成 SAM 初始标注')
    parser.add_argument('--input', '-i', required=True, help='输入图像目录')
    parser.add_argument('--output', '-o', required=True, help='输出 mask 目录')
    parser.add_argument('--model', '-m', default='models/sam_vit_b_01ec64.pth',
                        help='SAM 模型路径')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='配置文件路径（用于阈值弱标签回退）')
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'],
                        help='计算设备')
    parser.add_argument('--fallback', action='store_true',
                        help='SAM失败时使用阈值法回退')

    args = parser.parse_args()

    # 尝试使用 SAM
    try:
        generate_labels(
            input_dir=args.input,
            output_dir=args.output,
            model_path=args.model,
            device=args.device
        )
    except Exception as e:
        print(f"SAM 生成失败: {e}")

        if args.fallback:
            print("\n回退到阈值法生成弱标签...")
            generate_weak_labels_from_threshold(
                input_dir=args.input,
                output_dir=args.output,
                config_path=args.config
            )
        else:
            print("\n提示: 使用 --fallback 参数可回退到阈值法")


if __name__ == '__main__':
    main()
