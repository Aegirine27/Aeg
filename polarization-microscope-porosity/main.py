"""
偏光显微镜面孔率识别系统 - 主程序入口

使用方法：
    单张处理：
        python main.py --input path/to/image.jpg --output results/

    批量处理：
        python main.py --input path/to/folder/ --output results/ --batch

    使用分水岭算法：
        python main.py --input image.jpg --watershed

    交互式阈值调参：
        python main.py --input image.jpg --interactive
"""
import argparse
import sys
import cv2
import json
import time
from pathlib import Path
from tqdm import tqdm

from src.utils import load_config, load_image, save_image
from src.preprocessing import ImagePreprocessor
from src.threshold_segment import ThresholdSegmenter
from src.watershed_segment import WatershedSegmenter
from src.area_calculation import PorosityCalculator
from src.visualization import ResultVisualizer


class PorosityAnalyzer:
    """面孔率分析器 - 整合所有模块"""

    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.preprocessor = ImagePreprocessor(self.config)
        self.thresh_segmenter = ThresholdSegmenter(self.config)
        self.watershed_segmenter = WatershedSegmenter(self.config)
        self.calculator = PorosityCalculator(self.config)
        self.visualizer = ResultVisualizer(self.config)

    def analyze(self, image_path, use_watershed=False, save_intermediate=False):
        """
        分析单张图像

        Args:
            image_path: 图像路径
            use_watershed: 是否使用分水岭算法
            save_intermediate: 是否保存中间结果

        Returns:
            dict: 包含所有结果的字典
        """
        start_time = time.time()

        # 1. 加载图像
        original = load_image(image_path)
        filename = Path(image_path).stem

        # 2. 预处理
        processed = self.preprocessor.process(original)

        # 3. 阈值分割（始终执行，作为基础）
        thresh_result = self.thresh_segmenter.segment(processed)
        mask = thresh_result['mask']

        # 4. 分水岭算法（可选）
        method = 'threshold'
        if use_watershed:
            watershed_result = self.watershed_segmenter.segment(processed, mask)
            mask = watershed_result['mask']
            method = 'watershed'

        # 5. 面孔率计算
        stats = self.calculator.calculate(mask)
        stats['filename'] = filename
        stats['method'] = method
        stats['processing_time'] = round(time.time() - start_time, 3)

        # 6. 裂缝检测
        crack_info = self.calculator.detect_cracks(mask)
        stats['crack_count'] = crack_info['crack_count']

        # 7. 生成标注图像
        annotated = self.visualizer.create_annotated_image(mask)
        overlay = self.visualizer.overlay_on_original(original, mask)

        result = {
            'filename': filename,
            'original': original,
            'processed': processed['enhanced'],
            'mask': mask,
            'annotated': annotated,
            'overlay': overlay,
            'stats': stats,
            'method': method
        }

        return result

    def save_results(self, result, output_dir):
        """
        保存分析结果

        Args:
            result: analyze() 返回的结果字典
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = result['filename']

        # 1. 保存标注图像（纯蓝+纯白）
        annotated_path = output_dir / f"{filename}_annotated.png"
        self.visualizer.save_image(result['annotated'], annotated_path)

        # 2. 保存叠加图像
        overlay_path = output_dir / f"{filename}_overlay.png"
        self.visualizer.save_image(result['overlay'], overlay_path)

        # 3. 保存掩膜
        mask_path = output_dir / f"{filename}_mask.png"
        self.visualizer.save_image(result['mask'], mask_path)

        # 4. 保存对比图
        fig = self.visualizer.create_comparison_figure(
            result['original'], result['processed'],
            result['mask'], result['annotated'], result['stats']
        )
        comparison_path = output_dir / f"{filename}_comparison.png"
        self.visualizer.save_figure(fig, comparison_path)

        # 5. 保存统计信息（JSON）
        stats_path = output_dir / f"{filename}_stats.json"
        # 移除不可序列化的字段
        stats_clean = {k: v for k, v in result['stats'].items()
                       if k not in ['filename', 'processing_time']}
        stats_clean['filename'] = filename
        stats_clean['processing_time'] = result['stats']['processing_time']
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_clean, f, ensure_ascii=False, indent=2)

        # 6. 保存中间结果（可选）
        if self.config.get('output', {}).get('save_intermediate', False):
            intermediate_dir = output_dir / 'intermediate'
            intermediate_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(intermediate_dir / f"{filename}_denoised.png"), result['processed'])

        return {
            'annotated': str(annotated_path),
            'overlay': str(overlay_path),
            'mask': str(mask_path),
            'comparison': str(comparison_path),
            'stats': str(stats_path)
        }


def process_batch(analyzer, input_dir, output_dir, use_watershed=False):
    """批量处理文件夹"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    extensions = {'.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp'}
    image_files = [f for f in input_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in extensions]

    if not image_files:
        print(f"未找到图像文件: {input_dir}")
        return []

    print(f"找到 {len(image_files)} 张图像，开始处理...")

    all_stats = []
    for img_path in tqdm(image_files, desc="处理进度"):
        try:
            result = analyzer.analyze(str(img_path), use_watershed=use_watershed)
            paths = analyzer.save_results(result, output_dir)
            all_stats.append(result['stats'])
            print(f"✓ {img_path.name}: 面孔率={result['stats']['porosity_percent']:.4f}%")
        except Exception as e:
            print(f"✗ {img_path.name}: 错误 - {e}")
            continue

    # 保存汇总结果
    if all_stats:
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        print(f"\n汇总结果已保存: {summary_path}")

        # 生成对比图
        viz = ResultVisualizer({'output': {}})
        viz.plot_pore_size_distribution(all_stats, output_dir / 'porosity_comparison.png')

    return all_stats


def interactive_threshold_tuning(analyzer, image_path):
    """
    交互式阈值调参

    显示图像，用户点击蓝色树脂区域，自动计算阈值范围
    """
    original = load_image(image_path)
    processed = analyzer.preprocessor.process(original)
    hsv = processed['hsv']

    click_points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_points.append((x, y))
            # 取周围 10x10 区域
            x1, y1 = max(0, x-5), max(0, y-5)
            x2, y2 = min(hsv.shape[1], x+5), min(hsv.shape[0], y+5)
            roi = (x1, y1, x2-x1, y2-y1)

            tuned = analyzer.thresh_segmenter.auto_tune(hsv, roi)
            print(f"\n点击位置: ({x}, {y})")
            print(f"自动调整后的 HSV 阈值:")
            print(f"  Lower: {tuned['lower']}")
            print(f"  Upper: {tuned['upper']}")

            # 应用新阈值
            lower = np.array(tuned['lower'])
            upper = np.array(tuned['upper'])
            mask = cv2.inRange(hsv, lower, upper)

            # 显示结果
            display = original.copy()
            overlay = cv2.addWeighted(display, 0.7,
                                       cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
            cv2.imshow('Threshold Result', overlay)

    cv2.imshow('Original', original)
    cv2.setMouseCallback('Original', mouse_callback)

    print("在蓝色树脂区域点击鼠标左键，自动调整阈值...")
    print("按 ESC 退出")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='偏光显微镜面孔率识别系统')
    parser.add_argument('--input', '-i', required=True, help='输入图像路径或文件夹')
    parser.add_argument('--output', '-o', default='results', help='输出目录')
    parser.add_argument('--config', '-c', default='config.yaml', help='配置文件路径')
    parser.add_argument('--batch', '-b', action='store_true', help='批量处理模式')
    parser.add_argument('--watershed', '-w', action='store_true', help='使用分水岭算法')
    parser.add_argument('--interactive', action='store_true', help='交互式阈值调参')
    parser.add_argument('--save-intermediate', action='store_true', help='保存中间结果')

    args = parser.parse_args()

    # 初始化分析器
    analyzer = PorosityAnalyzer(args.config)

    if args.interactive:
        interactive_threshold_tuning(analyzer, args.input)
        return

    if args.batch or Path(args.input).is_dir():
        process_batch(analyzer, args.input, args.output, args.watershed)
    else:
        result = analyzer.analyze(args.input, use_watershed=args.watershed,
                                   save_intermediate=args.save_intermediate)
        paths = analyzer.save_results(result, args.output)

        print("\n" + "="*50)
        print(f"处理完成: {result['filename']}")
        print(f"方法: {result['method']}")
        print(f"面孔率: {result['stats']['porosity_percent']:.4f}%")
        print(f"孔隙数量: {result['stats']['pore_count']}")
        print(f"处理时间: {result['stats']['processing_time']}s")
        print("="*50)
        print(f"\n结果已保存到: {args.output}")
        for key, path in paths.items():
            print(f"  [{key}] {path}")


if __name__ == '__main__':
    main()
