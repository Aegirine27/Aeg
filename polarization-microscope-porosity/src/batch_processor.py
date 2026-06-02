"""
批量处理模块

支持：
    - 批量处理文件夹中的所有图像
    - 多线程/多进程加速（可选）
    - 结果汇总与导出（CSV/Excel/JSON）
    - 处理日志记录
"""
import csv
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .utils import load_image
from .preprocessing import ImagePreprocessor
from .threshold_segment import ThresholdSegmenter
from .watershed_segment import WatershedSegmenter
from .area_calculation import PorosityCalculator
from .visualization import ResultVisualizer


class BatchProcessor:
    """批量处理器"""

    def __init__(self, config, analyzer):
        """
        Args:
            config: 配置字典
            analyzer: PorosityAnalyzer 实例
        """
        self.config = config
        self.analyzer = analyzer
        self.supported_formats = {'.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp'}

    def collect_images(self, input_dir):
        """
        收集文件夹中的所有图像文件

        Args:
            input_dir: 输入目录路径

        Returns:
            list: 图像文件路径列表
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"目录不存在: {input_dir}")

        images = []
        for ext in self.supported_formats:
            images.extend(input_dir.glob(f'*{ext}'))
            images.extend(input_dir.glob(f'*{ext.upper()}'))

        # 去重并排序
        images = sorted(list(set(images)))
        return images

    def process_single(self, image_path, output_dir, use_watershed=False):
        """
        处理单张图像（供批量调用）

        Returns:
            dict or None: 成功返回 stats，失败返回 None
        """
        try:
            result = self.analyzer.analyze(
                str(image_path),
                use_watershed=use_watershed
            )
            self.analyzer.save_results(result, output_dir)
            return result['stats']
        except Exception as e:
            print(f"✗ {image_path.name}: {e}")
            return None

    def process_folder(self, input_dir, output_dir, use_watershed=False,
                        max_workers=1, progress_callback=None):
        """
        批量处理文件夹

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            use_watershed: 是否使用分水岭
            max_workers: 并行工作线程数（1=串行）
            progress_callback: 进度回调函数(当前, 总数)

        Returns:
            list: 所有成功处理的统计结果
        """
        images = self.collect_images(input_dir)
        total = len(images)

        if total == 0:
            print(f"未找到图像文件: {input_dir}")
            return []

        print(f"找到 {total} 张图像")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_stats = []
        failed = []

        start_time = time.time()

        if max_workers > 1:
            # 多线程并行处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_single, img, output_dir, use_watershed
                    ): img for img in images
                }

                for future in tqdm(as_completed(futures), total=total, desc="处理进度"):
                    stats = future.result()
                    if stats:
                        all_stats.append(stats)
                    else:
                        failed.append(futures[future].name)
        else:
            # 串行处理（推荐，OpenCV 多线程不稳定）
            for i, img_path in enumerate(tqdm(images, desc="处理进度"), 1):
                stats = self.process_single(img_path, output_dir, use_watershed)
                if stats:
                    all_stats.append(stats)
                else:
                    failed.append(img_path.name)

                if progress_callback:
                    progress_callback(i, total)

        elapsed = time.time() - start_time

        # 保存汇总
        self._save_summary(all_stats, output_dir, elapsed, failed)

        return all_stats

    def _save_summary(self, stats_list, output_dir, elapsed_time, failed_files):
        """
        保存汇总结果
        """
        output_dir = Path(output_dir)

        # 1. JSON 汇总
        summary = {
            'total_images': len(stats_list) + len(failed_files),
            'success_count': len(stats_list),
            'failed_count': len(failed_files),
            'failed_files': failed_files,
            'total_time_seconds': round(elapsed_time, 2),
            'avg_time_per_image': round(elapsed_time / max(len(stats_list), 1), 2),
            'avg_porosity': round(
                sum(s['porosity_percent'] for s in stats_list) / max(len(stats_list), 1), 4
            ) if stats_list else 0,
            'results': stats_list
        }

        with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 2. CSV 汇总
        if stats_list:
            csv_path = output_dir / 'summary.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                if stats_list:
                    writer = csv.DictWriter(f, fieldnames=stats_list[0].keys())
                    writer.writeheader()
                    writer.writerows(stats_list)

        # 3. 生成可视化对比图
        if len(stats_list) > 1:
            viz = ResultVisualizer({'output': {}})
            viz.plot_pore_size_distribution(
                stats_list,
                output_dir / 'porosity_comparison.png'
            )

        print(f"\n{'='*50}")
        print(f"批量处理完成")
        print(f"  成功: {len(stats_list)} 张")
        print(f"  失败: {len(failed_files)} 张")
        print(f"  总耗时: {elapsed_time:.1f}s")
        print(f"  平均面孔率: {summary['avg_porosity']:.4f}%")
        if failed_files:
            print(f"  失败文件: {', '.join(failed_files[:5])}")
        print(f"{'='*50}")
