"""
面孔率计算模块

面孔率 = (孔隙面积 + 裂缝面积) / 全视域面积 × 100%

输出：
    1. 面孔率数值
    2. 孔隙数量、平均孔隙面积、最大/最小孔隙面积
    3. 裂缝长度、宽度等参数（可选）
"""
import cv2
import numpy as np
from skimage.measure import label, regionprops


class PorosityCalculator:
    """面孔率计算器"""

    def __init__(self, config):
        self.config = config.get('area_calculation', {})

    def calculate(self, mask):
        """
        计算面孔率及相关统计信息

        Args:
            mask: 二值掩膜 (H, W), 255=孔隙/裂缝, 0=背景

        Returns:
            dict: 面孔率及详细统计信息
        """
        h, w = mask.shape
        total_pixels = h * w

        # 孔隙像素数
        pore_pixels = np.sum(mask > 0)

        # 面孔率
        porosity = (pore_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        # 连通区域分析
        labeled = label(mask > 0)
        regions = regionprops(labeled)

        # 过滤小区域（噪点）
        min_area = self.config.get('min_pore_area', 50)
        max_area = self.config.get('max_pore_area')

        valid_regions = []
        for region in regions:
            area = region.area
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            valid_regions.append(region)

        # 统计信息
        if valid_regions:
            areas = [r.area for r in valid_regions]
            perimeters = [r.perimeter for r in valid_regions]
            equivalent_diameters = [r.equivalent_diameter for r in valid_regions]

            stats = {
                'porosity_percent': round(porosity, 4),
                'pore_count': len(valid_regions),
                'total_pore_pixels': int(pore_pixels),
                'total_pixels': int(total_pixels),
                'avg_pore_area': round(float(np.mean(areas)), 2),
                'min_pore_area': int(np.min(areas)),
                'max_pore_area': int(np.max(areas)),
                'std_pore_area': round(float(np.std(areas)), 2),
                'avg_perimeter': round(float(np.mean(perimeters)), 2),
                'avg_equivalent_diameter': round(float(np.mean(equivalent_diameters)), 2),
                'pixel_scale_um_per_px': self.config.get('pixel_scale'),
            }

            # 如果有像素比例尺，计算实际面积
            scale = self.config.get('pixel_scale')
            if scale is not None:
                stats['total_pore_area_um2'] = round(pore_pixels * (scale ** 2), 2)
                stats['total_area_um2'] = round(total_pixels * (scale ** 2), 2)
                stats['avg_pore_area_um2'] = round(float(np.mean(areas)) * (scale ** 2), 2)
        else:
            stats = {
                'porosity_percent': 0.0,
                'pore_count': 0,
                'total_pore_pixels': 0,
                'total_pixels': int(total_pixels),
                'avg_pore_area': 0.0,
                'min_pore_area': 0,
                'max_pore_area': 0,
                'std_pore_area': 0.0,
                'avg_perimeter': 0.0,
                'avg_equivalent_diameter': 0.0,
                'pixel_scale_um_per_px': self.config.get('pixel_scale'),
            }

        return stats

    def detect_cracks(self, mask):
        """
        检测裂缝（细长形孔隙）

        策略：
            1. 筛选长宽比很大的连通区域
            2. 或使用形态学骨架化提取裂缝中心线

        Args:
            mask: 孔隙掩膜

        Returns:
            dict: 裂缝统计信息
        """
        labeled = label(mask > 0)
        regions = regionprops(labeled)

        cracks = []
        for region in regions:
            # 长宽比（major_axis / minor_axis）
            if region.minor_axis_length > 0:
                aspect_ratio = region.major_axis_length / region.minor_axis_length
            else:
                aspect_ratio = 0

            # 细长形且面积相对较小
            if aspect_ratio > 5 and region.area < 5000:
                cracks.append({
                    'area': region.area,
                    'length': region.major_axis_length,
                    'width': region.minor_axis_length,
                    'aspect_ratio': aspect_ratio,
                    'centroid': region.centroid
                })

        return {
            'crack_count': len(cracks),
            'cracks': cracks,
            'total_crack_pixels': sum(c['area'] for c in cracks)
        }

    def save_results(self, stats, output_path):
        """
        保存结果到文件
        """
        import json
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
