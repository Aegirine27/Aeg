"""
结果可视化模块

根据文档要求：
    - 孔隙与裂缝用纯蓝色填充标注
    - 剩余视域为纯白色

可选：
    - 在原图上叠加半透明标注
    - 生成分割边界图
    - 生成统计图表
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class ResultVisualizer:
    """结果可视化器"""

    def __init__(self, config):
        self.config = config.get('output', {})

    def create_annotated_image(self, mask, original_shape=None):
        """
        生成标注图像

        要求：
            - 孔隙与裂缝 → 纯蓝色 [255, 0, 0] (BGR)
            - 剩余视域 → 纯白色 [255, 255, 255] (BGR)

        Args:
            mask: 二值掩膜 (H, W)
            original_shape: 原始图像尺寸 (H, W, 3)，用于确保尺寸一致

        Returns:
            annotated: 标注图像 (H, W, 3), BGR
        """
        h, w = mask.shape

        # 创建纯白背景
        annotated = np.ones((h, w, 3), dtype=np.uint8) * 255

        # 获取颜色配置
        pore_color = self.config.get('pore_color', [255, 0, 0])  # BGR 纯蓝
        crack_color = self.config.get('crack_color', [255, 0, 0])  # BGR 纯蓝

        # 孔隙用蓝色填充
        # 目前不区分孔隙和裂缝，统一用蓝色
        annotated[mask > 0] = pore_color

        return annotated

    def overlay_on_original(self, original, mask, alpha=0.4):
        """
        在原图上叠加半透明标注（用于人工校验）

        Args:
            original: 原始图像 (H, W, 3)
            mask: 二值掩膜 (H, W)
            alpha: 叠加透明度

        Returns:
            overlay: 叠加后的图像
        """
        overlay = original.copy()
        pore_color = np.array(self.config.get('pore_color', [255, 0, 0]))

        # 创建彩色掩膜
        color_mask = np.zeros_like(original)
        color_mask[mask > 0] = pore_color

        # 混合
        overlay = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)

        # 绘制边界
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        return overlay

    def create_comparison_figure(self, original, processed, mask, annotated, stats):
        """
        创建对比图（6宫格或4宫格）

        展示：
            1. 原始图像
            2. 预处理后图像
            3. 阈值分割结果
            4. 分水岭结果（如有）
            5. 标注图像
            6. 叠加图
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 原图
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')

        # 2. 预处理
        axes[0, 1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Preprocessed')
        axes[0, 1].axis('off')

        # 3. 掩膜
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title('Segmentation Mask')
        axes[0, 2].axis('off')

        # 4. 标注图
        axes[1, 0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Annotated (Blue=Pore, White=Background)')
        axes[1, 0].axis('off')

        # 5. 叠加图
        overlay = self.overlay_on_original(original, mask)
        axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Overlay on Original')
        axes[1, 1].axis('off')

        # 6. 统计信息
        axes[1, 2].axis('off')
        text = self._format_stats_text(stats)
        axes[1, 2].text(0.1, 0.5, text, fontsize=12, verticalalignment='center',
                        fontfamily='monospace', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Statistics')

        plt.tight_layout()
        return fig

    def _format_stats_text(self, stats):
        """格式化统计信息为文本"""
        lines = [
            "=== 面孔率统计 ===",
            "",
            f"面孔率: {stats.get('porosity_percent', 0):.4f}%",
            f"孔隙数量: {stats.get('pore_count', 0)}",
            f"总孔隙像素: {stats.get('total_pore_pixels', 0)}",
            f"总像素数: {stats.get('total_pixels', 0)}",
            f"平均孔隙面积: {stats.get('avg_pore_area', 0):.2f} px",
            f"最小孔隙面积: {stats.get('min_pore_area', 0)} px",
            f"最大孔隙面积: {stats.get('max_pore_area', 0)} px",
            f"孔隙面积标准差: {stats.get('std_pore_area', 0):.2f} px",
            f"平均等效直径: {stats.get('avg_equivalent_diameter', 0):.2f} px",
        ]

        scale = stats.get('pixel_scale_um_per_px')
        if scale is not None:
            lines.extend([
                "",
                f"像素比例尺: {scale} μm/px",
                f"总孔隙面积: {stats.get('total_pore_area_um2', 0):.2f} μm²",
                f"总面积: {stats.get('total_area_um2', 0):.2f} μm²",
            ])

        return "\n".join(lines)

    def save_figure(self, fig, path, dpi=150):
        """保存 matplotlib 图表"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    def save_image(self, image, path):
        """保存 OpenCV 图像"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)

    def plot_pore_size_distribution(self, stats_list, output_path):
        """
        绘制多张图片的面孔率对比图

        Args:
            stats_list: 多个图像的统计结果列表
            output_path: 输出路径
        """
        names = [s.get('filename', f'Image_{i}') for i, s in enumerate(stats_list)]
        porosities = [s.get('porosity_percent', 0) for s in stats_list]
        counts = [s.get('pore_count', 0) for s in stats_list]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 面孔率柱状图
        bars = ax1.bar(range(len(names)), porosities, color='steelblue')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.set_ylabel('Porosity (%)')
        ax1.set_title('Porosity Comparison')
        ax1.axhline(y=np.mean(porosities), color='r', linestyle='--', label=f'Mean: {np.mean(porosities):.2f}%')
        ax1.legend()

        # 在柱子上标注数值
        for bar, val in zip(bars, porosities):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{val:.2f}%', ha='center', va='bottom', fontsize=8)

        # 孔隙数量柱状图
        ax2.bar(range(len(names)), counts, color='coral')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Pore Count')
        ax2.set_title('Pore Count Comparison')

        plt.tight_layout()
        self.save_figure(fig, output_path)
