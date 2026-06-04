"""
分水岭算法分割模块

适用场景：
    - 孔隙结构复杂，边缘模糊
    - 蓝色树脂颜色不均匀（岩石杂基不完全填充导致）
    - 多个孔隙粘连在一起，难以用简单阈值分开

算法流程：
    1. 对图像进行梯度计算
    2. 距离变换找到孔隙中心（标记）
    3. 分水岭算法分割粘连区域
"""
import cv2
import numpy as np
from scipy import ndimage

from .segmentation_base import BaseSegmenter


class WatershedSegmenter(BaseSegmenter):
    """基于分水岭算法的孔隙分割器"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config.get('watershed', {})

    def segment(self, image_dict, initial_mask=None):
        """
        执行分水岭分割

        Args:
            image_dict: 预处理后的图像字典
            initial_mask: 阈值分割得到的初始掩膜（可选）

        Returns:
            dict: {
                'mask': 最终二值掩膜,
                'labels': 分水岭标签图,
                'method': 'watershed',
                'num_regions': 分割出的区域数量
            }
        """
        enhanced = image_dict['enhanced']

        # 如果没有初始掩膜，先用颜色阈值生成
        if initial_mask is None:
            from .threshold_segment import ThresholdSegmenter
            thresh_segmenter = ThresholdSegmenter({'threshold_segmentation': {}})
            initial_mask = thresh_segmenter.segment(image_dict)['mask']

        # 1. 对初始掩膜进行距离变换
        distance = ndimage.distance_transform_edt(initial_mask)

        # 2. 找到局部最大值作为标记
        cfg = self.config.get('distance_transform', {})
        kernel_size = cfg.get('kernel_size', 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 对距离图进行局部最大值检测
        local_max = self._find_local_maxima(distance, kernel_size)

        # 3. 标记连通区域
        markers, num_features = ndimage.label(local_max)

        # 4. 确保标记在前景内部
        markers = markers * (initial_mask // 255)

        # 5. 准备分水岭输入
        # 使用梯度幅值作为"地形"
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

        # 6. 执行分水岭
        markers = markers.astype(np.int32)
        cv2.watershed(enhanced, markers)

        # 7. 生成分割结果
        #  watershed 结果：-1=边界, 1-num=不同区域, 0=背景
        labels = markers.copy()

        # 前景掩膜（排除边界和背景）
        mask = np.zeros_like(initial_mask)
        mask[labels > 0] = 255

        # 清理边界
        if self.config.get('border_clearing', {}).get('enabled', True):
            mask = self._clear_border(mask)

        # 填充孔洞
        mask = self._fill_holes(mask)

        return {
            'mask': mask,
            'labels': labels,
            'method': 'watershed',
            'num_regions': num_features
        }

    def _find_local_maxima(self, distance, min_distance=10):
        """
        使用形态学操作找到局部最大值
        """
        # 对距离图进行膨胀，然后与原图比较
        max_dist = ndimage.maximum_filter(distance, size=min_distance)
        local_max = (distance == max_dist) & (distance > 0)
        return local_max.astype(np.uint8) * 255

    def _clear_border(self, mask):
        """
        清除与图像边界相连的区域
        这些通常是截断的不完整孔隙
        """
        from skimage.segmentation import clear_border
        cleared = clear_border(mask > 0)
        return (cleared * 255).astype(np.uint8)

    def _fill_holes(self, mask):
        """
        填充掩膜中的孔洞
        """
        # 使用轮廓填充法
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, -1)
        return filled
