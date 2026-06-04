"""
分水岭算法分割模块

适用场景：
    - 孔隙结构复杂，边缘模糊
    - 多个孔隙粘连在一起，难以用简单阈值分开

算法流程：
    1. 对初始掩膜进行距离变换
    2. 找到孔隙中心（标记）
    3. 分水岭算法分割粘连区域
    4. 只保留原始阈值掩膜内的分割结果
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
        enhanced = image_dict.get('enhanced', image_dict.get('original'))

        # 如果没有初始掩膜，先用颜色阈值生成
        if initial_mask is None:
            from .threshold_segment import ThresholdSegmenter
            thresh_segmenter = ThresholdSegmenter({'threshold_segmentation': {}})
            initial_mask = thresh_segmenter.segment(image_dict)['mask']

        # 1. 对初始掩膜进行距离变换
        distance = ndimage.distance_transform_edt(initial_mask)

        # 2. 找到局部最大值作为标记（使用形态学操作）
        cfg = self.config.get('distance_transform', {})
        kernel_size = cfg.get('kernel_size', 5)
        local_max = self._find_local_maxima(distance, kernel_size)

        # 3. 标记连通区域
        markers, num_features = ndimage.label(local_max)

        # 4. 准备分水岭输入
        # markers: 0=未知(由watershed决定), 1=背景, 2+=前景种子
        # 将背景区域标记为1，前景种子从2开始
        markers = markers.astype(np.int32)
        markers[markers > 0] += 1  # 前景种子从2开始
        markers[initial_mask == 0] = 1  # 明确标记背景为1

        # 5. 执行分水岭
        # 使用梯度幅值作为"地形"
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

        cv2.watershed(enhanced, markers)

        # 6. 生成分割结果
        # watershed结果: -1=边界, 0=背景, 1=原始背景, 2+=前景区域
        # 只保留前景区域（原始掩膜内且被watershed标记为前景的）
        # 注意：watershed可能把边界(-1)和背景(0/1)都排除
        mask = np.zeros_like(initial_mask)
        mask[(markers > 1) & (initial_mask > 0)] = 255

        # 清理边界
        if self.config.get('border_clearing', {}).get('enabled', True):
            mask = self._clear_border(mask)

        return {
            'mask': mask,
            'labels': markers,
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
        """
        from skimage.segmentation import clear_border
        cleared = clear_border(mask > 0)
        return (cleared * 255).astype(np.uint8)
