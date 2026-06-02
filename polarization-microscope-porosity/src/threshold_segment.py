"""
颜色阈值分割模块

用于识别蓝色树脂填充的孔隙。
蓝色树脂在偏光显微镜单偏光下通常呈现明显的蓝色，
可以通过颜色阈值在 HSV 或 Lab 颜色空间中分割出来。

适用场景：
    - 蓝色树脂边缘清晰
    - 孔隙与矿物颜色对比明显
    - 计算简便，速度快
"""
import cv2
import numpy as np


class ThresholdSegmenter:
    """基于颜色阈值的孔隙分割器"""

    def __init__(self, config):
        self.config = config.get('threshold_segmentation', {})

    def segment(self, image_dict):
        """
        执行阈值分割

        Args:
            image_dict: 预处理后的图像字典，包含 'hsv', 'lab' 等

        Returns:
            dict: {
                'mask': 二值掩膜 (H, W), 255=孔隙, 0=背景
                'method': 'threshold',
                'params': 使用的参数
            }
        """
        color_space = self.config.get('color_space', 'HSV')

        if color_space.upper() == 'HSV':
            mask = self._segment_hsv(image_dict['hsv'])
        elif color_space.upper() == 'LAB':
            mask = self._segment_lab(image_dict['lab'])
        else:
            raise ValueError(f"不支持的颜色空间: {color_space}")

        # 形态学操作优化掩膜
        mask = self._morphological_cleanup(mask)

        return {
            'mask': mask,
            'method': 'threshold',
            'params': {
                'color_space': color_space,
                'hsv_range': self.config.get('hsv_range'),
                'lab_range': self.config.get('lab_range')
            }
        }

    def _segment_hsv(self, hsv_image):
        """
        在 HSV 颜色空间中进行阈值分割

        蓝色在 HSV 中的范围：
            H: 100-140 (青色到蓝色)
            S: 50-255 (有一定饱和度)
            V: 50-255 (有一定亮度)
        """
        hsv_range = self.config.get('hsv_range', {})
        lower = np.array(hsv_range.get('lower', [100, 50, 50]))
        upper = np.array(hsv_range.get('upper', [140, 255, 255]))

        mask = cv2.inRange(hsv_image, lower, upper)
        return mask

    def _segment_lab(self, lab_image):
        """
        在 Lab 颜色空间中进行阈值分割

        Lab 颜色空间的优势：
            - L: 亮度，与颜色无关
            - a: 绿-红轴
            - b: 蓝-黄轴（蓝色为负值）

        蓝色在 Lab 中：b 通道为负，且绝对值较大
        """
        lab_range = self.config.get('lab_range', {})
        lower = np.array(lab_range.get('lower', [0, 0, -128]))
        upper = np.array(lab_range.get('upper', [100, 127, -20]))

        mask = cv2.inRange(lab_image, lower, upper)
        return mask

    def _morphological_cleanup(self, mask):
        """
        形态学操作清理掩膜

        1. 开运算：去除小噪点
        2. 闭运算：填充小孔洞，连接断裂区域
        """
        morph_cfg = self.config.get('morphological_operations', {})
        iterations = morph_cfg.get('iterations', 2)

        # 开运算核（去除小噪点）
        open_k = morph_cfg.get('open_kernel', [3, 3])
        open_kernel = np.ones(open_k, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=iterations)

        # 闭运算核（填充孔洞）
        close_k = morph_cfg.get('close_kernel', [5, 5])
        close_kernel = np.ones(close_k, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=iterations)

        return mask

    def auto_tune(self, hsv_image, sample_region):
        """
        根据用户点击的区域自动调整阈值

        Args:
            hsv_image: HSV 图像
            sample_region: 用户选择的样本区域 (x, y, w, h)

        Returns:
            dict: 自动计算出的 HSV 上下限
        """
        x, y, w, h = sample_region
        roi = hsv_image[y:y+h, x:x+w]

        # 计算 ROI 的均值和标准差
        mean = np.mean(roi, axis=(0, 1))
        std = np.std(roi, axis=(0, 1))

        # 阈值 = 均值 ± 2*标准差
        lower = np.clip(mean - 2 * std, 0, 255).astype(np.uint8)
        upper = np.clip(mean + 2 * std, 0, 255).astype(np.uint8)

        return {
            'lower': lower.tolist(),
            'upper': upper.tolist()
        }
