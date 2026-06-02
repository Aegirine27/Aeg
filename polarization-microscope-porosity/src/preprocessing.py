"""
图像预处理模块

处理流程：
    1. 双边滤波去噪（保留边缘）
    2. 光照不均校正（背景减除）
    3. CLAHE 对比度增强
    4. 颜色空间转换（BGR -> HSV/Lab）
"""
import cv2
import numpy as np


class ImagePreprocessor:
    """图像预处理器"""

    def __init__(self, config):
        self.config = config.get('preprocessing', {})

    def process(self, image):
        """
        执行完整的预处理流程

        Args:
            image: 输入图像 (H, W, 3), BGR格式

        Returns:
            dict: {
                'original': 原图,
                'denoised': 去噪后图像,
                'illumination_corrected': 光照校正后,
                'enhanced': 最终增强图像,
                'hsv': HSV颜色空间,
                'lab': Lab颜色空间
            }
        """
        result = {'original': image.copy()}

        # 1. 去噪
        denoised = self.denoise(image)
        result['denoised'] = denoised

        # 2. 光照校正
        corrected = self.correct_illumination(denoised)
        result['illumination_corrected'] = corrected

        # 3. 对比度增强
        enhanced = self.enhance_contrast(corrected)
        result['enhanced'] = enhanced

        # 4. 颜色空间转换
        result['hsv'] = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        result['lab'] = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)

        return result

    def denoise(self, image):
        """去噪处理"""
        cfg = self.config.get('denoise', {})
        method = cfg.get('method', 'bilateral')

        if method == 'bilateral':
            # 双边滤波：保边去噪，适合显微镜图像
            return cv2.bilateralFilter(
                image,
                d=cfg.get('d', 9),
                sigmaColor=cfg.get('sigma_color', 75),
                sigmaSpace=cfg.get('sigma_space', 75)
            )
        elif method == 'median':
            # 中值滤波：去除椒盐噪声
            return cv2.medianBlur(image, ksize=cfg.get('d', 5))
        elif method == 'gaussian':
            # 高斯滤波
            ksize = cfg.get('d', 5)
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        else:
            return image

    def correct_illumination(self, image):
        """
        光照不均校正

        使用大核高斯模糊估计背景，然后原图减去背景
        """
        cfg = self.config.get('illumination_correction', {})
        if not cfg.get('enabled', True):
            return image

        kernel_size = cfg.get('kernel_size', 51)
        # 确保核大小为奇数
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        # 分别对三个通道进行背景估计和校正
        corrected = image.copy().astype(np.float32)
        for i in range(3):
            channel = image[:, :, i].astype(np.float32)
            # 估计背景（低频成分）
            background = cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)
            # 背景减除 + 灰度值恢复
            diff = channel - background + np.mean(background)
            corrected[:, :, i] = np.clip(diff, 0, 255)

        return corrected.astype(np.uint8)

    def enhance_contrast(self, image):
        """
        CLAHE 自适应直方图均衡化
        增强局部对比度，有助于区分蓝色树脂和矿物颗粒
        """
        cfg = self.config.get('contrast_enhancement', {})
        if not cfg.get('enabled', True):
            return image

        clip_limit = cfg.get('clip_limit', 2.0)
        tile_grid_size = tuple(cfg.get('tile_grid_size', [8, 8]))

        # 转换到 LAB 颜色空间，对 L 通道进行 CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
