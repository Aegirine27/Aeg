"""
工具函数模块
"""
import cv2
import numpy as np
import yaml
from pathlib import Path


def load_config(config_path="config.yaml"):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_image(image, path):
    """保存图像，自动创建目录"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def load_image(path, flag=cv2.IMREAD_COLOR):
    """加载图像"""
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    return image


def detect_scale_bar(image):
    """
    从图像中检测比例尺并计算像素-实际距离转换

    策略：寻找图像底部/角落的黑色/白色线条，
          通常比例尺是一条带有数字标注的线段

    Returns:
        scale: float or None, μm/pixel
    """
    # TODO: 实现比例尺自动识别
    # 暂时返回 None，使用用户配置或手动输入
    return None


def resize_for_display(image, max_size=800):
    """
    调整图像大小以便显示，保持长宽比
    """
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def apply_mask(image, mask, color=[255, 0, 0], alpha=0.5):
    """
    将掩膜叠加到原图上，用于可视化

    Args:
        image: 原图 (H, W, 3)
        mask: 二值掩膜 (H, W)
        color: 叠加颜色 [B, G, R]
        alpha: 透明度
    """
    overlay = image.copy()
    for i in range(3):
        overlay[:, :, i] = np.where(mask > 0,
                                     overlay[:, :, i] * (1 - alpha) + color[i] * alpha,
                                     overlay[:, :, i])
    return overlay.astype(np.uint8)


def safe_divide(a, b, default=0.0):
    """安全除法，避免除以零"""
    return a / b if b != 0 else default
