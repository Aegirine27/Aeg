"""
工具函数模块
"""
import cv2
import numpy as np
import yaml
from pathlib import Path
from PIL import Image


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
    """
    加载图像

    支持格式：
        - OpenCV原生: .jpg, .jpeg, .png, .bmp, .webp, .pbm, .pgm, .ppm
        - Pillow后备: .tif, .tiff, .gif(首帧), .ico, .pcx, .psd等
    """
    path_str = str(path)
    path_obj = Path(path_str)

    # 检查文件是否存在
    if not path_obj.exists():
        raise FileNotFoundError(f"文件不存在: {path_str}")

    # 首先尝试OpenCV加载
    image = cv2.imread(str(path_obj.resolve()), flag)
    if image is not None:
        return image

    # OpenCV失败，尝试Pillow加载并转换
    # 使用文件对象打开，避免路径解析问题
    try:
        with open(path_obj, 'rb') as f:
            pil_image = Image.open(f)

            # 处理多帧图像（如GIF），只取第一帧
            if hasattr(pil_image, 'n_frames') and pil_image.n_frames > 1:
                pil_image.seek(0)

            # 转换为RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # PIL Image -> numpy array (RGB)
            image = np.array(pil_image)

        # RGB -> BGR (OpenCV格式)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    except Exception as e:
        import traceback
        ext = path_obj.suffix.lower()
        supported = '.jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp, .gif'
        tb_str = traceback.format_exc()
        raise FileNotFoundError(
            f"无法加载图像: {path_str}\n"
            f"格式: {ext}\n"
            f"文件大小: {path_obj.stat().st_size / 1024:.1f} KB\n"
            f"支持的格式: {supported}\n"
            f"错误: {e}\n"
            f"堆栈:\n{tb_str[-500:]}"  # 只显示最后500字符
        )


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
