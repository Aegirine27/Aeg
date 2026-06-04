"""
统一分割接口基类模块

为阈值分割、分水岭分割、深度学习分割提供统一的抽象接口，
使三种方法可以无缝互换使用。
"""
from abc import ABC, abstractmethod


class BaseSegmenter(ABC):
    """孔隙分割器抽象基类

    所有分割方法（阈值、分水岭、深度学习）必须实现此接口，
    返回统一的字典格式，便于后续的面积统计和可视化处理。
    """

    def __init__(self, config):
        """
        Args:
            config: 包含该分割方法所需参数的字典或配置对象
        """
        self.config = config

    @abstractmethod
    def segment(self, image_dict):
        """执行图像分割

        Args:
            image_dict: 预处理后的图像字典，包含 'original', 'enhanced',
                       'hsv', 'lab' 等键

        Returns:
            dict: 必须包含以下键
                - mask: np.ndarray, 二值掩膜 (H, W), 255=孔隙, 0=背景
                - method: str, 分割方法名称 ('threshold', 'watershed', 'deep_learning')
                - params: dict, 使用的参数记录
                - probability: np.ndarray, 可选，概率图 (H, W) 值域 [0, 1]
        """
        pass

    def get_method_name(self):
        """获取方法名称（子类可覆盖）"""
        return self.__class__.__name__.replace('Segmenter', '').replace('DL', 'deep_learning').lower()
