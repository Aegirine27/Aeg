"""
深度学习分割模块（ONNX Runtime推理）

基于轻量U-Net模型，使用ONNX Runtime进行高效推理。
支持滑动窗口 + 高斯权重融合处理大尺寸图像。

使用方式:
    segmenter = DLSegmenter(config)
    result = segmenter.segment(image_dict)
    mask = result['mask']  # 二值掩膜
"""
import cv2
import numpy as np
from pathlib import Path

from .segmentation_base import BaseSegmenter


class DLSegmenter(BaseSegmenter):
    """基于深度学习的孔隙分割器（ONNX Runtime推理）

    使用滑动窗口对大尺寸图像（如3600x4800）进行分块推理，
    通过高斯权重融合消除拼接缝。
    """

    def __init__(self, config):
        super().__init__(config)
        dl_cfg = config.get('deep_learning', {}) if isinstance(config, dict) else {}
        self.model_path = dl_cfg.get('model_path', 'models/pore_segment.onnx')
        self.patch_size = dl_cfg.get('patch_size', 512)
        self.overlap = dl_cfg.get('overlap', 128)
        self.confidence_threshold = dl_cfg.get('confidence_threshold', 0.5)
        self.use_morphology = dl_cfg.get('use_morphology', True)
        self.use_gpu = dl_cfg.get('gpu', True)

        # 延迟加载：首次调用segment()时才加载模型
        self._session = None
        self._input_name = None
        self._input_shape = None
        self._warmup_done = False

        # 预计算高斯权重（用于patch融合）
        self._gaussian_weight = self._create_gaussian_weight(self.patch_size)

    def _create_gaussian_weight(self, size):
        """创建高斯权重图，中心权重高，边缘权重低"""
        y = np.linspace(-1, 1, size)
        x = np.linspace(-1, 1, size)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        # 二维高斯分布
        weight = np.exp(-(xx**2 + yy**2) / 0.5)
        return weight.astype(np.float32)

    def _load_model(self):
        """懒加载ONNX模型"""
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "使用深度学习方法需要安装ONNX Runtime:\n"
                "  pip install onnxruntime-gpu  (推荐，有GPU)\n"
                "  pip install onnxruntime      (仅CPU)"
            )

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"深度学习模型文件不存在: {self.model_path}\n"
                f"请先训练并导出ONNX模型，或从其他来源获取。"
            )

        # 配置ExecutionProvider
        providers = []
        if self.use_gpu:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        # 创建推理会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_file),
            sess_options=sess_options,
            providers=providers
        )

        # 获取输入信息
        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        raw_shape = input_meta.shape  # 可能是 [1, 3, 512, 512] 或 ['batch_size', 3, 'height', 'width']
        # 将字符串维度替换为 patch_size（ONNX dynamic axes 导致）
        self._input_shape = [
            self.patch_size if isinstance(s, str) else s for s in raw_shape
        ]

    def _warmup(self):
        """预热推理，首次GPU推理较慢"""
        if self._warmup_done:
            return

        dummy = np.zeros(
            (1, 3, self.patch_size, self.patch_size),
            dtype=np.float32
        )
        try:
            self._session.run(None, {self._input_name: dummy})
            self._warmup_done = True
        except Exception as e:
            print(f"模型预热失败: {e}")

    def segment(self, image_dict):
        """执行深度学习分割

        Args:
            image_dict: 预处理后的图像字典，建议使用 'enhanced' 键

        Returns:
            dict: {
                'mask': 二值掩膜 (H, W), 255=孔隙, 0=背景
                'method': 'deep_learning',
                'params': {...},
                'probability': 概率图 (H, W), 值域 [0, 1]
            }
        """
        self._load_model()
        self._warmup()

        # 使用增强后的图像作为输入
        image = image_dict.get('enhanced', image_dict.get('original'))
        if image is None:
            raise ValueError("image_dict 中必须包含 'enhanced' 或 'original' 图像")

        # 滑动窗口推理，得到概率图
        prob_map = self._sliding_window_inference(image)

        # 阈值化生成二值掩膜
        mask = (prob_map > self.confidence_threshold).astype(np.uint8) * 255

        # 可选：应用形态学操作（与阈值方法保持一致）
        if self.use_morphology:
            mask = self._apply_morphology(mask)

        return {
            'mask': mask,
            'method': 'deep_learning',
            'params': {
                'model_path': self.model_path,
                'patch_size': self.patch_size,
                'overlap': self.overlap,
                'confidence_threshold': self.confidence_threshold,
            },
            'probability': prob_map,
        }

    def _preprocess_patch(self, patch):
        """预处理patch为模型输入格式

        Args:
            patch: BGR图像 (patch_size, patch_size, 3)

        Returns:
            np.ndarray: (1, 3, H, W) float32, 归一化到 [0, 1]
        """
        # BGR -> RGB
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        # 归一化到 [0, 1]
        patch_norm = patch_rgb.astype(np.float32) / 255.0
        # HWC -> CHW -> NCHW
        patch_nchw = np.transpose(patch_norm, (2, 0, 1))[np.newaxis, ...]
        return patch_nchw

    def _infer_patch(self, patch):
        """对单个patch进行推理

        Args:
            patch: BGR图像 (patch_size, patch_size, 3)

        Returns:
            np.ndarray: 概率图 (patch_size, patch_size)
        """
        input_tensor = self._preprocess_patch(patch)

        # 如果模型期望的尺寸不同，进行resize
        expected_h = self._input_shape[2] if self._input_shape[2] is not None else self.patch_size
        expected_w = self._input_shape[3] if self._input_shape[3] is not None else self.patch_size

        if input_tensor.shape[2] != expected_h or input_tensor.shape[3] != expected_w:
            # 使用OpenCV resize NCHW -> 先转HWC -> resize -> 转回NCHW
            hwc = np.transpose(input_tensor[0], (1, 2, 0))  # (H, W, C)
            resized = cv2.resize(hwc, (expected_w, expected_h))
            input_tensor = np.transpose(resized, (2, 0, 1))[np.newaxis, ...]

        # 推理
        outputs = self._session.run(None, {self._input_name: input_tensor})

        # 假设输出是 (1, 1, H, W) 或 (1, H, W)
        output = outputs[0]
        if output.ndim == 4:
            prob = output[0, 0]  # (H, W)
        elif output.ndim == 3:
            prob = output[0]  # (H, W)
        else:
            prob = output

        # 如果输出尺寸与patch_size不同，resize回来
        if prob.shape[0] != self.patch_size or prob.shape[1] != self.patch_size:
            prob = cv2.resize(prob, (self.patch_size, self.patch_size))

        return prob.astype(np.float32)

    def _sliding_window_inference(self, image):
        """滑动窗口推理 + 高斯权重融合

        对大尺寸图像分块推理，使用高斯权重消除拼接缝。

        Args:
            image: BGR图像 (H, W, 3)

        Returns:
            np.ndarray: 概率图 (H, W), 值域 [0, 1]
        """
        h, w = image.shape[:2]
        patch = self.patch_size
        stride = patch - self.overlap

        # 计算需要padding的大小，使图像能被stride整除
        pad_h = (stride - h % stride) % stride
        pad_w = (stride - w % stride) % stride

        # 反射padding（保持边缘信息）
        padded = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            cv2.BORDER_REFLECT
        )
        ph, pw = padded.shape[:2]

        # 累加概率和权重
        prob_accum = np.zeros((ph, pw), dtype=np.float64)
        weight_accum = np.zeros((ph, pw), dtype=np.float64)

        # 滑动窗口
        y_positions = range(0, ph - patch + 1, stride)
        x_positions = range(0, pw - patch + 1, stride)

        for y in y_positions:
            for x in x_positions:
                # 提取patch
                patch_img = padded[y:y+patch, x:x+patch]

                # 推理
                patch_prob = self._infer_patch(patch_img)

                # 高斯权重融合
                prob_accum[y:y+patch, x:x+patch] += patch_prob * self._gaussian_weight
                weight_accum[y:y+patch, x:x+patch] += self._gaussian_weight

        # 归一化
        prob_map = prob_accum / (weight_accum + 1e-8)

        # 裁剪回原始尺寸
        return prob_map[:h, :w].astype(np.float32)

    def _apply_morphology(self, mask):
        """应用形态学操作（复用阈值方法的配置）"""
        morph_cfg = self.config.get('threshold_segmentation', {}).get('morphological_operations', {})
        if not morph_cfg:
            return mask

        iterations = morph_cfg.get('iterations', 2)

        # 开运算（去除小噪点）
        open_k = np.ones(tuple(morph_cfg.get('open_kernel', [3, 3])), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=iterations)

        # 闭运算（填充小孔洞）
        close_k = np.ones(tuple(morph_cfg.get('close_kernel', [5, 5])), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=iterations)

        return mask
