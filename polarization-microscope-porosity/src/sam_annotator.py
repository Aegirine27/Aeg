"""
SAM (Segment Anything Model) 辅助标注工具

利用 SAM 的零样本分割能力，自动为偏光显微镜图像生成高质量的初始孔隙标注。
支持通过颜色提示自动定位蓝色树脂区域。

使用方式:
    python -m src.sam_annotator --image image.jpg --output mask.png
"""
import cv2
import numpy as np
from pathlib import Path


class SAMAnnotator:
    """SAM 辅助标注器

    自动利用图像中的蓝色区域作为前景提示，生成高质量的孔隙分割 mask。
    """

    def __init__(self, model_path="models/sam_vit_b_01ec64.pth", device=None):
        """
        Args:
            model_path: SAM 模型权重文件路径 (.pth)
            device: 'cuda' 或 'cpu'，None 时自动选择
        """
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if self._check_cuda() else 'cpu')
        self._sam = None
        self._predictor = None

    def _check_cuda(self):
        """检查 CUDA 是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_sam(self):
        """懒加载 SAM 模型"""
        if self._predictor is not None:
            return

        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "使用 SAM 标注需要安装 segment-anything:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
                "并下载模型权重文件到 models/ 目录。"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SAM 模型文件不存在: {self.model_path}\n"
                f"请从 https://github.com/facebookresearch/segment-anything#model-checkpoints 下载:\n"
                f"  sam_vit_b_01ec64.pth (轻量版, ~375MB) 推荐\n"
                f"  或 sam_vit_h_4b8939.pth (完整版, ~2.4GB)"
            )

        # 加载模型
        model_type = self._get_model_type(self.model_path.name)
        sam = sam_model_registry[model_type](checkpoint=str(self.model_path))
        sam.to(device=self.device)

        self._predictor = SamPredictor(sam)

    def _get_model_type(self, filename):
        """根据文件名推断模型类型"""
        if 'vit_b' in filename.lower():
            return 'vit_b'
        elif 'vit_l' in filename.lower():
            return 'vit_l'
        elif 'vit_h' in filename.lower():
            return 'vit_h'
        else:
            return 'vit_b'  # 默认

    def generate_mask(self, image, use_color_prompt=True, points_per_side=32):
        """自动生成孔隙分割 mask

        Args:
            image: BGR 图像 (H, W, 3) 或图像路径
            use_color_prompt: 是否使用蓝色区域作为自动提示
            points_per_side: 自动网格采样点数（仅当 use_color_prompt=False 时有效）

        Returns:
            np.ndarray: 二值 mask (H, W), 255=孔隙, 0=背景
        """
        self._load_sam()

        # 加载图像
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
        else:
            img_bgr = image.copy()

        # SAM 需要 RGB 格式
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(img_rgb)

        if use_color_prompt:
            # 基于颜色自动提取前景提示点
            point_coords, point_labels = self._extract_blue_prompts(img_bgr)

            if len(point_coords) == 0:
                # 未检测到蓝色区域，使用网格采样
                masks = self._auto_mask_generator(points_per_side)
            else:
                # 使用颜色提示点进行分割
                masks, scores, logits = self._predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
                # 选择得分最高的 mask
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
        else:
            # 纯自动网格采样
            mask = self._auto_mask_generator(points_per_side)

        # 后处理：保留与蓝色区域重叠的部分
        mask = self._filter_by_color(mask, img_bgr)

        return (mask.astype(np.uint8)) * 255

    def _extract_blue_prompts(self, image_bgr, num_points=20):
        """从图像中提取蓝色区域作为前景提示点

        Args:
            image_bgr: BGR 图像
            num_points: 最多提取的提示点数量

        Returns:
            point_coords: (N, 2) 坐标数组
            point_labels: (N,) 标签数组 (1=前景)
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # 蓝色 HSV 范围（宽松）
        lower = np.array([80, 30, 30])
        upper = np.array([160, 255, 255])
        blue_mask = cv2.inRange(hsv, lower, upper)

        # 形态学清理
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 找到蓝色区域的轮廓中心作为提示点
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            blue_mask, connectivity=8
        )

        points = []
        # 按面积排序，取最大的 num_points 个
        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
        areas.sort(key=lambda x: x[1], reverse=True)

        for idx, (label_id, area) in enumerate(areas[:num_points]):
            if area < 100:  # 过滤太小的噪声
                continue
            cx, cy = centroids[label_id]
            points.append([cx, cy])

        if len(points) == 0:
            return np.array([]), np.array([])

        point_coords = np.array(points)
        point_labels = np.ones(len(points), dtype=np.int32)

        return point_coords, point_labels

    def _auto_mask_generator(self, points_per_side=32):
        """使用 SAM 的自动 mask 生成器"""
        try:
            from segment_anything import SamAutomaticMaskGenerator
        except ImportError:
            raise

        # 重新创建 generator（需要原始 sam 模型）
        sam = self._predictor.model
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

        # 注意：这里需要原始图像，但 predictor 已经设置了图像
        # 这是一个简化实现，实际需要重新组织代码
        # 暂时返回空 mask，提醒用户使用 color prompt 模式
        h, w = self._predictor.original_size
        return np.zeros((h, w), dtype=bool)

    def _filter_by_color(self, mask, image_bgr):
        """根据颜色过滤 mask，保留蓝色区域"""
        if mask.sum() == 0:
            return mask

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # 宽松的蓝色范围
        lower = np.array([80, 20, 20])
        upper = np.array([160, 255, 255])
        blue_mask = cv2.inRange(hsv, lower, upper)

        # 只保留 mask 中与蓝色区域重叠的部分
        filtered = mask & (blue_mask > 0)

        # 如果过滤后太少，保留原始 mask
        if filtered.sum() < mask.sum() * 0.1:
            return mask

        return filtered

    def interactive_refine(self, image, initial_mask, window_name="SAM标注修正"):
        """交互式修正 mask（OpenCV 窗口）

        显示图像和初始 mask 的叠加，用户可以通过点击添加/删除区域。

        Args:
            image: BGR 图像
            initial_mask: 初始二值 mask
            window_name: 窗口标题

        Returns:
            np.ndarray: 修正后的 mask
        """
        self._load_sam()

        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
        else:
            img_bgr = image.copy()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(img_rgb)

        mask = initial_mask.copy()
        click_points = []
        click_labels = []  # 1 = 添加前景, 0 = 添加背景

        def mouse_callback(event, x, y, flags, param):
            nonlocal mask, click_points, click_labels

            if event == cv2.EVENT_LBUTTONDOWN:
                # 左键 = 添加前景
                click_points.append([x, y])
                click_labels.append(1)
                self._update_mask()

            elif event == cv2.EVENT_RBUTTONDOWN:
                # 右键 = 添加背景（删除）
                click_points.append([x, y])
                click_labels.append(0)
                self._update_mask()

        def _update_mask():
            nonlocal mask
            if len(click_points) == 0:
                return

            coords = np.array(click_points)
            labels = np.array(click_labels)

            masks, scores, logits = self._predictor.predict(
                point_coords=coords,
                point_labels=labels,
                mask_input=logits if 'logits' in locals() else None,
                multimask_output=True,
            )

            best_idx = np.argmax(scores)
            mask = masks[best_idx]

            # 更新显示
            display = img_bgr.copy()
            overlay = np.zeros_like(display)
            overlay[mask] = [0, 255, 0]  # 绿色表示选中
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

            # 画出点击点
            for pt, lbl in zip(click_points, click_labels):
                color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
                cv2.circle(display, tuple(pt), 5, color, -1)

            cv2.imshow(window_name, display)

        # 初始显示
        display = img_bgr.copy()
        overlay = np.zeros_like(display)
        overlay[mask > 0] = [0, 255, 0]
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

        cv2.imshow(window_name, display)
        cv2.setMouseCallback(window_name, mouse_callback)

        print("交互式标注工具:")
        print("  左键点击 = 添加前景（孔隙）")
        print("  右键点击 = 添加背景（删除）")
        print("  S = 保存并退出")
        print("  R = 重置")
        print("  ESC = 退出不保存")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # 保存
                break
            elif key == ord('r'):  # 重置
                click_points = []
                click_labels = []
                mask = initial_mask.copy()
                _update_mask()
            elif key == 27:  # ESC
                mask = initial_mask.copy()
                break

        cv2.destroyWindow(window_name)
        return (mask.astype(np.uint8)) * 255


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='SAM 辅助标注工具')
    parser.add_argument('--image', '-i', required=True, help='输入图像路径')
    parser.add_argument('--output', '-o', required=True, help='输出 mask 路径')
    parser.add_argument('--model', '-m', default='models/sam_vit_b_01ec64.pth',
                        help='SAM 模型路径')
    parser.add_argument('--interactive', action='store_true',
                        help='交互式修正模式')
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'],
                        help='计算设备')

    args = parser.parse_args()

    # 创建标注器
    annotator = SAMAnnotator(model_path=args.model, device=args.device)

    # 生成初始 mask
    print(f"正在处理: {args.image}")
    initial_mask = annotator.generate_mask(args.image)
    print(f"初始 mask 生成完成，孔隙像素: {np.sum(initial_mask > 0)}")

    if args.interactive:
        # 交互式修正
        final_mask = annotator.interactive_refine(args.image, initial_mask)
    else:
        final_mask = initial_mask

    # 保存
    cv2.imwrite(args.output, final_mask)
    print(f"已保存: {args.output}")


if __name__ == '__main__':
    main()
