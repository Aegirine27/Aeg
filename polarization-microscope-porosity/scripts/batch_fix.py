"""
批量交互式修正 SAM 标注

自动逐个打开问题图像的交互式修正窗口。

使用方法:
    # 修正全部 10 张问题图像（自动开始）
    python scripts/batch_fix.py --auto

    # 只修正指定图像
    python scripts/batch_fix.py --names N9-4-(-)10X N10-1-(-)10X --auto

    # 修正所有 44 张图像
    python scripts/batch_fix.py --all --auto
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.sam_annotator import SAMAnnotator


# 默认的问题图像列表（按严重度排序）
DEFAULT_PROBLEM_IMAGES = [
    "N9-4-(-)10X",       # 76.33%
    "N2-3-(-)10X",       # 56.01%
    "N10-1-(-)10X",      # 55.39%
    "N10-5-(-)10X",      # 53.24%
    "N2-1-(-)5X",        # 52.21%
    "N10-8-(-)10X",      # 51.92%
    "N24-(11)-1-(-)10X", # 51.38%
    "N10-3-(-)5X",       # 47.81%
    "N9-6-(-)10X",       # 47.76%
    "N10-6-(-)10X",      # 46.72%
]


def interactive_fix(image_path: Path, mask_path: Path, sam_annotator: SAMAnnotator):
    """
    交互式修正单张图像的 mask。
    直接调用 SAMAnnotator 的方法，在同一进程中运行。
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"  ❌ 无法读取图像: {image_path}")
        return False

    # 读取当前 mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  ⚠️ 未找到现有 mask，将创建空白 mask")
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    # 初始化 SAM
    sam_annotator._load_sam()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    sam_annotator._predictor.set_image(img_rgb)

    # 当前 mask 状态
    current_mask = (mask > 0).astype(bool)
    click_points = []
    click_labels = []  # 1 = 前景, 0 = 背景
    # 窗口标题不能用 Windows 路径保留字符 (<>\"/\|?*:)
    window_name = f"Fix_{image_path.stem}"
    logits = None

    def update_display():
        """更新窗口显示"""
        display = img_bgr.copy()
        overlay = np.zeros_like(display)
        overlay[current_mask] = [0, 255, 0]  # 绿色表示 mask
        display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

        # 画出点击点
        for pt, lbl in zip(click_points, click_labels):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(display, tuple(pt), 6, color, -1)
            cv2.circle(display, tuple(pt), 8, (255, 255, 255), 2)

        # 显示孔隙率
        porosity = np.sum(current_mask) / current_mask.size * 100
        status_text = f"Porosity: {porosity:.1f}% | Points: {len(click_points)}"
        cv2.putText(display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # 操作提示
        hint = "L=Add R=Del S=Save Rst=Reset ESC=Quit"
        cv2.putText(display, hint, (10, img_bgr.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(display, hint, (10, img_bgr.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow(window_name, display)

    def predict_mask():
        """根据点击点重新预测 mask"""
        nonlocal current_mask, logits
        if len(click_points) == 0:
            return

        coords = np.array(click_points)
        labels = np.array(click_labels)

        masks, scores, new_logits = sam_annotator._predictor.predict(
            point_coords=coords,
            point_labels=labels,
            mask_input=logits[None, :, :] if logits is not None else None,
            multimask_output=True,
        )

        best_idx = np.argmax(scores)
        current_mask = masks[best_idx]
        logits = new_logits[best_idx]

    def mouse_callback(event, x, y, flags, param):
        nonlocal logits
        if event == cv2.EVENT_LBUTTONDOWN:
            click_points.append([x, y])
            click_labels.append(1)
            predict_mask()
            update_display()
        elif event == cv2.EVENT_RBUTTONDOWN:
            click_points.append([x, y])
            click_labels.append(0)
            predict_mask()
            update_display()

    # 创建窗口
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)
        cv2.setMouseCallback(window_name, mouse_callback)
    except Exception as e:
        print(f"  ❌ 创建窗口失败: {e}")
        return False

    # 初始显示
    try:
        update_display()
    except Exception as e:
        print(f"  ❌ 显示图像失败: {e}")
        cv2.destroyWindow(window_name)
        return False

    print("  操作说明:")
    print("    左键点击 = 添加前景（孔隙）")
    print("    右键点击 = 添加背景（删除误检）")
    print("    S        = 保存并继续下一张")
    print("    R        = 重置所有点击")
    print("    ESC      = 放弃保存，保持原 mask")
    print("  窗口标题栏可以拖动调整大小")

    saved = False
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('s'):  # 保存
            saved = True
            break
        elif key == ord('r'):  # 重置
            click_points.clear()
            click_labels.clear()
            logits = None
            current_mask = (mask > 0).astype(bool)
            update_display()
            print("  🔄 已重置")
        elif key == 27:  # ESC
            print("  ⏭️  已跳过（未保存）")
            break

    cv2.destroyWindow(window_name)

    if saved:
        # 保存 mask
        out_mask = (current_mask.astype(np.uint8)) * 255
        cv2.imwrite(str(mask_path), out_mask)
        porosity = np.sum(current_mask) / current_mask.size * 100
        print(f"  ✅ 已保存: {mask_path} (孔隙率: {porosity:.1f}%)")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description='批量交互式修正 SAM 标注')
    parser.add_argument('--names', '-n', nargs='+', default=None,
                        help='指定要修正的图像名（不含扩展名），默认修正全部问题图像')
    parser.add_argument('--img-dir', '-i', default='d:/AI/Test',
                        help='图像目录')
    parser.add_argument('--label-dir', '-o', default='data/labels',
                        help='标注输出目录')
    parser.add_argument('--all', '-a', action='store_true',
                        help='修正所有 44 张图像（不只是问题图像）')
    parser.add_argument('--auto', action='store_true',
                        help='自动模式：跳过开头确认')

    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)

    # 确定要修正的图像列表
    if args.all:
        image_files = sorted([f for f in img_dir.iterdir()
                              if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        names = [f.stem for f in image_files]
    elif args.names:
        names = args.names
    else:
        names = DEFAULT_PROBLEM_IMAGES

    print(f"准备修正 {len(names)} 张图像")
    print(f"图像目录: {img_dir}")
    print(f"标注目录: {label_dir}")
    print()

    if not args.auto:
        input("按 Enter 开始...")

    # 初始化 SAM（只初始化一次）
    print("正在加载 SAM 模型...")
    try:
        annotator = SAMAnnotator(model_path="models/sam_vit_b_01ec64.pth")
        annotator._load_sam()
        print("✅ SAM 模型加载完成\n")
    except Exception as e:
        print(f"❌ SAM 加载失败: {e}")
        return

    completed = []
    skipped = []

    for idx, name in enumerate(names, 1):
        img_path = img_dir / f"{name}.jpg"
        mask_path = label_dir / f"{name}_mask.png"

        if not img_path.exists():
            print(f"[{idx}/{len(names)}] ❌ 图像不存在: {img_path}")
            skipped.append(name)
            continue

        print(f"\n[{idx}/{len(names)}] 🖼️  {name}")
        print(f"    图像: {img_path}")

        # 读取当前孔隙率
        current_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if current_mask is not None:
            porosity = np.sum(current_mask > 0) / current_mask.size * 100
            print(f"    当前孔隙率: {porosity:.1f}%")

        # 启动交互式修正
        try:
            if interactive_fix(img_path, mask_path, annotator):
                completed.append(name)
            else:
                skipped.append(name)
        except Exception as e:
            print(f"  ❌ 修正过程出错: {e}")
            import traceback
            traceback.print_exc()
            skipped.append(name)

    print(f"\n{'='*60}")
    print(f"修正完成: {len(completed)}/{len(names)} 张")
    if completed:
        print("已修正:")
        for name in completed:
            print(f"  ✅ {name}")
    if skipped:
        print("未修正:")
        for name in skipped:
            print(f"  ⏭️ {name}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
