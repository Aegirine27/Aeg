"""
U-Net 训练脚本

训练孔隙分割模型，支持：
- 混合精度训练（节省显存）
- 学习率调度
- 早停
- TensorBoard 可视化
- 自动保存最佳模型

使用方法:
    python training/train.py --images data/images --labels data/labels --epochs 100
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time

from model import create_model, get_model_summary
from dataset import get_data_loaders


def dice_loss(pred, target, smooth=1.0):
    """Dice Loss"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def combined_loss(pred, target):
    """组合损失: BCE + Dice"""
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def train_epoch(model, train_loader, optimizer, scaler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(train_loader, desc="训练")
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()

        # 混合精度前向传播
        with autocast():
            outputs = model(images)
            loss = combined_loss(outputs, masks)

        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 计算Dice系数（用于显示）
        with torch.no_grad():
            pred = torch.sigmoid(outputs) > 0.5
            dice = (2. * (pred * masks).sum() / (pred.sum() + masks.sum() + 1e-8)).item()

        total_loss += loss.item()
        total_dice += dice

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
        })

    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)

    return avg_loss, avg_dice


@torch.no_grad()
def validate(model, val_loader, device):
    """验证"""
    model.eval()
    total_loss = 0
    total_dice = 0

    for batch in tqdm(val_loader, desc="验证"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        outputs = model(images)
        loss = combined_loss(outputs, masks)

        pred = torch.sigmoid(outputs) > 0.5
        dice = (2. * (pred * masks).sum() / (pred.sum() + masks.sum() + 1e-8)).item()

        total_loss += loss.item()
        total_dice += dice

    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)

    return avg_loss, avg_dice


def main():
    parser = argparse.ArgumentParser(description='训练孔隙分割U-Net')
    parser.add_argument('--images', '-i', required=True, help='图像目录')
    parser.add_argument('--labels', '-l', required=True, help='标签mask目录')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', '-b', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--patch-size', type=int, default=512, help='patch大小')
    parser.add_argument('--val-split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--output', '-o', default='checkpoints', help='模型保存目录')
    parser.add_argument('--encoder', default='mobilenet_v2', help='编码器类型')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='计算设备')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据加载器
    print("\n加载数据集...")
    train_loader, val_loader = get_data_loaders(
        image_dir=args.images,
        mask_dir=args.labels,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    if len(train_loader) == 0:
        print("错误: 训练集为空，请检查数据路径")
        return

    # 创建模型
    print("\n创建模型...")
    model = create_model(encoder=args.encoder)
    model = model.to(device)
    get_model_summary(model)

    # 优化器和学习率调度
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # 混合精度训练
    scaler = GradScaler()

    # 训练循环
    print(f"\n开始训练: {args.epochs} epochs")
    print("-" * 50)

    best_dice = 0
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = 15

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # 训练
        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, scaler, device
        )

        # 验证
        val_loss, val_dice = validate(model, val_loader, device)

        # 学习率调度
        scheduler.step(val_dice)

        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch}/{args.epochs}] "
              f"| 训练 Loss: {train_loss:.4f} Dice: {train_dice:.4f} "
              f"| 验证 Loss: {val_loss:.4f} Dice: {val_dice:.4f} "
              f"| 耗时: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            patience_counter = 0

            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'config': vars(args),
            }, checkpoint_path)
            print(f"  -> 保存最佳模型 (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= early_stop_patience:
            print(f"\n早停: {early_stop_patience} 轮无改善")
            break

    print("\n" + "=" * 50)
    print("训练完成!")
    print(f"最佳模型: Epoch {best_epoch}, Dice: {best_dice:.4f}")
    print(f"模型保存: {output_dir / 'best_model.pth'}")
    print("=" * 50)

    # 导出ONNX
    print("\n导出ONNX模型...")
    try:
        from export_onnx import export_onnx
        export_onnx(
            checkpoint_path=output_dir / 'best_model.pth',
            output_path=output_dir / 'model.onnx',
            model=model,
        )
        print(f"ONNX模型: {output_dir / 'model.onnx'}")
    except Exception as e:
        print(f"ONNX导出失败: {e}")
        print("可稍后手动运行: python training/export_onnx.py")


if __name__ == '__main__':
    main()
