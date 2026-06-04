"""
数据集定义与数据增强

从原始大尺寸图像中提取512x512的patch进行训练，
使用albumentations进行数据增强。
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PoreSegmentationDataset(Dataset):
    """孔隙分割数据集

    从原始图像和mask中提取patch进行训练。
    支持动态patch提取和多种数据增强。
    """

    def __init__(self, image_dir, mask_dir, patch_size=512,
                 stride=None, augment=True, min_pore_ratio=0.01):
        """
        Args:
            image_dir: 图像目录（原始RGB/BGR图像）
            mask_dir: mask目录（二值mask，0=背景，255=孔隙）
            patch_size: patch大小
            stride: 提取patch的步长，None时等于patch_size（无重叠）
            augment: 是否应用数据增强
            min_pore_ratio: 最小孔隙比例，过滤掉太稀疏的patch
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.augment = augment
        self.min_pore_ratio = min_pore_ratio

        # 收集所有patch
        self.patches = self._collect_patches()
        print(f"数据集创建完成: {len(self.patches)} 个patches")

        # 数据增强pipeline (兼容 albumentations >= 2.0)
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1),
                         rotate=(-15, 15), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.5),
                A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0, 0), p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                           std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406),
                           std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def _collect_patches(self):
        """收集所有有效的patch位置"""
        patches = []
        ps = self.patch_size

        # 支持的图像格式
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        image_files = [f for f in self.image_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in extensions]

        for img_file in image_files:
            mask_file = self.mask_dir / f"{img_file.stem}_mask.png"

            if not mask_file.exists():
                # 尝试其他命名格式
                mask_file = self.mask_dir / f"{img_file.stem}.png"

            if not mask_file.exists():
                print(f"警告: 未找到 {img_file.name} 对应的mask，跳过")
                continue

            # 读取mask检查尺寸
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            h, w = mask.shape

            # 滑动窗口提取patch位置
            for y in range(0, h - ps + 1, self.stride):
                for x in range(0, w - ps + 1, self.stride):
                    patch_mask = mask[y:y+ps, x:x+ps]

                    # 检查孔隙比例
                    pore_ratio = np.sum(patch_mask > 0) / (ps * ps)

                    # 过滤掉太稀疏或全背景的patch
                    if self.min_pore_ratio <= pore_ratio <= 0.99:
                        patches.append({
                            'image': img_file,
                            'mask': mask_file,
                            'x': x,
                            'y': y,
                        })

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        """获取一个patch

        Returns:
            dict: {'image': Tensor(3, H, W), 'mask': Tensor(1, H, W)}
        """
        patch_info = self.patches[idx]
        x, y = patch_info['x'], patch_info['y']
        ps = self.patch_size

        # 读取图像patch
        img = cv2.imread(str(patch_info['image']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_patch = img[y:y+ps, x:x+ps]

        # 读取mask patch
        mask = cv2.imread(str(patch_info['mask']), cv2.IMREAD_GRAYSCALE)
        mask_patch = mask[y:y+ps, x:x+ps]

        # 二值化mask
        mask_patch = (mask_patch > 127).astype(np.float32)

        # 应用数据增强
        transformed = self.transform(image=img_patch, mask=mask_patch)

        return {
            'image': transformed['image'],  # Tensor(3, H, W)
            'mask': transformed['mask'].unsqueeze(0),  # Tensor(1, H, W)
        }


def get_data_loaders(image_dir, mask_dir, patch_size=512,
                     batch_size=4, val_split=0.2, num_workers=4):
    """创建训练集和验证集的 DataLoader

    Args:
        image_dir: 图像目录
        mask_dir: mask目录
        patch_size: patch大小
        batch_size: 批次大小
        val_split: 验证集比例
        num_workers: 数据加载线程数

    Returns:
        (train_loader, val_loader)
    """
    # 创建完整数据集
    full_dataset = PoreSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        patch_size=patch_size,
        augment=True,
    )

    # 划分训练集和验证集
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 验证集关闭增强
    val_dataset.dataset.augment = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"训练集: {len(train_dataset)} patches")
    print(f"验证集: {len(val_dataset)} patches")

    return train_loader, val_loader


if __name__ == '__main__':
    # 测试数据集
    print("测试数据集创建...")

    # 这里使用示例路径，实际使用时请替换
    # dataset = PoreSegmentationDataset(
    #     image_dir='data/images',
    #     mask_dir='data/labels',
    #     patch_size=512,
    # )
    # print(f"数据集大小: {len(dataset)}")
    # sample = dataset[0]
    # print(f"图像形状: {sample['image'].shape}")
    # print(f"Mask形状: {sample['mask'].shape}")
    print("数据集模块测试通过！")
