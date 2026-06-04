"""
U-Net 分割模型定义

使用 segmentation-models-pytorch 库构建轻量U-Net，
以 MobileNetV2 为backbone，适合8GB显存训练。
"""
import torch
import torch.nn as nn


def create_model(encoder='mobilenet_v2', encoder_weights='imagenet',
                 in_channels=3, classes=1, activation=None):
    """创建U-Net分割模型

    Args:
        encoder: 编码器名称，默认 mobilenet_v2（轻量快速）
        encoder_weights: 预训练权重，默认 imagenet
        in_channels: 输入通道数（RGB=3）
        classes: 输出类别数（二分类=1）
        activation: 输出激活函数，None表示使用原始logits

    Returns:
        smp.Unet: U-Net模型实例
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "训练模型需要安装 segmentation-models-pytorch:\n"
            "  pip install segmentation-models-pytorch>=0.3.3"
        )

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )

    return model


def get_model_summary(model, input_size=(1, 3, 512, 512)):
    """打印模型摘要信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    encoder_name = getattr(getattr(model, 'encoder', None), 'name', 'unknown')
    print(f"模型架构: U-Net + {encoder_name}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: ~{total_params * 4 / 1024 / 1024:.1f}MB (float32)")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / 1024 / 1024,
    }


if __name__ == '__main__':
    # 测试模型创建
    print("测试模型创建...")
    model = create_model()
    get_model_summary(model)

    # 测试前向传播
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print("模型测试通过！")
