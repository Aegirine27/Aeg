"""
将训练好的 PyTorch 模型导出为 ONNX 格式

ONNX 模型可被 ONNX Runtime 高效推理，无需 PyTorch 依赖。

使用方法:
    python training/export_onnx.py --checkpoint checkpoints/best_model.pth --output models/pore_segment.onnx
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from model import create_model


def export_onnx(checkpoint_path, output_path, model=None, input_size=(1, 3, 512, 512)):
    """导出模型为 ONNX 格式

    Args:
        checkpoint_path: PyTorch 模型权重路径 (.pth)
        output_path: ONNX 输出路径
        model: 已创建的模型实例（可选，None时自动创建）
        input_size: 输入尺寸 (N, C, H, W)

    Returns:
        str: 输出文件路径
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建或加载模型
    if model is None:
        print("创建模型...")
        model = create_model()
        model = model.to(device)

    # 加载权重
    print(f"加载权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 创建 dummy 输入
    dummy_input = torch.randn(*input_size, device=device)

    # 导出 ONNX
    print(f"导出 ONNX: {output_path}")
    print(f"输入尺寸: {input_size}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'},
            },
        )

    print(f"ONNX 模型导出成功: {output_path}")

    # 验证导出的模型
    print("\n验证 ONNX 模型...")
    _verify_onnx(output_path, dummy_input.cpu().numpy())

    return str(output_path)


def _verify_onnx(onnx_path, pytorch_input):
    """验证 ONNX 模型输出与 PyTorch 一致"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("警告: 未安装 onnxruntime，跳过验证")
        return

    # 加载 ONNX 模型
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    # ONNX 推理
    onnx_output = session.run(None, {input_name: pytorch_input})[0]

    print(f"ONNX 输出形状: {onnx_output.shape}")
    print(f"ONNX 输出范围: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")
    print("验证通过！")


def main():
    parser = argparse.ArgumentParser(description='导出 ONNX 模型')
    parser.add_argument('--checkpoint', '-c', required=True,
                        help='PyTorch 模型权重路径 (.pth)')
    parser.add_argument('--output', '-o', default='models/pore_segment.onnx',
                        help='ONNX 输出路径')
    parser.add_argument('--encoder', default='mobilenet_v2',
                        help='模型编码器类型')

    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )

    print("\n使用 ONNX 模型进行推理:")
    print("  python main.py -i image.jpg --method dl")


if __name__ == '__main__':
    main()
