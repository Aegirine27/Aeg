# 偏光显微镜面孔率识别系统

基于计算机视觉的偏光显微镜单偏光照片孔隙与裂缝自动识别系统。

## 功能特性

- ✅ **自动识别蓝色树脂填充的孔隙和裂缝**
- ✅ **三种分割方法** — 颜色阈值 / 分水岭 / 深度学习(U-Net)
- ✅ **面孔率精确计算** — (孔隙+裂缝面积) / 全视域面积
- ✅ **批量处理** — 支持文件夹批量分析
- ✅ **结果可视化** — 孔隙纯蓝标注，背景纯白
- ✅ **统计导出** — JSON / CSV / 对比图表
- ✅ **SAM辅助标注** — 利用Segment Anything生成高质量训练数据
- ✅ **交互式标注修正** — 画笔工具人工修正mask

## 算法路线

```
输入图像
  ↓
预处理（去噪 → 光照校正 → 对比度增强）
  ↓
【三种分割方法可选】
  ├─ 颜色阈值分割（HSV/Lab 颜色空间）— 快速，适合边缘清晰
  ├─ 分水岭算法 — 处理粘连孔隙
  └─ 深度学习U-Net — 高精度，适合复杂场景
  ↓
面孔率计算 + 裂缝检测
  ↓
输出标注图像 + 统计结果
```

## 安装

### 基础安装（仅使用阈值/分水岭方法）

```bash
# 克隆仓库
git clone https://github.com/Aegirine27/Aeg.git
cd polarization-microscope-porosity

# 安装依赖
pip install -r requirements.txt
```

### 完整安装（包含深度学习训练）

```bash
# 安装基础依赖 + 训练依赖
pip install -r requirements.txt
pip install -r requirements-train.txt

# 下载 SAM 模型权重（用于辅助标注，推荐轻量版）
# 从 https://github.com/facebookresearch/segment-anything#model-checkpoints 下载
# 放置到 models/sam_vit_b_01ec64.pth
```

## 使用方法

### 单张图像处理

```bash
# 颜色阈值分割（默认，最快）
python main.py -i path/to/image.jpg -o results/

# 分水岭算法
python main.py -i path/to/image.jpg -o results/ --method watershed

# 深度学习（需要训练好的ONNX模型）
python main.py -i path/to/image.jpg -o results/ --method dl

# 交互式调参
python main.py -i path/to/image.jpg --interactive
```

### 批量处理

```bash
# 批量处理文件夹
python main.py -i path/to/folder/ -o results/ --batch

# 批量 + 指定方法
python main.py -i path/to/folder/ -o results/ --batch --method dl
```

### Python API

```python
from main import PorosityAnalyzer

analyzer = PorosityAnalyzer('config.yaml')

# 分析单张图像 — 选择分割方法
result = analyzer.analyze('image.jpg', method='threshold')   # 阈值
result = analyzer.analyze('image.jpg', method='watershed')   # 分水岭
result = analyzer.analyze('image.jpg', method='deep_learning')  # 深度学习

print(f"面孔率: {result['stats']['porosity_percent']:.4f}%")

# 保存结果
analyzer.save_results(result, 'output/')
```

## 三种分割方法对比

| 方法 | 速度 | 精度 | 适用场景 | 依赖 |
|------|------|------|----------|------|
| **颜色阈值** | ⭐⭐⭐ 最快 | ⭐⭐ 中等 | 蓝色均匀、边缘清晰 | 无 |
| **分水岭** | ⭐⭐ 中等 | ⭐⭐ 中等 | 孔隙粘连、结构复杂 | 无 |
| **深度学习** | ⭐ 较慢 | ⭐⭐⭐ 最高 | 颜色不均、边缘模糊 | ONNX模型 |

## 深度学习模块使用指南

### 1. 生成训练标注（SAM辅助）

利用SAM自动为图像生成初始标注，减少人工工作量：

```bash
# 使用SAM生成高质量初始标注
python scripts/generate_sam_labels.py \
    --input D:/AI/Test/ \
    --output data/labels/ \
    --model models/sam_vit_b_01ec64.pth

# 如果SAM未安装，使用阈值法生成弱标签（--fallback）
python scripts/generate_sam_labels.py \
    --input D:/AI/Test/ \
    --output data/labels/ \
    --fallback
```

### 2. 人工修正标注（GUI画笔工具）

启动GUI，加载图像后：
1. 勾选 **"启用画笔标注修正"**
2. 选择 **"前景(孔隙)"** 或 **"背景(删除)"**
3. 在右侧标注图像上拖动鼠标绘制
4. 点击 **"保存标注"** 导出mask

```bash
python gui_main.py
```

### 3. 训练U-Net模型

```bash
# 训练（使用生成的标注数据）
python training/train.py \
    --images data/images/ \
    --labels data/labels/ \
    --epochs 100 \
    --batch-size 4

# 输出
# checkpoints/best_model.pth   — PyTorch模型
# checkpoints/model.onnx       — ONNX部署模型
```

训练参数说明：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--images` | - | 原始图像目录 |
| `--labels` | - | 标注mask目录 |
| `--epochs` | 100 | 训练轮数 |
| `--batch-size` | 4 | 批次大小（8GB显存建议4） |
| `--lr` | 1e-4 | 学习率 |
| `--patch-size` | 512 | 训练patch大小 |
| `--encoder` | mobilenet_v2 | 编码器（轻量快速） |

### 4. 导出ONNX模型

```bash
python training/export_onnx.py \
    --checkpoint checkpoints/best_model.pth \
    --output models/pore_segment.onnx
```

### 5. 使用深度学习推理

导出ONNX后，系统会自动识别模型文件：

```bash
# CLI
python main.py -i image.jpg --method dl

# GUI — 在"分割方法"下拉框选择"deep_learning"
python gui_main.py
```

## 输出说明

每张图像处理后会生成以下文件：

| 文件 | 说明 |
|------|------|
| `{name}_annotated.png` | 标注图像（孔隙纯蓝，背景纯白）|
| `{name}_overlay.png` | 在原图上叠加标注 |
| `{name}_mask.png` | 二值掩膜 |
| `{name}_comparison.png` | 处理流程对比图 |
| `{name}_stats.json` | 详细统计信息 |
| `summary.json/csv` | 批量处理的汇总结果 |
| `porosity_comparison.png` | 面孔率对比柱状图 |

## 配置调参

编辑 `config.yaml`：

```yaml
# 分割方法选择: threshold | watershed | deep_learning
segmentation:
  method: "threshold"

# 深度学习配置
deep_learning:
  enabled: false
  model_path: "models/pore_segment.onnx"  # ONNX模型路径
  patch_size: 512                         # 推理patch大小
  overlap: 128                            # patch重叠像素
  confidence_threshold: 0.5               # 置信度阈值
  use_morphology: true                    # 是否应用形态学后处理
  gpu: true                               # 使用GPU推理

# 蓝色树脂的 HSV 颜色范围
threshold_segmentation:
  color_space: "HSV"
  hsv_range:
    lower: [100, 50, 50]   # H, S, V 下限
    upper: [140, 255, 255] # H, S, V 上限

# 最小孔隙面积过滤
area_calculation:
  min_pore_area: 50  # 像素
  pixel_scale: null  # μm/pixel，有比例尺时填写
```

## 项目结构

```
.
├── data/
│   ├── raw/              # 原始照片
│   ├── processed/        # 预处理结果
│   ├── labels/           # 训练标注（人工修正后）
│   └── results/          # 最终输出
├── models/               # 模型文件
│   ├── sam_vit_b_01ec64.pth      # SAM模型（辅助标注）
│   └── pore_segment.onnx         # U-Net ONNX（推理部署）
├── src/
│   ├── segmentation_base.py      # 统一分割接口
│   ├── preprocessing.py          # 图像预处理
│   ├── threshold_segment.py      # 阈值分割
│   ├── watershed_segment.py      # 分水岭算法
│   ├── dl_segment.py             # 深度学习ONNX推理
│   ├── sam_annotator.py          # SAM辅助标注
│   ├── area_calculation.py       # 面孔率计算
│   ├── visualization.py          # 结果可视化
│   ├── gui.py                    # GUI主窗口
│   └── gui_components.py         # 自定义组件
├── scripts/
│   └── generate_sam_labels.py    # 批量SAM标注生成
├── training/
│   ├── model.py                  # U-Net定义
│   ├── dataset.py                # 数据集与增强
│   ├── train.py                  # 训练脚本
│   └── export_onnx.py            # ONNX导出
├── config.yaml           # 配置文件
├── main.py               # CLI主程序入口
├── gui_main.py           # GUI入口
├── requirements.txt      # 基础依赖
├── requirements-train.txt # 训练依赖
└── README.md             # 本文件
```

## 深度学习模块架构

```
┌─────────────────────────────────────────────────────────┐
│                    分割方法选择                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  颜色阈值     │  │  分水岭算法   │  │  深度学习U-Net│   │
│  │ Threshold    │  │  Watershed   │  │  Deep Learning│   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │      BaseSegmenter       │  ← 统一接口
              └────────────┬────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
┌───┴───┐           ┌─────┴─────┐         ┌─────┴─────┐
│预处理  │           │  滑动窗口  │         │  高斯融合  │
│Pipeline│           │  512×512  │         │  消除拼接缝│
└───┬───┘           └─────┬─────┘         └─────┬─────┘
    │                      │                      │
    └──────────────────────┼──────────────────────┘
                           │
              ┌────────────┴────────────┐
              │     ONNX Runtime推理      │
              │   models/pore_segment.onnx│
              └────────────┬────────────┘
                           │
                    ┌──────┴──────┐
                    │  面孔率计算   │
                    └─────────────┘
```

## 常见问题

### Q: 选择深度学习方法时提示"模型未找到"
**A:** 需要先训练或获取ONNX模型文件：
1. 准备标注数据 → 2. 运行 `training/train.py` → 3. 导出ONNX

### Q: SAM标注工具安装失败
**A:** SAM需要单独安装：
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```
然后从[官网](https://github.com/facebookresearch/segment-anything#model-checkpoints)下载模型权重。

### Q: 8GB显存训练时OOM
**A:** 减小batch_size或使用梯度累积：
```bash
python training/train.py --batch-size 2
```

### Q: 深度学习推理比阈值慢多少？
**A:** 3600×4800图像在RTX 4060上约需3-5秒（滑动窗口推理），阈值法约0.5秒。建议批量分析时使用DL，单张实时预览用阈值。

## 后续优化路线

1. ✅ **深度学习分割** — U-Net / SAM 提升精度至 ±1%
2. **用户校正反馈** — 记录用户手动修正的结果用于增量训练
3. **比例尺自动识别** — 从图像中提取比例尺信息
4. **裂缝智能分类** — 区分收缩裂缝、构造裂缝等类型

## 许可证

MIT License
