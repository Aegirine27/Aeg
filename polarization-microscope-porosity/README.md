# 偏光显微镜面孔率识别系统

基于计算机视觉的偏光显微镜单偏光照片孔隙与裂缝自动识别系统。

## 功能特性

- ✅ **自动识别蓝色树脂填充的孔隙和裂缝**
- ✅ **颜色阈值分割** — 边缘清晰时快速计算
- ✅ **分水岭算法** — 处理边缘模糊、结构复杂的孔隙
- ✅ **面孔率精确计算** — (孔隙+裂缝面积) / 全视域面积
- ✅ **批量处理** — 支持文件夹批量分析
- ✅ **结果可视化** — 孔隙纯蓝标注，背景纯白
- ✅ **统计导出** — JSON / CSV / 对比图表

## 算法路线

```
输入图像
  ↓
预处理（去噪 → 光照校正 → 对比度增强）
  ↓
颜色阈值分割（HSV/Lab 颜色空间）
  ↓
分水岭算法（可选，处理复杂孔隙）
  ↓
面孔率计算 + 裂缝检测
  ↓
输出标注图像 + 统计结果
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/Aegirine27/Aeg.git
cd polarization-microscope-porosity

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 单张图像处理

```bash
# 基础处理（颜色阈值分割）
python main.py -i path/to/image.jpg -o results/

# 使用分水岭算法
python main.py -i path/to/image.jpg -o results/ --watershed

# 交互式调参
python main.py -i path/to/image.jpg --interactive
```

### 批量处理

```bash
# 批量处理文件夹
python main.py -i path/to/folder/ -o results/ --batch

# 批量 + 分水岭
python main.py -i path/to/folder/ -o results/ --batch --watershed
```

### Python API

```python
from main import PorosityAnalyzer

analyzer = PorosityAnalyzer('config.yaml')

# 分析单张图像
result = analyzer.analyze('image.jpg', use_watershed=True)
print(f"面孔率: {result['stats']['porosity_percent']:.4f}%")

# 保存结果
analyzer.save_results(result, 'output/')
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
│   ├── raw/           # 原始照片
│   ├── processed/     # 预处理结果
│   └── results/       # 最终输出
├── src/
│   ├── preprocessing.py      # 图像预处理
│   ├── threshold_segment.py  # 阈值分割
│   ├── watershed_segment.py  # 分水岭算法
│   ├── area_calculation.py   # 面孔率计算
│   ├── visualization.py      # 结果可视化
│   └── batch_processor.py    # 批量处理
├── config.yaml        # 配置文件
├── main.py            # 主程序入口
├── requirements.txt   # Python 依赖
└── README.md          # 本文件
```

## 后续优化路线

1. **用户校正反馈** — 记录用户手动修正的结果
2. **深度学习优化** — 使用 U-Net / SAM 提升精度至 ±1%
3. **比例尺自动识别** — 从图像中提取比例尺信息

## 许可证

MIT License
