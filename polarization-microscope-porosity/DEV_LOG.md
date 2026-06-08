# 面孔率识别系统 - 开发备忘录

## 当前状态（2026-06-05）

### 今日操作

1. **删除旧标注并重新生成** (`2026-06-05`)
   - 删除 `data/labels/` 下 127 个旧标注（含质量问题的 mask）
   - 下载 SAM 模型 `sam_vit_b_01ec64.pth`（~375MB）到 `models/`
   - 使用 `scripts/generate_sam_labels.py` 为 `./path/to/your/images/` 中 44 张图像重新生成 SAM 初始标注
   - 生成完成：44/44 张，0 失败
   - 输出：44 个 `_mask.png` + 44 个 `_vis.jpg` 供人工审核

2. **Top 10 问题标注人工修正完成** (`2026-06-05`)
   - 通过 SAM vs 阈值法差异分析，识别出 10 张质量最差的标注
   - 使用 GUI 放大标注功能人工逐张修正：
     - `N12-1-(-)10X`, `N1-1-(-)10X`, `N16-(3)-1-(-)10X`, `N10-8-(-)10X`
     - `N10-1-(-)10X`, `N10-5-(-)10X`, `N16-(2)-1-(-)10X`, `N2-3-(-)10X`
     - `N24-(11)-1-(-)10X`, `N9-4-(-)10X`
   - 修正后的 mask 已覆盖保存到 `data/labels/`
   - 当前 44 张标注全部通过人工审核

## 历史状态（2026-06-04）

### 已完成的修复

1. **阈值分割和分水岭算法修复** (`5897f3d`)
   - 禁用 config.yaml 中的光照校正和 CLAHE 增强（会改变颜色分布导致误检）
   - 恢复默认 HSV 范围 `[100,50,50]-[140,255,255]`
   - main.py 阈值分割改用原始图像 HSV（与 GUI 预览一致）
   - 重写 watershed_segment.py（正确设置背景/前景标记，移除过度填充）
   - 修复 albumentations 2.0 兼容性（ShiftScaleRotate → Affine）

2. **GUI 启动闪退修复** (`aec9409`)
   - 修复 `_build_annotation_panel(left_frame)` 参数名错误 → 改为 `parent`

3. **方法独立 + 预览实时更新** (`1d28417`)
   - 移除未定义的 `watershed_var` 变量
   - 阈值/分水岭完全由 `method_var` 下拉框控制
   - 预览回调链恢复正常：滑块拖动 → 200ms 防抖 → 实时预览

### 当前可用功能

| 功能 | 状态 |
|------|------|
| 阈值分割（单张/批量） | ✅ 可用 |
| 分水岭分割（单张/批量） | ✅ 可用 |
| GUI 交互式分析 | ✅ 可用（启动：python gui_main.py） |
| 深度学习分割 | ✅ 可用（ONNX 模型已就绪） |

### 深度学习模型训练完成

- **模型架构**：U-Net + MobileNetV2
- **训练数据**：44张图像，~2,323 valid patches
- **最佳 Epoch**：34 / 100
- **验证 Dice**：**0.8810** ✅
- **模型文件**：`checkpoints/best_model.pth`（77MB）
- **ONNX 导出**：`checkpoints/model.onnx`（26MB）
- **训练时长**：约 1 小时（RTX 4060 Laptop）

### 测试验证结果

- 阈值法：N1-1(-)10X = 9.39%, N10-1(-)10X = 28.85%, N2-1(-)5X = 37.50%
- 分水岭法：N1-1(-)10X = 2.92%, N10-1(-)10X = 13.11%

### 已知问题 / 待办

- [x] **深度学习模型训练** — ✅ 已完成（44张标注，Dice=0.881）
- [ ] **GUI 深度学习分割集成** — ONNX 模型已就绪，需配置 GUI 加载 model.onnx
- [ ] **面积计算模块** — scikit-image `equivalent_diameter` 已弃用，需替换为 `equivalent_diameter_area`
- [ ] **config.yaml 注释** — 当前值已调整，但注释仍标注"默认"，需更新
- [ ] **不同偏光方向图像** — (-) 和 (+) 方向的 HSV 分布差异大，可能需要自适应阈值或按方向分组处理
- [ ] **后续增量训练** — 收集更多标注后，使用 `--checkpoint` 参数继续训练优化模型

### 关键文件变更

```
config.yaml              - 禁用光照校正/CLAHE，调整HSV范围和形态学参数
main.py                  - 阈值分割使用原始图像HSV
gui.py                   - 移除watershed_var，方法由method_var控制
src/watershed_segment.py - 重写分水岭流程
src/threshold_segment.py - 未修改（核心逻辑正确）
training/dataset.py      - albumentations 2.0 兼容性修复
```

### 快速启动命令

```bash
cd "d:\AI\Aeg for github\polarization-microscope-porosity"

# GUI 交互式分析
python gui_main.py

# 单张分析
python main.py -i "./path/to/your/images/N1-1-(-)10X.jpg" -o results/

# 批量分析
python main.py -i "./path/to/your/images/" -o results/ --batch

# 分水岭方法
python main.py -i "./path/to/your/images/N1-1-(-)10X.jpg" --watershed
```

### 数据目录

- 原始图像：`./path/to/your/images/`（80 张）
- 标注 mask：`data/labels/`（127 个，质量待确认）
