# 面孔率识别系统 - 开发备忘录

## 当前状态（2026-06-04）

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
| 深度学习分割 | ❌ 不可用（缺少 ONNX 模型） |

### 测试验证结果

- 阈值法：N1-1(-)10X = 9.39%, N10-1(-)10X = 28.85%, N2-1(-)5X = 37.50%
- 分水岭法：N1-1(-)10X = 2.92%, N10-1(-)10X = 13.11%

### 已知问题 / 待办

- [ ] **深度学习模型训练** — data/labels/ 有 127 个标注，其中 63 个（49.6%）质量有问题（高/低比率异常），需 SAM 重新生成或人工修正
- [ ] **面积计算模块** — scikit-image `equivalent_diameter` 已弃用，需替换为 `equivalent_diameter_area`
- [ ] **config.yaml 注释** — 当前值已调整，但注释仍标注"默认"，需更新
- [ ] **不同偏光方向图像** — (-) 和 (+) 方向的 HSV 分布差异大，可能需要自适应阈值或按方向分组处理

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
python main.py -i "d:/AI/Test/N1-1-(-)10X.jpg" -o results/

# 批量分析
python main.py -i "d:/AI/Test/" -o results/ --batch

# 分水岭方法
python main.py -i "d:/AI/Test/N1-1-(-)10X.jpg" --watershed
```

### 数据目录

- 原始图像：`d:/AI/Test/`（80 张）
- 标注 mask：`data/labels/`（127 个，质量待确认）
