"""
GUI 主窗口模块

整合所有组件，实现完整的交互逻辑：
    - 文件/文件夹选择
    - 参数调整（含Photoshop式取色分割）
    - 图像分析（后台线程）
    - 结果展示
    - 结果保存
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import json
import csv
import time
import types
import cv2
import numpy as np
import traceback
import sys

# 设置日志文件
log_file = Path('debug.log')
def log(msg):
    """写入日志"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")

log("="*50)
log("程序启动")

from .gui_components import (
    COLORS, ImageViewer, ResultCard, ParameterSlider, RangeSlider,
    CustomButton, StatusBar
)
from .utils import load_config, load_image, save_image
from .preprocessing import ImagePreprocessor
from .threshold_segment import ThresholdSegmenter
from .watershed_segment import WatershedSegmenter
from .area_calculation import PorosityCalculator
from .visualization import ResultVisualizer


class PorosityGUI:
    """面孔率识别系统 GUI 主窗口"""

    def __init__(self, config_path="config.yaml"):
        # 加载配置
        self.config = load_config(config_path)
        self.config_path = config_path

        # 初始化分析器
        self.analyzer = None
        self._init_analyzer()

        # 当前结果缓存
        self.current_result = None
        self.batch_results = []
        self.selected_path = None

        # 取色相关状态
        self.original_image = None       # 原始BGR图像
        self.original_image_hsv = None   # HSV图像
        self.pick_mode = False           # 是否处于取色模式
        self.sampled_colors = []         # 用户取样的颜色列表 [(h,s,v), ...]
        self.tolerance_h = 10            # H容差
        self.tolerance_s = 40            # S容差
        self.tolerance_v = 40            # V容差

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("偏光显微镜面孔率识别系统 v1.0")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg=COLORS['bg_main'])

        # 尝试设置窗口图标（可选）
        try:
            self.root.iconbitmap('')
        except:
            pass

        self._build_ui()

    def _init_analyzer(self):
        """初始化分析器"""
        try:
            self.analyzer = types.SimpleNamespace(
                config=self.config,
                preprocessor=ImagePreprocessor(self.config),
                thresh_segmenter=ThresholdSegmenter(self.config),
                watershed_segmenter=WatershedSegmenter(self.config),
                calculator=PorosityCalculator(self.config),
                visualizer=ResultVisualizer(self.config),
            )
        except Exception as e:
            messagebox.showerror("初始化错误", f"分析器初始化失败:\n{e}")

    def _build_ui(self):
        """构建 UI 界面"""
        # ============ 顶部标题栏 ============
        title_frame = tk.Frame(self.root, bg=COLORS['primary'], height=50)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)

        tk.Label(title_frame, text="偏光显微镜面孔率识别系统",
                 bg=COLORS['primary'], fg='white',
                 font=('Microsoft YaHei', 14, 'bold')).pack(side='left', padx=15, pady=8)

        tk.Label(title_frame, text="v1.0",
                 bg=COLORS['primary'], fg='#BDC3C7',
                 font=('Microsoft YaHei', 10)).pack(side='left', pady=8)

        # ============ 主体区域（左右分栏） ============
        main_frame = tk.Frame(self.root, bg=COLORS['bg_main'])
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 左侧面板（控制区）
        self._build_left_panel(main_frame)

        # 右侧面板（预览+结果区）
        self._build_right_panel(main_frame)

        # ============ 底部状态栏 ============
        self.status_bar = StatusBar(self.root)
        self.status_bar.pack(fill='x', side='bottom')

    def _build_left_panel(self, parent):
        """构建左侧面板（带滚动条）"""
        # 外层容器固定宽度
        left_container = tk.Frame(parent, bg=COLORS['bg_panel'], width=300)
        left_container.pack(side='left', fill='y', padx=(0, 10))
        left_container.pack_propagate(False)

        # Canvas + Scrollbar 实现滚动
        canvas = tk.Canvas(left_container, bg=COLORS['bg_panel'], highlightthickness=0)
        scrollbar = tk.Scrollbar(left_container, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        # 内部Frame放置所有控件
        left_frame = tk.Frame(canvas, bg=COLORS['bg_panel'], width=280)
        canvas.create_window((0, 0), window=left_frame, anchor='nw', width=280)

        # 绑定鼠标滚轮
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _on_mousewheel)

        # 内容变化时更新滚动区域
        def _configure_canvas(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
        left_frame.bind('<Configure>', _configure_canvas)

        self.left_canvas = canvas  # 保存引用

        # --- 输入选择区 ---
        input_frame = tk.LabelFrame(left_frame, text=" 输入选择 ",
                                     bg=COLORS['bg_panel'], fg=COLORS['primary'],
                                     font=('Microsoft YaHei', 10, 'bold'),
                                     padx=10, pady=10)
        input_frame.pack(fill='x', padx=10, pady=(10, 5))

        # 单选按钮
        self.mode_var = tk.StringVar(value='single')
        tk.Radiobutton(input_frame, text="单张图像", variable=self.mode_var,
                       value='single', bg=COLORS['bg_panel'], fg=COLORS['text'],
                       font=('Microsoft YaHei', 9), command=self._on_mode_change).pack(anchor='w')
        tk.Radiobutton(input_frame, text="批量文件夹", variable=self.mode_var,
                       value='batch', bg=COLORS['bg_panel'], fg=COLORS['text'],
                       font=('Microsoft YaHei', 9), command=self._on_mode_change).pack(anchor='w')

        # 路径显示
        self.path_var = tk.StringVar(value="未选择")
        tk.Label(input_frame, textvariable=self.path_var, bg=COLORS['bg_panel'],
                 fg=COLORS['text_secondary'], font=('Microsoft YaHei', 9),
                 wraplength=250).pack(fill='x', pady=(5, 0))

        # 选择按钮
        btn_frame = tk.Frame(input_frame, bg=COLORS['bg_panel'])
        btn_frame.pack(fill='x', pady=(5, 0))

        self.select_btn = CustomButton(btn_frame, text="选择文件", style='secondary',
                                        command=self._select_file)
        self.select_btn.pack(fill='x', pady=(0, 3))

        # --- 取色校正面板（新增） ---
        self._build_color_pick_panel(left_frame)

        # --- 参数设置区 ---
        param_frame = tk.LabelFrame(left_frame, text=" 参数设置 ",
                                     bg=COLORS['bg_panel'], fg=COLORS['primary'],
                                     font=('Microsoft YaHei', 10, 'bold'),
                                     padx=10, pady=10)
        param_frame.pack(fill='x', padx=10, pady=5)

        # 分割方法选择
        method_frame = tk.Frame(param_frame, bg=COLORS['bg_panel'])
        method_frame.pack(fill='x', pady=(0, 5))

        tk.Label(method_frame, text="分割方法:", bg=COLORS['bg_panel'],
                 fg=COLORS['text'], font=('Microsoft YaHei', 9)).pack(side='left')

        self.method_var = tk.StringVar(value='threshold')
        self.method_combo = ttk.Combobox(method_frame, textvariable=self.method_var,
                                          values=['threshold', 'watershed', 'deep_learning'],
                                          width=15, state='readonly')
        self.method_combo.pack(side='left', padx=(5, 0))
        self.method_combo.bind('<<ComboboxSelected>>', self._on_method_changed)

        # DL模型状态指示
        self.dl_status_label = tk.Label(param_frame, text="DL模型: 未使用",
                                         bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
                                         font=('Microsoft YaHei', 8))
        self.dl_status_label.pack(anchor='w', pady=(0, 3))

        # HSV 阈值设置面板（可折叠，默认折叠）
        self.thresh_collapsed = True
        thresh_frame = tk.LabelFrame(param_frame, text="",
                                      bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
                                      font=('Microsoft YaHei', 9), padx=5, pady=5)
        thresh_frame.pack(fill='x', pady=(0, 5))

        # 标题栏：标题 + 折叠按钮（始终可见）
        title_bar = tk.Frame(thresh_frame, bg=COLORS['bg_panel'])
        title_bar.pack(fill='x', pady=(0, 3))

        tk.Label(title_bar, text="颜色阈值范围 (HSV)", bg=COLORS['bg_panel'],
                 fg=COLORS['primary'], font=('Microsoft YaHei', 9, 'bold')).pack(side='left')

        self.toggle_thresh_btn = tk.Button(title_bar, text="▼ 展开", bg=COLORS['bg_panel'],
                                            fg=COLORS['accent'], font=('Microsoft YaHei', 8),
                                            relief='flat', cursor='hand2',
                                            command=self._toggle_thresh_panel)
        self.toggle_thresh_btn.pack(side='right')

        # 预设选择（始终可见）
        preset_frame = tk.Frame(thresh_frame, bg=COLORS['bg_panel'])
        preset_frame.pack(fill='x', pady=(0, 3))

        tk.Label(preset_frame, text="快速选择:", bg=COLORS['bg_panel'],
                 fg=COLORS['text'], font=('Microsoft YaHei', 9)).pack(side='left')

        self.preset_var = tk.StringVar(value='蓝色树脂')
        self.preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                          values=['蓝色树脂', '青色', '紫色', '绿色', '自定义'],
                                          width=10, state='readonly')
        self.preset_combo.pack(side='left', padx=(5, 0))
        self.preset_combo.bind('<<ComboboxSelected>>', self._on_preset_changed)

        # 可折叠的内容区
        self.thresh_content = tk.Frame(thresh_frame, bg=COLORS['bg_panel'])
        # 默认折叠，不 pack

        # 从配置读取默认值
        lower = self.config.get('threshold_segmentation', {}).get('hsv_range', {}).get('lower', [100, 50, 50])
        upper = self.config.get('threshold_segmentation', {}).get('hsv_range', {}).get('upper', [140, 255, 255])

        # H 色相范围
        self.h_range = RangeSlider(self.thresh_content, label="H · 色相范围", from_=0, to=180,
                                    lower_default=lower[0], upper_default=upper[0],
                                    command=lambda lo, hi: self._on_param_changed())
        self.h_range.pack(fill='x', pady=3)

        # S 饱和度范围
        self.s_range = RangeSlider(self.thresh_content, label="S · 饱和度范围", from_=0, to=255,
                                    lower_default=lower[1], upper_default=upper[1],
                                    command=lambda lo, hi: self._on_param_changed())
        self.s_range.pack(fill='x', pady=3)

        # V 亮度范围
        self.v_range = RangeSlider(self.thresh_content, label="V · 亮度范围", from_=0, to=255,
                                    lower_default=lower[2], upper_default=upper[2],
                                    command=lambda lo, hi: self._on_param_changed())
        self.v_range.pack(fill='x', pady=3)

        # 操作提示
        tk.Label(self.thresh_content, text="拖动两端白色圆点调节范围，中间蓝色为选中区域",
                 bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
                 font=('Microsoft YaHei', 8)).pack(anchor='w', pady=(3, 0))

        # 最小孔隙面积
        min_area = self.config.get('area_calculation', {}).get('min_pore_area', 50)
        self.min_area_slider = ParameterSlider(param_frame, label="最小孔隙面积", from_=0, to=500, default=min_area,
                                                command=lambda v: self._on_param_changed())
        self.min_area_slider.pack(fill='x', pady=(5, 0))

        # --- 操作按钮区 ---
        action_frame = tk.Frame(left_frame, bg=COLORS['bg_panel'])
        action_frame.pack(fill='x', padx=10, pady=(10, 5))

        self.analyze_btn = CustomButton(action_frame, text="开始分析", style='primary',
                                         command=self._start_analysis)
        self.analyze_btn.pack(fill='x', pady=(0, 5))

        self.preview_btn = CustomButton(action_frame, text="预览分割", style='secondary',
                                         command=self._preview_threshold)
        self.preview_btn.pack(fill='x', pady=(0, 5))

        self.reset_btn = CustomButton(action_frame, text="重置参数", style='secondary',
                                       command=self._reset_params)
        self.reset_btn.pack(fill='x', pady=(0, 5))

        self.save_btn = CustomButton(action_frame, text="保存结果", style='success',
                                      command=self._save_results, state='disabled')
        self.save_btn.pack(fill='x', pady=(0, 5))

        self.export_btn = CustomButton(action_frame, text="导出CSV", style='secondary',
                                        command=self._export_csv, state='disabled')
        self.export_btn.pack(fill='x')

    def _toggle_thresh_panel(self):
        """切换 HSV 阈值面板的折叠/展开状态"""
        if self.thresh_collapsed:
            # 展开
            self.thresh_content.pack(fill='x', pady=(5, 0))
            self.toggle_thresh_btn.config(text="▲ 折叠")
            self.thresh_collapsed = False
        else:
            # 折叠
            self.thresh_content.pack_forget()
            self.toggle_thresh_btn.config(text="▼ 展开")
            self.thresh_collapsed = True

    def _on_method_changed(self, event=None):
        """处理分割方法切换"""
        method = self.method_var.get()

        if method == 'deep_learning':
            # 检查模型文件是否存在
            model_path = Path(self.config.get('deep_learning', {}).get('model_path', 'models/pore_segment.onnx'))
            if model_path.exists():
                self.dl_status_label.config(text=f"DL模型: 就绪 ({model_path.name})", fg=COLORS['success'])
            else:
                self.dl_status_label.config(
                    text=f"DL模型: 未找到 ({model_path})", fg=COLORS['danger']
                )
                # 自动切回阈值方法
                self.method_var.set('threshold')
                messagebox.showwarning(
                    "模型未找到",
                    f"深度学习模型文件不存在:\n{model_path}\n\n"
                    f"请先训练并导出ONNX模型，或从其他来源获取。\n"
                    f"已自动切换回阈值分割方法。"
                )
                return
        else:
            self.dl_status_label.config(text="DL模型: 未使用", fg=COLORS['text_secondary'])

        # 方法切换后自动预览
        self._preview_threshold()

    def _get_segmenter_and_mask(self):
        """根据当前选择的方法获取分割器和mask"""
        method = self.method_var.get()

        if method == 'deep_learning':
            try:
                from .dl_segment import DLSegmenter
                segmenter = DLSegmenter(self.config)
                return segmenter, 'deep_learning'
            except Exception as e:
                log(f"DL模型加载失败: {e}")
                self.status_bar.set_status(f"DL模型错误: {str(e)[:50]}", COLORS['danger'])
                # 回退到阈值方法
                return self.analyzer.thresh_segmenter, 'threshold'
        elif method == 'watershed':
            return self.analyzer.watershed_segmenter, 'watershed'
        else:
            return self.analyzer.thresh_segmenter, 'threshold'

    def _on_preset_changed(self, event=None):
        """处理 HSV 预设选择变更"""
        preset = self.preset_var.get()

        presets = {
            '蓝色树脂': {'lower': [100, 50, 50], 'upper': [140, 255, 255]},
            '青色': {'lower': [80, 50, 50], 'upper': [100, 255, 255]},
            '紫色': {'lower': [140, 50, 50], 'upper': [160, 255, 255]},
            '绿色': {'lower': [40, 50, 50], 'upper': [80, 255, 255]},
            '自定义': None,
        }

        if preset == '自定义':
            return  # 不做任何更改，保持当前值

        cfg = presets.get(preset)
        if cfg:
            lower = cfg['lower']
            upper = cfg['upper']

            self.h_range.set(lower[0], upper[0])
            self.s_range.set(lower[1], upper[1])
            self.v_range.set(lower[2], upper[2])

            self.status_bar.set_status(
                f"已应用预设 [{preset}]: H{lower[0]}-{upper[0]} S{lower[1]}-{upper[1]} V{lower[2]}-{upper[2]}",
                COLORS['success']
            )
            # RangeSlider.set() 已触发 _on_param_changed()，200ms 后自动预览

    def _build_color_pick_panel(self, parent):
        """构建取色校正面板（简化版：点击即应用，无容差设置）"""
        pick_frame = tk.LabelFrame(parent, text=" 取色校正 ",
                                    bg=COLORS['bg_panel'], fg=COLORS['primary'],
                                    font=('Microsoft YaHei', 10, 'bold'),
                                    padx=10, pady=10)
        pick_frame.pack(fill='x', padx=10, pady=5)

        # 步骤1：自动提取蓝色
        tk.Label(pick_frame, text="Step 1: 自动提取蓝色", bg=COLORS['bg_panel'],
                 fg=COLORS['text'], font=('Microsoft YaHei', 9, 'bold')).pack(anchor='w')
        self.auto_blue_btn = CustomButton(pick_frame, text="自动提取蓝色", style='secondary',
                                           command=self._auto_extract_blue)
        self.auto_blue_btn.pack(fill='x', pady=(2, 5))

        # 步骤2：取色笔
        tk.Label(pick_frame, text="Step 2: 取色笔精确校正", bg=COLORS['bg_panel'],
                 fg=COLORS['text'], font=('Microsoft YaHei', 9, 'bold')).pack(anchor='w')
        self.pick_btn = CustomButton(pick_frame, text="取色提示", style='secondary',
                                      command=self._enter_pick_mode)
        self.pick_btn.pack(fill='x', pady=(2, 2))

        # 取样状态
        self.sample_label = tk.Label(pick_frame, text="已取样: 0 个", bg=COLORS['bg_panel'],
                                      fg=COLORS['text_secondary'], font=('Microsoft YaHei', 9))
        self.sample_label.pack(anchor='w')

        # 取样操作按钮区
        sample_btn_frame = tk.Frame(pick_frame, bg=COLORS['bg_panel'])
        sample_btn_frame.pack(fill='x', pady=(5, 0))

        self.undo_sample_btn = CustomButton(sample_btn_frame, text="↩ 撤销上一取样", style='secondary',
                                             command=self._undo_last_sample)
        self.undo_sample_btn.pack(fill='x', pady=(0, 3))

        self.clear_sample_btn = CustomButton(sample_btn_frame, text="清空取样", style='danger',
                                              command=self._clear_samples)
        self.clear_sample_btn.pack(fill='x')

        # --- 标注修正面板（深度学习训练用） ---
        self._build_annotation_panel(parent)

    def _build_annotation_panel(self, parent):
        """构建标注修正面板"""
        anno_frame = tk.LabelFrame(parent, text=" 标注修正 (训练用) ",
                                    bg=COLORS['bg_panel'], fg=COLORS['primary'],
                                    font=('Microsoft YaHei', 10, 'bold'),
                                    padx=10, pady=10)
        anno_frame.pack(fill='x', padx=10, pady=5)

        tk.Label(anno_frame, text="生成高质量训练标注",
                 bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
                 font=('Microsoft YaHei', 9)).pack(anchor='w')

        # 标注模式开关
        self.annotate_mode_var = tk.BooleanVar(value=False)
        self.annotate_check = tk.Checkbutton(anno_frame, text="启用画笔标注修正",
                                              variable=self.annotate_mode_var,
                                              bg=COLORS['bg_panel'], fg=COLORS['text'],
                                              font=('Microsoft YaHei', 9),
                                              command=self._toggle_annotation_mode)
        self.annotate_check.pack(anchor='w', pady=(3, 0))

        # 画笔设置
        brush_frame = tk.Frame(anno_frame, bg=COLORS['bg_panel'])
        brush_frame.pack(fill='x', pady=(5, 0))

        tk.Label(brush_frame, text="画笔:", bg=COLORS['bg_panel'],
                 fg=COLORS['text'], font=('Microsoft YaHei', 9)).pack(side='left')
        self.brush_size_var = tk.IntVar(value=10)
        tk.Spinbox(brush_frame, from_=3, to=50, textvariable=self.brush_size_var,
                   width=5, font=('Microsoft YaHei', 9)).pack(side='left', padx=(5, 0))

        # 前景/背景选择
        self.annotation_color_var = tk.StringVar(value='foreground')
        tk.Radiobutton(anno_frame, text="前景(孔隙)", variable=self.annotation_color_var,
                       value='foreground', bg=COLORS['bg_panel'], fg=COLORS['text'],
                       font=('Microsoft YaHei', 9),
                       command=self._set_annotation_color).pack(anchor='w')
        tk.Radiobutton(anno_frame, text="背景(删除)", variable=self.annotation_color_var,
                       value='background', bg=COLORS['bg_panel'], fg=COLORS['text'],
                       font=('Microsoft YaHei', 9),
                       command=self._set_annotation_color).pack(anchor='w')

        # 操作按钮
        anno_btn_frame = tk.Frame(anno_frame, bg=COLORS['bg_panel'])
        anno_btn_frame.pack(fill='x', pady=(5, 0))

        self.clear_anno_btn = CustomButton(anno_btn_frame, text="清空标注", style='danger',
                                            command=self._clear_annotation)
        self.clear_anno_btn.pack(fill='x', pady=(0, 3))

        self.save_anno_btn = CustomButton(anno_btn_frame, text="保存标注", style='success',
                                           command=self._save_annotation)
        self.save_anno_btn.pack(fill='x')

        # 初始禁用
        self.annotate_check.config(state='disabled')
        self.save_anno_btn.config(state='disabled')

    def _toggle_annotation_mode(self):
        """切换标注模式"""
        enabled = self.annotate_mode_var.get()
        color = 255 if self.annotation_color_var.get() == 'foreground' else 0
        size = self.brush_size_var.get()
        self.annotated_viewer.set_annotate_mode(enabled, brush_size=size, annotation_color=color)
        self.status_bar.set_status("画笔标注" + ("开启" if enabled else "关闭"), COLORS['warning'] if enabled else COLORS['success'])

    def _set_annotation_color(self):
        """设置标注颜色"""
        color = 255 if self.annotation_color_var.get() == 'foreground' else 0
        size = self.brush_size_var.get()
        if self.annotate_mode_var.get():
            self.annotated_viewer.set_annotate_mode(True, brush_size=size, annotation_color=color)

    def _clear_annotation(self):
        """清空标注"""
        self.annotated_viewer.clear_annotation()
        self.status_bar.set_status("标注已清空", COLORS['text_secondary'])

    def _save_annotation(self):
        """保存标注为训练数据"""
        mask = self.annotated_viewer.get_annotation_mask()
        if mask is None:
            messagebox.showwarning("提示", "没有可保存的标注")
            return
        output_dir = Path('data/labels')
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / f"annotated_mask_{len(list(output_dir.glob('*.png')))}.png"
        cv2.imwrite(str(mask_path), mask)
        self.status_bar.set_status(f"标注已保存: {mask_path.name}", COLORS['success'])
        messagebox.showinfo("完成", f"标注已保存:\n{mask_path}")

    def _build_right_panel(self, parent):
        """构建右侧面板"""
        right_frame = tk.Frame(parent, bg=COLORS['bg_main'])
        right_frame.pack(side='left', fill='both', expand=True)

        # --- 图像预览区 ---
        preview_frame = tk.LabelFrame(right_frame, text=" 图像预览 ",
                                       bg=COLORS['bg_main'], fg=COLORS['primary'],
                                       font=('Microsoft YaHei', 10, 'bold'),
                                       padx=5, pady=5)
        preview_frame.pack(fill='both', expand=True, pady=(0, 5))

        # 左右图像并排
        img_container = tk.Frame(preview_frame, bg=COLORS['bg_main'])
        img_container.pack(fill='both', expand=True)
        img_container.grid_columnconfigure(0, weight=1)
        img_container.grid_columnconfigure(1, weight=1)
        img_container.grid_rowconfigure(0, weight=1)

        # 原图支持点击取色
        self.original_viewer = ImageViewer(img_container, title="原始图像(点击取色)",
                                            width=420, height=420,
                                            click_callback=self._on_image_pick)
        self.original_viewer.grid(row=0, column=0, padx=(0, 5), sticky='nsew')

        self.annotated_viewer = ImageViewer(img_container, title="标注图像（孔隙=蓝色，背景=白色）",
                                             width=420, height=420)
        self.annotated_viewer.grid(row=0, column=1, padx=(5, 0), sticky='nsew')

        # --- 结果展示区 ---
        result_frame = tk.LabelFrame(right_frame, text=" 分析结果 ",
                                      bg=COLORS['bg_panel'], fg=COLORS['primary'],
                                      font=('Microsoft YaHei', 10, 'bold'),
                                      padx=10, pady=10)
        result_frame.pack(fill='x', pady=(5, 0))

        # 面孔率大数字
        self.porosity_card = ResultCard(result_frame, label="面孔率", value="--",
                                         unit="%", font_size=28, value_color=COLORS['accent'])
        self.porosity_card.pack(side='left', padx=(0, 20))

        # 其他指标
        stats_container = tk.Frame(result_frame, bg=COLORS['bg_panel'])
        stats_container.pack(side='left', fill='both', expand=True)

        self.count_card = ResultCard(stats_container, label="孔隙数量", value="--", unit="个")
        self.count_card.pack(side='left', padx=10)

        self.avg_area_card = ResultCard(stats_container, label="平均面积", value="--", unit="px")
        self.avg_area_card.pack(side='left', padx=10)

        self.crack_card = ResultCard(stats_container, label="裂缝数量", value="--", unit="条")
        self.crack_card.pack(side='left', padx=10)

        self.time_card = ResultCard(stats_container, label="处理时间", value="--", unit="s")
        self.time_card.pack(side='left', padx=10)

    # ============ 取色功能 ============

    def _auto_extract_blue(self):
        """自动提取蓝色：基于图像中蓝色像素的统计自动设置阈值"""
        if self.original_image_hsv is None:
            messagebox.showwarning("提示", "请先选择图像！")
            return

        hsv = self.original_image_hsv
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

        # 粗略筛选蓝色区域
        blue_mask = (h > 90) & (h < 140) & (s > 50) & (v > 40)
        blue_pixels = hsv[blue_mask]

        if len(blue_pixels) == 0:
            messagebox.showwarning("提示", "未检测到明显的蓝色区域！")
            return

        # 计算蓝色区域的统计值
        h_mean = int(np.mean(blue_pixels[:, 0]))
        s_mean = int(np.mean(blue_pixels[:, 1]))
        v_mean = int(np.mean(blue_pixels[:, 2]))

        h_std = int(np.std(blue_pixels[:, 0]))
        s_std = int(np.std(blue_pixels[:, 1]))
        v_std = int(np.std(blue_pixels[:, 2]))

        # 设置阈值为 均值 ± 2*标准差
        h_lower = max(0, h_mean - max(10, h_std * 2))
        h_upper = min(180, h_mean + max(10, h_std * 2))
        s_lower = max(0, s_mean - max(30, s_std))
        s_upper = min(255, s_mean + max(30, s_std))
        v_lower = max(0, v_mean - max(30, v_std))
        v_upper = min(255, v_mean + max(30, v_std))

        self.h_range.set(h_lower, h_upper)
        self.s_range.set(s_lower, s_upper)
        self.v_range.set(v_lower, v_upper)

        self.status_bar.set_status(f"自动提取: H{h_mean} S{s_mean} V{v_mean}", COLORS['success'])

        # 自动预览
        self._preview_threshold()

    def _enter_pick_mode(self):
        """取色提示"""
        if self.original_image is None:
            messagebox.showwarning("提示", "请先选择图像！")
            return

        self.status_bar.set_status("请点击原图上的蓝色孔隙区域，任意点数均可", COLORS['warning'])
        messagebox.showinfo("取色提示", "直接点击原图上的蓝色孔隙区域即可\n任意点数均可，每次点击自动应用并预览")

    def _on_image_pick(self, img_x, img_y):
        """处理图像点击取色——点击直接以固定窗口设置阈值并预览"""
        if self.original_image_hsv is None:
            return

        # 获取点击位置的HSV值（取3x3区域平均，减少单点噪声）
        h, w = self.original_image_hsv.shape[:2]
        x1, y1 = max(0, img_x - 1), max(0, img_y - 1)
        x2, y2 = min(w, img_x + 2), min(h, img_y + 2)

        roi = self.original_image_hsv[y1:y2, x1:x2]
        sample_h = int(np.mean(roi[:, :, 0]))
        sample_s = int(np.mean(roi[:, :, 1]))
        sample_v = int(np.mean(roi[:, :, 2]))

        self.sampled_colors.append((sample_h, sample_s, sample_v))
        self.sample_label.config(text=f"已取样: {len(self.sampled_colors)} 个")

        # 固定窗口大小：以取样颜色为中心，向外扩展固定范围
        H_WINDOW, S_WINDOW, V_WINDOW = 15, 50, 50

        # 基于所有取样点的范围，加上固定窗口
        h_vals = [c[0] for c in self.sampled_colors]
        s_vals = [c[1] for c in self.sampled_colors]
        v_vals = [c[2] for c in self.sampled_colors]

        h_min, h_max = min(h_vals), max(h_vals)
        s_min, s_max = min(s_vals), max(s_vals)
        v_min, v_max = min(v_vals), max(v_vals)

        self.h_range.set(max(0, h_min - H_WINDOW), min(180, h_max + H_WINDOW))
        self.s_range.set(max(0, s_min - S_WINDOW), min(255, s_max + S_WINDOW))
        self.v_range.set(max(0, v_min - V_WINDOW), min(255, v_max + V_WINDOW))

        # 状态栏显示
        self.status_bar.set_status(
            f"取样#{len(self.sampled_colors)}: H={sample_h} S={sample_s} V={sample_v}（已自动应用）",
            COLORS['success']
        )

        # 自动预览
        self._preview_threshold()

    def _undo_last_sample(self):
        """撤销上一个取样点，并自动重新计算阈值"""
        if not self.sampled_colors:
            self.status_bar.set_status("没有可撤销的取样点", COLORS['warning'])
            return

        # 删除最后一个取样点
        removed = self.sampled_colors.pop()
        count = len(self.sampled_colors)
        self.sample_label.config(text=f"已取样: {count} 个")

        # 固定窗口大小（与 _on_image_pick 保持一致）
        H_WINDOW, S_WINDOW, V_WINDOW = 15, 50, 50

        if count > 0:
            # 还有剩余取样点：基于剩余点重新计算阈值
            h_vals = [c[0] for c in self.sampled_colors]
            s_vals = [c[1] for c in self.sampled_colors]
            v_vals = [c[2] for c in self.sampled_colors]

            h_min, h_max = min(h_vals), max(h_vals)
            s_min, s_max = min(s_vals), max(s_vals)
            v_min, v_max = min(v_vals), max(v_vals)

            self.h_range.set(max(0, h_min - H_WINDOW), min(180, h_max + H_WINDOW))
            self.s_range.set(max(0, s_min - S_WINDOW), min(255, s_max + S_WINDOW))
            self.v_range.set(max(0, v_min - V_WINDOW), min(255, v_max + V_WINDOW))

            self.status_bar.set_status(
                f"已撤销取样: H={removed[0]} S={removed[1]} V={removed[2]}，剩余{count}个",
                COLORS['success']
            )
        else:
            # 没有取样点了：重置阈值为默认值
            default_config = load_config(self.config_path)
            lower = default_config.get('threshold_segmentation', {}).get('hsv_range', {}).get('lower', [100, 50, 50])
            upper = default_config.get('threshold_segmentation', {}).get('hsv_range', {}).get('upper', [140, 255, 255])

            self.h_range.set(lower[0], upper[0])
            self.s_range.set(lower[1], upper[1])
            self.v_range.set(lower[2], upper[2])

            self.status_bar.set_status(
                f"已撤销最后一个取样，阈值已重置为默认值",
                COLORS['text_secondary']
            )

        # 自动预览更新
        self._preview_threshold()

    def _clear_samples(self):
        """清空所有取样"""
        self.sampled_colors = []
        self.sample_label.config(text="已取样: 0 个")
        self.pick_mode = False
        self.status_bar.set_status("取样已清空", COLORS['text_secondary'])

    def _apply_watershed(self, mask, image):
        """应用分水岭算法分割粘连孔隙"""
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, markers = cv2.threshold(distance, 0.5 * distance.max(), 255, cv2.THRESH_BINARY)
        markers = np.uint8(markers)
        _, markers = cv2.connectedComponents(markers)
        markers = markers + 1
        markers[mask == 0] = 0
        markers = markers.astype(np.int32)
        cv2.watershed(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), markers)
        result = np.zeros_like(mask)
        result[markers > 1] = 255
        return result

    def _preview_threshold(self):
        """实时预览当前阈值的分割效果（使用原始HSV，与用户看到的图像一致）"""
        if self.original_image is None or self.original_image_hsv is None:
            return

        try:
            # 先更新配置（确保阈值是最新的）
            self._update_config_from_ui()

            # 使用原始HSV进行预览（与用户看到的图像一致）
            hsv = self.original_image_hsv

            # 使用当前UI的阈值参数进行分割
            h_l, h_u = self.h_range.get()
            s_l, s_u = self.s_range.get()
            v_l, v_u = self.v_range.get()

            lower = np.array([h_l, s_l, v_l])
            upper = np.array([h_u, s_u, v_u])

            # 执行阈值分割
            mask = cv2.inRange(hsv, lower, upper)

            # 形态学操作（与配置一致）
            morph_cfg = self.config.get('threshold_segmentation', {}).get('morphological_operations', {})
            open_k = np.ones(tuple(morph_cfg.get('open_kernel', [3, 3])), np.uint8)
            close_k = np.ones(tuple(morph_cfg.get('close_kernel', [5, 5])), np.uint8)
            iterations = morph_cfg.get('iterations', 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=iterations)

            # 分水岭算法（预览与分析保持一致）
            if self.watershed_var.get():
                mask = self._apply_watershed(mask, self.original_image)

            # 应用最小孔隙面积过滤
            min_area = self.min_area_slider.get()
            if min_area > 0:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                filtered_mask = np.zeros_like(mask)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= min_area:
                        filtered_mask[labels == i] = 255
                mask = filtered_mask

            # 生成标注图像
            annotated = np.ones_like(self.original_image) * 255
            annotated[mask > 0] = [255, 0, 0]  # BGR蓝色

            # 计算并显示预览面孔率
            pore_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            porosity = (pore_pixels / total_pixels) * 100

            # 显示预览
            self.annotated_viewer.show_image(annotated)

            self.status_bar.set_status(
                f"预览: 面孔率={porosity:.2f}% 像素={pore_pixels}",
                COLORS['accent']
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_bar.set_status(f"预览错误: {str(e)[:50]}", COLORS['danger'])

    # ============ 事件处理 ============

    def _on_mode_change(self):
        """切换单张/批量模式"""
        if self.mode_var.get() == 'single':
            self.select_btn.config(text="选择文件")
        else:
            self.select_btn.config(text="选择文件夹")
        self.selected_path = None
        self.path_var.set("未选择")

    def _select_file(self):
        """选择文件或文件夹"""
        if self.mode_var.get() == 'single':
            # 简化文件对话框，避免某些tkinter版本的兼容性问题
            try:
                path = filedialog.askopenfilename(
                    title="选择显微镜照片",
                    filetypes=[("所有文件", "*.*")]
                )
            except Exception as e:
                messagebox.showerror("对话框错误", f"文件对话框出错:\n{e}\n\n尝试使用简化对话框...")
                try:
                    path = filedialog.askopenfilename()
                except Exception as e2:
                    messagebox.showerror("对话框错误", f"简化对话框也出错:\n{e2}")
                    return
        else:
            path = filedialog.askdirectory(title="选择照片文件夹")

        if path:
            self.selected_path = path
            self.path_var.set(f"已选择: {path}")
            self.status_bar.set_file(Path(path).name)

            # 如果是单张，加载并显示原图
            if self.mode_var.get() == 'single':
                log(f"尝试加载图像: {path}")
                try:
                    self.original_image = load_image(path)
                    self.original_image_hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
                    self.original_viewer.show_image(self.original_image)
                    self.annotated_viewer.clear()
                    self._clear_results()
                    self._clear_samples()
                    # 启用标注修正功能
                    self.annotate_check.config(state='normal')
                    self.save_anno_btn.config(state='normal')
                    self.annotated_viewer.load_annotation_mask(
                        np.zeros(self.original_image.shape[:2], dtype=np.uint8)
                    )
                    self.status_bar.set_status("图像已加载，可调整阈值或取色", COLORS['success'])
                    log(f"图像加载成功: {self.original_image.shape}")
                except Exception as e:
                    error_msg = f"无法加载图像:\n{e}"
                    log(f"图像加载失败: {error_msg}")
                    log(traceback.format_exc())
                    messagebox.showerror("加载错误", error_msg)
                    self.original_image = None
                    self.original_image_hsv = None

    def _start_analysis(self):
        """开始分析"""
        if not self.selected_path:
            messagebox.showwarning("提示", "请先选择文件或文件夹！")
            return

        # 禁用分析按钮
        self.analyze_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        self.export_btn.config(state='disabled')

        # 更新配置
        self._update_config_from_ui()

        # 在后台线程执行分析
        self.status_bar.set_status("分析中...", COLORS['warning'])
        thread = threading.Thread(target=self._analyze_worker, daemon=True)
        thread.start()

    def _analyze_worker(self):
        """后台分析线程"""
        try:
            if self.mode_var.get() == 'single':
                self._analyze_single()
            else:
                self._analyze_batch()
        except Exception as e:
            self.root.after(0, lambda: self._on_analysis_error(str(e)))

    def _analyze_single(self):
        """单张分析（支持阈值/分水岭/深度学习三种方法）"""
        path = self.selected_path
        method = self.method_var.get()

        # 检查图像是否已加载
        if self.original_image is None or self.original_image_hsv is None:
            self.root.after(0, lambda: messagebox.showerror("错误", "请先选择图像"))
            return

        start = time.time()

        # 使用已加载的图像进行分析
        original = self.original_image

        if method == 'deep_learning':
            # 深度学习方法
            try:
                from .dl_segment import DLSegmenter
                segmenter = DLSegmenter(self.config)
                # DL需要增强后的图像
                image_dict = {'enhanced': original, 'original': original}
                seg_result = segmenter.segment(image_dict)
                mask = seg_result['mask']
                method_name = 'deep_learning'
            except Exception as e:
                log(f"深度学习分析失败: {e}")
                self.root.after(0, lambda: self._on_analysis_error(
                    f"深度学习分析失败:\n{str(e)}\n\n已回退到阈值分割。"
                ))
                # 回退到阈值方法
                method = 'threshold'
                method_name = 'threshold'
                hsv = self.original_image_hsv
                h_l = self.config['threshold_segmentation']['hsv_range']['lower'][0]
                h_u = self.config['threshold_segmentation']['hsv_range']['upper'][0]
                s_l = self.config['threshold_segmentation']['hsv_range']['lower'][1]
                s_u = self.config['threshold_segmentation']['hsv_range']['upper'][1]
                v_l = self.config['threshold_segmentation']['hsv_range']['lower'][2]
                v_u = self.config['threshold_segmentation']['hsv_range']['upper'][2]
                lower = np.array([h_l, s_l, v_l])
                upper = np.array([h_u, s_u, v_u])
                mask = cv2.inRange(hsv, lower, upper)
        else:
            # 阈值或分水岭方法
            hsv = self.original_image_hsv
            h_l = self.config['threshold_segmentation']['hsv_range']['lower'][0]
            h_u = self.config['threshold_segmentation']['hsv_range']['upper'][0]
            s_l = self.config['threshold_segmentation']['hsv_range']['lower'][1]
            s_u = self.config['threshold_segmentation']['hsv_range']['upper'][1]
            v_l = self.config['threshold_segmentation']['hsv_range']['lower'][2]
            v_u = self.config['threshold_segmentation']['hsv_range']['upper'][2]

            lower = np.array([h_l, s_l, v_l])
            upper = np.array([h_u, s_u, v_u])

            mask = cv2.inRange(hsv, lower, upper)

            # 形态学操作
            morph_cfg = self.config.get('threshold_segmentation', {}).get('morphological_operations', {})
            open_k = np.ones(tuple(morph_cfg.get('open_kernel', [3, 3])), np.uint8)
            close_k = np.ones(tuple(morph_cfg.get('close_kernel', [5, 5])), np.uint8)
            iterations = morph_cfg.get('iterations', 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=iterations)

            # 分水岭算法（可选）
            if method == 'watershed':
                mask = self._apply_watershed(mask, original)
                method_name = 'watershed'
            else:
                method_name = 'threshold'

        # 最小孔隙面积过滤
        min_area = self.config.get('area_calculation', {}).get('min_pore_area', 50)
        if min_area > 0:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            filtered_mask = np.zeros_like(mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    filtered_mask[labels == i] = 255
            mask = filtered_mask

        # 计算统计信息
        h, w = mask.shape
        total_pixels = h * w
        pore_pixels = np.sum(mask > 0)
        porosity = (pore_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        pore_count = num_labels - 1  # 减去背景

        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        avg_area = sum(areas) / len(areas) if areas else 0.0

        stats_dict = {
            'porosity_percent': round(porosity, 4),
            'pore_count': pore_count,
            'total_pore_pixels': int(pore_pixels),
            'total_pixels': int(total_pixels),
            'avg_pore_area': round(avg_area, 2),
            'method': method_name,
            'processing_time': round(time.time() - start, 3),
        }

        # 生成标注图像
        annotated = np.ones_like(original) * 255
        annotated[mask > 0] = [255, 0, 0]

        # 生成叠加图像
        overlay = original.copy()
        for i in range(3):
            overlay[:, :, i] = np.where(mask > 0,
                                         overlay[:, :, i] * 0.5 + [255, 0, 0][i] * 0.5,
                                         overlay[:, :, i])
        overlay = overlay.astype(np.uint8)

        self.current_result = {
            'filename': Path(path).stem,
            'original': original,
            'annotated': annotated,
            'overlay': overlay,
            'mask': mask,
            'stats': stats_dict
        }

        # 更新 UI
        self.root.after(0, self._update_single_result)

    def _analyze_batch(self):
        """批量分析"""
        input_dir = Path(self.selected_path)
        output_dir = input_dir / 'results'
        use_watershed = self.watershed_var.get()

        # 收集图像（支持所有常见格式，大小写不敏感）
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.gif'}
        image_files = [f for f in input_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in extensions]

        total = len(image_files)
        if total == 0:
            self.root.after(0, lambda: messagebox.showwarning("提示", "文件夹中没有图像文件！"))
            return

        self.batch_results = []

        for i, img_path in enumerate(image_files):
            # 更新进度
            self._update_progress(i + 1, total, img_path.name)

            try:
                self._init_analyzer()
                original = load_image(str(img_path))

                processed = self.analyzer.preprocessor.process(original)
                thresh_result = self.analyzer.thresh_segmenter.segment(processed)
                mask = thresh_result['mask']

                if use_watershed:
                    watershed_result = self.analyzer.watershed_segmenter.segment(processed, mask)
                    mask = watershed_result['mask']

                stats = self.analyzer.calculator.calculate(mask)
                stats['filename'] = img_path.stem
                stats['method'] = 'watershed' if use_watershed else 'threshold'

                self.batch_results.append(stats)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # 保存汇总
        if self.batch_results:
            output_dir.mkdir(exist_ok=True)
            self._write_json(output_dir / 'summary.json', self.batch_results)
            self._write_csv(output_dir / 'summary.csv', self.batch_results)

        self.root.after(0, self._update_batch_result)

    def _update_batch_result(self):
        """更新批量分析结果到 UI"""
        if not self.batch_results:
            messagebox.showwarning("提示", "没有成功处理任何图像！")
            self.analyze_btn.config(state='normal')
            return

        # 显示汇总
        avg_porosity = sum(r['porosity_percent'] for r in self.batch_results) / len(self.batch_results)
        total_pores = sum(r['pore_count'] for r in self.batch_results)

        self.porosity_card.set_value(f"{avg_porosity:.4f}")
        self.count_card.set_value(f"{len(self.batch_results)} 张")
        self.avg_area_card.set_value("--")
        self.crack_card.set_value(str(total_pores))
        self.time_card.set_value("批量")

        self.original_viewer.clear()
        self.annotated_viewer.clear()

        # 启用按钮
        self.analyze_btn.config(state='normal')
        self.save_btn.config(state='normal')
        self.export_btn.config(state='normal')

        self.status_bar.set_status(f"批量完成: {len(self.batch_results)} 张", COLORS['success'])
        self.status_bar.set_progress(100)

        messagebox.showinfo("完成", f"批量处理完成！\n"
                            f"成功: {len(self.batch_results)} 张\n"
                            f"平均面孔率: {avg_porosity:.4f}%\n"
                            f"结果已保存到: {Path(self.selected_path) / 'results'}")

    def _on_analysis_error(self, error_msg):
        """分析出错回调"""
        self.analyze_btn.config(state='normal')
        self.status_bar.set_status("分析失败", COLORS['danger'])
        messagebox.showerror("分析错误", f"处理过程中出现错误:\n{error_msg}")

    def _update_progress(self, current, total, name):
        """更新进度条（从后台线程安全调用）"""
        self.root.after(0, lambda: (
            self.status_bar.set_progress(current, total),
            self.status_bar.set_file(name)
        ))

    def _write_json(self, path, data):
        """写入 JSON 文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _write_csv(self, path, data):
        """写入 CSV 文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

    def _clear_results(self):
        """清空结果显示"""
        self.porosity_card.set_value("--")
        self.count_card.set_value("--")
        self.avg_area_card.set_value("--")
        self.crack_card.set_value("--")
        self.time_card.set_value("--")

    def _update_config_from_ui(self):
        """从 UI 更新配置"""
        # 更新 HSV 阈值
        h_l, h_u = self.h_range.get()
        s_l, s_u = self.s_range.get()
        v_l, v_u = self.v_range.get()
        self.config['threshold_segmentation']['hsv_range']['lower'] = [h_l, s_l, v_l]
        self.config['threshold_segmentation']['hsv_range']['upper'] = [h_u, s_u, v_u]

        # 更新最小孔隙面积
        self.config['area_calculation']['min_pore_area'] = self.min_area_slider.get()

    def _reset_params(self):
        """重置参数到默认值"""
        default_config = load_config(self.config_path)

        lower = default_config.get('threshold_segmentation', {}).get('hsv_range', {}).get('lower', [100, 50, 50])
        upper = default_config.get('threshold_segmentation', {}).get('hsv_range', {}).get('upper', [140, 255, 255])
        min_area = default_config.get('area_calculation', {}).get('min_pore_area', 50)

        self.h_range.set(lower[0], upper[0])
        self.s_range.set(lower[1], upper[1])
        self.v_range.set(lower[2], upper[2])
        self.min_area_slider.set(min_area)
        self.watershed_var.set(False)

        self.config = default_config
        self._clear_samples()
        messagebox.showinfo("提示", "参数已重置为默认值！")

    def _save_results(self):
        """保存结果"""
        if not self.current_result and not self.batch_results:
            messagebox.showwarning("提示", "没有可保存的结果！")
            return

        output_dir = filedialog.askdirectory(title="选择保存目录")
        if not output_dir:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if self.mode_var.get() == 'single' and self.current_result:
                result = self.current_result

                save_image(result['annotated'], output_dir / f"{result['filename']}_annotated.png")
                save_image(result['overlay'], output_dir / f"{result['filename']}_overlay.png")
                save_image(result['mask'], output_dir / f"{result['filename']}_mask.png")
                self._write_json(output_dir / f"{result['filename']}_stats.json", result['stats'])

            elif self.mode_var.get() == 'batch' and self.batch_results:
                self._write_json(output_dir / 'summary.json', self.batch_results)
                self._write_csv(output_dir / 'summary.csv', self.batch_results)

            messagebox.showinfo("完成", f"结果已保存到:\n{output_dir}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _export_csv(self):
        """导出 CSV"""
        if not self.batch_results:
            messagebox.showwarning("提示", "没有可导出的数据！")
            return

        path = filedialog.asksaveasfilename(
            title="导出 CSV",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv")]
        )
        if not path:
            return

        try:
            self._write_csv(path, self.batch_results)
            messagebox.showinfo("完成", f"CSV 已导出到:\n{path}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    def _on_param_changed(self):
        """参数改变时实时预览（防抖）"""
        log(f"_on_param_changed 被调用, original_image={self.original_image is not None}")
        if self.original_image is None:
            self.status_bar.set_status("请先选择图像", COLORS['warning'])
            return

        # 取消之前的定时器
        if hasattr(self, '_preview_after_id'):
            try:
                self.root.after_cancel(self._preview_after_id)
            except:
                pass

        # 延迟200ms后预览，避免滑动时频繁计算
        self._preview_after_id = self.root.after(200, self._preview_threshold)
        log("已设置200ms后预览")

    def _update_single_result(self):
        """更新单张分析结果到 UI"""
        result = self.current_result
        if result is None:
            return

        stats = result['stats']

        # 显示图像
        self.original_viewer.show_image(result['original'])
        self.annotated_viewer.show_image(result['annotated'])

        # 更新数值
        self.porosity_card.set_value(f"{stats['porosity_percent']:.4f}")
        self.count_card.set_value(str(stats['pore_count']))
        self.avg_area_card.set_value(f"{stats['avg_pore_area']:.2f}")
        self.crack_card.set_value(str(stats.get('crack_count', 0)))
        self.time_card.set_value(f"{stats['processing_time']}")

        # 启用按钮
        self.analyze_btn.config(state='normal')
        self.save_btn.config(state='normal')
        self.export_btn.config(state='disabled')  # 单张不启用导出CSV

        self.status_bar.set_status("分析完成", COLORS['success'])
        self.status_bar.set_progress(100)

    def run(self):
        """启动 GUI"""
        self.root.mainloop()
