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
    COLORS, ImageViewer, ResultCard, ParameterSlider,
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

        # 分水岭算法
        self.watershed_var = tk.BooleanVar(value=False)
        tk.Checkbutton(param_frame, text="启用分水岭算法", variable=self.watershed_var,
                       bg=COLORS['bg_panel'], fg=COLORS['text'],
                       font=('Microsoft YaHei', 9)).pack(anchor='w', pady=(0, 5))

        # HSV 阈值设置
        thresh_frame = tk.LabelFrame(param_frame, text="颜色阈值 (HSV)",
                                      bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
                                      font=('Microsoft YaHei', 9), padx=5, pady=5)
        thresh_frame.pack(fill='x', pady=(0, 5))

        # 下限
        lower = self.config.get('threshold_segmentation', {}).get('hsv_range', {}).get('lower', [100, 50, 50])
        self.h_lower = ParameterSlider(thresh_frame, label="H下限(色相)", from_=0, to=180, default=lower[0],
                                        command=lambda v: self._on_param_changed())
        self.h_lower.pack(fill='x', pady=2)
        self.s_lower = ParameterSlider(thresh_frame, label="S下限(饱和度)", from_=0, to=255, default=lower[1],
                                        command=lambda v: self._on_param_changed())
        self.s_lower.pack(fill='x', pady=2)
        self.v_lower = ParameterSlider(thresh_frame, label="V下限(亮度)", from_=0, to=255, default=lower[2],
                                        command=lambda v: self._on_param_changed())
        self.v_lower.pack(fill='x', pady=2)

        # 上限
        upper = self.config.get('threshold_segmentation', {}).get('hsv_range', {}).get('upper', [140, 255, 255])
        self.h_upper = ParameterSlider(thresh_frame, label="H上限(色相)", from_=0, to=180, default=upper[0],
                                        command=lambda v: self._on_param_changed())
        self.h_upper.pack(fill='x', pady=2)
        self.s_upper = ParameterSlider(thresh_frame, label="S上限(饱和度)", from_=0, to=255, default=upper[1],
                                        command=lambda v: self._on_param_changed())
        self.s_upper.pack(fill='x', pady=2)
        self.v_upper = ParameterSlider(thresh_frame, label="V上限(亮度)", from_=0, to=255, default=upper[2],
                                        command=lambda v: self._on_param_changed())
        self.v_upper.pack(fill='x', pady=2)

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

    def _build_color_pick_panel(self, parent):
        """构建取色校正面板"""
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
        self.pick_btn = CustomButton(pick_frame, text="开始取色", style='secondary',
                                      command=self._enter_pick_mode)
        self.pick_btn.pack(fill='x', pady=(2, 2))

        # 取样状态
        self.sample_label = tk.Label(pick_frame, text="已取样: 0 个", bg=COLORS['bg_panel'],
                                      fg=COLORS['text_secondary'], font=('Microsoft YaHei', 9))
        self.sample_label.pack(anchor='w')

        # 步骤3：容差调整
        tk.Label(pick_frame, text="Step 3: 调整容差", bg=COLORS['bg_panel'],
                 fg=COLORS['text'], font=('Microsoft YaHei', 9, 'bold')).pack(anchor='w', pady=(5, 0))

        # H容差：色相范围，控制颜色偏蓝还是偏绿/紫
        self.h_tol = ParameterSlider(pick_frame, label="H容差(色相)", from_=1, to=30, default=10,
                                     command=lambda v: self._on_param_changed())
        self.h_tol.pack(fill='x', pady=1)
        tk.Label(pick_frame, text="  色相范围：越大包含越多蓝绿色",
                 bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
                 font=('Microsoft YaHei', 8)).pack(anchor='w')

        # S容差：饱和度范围，控制颜色鲜艳程度
        self.s_tol = ParameterSlider(pick_frame, label="S容差(饱和度)", from_=10, to=100, default=40,
                                     command=lambda v: self._on_param_changed())
        self.s_tol.pack(fill='x', pady=1)
        tk.Label(pick_frame, text="  饱和度范围：越大包含越淡的蓝色",
                 bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
                 font=('Microsoft YaHei', 8)).pack(anchor='w')

        # V容差：亮度范围，控制明暗程度
        self.v_tol = ParameterSlider(pick_frame, label="V容差(亮度)", from_=10, to=100, default=40,
                                     command=lambda v: self._on_param_changed())
        self.v_tol.pack(fill='x', pady=1)
        tk.Label(pick_frame, text="  亮度范围：越大包含越暗/越亮的蓝色",
                 bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
                 font=('Microsoft YaHei', 8)).pack(anchor='w')

        # 应用取样按钮
        self.apply_sample_btn = CustomButton(pick_frame, text="应用取样到阈值", style='primary',
                                              command=self._apply_sampled_threshold)
        self.apply_sample_btn.pack(fill='x', pady=(5, 0))

        # 清空取样
        self.clear_sample_btn = CustomButton(pick_frame, text="清空取样", style='danger',
                                              command=self._clear_samples)
        self.clear_sample_btn.pack(fill='x', pady=(2, 0))

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

        self.h_lower.set(h_lower)
        self.s_lower.set(s_lower)
        self.v_lower.set(v_lower)
        self.h_upper.set(h_upper)
        self.s_upper.set(s_upper)
        self.v_upper.set(v_upper)

        self.status_bar.set_status(f"自动提取: H{h_mean} S{s_mean} V{v_mean}", COLORS['success'])

        # 自动预览
        self._preview_threshold()

    def _enter_pick_mode(self):
        """进入取色模式"""
        if self.original_image is None:
            messagebox.showwarning("提示", "请先选择图像！")
            return

        self.pick_mode = True
        self.pick_btn.config(text="取色中...")
        self.pick_btn.set_style('primary')
        self.status_bar.set_status("请点击原图上的蓝色孔隙区域", COLORS['warning'])
        messagebox.showinfo("取色提示", "请点击原图上的蓝色孔隙区域\n可以点击多个位置取平均")

    def _on_image_pick(self, img_x, img_y):
        """处理图像点击取色"""
        if not self.pick_mode or self.original_image_hsv is None:
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

        # 在状态栏显示最新取样
        self.status_bar.set_status(
            f"取样#{len(self.sampled_colors)}: H={sample_h} S={sample_s} V={sample_v}",
            COLORS['success']
        )

        # 如果取样超过5个，自动应用
        if len(self.sampled_colors) >= 5:
            self._apply_sampled_threshold()
            self.pick_mode = False
            self.pick_btn.config(text="开始取色")
            self.pick_btn.set_style('secondary')
            messagebox.showinfo("取色完成", "已自动应用5个取样点到阈值")

    def _apply_sampled_threshold(self):
        """将取样颜色应用到阈值设置"""
        if not self.sampled_colors:
            messagebox.showwarning("提示", "请先取样！")
            return

        # 获取容差
        tol_h = self.h_tol.get()
        tol_s = self.s_tol.get()
        tol_v = self.v_tol.get()

        # 计算所有取样颜色的范围
        h_vals = [c[0] for c in self.sampled_colors]
        s_vals = [c[1] for c in self.sampled_colors]
        v_vals = [c[2] for c in self.sampled_colors]

        h_min, h_max = min(h_vals), max(h_vals)
        s_min, s_max = min(s_vals), max(s_vals)
        v_min, v_max = min(v_vals), max(v_vals)

        # 设置阈值：取样范围 ± 容差
        self.h_lower.set(max(0, h_min - tol_h))
        self.h_upper.set(min(180, h_max + tol_h))
        self.s_lower.set(max(0, s_min - tol_s))
        self.s_upper.set(min(255, s_max + tol_s))
        self.v_lower.set(max(0, v_min - tol_v))
        self.v_upper.set(min(255, v_max + tol_v))

        self.status_bar.set_status(
            f"已应用{len(self.sampled_colors)}个取样: H{h_min}-{h_max} S{s_min}-{s_max} V{v_min}-{v_max}",
            COLORS['success']
        )

        # 自动预览
        self._preview_threshold()

    def _clear_samples(self):
        """清空所有取样"""
        self.sampled_colors = []
        self.sample_label.config(text="已取样: 0 个")
        self.pick_mode = False
        self.pick_btn.config(text="开始取色")
        self.pick_btn.set_style('secondary')
        self.status_bar.set_status("取样已清空", COLORS['text_secondary'])

    def _preview_threshold(self):
        """实时预览当前阈值的分割效果（使用与正式分析相同的预处理流程）"""
        if self.original_image is None:
            return

        try:
            # 先更新配置（确保阈值是最新的）
            self._update_config_from_ui()

            # 使用与正式分析相同的预处理流程
            preprocessor = ImagePreprocessor(self.config)
            processed = preprocessor.process(self.original_image)
            hsv = processed['hsv']

            # 使用当前UI的阈值参数进行分割
            h_l = self.h_lower.get()
            h_u = self.h_upper.get()
            s_l = self.s_lower.get()
            s_u = self.s_upper.get()
            v_l = self.v_lower.get()
            v_u = self.v_upper.get()

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

            # 显示预览
            self.annotated_viewer.show_image(annotated)

            # 计算并显示预览面孔率
            pore_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            porosity = (pore_pixels / total_pixels) * 100

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
        """单张分析"""
        path = self.selected_path
        use_watershed = self.watershed_var.get()

        # 检查图像是否已加载
        if self.original_image is None:
            self.root.after(0, lambda: messagebox.showerror("错误", "请先选择图像"))
            return

        # 重新初始化分析器（使用最新配置）
        self._init_analyzer()

        start = time.time()

        # 使用已加载的图像进行分析（避免重复加载，确保与预览一致）
        original = self.original_image
        processed = self.analyzer.preprocessor.process(original)
        thresh_result = self.analyzer.thresh_segmenter.segment(processed)
        mask = thresh_result['mask']

        if use_watershed:
            watershed_result = self.analyzer.watershed_segmenter.segment(processed, mask)
            mask = watershed_result['mask']

        stats = self.analyzer.calculator.calculate(mask)
        stats['filename'] = Path(path).stem
        stats['method'] = 'watershed' if use_watershed else 'threshold'
        stats['processing_time'] = round(time.time() - start, 3)

        crack_info = self.analyzer.calculator.detect_cracks(mask)
        stats['crack_count'] = crack_info['crack_count']

        annotated = self.analyzer.visualizer.create_annotated_image(mask)
        overlay = self.analyzer.visualizer.overlay_on_original(original, mask)

        self.current_result = {
            'filename': Path(path).stem,
            'original': original,
            'annotated': annotated,
            'overlay': overlay,
            'mask': mask,
            'stats': stats
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
        self.config['threshold_segmentation']['hsv_range']['lower'] = [
            self.h_lower.get(), self.s_lower.get(), self.v_lower.get()
        ]
        self.config['threshold_segmentation']['hsv_range']['upper'] = [
            self.h_upper.get(), self.s_upper.get(), self.v_upper.get()
        ]

        # 更新最小孔隙面积
        self.config['area_calculation']['min_pore_area'] = self.min_area_slider.get()

    def _reset_params(self):
        """重置参数到默认值"""
        default_config = load_config(self.config_path)

        lower = default_config.get('threshold_segmentation', {}).get('hsv_range', {}).get('lower', [100, 50, 50])
        upper = default_config.get('threshold_segmentation', {}).get('hsv_range', {}).get('upper', [140, 255, 255])
        min_area = default_config.get('area_calculation', {}).get('min_pore_area', 50)

        self.h_lower.set(lower[0])
        self.s_lower.set(lower[1])
        self.v_lower.set(lower[2])
        self.h_upper.set(upper[0])
        self.s_upper.set(upper[1])
        self.v_upper.set(upper[2])
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
