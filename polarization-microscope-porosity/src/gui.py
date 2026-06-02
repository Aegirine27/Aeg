"""
GUI 主窗口模块

整合所有组件，实现完整的交互逻辑：
    - 文件/文件夹选择
    - 参数调整
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

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("🧪 偏光显微镜面孔率识别系统 v1.0")
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

        tk.Label(title_frame, text="🧪 偏光显微镜面孔率识别系统",
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
        """构建左侧面板"""
        left_frame = tk.Frame(parent, bg=COLORS['bg_panel'], width=300)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        left_frame.pack_propagate(False)

        # --- 输入选择区 ---
        input_frame = tk.LabelFrame(left_frame, text=" 📂 输入选择 ",
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

        # --- 参数设置区 ---
        param_frame = tk.LabelFrame(left_frame, text=" ⚙️ 参数设置 ",
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
        self.h_lower = ParameterSlider(thresh_frame, label="H 下限", from_=0, to=180, default=lower[0])
        self.h_lower.pack(fill='x', pady=2)
        self.s_lower = ParameterSlider(thresh_frame, label="S 下限", from_=0, to=255, default=lower[1])
        self.s_lower.pack(fill='x', pady=2)
        self.v_lower = ParameterSlider(thresh_frame, label="V 下限", from_=0, to=255, default=lower[2])
        self.v_lower.pack(fill='x', pady=2)

        # 上限
        upper = self.config.get('threshold_segmentation', {}).get('hsv_range', {}).get('upper', [140, 255, 255])
        self.h_upper = ParameterSlider(thresh_frame, label="H 上限", from_=0, to=180, default=upper[0])
        self.h_upper.pack(fill='x', pady=2)
        self.s_upper = ParameterSlider(thresh_frame, label="S 上限", from_=0, to=255, default=upper[1])
        self.s_upper.pack(fill='x', pady=2)
        self.v_upper = ParameterSlider(thresh_frame, label="V 上限", from_=0, to=255, default=upper[2])
        self.v_upper.pack(fill='x', pady=2)

        # 最小孔隙面积
        min_area = self.config.get('area_calculation', {}).get('min_pore_area', 50)
        self.min_area_slider = ParameterSlider(param_frame, label="最小孔隙面积", from_=0, to=500, default=min_area)
        self.min_area_slider.pack(fill='x', pady=(5, 0))

        # --- 操作按钮区 ---
        action_frame = tk.Frame(left_frame, bg=COLORS['bg_panel'])
        action_frame.pack(fill='x', padx=10, pady=(10, 5))

        self.analyze_btn = CustomButton(action_frame, text="▶ 开始分析", style='primary',
                                         command=self._start_analysis)
        self.analyze_btn.pack(fill='x', pady=(0, 5))

        self.reset_btn = CustomButton(action_frame, text="↺ 重置参数", style='secondary',
                                       command=self._reset_params)
        self.reset_btn.pack(fill='x', pady=(0, 5))

        self.save_btn = CustomButton(action_frame, text="💾 保存结果", style='success',
                                      command=self._save_results, state='disabled')
        self.save_btn.pack(fill='x', pady=(0, 5))

        self.export_btn = CustomButton(action_frame, text="📊 导出CSV", style='secondary',
                                        command=self._export_csv, state='disabled')
        self.export_btn.pack(fill='x')

    def _build_right_panel(self, parent):
        """构建右侧面板"""
        right_frame = tk.Frame(parent, bg=COLORS['bg_main'])
        right_frame.pack(side='left', fill='both', expand=True)

        # --- 图像预览区 ---
        preview_frame = tk.LabelFrame(right_frame, text=" 📷 图像预览 ",
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

        self.original_viewer = ImageViewer(img_container, title="原始图像", width=420, height=420)
        self.original_viewer.grid(row=0, column=0, padx=(0, 5), sticky='nsew')

        self.annotated_viewer = ImageViewer(img_container, title="标注图像（孔隙=蓝色，背景=白色）",
                                             width=420, height=420)
        self.annotated_viewer.grid(row=0, column=1, padx=(5, 0), sticky='nsew')

        # --- 结果展示区 ---
        result_frame = tk.LabelFrame(right_frame, text=" 📊 分析结果 ",
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
            path = filedialog.askopenfilename(
                title="选择显微镜照片",
                filetypes=[("图像文件", "*.jpg *.jpeg *.tif *.tiff *.png *.bmp"),
                           ("所有文件", "*.*")]
            )
        else:
            path = filedialog.askdirectory(title="选择照片文件夹")

        if path:
            self.selected_path = path
            self.path_var.set(f"已选择: {path}")
            self.status_bar.set_file(Path(path).name)

            # 如果是单张，先显示原图
            if self.mode_var.get() == 'single':
                try:
                    img = load_image(path)
                    self.original_viewer.show_image(img)
                    self.annotated_viewer.clear()
                    self._clear_results()
                except Exception:
                    pass

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

        # 重新初始化分析器（使用最新配置）
        self._init_analyzer()

        start = time.time()

        # 执行分析
        original = load_image(path)
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

        # 收集图像
        extensions = {'.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp'}
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

    def _update_single_result(self):
        """更新单张分析结果到 UI"""
        result = self.current_result
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

    def run(self):
        """启动 GUI"""
        self.root.mainloop()
