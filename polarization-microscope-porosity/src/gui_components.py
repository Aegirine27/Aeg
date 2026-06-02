"""
GUI 可复用组件模块

包含：
    - ImageViewer: 图像预览组件（支持自适应缩放）
    - ResultCard: 结果展示卡片
    - ParameterSlider: 参数滑块（带数值显示）
    - StatusBar: 状态栏
    - ProgressBar: 自定义进度条
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2


# ============ 配色常量 ============
COLORS = {
    'bg_main': '#F5F6FA',
    'bg_panel': '#FFFFFF',
    'primary': '#2C3E50',
    'accent': '#3498DB',
    'success': '#27AE60',
    'danger': '#E74C3C',
    'warning': '#F39C12',
    'text': '#2C3E50',
    'text_secondary': '#7F8C8D',
    'border': '#BDC3C7',
    'hover': '#2980B9',
}


class ImageViewer(tk.Canvas):
    """
    图像预览组件

    功能：
        - 自适应缩放显示（保持长宽比）
        - 支持 OpenCV BGR 图像直接传入
        - 双击可放大查看
    """

    def __init__(self, parent, title="图像", width=400, height=400, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg='#E8E8E8', highlightthickness=1,
                         highlightbackground=COLORS['border'], **kwargs)

        self.title = title
        self.canvas_width = width
        self.canvas_height = height
        self.current_image = None  # 当前显示的 PIL Image
        self.photo_image = None    # PhotoImage 引用（防止GC）

        # 标题文字
        self.create_text(width//2, height//2, text=f"{title}\n(暂无图像)",
                         fill=COLORS['text_secondary'], font=('Microsoft YaHei', 12),
                         justify='center')

    def show_image(self, cv_image):
        """
        显示 OpenCV BGR 图像

        Args:
            cv_image: numpy array, BGR格式
        """
        if cv_image is None:
            self.clear()
            return

        # BGR -> RGB
        if len(cv_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv_image

        # 转为 PIL Image
        pil_image = Image.fromarray(rgb_image)

        # 自适应缩放
        pil_image = self._fit_image(pil_image)

        # 转为 PhotoImage
        self.photo_image = ImageTk.PhotoImage(pil_image)

        # 清空画布并显示
        self.delete("all")
        self.create_image(self.canvas_width//2, self.canvas_height//2,
                          image=self.photo_image, anchor='center')

        # 添加标题
        self.create_text(10, 10, text=self.title, anchor='nw',
                         fill=COLORS['primary'], font=('Microsoft YaHei', 10, 'bold'),
                         tags='title')

    def _fit_image(self, pil_image):
        """按比例缩放图像以适应画布"""
        img_w, img_h = pil_image.size
        canvas_w, canvas_h = self.canvas_width - 20, self.canvas_height - 40

        if img_w == 0 or img_h == 0:
            return pil_image

        scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        return pil_image.resize((new_w, new_h), Image.LANCZOS)

    def clear(self):
        """清空显示"""
        self.delete("all")
        self.create_text(self.canvas_width//2, self.canvas_height//2,
                         text=f"{self.title}\n(暂无图像)",
                         fill=COLORS['text_secondary'], font=('Microsoft YaHei', 12),
                         justify='center')


class ResultCard(tk.Frame):
    """
    结果展示卡片

    显示单项指标的数值和标签
    """

    def __init__(self, parent, label="指标", value="--", unit="",
                 font_size=14, value_color=None, **kwargs):
        super().__init__(parent, bg=COLORS['bg_panel'], **kwargs)

        if value_color is None:
            value_color = COLORS['accent']

        # 标签
        self.label_widget = tk.Label(self, text=label, bg=COLORS['bg_panel'],
                                      fg=COLORS['text_secondary'], font=('Microsoft YaHei', 10))
        self.label_widget.pack()

        # 数值
        self.value_widget = tk.Label(self, text=f"{value}{unit}", bg=COLORS['bg_panel'],
                                      fg=value_color, font=('Microsoft YaHei', font_size, 'bold'))
        self.value_widget.pack()

    def set_value(self, value):
        """更新数值"""
        self.value_widget.config(text=str(value))


class ParameterSlider(tk.Frame):
    """
    参数滑块组件

    包含标签、滑块、数值显示
    """

    def __init__(self, parent, label="参数", from_=0, to=255, default=0,
                 orient='horizontal', command=None, **kwargs):
        super().__init__(parent, bg=COLORS['bg_panel'], **kwargs)

        self.command = command

        # 标签框架
        label_frame = tk.Frame(self, bg=COLORS['bg_panel'])
        label_frame.pack(fill='x')

        tk.Label(label_frame, text=label, bg=COLORS['bg_panel'],
                 fg=COLORS['text'], font=('Microsoft YaHei', 9)).pack(side='left')

        self.value_label = tk.Label(label_frame, text=str(default), bg=COLORS['bg_panel'],
                                     fg=COLORS['accent'], font=('Microsoft YaHei', 9, 'bold'),
                                     width=4)
        self.value_label.pack(side='right')

        # 滑块
        self.slider = tk.Scale(self, from_=from_, to=to, orient=orient,
                               showvalue=False, length=180,
                               bg=COLORS['bg_panel'], highlightthickness=0,
                               troughcolor=COLORS['border'], activebackground=COLORS['accent'],
                               sliderrelief='flat', sliderlength=15,
                               command=self._on_change)
        self.slider.set(default)
        self.slider.pack(fill='x', pady=(2, 0))

    def _on_change(self, value):
        """滑块值改变回调"""
        self.value_label.config(text=str(int(float(value))))
        if self.command:
            self.command(int(float(value)))

    def get(self):
        """获取当前值"""
        return self.slider.get()

    def set(self, value):
        """设置值"""
        self.slider.set(value)
        self.value_label.config(text=str(value))


class CustomButton(tk.Button):
    """
    自定义样式按钮
    """

    def __init__(self, parent, text="按钮", style='primary', command=None, **kwargs):
        """
        style: 'primary' | 'secondary' | 'success' | 'danger'
        """
        styles = {
            'primary': {'bg': COLORS['accent'], 'fg': 'white', 'activebg': COLORS['hover']},
            'secondary': {'bg': COLORS['bg_panel'], 'fg': COLORS['text'], 'activebg': COLORS['border']},
            'success': {'bg': COLORS['success'], 'fg': 'white', 'activebg': '#229954'},
            'danger': {'bg': COLORS['danger'], 'fg': 'white', 'activebg': '#C0392B'},
        }

        style_cfg = styles.get(style, styles['primary'])

        super().__init__(parent, text=text, command=command,
                         font=('Microsoft YaHei', 10),
                         bg=style_cfg['bg'], fg=style_cfg['fg'],
                         activebackground=style_cfg['activebg'],
                         activeforeground='white',
                         relief='flat', cursor='hand2',
                         padx=15, pady=5, **kwargs)


class StatusBar(tk.Frame):
    """
    底部状态栏
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS['primary'], height=28, **kwargs)
        self.pack_propagate(False)

        # 状态文字
        self.status_label = tk.Label(self, text="就绪", bg=COLORS['primary'],
                                      fg='white', font=('Microsoft YaHei', 9))
        self.status_label.pack(side='left', padx=(10, 0))

        # 当前文件
        self.file_label = tk.Label(self, text="当前文件: 无", bg=COLORS['primary'],
                                    fg='#BDC3C7', font=('Microsoft YaHei', 9))
        self.file_label.pack(side='left', padx=(20, 0))

        # 进度条框架
        self.progress_frame = tk.Frame(self, bg=COLORS['primary'])
        self.progress_frame.pack(side='right', padx=(0, 10))

        self.progress_label = tk.Label(self.progress_frame, text="0%", bg=COLORS['primary'],
                                        fg='white', font=('Microsoft YaHei', 9))
        self.progress_label.pack(side='left', padx=(0, 5))

        self.progress_bar = ttk.Progressbar(self.progress_frame, orient='horizontal',
                                             length=150, mode='determinate')
        self.progress_bar.pack(side='left')

    def set_status(self, text, color='white'):
        """设置状态文字"""
        self.status_label.config(text=text, fg=color)

    def set_file(self, filename):
        """设置当前文件名"""
        self.file_label.config(text=f"当前文件: {filename}")

    def set_progress(self, value, maximum=100):
        """设置进度"""
        self.progress_bar.config(maximum=maximum, value=value)
        pct = int((value / maximum) * 100) if maximum > 0 else 0
        self.progress_label.config(text=f"{pct}%")

    def reset(self):
        """重置状态栏"""
        self.set_status("就绪")
        self.set_file("无")
        self.set_progress(0)
