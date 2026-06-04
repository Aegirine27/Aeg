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

    def __init__(self, parent, title="图像", width=400, height=400, click_callback=None, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg='#E8E8E8', highlightthickness=1,
                         highlightbackground=COLORS['border'], **kwargs)

        self.title = title
        self.canvas_width = width
        self.canvas_height = height
        self.current_image = None  # 当前显示的 PIL Image
        self.photo_image = None    # PhotoImage 引用（防止GC）
        self.click_callback = click_callback  # 鼠标点击回调
        self.display_offset = (0, 0)  # 图像在画布中的偏移
        self.display_scale = 1.0      # 图像缩放比例

        # 标题文字
        self.create_text(width//2, height//2, text=f"{title}\n(暂无图像)",
                         fill=COLORS['text_secondary'], font=('Microsoft YaHei', 12),
                         justify='center')

        # 绑定鼠标点击事件
        if self.click_callback:
            self.bind('<Button-1>', self._on_click)
            self.config(cursor='crosshair')

    def _on_click(self, event):
        """处理鼠标点击，将画布坐标转换为图像坐标"""
        if self.current_image is None or self.click_callback is None:
            return

        # 计算图像在画布中的实际位置
        img_w = int(self.current_image.width * self.display_scale)
        img_h = int(self.current_image.height * self.display_scale)
        offset_x = (self.canvas_width - img_w) // 2
        offset_y = (self.canvas_height - img_h) // 2

        # 检查点击是否在图像范围内
        if offset_x <= event.x <= offset_x + img_w and offset_y <= event.y <= offset_y + img_h:
            # 转换回原始图像坐标
            img_x = int((event.x - offset_x) / self.display_scale)
            img_y = int((event.y - offset_y) / self.display_scale)
            self.click_callback(img_x, img_y)

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
        self.current_image = pil_image  # 保存原始PIL图像

        # 自适应缩放
        scaled_image, self.display_scale = self._fit_image(pil_image)

        # 转为 PhotoImage
        self.photo_image = ImageTk.PhotoImage(scaled_image)

        # 清空画布并显示
        self.delete("all")
        self.create_image(self.canvas_width//2, self.canvas_height//2,
                          image=self.photo_image, anchor='center')

        # 添加标题
        self.create_text(10, 10, text=self.title, anchor='nw',
                         fill=COLORS['primary'], font=('Microsoft YaHei', 10, 'bold'),
                         tags='title')

        # 如果有取色回调，显示提示
        if self.click_callback:
            self.create_text(self.canvas_width//2, self.canvas_height - 15,
                             text="点击图像取色", anchor='s',
                             fill=COLORS['accent'], font=('Microsoft YaHei', 9),
                             tags='hint')

    def _fit_image(self, pil_image):
        """按比例缩放图像以适应画布，返回缩放后的图像和缩放比例"""
        img_w, img_h = pil_image.size
        canvas_w, canvas_h = self.canvas_width - 20, self.canvas_height - 40

        if img_w == 0 or img_h == 0:
            return pil_image, 1.0

        scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        return pil_image.resize((new_w, new_h), Image.LANCZOS), scale

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


class RangeSlider(tk.Canvas):
    """
    双滑块范围选择器（下限/上限合并到同一滑动条）

    用 Canvas 绘制轨道和两个可拖动手柄，
    手柄之间区域高亮显示当前选中范围。
    """

    TRACK_Y = 22
    TRACK_START = 35
    TRACK_END = 215
    HANDLE_RADIUS = 7
    MIN_GAP = 8  # 两个手柄之间的最小像素间距

    def __init__(self, parent, label="范围", from_=0, to=255,
                 lower_default=0, upper_default=255, command=None, **kwargs):
        super().__init__(parent, width=250, height=45,
                         bg=COLORS['bg_panel'], highlightthickness=0, **kwargs)

        self.from_ = from_
        self.to = to
        self.command = command
        self._dragging = None  # 'lower', 'upper', 或 None

        # 当前值
        self.lower_value = lower_default
        self.upper_value = upper_default

        # 标签
        self.create_text(5, 8, text=label, anchor='nw',
                         fill=COLORS['text'], font=('Microsoft YaHei', 9))

        # 数值显示
        self.value_text = self.create_text(245, 8, text=self._fmt_value(), anchor='ne',
                                            fill=COLORS['accent'], font=('Microsoft YaHei', 9, 'bold'))

        # 绑定鼠标事件
        self.bind('<Button-1>', self._on_mouse_down)
        self.bind('<B1-Motion>', self._on_mouse_drag)
        self.bind('<ButtonRelease-1>', self._on_mouse_up)

        # 初始绘制
        self._draw()

    def _value_to_x(self, value):
        """将数值映射到轨道像素坐标"""
        ratio = (value - self.from_) / (self.to - self.from_)
        return self.TRACK_START + ratio * (self.TRACK_END - self.TRACK_START)

    def _x_to_value(self, x):
        """将轨道像素坐标映射到数值"""
        ratio = (x - self.TRACK_START) / (self.TRACK_END - self.TRACK_START)
        value = self.from_ + ratio * (self.to - self.from_)
        return int(round(max(self.from_, min(self.to, value))))

    def _fmt_value(self):
        """格式化数值显示"""
        return f"{self.lower_value} - {self.upper_value}"

    def _draw(self):
        """绘制轨道和手柄"""
        self.delete('track', 'fill', 'handle')

        x_low = self._value_to_x(self.lower_value)
        x_high = self._value_to_x(self.upper_value)
        y = self.TRACK_Y
        r = self.HANDLE_RADIUS

        # 轨道背景（灰色）
        self.create_line(self.TRACK_START, y, self.TRACK_END, y,
                         fill=COLORS['border'], width=4, tags='track', capstyle='round')

        # 选中区域（蓝色高亮）
        self.create_line(x_low, y, x_high, y,
                         fill=COLORS['accent'], width=4, tags='fill', capstyle='round')

        # 下限手柄
        self.create_oval(x_low - r, y - r, x_low + r, y + r,
                         fill='white', outline=COLORS['accent'], width=2, tags='handle')

        # 上限手柄
        self.create_oval(x_high - r, y - r, x_high + r, y + r,
                         fill='white', outline=COLORS['accent'], width=2, tags='handle')

        # 更新数值显示
        self.itemconfig(self.value_text, text=self._fmt_value())

    def _on_mouse_down(self, event):
        """鼠标按下：判断选中哪个手柄（只有点击手柄附近才能拖动）"""
        x_low = self._value_to_x(self.lower_value)
        x_high = self._value_to_x(self.upper_value)
        y = self.TRACK_Y

        # 计算点击位置到两个手柄的距离
        dist_low = abs(event.x - x_low) + abs(event.y - y)
        dist_high = abs(event.x - x_high) + abs(event.y - y)

        # 判断点击了哪个手柄（考虑吸附范围）
        if dist_low < dist_high and dist_low < 20:
            self._dragging = 'lower'
        elif dist_high <= dist_low and dist_high < 20:
            self._dragging = 'upper'
        else:
            # 点击在轨道上但不靠近手柄：不开始拖动，避免误触
            self._dragging = None

    def _on_mouse_drag(self, event):
        """鼠标拖动：移动手柄"""
        if self._dragging == 'lower':
            self._update_lower(event.x)
        elif self._dragging == 'upper':
            self._update_upper(event.x)

    def _update_lower(self, x):
        """更新下限值"""
        x = max(self.TRACK_START, min(x, self.TRACK_END))
        # 确保不超过上限
        x_high = self._value_to_x(self.upper_value)
        x = min(x, x_high - self.MIN_GAP)
        self.lower_value = self._x_to_value(x)
        self._draw()
        if self.command:
            self.command(self.lower_value, self.upper_value)

    def _update_upper(self, x):
        """更新上限值"""
        x = max(self.TRACK_START, min(x, self.TRACK_END))
        # 确保不低于下限
        x_low = self._value_to_x(self.lower_value)
        x = max(x, x_low + self.MIN_GAP)
        self.upper_value = self._x_to_value(x)
        self._draw()
        if self.command:
            self.command(self.lower_value, self.upper_value)

    def _on_mouse_up(self, event):
        """鼠标释放：停止拖动"""
        self._dragging = None

    def get(self):
        """获取当前范围 (下限, 上限)"""
        return (self.lower_value, self.upper_value)

    def set(self, lower, upper):
        """设置范围"""
        self.lower_value = max(self.from_, min(lower, self.to))
        self.upper_value = max(self.from_, min(upper, self.to))
        # 确保下限 <= 上限
        if self.lower_value > self.upper_value:
            self.lower_value, self.upper_value = self.upper_value, self.lower_value
        self._draw()
        if self.command:
            self.command(self.lower_value, self.upper_value)


class CustomButton(tk.Button):
    """
    自定义样式按钮
    """

    STYLES = {
        'primary': {'bg': COLORS['accent'], 'fg': 'white', 'activebackground': COLORS['hover']},
        'secondary': {'bg': COLORS['bg_panel'], 'fg': COLORS['text'], 'activebackground': COLORS['border']},
        'success': {'bg': COLORS['success'], 'fg': 'white', 'activebackground': '#229954'},
        'danger': {'bg': COLORS['danger'], 'fg': 'white', 'activebackground': '#C0392B'},
    }

    def __init__(self, parent, text="按钮", style='primary', command=None, **kwargs):
        """
        style: 'primary' | 'secondary' | 'success' | 'danger'
        """
        style_cfg = self.STYLES.get(style, self.STYLES['primary'])

        super().__init__(parent, text=text, command=command,
                         font=('Microsoft YaHei', 10),
                         bg=style_cfg['bg'], fg=style_cfg['fg'],
                         activebackground=style_cfg['activebackground'],
                         activeforeground='white',
                         relief='flat', cursor='hand2',
                         padx=15, pady=5, **kwargs)

    def set_style(self, style):
        """修改按钮样式（tk.Button不支持config(style=...)）"""
        style_cfg = self.STYLES.get(style, self.STYLES['primary'])
        self.config(
            bg=style_cfg['bg'],
            fg=style_cfg['fg'],
            activebackground=style_cfg['activebackground'],
            activeforeground='white'
        )


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
