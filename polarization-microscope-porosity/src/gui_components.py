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
import numpy as np


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
        - 支持画笔标注模式（用于修正mask）
    """

    def __init__(self, parent, title="图像", width=400, height=400, click_callback=None, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg='#E8E8E8', highlightthickness=1,
                         highlightbackground=COLORS['border'], **kwargs)

        self.title = title
        self.canvas_width = width
        self.canvas_height = height
        self.current_image = None  # 当前显示的 PIL Image（原始尺寸）
        self.photo_image = None    # PhotoImage 引用（防止GC）
        self.click_callback = click_callback  # 鼠标点击回调
        self.display_scale = 1.0   # 基础自适应缩放（让图像适应画布）
        self.user_zoom = 1.0       # 用户额外缩放（滚轮控制）
        self.pan_x = 0             # 平移偏移X（中键拖拽）
        self.pan_y = 0             # 平移偏移Y
        self._pan_dragging = False
        self._pan_drag_start = (0, 0)

        # 标注模式
        self.annotate_mode = False
        self.brush_size = 10
        self.annotation_color = 255
        self.annotation_mask = None
        self._drawing = False

        # 叠加层显示
        self._base_cv_image = None
        self._annotation_overlay_photo = None

        # 标题文字
        self.create_text(width//2, height//2, text=f"{title}\n(暂无图像)",
                         fill=COLORS['text_secondary'], font=('Microsoft YaHei', 12),
                         justify='center')

        # 绑定鼠标事件
        if self.click_callback:
            self.bind('<Button-1>', self._on_click)
        self.bind('<Button-1>', self._on_brush_down, add='+')
        self.bind('<B1-Motion>', self._on_brush_drag)
        self.bind('<ButtonRelease-1>', self._on_brush_up)
        # 滚轮缩放
        self.bind('<MouseWheel>', self._on_mousewheel)
        # 中键/右键拖拽平移
        self.bind('<Button-2>', self._on_pan_down)
        self.bind('<B2-Motion>', self._on_pan_drag)
        self.bind('<ButtonRelease-2>', self._on_pan_up)
        self.bind('<Button-3>', self._on_pan_down)
        self.bind('<B3-Motion>', self._on_pan_drag)
        self.bind('<ButtonRelease-3>', self._on_pan_up)

    def _get_image_display_rect(self):
        """获取图像在画布中的显示矩形 (x, y, w, h)，考虑缩放和平移"""
        if self.current_image is None:
            return 0, 0, 0, 0
        total_scale = self.display_scale * self.user_zoom
        img_w = int(self.current_image.width * total_scale)
        img_h = int(self.current_image.height * total_scale)
        x = (self.canvas_width - img_w) // 2 + self.pan_x
        y = (self.canvas_height - img_h) // 2 + self.pan_y
        return x, y, img_w, img_h

    def _get_image_coords(self, event_x, event_y):
        """将画布坐标转换为原始图像坐标（考虑缩放和平移）"""
        if self.current_image is None:
            return None, None
        x, y, w, h = self._get_image_display_rect()
        if x <= event_x <= x + w and y <= event_y <= y + h:
            total_scale = self.display_scale * self.user_zoom
            img_x = int((event_x - x) / total_scale)
            img_y = int((event_y - y) / total_scale)
            img_x = max(0, min(img_x, self.current_image.width - 1))
            img_y = max(0, min(img_y, self.current_image.height - 1))
            return img_x, img_y
        return None, None

    def _on_mousewheel(self, event):
        """滚轮缩放"""
        if hasattr(event, 'delta'):
            delta = event.delta
        else:
            delta = 120 if event.num == 4 else -120
        factor = 1.1 if delta > 0 else 0.9
        self._set_user_zoom(self.user_zoom * factor)

    def _set_user_zoom(self, zoom):
        """设置用户缩放，以画布中心为锚点"""
        old_zoom = self.user_zoom
        self.user_zoom = max(0.1, min(10.0, zoom))
        if self.current_image is not None:
            # 以画布中心为锚点调整平移
            scale_ratio = self.user_zoom / old_zoom
            self.pan_x = int(self.pan_x * scale_ratio)
            self.pan_y = int(self.pan_y * scale_ratio)
            self._refresh_display()

    def zoom_in(self):
        self._set_user_zoom(self.user_zoom * 1.2)

    def zoom_out(self):
        self._set_user_zoom(self.user_zoom / 1.2)

    def reset_zoom(self):
        self.user_zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._refresh_display()

    def fit_zoom(self):
        self.user_zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._refresh_display()

    def _on_pan_down(self, event):
        """中键/右键按下开始平移"""
        self._pan_dragging = True
        self._pan_drag_start = (event.x, event.y)
        self.config(cursor='fleur')

    def _on_pan_drag(self, event):
        """中键/右键拖拽平移"""
        if self._pan_dragging:
            dx = event.x - self._pan_drag_start[0]
            dy = event.y - self._pan_drag_start[1]
            self.pan_x += dx
            self.pan_y += dy
            self._pan_drag_start = (event.x, event.y)
            self._refresh_display()

    def _on_pan_up(self, event):
        """中键/右键释放结束平移"""
        self._pan_dragging = False
        if self.annotate_mode:
            self.config(cursor='circle')
        else:
            self.config(cursor='' if self.click_callback else '')

    def _on_click(self, event):
        """处理鼠标点击，将画布坐标转换为图像坐标"""
        img_x, img_y = self._get_image_coords(event.x, event.y)
        if img_x is not None and self.click_callback is not None:
            self.click_callback(img_x, img_y)

    def _on_brush_down(self, event):
        if not self.annotate_mode or self._pan_dragging:
            return
        self._drawing = True
        self._draw_brush(event.x, event.y)

    def _on_brush_drag(self, event):
        if not self.annotate_mode or not self._drawing or self._pan_dragging:
            return
        self._draw_brush(event.x, event.y)

    def _on_brush_up(self, event):
        self._drawing = False

    def _draw_brush(self, canvas_x, canvas_y):
        """在指定位置画笔画"""
        img_x, img_y = self._get_image_coords(canvas_x, canvas_y)
        if img_x is None:
            return

        # 在mask上绘制
        if self.annotation_mask is not None:
            h, w = self.annotation_mask.shape
            if 0 <= img_x < w and 0 <= img_y < h:
                cv2.circle(self.annotation_mask, (img_x, img_y), self.brush_size,
                          self.annotation_color, -1)

        # 实时刷新mask叠加层（让用户立即看到mask变化）
        self._update_annotation_overlay()

    def set_annotate_mode(self, enabled, brush_size=10, annotation_color=255):
        """设置标注模式

        Args:
            enabled: True=启用标注模式, False=禁用
            brush_size: 画笔大小（像素）
            annotation_color: 255=前景(白色), 0=背景(黑色)
        """
        self.annotate_mode = enabled
        self.brush_size = brush_size
        self.annotation_color = annotation_color

        if enabled:
            self.config(cursor='circle')
            if self.current_image is not None:
                h, w = self.current_image.height, self.current_image.width
                if self.annotation_mask is None or self.annotation_mask.shape != (h, w):
                    self.annotation_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            self.config(cursor='' if self.click_callback else '')

    def _update_annotation_overlay(self):
        """叠加显示annotation_mask（考虑用户缩放和平移）"""
        self.delete('annotation_overlay')
        self._annotation_overlay_photo = None
        if self.annotation_mask is None or self.current_image is None:
            return
        if self.annotation_mask.sum() == 0:
            return
        try:
            x, y, w, h = self._get_image_display_rect()
            if w <= 0 or h <= 0:
                return
            display_mask = cv2.resize(self.annotation_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            overlay_rgba[display_mask > 0] = [255, 255, 0, 220]
            overlay_pil = Image.fromarray(overlay_rgba, 'RGBA')
            self._annotation_overlay_photo = ImageTk.PhotoImage(overlay_pil)
            self.create_image(x + w // 2, y + h // 2, image=self._annotation_overlay_photo,
                              anchor='center', tags='annotation_overlay')
            # 红色轮廓
            contours, _ = cv2.findContours(display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                points = [(int(pt[0][0] + x), int(pt[0][1] + y)) for pt in cnt]
                if len(points) > 2:
                    self.create_polygon(points, outline='#FF0000', fill='',
                                        width=1, tags='annotation_overlay')
        except Exception as e:
            pass

    def _refresh_display(self):
        """刷新显示（应用用户缩放和平移）"""
        if self.current_image is None or self._base_cv_image is None:
            return
        try:
            total_scale = self.display_scale * self.user_zoom
            total_w = int(self.current_image.width * total_scale)
            total_h = int(self.current_image.height * total_scale)
            total_w = max(1, total_w)
            total_h = max(1, total_h)
            rgb = cv2.cvtColor(self._base_cv_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            resized = pil_img.resize((total_w, total_h), Image.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(resized)
            x = (self.canvas_width - total_w) // 2 + self.pan_x
            y = (self.canvas_height - total_h) // 2 + self.pan_y
            self.delete("all")
            self.create_image(x + total_w // 2, y + total_h // 2,
                              image=self.photo_image, anchor='center', tags='image')
            # 标题
            self.create_text(10, 10, text=self.title, anchor='nw',
                             fill=COLORS['primary'], font=('Microsoft YaHei', 10, 'bold'),
                             tags='title')
            # 缩放提示
            zoom_text = f"缩放: {total_scale:.2f}x | 滚轮缩放 | 中键拖拽"
            self.create_text(self.canvas_width - 10, 10, text=zoom_text, anchor='ne',
                             fill=COLORS['accent'], font=('Microsoft YaHei', 8),
                             tags='zoom_hint')
            # 模式提示
            hint_text = None
            if self.annotate_mode:
                hint_text = "画笔标注模式"
            elif self.click_callback:
                hint_text = "点击图像取色"
            if hint_text:
                self.create_text(self.canvas_width//2, self.canvas_height - 15,
                                 text=hint_text, anchor='s',
                                 fill=COLORS['accent'], font=('Microsoft YaHei', 9),
                                 tags='hint')
            self._update_annotation_overlay()
        except Exception as e:
            print(f"ImageViewer刷新失败: {e}")

    def clear_annotation(self):
        """清空标注"""
        self.annotation_mask = None
        # 删除叠加层
        self.delete('annotation_overlay')
        self._annotation_overlay_photo = None
        # 重新显示基础图像（去除mask叠加）
        if self._base_cv_image is not None:
            self.show_image(self._base_cv_image.copy())

    def get_annotation_mask(self):
        """获取标注mask"""
        return self.annotation_mask

    def load_annotation_mask(self, mask):
        """加载已有mask进行修正"""
        if self.current_image is None:
            return
        self.annotation_mask = mask.copy()
        # 刷新叠加层显示
        self._update_annotation_overlay()

    def show_image(self, cv_image):
        """显示 OpenCV BGR 图像"""
        if cv_image is None:
            self.clear()
            return
        if len(cv_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv_image
        pil_image = Image.fromarray(rgb_image)
        self.current_image = pil_image
        self._base_cv_image = cv_image.copy()
        # 计算基础自适应缩放
        canvas_w, canvas_h = self.canvas_width - 20, self.canvas_height - 40
        img_w, img_h = pil_image.size
        if img_w > 0 and img_h > 0:
            self.display_scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        else:
            self.display_scale = 1.0
        # 重置用户缩放和平移
        self.user_zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._refresh_display()

    def _fit_image(self, pil_image):
        """按比例缩放图像以适应画布，返回缩放后的图像和缩放比例"""
        img_w, img_h = pil_image.size
        canvas_w, canvas_h = self.canvas_width - 20, self.canvas_height - 40

        if img_w == 0 or img_h == 0:
            return pil_image, 1.0

    def clear(self):
        """清空显示"""
        self.delete("all")
        self.current_image = None
        self.photo_image = None
        self._base_cv_image = None
        self.annotation_mask = None
        self.user_zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.create_text(self.canvas_width//2, self.canvas_height//2,
                         text=f"{self.title}\n(暂无图像)",
                         fill=COLORS['text_secondary'], font=('Microsoft YaHei', 12),
                         justify='center')


class ZoomWindow(tk.Toplevel):
    """
    放大查看窗口（用于精确取色和精确标注）

    功能：
        - 滑块/滚轮调整缩放比例（0.5x - 5.0x）
        - 中键拖拽平移视口
        - 左键点击取色（取色模式）
        - 左键涂抹标注（标注模式）
        - 实时显示当前坐标和RGB值
    """

    def __init__(self, parent, cv_image, mode='view', click_callback=None,
                 annotation_mask=None, brush_size=10, annotation_color=255,
                 title="放大查看"):
        """
        Args:
            parent: 父窗口
            cv_image: OpenCV BGR 图像
            mode: 'view'查看 / 'pick'取色 / 'annotate'标注
            click_callback: 取色回调函数(img_x, img_y)
            annotation_mask: 当前标注mask（标注模式用）
            brush_size: 画笔大小
            annotation_color: 255=前景, 0=背景
            title: 窗口标题
        """
        super().__init__(parent)
        self.title(title)
        self.geometry("900x700")
        self.minsize(600, 400)

        self.cv_image = cv_image
        self.mode = mode
        self.click_callback = click_callback
        self.brush_size = brush_size
        self.annotation_color = annotation_color
        self._dragging = False
        self._drag_start = (0, 0)
        self._view_offset = [0, 0]  # 视口偏移（图像坐标系）

        # 图像尺寸
        self.img_h, self.img_w = cv_image.shape[:2]

        # 计算初始缩放（让图像适应窗口）
        self.zoom_scale = min(800 / self.img_w, 550 / self.img_h, 1.0)
        self.zoom_scale = max(self.zoom_scale, 0.5)

        # 标注模式相关
        self.annotation_mask = annotation_mask.copy() if annotation_mask is not None else None
        self._drawing = False

        self._build_ui()
        self._refresh_display()

        # 绑定快捷键
        self.bind('<Escape>', lambda e: self.destroy())
        self.bind('<MouseWheel>', self._on_mousewheel)  # Windows
        self.bind('<Button-4>', self._on_mousewheel)     # Linux向上
        self.bind('<Button-5>', self._on_mousewheel)     # Linux向下

    def _build_ui(self):
        """构建UI"""
        # 顶部信息栏
        info_frame = tk.Frame(self, bg=COLORS['bg_panel'], height=35)
        info_frame.pack(fill='x', padx=5, pady=5)
        info_frame.pack_propagate(False)

        self.coord_label = tk.Label(info_frame, text="坐标: --",
                                     bg=COLORS['bg_panel'], fg=COLORS['text'],
                                     font=('Microsoft YaHei', 9))
        self.coord_label.pack(side='left', padx=10)

        self.rgb_label = tk.Label(info_frame, text="RGB: --",
                                   bg=COLORS['bg_panel'], fg=COLORS['text'],
                                   font=('Microsoft YaHei', 9))
        self.rgb_label.pack(side='left', padx=10)

        self.scale_label = tk.Label(info_frame, text=f"缩放: {self.zoom_scale:.1f}x",
                                     bg=COLORS['bg_panel'], fg=COLORS['accent'],
                                     font=('Microsoft YaHei', 9, 'bold'))
        self.scale_label.pack(side='right', padx=10)

        # 模式提示
        mode_text = {
            'pick': '【取色模式】点击图像取色',
            'annotate': '【标注模式】左键涂抹，中键拖拽平移，滚轮缩放',
            'view': '【查看模式】中键拖拽平移，滚轮缩放'
        }.get(self.mode, '')
        tk.Label(info_frame, text=mode_text,
                 bg=COLORS['bg_panel'], fg=COLORS['success'],
                 font=('Microsoft YaHei', 9)).pack(side='left', padx=10)

        # Canvas显示区域
        canvas_frame = tk.Frame(self, bg=COLORS['border'])
        canvas_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg='#2C2C2C',
                                highlightthickness=1,
                                highlightbackground=COLORS['border'])
        self.canvas.pack(fill='both', expand=True)

        # Canvas实际尺寸在窗口显示后才能获取，这里用默认值
        self._canvas_width = 880
        self._canvas_height = 570

        # 绑定Canvas事件
        self.canvas.bind('<Motion>', self._on_mouse_move)
        self.canvas.bind('<Button-1>', self._on_left_click)
        self.canvas.bind('<B1-Motion>', self._on_left_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_left_release)
        self.canvas.bind('<Button-2>', self._on_middle_down)      # 中键按下（Linux）
        self.canvas.bind('<B2-Motion>', self._on_middle_drag)
        self.canvas.bind('<Button-3>', self._on_middle_down)      # 右键按下（Windows/Mac用作中键替代）
        self.canvas.bind('<B3-Motion>', self._on_middle_drag)
        self.canvas.bind('<ButtonRelease-2>', self._on_middle_up)
        self.canvas.bind('<ButtonRelease-3>', self._on_middle_up)

        # 底部控制栏
        ctrl_frame = tk.Frame(self, bg=COLORS['bg_panel'], height=50)
        ctrl_frame.pack(fill='x', padx=5, pady=5)
        ctrl_frame.pack_propagate(False)

        tk.Label(ctrl_frame, text="缩放:", bg=COLORS['bg_panel'],
                 fg=COLORS['text'], font=('Microsoft YaHei', 10)).pack(side='left', padx=10)

        self.zoom_slider = tk.Scale(ctrl_frame, from_=0.5, to=5.0, resolution=0.1,
                                     orient='horizontal', length=400,
                                     command=self._on_zoom_changed)
        self.zoom_slider.set(self.zoom_scale)
        self.zoom_slider.pack(side='left', padx=5)

        # 按钮
        CustomButton(ctrl_frame, text="1:1", style='primary',
                     command=lambda: self._set_zoom(1.0)).pack(side='left', padx=5)
        CustomButton(ctrl_frame, text="适应窗口", style='primary',
                     command=self._fit_zoom).pack(side='left', padx=5)

        if self.mode == 'annotate':
            CustomButton(ctrl_frame, text="清空标注", style='danger',
                         command=self._clear_annotation).pack(side='right', padx=10)

        CustomButton(ctrl_frame, text="关闭", style='secondary',
                     command=self.destroy).pack(side='right', padx=5)

    # ============ 显示刷新 ============

    def _get_canvas_size(self):
        """获取Canvas实际尺寸（窗口显示后调用）"""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        # 窗口未显示时winfo返回1，用默认值
        if cw <= 1:
            cw = self._canvas_width
        if ch <= 1:
            ch = self._canvas_height
        return cw, ch

    def _refresh_display(self):
        """刷新Canvas显示"""
        if self.cv_image is None:
            return

        try:
            # 获取Canvas实际尺寸
            canvas_w, canvas_h = self._get_canvas_size()

            # 计算显示尺寸
            disp_w = int(self.img_w * self.zoom_scale)
            disp_h = int(self.img_h * self.zoom_scale)

            # 限制视口偏移，防止图像完全移出画布
            max_offset_x = max(0, disp_w - canvas_w)
            max_offset_y = max(0, disp_h - canvas_h)
            self._view_offset[0] = max(0, min(int(self._view_offset[0]), max_offset_x))
            self._view_offset[1] = max(0, min(int(self._view_offset[1]), max_offset_y))

            # 从原始图像裁剪可视区域
            src_x = int(self._view_offset[0] / self.zoom_scale)
            src_y = int(self._view_offset[1] / self.zoom_scale)
            src_w = min(int(canvas_w / self.zoom_scale) + 1, self.img_w - src_x)
            src_h = min(int(canvas_h / self.zoom_scale) + 1, self.img_h - src_y)

            if src_w <= 0 or src_h <= 0:
                # 显示空白提示
                self.canvas.delete('all')
                self.canvas.create_text(
                    canvas_w // 2, canvas_h // 2,
                    text="视口越界\n请重置缩放",
                    fill='white', font=('Microsoft YaHei', 14),
                    justify='center'
                )
                return

            # 裁剪并缩放
            roi = self.cv_image[src_y:src_y+src_h, src_x:src_x+src_w]
            target_w = min(int(src_w * self.zoom_scale), canvas_w)
            target_h = min(int(src_h * self.zoom_scale), canvas_h)
            display_roi = cv2.resize(roi, (target_w, target_h),
                                      interpolation=cv2.INTER_LINEAR)

            # 如果有标注mask，叠加显示
            if self.annotation_mask is not None and self.mode == 'annotate':
                mask_roi = self.annotation_mask[src_y:src_y+src_h, src_x:src_x+src_w]
                mask_display = cv2.resize(mask_roi, (display_roi.shape[1], display_roi.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                overlay = display_roi.copy()
                overlay[mask_display > 0] = [0, 255, 255]  # BGR黄色
                display_roi = cv2.addWeighted(display_roi, 0.5, overlay, 0.5, 0)

            # 转为PIL并显示
            rgb = cv2.cvtColor(display_roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            self._photo = ImageTk.PhotoImage(pil_img)

            self.canvas.delete('all')
            self.canvas.create_image(0, 0, image=self._photo, anchor='nw', tags='image')

            # 绘制视口边框（如果图像小于画布）
            if disp_w < canvas_w or disp_h < canvas_h:
                self.canvas.create_rectangle(0, 0, disp_w, disp_h, outline='#FF0000', width=2)

        except Exception as e:
            print(f"ZoomWindow刷新失败: {e}")
            import traceback
            traceback.print_exc()

    def _set_zoom(self, scale):
        """设置缩放比例（以画布中心为锚点）"""
        old_scale = self.zoom_scale
        if old_scale <= 0:
            old_scale = 0.5
        self.zoom_scale = max(0.5, min(5.0, scale))

        # 调整偏移，以画布中心为缩放中心
        canvas_w, canvas_h = self._get_canvas_size()
        cx = canvas_w / 2
        cy = canvas_h / 2
        self._view_offset[0] = int((self._view_offset[0] + cx) * (self.zoom_scale / old_scale) - cx)
        self._view_offset[1] = int((self._view_offset[1] + cy) * (self.zoom_scale / old_scale) - cy)

        self.zoom_slider.set(self.zoom_scale)
        self.scale_label.config(text=f"缩放: {self.zoom_scale:.1f}x")
        self._refresh_display()

    def _fit_zoom(self):
        """适应窗口"""
        canvas_w, canvas_h = self._get_canvas_size()
        self._view_offset = [0, 0]
        self.zoom_scale = min(canvas_w / self.img_w,
                              canvas_h / self.img_h, 1.0)
        self.zoom_scale = max(self.zoom_scale, 0.5)
        self.zoom_slider.set(self.zoom_scale)
        self.scale_label.config(text=f"缩放: {self.zoom_scale:.1f}x")
        self._refresh_display()

    # ============ 事件处理 ============

    def _on_mousewheel(self, event):
        """滚轮缩放"""
        if hasattr(event, 'delta'):
            # Windows/Mac
            delta = event.delta
        else:
            # Linux
            delta = 120 if event.num == 4 else -120

        factor = 1.1 if delta > 0 else 0.9
        self._set_zoom(self.zoom_scale * factor)

    def _on_zoom_changed(self, value):
        """滑块缩放"""
        self._set_zoom(float(value))

    def _on_mouse_move(self, event):
        """鼠标移动，更新坐标和RGB显示"""
        img_x, img_y = self._canvas_to_image(event.x, event.y)
        if img_x is not None:
            self.coord_label.config(text=f"坐标: ({img_x}, {img_y})")
            b, g, r = self.cv_image[img_y, img_x]
            self.rgb_label.config(text=f"RGB: ({r}, {g}, {b})")
        else:
            self.coord_label.config(text="坐标: --")
            self.rgb_label.config(text="RGB: --")

    def _on_left_click(self, event):
        """左键点击/开始涂抹"""
        img_x, img_y = self._canvas_to_image(event.x, event.y)
        if img_x is None:
            return

        if self.mode == 'pick' and self.click_callback:
            self.click_callback(img_x, img_y)
            self.destroy()  # 取色后自动关闭
        elif self.mode == 'annotate':
            self._drawing = True
            self._draw_brush(img_x, img_y)

    def _on_left_drag(self, event):
        """左键拖拽涂抹"""
        if self.mode == 'annotate' and self._drawing:
            img_x, img_y = self._canvas_to_image(event.x, event.y)
            if img_x is not None:
                self._draw_brush(img_x, img_y)

    def _on_left_release(self, event):
        """左键释放"""
        self._drawing = False

    def _on_middle_down(self, event):
        """中键/右键按下开始拖拽"""
        self._dragging = True
        self._drag_start = (event.x, event.y)

    def _on_middle_drag(self, event):
        """中键/右键拖拽平移"""
        if self._dragging:
            dx = self._drag_start[0] - event.x
            dy = self._drag_start[1] - event.y
            self._view_offset[0] += dx
            self._view_offset[1] += dy
            self._drag_start = (event.x, event.y)
            self._refresh_display()

    def _on_middle_up(self, event):
        """中键/右键释放"""
        self._dragging = False

    # ============ 工具方法 ============

    def _canvas_to_image(self, canvas_x, canvas_y):
        """将Canvas坐标转换为原始图像坐标"""
        img_x = int((canvas_x + self._view_offset[0]) / self.zoom_scale)
        img_y = int((canvas_y + self._view_offset[1]) / self.zoom_scale)
        if 0 <= img_x < self.img_w and 0 <= img_y < self.img_h:
            return img_x, img_y
        return None, None

    def _draw_brush(self, img_x, img_y):
        """在mask上绘制"""
        if self.annotation_mask is not None:
            cv2.circle(self.annotation_mask, (img_x, img_y), self.brush_size,
                      self.annotation_color, -1)
            self._refresh_display()

    def _clear_annotation(self):
        """清空标注"""
        if self.annotation_mask is not None:
            self.annotation_mask.fill(0)
            self._refresh_display()

    def get_annotation_mask(self):
        """获取标注后的mask"""
        return self.annotation_mask


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
