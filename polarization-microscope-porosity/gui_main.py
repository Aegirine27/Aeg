"""
偏光显微镜面孔率识别系统 - GUI 入口

使用方法：
    python gui_main.py

打包为 exe：
    pyinstaller --windowed --onefile --name "面孔率识别系统" gui_main.py
"""
import sys
import os

# 确保 src 目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gui import PorosityGUI


def main():
    """主函数"""
    # 查找配置文件
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        # 尝试 exe 同级目录
        exe_dir = os.path.dirname(sys.executable)
        alt_path = os.path.join(exe_dir, "config.yaml")
        if os.path.exists(alt_path):
            config_path = alt_path

    app = PorosityGUI(config_path=config_path)
    app.run()


if __name__ == '__main__':
    main()
