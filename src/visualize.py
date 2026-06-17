import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from src.config import FIGURE_DPI, CHINESE_FONT_CANDIDATES

_FONT_NAME: str | None = None


def _find_chinese_font() -> str | None:
    """在系统中查找可用的中文字体文件，返回字体名，找不到则返回 None。"""
    # 策略1：用 fm.findfont 查找实际字体文件路径
    for font_name in CHINESE_FONT_CANDIDATES:
        try:
            font_path = fm.findfont(font_name, fallback_to_default=False)
        except Exception:
            continue
        if font_path and os.path.exists(font_path):
            try:
                fm.fontManager.addfont(font_path)
            except Exception:
                pass
            return font_name

    # 策略2：检查 fontManager 中已注册的字体名
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in CHINESE_FONT_CANDIDATES:
        if font_name in available:
            return font_name

    return None


def _apply_chinese_font(font_name: str) -> None:
    """将中文字体应用到 matplotlib 的全局 rcParams 配置。"""
    plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def setup_chinese_font() -> str | None:
    """检测并配置 matplotlib 可用的中文字体，返回字体名。"""
    global _FONT_NAME
    if _FONT_NAME is not None:
        return _FONT_NAME

    font_name = _find_chinese_font()
    if font_name is not None:
        _apply_chinese_font(font_name)
        _FONT_NAME = font_name
        return font_name

    print("WARNING: No Chinese font found. Chinese text may render as boxes.")
    _FONT_NAME = None
    return None


def set_style():
    """应用 seaborn 全局样式，然后重新覆盖中文字体。

    重要：sns.set_style() 内部会重置 font.sans-serif 为
    ['Arial', ...]，覆盖我们的中文字体设置。因此必须在
    seaborn 设置完成之后重新应用中文字体。
    """
    global _FONT_NAME
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    # 如果已检测到中文字体则重新应用（seaborn 会重置字体）
    if _FONT_NAME is not None:
        _apply_chinese_font(_FONT_NAME)
    elif not hasattr(set_style, "_first_call"):
        # 首次调用时，在 seaborn 重置字体后重新检测
        set_style._first_call = True
        font_name = _find_chinese_font()
        if font_name is not None:
            _apply_chinese_font(font_name)
            _FONT_NAME = font_name


def save_figure(fig: plt.Figure, filename: str, output_dir: str | None = None) -> str:
    """以 300 DPI 高分辨率保存图片，返回文件路径。"""
    from src.config import FIGURES_DIR

    if output_dir is None:
        output_dir = FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return filepath


def get_font_name() -> str | None:
    """返回已检测到的中文字体名，尚未检测则触发检测。"""
    if _FONT_NAME is None:
        setup_chinese_font()
    return _FONT_NAME
