import os

# ── Paths ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = BASE_DIR  # 2025MM-capitalbikeshare-tripdata/ dirs are here
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# ── Sampling ───────────────────────────────────────
TOTAL_SAMPLE_SIZE = 20000  # 2万条：更多散客样本可学，20K×17特征×8B=2.7MB，内存安全
RANDOM_SEED = 42

# ── Parallelism ─────────────────────────────────────
# 关键修复：i9-13900HX有32逻辑核心，n_jobs=-1会fork 32个进程
# 每个进程复制数据+SVC cache → 内存爆炸。限制为4个。
N_JOBS = 4
SVC_CACHE_SIZE = 500  # MB，每个SVC实例分配500MB

# ── Train / Test Split ─────────────────────────────
TEST_SIZE = 0.2

# ── Cleaning Thresholds ────────────────────────────
MIN_DURATION_MINUTES = 1
MAX_DURATION_MINUTES = 1440  # 24 hours
MIN_AVG_SPEED_KMH = 0.5
MAX_AVG_SPEED_KMH = 50

# ── GridSearchCV ───────────────────────────────────
CV_FOLDS = 5
SCORING = "f1_macro"  # 用f1_macro替代accuracy：70/30不均衡数据下，
                       # f1_macro平等对待member和casual两个类别，
                       # 迫使模型真正学会区分散客，而非盲猜member

# 扩大C搜���范围：C=0.01是边界 → 加入0.001和0.005探索更小值
# 8个C × 5个gamma × 5-fold = 200 fits
PARAM_GRID = {
    "C": [0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 100],
    "gamma": ["scale", 0.1, 0.05, 0.01, 0.001],
    "kernel": ["rbf"],
}
# 扩展搜索：如果最优C在100（上边界），继续向上搜
PARAM_GRID_EXTENDED = {
    "C": [100, 200, 500, 1000],
    "gamma": ["scale", 0.1, 0.05, 0.01],
    "kernel": ["rbf"],
}

# ── Visualization ──────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
CHINESE_FONT_CANDIDATES = ["SimHei", "Microsoft YaHei", "KaiTi", "FangSong"]

# ── Target Encoding ────────────────────────────────
TARGET_MAP = {"casual": 0, "member": 1}


def ensure_dirs():
    for d in [FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)
