# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

这是"机器学习实验课"课程项目 — 使用 SVM（支持向量机）对华盛顿 Capital Bikeshare 共享单车用户类型进行二分类预测（member vs casual）。

# Machine Learning Course Project

## Project Goal

Build a high-quality machine learning experiment based on Support Vector Machine (SVM) to classify Capital Bikeshare users into:

- Member
- Casual

The final project must maximize:

- Prediction accuracy
- Experimental rigor
- Report quality
- Code readability

This project is evaluated as a university machine learning laboratory assignment.

---

# Scoring Priorities

The report and code should be designed to achieve the highest possible score.

Evaluation criteria:

- Model Accuracy: 40%
- Report Quality: 40%
- Code Quality: 20%

Therefore:

DO NOT only train an SVM.

The project must include:

1. Complete data cleaning
2. Feature engineering
3. Exploratory Data Analysis (EDA)
4. SVM optimization
5. Model evaluation
6. Feature importance analysis
7. Interesting findings
8. Publication-quality visualizations
9. Reproducible pipeline

---

# Dataset Description

Dataset:

Capital Bikeshare trip records from Washington D.C.

Period:

January 2025 – December 2025

Target Variable:

User Type

Binary classification:

- Member
- Casual

---

# Required Workflow

Follow the workflow strictly.

## Step 1 Data Sampling

The original dataset is large.

Create a representative dataset by:

### Monthly Stratified Sampling

For each month:

1. Separate Member and Casual users
2. Sample proportionally
3. Keep class distribution balanced

Requirements:

- Explain sampling strategy
- Record sample size for each month
- Generate table of monthly sample counts

Create visualization:

- Monthly sample distribution
- User type distribution

---

## Step 2 Data Cleaning

Perform complete preprocessing.

### Missing Values

Detect and report:

- Missing count
- Missing percentage

Apply suitable methods:

- Median imputation
- Mode imputation
- Removal if necessary

### Duplicate Records

Detect and remove duplicates.

### Outlier Detection

Analyze:

- Trip duration
- Ride distance (if available)

Methods:

- IQR
- Boxplot

Create before/after comparison plots.

---

## Step 3 Feature Engineering

Create meaningful features.

### Time Features

From start time:

Generate:

- Hour (cyclic: sin/cos)
- Day of Week (cyclic: sin/cos)
- Month (cyclic: sin/cos)
- Is Weekend
- Is Morning Rush (7-9h, weekday)
- Is Evening Rush (16-19h, weekday)

### Trip Features

Generate:

- Duration (log-transformed)
- Haversine Distance (km)
- Is Electric Bike

### Location Features

Use:

- Start Station frequency encoding
- End Station frequency encoding
- Has Start Station (binary)
- Has End Station (binary)

Apply:

- Frequency Encoding

Avoid extremely high-dimensional one-hot encoding.

### Composite Features

- Average Speed (km/h), derived from distance / duration

### Current Feature List (17 features)

```
hour_sin, hour_cos, dayofweek_sin, dayofweek_cos, month_sin, month_cos,
is_weekend, is_morning_rush, is_evening_rush, duration_minutes_log,
haversine_distance_km, is_electric_bike, has_start_station, has_end_station,
start_station_freq, end_station_freq, avg_speed_kmh
```

---

## Step 4 Exploratory Data Analysis

This section is extremely important.

Produce high-quality figures.

### 15 EDA Figures

1. User type distribution (pie chart)
2. Monthly ride volume (stacked bar)
3. Hourly usage patterns (line chart)
4. Day-of-week patterns (grouped bar)
5. Seasonal usage (dual-axis: bar + line)
6. Duration distribution KDE (member vs casual)
7. Distance distribution KDE (member vs casual)
8. Average speed boxplot (member vs casual)
9. Rush hour split (grouped bar)
10. Top 15 start stations (horizontal bar)
11. Bike type preference (grouped bar)
12. Dockless ride proportion (grouped bar)
13. Feature correlation heatmap
14. Duration vs Distance scatter
15. Monthly class balance trend

---

# High Score Requirement

Generate at least 12–15 figures.

Every figure must be:

- Saved separately
- High resolution
- Publication style

Minimum DPI:

300

Use:

plt.savefig(..., dpi=300)

---

# Step 5 Feature Selection

Use multiple methods.

## Method 1

Mutual Information

## Method 2

Chi-Square (after MinMax scaling to [0,1])

## Method 3

Random Forest Importance (150 trees, max_depth=10)

Even though SVM is required for classification,
Random Forest may be used ONLY for feature importance analysis.

### Ensemble Ranking

Average normalized ranks from all 3 methods to produce consensus importance ranking.

Generate ranking charts.

Select Top Features.

---

# Step 6 Data Standardization

SVM requires scaling.

Apply:

StandardScaler

Pipeline:

Train Set

→ Fit Scaler

Test Set

→ Transform Only

Prevent data leakage.

---

# Step 7 Train-Test Split

Use:

80% Training

20% Testing

Maintain class balance:

stratify=y

Random Seed:

42

---

# Step 8 SVM Modeling

Build multiple SVM models.

## Model A

LinearSVC (dual=False, max_iter=5000)

## Model B

SVC-Poly (degree=2)

## Model C

SVC-RBF (gamma='scale')

All with class_weight='balanced'.

Compare all models.

---

# Step 9 Hyperparameter Optimization

Use GridSearchCV with 5-fold CV.

### Current Search Grid

```
C:      [0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 100]   (8 values)
gamma:  ['scale', 0.1, 0.05, 0.01, 0.001]               (5 values)
kernel: ['rbf']

Total: 8 × 5 × 5 = 200 fits
```

### Scoring Metric

`f1_macro` — 平等对待 member 和 casual 两个类别。在 70/30 不均衡数据下，`accuracy` 评分会偏向多数类（member），导致模型只需盲猜 member 即可获得 ~70% 准确率。`f1_macro` 迫使模型真正学会区分散客。

### Extended Search

If best C = 100 (upper boundary), extend upward:
```
C: [100, 200, 500, 1000]
gamma: ['scale', 0.1, 0.05, 0.01]
```

### Output

- Best parameters
- Best CV score
- CV results CSV

---

# Step 10 Evaluation

Generate comprehensive evaluation.

Metrics:

## Accuracy

## Precision (per-class + weighted)

## Recall (per-class + weighted)

## F1 Score (per-class + weighted)

## ROC-AUC

Generate:

### Confusion Matrix (raw + normalized)

### ROC Curve

### Precision-Recall Curve

### Classification Report Heatmap

### Learning Curve

All figures saved.

---

# Step 11 Feature Importance Analysis

Because SVM lacks direct interpretability:

Use:

Permutation Importance (n_repeats=20)

Plus:

Logistic Regression Coefficients (auxiliary interpretability)

Analyze:

Top 15 Features

Create:

- Permutation importance bar chart (with error bars)
- Logistic regression coefficient chart (positive/negative direction)

Discuss:

Which factors most influence user type.

This section contributes significantly to report score.

---

# Step 12 Interesting Findings

Actively search for discoveries. All findings backed by statistical tests.

### Finding 1: Commuting Patterns
- Test: Chi-square test on rush-hour vs non-rush crosstab
- Finding: Members concentrate on commuting hours (statistically significant)

### Finding 2: Duration Difference
- Test: Mann-Whitney U test
- Finding: Casual riders have significantly longer ride durations

### Finding 3: Weekend Effect
- Test: Chi-square test on weekend vs weekday crosstab
- Finding: Casual proportion is significantly higher on weekends

### Finding 4: Dockless Pattern
- Test: Chi-square test
- Finding: Dockless (no station) rides have higher casual proportion

### Finding 5: Seasonal Shift
- Analysis: Monthly casual proportion by season
- Finding: Casual proportion peaks in summer, troughs in winter

### Finding 6: Electric Bike Preference
- Test: Chi-square test
- Finding: Casual users prefer electric bikes more than members

---

# Code Requirements

## Actual Project Structure

```
机器学习实验课/
├── main.py                          # 主入口，完整 12 步流程编排
├── src/
│   ├── __init__.py
│   ├── config.py                    # 路径、超参数、内存优化配置
│   ├── data_loader.py              # CSV 文件发现与月度加载
│   ├── sampling.py                 # 分层抽样（chunksize=100K 分块加载）
│   ├── cleaning.py                 # 数据清洗（DST修复、IQR截断、Haversine距离）
│   ├── feature_engineering.py      # 特征工程（时间循环编码、频率编码）
│   ├── eda.py                      # 探索性数据分析（15 张图）
│   ├── feature_selection.py        # 特征选择（MI、Chi2、RF + 集成排名）
│   ├── preprocessing.py            # 标准化 + 训练/测试划分
│   ├── train.py                    # SVM 建模 + GridSearchCV
│   ├── evaluate.py                 # 模型评估（混淆矩阵、ROC、PR、学习曲线）
│   ├── feature_analysis.py         # 特征重要性（Permutation + Logistic Regression）
│   ├── findings.py                 # 统计发现（卡方检验、Mann-Whitney U）
│   └── visualize.py                # 中文字体、seaborn 样式、图片保存
├── outputs/
│   ├── figures/                    # 所有图表（300 DPI PNG）
│   ├── tables/                     # 所有表格（CSV）
│   └── models/                     # 训练好的模型 + scaler（joblib）
├── 2025MM-capitalbikeshare-tripdata/  # 12 个月原始 CSV 数据
└── report/                         # 实验报告
```

## Code Style Requirements

- Type hints
- Function documentation
- Modular design
- Clear variable names

Avoid notebook-only implementation.

---

# Bug Fixes & Optimization History

> 本章节记录了从原始代码到当前版本的每一次重要修改、发现的 Bug 及其根因、以及优化决策的依据。

---

## 第一轮：内存爆炸修复 (2025-06-14)

### 问题现象

在 i9-13900HX (32 线程) + 32GB RAM + RTX 4080 Laptop 上运行 `main.py`，内存占用爆满，系统卡死。

### 根因分析

**四个内存杀手叠加**：

#### 杀手 1：`n_jobs=-1` fork 炸弹（最主要原因）

```python
# 原代码
GridSearchCV(svc, PARAM_GRID, cv=5, n_jobs=-1)
```

i9-13900HX 有 32 个逻辑核心，`n_jobs=-1` 告诉 sklearn 使用全部核心。joblib 会 fork 32 个子进程，每个子进程通过 copy-on-write 复制父进程内存。但 Python 的引用计数会导致 COW 几乎立即触发实际内存复制：

```
父进程内存（已加载训练数据）: ~1-2 GB
32 个子进程各 fork 一份:      32 × 1GB ≈ 32 GB（已接近物理内存上限）
```

GridSearchCV 默认 `pre_dispatch='2*n_jobs'` = 64 个任务预分配，进一步加剧了问题。

#### 杀手 2：`cache_size=2000` 的 SVC 内核缓存

```python
# 原代码 — 6 处 SVC 实例化均使用 cache_size=2000
SVC(cache_size=2000)  # 2000 MB = 2 GB per SVC instance
```

GridSearchCV 期间，最多 32 个 SVC 实例同时拟合，每个分配 2GB 内核缓存：

```
峰值缓存占用: 32 × 2 GB = 64 GB ← 仅缓存就超过物理内存
```

SVM 的核矩阵本身也占大量内存：n 个样本的 RBF 核矩阵为 O(n²)。

#### 杀手 3：全量 CSV 加载

```python
# 原 sampling.py
for month, path in files.items():
    df = load_single_month(path)  # 加载整个 70 万行 CSV
    # 然后从中只取 ~5000 行...
```

单月 70 万行 × 13 列（含长字符串的站点名）DataFrame 内存占用约 1.2 GB。虽然每月加载后会被 GC，但 Python GC 不及时回收，可能和下个月的加载叠加。

#### 杀手 4：其他并行热点

- `learning_curve(n_jobs=-1)` — 又 32 进程
- `RandomForestClassifier(n_estimators=200, n_jobs=-1)` — 200 棵树 × 32 进程
- `permutation_importance(n_repeats=30, n_jobs=-1)` — 30 轮 × 32 进程

### 内存时间线还原

```
阶段                        估算内存    备注
──────────────────────────────────────────────────
启动                         0.5 GB    Python + imports
Step 1 加载 CSV (单月)       1.5 GB    710K行 DataFrame
Step 3 特征工程              0.8 GB    X + y + df_clean
Step 5 RF 200树×32进程       14 GB     峰值
        ↓ GC回收后            2 GB
Step 8 Baseline SVM          5 GB      SVC(cache=2GB) × 3
Step 9 GridSearchCV          64 GB ←  💀 32进程 × 2GB cache
        物理内存不足 →
        Windows swap →
        磁盘 I/O 100% →
        系统卡死
──────────────────────────────────────────────────
```

### 修改方案

| 文件 | 修改项 | 旧值 | 新值 | 效果 |
|------|--------|------|------|------|
| `config.py` | 新增 N_JOBS | 无 | `4` | 并行进程数 ÷8 |
| `config.py` | 新增 SVC_CACHE_SIZE | 无 | `500` | 每实例缓存 ÷4 |
| `config.py` | TOTAL_SAMPLE_SIZE | `60000` | `40000` | 训练样本 -33% |
| `train.py` | GridSearchCV n_jobs | `-1` | `N_JOBS` | |
| `train.py` | GridSearchCV pre_dispatch | 默认 | `"2*n_jobs"` | 限制预分配 |
| `train.py` | 所有 SVC cache_size | `2000` | `SVC_CACHE_SIZE` | 6 处统一 |
| `evaluate.py` | learning_curve n_jobs | `-1` | `N_JOBS` | |
| `feature_selection.py` | RF n_estimators | `200` | `100` | 树数减半 |
| `feature_selection.py` | RF n_jobs | `-1` | `N_JOBS` | |
| `feature_analysis.py` | permutation n_repeats | `30` | `15` | 轮次减半 |
| `feature_analysis.py` | permutation n_jobs | `-1` | `N_JOBS` | |
| `sampling.py` | 加载方式 | 全量加载 | chunksize=100K 分块 | 单月峰值 1.2GB→200MB |
| `sampling.py` | 列过滤 | 全部列 | usecols=13列 | 减少不必要的数据 |
| `main.py` | 内存监控 | 无 | _mem_mb() + gc.collect() | 每步打印内存 |

### 修改后的峰值估算

```
GridSearchCV 阶段:
  4 进程 × (500MB SVC缓存 + ~200MB 数据副本) = ~2.8 GB
  Python 基础 + DataFrame                     = ~1.5 GB
  总峰值                                       ≈ 4-5 GB  ✅ 在 32GB 中安全
```

---

## 第二轮：GridSearchCV 耗时过长 (2025-06-15)

### 问题现象

第一轮修复后，内存不再爆炸，但 Step 9 (GridSearchCV) 运行了超过 1 小时仍未完成。

### 根因分析

第一轮将样本量从 6 万降到 4 万，**内存安全了但计算量仍然巨大**。

RBF 核 SVM 的训练复杂度是 $O(n^2) \sim O(n^3)$：

```
训练样本: 40,000 × 80% = 32,000 条
参数网格: 5 C × 4 gamma = 20 组合
5-fold CV: 20 × 5 = 100 次 SVC.fit()

单次 SVC.fit() 耗时估算:
  C=0.01  → ~30-60 秒
  C=100   → ~120-300 秒（大 C → 更支持向量 → 收敛极慢）

100 次拟合总耗时:
  C=100 拖慢整体: 可能 >1 小时
```

**核心认知**：SVM 不是神经网络——它不需要海量数据。数据量超过一定阈值后，边际收益极低，但计算成本呈指数增长。

### 修改方案

| 修改项 | 旧值 | 新值 | 理由 |
|--------|------|------|------|
| TOTAL_SAMPLE_SIZE | `40000` | `12000` | SVM 的 O(n²) 决定样本量不宜大 |

### 修改后的耗时

```
训练样本: 9,600 条
100 次 SVC.fit():
  单次拟合: ~5-15 秒
  4 进程并行: 100 × 10s / 4 ≈ 4-5 分钟
  Step 9 总计: ~5-8 分钟  ✅
```

### 结果

- 内存峰值: ~350 MB
- GridSearchCV: ~106 秒
- 总耗时: ~3.6 分钟
- Best params: `C=0.01, gamma=0.1`
- **Test Accuracy: 71.33%**（仅略高于 70% 多数类基线）
- **Casual Recall: 27.31%**（模型几乎不识别散客）

---

## 第三轮：Bug 修复 (2025-06-15)

### Bug 1: Step 12 崩溃 — 列名冲突

**现象**：

```
ValueError: No data; `observed` has size 0.
  at findings.py:146 → analyze_electric_preference()
```

**根因**：`feature_engineering.py` 中创建了一个名为 `rideable_type` 的特征列（值为 0/1），与原始 DataFrame 中的 `rideable_type` 列（值为 "electric_bike"/"classic_bike"）同名。`main.py` 将特征矩阵写回 `df_clean` 时，原始字符串列被覆盖为 0/1 数值：

```python
# feature_engineering.py (原代码)
feats["rideable_type"] = (df["rideable_type"] == "electric_bike").astype(int)
#                                   ↑ 新特征与原始列同名！

# main.py
for i, name in enumerate(feature_names):
    df_clean[name] = X[:, i]   # df_clean["rideable_type"] 被覆盖为 0/1

# findings.py
df["rideable_type"].map({"electric_bike": "电动单车", ...})
# → 全部不匹配 → 全部 NaN → crosstab 为空 → chi2 报错
```

**修复**：`rideable_type` → `is_electric_bike`（避免与原始列名冲突）

**经验教训**：特征命名时避免与原始数据列同名，尤其是特征会被写回原 DataFrame 时。

### Bug 2: tkinter 线程报错

**现象**：程序退出时大量 `RuntimeError: main thread is not in main loop`

**根因**：Windows 下 matplotlib 默认使用 tkinter 后端，程序退出时 tkinter 在主线程外清理资源导致。

**修复**：`main.py` 开头添加 `matplotlib.use("Agg")` 使用非交互后端。

### Bug 3: seaborn boxplot FutureWarning

**现象**：`FutureWarning: Passing palette without assigning hue is deprecated`

**修复**：`sns.boxplot(...)` 添加 `hue="用户类型", legend=False`。

### Bug 4: sklearn SVC(probability=True) 废弃警告

**现象**：`FutureWarning: The probability parameter was deprecated in 1.9`

**修复**：`SVC(probability=True)` → `CalibratedClassifierCV(SVC(), cv=5, ensemble=False)`（Platt scaling 校准）。

### Bug 5: sklearn LogisticRegression penalty 警告

**修复**：移除 `penalty="l2"`（已是新版本默认行为）。

---

## 第四轮：模型精度优化 (2025-06-15)

### 问题分析

第三轮后程序稳定运行，但模型质量不佳：

| 指标 | 值 | 评估 |
|------|-----|------|
| Accuracy | 71.33% | 仅比 70% 基线高 1.3% |
| Casual Recall | 27.31% | 模型几乎不识别散客 |
| Member Recall | 92.03% | 模型几乎全猜 member |
| Best C | 0.01 | **在搜索边界上** |
| GridSearchCV scoring | `accuracy` | 在不均衡数据上偏向多数类 |

模型在"偷懒"——找到一个简单的决策面：把所有样本预测为 member，accuracy 就能有 ~70%。GridSearchCV 用 accuracy 做评分，反而奖励了这种行为。

### 优化方案

| 修改项 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| TOTAL_SAMPLE_SIZE | `12000` | `20000` | 更多散客样本（~5500），给模型更多少数类信息 |
| PARAM_GRID C | `[0.01, 0.1, 1, 10, 100]` | `[0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 100]` | C=0.01 在边界，需探索更小值 |
| PARAM_GRID gamma | `['scale', 0.1, 0.01, 0.001]` | `['scale', 0.1, 0.05, 0.01, 0.001]` | 增加 0.05 粒度 |
| SCORING (新增) | 无（硬编码 accuracy） | `f1_macro` | 平等对待两个类别 |
| PARAM_GRID_EXTENDED C | `[100, 500, 1000]` | `[100, 200, 500, 1000]` | 增加 200 |
| RF n_estimators | `100` | `150` | 样本增加，RF 可以更稳定 |
| Permutation n_repeats | `15` | `20` | 更多重复 → 更稳定的重要性估计 |

### 结果

```
总耗时: 15.3 分钟
内存峰值: 357 MB
GridSearchCV: 200 fits, 10.4 分钟
Best params: C=0.05, gamma=0.1  ← C 不再卡边界

Test Accuracy:     73.27%
F1 weighted:        0.6969
Member Recall:     92.28%
Casual Recall:     27.46%   ← 几乎无改善
ROC-AUC:            0.6928
```

### 关键发现

**即使翻倍样本量、扩大搜索范围、换成 f1_macro 评分，散客召回率仍然只有 27%。这不是参数问题，是特征区分度不够。**

### 根因深度分析

Member 和 Casual 的骑行行为模式高度重叠：

1. **两类用户都在骑共享单车** — 时间、距离、速度的分布有大量重叠区域
2. **Member 有规律但 Casual 也有规律** — 游客在景点附近、休闲骑行者在周末、偶尔通勤者
3. **17 个特征主要描述"如何骑行"** — 而非"为什么骑行"（缺失关键上下文：是否有季票、骑行目的、用户注册时长等）

这其实是一个**有价值的实验发现**：仅靠骑行行为特征（时间、位置、距离、速度）难以高精度区分 member 和 casual，两类用户的行为模式存在本质性的重叠。

### 对实验报告的定位

1. **73.27% > 70% 多数类基线** — 模型确实学到了一些区分模式
2. **散客识别难** — 说明 casual 用户行为高度多样化，不是简单的"非通勤"模式
3. **ROC-AUC 0.69** — 模型有一定排序能力，但区分度有限
4. **特征重要性一致指向** — `duration_minutes_log`、`avg_speed_kmh` 始终是 Top 特征，说明骑行时长和速度是最有区分力的信号
5. 这也解释了为什么共享单车公司很难精确预测用户类型 — 需要更多维度的数据

---

# Hardware & Performance Profile

## 当前配置 (i9-13900HX + RTX 4080 Laptop + 32GB RAM)

| 参数 | 值 | 说明 |
|------|-----|------|
| `TOTAL_SAMPLE_SIZE` | `20000` | 全年 660 万条中抽 2 万，训练集约 1.5 万 |
| `N_JOBS` | `4` | 并行进程数（32 核中只开 4 个，内存安全） |
| `SVC_CACHE_SIZE` | `500` | MB/实例，4 实例峰值仅 2GB |
| `SCORING` | `f1_macro` | 平等对待 member/casual |
| `CV_FOLDS` | `5` | 5 折交叉验证 |
| `TEST_SIZE` | `0.2` | 80/20 划分 |

## 当前性能

| 阶段 | 耗时 | 内存 |
|------|------|------|
| Step 1 抽样 | ~17s | ~243 MB |
| Step 2-7 清洗+特征+EDA+划分 | ~7s | ~357 MB |
| Step 8 基准模型 (3 models) | ~11s | ~357 MB |
| Step 9 GridSearchCV (200 fits) | ~10 min | ~357 MB |
| Step 10 评估 | ~15s | ~357 MB |
| Step 11 特征重要性 | ~3.5 min | ~357 MB |
| Step 12 发现分析 | <1s | ~357 MB |
| **总计** | **~15 min** | **~360 MB 峰值** |

## 如果要进一步调整

- **想提速 (<10 min)** → `TOTAL_SAMPLE_SIZE=15000`, `PARAM_GRID` C 减为 6 个值
- **想提高散客召回** → 考虑构造交互特征（如 weekend × hour, season × station_freq），或尝试不同编码策略
- **显存加速 (RTX 4080 12GB)** → `thundersvm` GPU 加速可尝试，但 sklearn 兼容性需验证
- **内存更紧张（16GB 机器）** → `N_JOBS=2`, `TOTAL_SAMPLE_SIZE=10000`

---

# Important Constraints

Must use:

Support Vector Machine (SVM)

Do NOT replace SVM with:

- XGBoost
- LightGBM
- CatBoost
- Neural Networks

These may only be used for auxiliary analysis if necessary.

The final classification model must be SVM.

---

# Final Deliverables

Generate:

1. Clean dataset
2. Trained SVM model
3. Evaluation metrics
4. All figures (15 EDA + 5 evaluation + 5 feature analysis = 25+ total)
5. All tables (sampling, cleaning, CV results, metrics, feature rankings, findings)
6. Feature importance analysis (permutation + logistic regression)
7. Report-ready outputs

The final result should resemble a small academic research project rather than a simple homework submission.

## Data

12 个月的 CSV 数据（2025-01 到 2025-12），位于 `2025MM-capitalbikeshare-tripdata/` 目录下。每月 29 万到 71 万行不等，全年总计约 660 万条记录。

### CSV Schema

| Column                                    | Type     | Description              |
| ----------------------------------------- | -------- | ------------------------ |
| `ride_id`                                 | str      | 行程唯一 ID                  |
| `rideable_type`                           | str      | 单车类型（electric_bike 等）    |
| `started_at` / `ended_at`                 | datetime | 起止时间（精度到毫秒）              |
| `start_station_name` / `start_station_id` | str      | 起始站点名称/ID                |
| `end_station_name` / `end_station_id`     | str      | 终点站点名称/ID                |
| `start_lat` / `start_lng`                 | float    | 起始经纬度                    |
| `end_lat` / `end_lng`                     | float    | 终点经纬度                    |
| `member_casual`                           | str      | 目标变量，"member" 或 "casual" |

每个 CSV 目录下有一个 `__MACOSX/` 隐藏文件夹（Mac 系统产生的元数据），读取数据时需忽略。

## Assignment Requirements（来自 PPTX）

1. **数据采样**：全年数据量大，需从每月按比例分层抽样（stratified sampling by month），确保 member/casual 比例与原始分布一致，解决类别不平衡问题
2. **特征工程**：从起止站点、经纬度、时间等字段构建特征（如计算骑行距离、提取时段/星期/月份）
3. **数据预处理**：处理缺失值、异常值、归一化/标准化
4. **模型**：SVM 分类，80/20 训练/测试划分
5. **评估**：准确率及验证指标
6. **分析**：不同特征对用户类型的影响强度，探索性发现加分

## 实验报告要求

- 内容：实验主题、目标、数据介绍（含时间/空间/数量分布图）、方法（算法流程+公式、数据处理流程、验证流程）、结果（准确率、SVM 支持向量图、特征重要性等）、讨论
- 字数：3500–4000 字左右
- 评分：模型准确率 40% + 报告与分析质量 40% + 可读性 20%

## Tech Stack Guidance

推荐使用 Python 生态：

- **pandas** — 数据加载、清洗、采样
- **scikit-learn** — `SVC`/`LinearSVC`、`train_test_split`、`StandardScaler`、`classification_report`
- **matplotlib / seaborn** — 可视化
- **numpy** — 数值计算
- **geopy** — 可选，计算站点间实际地理距离

由于数据量大（全年 660 万行），已实施以下优化：

- 分块读取 CSV（`chunksize=100_000`, `usecols=13`），避免一次性加载全月数据
- 分层抽样到 2 万条（`TOTAL_SAMPLE_SIZE=20000`），SVM O(n²) 复杂度下平衡精度与耗时
- 限制并行进程数为 4（`N_JOBS=4`），防止 32 核 fork 炸弹
- SVC 内核缓存限制为 500MB/实例（`SVC_CACHE_SIZE=500`），峰值仅 2GB
- GridSearchCV 评分使用 `f1_macro`，避免 70/30 不均衡数据下的 accuracy 偏向
- `CalibratedClassifierCV` 替代 `SVC(probability=True)` 进行概率校准
- matplotlib `Agg` 后端，避免 Windows tkinter 线程报错
