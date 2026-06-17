# 🚲 SVM 共享单车用户分类 — 机器学习实验课

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 使用支持向量机（SVM）对 Washington Capital Bikeshare 用户类型进行二分类预测 — member vs casual。

**最终结果**: Test Accuracy **73.27%** | ROC-AUC **0.69** | F1 Macro **0.60**

---

## 📋 项目简介

本项目是大学机器学习实验课作业，基于 Capital Bikeshare 2025 全年数据（~660 万条行程记录），构建 SVM 分类模型区分**年卡会员（member）**与**散客（casual）**。

完整实现 12 步机器学习流水线：数据采样 → 清洗 → 特征工程 → EDA → 特征选择 → 标准化 → SVM 建模 → 超参数优化 → 评估 → 特征重要性分析 → 统计发现。

---

## 📊 数据集

| 属性 | 描述 |
|------|------|
| 来源 | [Capital Bikeshare](https://capitalbikeshare.com/system-data) |
| 时间 | 2025 年 1 月 – 12 月 |
| 原始规模 | ~660 万条记录 (12 个 CSV, ~1.1 GB) |
| 抽样后 | 20,000 条 (每月按比例分层抽样) |
| 目标变量 | `member_casual` — member (≈70%) / casual (≈30%) |
| 特征维度 | 17 维 (时间循环编码 + 骑行特征 + 站点频率编码) |

---

## 🏗️ 项目结构

```
├── main.py                          # 主入口，完整 12 步流程编排
├── src/
│   ├── config.py                    # 全局配置 (路径、超参数、内存优化)
│   ├── data_loader.py               # CSV 文件发现与月度加载
│   ├── sampling.py                  # 每月分层抽样 (chunksize 分块)
│   ├── cleaning.py                  # DST 修复、IQR 缩尾、Haversine 距离
│   ├── feature_engineering.py       # 循环编码、频率编码、对数变换
│   ├── eda.py                       # 15 张探索性数据分析图
│   ├── feature_selection.py         # MI + 卡方 + RF 集成排名
│   ├── preprocessing.py             # StandardScaler + 80/20 分层划分
│   ├── train.py                     # 3 种 SVM + GridSearchCV (f1_macro)
│   ├── evaluate.py                  # 混淆矩阵、ROC、PR、学习曲线
│   ├── feature_analysis.py          # 排列重要性 + 逻辑回归系数
│   ├── findings.py                  # 6 项统计检验 (卡方 / Mann-Whitney U)
│   └── visualize.py                 # 中文字体、seaborn 样式、300 DPI 保存
├── outputs/
│   ├── figures/                     # 27 张高质量图表 (300 DPI PNG)
│   ├── tables/                      # 14 张数据表 (CSV)
│   └── models/                      # 训练好的模型 + scaler (joblib)
├── report/                          # 实验报告 (.docx)
├── 代码过程与原理解释.md               # 深度技术文档 (推荐阅读)
└── 实验报告撰写模版与参考.md           # 报告撰写指南
```

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- 依赖: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

### 运行

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. 安装依赖
pip install numpy pandas scikit-learn matplotlib seaborn

# 3. 下载数据
# 从 https://capitalbikeshare.com/system-data 下载 2025 年 1-12 月 CSV
# 放入 2025MM-capitalbikeshare-tripdata/ 目录

# 4. 运行完整流水线
python main.py
```

**预计耗时**: ~15 分钟 (i9-13900HX, 32GB RAM)  
**峰值内存**: ~360 MB

---

## 📈 核心结果

### 模型性能

| 指标 | 值 |
|------|-----|
| Test Accuracy | **73.27%** |
| F1 (Weighted) | 0.70 |
| Member Recall | 92.28% |
| Casual Recall | 27.46% |
| ROC-AUC | 0.69 |
| Best Params | C=0.05, γ=0.1 (RBF) |

### 15 张 EDA 图表预览

| # | 图表 | 分析维度 |
|---|------|----------|
| 1 | 饼图 | 用户类型分布 (≈70/30) |
| 2 | 堆叠柱状图 | 月度骑行量 |
| 3 | 折线图 | 小时级使用规律 |
| 4 | 分组柱状图 | 星期级模式 |
| 5 | 组合图 | 季节性变化 |
| 6-7 | KDE 分布 | 时长 & 距离对比 |
| 8 | 箱线图 | 平均速度差异 |
| 9 | 分组柱状图 | 高峰时段分布 |
| 10 | 水平柱状图 | Top 15 热门站点 |
| 11-12 | 分组柱状图 | 单车类型 & 无桩比例 |
| 13 | 热力图 | 特征相关性 |
| 14 | 散点图 | 时长 vs 距离 |
| 15 | 折线图 | 散客占比月度趋势 |

---

## 🔬 6 项统计发现

| # | 发现 | 检验方法 | 结论 |
|---|------|----------|------|
| 1 | **通勤模式** | 卡方检验 | 会员集中在通勤时段 (p < 0.001) |
| 2 | **时长差异** | Mann-Whitney U | 散客骑行时长显著更长 (p < 0.001) |
| 3 | **周末效应** | 卡方检验 | 周末散客比例显著更高 (p < 0.001) |
| 4 | **无桩骑行** | 卡方检验 | 无桩骑行中散客比例更高 (p < 0.001) |
| 5 | **季节性转变** | 描述统计 | 散客占比夏季峰值、冬季低谷 |
| 6 | **电动单车偏好** | 卡方检验 | 散客更偏好电动单车 (p < 0.001) |

---

## 🧠 关键技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 时间编码 | sin/cos 循环编码 | 保留小时/星期/月份的循环拓扑 |
| 站点编码 | 频率编码 | 避免高维 one-hot，保留热门程度 |
| 时长变换 | log1p | 压缩长尾分布，适配 SVM 欧氏距离 |
| 异常值 | IQR 缩尾 | 保留样本量，降低极端值影响 |
| 评分指标 | f1_macro | 70/30 不均衡下公平对待两个类别 |
| 并行控制 | n_jobs=4 | 防止 32 核 fork 炸弹 |
| 概率校准 | CalibratedClassifierCV | 替代废弃的 SVC(probability=True) |

---

## ⚡ 性能优化历程

项目经历了四轮迭代优化：

1. **内存爆炸修复** — `n_jobs` 从 32 → 4, `cache_size` 从 2GB → 500MB, CSV 分块加载
2. **GridSearchCV 加速** — 样本量从 6 万 → 1.2 万 (SVM O(n²) 特性)
3. **5 个 Bug 修复** — 列名冲突、tkinter 线程、API 废弃警告
4. **模型精度优化** — `scoring='f1_macro'` 替代 accuracy, 扩大 C 搜索范围

详见 [代码过程与原理解释.md](代码过程与原理解释.md) 第 14 章。

---

## 📖 延伸阅读

- [代码过程与原理解释.md](代码过程与原理解释.md) — 每个模块的原理、公式、设计决策 (~900 行深度文档)

---

## 📝 License

MIT License — 仅供学习参考，数据版权归 Capital Bikeshare 所有。
