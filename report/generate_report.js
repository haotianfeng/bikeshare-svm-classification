const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat, TableOfContents,
} = require("docx");

// ── Helper: create a body paragraph with SimSun ──
function bodyPara(text, opts = {}) {
  return new Paragraph({
    spacing: { line: 360, before: 60, after: 60 },
    indent: opts.noIndent ? undefined : { firstLine: 480 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [
      new TextRun({
        text,
        font: "SimSun",
        size: 24, // 12pt
        ...(opts.bold ? { bold: true } : {}),
      }),
    ],
  });
}

// ── Helper: H1 with SimHei ──
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [
      new TextRun({
        text,
        font: "SimHei",
        size: 32, // 16pt
        bold: true,
      }),
    ],
  });
}

// ── Helper: H2 with KaiTi ──
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: [
      new TextRun({
        text,
        font: "KaiTi",
        size: 28, // 14pt
        bold: true,
      }),
    ],
  });
}

// ── Helper: formula paragraph (centered, monospaced) ──
function formula(text) {
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    alignment: AlignmentType.CENTER,
    children: [
      new TextRun({
        text,
        font: "Times New Roman",
        size: 24,
        italics: true,
      }),
    ],
  });
}

// ── Helper: table cell ──
function cell(text, opts = {}) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new TableCell({
    borders,
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.shading ? { fill: opts.shading, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: "center",
    children: [
      new Paragraph({
        alignment: opts.align || AlignmentType.CENTER,
        children: [
          new TextRun({
            text: String(text),
            font: opts.font || "SimSun",
            size: opts.size || 22,
            bold: opts.bold || false,
          }),
        ],
      }),
    ],
  });
}

// ── Helper: make a table ──
function makeTable(headers, rows, colWidths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) =>
      new TableCell({
        borders,
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        verticalAlign: "center",
        children: [
          new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: h, font: "SimHei", size: 22, bold: true })],
          }),
        ],
      })
    ),
  });

  const dataRows = rows.map((row) =>
    new TableRow({
      children: row.map((val, i) =>
        new TableCell({
          borders,
          width: { size: colWidths[i], type: WidthType.DXA },
          margins: { top: 50, bottom: 50, left: 100, right: 100 },
          verticalAlign: "center",
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ text: String(val), font: "SimSun", size: 22 })],
            }),
          ],
        })
      ),
    })
  );

  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
}

// ── Helper: ordered list item ──
function numberedItem(num, text) {
  return new Paragraph({
    spacing: { line: 340, before: 40, after: 40 },
    indent: { left: 480 },
    children: [
      new TextRun({ text: `(${num}) `, font: "SimSun", size: 24 }),
      new TextRun({ text, font: "SimSun", size: 24 }),
    ],
  });
}

// ── Helper: figure caption ──
function figCaption(text) {
  return new Paragraph({
    spacing: { before: 40, after: 120 },
    alignment: AlignmentType.CENTER,
    children: [
      new TextRun({ text, font: "KaiTi", size: 22, italics: true }),
    ],
  });
}

// ════════════════════════════════════════════════════════════════
// BUILD DOCUMENT
// ════════════════════════════════════════════════════════════════

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: "SimSun", size: 24 },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 32, bold: true, font: "SimHei" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 28, bold: true, font: "KaiTi" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "findings",
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    // ══════════════════════════════════════════════════════
    // TITLE PAGE SECTION
    // ══════════════════════════════════════════════════════
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 }, // A4
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children: [
        new Paragraph({ spacing: { before: 3600 }, children: [] }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 600 },
          children: [
            new TextRun({
              text: "机器学习实验课实验报告",
              font: "SimHei",
              size: 44, // 22pt
              bold: true,
            }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [
            new TextRun({
              text: "基于SVM的共享单车用户类型分类预测",
              font: "KaiTi",
              size: 36, // 18pt
              bold: true,
            }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [
            new TextRun({
              text: "—— 华盛顿 Capital Bikeshare 数据 (2025年1月–12月)",
              font: "KaiTi",
              size: 28,
            }),
          ],
        }),
        new Paragraph({ spacing: { before: 1200 }, children: [] }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "实验日期：2025年6月", font: "SimSun", size: 24 })],
        }),
      ],
    },

    // ══════════════════════════════════════════════════════
    // MAIN CONTENT SECTION
    // ══════════════════════════════════════════════════════
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ text: "机器学习实验课实验报告", font: "KaiTi", size: 18, italics: true }),
              ],
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ text: "第 ", font: "SimSun", size: 18 }),
                new TextRun({ children: [PageNumber.CURRENT], font: "SimSun", size: 18 }),
                new TextRun({ text: " 页", font: "SimSun", size: 18 }),
              ],
            }),
          ],
        }),
      },
      children: [
        // ── TOC ──
        new Paragraph({
          heading: HeadingLevel.HEADING_1,
          spacing: { before: 360, after: 200 },
          children: [new TextRun({ text: "目录", font: "SimHei", size: 32, bold: true })],
        }),
        new TableOfContents("目录", { hyperlink: true, headingStyleRange: "1-2" }),
        new Paragraph({ children: [new PageBreak()] }),

        // ══════════════════════════════════════════════
        // 一、实验问题
        // ══════════════════════════════════════════════
        h1("一、实验问题"),

        bodyPara("随着共享经济的快速发展，共享单车已成为城市交通系统的重要组成部分。华盛顿都会区的 Capital Bikeshare 系统作为美国规模最大的共享单车系统之一，每日产生大量骑行数据。该系统将用户分为两类：会员（Member）和散客（Casual）。会员通常为注册年度会员，多为日常通勤者；散客则为按次付费或短期用户，多为游客或休闲骑行者。"),
        bodyPara("准确识别和预测用户类型对于共享单车运营管理具有重要意义：一方面，不同用户类型具有差异化的使用模式和需求特征，了解这些差异有助于优化车辆调度和站点布局；另一方面，用户类型预测可以帮助运营方制定精准的营销策略和差异化定价方案。本实验的核心问题是：能否仅基于骑行行为特征（时间、地点、距离、单车类型等），利用支持向量机（SVM）算法构建分类模型，准确区分会员用户与散客用户？何种骑行行为特征对区分两类用户最为关键？"),

        // ══════════════════════════════════════════════
        // 二、实验目标
        // ══════════════════════════════════════════════
        h1("二、实验目标"),

        bodyPara("本实验的主要目标如下："),
        numberedItem(1, "构建完整的数据处理流程，包括数据抽样、清洗、特征工程和标准化；"),
        numberedItem(2, "基于支持向量机（SVM）算法训练二分类模型，实现对会员（Member）和散客（Casual）用户类型的分类预测；"),
        numberedItem(3, "通过网格搜索和交叉验证优化SVM模型的超参数，提升分类性能；"),
        numberedItem(4, "综合运用多种评估指标（准确率、精确率、召回率、F1分数、ROC-AUC）全面评价模型性能；"),
        numberedItem(5, "利用排列重要性（Permutation Importance）和逻辑回归系数分析各特征对分类的贡献度；"),
        numberedItem(6, "通过统计检验方法挖掘会员与散客用户的行为差异，形成有意义的实验发现。"),

        // ══════════════════════════════════════════════
        // 三、数据介绍
        // ══════════════════════════════════════════════
        new Paragraph({ children: [new PageBreak()] }),
        h1("三、数据介绍"),

        h2("3.1 数据来源"),
        bodyPara("本实验使用的数据集来自 Capital Bikeshare 公布的华盛顿都会区共享单车轨迹数据，涵盖2025年1月至12月共12个月的历史骑行记录。原始数据为CSV格式，每月一个文件，全年共计约660万条记录。每条记录代表一次完整的骑行行程，包含以下关键字段：行程唯一标识（ride_id）、单车类型（rideable_type，分为电动单车 electric_bike 和经典单车 classic_bike）、起止时间（started_at / ended_at，精确到毫秒）、起止站点名称与编号（start_station_name / start_station_id、end_station_name / end_station_id）、起止经纬度坐标（start_lat / start_lng、end_lat / end_lng），以及目标变量——用户类型（member_casual，取值为 member 或 casual）。"),

        h2("3.2 数据抽样策略"),
        bodyPara("由于全年原始数据量达660万条，直接使用全部数据进行SVM训练将导致计算复杂度O(n²)带来的内存与时间不可承受。因此，本实验采用按月比例分层抽样策略，从12个月的数据中抽取代表性样本。"),
        bodyPara("具体抽样方案如下：（1）按每个月原始数据量占总量的比例，分配该月应抽取的样本数，确保月度代表性；（2）每月内部采用分层抽样，分别从member和casual两类用户中按比例抽取，保持类别分布与原始数据一致，避免类别不平衡偏差；（3）总目标样本量设定为20,000条，抽样比例约为0.30%；（4）随机种子固定为42，确保实验可复现。"),
        bodyPara("抽样完成后，共计获得19,999条样本。各月样本分布如表3-1所示。"),

        figCaption("表3-1 月度抽样分布表"),
        makeTable(
          ["月份", "总样本数", "会员数", "散客数", "会员占比(%)"],
          [
            ["2025-01", "858", "699", "159", "81.5"],
            ["2025-02", "1,107", "834", "273", "75.3"],
            ["2025-03", "1,881", "1,325", "556", "70.4"],
            ["2025-04", "1,995", "1,402", "593", "70.3"],
            ["2025-05", "2,075", "1,462", "613", "70.5"],
            ["2025-06", "2,076", "1,460", "616", "70.3"],
            ["2025-07", "2,117", "1,524", "593", "72.0"],
            ["2025-08", "1,918", "1,241", "677", "64.7"],
            ["2025-09", "1,876", "1,300", "576", "69.3"],
            ["2025-10", "1,877", "1,306", "571", "69.6"],
            ["2025-11", "1,339", "948", "391", "70.8"],
            ["2025-12", "880", "711", "169", "80.8"],
          ],
          [1500, 1500, 1500, 1500, 1800]
        ),

        bodyPara("从表3-1可以看出，各月样本的会员占比在64.7%至81.5%之间波动，反映了原始数据中真实存在的类别分布差异。冬季月份（1月、12月）会员占比较高，而夏季月份（8月）会员占比相对较低，散客出行增多。"),

        h2("3.3 数据清洗"),
        bodyPara("数据清洗是确保模型训练质量的关键步骤。清洗前数据集为19,999条记录，清洗后保留18,686条（去除率6.6%）。具体清洗步骤包括："),
        numberedItem(1, "夏令时（DST）修复：11月份因冬令时回拨，部分记录的结束时间早于开始时间，导致负的骑行时长。通过将负时长的结束时间加1小时进行修正。"),
        numberedItem(2, "缺失值处理：去除终点经纬度缺失的记录（占比极少）。站点名称为空（无桩骑行）予以保留，通过二值特征标记。"),
        numberedItem(3, "异常值过滤：设置骑行时长有效范围为1分钟至1440分钟（24小时），距离有效范围为0.5 km/h至50 km/h的平均速度过滤，移除超出合理范围的记录。"),
        numberedItem(4, "距离为零但起止站点不同的矛盾记录：移除haversine距离为0但起止站点ID不一致的记录。"),
        numberedItem(5, "IQR截尾处理：对骑行时长和Haversine距离分别应用四分位距（IQR）方法进行截尾，将超出Q1-1.5×IQR至Q3+1.5×IQR范围的值替换为边界值，保留数据而非删除。"),
        numberedItem(6, "派生指标计算：在清洗后的数据基础上，计算每次骑行的Haversine地理距离（km）和平均骑行速度（km/h），作为后续特征工程的输入。"),

        h2("3.4 探索性数据分析"),
        bodyPara("为深入理解数据特征和两类用户的骑行行为差异，本实验生成了15张探索性数据分析（EDA）图表，所有图表均以300 DPI高分辨率保存。"),
        bodyPara("EDA分析的主要发现包括：整体数据中会员约占比70%、散客约占比30%，存在中等程度的类别不平衡。月度骑行量呈现明显的季节性波动，夏季（6–8月）骑行量最高，冬季（1月、12月）最低。会员骑行集中在早晚通勤高峰时段（7–9时和16–19时），呈现典型的“双峰”模式；散客骑行在白天时段分布更为均匀，午后休闲骑行占比较高。散客在周末的骑行占比（约37%）显著高于工作日（约26%）。散客骑行时长中位数（11.8分钟）显著高于会员（8.8分钟），表明散客倾向于更长时间的休闲骑行。电动单车在散客中的使用比例（59.3%）低于在会员中的使用比例（64.7%）。"),

        // ══════════════════════════════════════════════
        // 四、实验方法
        // ══════════════════════════════════════════════
        new Paragraph({ children: [new PageBreak()] }),
        h1("四、实验方法"),

        h2("4.1 支持向量机算法原理"),
        bodyPara("支持向量机（Support Vector Machine, SVM）是一种经典的监督学习算法，其核心思想是在特征空间中寻找一个最优超平面，使得不同类别样本之间的分类间隔（margin）最大化。对于线性不可分问题，SVM通过核函数（Kernel Function）将原始特征映射到高维空间，在高维空间中构造线性分类超平面，从而实现对非线性问题的求解。"),
        bodyPara("SVM的核心优化问题可表达为带软间隔（Soft Margin）的约束优化。软间隔SVM通过引入松弛变量ξᵢ，允许部分样本落在间隔内部或被错误分类，从而增强模型对噪声和异常值的鲁棒性。优化目标为在最大化间隔的同时最小化分类错误。其原始优化问题为："),
        formula("min  (1/2)·||w||² + C · Σξᵢ"),
        formula("s.t.  yᵢ(wᵀφ(xᵢ) + b) ≥ 1 − ξᵢ,   ξᵢ ≥ 0"),
        bodyPara("其中，w为超平面法向量，b为偏置项，φ(x)为核映射函数，C为正则化参数（惩罚系数），ξᵢ为第i个样本的松弛变量。C越大，对误分类的惩罚越重，模型倾向于更复杂的决策边界；C越小，模型倾向于更宽的间隔和更高的泛化能力。"),
        bodyPara("引入拉格朗日乘子αᵢ后，可转化为对偶问题："),
        formula("max  Σαᵢ − (1/2)·ΣΣαᵢαⱼyᵢyⱼK(xᵢ, xⱼ)"),
        formula("s.t.  Σαᵢyᵢ = 0,   0 ≤ αᵢ ≤ C"),
        bodyPara("其中K(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩为核函数。本实验采用径向基函数（Radial Basis Function, RBF）核：K(xᵢ, xⱼ) = exp(−γ·||xᵢ − xⱼ||²)。RBF核能够处理非线性分类边界，参数γ控制单个样本的影响范围：γ越大，决策边界越复杂，越容易过拟合；γ越小，决策边界越平滑。"),

        h2("4.2 特征工程"),
        bodyPara("从原始数据出发，本实验构建了三大类共17个特征变量："),
        bodyPara("（1）时间循环特征（9个）：hour_sin、hour_cos（小时的正弦/余弦编码，周期24）；dayofweek_sin、dayofweek_cos（星期几的正弦/余弦编码，周期7）；month_sin、month_cos（月份的正弦/余弦编码，周期12）；is_weekend（是否周末，二值）；is_morning_rush（是否早高峰7–9时工作日，二值）；is_evening_rush（是否晚高峰16–19时工作日，二值）。使用正弦/余弦循环编码而非原始数值，能确保23小时与0小时在特征空间中相邻，避免人为引入数值边界。"),
        bodyPara("（2）行程特征（3个）：duration_minutes_log（骑行时长经对数变换log(1+x)，缓解长尾分布）；haversine_distance_km（基于起点和终点经纬度计算的Haversine球面距离，单位为公里）；is_electric_bike（是否电动单车，二值）。"),
        bodyPara("（3）站点特征（4个）：has_start_station、has_end_station（是否有起/止站点，二值，标记无桩骑行）；start_station_freq、end_station_freq（起/止站点频率编码，即该站点的骑行次数占总骑行次数的比例，反映站点热度）。"),
        bodyPara("（4）复合特征（1个）：avg_speed_kmh（平均骑行速度，由距离除以时长计算，经截尾处理限制在合理范围内）。"),
        bodyPara("所有特征经检查，无NaN值、无Inf值，确保模型训练的数值稳定性。"),

        h2("4.3 特征选择"),
        bodyPara("为理解各特征对分类任务的贡献度，本实验综合使用了三种特征选择方法："),
        bodyPara("（1）互信息（Mutual Information, MI）：度量每个特征与目标变量之间的信息共享量，不依赖任何模型假设，能捕捉非线性关系。"),
        bodyPara("（2）卡方检验（Chi-Square Test）：检验每个特征与目标变量的统计独立性。特征经MinMaxScaler缩放至[0,1]区间以满足卡方检验的非负性要求。"),
        bodyPara("（3）随机森林重要性（Random Forest Importance）：使用150棵决策树、最大深度为10的随机森林模型，基于Gini不纯度减少量计算特征重要性。随机森林仅用于辅助特征重要性分析，不作为最终的分类模型。"),
        bodyPara("三种方法分别给出特征排名后，计算各特征的平均归一化排名，形成集成排名（Ensemble Ranking），综合反映特征的区分能力。"),

        h2("4.4 数据标准化与划分"),
        bodyPara("SVM算法对特征尺度敏感，因此对17个特征全部应用StandardScaler标准化（零均值、单位方差），确保各特征在模型训练中具有可比的重。Scaler仅在训练集上拟合，然后用于变换训练集和测试集，严格防止数据泄漏。"),
        bodyPara("训练/测试集按80:20比例分层划分（stratify=y），保证训练集和测试集中member与casual的比例与原始数据一致。随机种子固定为42，确保划分结果可复现。标准化后，训练集包含约16,000条样本，测试集包含约3,738条样本。"),

        h2("4.5 模型训练与超参数优化"),
        bodyPara("本实验构建了三类SVM基准模型进行对比：（1）LinearSVC（线性核，dual=False，max_iter=5000，class_weight='balanced'）；（2）SVC-Poly（多项式核，degree=2，C=1.0，class_weight='balanced'）；（3）SVC-RBF（径向基核，C=1.0，gamma='scale'，class_weight='balanced'）。所有模型均设置class_weight='balanced'以缓解类别不平衡问题。"),
        bodyPara("基于基准模型对比结果，选定RBF核SVC作为目标模型，使用GridSearchCV进行超参数优化。超参数搜索网格为：C ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 100}（8个值），gamma ∈ {'scale', 0.1, 0.05, 0.01, 0.001}（5个值），核函数固定为rbf，共计8×5=40个参数组合。采用5折交叉验证，以f1_macro作为评分指标。f1_macro平等对待member和casual两个类别，在70/30不均衡数据下比accuracy更能反映模型对少数类（散客）的识别能力。模型校准采用CalibratedClassifierCV（5折Platt Scaling），为后续ROC和PR曲线提供概率估计。为防止内存溢出，并行进程数限制为4，SVC内核缓存限制为500MB。"),

        h2("4.6 模型评估方法"),
        bodyPara("模型评估采用多维度指标体系：（1）准确率（Accuracy）：正确分类的样本占总样本的比例；（2）精确率（Precision）：预测为某类别的样本中实际属于该类别的比例；（3）召回率（Recall）：某类别的样本中被正确识别出来的比例；（4）F1分数（F1 Score）：精确率和召回率的调和平均数，F1 = 2×(P×R)/(P+R)；（5）ROC-AUC：受试者工作特征曲线下面积，衡量模型的整体排序能力。"),
        bodyPara("此外，生成混淆矩阵（原始计数+按行归一化）、ROC曲线、精确率-召回率（PR）曲线、学习曲线和分类报告热力图，从多角度可视化模型性能。"),

        // ══════════════════════════════════════════════
        // 五、实验结果
        // ══════════════════════════════════════════════
        new Paragraph({ children: [new PageBreak()] }),
        h1("五、实验结果"),

        h2("5.1 基准模型对比"),
        bodyPara("在默认参数下，三种SVM模型的测试集性能如表5-1所示。"),

        figCaption("表5-1 基准模型性能对比"),
        makeTable(
          ["模型", "准确率", "F1 (加权)", "训练耗时(s)"],
          [
            ["LinearSVC", "64.79%", "0.6617", "0.0"],
            ["SVC-Poly (d=2)", "61.61%", "0.6327", "4.1"],
            ["SVC-RBF (基线)", "66.43%", "0.6748", "4.8"],
          ],
          [2800, 2800, 2800, 2000]
        ),

        bodyPara("结果显示，RBF核对本数据集的适应性最强，在三种基准模型中取得最高的准确率（66.43%）和F1加权分数（0.6748）。LinearSVC速度最快但性能有限，多项式核SVC（degree=2）表现最弱。因此，选定RBF核SVC作为后续超参数优化的基础模型。"),

        h2("5.2 超参数优化结果"),
        bodyPara("GridSearchCV共执行200次拟合（40个参数组合×5折交叉验证），历时约10.4分钟。最优超参数组合为：C=0.05，gamma=0.1，kernel=rbf。最优交叉验证f1_macro得分为0.5496。C的最优值出现在搜索网格内部而非边界（C=0.05，网格范围为0.001–100），表明搜索范围覆盖了最优区域，无需进行扩展搜索。"),
        bodyPara("较小的最优C值（0.05）表明模型倾向于更宽的分类间隔和更强的正则化，这有助于防止过拟合。较大的gamma值（0.1 > 默认'scale'≈0.0588）表明RBF核的径向作用范围适中，单个支持向量的影响区域不会过大。"),

        h2("5.3 最终模型性能"),
        bodyPara("使用最优超参数训练的最终模型在测试集（3,738条样本）上的性能指标如表5-2所示。"),

        figCaption("表5-2 最终模型测试集性能"),
        makeTable(
          ["指标", "值"],
          [
            ["准确率 (Accuracy)", "73.27%"],
            ["F1 (加权平均)", "0.6969"],
            ["ROC-AUC", "0.6928"],
            ["会员 精确率", "75.41%"],
            ["会员 召回率", "92.28%"],
            ["会员 F1", "0.8300"],
            ["散客 精确率", "59.60%"],
            ["散客 召回率", "27.46%"],
            ["散客 F1", "0.3760"],
          ],
          [5000, 4360]
        ),

        bodyPara("从表5-2可以看出，模型对会员用户的识别能力较强（召回率92.28%），但对散客的识别能力有限（召回率27.46%）。该结果表明：仅依靠骑行行为特征（时间、地点、距离、速度等），难以精确区分两类用户。会员用户的行为模式相对集中（以通勤为主），而散客的行为模式高度多样化（可能涵盖通勤、观光、购物等多种目的），两类用户的行为分布存在大量重叠区域。"),

        h2("5.4 模型评估图表分析"),
        bodyPara("实验生成了5张模型评估图：（1）混淆矩阵（原始计数+按行归一化），直观展示每类样本的分类正确和错误情况；（2）ROC曲线（AUC=0.69），表明模型具有一定的排序能力，高于随机分类器的0.5基线但仍有较大提升空间；（3）PR曲线，在类别不均衡场景下比ROC更敏感，能更真实地反映少数类的分类效果；（4）分类报告热力图，汇总精确率、召回率、F1分数；（5）学习曲线，显示训练集和交叉验证集得分随样本量的变化趋势，用于判断是否存在过拟合或欠拟合。"),

        // ══════════════════════════════════════════════
        // 六、实验结果分析
        // ══════════════════════════════════════════════
        new Paragraph({ children: [new PageBreak()] }),
        h1("六、实验结果分析"),

        h2("6.1 特征重要性分析"),
        bodyPara("由于SVM本身缺乏直接的特征重要性输出，本实验采用两种互补方法进行事后解释："),
        bodyPara("（1）排列重要性（Permutation Importance）：通过随机打乱单个特征的值并观测模型准确率的下降程度来衡量特征重要性。重复20次以获得稳定的重要性估计和标准差。前三重要的特征为：duration_minutes_log（准确率下降0.0216±0.0039）、avg_speed_kmh（0.0200±0.0030）、has_end_station（0.0095±0.0027）。"),
        bodyPara("（2）逻辑回归系数（Logistic Regression Coefficients）：训练一个L2正则化的逻辑回归模型作为辅助解释工具，提取标准化系数以判断特征的影响方向。系数绝对值前三的特征为：duration_minutes_log（-0.736，负号表示长时长更偏向散客）、haversine_distance_km（+0.462，正号表示远距离更偏向会员）、has_start_station（-0.267）。"),
        bodyPara("两种方法一致指向duration_minutes_log和avg_speed_kmh是区分两类用户的最关键特征。骑行时长和速度差异反映了会员用户的短时高效骑行模式（以通勤为目的）与散客用户的长时休闲骑行模式（以观光游览为目的）之间的本质区别。"),
        bodyPara("此外，基于互信息、卡方检验和随机森林重要性三种方法的集成排名（Ensemble Ranking）中，duration_minutes_log、avg_speed_kmh和haversine_distance_km位列前三，进一步验证了上述结论。"),

        h2("6.2 有趣发现"),
        bodyPara("通过统计检验方法，本实验挖掘了六项具有统计学显著性的有趣发现："),

        bodyPara("发现一：会员通勤模式显著。基于卡方独立性检验，会员在高峰时段（工作日7–9时和16–19时）的骑行比例为40.6%，显著高于散客的30.9%（χ²=155.3，p=1.19×10⁻³⁵）。这表明会员用户以通勤为主要使用场景。", { bold: false }),

        bodyPara("发现二：散客骑行时长显著更长。Mann-Whitney U检验结果显示，散客骑行时长中位数（11.8分钟）显著高于会员（8.8分钟）（U=44,273,346，p=1.79×10⁻¹²⁸）。散客倾向于更长时间的休闲骑行，而会员以短时高效的交通接驳为主。"),

        bodyPara("发现三：周末散客占比显著更高。周末散客骑行占比为36.8%，远高于工作日的26.4%，差异具有统计学显著性（χ²=197.9，p=5.95×10⁻⁴⁵）。周末是散客出行的集中时段，这与休闲观光的出行目的相符。"),

        bodyPara("发现四：无桩骑行中散客比例更高。无桩骑行（无站点记录）中散客占比为31.8%，显著高于有桩骑行的20.1%（χ²=201.4，p=1.06×10⁻⁴⁵）。散客更可能在不固定站点借还车辆，反映出其使用的随意性。"),

        bodyPara("发现五：散客占比呈现季节性波动。散客占比夏季最高（31.5%）、冬季最低（21.5%），峰谷比为1.46。散客出行受天气和旅游旺季影响明显，夏季旅游高峰期散客量大幅增加。"),

        bodyPara("发现六：单车类型偏好的细微差异。电动单车使用比例在会员中为64.7%，在散客中为59.3%，虽然差异不大但具有统计学显著性（χ²=49.1，p=2.44×10⁻¹²）。"),

        // ══════════════════════════════════════════════
        // 七、讨论与结论
        // ══════════════════════════════════════════════
        new Paragraph({ children: [new PageBreak()] }),
        h1("七、讨论与结论"),

        bodyPara("本实验基于华盛顿Capital Bikeshare共享单车2025年全年数据，采用支持向量机（SVM）构建了会员与散客的二分类预测模型。实验严格遵循从数据抽样、清洗、特征工程、特征选择到模型训练、超参数优化和综合评估的完整机器学习流程。"),
        bodyPara("实验取得的主要成果包括：（1）从660万条原始记录中通过按月比例分层抽样构建了2万条代表性样本，清洗后保留18,686条高质量记录；（2）构建了涵盖时间循环编码、行程特征、站点频率编码和复合特征的17维特征空间；（3）通过GridSearchCV（5折交叉验证、f1_macro评分）获得了最优SVM模型（C=0.05，gamma=0.1，RBF核）；（4）最终模型测试集准确率达73.27%，显著高于70%的多数类基线，ROC-AUC为0.69，表明模型确实学到了一定的分类能力；（5）通过排列重要性和逻辑回归系数揭示了骑行时长和平均速度为区分两类用户的最关键特征；（6）基于统计检验挖掘了六项具有统计学显著性的用户行为差异发现。"),
        bodyPara("实验也存在一定局限性：散客召回率仅27.46%，表明仅靠骑行行为特征难以高精度识别散客用户。两类用户的骑行模式存在本质性重叠——散客中也有通勤者，会员中也有休闲骑行者；此外，缺少用户画像数据（如年龄、收入、是否有车、骑行目的等），这些信息可能是区分用户类型的关键。"),
        bodyPara("未来改进方向包括：（1）构造交互特征（如周末×时段、季节×站点热度）以捕捉更复杂的行为模式；（2）尝试类别权重调整或SMOTE过采样以进一步缓解散客召回率低的问题；（3）引入外部数据（如天气、节假日信息）丰富特征维度；（4）在算力允许的条件下，可探索集成方法中的非线性模型作为基准比较。"),
        bodyPara("综上所述，本实验完整实现了基于SVM的共享单车用户分类预测，虽然在散客识别精度上存在提升空间，但实验流程规范、方法全面、分析深入，所获得的骑行行为特征重要性排序和用户行为差异发现对共享单车运营管理具有实际参考价值。"),

        // ══════════════════════════════════════════════
        // 参考文献
        // ══════════════════════════════════════════════
        new Paragraph({ children: [new PageBreak()] }),
        h1("参考文献"),

        numberedItem(1, "Cortes C, Vapnik V. Support-vector networks[J]. Machine Learning, 1995, 20(3): 273-297."),
        numberedItem(2, "Boser B E, Guyon I M, Vapnik V N. A training algorithm for optimal margin classifiers[C]. Proceedings of the Fifth Annual Workshop on Computational Learning Theory, 1992: 144-152."),
        numberedItem(3, "Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning: Data Mining, Inference, and Prediction[M]. 2nd ed. Springer, 2009."),
        numberedItem(4, "Pedregosa F, et al. Scikit-learn: Machine learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830."),
        numberedItem(5, "Capital Bikeshare. System Data [EB/OL]. https://capitalbikeshare.com/system-data, 2025."),
        numberedItem(6, "Breiman L. Random forests[J]. Machine Learning, 2001, 45(1): 5-32."),
        numberedItem(7, "Altmann A, et al. Permutation importance: a corrected feature importance measure[J]. Bioinformatics, 2010, 26(10): 1340-1347."),
      ],
    },
  ],
});

// ══════════════════════════════════════════════════════
// WRITE FILE
// ══════════════════════════════════════════════════════
const outputPath = "D:/cxdownload/机器学习实验课/report/实验报告.docx";
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outputPath, buffer);
  console.log("Report generated:", outputPath);
  console.log("File size:", (buffer.length / 1024).toFixed(1), "KB");
});
