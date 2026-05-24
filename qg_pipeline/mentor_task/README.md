# 生物医学关系提取与代表性关系选择

从生物医学论文中提取概念和关系，使用 LLM 智能选择最具代表性的关系，生成符合导师要求的 RDF 格式本体。

---

## 🚀 快速开始（完整流程）

只需两步，从论文到最终的本体文件：

### 步骤 1: 提取关系

```bash
python3 mentor_task/mentor_extract_to_csv.py \
    --config config.yaml \
    --input-dir data/processed_data \
    --n 50 \
    --seed 42
```

**作用**: 从 50 篇论文中提取所有重要概念和关系
**输出**: `mentor_task/mentor_50papers.csv`（原始提取结果，包含数百个关系）

---

### 步骤 2: LLM 智能选择代表性关系

```bash
python3 mentor_task/cluster_and_select_representative_relations.py \
    --config config.yaml
```

**注意**: 默认会读取 `mentor_task/mentor_50papers.csv`，这是步骤1的输出

**作用**: 使用 LLM 自动聚类和筛选，选出 20-30 个最具代表性的关系
**输出**:
- `mentor_task/representative_relations.csv` - 符合格式的最终本体文件 ✅ **给导师的**
- `mentor_task/representative_relations_report.md` - 详细分析报告

---

## 📋 工具说明

### 1. mentor_extract_to_csv.py

**功能**: 从论文中提取概念和关系

**输入**:
- `data/processed_data/*.json` - 预处理后的论文文件

**输出**:
- `mentor_task/mentor_50papers.csv` - 原始关系数据

**参数**:
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | config.yaml |
| `--input-dir` | 输入目录 | data/processed_data |
| `--out` | 输出文件路径 | mentor_task/mentor_50papers.csv |
| `--n` | 采样论文数量 | 50 |
| `--seed` | 随机种子 | 42 |
| `--all-sections` | 包含所有章节 | 仅摘要/前言/讨论 |
| `--dry-run` | 仅显示 prompt 不调用 API | - |

---

### 2. cluster_and_select_representative_relations.py ⭐

**功能**: 使用 LLM 智能聚类和选择代表性关系

**为什么需要这个脚本？**
- 第一步提取了数百个关系，太多了
- 需要筛选出 20-30 个最重要的
- 使用 LLM 理解语义，自动聚类，比手工维护规则更可靠

**输入**:
- `mentor_task/mentor_50papers.csv` - 第一步的输出

**输出**:
1. **`mentor_task/representative_relations.csv`** - RDF 格式的关系表
   - 符合导师要求的格式
   - 包含完整的元数据（出处、上下文等）
   - Subject/Predicate/Object 都是 URI 格式（带 `:` 前缀）

2. **`mentor_task/representative_relations_report.md`** - 详细报告
   - 聚类分析结果
   - 每个关系的选择理由
   - URI 映射表

**参数**:
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | config.yaml |
| `--input` | 输入 CSV 路径 | mentor_task/mentor_50papers.csv |
| `--output` | 输出 CSV 路径 | mentor_task/representative_relations.csv |
| `--report` | 报告输出路径 | mentor_task/representative_relations_report.md |
| `--n-relations` | 目标关系数量 | 30 |
| `--n-clusters` | 聚类数量 | 25 |
| `--dry-run` | 仅显示 prompt 不调用 API | - |

**工作原理**:
1. 读取所有关系，统计频率
2. **阶段 1 - LLM 聚类**: 按语义相似性分组（如"inhibits"和"suppresses"归为一类）
3. **阶段 2 - 代表性选择**: 从每个聚类选择最重要的关系
4. 转换为 RDF URI 格式（`:Entity_Name`）
5. 选择最佳上下文证据片段

---

## 📊 输出格式示例

### representative_relations.csv

```csv
subject,predicate,object,source_section,source_chunk,context_text,出处
:Simplified_Two_Tier_System,:improves_metric,:Operability,前言,0,"对一线临床医师而言，分层架构可提升操作性。",paper_123
:Hypertension_Diagnosis,:can_be_based_on,:OBPM_Monitoring,4obpm24h,3,"高血压的诊断可依据OBPM、24小时动态血压监测等方法。",paper_456
:Relugolix,:First_Line_Drug_For,:Prostate_Cancer,前言,0,"Leuprolide is the first-line drug for these diseases...",paper_295
```

**格式说明**:
- Subject/Predicate/Object 都是 URI 格式（带 `:` 前缀）
- 空格替换为下划线
- 保留中文字符
- 完整保留出处、位置、上下文证据

---

## ⚙️ 配置

所有脚本从 `config.yaml` 读取配置：

```yaml
api:
  openai_api_key: ${OPENAI_API_KEY}
  openai_base_url: ${OPENAI_BASE_URL}

extractor:
  model: "deepseek-chat"
  temperature: 0.1
  max_tokens: 3000
```

**环境变量**:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

---

## 🔧 常见问题

### Q: 调整代表性关系的数量？

```bash
python3 mentor_task/cluster_and_select_representative_relations.py \
    --n-relations 25  # 选择 25 个关系
```

### Q: 测试一下但不调用 API？

```bash
python3 mentor_task/cluster_and_select_representative_relations.py \
    --dry-run
```

### Q: 为什么要用 LLM 而不是规则？

**LLM 方法的优势**:
- ✅ 理解语义，"inhibits" 和 "suppresses" 自动归为一类
- ✅ 无需手工维护映射规则
- ✅ 新关系类型自动适应
- ✅ 提供选择理由和解释

**旧方法**（已归档到 `archived/` 目录）需要手工维护数百条映射规则。

---

## 📁 目录结构

```
mentor_task/
├── README.md                           # 本文档
├── mentor_extract_to_csv.py            # 步骤1: 提取关系
├── cluster_and_select_representative_relations.py  # 步骤2: LLM选择
├── __init__.py
└── archived/                           # 已归档的旧方法（基于规则）
    ├── transform_csv_to_triples.py
    ├── analyze_relations.py
    ├── normalize_and_merge_relations.py
    └── generate_final_report.py
```

---

## 📚 输出文件

| 文件 | 用途 | 位置 |
|------|------|------|
| `mentor_50papers.csv` | 原始提取结果 | `mentor_task/` |
| `representative_relations.csv` | 最终本体文件（给导师） | `mentor_task/` |
| `representative_relations_report.md` | 分析报告（给自己看） | `mentor_task/` |

---

**最后更新**: 2025-01-14
**维护者**: Claude Code
