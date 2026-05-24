# Multidoc-KG-zya

这是一个面向 **PubMed / 医学论文 PDF** 的知识图谱构建项目。  
当前主线目标是：

```text
PDF
-> 预处理
-> 论文 JSON
-> 子领域识别
-> 批次化子领域 refinement
-> 三元组抽取
-> 实体对齐
-> 验证
-> 写入 Neo4j
```

这份 README 按“**准备转手**”来写，重点说明：

- `data/` 怎么维护
- 新人怎么把项目跑起来
- 各种常用命令怎么用
- 接手时最容易踩的坑是什么

## 1. 项目现状

当前代码已经完成从“古籍/中医实验项目”到“医学文献知识图谱项目”的迁移。  
现在可以直接做：

- 医学论文 PDF 预处理
- 子领域识别与批次化 taxonomy 维护
- 受控 biomedical relation 抽取
- 实体 grounding
- claim validation
- Neo4j 写库

当前默认向量模型：

- `BAAI/bge-m3`

当前默认 LLM 入口：

- OpenAI-compatible API

## 2. 目录结构

最重要的目录如下：

```text
.
├── agents/                 # 各阶段 agent
├── config/                 # 配置文件
├── core/                   # 基础设施：LLM、Neo4j、向量库、日志等
├── data/
│   ├── raw_data_papers/    # 原始 PDF
│   ├── processed_papers/   # 预处理后的 section JSON
│   ├── cleaned_papers/     # 后清洗后的 JSON（主线默认输入）
│   └── preprocess/         # 预处理脚本
├── scripts/                # 辅助脚本
├── main.py                 # 主入口
├── data_loader.py          # 论文加载
├── schema.py               # 主要数据结构
├── .env.example            # 环境变量模板
└── requirements.txt
```

## 3. `data/` 怎么操作

这一部分是最关键的交接内容。

### 3.1 `data/raw_data_papers/`

这里放原始 PDF。

约定：

- 一个 PDF 对应一篇论文
- 文件名建议直接用 `PMID.pdf`
- 例如：
  - `39669840.pdf`
  - `39677775.pdf`

建议：

- 新增论文时，优先保持这个命名规则
- 不要随意混入中文文件名或描述性长标题，后续追踪会麻烦

### 3.2 `data/processed_papers/`

这里放 **预处理后的 section JSON**。  
它是由 `raw_data_papers/` 经过 PDF 切分后得到的中间结果。

这一步的作用：

- 按章节组织论文
- 恢复 section title
- 清掉一部分页眉页脚、书签噪声

这批文件属于：

- **中间结果**
- 可重新生成

如果你只想直接跑主线，不一定要手动改这里。

### 3.3 `data/cleaned_papers/`

这里放 **后清洗后的 JSON**，也是当前主线默认输入目录。

主线默认使用：

```bash
python main.py --data-dir data/cleaned_papers --batch-size 5
```

这一步相对 `processed_papers/` 额外做了：

- 首页噪声清洗
- 尾部 section 去噪
- 一部分图注 / 版权 / author info 压缩

如果你只是要继续实验：

- **优先用 `cleaned_papers/`**

### 3.4 新增一批 PDF 时的推荐流程

如果接手人要加新论文，建议这样做：

1. 把 PDF 放进 `data/raw_data_papers/`
2. 运行预处理，生成 `processed_papers/`
3. 运行后清洗，生成 `cleaned_papers/`
4. 抽样检查 2 到 3 篇 `cleaned_papers/`
5. 再跑主线

### 3.5 什么时候需要重跑预处理

只有这些情况才建议重跑：

- 新增了原始 PDF
- 修改了 `data/preprocess/` 下的规则
- 当前 `cleaned_papers/` 质量明显影响抽取结果

否则：

- 不要频繁重跑 `processed_papers/` / `cleaned_papers/`
- 直接用已有 `cleaned_papers/` 做主线实验即可

## 4. 环境准备

### 4.1 Python 环境

推荐 Python 3.11。

安装依赖：

```bash
pip install -r requirements.txt
```

### 4.2 环境变量

先复制模板：

```bash
cp .env.example .env
```

再填写：

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
LLM_MAX_RETRIES=3
LLM_RETRY_BASE_DELAY=2
LLM_RETRY_MAX_DELAY=12
```

说明：

- `OPENAI_BASE_URL` 可以替换成你实际使用的兼容 API 地址
- `LLM_MAX_RETRIES` 等参数是可选的
- Neo4j 必须可连通，否则完整主线会停在写库阶段

## 5. 最常用的运行方式

### 5.1 跑完整主线

```bash
python main.py --data-dir data/cleaned_papers --batch-size 5
```

这条会执行：

- 子领域识别
- 每个 batch 自动 refinement
- Extraction
- Grounding
- Validation
- Evolution

### 5.2 从空库开始跑

```bash
python main.py --data-dir data/cleaned_papers --batch-size 5 --clear-db
```

适用场景：

- 你想完全重建图谱
- 你不想受旧 Neo4j 内容影响

### 5.3 只看子领域

```bash
python main.py \
  --data-dir data/cleaned_papers \
  --subdomain-only \
  --subdomain-report reports/subdomain_assignments.csv
```

### 5.4 只看抽取

```bash
python main.py \
  --data-dir data/cleaned_papers \
  --extraction-only \
  --extraction-report reports/extraction_claims.csv
```

注意：

- `extraction-only` 仍会先跑子领域
- 因为 extraction prompt 会用到 `subdomain`

### 5.5 测试 embedding 模型

```bash
python scripts/test_vector_model.py --model BAAI/bge-m3
```

### 5.6 导出 5 篇论文的分阶段结果

这个脚本适合做汇报或人工检查：

```bash
python scripts/export_stage_outputs.py --clear-db
```

它会导出：

- 预处理摘要
- processed / cleaned JSON
- 子领域候选与最终结果
- extraction claims
- grounding claims
- validation claims
- evolution 写库结果

默认输出目录类似：

- `reports/stage_outputs_<timestamp>/`

## 6. 子领域模块怎么理解

当前不是“每篇论文直接决定最终 taxonomy”，而是：

```text
论文
-> 子领域候选
-> batch 结束
-> refinement
-> confirmed subdomain
```

图里会同时看到：

- `Subdomain`
- `SubdomainCandidate`

这不是 bug。  
`SubdomainCandidate` 是过程对象，用来保留候选、归并、提升的轨迹。

## 7. 各阶段输出大概长什么样

### 7.1 预处理

输出位置：

- `data/processed_papers/*.json`
- `data/cleaned_papers/*.json`

### 7.2 子领域

运行时会写到：

- `paper.metadata`
- Neo4j 的 `Paper / Subdomain / SubdomainCandidate`

常见导出：

- `reports/subdomain_assignments.csv`

### 7.3 抽取

输出是 `KnowledgeClaim` 列表，分：

- ontology layer
- instance layer

常见导出：

- `reports/extraction_claims.csv`

### 7.4 对齐

对齐后每条 claim 会有：

- `subject_id`
- `object_id`

### 7.5 验证

验证后每条 claim 会带：

- `status`
- `validation_type`
- `confidence`

### 7.6 入库

通过验证的 claim 会写入 Neo4j。

## 8. 最容易踩的坑

### 8.1 直接用系统 Python 跑

如果 `python3` 指向系统环境，可能会缺：

- `pydantic`
- `neo4j`
- `sentence_transformers`
- `faiss`

所以要确保你在项目依赖环境里运行。

### 8.2 Neo4j 没起

完整主线需要 Neo4j。  
如果数据库没开，主线不会正常完成。

### 8.3 以为 `SubdomainCandidate` 说明 refinement 没跑

不是。  
candidate 可能会被保留作生命周期记录。

### 8.4 多栏综述类 PDF 质量不如病例报告类 PDF

预处理目前对复杂版面并不是完美的。  
如果发现某篇抽取质量很差，先反查 `cleaned_papers/`，再决定是不是要动预处理。

## 9. 建议接手顺序

如果一个新人接手，建议按这个顺序熟悉：

1. 看 `data/cleaned_papers/` 的结构
2. 看 `data_loader.py`
3. 看 `main.py`
4. 看 `agents/subdomain.py` 和 `agents/subdomain_refinement.py`
5. 看 `agents/extraction.py`
6. 看 `agents/grounding.py`
7. 看 `agents/validation.py`
8. 最后看 `core/neo4j_store.py`

## 10. 一句话总结

这个项目现在已经不是“研究脚本堆”，而是一条能跑通的医学文献知识图谱流水线。  
接手时，最重要的是先理解 `data/` 的三层结构，再按 `cleaned_papers -> main.py -> Neo4j` 这条线往下看。
