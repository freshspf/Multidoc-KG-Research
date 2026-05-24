# 会话交接文档（2026-03-26，Extraction 阶段前）

这份文档是给后续 AI / 新会话续接用的，尽量把这次对话里已经确认的项目目标、已完成改动、当前实验状态、下一步建议都整理清楚。

---

## 1. 项目当前真实目标

这个项目最初是“古籍 / 中医知识抽取”框架，但当前真实目标已经明确为：

- 面向 **医学文献 / PubMed / PDF 论文** 做知识抽取
- 不再继续围绕古籍语义本身做实验
- 保留原项目已有的工程骨架和多阶段 pipeline
- 当前阶段不是重写整个系统，而是：
  - 先把 PDF 预处理做到“够用”
  - 再逐步把主线模块迁移到医学文献场景

一句话概括：

> 当前项目是在复用原有多阶段抽取框架的前提下，把系统从古籍领域迁移到医学文献知识抽取。

---

## 2. 当前整体流程理解

现在对整个项目的理解已经比较稳定：

`PDF -> 预处理 -> JSON论文 -> 子领域识别 -> Extraction -> Grounding -> Validation -> Evolution -> 写入图谱`

其中：

- **主线** 仍然是 `Extraction / Grounding / Validation / Evolution`
- **子领域识别** 是前置辅助步骤，用来给论文打标签、帮助 prompt 更聚焦

当前推荐实验顺序：

1. 先验证子领域分类是否合理
2. 再单独验证 extraction 输出
3. 然后再接 grounding / validation / evolution

---

## 3. 这次会话里已经完成的主要工作

### 3.1 数据结构与加载层

已经完成：

- `schema.py`
  - `Paper.metadata` 已稳定可用
  - 已支持：
    - `get_abstract()`
    - `get_keywords()`
    - `build_classification_text()`
  - 新增 `SubdomainAssignment`
- `data_loader.py`
  - 已适配医学论文 JSON
  - 会尽量抽取：
    - `title`
    - `abstract`
    - `keywords`
    - `pmid / doi / journal / year / authors / mesh_terms`
    - sections

当前重要补充：

- 当标题只是 PMID 数字、且缺少 `abstract / keywords` 时，
  `Paper.build_classification_text()` 现在会自动从正文前几个实质 section 生成一个 `content summary fallback`
- 这一步对后续子领域分类非常关键，已经证明有效

---

### 3.2 PDF 预处理

已经完成较多改动：

- `data/preprocess/bookmark_based_splitter.py`
- `data/preprocess/clean_processed_papers.py`
- `data/preprocess/preprocess.py`

总体结论：

- 预处理现在已经达到“够用版本”
- 对复杂双栏综述类 PDF 仍不完美
- 但已经不再是当前主阻塞项

当前保留状态：

- `data/raw_data_papers` 保留
- `data/processed_papers` 保留
- `data/cleaned_papers` 保留

判断：

- 这一步现在足够支撑主线实验继续往下走
- 不建议当前再继续深挖 PDF 解析

---

### 3.3 子领域识别模块

这是本次会话里最明确完成的一块。

已完成：

- 新增 `agents/subdomain.py`
  - `SubdomainClassifierAgent`
- 新增 `config/subdomain_config.yaml`
- 主流程已经接入子领域识别
- 结果会写回 `paper.metadata`
- 支持导出 CSV

当前写回字段：

- `subdomain`
- `parent_domain`
- `subdomain_reason`
- `subdomain_confidence`
- `subdomain_new_relations`
- `subdomain_assignment`

新增运行方式：

- `--subdomain-only`
- `--subdomain-report`

例如：

```bash
python main.py \
  --data-dir data/cleaned_papers \
  --subdomain-only \
  --subdomain-report reports/subdomain_assignments.csv
```

---

## 4. 子领域识别实验结果总结

### 4.1 第一轮结果

第一轮结果问题很明显：

- 多篇论文被误判成：
  - `biomedical literature indexing`
  - `radiology and imaging`
  - `neuroimaging biomarkers`

根因不是分类器本身完全失效，而是：

- 论文标题经常只是 PMID 数字
- 很多样本没有 `abstract`
- 也没有 `keywords`
- 于是分类器实际拿到的输入过弱

---

### 4.2 第二轮结果

在加入 `content summary fallback` 后，第二轮结果明显改善。

当前这一版 `reports/subdomain_assignments.csv` 的结果已经基本可用，例如：

- `39669840 -> liver cancer immunotherapy`
- `39670055 -> oncology shared decision making`
- `39670162 -> cancer nanotechnology`
- `39671082 -> cancer liquid biopsy`
- `39672820 -> cancer drug discovery`
- `39677122 -> testicular germ cell tumor oncology`
- `39677775 -> breast cancer dna repair dysfunction`
- `39687165 -> single-cell rna sequencing computational methods`
- `39764068 -> cancer drug delivery systems`
- `39767185 -> lymphedema imaging and therapy`

当前判断：

- 这版结果已经从“不可评估”提升到“基本可评估”
- 已经足够继续推进到 extraction

残留观察：

- `parent_domain` 仍然偏向 `oncology`
- 但考虑当前样本本身癌症相关较多，这不一定是错

---

## 5. Extraction 当前状态

### 5.1 已完成的迁移

`agents/extraction.py` 已经不再是原始古籍 prompt，当前已完成这些工作：

- 核心 prompt 已迁到医学文献语境
- few-shot 默认示例已经是 biomedical 风格
- chunk 逻辑已更适配论文 section 文本
- 基础过滤逻辑已偏向 biomedical claims

另外，本次会话进一步做了这些增强：

- 显式构造 paper-level extraction context
  - 包含：
    - `Title`
    - `Assigned subdomain`
    - `Parent domain`
    - `Abstract`
    - `Keywords`
    - 必要时 `Content summary`
- 在 extraction prompt 里明确要求：
  - `subdomain` 只能作为 soft prior
  - 不能据此臆造三元组
- 每条抽取出的 `KnowledgeClaim` 会附加 metadata，包含：
  - `paper_title`
  - `chunk_index`
  - `paper_subdomain`
  - `paper_parent_domain`
  - `section_title`（如果可推断）

---

### 5.2 当前仍然存在的问题

虽然 extraction 已明显迁到新域，但仍有残留问题：

1. `agents/extraction.py` 里还有一些旧域的中文 relation / entity normalization 分支没有彻底清干净
2. 目前还没有正式跑过一轮真实的 `extraction-only` 结果检查
3. 还没有验证：
   - 提取出来的 ontology / instance claims 是否合理
   - 证据字段是否足够稳定
   - 噪音过滤是否还会误伤

所以当前更准确的判断是：

- Extraction 已经“进入可测试状态”
- 但还没有完成效果验证

---

## 6. 主流程轻量入口

本次会话已经补了轻量实验入口，避免在做单点验证时误初始化后续依赖。

### 6.1 子领域分类单独运行

```bash
python main.py \
  --data-dir data/cleaned_papers \
  --subdomain-only \
  --subdomain-report reports/subdomain_assignments.csv
```

这个模式下：

- 不会初始化 Neo4j
- 不会初始化 VectorStore
- 不会下载 `BAAI/bge-m3`
- 不会进入 extraction / grounding / validation / evolution

之前出现过一次 bug：

- `--subdomain-only` 仍然初始化了 Neo4j 和 VectorStore
- 已修复

---

### 6.2 Extraction 单独运行

新增：

- `--extraction-only`
- `--extraction-report`

设计目标是：

- 先跑子领域分类
- 再单独跑 extraction
- 结果导出到 CSV
- 不进入 grounding / validation / evolution

推荐命令：

```bash
python main.py \
  --data-dir data/cleaned_papers \
  --extraction-only \
  --extraction-report reports/extraction_claims.csv
```

当前状态：

- 代码入口已经补上
- 但还没有跑过真实结果验证

---

## 7. 当前仓库清理状态

本次会话里已经做过一次清理，删除了：

- 根目录旧导出 `graph_export_*.xlsx`
- `results/` 里旧实验结果
- `evaluation/` 里旧评估产物
- 旧报告输出
- `__pycache__`
- 杂项临时文件

当前保留：

- 代码
- 配置
- 原始 PDF
- `processed_papers`
- `cleaned_papers`
- 当前 handoff / pipeline / progress 文档

---

## 8. 当前最重要的文档

建议后续 AI 先看这几个文件：

1. `reports/current_experiment_pipeline.md`
   - 当前整体流程图和模块说明
2. `reports/progress_tracker.md`
   - 当前阶段、待办和运行命令
3. `reports/subdomain_assignments.csv`
   - 当前子领域分类结果

---

## 9. 下一步最合理的工作顺序

如果是后续 AI 接手，建议按这个顺序继续：

1. 先跑一轮 `--extraction-only`
2. 检查 `reports/extraction_claims.csv`
   - claim 数量是否合理
   - ontology / instance 区分是否合理
   - evidence 是否可核查
   - section_title / subdomain metadata 是否带上
3. 如果 extraction 输出基本可用，再进入：
   - `grounding.py`
   - `validation.py`
4. 不建议当前再把时间主要花在 PDF 预处理上

---

## 10. 一句话交接结论

当前项目状态可以概括为：

> PDF 预处理已经够用，子领域识别已接入并完成第一轮有效验证，现在最合理的主线是正式验证 Extraction 输出，而不是继续回头修 PDF。
