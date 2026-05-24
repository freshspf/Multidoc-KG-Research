# 项目进度跟踪

最后更新：2026-04-11

## 1. 当前目标

将项目主线从“古籍/中医知识抽取”迁移到“医学文献知识抽取”，并按以下顺序逐步推进：

```text
PDF 预处理
-> 论文加载
-> 子领域识别
-> Extraction
-> Grounding
-> Validation
-> Evolution
```

## 2. 当前阶段

当前正在推进：
- 完整主线已经打通到 Neo4j 写库
- 子领域分类已切到“按 batch 自动 refinement”
- 已完成一轮 5 篇 PubMed 样本的分阶段导出

当前判断：
- PDF 预处理已达到“够用版本”，暂不继续深挖
- 子领域层已经不再只是前置标签，而是图谱中的独立层
- Extraction / Grounding / Validation / Evolution 已能串成完整主线

## 3. 已完成事项

### 3.1 数据结构与加载

已完成：
- `schema.py` 支持 `abstract / keywords`
- `Paper.build_classification_text()` 已用于分类任务
- `data_loader.py` 已适配医学论文 JSON
- 默认主入口数据目录已切到医学文献目录

### 3.2 PDF 预处理

已完成：
- `bookmark_based_splitter.py` 已改造成主预处理器
- `clean_processed_papers.py` 已补充首页噪声、尾部噪声、caption 清理
- `processed_papers` 与 `cleaned_papers` 已多轮重跑

当前结论：
- 适合做主线实验
- 不再作为当前阻塞项

### 3.3 子领域识别与 refinement

已完成：
- 新增 `agents/subdomain.py`
- 新增 `config/subdomain_config.yaml`
- 新增 `agents/subdomain_refinement.py`
- 主流程已支持 `--subdomain-only`
- 子领域结果支持导出到 `reports/subdomain_assignments.csv`
- 当标题仅为 PMID 且缺少摘要/关键词时，已加入正文 summary fallback
- 子领域分类已支持 `batch_size`
- 每个 batch 分类完成后，现已自动执行一次 refinement
- Neo4j 中已区分 `Subdomain`(confirmed) 与 `SubdomainCandidate`(candidate)

当前结论：
- 当前版本已从“单独实验”推进到“主线前置模块”
- 子领域层已可直接服务后续 extraction / grounding / validation

### 3.4 主流程轻量与调试模式

已完成：
- `--subdomain-only`
- `--subdomain-report`
- `--subdomain-graph-only`
- `--refine-subdomains-only`
- `--extraction-only`
- `--extraction-report`
- `--vector-model`
- `scripts/test_vector_model.py`
- `scripts/export_stage_outputs.py`

目的：
- 支持按阶段单独调试
- 支持单独测试 embedding 模型加载
- 支持在主线环境不稳定时快速定位问题
- 支持固定 5 篇论文的整链路阶段导出，便于人工检查

## 4. 当前待办

### 高优先级

- 观察完整主线端到端运行日志
- 检查 Neo4j 中正式 taxonomy 与最终 claims 的结构质量

### 中优先级

- 增加图谱中文显示层（如 `name_zh / display_name`）
- 为 Neo4j Browser 提供更稳定的“最终视图”查询

### 低优先级

- 进一步提升复杂双栏 PDF 的版面恢复
- 继续优化 candidate 清理策略，降低 Neo4j 可视化噪声

## 5. 当前实验命令

### 只跑子领域分类

```bash
python main.py \
  --data-dir data/cleaned_papers \
  --subdomain-only \
  --subdomain-report reports/subdomain_assignments.csv
```

### 只跑 Extraction

```bash
python main.py \
  --data-dir data/cleaned_papers \
  --extraction-only \
  --extraction-report reports/extraction_claims.csv
```

说明：
- `extraction-only` 默认仍会先跑子领域分类，然后再跑 extraction
- 这样 extraction 可以直接利用 `subdomain`

### 跑完整主流程

```bash
python main.py --data-dir data/cleaned_papers --batch-size 5
```

### 从空库开始跑完整主流程

```bash
python main.py --data-dir data/cleaned_papers --batch-size 5 --clear-db
```

### 测试 embedding 模型

```bash
python scripts/test_vector_model.py --model BAAI/bge-m3
```

### 导出 5 篇论文的分阶段结果

```bash
python scripts/export_stage_outputs.py --clear-db
```

## 6. 最近更新记录

### 2026-03-26

- 接入了子领域分类模块
- 加入了子领域 CSV 导出
- 修复了 `--subdomain-only` 仍初始化 Neo4j / VectorStore 的问题
- 修复了子领域分类在“标题仅为 PMID”场景下的输入不足问题
- 新增 `--extraction-only` 模式
- 开始将 `subdomain` 显式注入 extraction 上下文
- 删除了旧域残留文件 `config/hdnj_examples.yaml`
- 清理了 `Extraction` 中残留的古籍/中医配置与归一化逻辑
- 删除了 `schema.py` 中残留的 `CorpusType / corpus_type` 古籍字段
- 删除了过时的旧 handoff 文档和旧改进报告，并重写了 `README.md`
- 已将 `validation.py` 和 `grounding.py` 的主提示词迁到医学文献语境

### 2026-04-02

- 新增子领域落库第一阶段方案文档：`reports/子领域融合与关系维护改造方案_2026-04-02.md`
- `core/neo4j_store.py` 已支持 `Paper / Subdomain / CLASSIFIED_AS / SUBCLASS_OF` 基础写库接口
- 完整主流程在进入 Extraction 前，现已可先把论文子领域信息持久化到 Neo4j
- `agents/subdomain.py` 已切到 hierarchy-aware prompt，主逻辑开始读取 Neo4j 中已有的 `Subdomain` 层级
- `config/subdomain_config.yaml` 已瘦身为运行参数 + 轻量回退规则，不再以静态父领域候选作为主逻辑
- 当前选择了“从空图开始”的方案：完整主流程会先清空 Neo4j，再初始化 `Biomedicine` 根节点，再执行子领域分类
- 新增 `--subdomain-graph-only` 模式：只做子领域分类 + Neo4j 写入，便于单独验证子领域层
- 子领域模块已加入“优先复用已有子领域”逻辑：对语义接近的候选标签会自动归并，避免图中出现大量近义节点
- 子领域模块开始优先为新节点选择“更具体的已有父节点”，不再默认全部挂到 `Biomedicine`
- 新增批次化维护方案文档：`reports/批次化子领域维护方案_2026-04-02.md`
- 当前已明确后续大批量运行策略：按批次分类、按批次 refinement、按版本维护 taxonomy
- `schema.py` 已为子领域分配结果加入 `status / is_new_subdomain / batch_id / taxonomy_version`
- `main.py` 已支持 `--batch-size`，子领域分类现在按批次冻结 hierarchy 快照执行
- `core/neo4j_store.py` 已开始区分 `Subdomain`(confirmed) 与 `SubdomainCandidate`(candidate)
- 已新增 `SubdomainHierarchyRefinementAgent` 与 `--refine-subdomains-only` 入口，可单独运行 candidate -> confirmed 的最小 refinement 流程
- 已回到主线推进 `Extraction`：收紧为受控 biomedical relations，增强元叙事/句子型实体过滤，并加入 `Introduction/Methods` section 级降噪
- `config/extraction_config.yaml` 已将 `max_claims_per_chunk` 默认收紧到 `15`，用于压制过抽取
- 已打通主线关键兼容层：Neo4j 英文 relation 写库判型已与当前 biomedical extraction 对齐，完整流程可使用 `--vector-model` 指定 grounding 模型
- 若向量模型初始化失败，主流程现在会自动退回 `MockVectorStore`，保证今晚可先把端到端链路跑通
- `core/llm_client.py` 已加入自动重试与指数退避，用于缓解第三方 API 间歇性连接失败
- `main.py` 已接入“每个 batch 分类结束后自动 refinement”逻辑
- 已新增 prompt 总结文档：`reports/各阶段核心Prompt设计总结_2026-04-03.md`

### 2026-04-11

- 已清理旧日志与旧实验 CSV 导出，避免与新一轮阶段检查混淆
- 新增 `scripts/export_stage_outputs.py`
- 该脚本固定面向 5 篇代表性 PubMed 样本导出：
  - 预处理结果
  - 加载后的论文视图
  - 子领域候选与 refinement 后结果
  - Extraction 输出
  - Grounding 输出
  - Validation 输出
  - Evolution 写库结果
- 当前导出目录约定为：`reports/stage_outputs_<timestamp>/`
- 已实际完成一轮导出：
  - 输出目录：`reports/stage_outputs_20260411_fullrun/`
  - 汇总说明：`reports/五篇PubMed分阶段导出说明_2026-04-11.md`
- 已补充导师汇报版材料：
  - `reports/导师汇报版_五篇PubMed阶段总览_2026-04-11.md`
  - `reports/导师汇报版_五篇PubMed论文卡片_2026-04-11.md`
  - `reports/导师汇报版_五篇PubMed阶段总览_2026-04-11.csv`

## 7. 相关文档

- `reports/session_handoff_2026-03-26_extraction_stage.md`
- `reports/current_experiment_pipeline.md`
- `reports/各阶段核心Prompt设计总结_2026-04-03.md`
- `reports/子领域识别实验分析_2026-03-26.md`
- `reports/抽取三元组实验分析_2026-03-26.md`
- `reports/批次化子领域维护方案_2026-04-02.md`
