# Multidoc-KG-Research

多文档知识图谱构建与多跳问答评测的研究项目集合。从学术论文 PDF 出发，经过知识图谱抽取、质量评估、多跳问答生成，最终评测大模型的科学推理能力。

## 项目总览

```
学术论文 PDF
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  qg_pipeline                                        │
│  ① 章节级知识图谱抽取 (TTL/RDF)                      │
│  ② TTL 质量评估 + 自动重试                           │
│  ③ 多跳推理问答生成                                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  autoqg-7k                                          │
│  QA 基准数据集 (2,278 篇论文 + 多跳选择题)             │
│  21 个 LLM 评测 (o3, GPT-4.1, Qwen, DeepSeek 等)    │
└─────────────────────────────────────────────────────┘
```

## 各项目详情

### Multidoc-KG-new — 知识图谱构建骨架

通用型学术论文知识图谱构建框架，是整条链路的早期原型。

- **4 个 Agent**：抽取 → 语义对齐 → 知识验证 → 图谱演化
- 多线程管道（队列连接），Neo4j 存储，FAISS 向量检索
- 嵌入模型：`all-MiniLM-L6-v2`
- LLM：OpenAI 兼容 API（GPT-4o）

### Multidoc-KG-sub — 生物医学知识图谱构建（生产版）

在 `Multidoc-KG-new` 基础上面向生物医学文献的成熟版本。

- **6 个 Agent**：新增子领域分类 + 子领域动态演化
- 针对 PubMed 文献优化，支持受控关系抽取
- 嵌入模型升级为 `BAAI/bge-m3`
- 支持 `--subdomain-only` / `--extraction-only` CLI 模式
- YAML 配置、指数退避重试、CLAUDE.md 文档

### qg_pipeline — 图谱抽取 + QA 生成管道

用 LangGraph 编排的端到端管道，从论文生成多跳问答。

- **3 个 Agent**：章节抽取 → TTL 质量评估 (1-10 分，<7 分自动重试) → 多跳 QA 生成
- LangGraph StateGraph 状态机编排
- 支持大规模运行 (`run_large_scale.py`，1000+ 论文并发)
- 断点续测、独立模型配置

### autoqg-7k — 多跳问答基准测试

基于知识图谱多跳路径的 QA 基准数据集与 LLM 评测工具。

- 2,278 篇论文 + 对应的多跳推理选择题（4 选 1）
- 评测场景：无原文上下文，纯测模型推理能力
- 已评测 21 个模型：o3 (~37%) > GPT-4.1 (~32%) > 小模型 (~4%)
- 并行 API 调用 + 断点续测

## 技术栈

| 组件 | 技术 |
|------|------|
| 图数据库 | Neo4j |
| 向量检索 | FAISS + sentence-transformers |
| 工作流编排 | LangGraph / 自研线程管道 |
| LLM | OpenAI 兼容 API (GPT-4o, DeepSeek 等) |
| 数据模型 | Pydantic |
| 知识表示 | TTL/RDF |
| 部署 | Docker / docker-compose |

## 快速开始

```bash
# 1. 知识图谱抽取 + QA 生成
cd qg_pipeline
cp .env.example .env  # 填入 API key
python run_pipeline.py

# 2. 大规模运行
python run_large_scale.py

# 3. LLM 评测
cd ../autoqg-7k
python qa_evaluation.py
```

## 目录结构

```
Multidoc-KG-Research/
├── README.md
├── Multidoc-KG-new/      # 通用知识图谱构建骨架
├── Multidoc-KG-sub/      # 生物医学知识图谱构建 (生产版)
├── qg_pipeline/          # 图谱抽取 + QA 生成管道
└── autoqg-7k/            # 多跳问答基准测试
```
