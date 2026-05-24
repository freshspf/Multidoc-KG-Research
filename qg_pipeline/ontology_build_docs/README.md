# 知识图谱自动构建入口点文档

## 文档概述

本文件夹包含了多智能体知识图谱管道项目中，自动构建知识图谱/本体的代码入口点分析和技术实现指南。

## 文档结构

### 1. [knowledge_graph_construction_entrypoints.md](./knowledge_graph_construction_entrypoints.md)
**主要内容**:
- 详细的代码入口点分析
- 核心架构和工作流程
- 关键文件路径和函数位置
- 如何集成现有知识图谱的建议

**适合人群**:
- 需要了解系统架构的开发者
- 希望集成现有知识图谱的研究人员
- 进行系统扩展和维护的工程师

### 2. [integration_guide.md](./integration_guide.md)
**主要内容**:
- 详细的技术实现方案
- 三种不同的集成策略
- 完整的代码示例和配置方法
- 最佳实践和注意事项

**适合人群**:
- 需要具体实现代码的开发者
- 进行系统集成的工程师
- 希望深度定制的研究人员

## 核心发现

### 主要入口点

1. **管道级入口**: `run_pipeline.py:27`
   - 创建 `MultiAgentKGPipeline` 实例
   - 启动完整的处理流程

2. **工作流级入口**: `workflow.py:350`
   - `extract_knowledge()` 方法
   - 调用 `extractor.process_single_paper()`

3. **提取器级入口**: `agents/extractor/section_based_extraction.py:433`
   - `process_single_paper()` 方法
   - 核心的知识图谱提取逻辑

### 关键组件

- **SectionBasedExtractor**: 基于章节的知识图谱提取器
- **TTLEvaluator**: 知识图谱质量评估器
- **MultiHopQAGenerator**: 多跳推理QA生成器
- **LangGraph StateGraph**: 工作流协调器

## 集成现有知识图谱的策略

### 方案1: 知识图谱扩展模式 (推荐)
- 在现有知识图谱基础上扩展
- 保持一致性和连贯性
- 支持实体复用和关系丰富

### 方案2: 本体驱动模式
- 基于预定义本体结构进行提取
- 确保生成的知识图谱符合特定模式
- 支持严格的数据质量控制

### 方案3: 配置驱动模式
- 通过配置文件控制集成策略
- 灵活的参数化控制
- 易于维护和调整

## 快速开始

### 1. 现有功能使用
```bash
# 基本运行
python run_pipeline.py

# 指定配置
python run_pipeline.py --config config.yaml

# 恢复处理
python run_pipeline.py --resume
```

### 2. 集成现有知识图谱
```python
from agents.extractor.section_based_extraction import SectionBasedExtractor

# 加载现有知识图谱
with open("your_kg.ttl", "r") as f:
    existing_kg = f.read()

# 初始化增强的提取器
extractor = SectionBasedExtractor(
    existing_knowledge_graph=existing_kg,
    enable_kg_extension=True
)

# 处理论文
result = extractor.process_paper_from_text(
    paper_text="论文内容...",
    paper_name="论文标题"
)
```

## 重要文件路径

### 核心代码文件
- `run_pipeline.py` - 主入口脚本
- `workflow.py` - 工作流协调器
- `agents/extractor/section_based_extraction.py` - 知识提取器实现
- `config.yaml` - 主配置文件

### 输入输出目录
- `data/` - 输入论文JSON文件
- `outputs/section_based_extractions/` - 生成的TTL文件
- `outputs/multi_hop_qa/` - 生成的QA对
- `outputs/state/` - 工作流状态和检查点

### 工具和配置
- `agents/extractor/section_based_extraction_prompt.txt` - 提取提示模板
- `utils/helpers.py` - 配置加载和工具函数
- `utils/storage.py` - 状态管理

## 配置要点

### 提取器配置 (config.yaml)
```yaml
extractor:
  model: "deepseek-v3.1"           # 提取模型
  temperature: 0.1                 # 生成温度
  max_tokens: 3000                # 令牌限制
  output_dir: "outputs/section_based_extractions"  # 输出目录
```

### 评估配置
```yaml
evaluator:
  model: "deepseek-v3.1"          # 评估模型
  threshold: 6                     # 通过阈值
```

### 工作流配置
```yaml
workflow:
  batch_size: 1                    # 批处理大小
  max_retries: 3                   # 最大重试次数
```

## 扩展建议

### 1. 立即可实现的扩展
- 在 `SectionBasedExtractor` 构造函数中添加 `existing_knowledge_graph` 参数
- 修改 `extract_single_section` 方法以支持知识图谱上下文
- 在配置文件中添加知识图谱扩展相关配置

### 2. 中期扩展
- 实现智能实体匹配和去重算法
- 添加知识图谱验证和质量检查机制
- 支持增量更新和版本控制

### 3. 长期扩展
- 支持多种知识图谱格式输入输出
- 实现知识图谱演化和管理功能
- 添加可视化和分析工具

## 联系和支持

如有问题或建议，请参考：
- 项目CLAUDE.md文档了解整体架构
- 查看源代码了解具体实现
- 检查配置文件了解可配置选项

## 更新日志

- **2024-12-01**: 初始版本，包含入口点分析和集成指南
- 根据项目进展持续更新