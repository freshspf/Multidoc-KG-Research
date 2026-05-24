# 知识图谱自动构建入口点分析

## 项目概述

这个项目是一个多智能体知识图谱管道，使用LangGraph协调三个专门的智能体来从学术论文中提取、评估和生成QA对。系统最终生成TTL格式的知识图谱，支持多跳推理。

## 核心架构

### 1. 主要智能体
- **SectionBasedExtractor**: 基于章节的知识图谱提取器
- **TTLEvaluator**: 知识图谱质量评估器
- **MultiHopQAGenerator**: 多跳推理QA生成器

### 2. 工作流协调
- **LangGraph StateGraph**: 管理智能体协调和条件路由
- **状态管理**: 通过 `utils/storage.py` 进行持久化状态跟踪
- **重试逻辑**: 自动重试并提供改进建议

## 主要入口点

### 1. 管道主入口
**文件**: [`run_pipeline.py`](../run_pipeline.py:27)
```python
def main():
    # 行27-84
    parser = argparse.ArgumentParser(description='Run multi-agent KG pipeline')
    # ...
    pipeline = MultiAgentKGPipeline(config_path=args.config)  # 行83
    result = pipeline.run_batch(resume_from_checkpoint=args.resume)  # 行84
```

### 2. 大规模处理入口
**文件**: [`run_large_scale.py`](../run_large_scale.py)
- 用于处理1000+论文的优化版本
- 支持并行批次处理

### 3. 工作流协调器
**文件**: [`workflow.py`](../workflow.py:66)
```python
class MultiAgentKGPipeline:
    def __init__(self, config_path: str = "config.yaml"):  # 行67
        self.config = load_config(config_path)
        self.state_manager = StateManager(self.config)
        self._initialize_agents()  # 行100
```

## 知识图谱构建核心

### 1. SectionBasedExtractor 智能体

**实现文件**: [`agents/extractor/section_based_extraction.py`](../agents/extractor/section_based_extraction.py:17)

#### 初始化
```python
class SectionBasedExtractor:
    def __init__(self, output_dir: str = None, config: Dict = None):  # 行18-50
        # API配置
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = os.getenv('OPENAI_BASE_URL')

        # 模型配置 (从config.yaml读取)
        self.model = extractor_config.get('model', 'gpt-4o-mini')  # 行38
        self.temperature = extractor_config.get('temperature', 0.1)   # 行39
        self.max_tokens = extractor_config.get('max_tokens', 3000)   # 行40

        # 加载提取提示
        prompt_file = script_dir / 'section_based_extraction_prompt.txt'  # 行48
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.section_prompt = f.read()  # 行50
```

#### 核心方法

**1. 主入口方法**: [`process_single_paper()`](../agents/extractor/section_based_extraction.py:433)
```python
def process_single_paper(self, paper_file: str, evaluation_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """处理单个论文文件 - 工作流集成的主接口"""
    if not Path(paper_file).exists():
        return {"error": f"File not found: {paper_file}", "paper_file": paper_file}

    return self.process_paper(paper_file, evaluation_result)  # 行438
```

**2. 直接文本处理方法**: [`process_paper_from_text()`](../agents/extractor/section_based_extraction.py:440)
```python
def process_paper_from_text(self, paper_text: str, paper_name: str = "unknown",
                           evaluation_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """直接从文本内容处理论文"""
    # 获取改进建议
    improvement_suggestions = ""
    if evaluation_result and not evaluation_result.get("passed", True):
        improvement_suggestions = evaluation_result.get("suggestions", "")

    # 执行知识提取
    knowledge_graph = self.extract_section_knowledge(paper_text, metadata, improvement_suggestions)  # 行460
```

**3. 基于章节的知识提取**: [`extract_section_knowledge()`](../agents/extractor/section_based_extraction.py:183)
```python
def extract_section_knowledge(self, paper_text: str, metadata: Dict, improvement_suggestions: str = "") -> str:
    """使用逐章节方法提取知识然后合并"""

    # 解析章节
    sections = self.parse_sections_from_text(paper_text)  # 行192

    # 逐个提取每个章节的知识
    for section_info in sections:  # 行203-222
        section_name = section_info['name']
        content = section_info['content']

        # 提取单个章节的知识
        section_kg = self.extract_single_section(content, section_name, chunk_id, metadata, improvement_suggestions)

        all_knowledge_graphs.append(f"# ===== SECTION: {section_name.upper()} =====\n{section_kg}")

    # 合并所有知识图谱
    combined_kg = "\n\n".join(all_knowledge_graphs)  # 行224
    return combined_kg
```

**4. 单章节提取**: [`extract_single_section()`](../agents/extractor/section_based_extraction.py:117)
```python
def extract_single_section(self, section_text: str, section_name: str, chunk_id: int,
                          metadata: Dict, improvement_suggestions: str = "") -> str:
    """从单个章节提取知识"""

    # 准备改进指导
    if improvement_suggestions:
        improvement_guidance = f"""
IMPROVEMENT GUIDANCE (from evaluator):
{improvement_suggestions}
Please incorporate these suggestions in your extraction to improve the quality of the knowledge graph."""

    # 构建系统提示
    system_prompt = f"""You are an expert knowledge extractor...
{self.section_prompt}
TARGET: 20-50 triples from this section alone."""

    # 调用API提取
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return self.call_openai_api(messages, max_tokens=3000)  # 行181
```

### 2. 工作流集成

**提取节点**: [`workflow.py:extract_knowledge()`](../workflow.py:350)
```python
def extract_knowledge(self, state: WorkflowState) -> WorkflowState:
    """从论文提取知识图谱"""
    paper_name = state["current_paper"]
    paper_path = state["current_paper_path"]

    # 准备评估结果用于改进建议
    evaluation_result = None
    if state.get("improvement_suggestions"):
        evaluation_result = {
            "passed": False,
            "suggestions": state["improvement_suggestions"]
        }

    # 提取知识 - 关键调用点
    result = self.extractor.process_single_paper(paper_path, evaluation_result)  # 行368

    if result.get("success"):
        state["extraction_file"] = result["saved_file"]
        # 更新论文状态...
```

## 如何集成现有知识图谱/本体

### 当前系统分析

**现有状态**: 目前的 `SectionBasedExtractor` 没有直接支持接收现有知识图谱/本体的参数。系统主要设计为从论文文本中从头提取知识图谱。

### 集成方案

为了支持将现有知识图谱或本体传递到提取过程中，需要在以下几个关键点进行修改：

#### 1. 修改 SectionBasedExtractor 构造函数

在 [`agents/extractor/section_based_extraction.py:18`](../agents/extractor/section_based_extraction.py:18) 添加参数：

```python
def __init__(self, output_dir: str = None, config: Dict = None,
             existing_knowledge_graph: str = None, ontology: Dict = None):
    # 现有代码...

    # 新增：加载现有知识图谱
    self.existing_knowledge_graph = existing_knowledge_graph
    self.ontology = ontology  # 可以包含预定义的实体类型、关系类型等

    # 新增：基于本体修改提取提示
    if ontology:
        self.section_prompt = self._enhance_prompt_with_ontology(self.section_prompt, ontology)
```

#### 2. 增强 extract_single_section 方法

在 [`agents/extractor/section_based_extraction.py:117`](../agents/extractor/section_based_extraction.py:117) 中添加现有知识图谱上下文：

```python
def extract_single_section(self, section_text: str, section_name: str, chunk_id: int,
                          metadata: Dict, improvement_suggestions: str = "") -> str:

    # 新增：添加现有知识图谱上下文
    existing_context = ""
    if self.existing_knowledge_graph:
        existing_context = f"""
EXISTING KNOWLEDGE GRAPH CONTEXT:
{self.existing_knowledge_graph}

Please extend and enrich this existing knowledge graph with new information from the current section.
Maintain consistency with existing entities and relationships."""

    # 修改系统提示
    system_prompt = f"""You are an expert knowledge extractor...
{self.section_prompt}
{existing_context}
TARGET: 20-50 triples from this section alone."""
```

#### 3. 修改工作流初始化

在 [`workflow.py:105`](../workflow.py:105) 中传递现有知识图谱：

```python
def _initialize_agents(self):
    """初始化智能体"""
    extractor_config = self.config.get('extractor', {})

    # 新增：加载现有知识图谱
    existing_kg_path = extractor_config.get('existing_knowledge_graph_path')
    existing_kg = None
    if existing_kg_path and Path(existing_kg_path).exists():
        with open(existing_kg_path, 'r', encoding='utf-8') as f:
            existing_kg = f.read()

    # 新增：加载本体配置
    ontology_path = extractor_config.get('ontology_path')
    ontology = None
    if ontology_path and Path(ontology_path).exists():
        with open(ontology_path, 'r', encoding='utf-8') as f:
            ontology = json.load(f)

    self.extractor = SectionBasedExtractor(
        output_dir=extractor_config.get('output_dir'),
        config=self.config,
        existing_knowledge_graph=existing_kg,
        ontology=ontology
    )
```

#### 4. 配置文件修改

在 [`config.yaml`](../config.yaml) 中添加配置项：

```yaml
extractor:
  model: "deepseek-v3.1"
  temperature: 0.1
  max_tokens: 3000
  # 新增：现有知识图谱路径
  existing_knowledge_graph_path: "path/to/existing/kg.ttl"
  # 新增：本体配置路径
  ontology_path: "path/to/ontology.json"
  # 新增：是否启用知识图谱扩展模式
  enable_kg_extension: true
```

## 关键文件路径

### 核心实现文件
- **主入口**: [`run_pipeline.py`](../run_pipeline.py:27)
- **工作流协调**: [`workflow.py`](../workflow.py:66)
- **知识提取器**: [`agents/extractor/section_based_extraction.py`](../agents/extractor/section_based_extraction.py:17)
- **提取提示**: [`agents/extractor/section_based_extraction_prompt.txt`](../agents/extractor/section_based_extraction_prompt.txt:1)

### 配置和输出
- **主配置**: [`config.yaml`](../config.yaml:23)
- **输出目录**: `outputs/section_based_extractions/`
- **状态管理**: `outputs/state/`

### 工具和实用程序
- **配置加载**: [`utils/helpers.py`](../utils/helpers.py)
- **状态管理**: [`utils/storage.py`](../utils/storage.py)

## 使用示例

### 1. 基本使用（现有功能）
```bash
# 运行管道
python run_pipeline.py

# 指定配置
python run_pipeline.py --config my_config.yaml

# 恢复检查点
python run_pipeline.py --resume
```

### 2. 集成现有知识图谱（建议的修改）
```python
# 直接使用提取器
from agents.extractor.section_based_extraction import SectionBasedExtractor

# 加载现有知识图谱
with open("existing_kg.ttl", "r") as f:
    existing_kg = f.read()

# 加载本体配置
with open("ontology.json", "r") as f:
    ontology = json.load(f)

# 初始化提取器
extractor = SectionBasedExtractor(
    config=config,
    existing_knowledge_graph=existing_kg,
    ontology=ontology
)

# 处理论文
result = extractor.process_paper_from_text(
    paper_text="论文内容...",
    paper_name="论文标题"
)
```

## 总结

当前系统的知识图谱构建入口点主要集中在：

1. **管道级入口**: `run_pipeline.py` → `MultiAgentKGPipeline`
2. **工作流级入口**: `workflow.py` → `extract_knowledge()`
3. **提取器级入口**: `section_based_extraction.py` → `process_single_paper()`

要集成现有知识图谱/本体，主要需要修改 `SectionBasedExtractor` 类的构造函数和核心提取方法，以及在配置和工作流中添加相应的支持。这样可以在现有提取能力的基础上，实现知识图谱的扩展和丰富。