# TTL知识图谱评估器分析

## 评估器概述

[`TTLEvaluator`](../agents/evaluator/ttl_evaluator.py:33) 是一个专门评估从学术论文提取的TTL格式知识图谱质量的智能体，使用GPT模型进行多维度质量评估。

## 核心功能

### 1. 评估对象
- **TTL文件**: 从学术论文提取的知识图谱
- **格式**: Turtle/RDF格式的三元组数据
- **来源**: SectionBasedExtractor的输出

### 2. 评估目标
- 确保提取的知识图谱符合质量标准
- 为后续QA生成提供质量门槛
- 提供具体的改进建议
- 支持重试机制和持续改进

## 评估维度和方法

### 五个核心评估维度

#### 1. Domain Fit (领域适配) - 权重20%
- **评估内容**: 知识图谱与任务/论文领域的对齐程度
- **章节权重**:
  - Abstract/Conclusion: 侧重高层次相关性
  - Methods/Results: 侧重任务级相关性
- **评分标准**: 0-10分

#### 2. Accuracy (准确性) - 权重30%
- **评估内容**: 事实支持度，边与证据的显式蕴含关系比例
- **特殊处理**:
  - 推测性声明会被扣分（除非在Discussion章节）
  - 需要显式的证据支持
- **评分标准**: 0-10分

#### 3. Consistency (一致性) - 权重20%
- **评估内容**: 跨章节的连贯性
- **检查项**:
  - 矛盾检测 (conflict_count)
  - 单位/时间/方向不匹配
  - 本体违规
- **评分标准**: 0-10分

#### 4. Completeness (完整性) - 权重15%
- **评估内容**: 章节适当的槽位/元数据覆盖度
- **期望内容**:
  - Methods/Results: 数据集/指标/值/版本
  - Abstract: 可能较轻量
- **度量**: missing_slot_rate
- **评分标准**: 0-10分

#### 5. Granularity (粒度) - 权重15%
- **评估内容**: 细节层次的适当性
- **检查项**:
  - 标准化术语
  - 别名解析
  - 模型/数据集版本
  - 指标名称/值
- **注意**: 惩罚系统性过粗/过细的模式
- **评分标准**: 0-10分

### 最终得分计算
```python
final_score = 0.20 * domain_fit +
              0.30 * accuracy +
              0.20 * consistency +
              0.15 * completeness +
              0.15 * granularity
```

## 技术实现

### 核心类和方法

#### 1. TTLEvaluator初始化 ([ttl_evaluator.py:36](../agents/evaluator/ttl_evaluator.py:36))
```python
def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
             threshold: float = 7, config: Dict = None):
    # 模型配置
    self.model = evaluator_config.get('model', 'gpt-4o-mini')
    self.temperature = evaluator_config.get('temperature', 0.1)
    self.max_tokens = evaluator_config.get('max_tokens', 2000)
    self.threshold = evaluator_config.get('threshold', threshold)

    # 加载评估提示模板
    self.prompt_template = self._load_prompt_template()
```

#### 2. 主要评估方法 ([ttl_evaluator.py:181](../agents/evaluator/ttl_evaluator.py:181))
```python
def evaluate_ttl(self, ttl_file_path: str) -> EvaluationResult:
    """全面评估TTL文件"""
    # 提取TTL内容
    ttl_content = self._extract_ttl_content(ttl_file_path)

    # 统计基础信息
    triple_count, entity_count = self._count_triples_and_entities(ttl_content)
    sections = self._extract_sections(ttl_content)

    # 构建评估提示
    prompt = self.prompt_template.format(
        QUESTION="全面评估从学术论文提取的知识图谱质量",
        ONTOLOGY_OR_RULES="标准知识图谱评估标准",
        SECTIONED_TURTLE_BLOCK=ttl_content
    )

    # 调用GPT模型进行评估
    response = self.client.chat.completions.create(...)
```

#### 3. TTL内容解析 ([ttl_evaluator.py:83](../agents/evaluator/ttl_evaluator.py:83))
```python
def _extract_ttl_content(self, ttl_file_path: str) -> str:
    """从TTL文件提取内容"""
    # 提取 ```turtle ``` 块
    # 支持多个turtle块合并
    turtle_blocks = []
    for line in content.split('\n'):
        if line.strip() == '```turtle':
            in_turtle_block = True
        elif line.strip() == '```' and in_turtle_block:
            in_turtle_block = False
            if current_block:
                turtle_blocks.append('\n'.join(current_block))

    return '\n\n'.join(turtle_blocks)
```

### 评估结果结构

#### EvaluationResult数据类 ([ttl_evaluator.py:22](../agents/evaluator/ttl_evaluator.py:22))
```python
@dataclass
class EvaluationResult:
    meta: Dict[str, Any]                    # 元数据统计
    scores: Dict[str, Dict[str, Any]]       # 各维度得分
    final_score: float                      # 最终得分
    summary_advice: str                     # 改进建议
    top_fixes: List[str]                    # 优先修复项目
    passed_threshold: bool                  # 是否通过阈值
```

#### 详细输出格式
```json
{
  "meta": {
    "triple_count": <三元组数量>,
    "entity_count": <实体数量>,
    "section_coverage": {
      "present": ["Abstract", "Methods", "..."],
      "missing": ["Results", "..."]
    },
    "evidence_coverage": <0-1>,
    "conflict_count": <矛盾数量>,
    "redundancy_rate": <0-1>,
    "missing_slot_rate": <0-1>,
    "granularity_notes": ["粒度说明"]
  },
  "scores": {
    "domain_fit": {"score": <0-10>, "reason": "简短原因"},
    "accuracy": {"score": <0-10>, "reason": "简短原因"},
    "consistency": {"score": <0-10>, "reason": "简短原因"},
    "completeness": {"score": <0-10>, "reason": "简短原因"},
    "granularity": {"score": <0-10>, "reason": "简短原因"}
  },
  "final_score": <0-10>,
  "summary_advice": "整体改进建议 (≤120字符)",
  "top_fixes": ["修复建议1", "修复建议2", "修复建议3"]
}
```

## 工作流集成

### 在工作流中的角色 ([workflow.py](../workflow.py))
评估器在知识提取和QA生成之间起到质量控制的作用：

```python
def should_retry_extraction(self, state: WorkflowState) -> WorkflowState:
    """判断是否需要重试提取"""
    evaluation_passed = state.get("evaluation_passed", False)

    if evaluation_passed:
        # 评估通过，进入QA生成阶段
        return {"next_step": "generate_qa"}
    else:
        # 评估失败，使用改进建议重试提取
        improvement_suggestions = state.get("improvement_suggestions", "")
        return {"next_step": "retry_extraction", "improvement_suggestions": improvement_suggestions}
```

### 重试机制
1. **失败处理**: 当评估未通过阈值时，触发重试
2. **改进建议**: 将评估器的改进建议传递给提取器
3. **循环控制**: 最多重试3次，避免无限循环

## 评估标准和阈值

### 默认配置 ([config.yaml](../config.yaml))
```yaml
evaluator:
  model: "deepseek-v3.1"        # 评估模型
  threshold: 6                  # 通过阈值
  temperature: 0.1              # 低温度保证一致性
  max_tokens: 2000             # 响应长度限制
```

### 阈值设置策略
- **严格模式**: threshold=8-9, 仅允许高质量知识图谱进入QA阶段
- **标准模式**: threshold=6-7, 平衡质量和通过率
- **宽松模式**: threshold=4-5, 允许更多知识图谱进入QA阶段

### 质量等级
- **优秀**: 8.0-10.0, 可以直接用于QA生成
- **良好**: 6.0-7.9, 通常可以通过，可能需要小幅改进
- **及格**: 4.0-5.9, 建议重试或大幅改进
- **不合格**: 0.0-3.9, 必须重新提取

## 评估器的优势和局限

### 优势
1. **多维度评估**: 覆盖知识图谱质量的各个重要方面
2. **章节感知**: 考虑不同章节的特点和权重
3. **可配置性**: 支持不同质量标准和阈值
4. **改进导向**: 提供具体的改进建议
5. **自动化**: 无需人工干预的批量评估

### 局限
1. **依赖GPT**: 评估结果受模型能力和一致性的影响
2. **成本考虑**: 每次评估都需要调用API
3. **语言依赖**: 主要针对英文论文优化
4. **主观性**: 某些维度（如粒度）可能存在主观判断

### 改进建议
1. **缓存机制**: 对相似知识图谱的评估结果进行缓存
2. **规则增强**: 结合传统规则检查提高评估一致性
3. **自定义标准**: 支持领域特定的评估标准
4. **批处理优化**: 支持批量知识图谱评估以提高效率

## 使用示例

### 单独使用评估器
```python
from agents.evaluator.ttl_evaluator import TTLEvaluator

# 初始化评估器
evaluator = TTLEvaluator(
    api_key="your-api-key",
    threshold=7.0  # 设置通过阈值
)

# 评估TTL文件
result = evaluator.evaluate_and_save(
    ttl_file_path="path/to/extraction.ttl",
    output_path="path/to/evaluation.json"
)

if result.passed_threshold:
    print(f"评估通过！得分: {result.final_score:.1f}")
    print(f"改进建议: {result.summary_advice}")
else:
    print(f"评估未通过，得分: {result.final_score:.1f}")
    print(f"需要修复: {', '.join(result.top_fixes[:3])}")
```

### 在工作流中使用
评估器由工作流自动调用，用户主要通过配置文件控制：
```yaml
evaluator:
  threshold: 6.5  # 调整通过阈值
  model: "gpt-4"   # 使用更强大的模型
  temperature: 0.1 # 保持低温度
```

## 总结

TTLEvaluator是一个关键的质量控制组件，确保只有高质量的知识图谱进入QA生成阶段。它的多维度评估方法、改进建议机制和可配置性，使其能够适应不同的质量要求和领域需求。通过自动化的评估和重试机制，系统可以持续改进知识图谱的质量。