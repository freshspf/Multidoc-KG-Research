# 多论文知识图谱构建框架

这是一个基于多智能体系统的知识图谱构建框架，用于从学术论文中提取、验证和演化知识。

## 架构概述

该系统采用 "LLM-as-a-Judge" 模式，使用大型语言模型验证小型模型的输出。系统由4个主要智能体组成，按顺序处理数据：

1. **Extraction Agent (提取智能体)**: 对论文进行分块并提取"知识声明"
2. **Semantic Grounding Agent (语义对齐智能体)**: 使用向量搜索 + LLM判断将实体与现有图谱对齐
3. **Knowledge Validation Agent (知识验证智能体)**: 使用LLM判断检查与历史知识的逻辑冲突
4. **Knowledge Evolution Agent (知识演化智能体)**: 处理知识图谱数据库的版本化写入

## 文件结构

```
project_root/
├── schema.py           # Pydantic数据模型
├── main.py             # 主编排入口点
├── requirements.txt    # Python依赖
├── README.md           # 项目文档
├── agents/             # 智能体模块
│   ├── __init__.py
│   ├── extraction.py   # 提取智能体
│   ├── grounding.py    # 语义对齐智能体
│   ├── validation.py   # 知识验证智能体
│   └── evolution.py    # 知识演化智能体
└── core/               # 外部服务的模拟接口
    ├── __init__.py
    ├── llm_client.py   # LLM客户端接口
    └── graph_store.py  # 图数据库存储接口
```

## 安装

1. 安装Python依赖：

```bash
pip install -r requirements.txt
```

2. (可选) 配置环境变量：

如果要使用真实的OpenAI API，创建 `.env` 文件：

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选
```

## 运行

运行主程序：

```bash
python main.py
```

该程序将：
1. 初始化所有模拟依赖
2. 初始化所有4个智能体
3. 创建一个虚拟论文对象
4. 运行顺序处理流水线：提取 → 对齐 → 验证 → 演化
5. 打印处理摘要

## 技术规范

- **Python版本**: 3.10+
- **类型提示**: 使用完整的Python类型提示
- **数据模型**: 使用Pydantic进行数据验证
- **依赖注入**: 所有智能体使用依赖注入模式
- **架构模式**: LLM-as-a-Judge模式

## 数据模型

### KnowledgeClaim (知识声明)

```python
{
    "subject": str,           # 主体实体
    "relation": str,          # 关系/谓词
    "object": str,            # 客体实体
    "evidence": str,          # 支持证据
    "source_paper_id": str,   # 来源论文ID
    "status": ClaimStatus,    # 处理状态
    "grounded_ids": List[str] # 对齐后的图谱ID (可选)
}
```

### ClaimStatus (声明状态)

- `EXTRACTED`: 已提取
- `GROUNDED`: 已对齐
- `VALIDATED`: 已验证
- `REJECTED`: 已拒绝

## 开发说明

当前实现是"骨架优先"版本：
- 所有智能体类已定义
- 所有方法签名已完成
- 使用模拟实现进行演示
- 可以立即运行而不会出错

下一步开发可以：
1. 实现真实的LLM调用逻辑
2. 集成真实的图数据库（如Neo4j）
3. 实现向量搜索功能
4. 添加日志和监控
5. 添加单元测试和集成测试

## 许可证

待定
