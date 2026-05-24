# AutoQG-7K 数据集说明文档

## 目录结构

```
autoqg-7k/
├── papers/          # 论文原文文件
├── qa-data/         # QA问答对文件
```

## 文件说明

### 1. Papers 文件夹（原文文件）

**文件命名**: `paper_X.json`（X为重新编号的序号）

**文件结构**:
```json
[
  {
    "id": "章节ID",
    "metadata": {
      "lang": "语言代码（如en）",
      "section_title": "章节标题",
      "page_start": 起始页码,
      "page_end": 结束页码
    },
    "text": "章节文本内容"
  },
  ...
]
```

**字段说明**:
- `id`: 章节标识符（如introduction, relatedwork, experiments等）
- `metadata`: 元数据信息
  - `lang`: 文档语言
  - `section_title`: 章节标题
  - `page_start`: 章节起始页码
  - `page_end`: 章节结束页码
- `text`: 章节的完整文本内容

### 2. QA-Data 文件夹（问答对文件）

**文件命名**: `paper_X_QA_pair.json`（X与对应原文文件编号一致）

**文件结构**:
```json
[
  {
    "section": "章节ID",
    "path": {
      "section": "章节ID",
      "start_entity": "起始实体",
      "end_entity": "结束实体",
      "hops": [
        {
          "from": "起点实体",
          "relation": "关系类型",
          "to": "终点实体",
          "context": "上下文文本",
          "chunk": "文本块编号"
        },
        ...
      ]
    },
    "question": "问题文本",
    "options": {
      "A": "选项A内容",
      "B": "选项B内容",
      "C": "选项C内容",
      "D": "选项D内容"
    },
    "correct_answer": "正确答案（A/B/C/D）",
    "explanation": "答案解释",
    "path_description": "路径描述文本",
    "context": "相关上下文",
    "paper_id": "论文编号"
  },
  ...
]
```

**字段说明**:

- **section**: 问题所属的章节ID
- **path**: 知识路径信息
  - `section`: 章节ID
  - `start_entity`: 推理路径的起始实体
  - `end_entity`: 推理路径的结束实体
  - `hops`: 推理跳数数组，每一跳包含：
    - `from`: 当前跳的起点实体
    - `relation`: 实体间的关系类型
    - `to`: 当前跳的终点实体
    - `context`: 支撑该关系的上下文文本
    - `chunk`: 文本块编号
- **question**: 生成的问题文本
- **options**: 四个选项（A/B/C/D）
- **correct_answer**: 正确答案选项
- **explanation**: 答案的详细解释
- **path_description**: 推理路径的文字描述
- **context**: 所有相关上下文的拼接
- **paper_id**: 对应的论文编号



