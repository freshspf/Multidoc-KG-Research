# 多篇论文图谱抽取分析报告


---

## 1. 图谱整体质量评估

### 1.1 总体评价

该图谱展现出**优秀的连通性和清晰的层级结构**，是一个"骨架完备、枝叶待展"的结构化知识库。

| 评估维度 | 评分 | 说明 |
|----------|------|------|
| 连通性 | ★★★★★ | 0个孤立节点，完全连通 |
| 层级结构 | ★★★★☆ | IS_A本体挂载策略成功 |
| 语义丰富度 | ★★★☆☆ | 语义关系占比34.7%，偏低 |
| 分类特异性 | ★★★☆☆ | Concept占比50%，偏高 |

### 1.2 核心统计

| 指标 | 数值 |
|------|------|
| 总实体数 | 386 |
| 总关系数 | 190 |
| 平均度数 | 2.92 |
| IS_A关系 | 124条 (65.3%) |
| 语义关系 | 66条 (34.7%) |

### 1.3 本体类型分布

| 本体类别 | 实体数量 | 占比 |
|----------|----------|------|
| Concept | 196 | 50.8% |
| Tool | 44 | 11.4% |
| Metric | 43 | 11.1% |
| Method | 37 | 9.6% |
| Finding | 34 | 8.8% |
| Task | 27 | 7.0% |
| Dataset | 5 | 1.3% |

---

## 2. 上下位词关系分析（IS_A层次结构）

上下位关系表示「实体 IS_A 本体类别」的归属关系：
- `BERT` IS_A `Method`（BERT 是一种方法）
- `GLUE` IS_A `Dataset`（GLUE 是一个数据集）

### 2.1 按本体类别分组

#### Task (27个)
> 定义：研究目标或任务，如image classification、sentiment analysis

**典型实体**：eye blink detection, rehabilitation training, voluntary blink detection, face detection, robot control, navigation performance

#### Method (37个)
> 定义：算法、模型或技术方法，如Transformer、BERT、CNN

**典型实体**：Infrared-Oculography (IR-OG), SSVEP, CSP algorithm, Savitzky-Golay filter, Viola-Jones object detector, K-means method

#### Metric (43个)
> 定义：评估指标，如accuracy、F1-score、BLEU

**典型实体**：ITR, sensitivity, accuracy, time efficiency, 95.35% accuracy, 99% accuracy

#### Dataset (5个)
> 定义：数据集，如ImageNet、MNIST

**典型实体**：Experiment I, Experiment II, blink samples, EAR signal, EEG data

#### Tool (44个)
> 定义：软件框架、硬件设备或系统

**典型实体**：HMD, BCI system, HTC VIVE, EEG acquisition headset, AFFDEX SDK, Eye-Tracker Interfaces

#### Finding (34个)
> 定义：研究发现或结论

**典型实体**：significant frontal gamma correlations, 94.4%, speeding up interaction with computers

#### Concept (196个)
> 定义：通用概念（默认兜底类别）

**典型实体**：EEG signal, Brain-Computer Interface (BCI), gamma-band oscillations, double blinks

---

## 3. 同义词合并情况

### 3.1 执行摘要

| 操作类型 | 处理数量 |
|----------|----------|
| 垃圾回收（孤立节点） | 0 |
| 类型冲突解决 | 0 |
| 同义词合并 | 9 |

### 3.2 合并详情

| 保留实体 | 被合并实体 | 最终类型 |
|----------|------------|----------|
| short blinks | short blinks | Finding |
| SSVEP-based BCI | SSVEP-based BCI | Concept |
| Asynchronous BCI | asynchronous BCI | Concept |
| motor imagery | motor imagery | Concept |
| voluntary blinking | voluntary blinking | Method |
| EOG signal | EOG signal | Concept |
| EOG | EOG | Tool |
| VOG | VOG | Tool |
| long blinks | long blinks | Concept |

**合并规则**：保留关系数最多的实体作为主实体，将被合并实体的所有关系转移后删除。

---

## 4. 核心概念（高连接度实体）

| 排名 | 实体名称 | 本体类别 | 关系数 |
|------|----------|----------|--------|
| 1 | Gamma-band oscillations | Concept | 5 |
| 2 | Haptic preference scores (HPS) | Metric | 4 |
| 3 | HMD | Tool | 2 |
| 4 | SSVEP-based BCI system | Tool | 2 |
| 5 | Brain-Computer Interface | Concept | 2 |
| 6 | Response times | Metric | 2 |
| 7 | Experiment I | Dataset | 2 |

**发现**：`Gamma-band oscillations` 和 `Haptic preference scores` 是跨文档的高频关注点。

---

## 5. 关系类型统计

| 关系类型 | 出现次数 |
|----------|----------|
| HIS_RESEARCH_INTERESTS_INCLUDE | 12 |
| WAS | 10 |
| IS | 8 |
| HAS | 5 |
| USES | 5 |
| IS_PERFORMED_BY | 5 |
| CONSISTS_OF | 3 |
| CONTAINS | 3 |
| 其他 (156种) | ... |

---

## 6. 质量问题与改进建议

### 6.1 主要问题

| 问题 | 现状 | 影响 |
|------|------|------|
| Concept占比过高 | 50.8% | 分类不够精确 |
| 语义关系偏少 | 34.7% | 知识表达不够丰富 |
| 数值实体未独立 | 混入Concept | 类型不清晰 |

### 6.2 改进建议

1. **Schema升级**：增加 `Parameter`（参数）、`Value`（数值）类型
2. **Prompt调优**：明确指示"除非万不得已，不要使用Concept"
3. **关系增强**：鼓励提取 `CAUSES`、`MEASURES`、`OPTIMIZES` 等语义关系

---

## 7. 抽取效果对比分析

### 7.1 评估指标

| 系统 | 关系F1 |
|------|--------|
| 本框架 | 52.9% |
| AutoQG | **72.3%** |
| GPT-o1 | 35.4% |

### 7.2 差距原因分析

| 原因 | 说明 |
|------|------|
| 数值结果未抽取  | 标注中30%是achievesScore关系 |
| 关系数量不足  | 预测数量少于标注 |
| 抽取偏好差异  | 标注侧重实验结果，预测侧重概念关系 |

**标注偏好**：achievesScore(30%), uses(20%), resultIn(13%)

**预测偏好**：IS, DEFINED_AS, IMPROVES, OUTPERFORMS, CONTAINS

**结论**：当前系统设计目标是概念和方法的语义关系，而标注更侧重实验结果和性能对比。

