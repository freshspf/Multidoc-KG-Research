# KG构建方式

#### 1. 每篇论文独立处理
- 一篇论文 → 一个独立的TTL文件
- 输出：`paper_0_extraction.ttl`、`paper_1_extraction.ttl`...
- 每次处理都是全新开始，不继承之前的结果

#### 2. 处理流程
```python
# 论文1处理
process_paper("paper_0.json") → paper_0_extraction.ttl
evaluate_kg("paper_0_extraction.ttl") → 评分9.2分
generate_qa("paper_0_extraction.ttl") → QA对

# 论文2处理 (重新开始，不会累积paper_0的知识)
process_paper("paper_1.json") → paper_1_extraction.ttl
evaluate_kg("paper_1_extraction.ttl") → 评分8.5分
generate_qa("paper_1_extraction.ttl") → QA对
```

#### 3. 三元组格式
```turtle
# 每个三元组都有来源标记，不与其他论文混合
:BlockchainTechnology rdf:type :Technology ;
    :sourceChunk "0" ;
    :sourceSection "introduction" ;
    :contextText "原始文本片段" .
```

#### 4. 输出结构
```
outputs/
├── section_based_extractions/
│   ├── paper_0_extraction.ttl  # 论文1的独立KG
│   ├── paper_1_extraction.ttl  # 论文2的独立KG
│   └── ...
```

## 关键特点

1. **无状态**: 每篇论文处理完就结束，不保存中间知识
2. **独立文件**: 每个论文一个TTL文件，互不干扰
3. **来源追踪**: 每个三元组都知道来自哪篇论文的哪个章节
4. **质量门槛**: 只有高分KG才会生成QA对

## 简单总结

**这个项目不是在构建一个大的统一知识图谱，而是为每篇论文单独创建小型的、独立的知识图谱文件。**