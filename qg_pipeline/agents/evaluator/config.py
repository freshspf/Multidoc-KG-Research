#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTL评估器配置文件
包含默认参数和设置
"""

# API配置
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2000

# 评估配置
DEFAULT_THRESHOLD = 0.8
DEFAULT_QUESTION = "评估从学术论文中提取的知识图谱质量"

# 评估维度权重（必须总和为1.0）
DIMENSION_WEIGHTS = {
    "domain_fit": 0.20,      # 领域适配性
    "accuracy": 0.30,        # 准确性  
    "consistency": 0.20,     # 一致性
    "completeness": 0.15,    # 完整性
    "granularity": 0.15      # 粒度
}

# 文件配置
DEFAULT_OUTPUT_SUFFIX = "_evaluation.json"
DEFAULT_ENCODING = "utf-8"

# 日志配置
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# TTL解析配置
TURTLE_BLOCK_START = "```turtle"
TURTLE_BLOCK_END = "```"
SECTION_PATTERN = r'# ===== SECTION: ([^=]+) ====='
SOURCE_SECTION_PATTERN = r':sourceSection\s+"([^"]+)"'

# 实体和关系类型（用于验证）
EXPECTED_ENTITY_TYPES = [
    "Author", "Model", "Language", "Method", "Task", 
    "Organization", "Publication", "Concept", "Dataset",
    "Metric", "Component", "Architecture", "Tool"
]

EXPECTED_RELATION_TYPES = [
    "rdf:type", "partOf", "uses", "trainedOn", "evaluatedOn",
    "appliedTo", "relatedTo", "partiallyOrthogonalTo", "mappedTo",
    "usedFor", "contributesTo", "addresses", "shows"
]

# 质量检查配置
MIN_TRIPLES_PER_SECTION = 5
MIN_ENTITIES_PER_SECTION = 3
MAX_REDUNDANCY_RATE = 0.3
MIN_EVIDENCE_COVERAGE = 0.6

# 评估提示模板变量
PROMPT_VARIABLES = [
    "QUESTION",
    "ONTOLOGY_OR_RULES", 
    "SECTIONED_TURTLE_BLOCK"
]