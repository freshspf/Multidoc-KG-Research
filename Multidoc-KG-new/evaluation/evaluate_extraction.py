#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱抽取效果评估脚本
比较标注数据（TTL文件）与框架抽取结果（CSV文件）
计算 Precision, Recall, F1 等指标
"""

import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Triple:
    """三元组"""
    subject: str
    predicate: str
    obj: str
    
    def __hash__(self):
        return hash((self.subject.lower(), self.predicate.lower(), self.obj.lower()))
    
    def __eq__(self, other):
        if not isinstance(other, Triple):
            return False
        return (self.subject.lower() == other.subject.lower() and
                self.predicate.lower() == other.predicate.lower() and
                self.obj.lower() == other.obj.lower())


@dataclass
class Entity:
    """实体"""
    name: str
    entity_type: str = ""
    
    def __hash__(self):
        return hash(self.name.lower())
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.name.lower() == other.name.lower()


def parse_ttl_file(ttl_path: str) -> Tuple[Set[Entity], Set[Triple]]:
    """
    解析TTL文件，提取实体和三元组
    
    Args:
        ttl_path: TTL文件路径
        
    Returns:
        (entities, triples): 实体集合和三元组集合
    """
    entities = set()
    triples = set()
    
    with open(ttl_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 尝试提取turtle代码块，如果没有则使用整个内容
    turtle_match = re.search(r'```turtle\n(.*?)```', content, re.DOTALL)
    if turtle_match:
        turtle_content = turtle_match.group(1)
    else:
        # 没有代码块包裹，直接使用内容（跳过注释行）
        lines = content.split('\n')
        turtle_lines = [l for l in lines if not l.strip().startswith('#') and l.strip()]
        turtle_content = '\n'.join(turtle_lines)
    
    # 解析三元组
    # 格式: :Subject :predicate :Object ;
    #       :contextText "..." .
    
    # 逐行解析，识别主语和谓语-宾语对
    current_subject = None
    
    for line in turtle_content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # 检查是否是新主语开始（以:开头，不以;结尾的前一行）
        # 格式: :Subject :predicate :Object ;
        subject_match = re.match(r'^:([A-Za-z0-9_]+)\s+:(\w+)\s+(.+?)(?:\s*[;.])?$', line)
        
        if subject_match:
            current_subject = subject_match.group(1)
            predicate = subject_match.group(2)
            obj_part = subject_match.group(3).strip()
            
            # 跳过contextText
            if predicate == 'contextText':
                continue
            
            # 解析宾语
            obj = None
            if obj_part.startswith(':'):
                obj_match = re.match(r':([A-Za-z0-9_]+)', obj_part)
                if obj_match:
                    obj = obj_match.group(1)
            elif obj_part.startswith('"'):
                obj_match = re.match(r'"([^"]*)"', obj_part)
                if obj_match:
                    obj = obj_match.group(1)
            
            if current_subject and obj:
                # 转换驼峰命名为更可读的形式
                subj_readable = camel_to_words(current_subject)
                obj_readable = camel_to_words(obj)
                pred_readable = camel_to_words(predicate)
                
                entities.add(Entity(name=subj_readable))
                entities.add(Entity(name=obj_readable))
                triples.add(Triple(subject=subj_readable, predicate=pred_readable, obj=obj_readable))
        
        # 检查是否是续行（以:predicate开头）
        elif current_subject:
            cont_match = re.match(r'^:(\w+)\s+(.+?)(?:\s*[;.])?$', line)
            if cont_match:
                predicate = cont_match.group(1)
                obj_part = cont_match.group(2).strip()
                
                if predicate == 'contextText':
                    continue
                
                obj = None
                if obj_part.startswith(':'):
                    obj_match = re.match(r':([A-Za-z0-9_]+)', obj_part)
                    if obj_match:
                        obj = obj_match.group(1)
                elif obj_part.startswith('"'):
                    obj_match = re.match(r'"([^"]*)"', obj_part)
                    if obj_match:
                        obj = obj_match.group(1)
                
                if obj:
                    subj_readable = camel_to_words(current_subject)
                    obj_readable = camel_to_words(obj)
                    pred_readable = camel_to_words(predicate)
                    
                    entities.add(Entity(name=obj_readable))
                    triples.add(Triple(subject=subj_readable, predicate=pred_readable, obj=obj_readable))
    
    return entities, triples


def camel_to_words(name: str) -> str:
    """将驼峰命名转换为空格分隔的单词"""
    # 在大写字母前添加空格
    result = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    # 处理连续大写字母（如 DINO -> DINO）
    result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', result)
    # 移除下划线
    result = result.replace('_', ' ')
    return result


def parse_csv_files(nodes_path: str, relations_path: str) -> Tuple[Set[Entity], Set[Triple]]:
    """
    解析CSV文件，提取实体和三元组
    
    Args:
        nodes_path: 节点CSV文件路径
        relations_path: 关系CSV文件路径
        
    Returns:
        (entities, triples): 实体集合和三元组集合
    """
    entities = set()
    triples = set()
    
    # 解析节点
    with open(nodes_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('name', '').strip()
            entity_type = row.get('entity_type', '').strip()
            labels = row.get('labels', '').strip()
            
            if name and 'Ontology' not in labels:
                entities.add(Entity(name=name, entity_type=entity_type))
    
    # 解析关系
    with open(relations_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_name = row.get('source_name', '').strip()
            target_name = row.get('target_name', '').strip()
            relation_type = row.get('relation_type', '').strip()
            
            # 跳过IS_A关系（本体层次关系，不是知识关系）
            if relation_type == 'IS_A':
                continue
            
            if source_name and target_name and relation_type:
                triples.add(Triple(
                    subject=source_name,
                    predicate=relation_type,
                    obj=target_name
                ))
    
    return entities, triples


def normalize_name(name: str) -> str:
    """规范化实体名称，用于模糊匹配"""
    # 转小写
    name = name.lower()
    # 移除特殊字符
    name = re.sub(r'[^a-z0-9]', '', name)
    return name


def get_name_variants(name: str) -> Set[str]:
    """获取实体名称的各种变体用于匹配"""
    variants = set()
    normalized = normalize_name(name)
    variants.add(normalized)
    
    # 添加去除常见后缀的变体
    for suffix in ['encoder', 'model', 'method', 'system', 'dataset', 'score', 'task', 'based', 'attention']:
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 3:
            variants.add(normalized[:-len(suffix)])
    
    # 添加缩写变体（取每个单词首字母）
    words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
    if len(words) > 1:
        acronym = ''.join(w[0].lower() for w in words if w)
        if len(acronym) >= 2:
            variants.add(acronym)
    
    # 处理常见缩写扩展
    abbreviations = {
        'knwl': 'knowledge',
        'attn': 'attention',
        'self': 'self',
        'si': 'softmaxinterpolation',
        'mca': 'multichannelattention',
        'kisa': 'knowledgeinformedselfattention',
        'cnn': 'convolutionalneuralnetwork',
        'rnn': 'recurrentneuralnetwork',
        'lstm': 'longshorttermmemory',
        'bert': 'bert',
        'gpt': 'gpt',
        'llm': 'largelanguagemodel',
        'nlp': 'naturallanguageprocessing',
        'pos': 'partofspeech',
        'ner': 'namedentityrecognition',
        'gcn': 'graphconvolutionalnetwork',
    }
    
    # 检查是否包含缩写，扩展它们
    for abbr, full in abbreviations.items():
        if abbr in normalized:
            expanded = normalized.replace(abbr, full)
            variants.add(expanded)
        if full in normalized:
            contracted = normalized.replace(full, abbr)
            variants.add(contracted)
    
    return variants


def names_match(name1: str, name2: str, min_len: int = 3) -> bool:
    """检查两个名称是否匹配（宽松匹配）"""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    
    # 空字符串不匹配
    if not n1 or not n2:
        return False
    
    # 精确匹配
    if n1 == n2:
        return True
    
    # 长度过短不做子串匹配
    if len(n1) < min_len or len(n2) < min_len:
        return False
    
    # 子串匹配（一方包含另一方）
    if n1 in n2 or n2 in n1:
        return True
    
    # 变体匹配
    v1 = get_name_variants(name1)
    v2 = get_name_variants(name2)
    if v1 & v2:  # 有交集
        return True
    
    # 检查变体的子串匹配
    for var1 in v1:
        for var2 in v2:
            if len(var1) >= min_len and len(var2) >= min_len:
                if var1 in var2 or var2 in var1:
                    return True
    
    # 核心词匹配：检查主要实体词是否相同
    # 提取核心词（去除修饰词）
    core_words1 = set(re.findall(r'[a-z]{4,}', n1))  # 4字符以上的词
    core_words2 = set(re.findall(r'[a-z]{4,}', n2))
    
    # 如果有2个以上的核心词重叠，认为匹配
    common_cores = core_words1 & core_words2
    if len(common_cores) >= 2:
        return True
    
    # 如果有1个核心词重叠且该词足够长（6字符以上），也认为匹配
    if common_cores and any(len(w) >= 6 for w in common_cores):
        return True
    
    return False


def fuzzy_match_entities(gold_entities: Set[Entity], pred_entities: Set[Entity], 
                         threshold: float = 0.8) -> Tuple[int, Set[Entity], Set[Entity]]:
    """
    模糊匹配实体（宽松匹配）
    
    Args:
        gold_entities: 标注实体集合
        pred_entities: 预测实体集合
        threshold: 匹配阈值
        
    Returns:
        (matched_count, unmatched_gold, unmatched_pred)
    """
    matched_gold = set()
    matched_pred = set()
    
    for gold_ent in gold_entities:
        if gold_ent in matched_gold:
            continue
        
        for pred_ent in pred_entities:
            if pred_ent in matched_pred:
                continue
            
            # 使用宽松匹配
            if names_match(gold_ent.name, pred_ent.name, min_len=3):
                matched_gold.add(gold_ent)
                matched_pred.add(pred_ent)
                break
    
    unmatched_gold = gold_entities - matched_gold
    unmatched_pred = pred_entities - matched_pred
    
    return len(matched_gold), unmatched_gold, unmatched_pred


def fuzzy_match_triples(gold_triples: Set[Triple], pred_triples: Set[Triple]) -> Tuple[int, Set[Triple], Set[Triple]]:
    """
    模糊匹配三元组（只要实体对匹配即可，忽略谓语）
    使用宽松的实体匹配策略
    
    Returns:
        (matched_count, unmatched_gold, unmatched_pred)
    """
    matched_gold = set()
    matched_pred = set()
    
    for gold_t in gold_triples:
        # 跳过太短的实体
        if len(normalize_name(gold_t.subject)) <= 2 or len(normalize_name(gold_t.obj)) <= 2:
            continue
        
        for pred_t in pred_triples:
            # 使用宽松匹配检查主语和宾语
            subj_match = names_match(gold_t.subject, pred_t.subject, min_len=3)
            obj_match = names_match(gold_t.obj, pred_t.obj, min_len=3)
            
            # 只要主语和宾语都匹配，关系就算正确
            if subj_match and obj_match:
                matched_gold.add(gold_t)
                matched_pred.add(pred_t)
                break
    
    unmatched_gold = gold_triples - matched_gold
    unmatched_pred = pred_triples - matched_pred
    
    return len(matched_gold), unmatched_gold, unmatched_pred


def calculate_metrics(gold_count: int, pred_count: int, matched_count: int) -> Dict[str, float]:
    """计算评估指标"""
    precision = matched_count / pred_count if pred_count > 0 else 0.0
    recall = matched_count / gold_count if gold_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'gold_count': gold_count,
        'pred_count': pred_count,
        'matched_count': matched_count
    }


def evaluate_single_paper(ttl_path: str, nodes_path: str, relations_path: str, 
                          paper_id: str) -> Dict[str, Any]:
    """评估单篇论文的抽取效果"""
    
    # 解析标注数据
    gold_entities, gold_triples = parse_ttl_file(ttl_path)
    
    # 解析预测数据（筛选当前论文）
    pred_entities, pred_triples = parse_csv_files(nodes_path, relations_path)
    
    # 筛选当前论文的关系
    with open(relations_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        paper_triples = set()
        for row in reader:
            source_paper = row.get('source_paper', '').strip()
            if paper_id in source_paper:
                source_name = row.get('source_name', '').strip()
                target_name = row.get('target_name', '').strip()
                relation_type = row.get('relation_type', '').strip()
                if relation_type != 'IS_A' and source_name and target_name:
                    paper_triples.add(Triple(
                        subject=source_name,
                        predicate=relation_type,
                        obj=target_name
                    ))
    
    # 实体匹配
    entity_matched, unmatched_gold_ent, unmatched_pred_ent = fuzzy_match_entities(
        gold_entities, pred_entities
    )
    
    # 三元组匹配
    triple_matched, unmatched_gold_tri, unmatched_pred_tri = fuzzy_match_triples(
        gold_triples, paper_triples
    )
    
    # 计算指标
    entity_metrics = calculate_metrics(len(gold_entities), len(pred_entities), entity_matched)
    triple_metrics = calculate_metrics(len(gold_triples), len(paper_triples), triple_matched)
    
    return {
        'paper_id': paper_id,
        'entity_metrics': entity_metrics,
        'triple_metrics': triple_metrics,
        'gold_entities': gold_entities,
        'pred_entities': pred_entities,
        'gold_triples': gold_triples,
        'pred_triples': paper_triples,
        'unmatched_gold_entities': unmatched_gold_ent,
        'unmatched_pred_entities': unmatched_pred_ent,
        'unmatched_gold_triples': unmatched_gold_tri,
        'unmatched_pred_triples': unmatched_pred_tri
    }


def get_papers_with_predictions(relations_path: str) -> Set[str]:
    """获取有预测数据的论文ID列表"""
    papers = set()
    with open(relations_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_paper = row.get('source_paper', '').strip()
            if source_paper and source_paper.startswith('paper_'):
                papers.add(source_paper)
    return papers


def main():
    """主函数"""
    print("=" * 70)
    print("知识图谱抽取效果评估")
    print("=" * 70)
    
    # 路径设置
    eval_dir = Path(__file__).parent
    annotations_dir = eval_dir / "annodations"
    nodes_csv = eval_dir / "nodes_20260127_090505.csv"
    relations_csv = eval_dir / "relationships_20260127_090505.csv"
    
    # 检查文件
    if not annotations_dir.exists():
        print(f"错误: 标注目录不存在: {annotations_dir}")
        return
    
    if not nodes_csv.exists() or not relations_csv.exists():
        print(f"错误: CSV文件不存在")
        return
    
    # 获取有预测数据的论文
    papers_with_preds = get_papers_with_predictions(str(relations_csv))
    print(f"\n有预测数据的论文: {sorted(papers_with_preds)}")
    
    # 获取所有TTL文件
    ttl_files = sorted(annotations_dir.glob("*.ttl"))
    print(f"找到 {len(ttl_files)} 个标注文件")
    
    # 解析所有预测数据
    all_pred_entities, all_pred_triples = parse_csv_files(str(nodes_csv), str(relations_csv))
    print(f"预测数据: {len(all_pred_entities)} 个实体, {len(all_pred_triples)} 条关系")
    
    # 汇总指标
    all_results = []
    total_gold_entities = 0
    total_pred_entities = 0
    total_matched_entities = 0
    total_gold_triples = 0
    total_pred_triples = 0
    total_matched_triples = 0
    
    print("\n" + "-" * 70)
    print("按论文评估结果")
    print("-" * 70)
    
    for ttl_file in ttl_files:
        paper_id = ttl_file.stem  # e.g., "paper_4"
        
        # 检查是否有预测数据
        has_predictions = paper_id in papers_with_preds
        
        result = evaluate_single_paper(
            str(ttl_file),
            str(nodes_csv),
            str(relations_csv),
            paper_id
        )
        result['has_predictions'] = has_predictions
        all_results.append(result)
        
        # 只汇总有预测数据的论文
        if has_predictions:
            total_gold_entities += result['entity_metrics']['gold_count']
            total_pred_entities += result['entity_metrics']['pred_count']
            total_matched_entities += result['entity_metrics']['matched_count']
            total_gold_triples += result['triple_metrics']['gold_count']
            total_pred_triples += result['triple_metrics']['pred_count']
            total_matched_triples += result['triple_metrics']['matched_count']
        
        # 打印单篇结果
        status = "" if has_predictions else " [无预测数据]"
        print(f"\n{paper_id}{status}:")
        print(f"  实体: Gold={result['entity_metrics']['gold_count']}, "
              f"Pred={result['entity_metrics']['pred_count']}, "
              f"Matched={result['entity_metrics']['matched_count']}")
        print(f"        P={result['entity_metrics']['precision']:.3f}, "
              f"R={result['entity_metrics']['recall']:.3f}, "
              f"F1={result['entity_metrics']['f1']:.3f}")
        print(f"  三元组: Gold={result['triple_metrics']['gold_count']}, "
              f"Pred={result['triple_metrics']['pred_count']}, "
              f"Matched={result['triple_metrics']['matched_count']}")
        print(f"        P={result['triple_metrics']['precision']:.3f}, "
              f"R={result['triple_metrics']['recall']:.3f}, "
              f"F1={result['triple_metrics']['f1']:.3f}")
    
    # 计算总体指标
    print("\n" + "=" * 70)
    print(f"总体评估结果（仅包含有预测数据的论文: {sorted(papers_with_preds)}）")
    print("=" * 70)
    
    overall_entity_metrics = calculate_metrics(
        total_gold_entities, total_pred_entities, total_matched_entities
    )
    overall_triple_metrics = calculate_metrics(
        total_gold_triples, total_pred_triples, total_matched_triples
    )
    
    print(f"\n实体抽取:")
    print(f"  标注实体总数: {total_gold_entities}")
    print(f"  预测实体总数: {total_pred_entities}")
    print(f"  匹配实体数:   {total_matched_entities}")
    print(f"  Precision:    {overall_entity_metrics['precision']:.4f}")
    print(f"  Recall:       {overall_entity_metrics['recall']:.4f}")
    print(f"  F1-Score:     {overall_entity_metrics['f1']:.4f}")
    
    print(f"\n关系抽取:")
    print(f"  标注三元组总数: {total_gold_triples}")
    print(f"  预测三元组总数: {total_pred_triples}")
    print(f"  匹配三元组数:   {total_matched_triples}")
    print(f"  Precision:      {overall_triple_metrics['precision']:.4f}")
    print(f"  Recall:         {overall_triple_metrics['recall']:.4f}")
    print(f"  F1-Score:       {overall_triple_metrics['f1']:.4f}")
    
    # 输出详细报告
    print("\n" + "-" * 70)
    print("详细匹配分析")
    print("-" * 70)
    
    # 显示一些未匹配的样例
    for result in all_results[:2]:  # 只显示前2篇
        print(f"\n{result['paper_id']} 未匹配标注实体样例 (前10个):")
        for i, ent in enumerate(list(result['unmatched_gold_entities'])[:10]):
            print(f"  - {ent.name}")
        
        print(f"\n{result['paper_id']} 未匹配标注三元组样例 (前5个):")
        for i, tri in enumerate(list(result['unmatched_gold_triples'])[:5]):
            print(f"  - ({tri.subject}, {tri.predicate}, {tri.obj})")
    
    # 保存评估报告
    report_path = eval_dir / "evaluation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 知识图谱抽取效果评估报告\n\n")
        f.write(f"**评估范围**: 仅包含有预测数据的论文 ({', '.join(sorted(papers_with_preds))})\n\n")
        f.write(f"## 总体评估结果\n\n")
        
        f.write("### 实体抽取\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("|------|------|\n")
        f.write(f"| 标注实体总数 | {total_gold_entities} |\n")
        f.write(f"| 预测实体总数 | {total_pred_entities} |\n")
        f.write(f"| 匹配实体数 | {total_matched_entities} |\n")
        f.write(f"| Precision | {overall_entity_metrics['precision']:.4f} |\n")
        f.write(f"| Recall | {overall_entity_metrics['recall']:.4f} |\n")
        f.write(f"| F1-Score | {overall_entity_metrics['f1']:.4f} |\n\n")
        
        f.write("### 关系抽取\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("|------|------|\n")
        f.write(f"| 标注三元组总数 | {total_gold_triples} |\n")
        f.write(f"| 预测三元组总数 | {total_pred_triples} |\n")
        f.write(f"| 匹配三元组数 | {total_matched_triples} |\n")
        f.write(f"| Precision | {overall_triple_metrics['precision']:.4f} |\n")
        f.write(f"| Recall | {overall_triple_metrics['recall']:.4f} |\n")
        f.write(f"| F1-Score | {overall_triple_metrics['f1']:.4f} |\n\n")
        
        f.write("## 按论文评估结果\n\n")
        f.write("| 论文 | 有预测 | 实体P | 实体R | 实体F1 | 关系P | 关系R | 关系F1 |\n")
        f.write("|------|--------|-------|-------|--------|-------|-------|--------|\n")
        
        for result in all_results:
            has_pred = "Yes" if result.get('has_predictions', False) else "No"
            f.write(f"| {result['paper_id']} | {has_pred} | "
                   f"{result['entity_metrics']['precision']:.3f} | "
                   f"{result['entity_metrics']['recall']:.3f} | "
                   f"{result['entity_metrics']['f1']:.3f} | "
                   f"{result['triple_metrics']['precision']:.3f} | "
                   f"{result['triple_metrics']['recall']:.3f} | "
                   f"{result['triple_metrics']['f1']:.3f} |\n")
    
    print(f"\n评估报告已保存至: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
