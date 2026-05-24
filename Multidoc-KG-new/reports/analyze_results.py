"""
实验结果分析工具
分析知识图谱构建系统的运行结果
"""
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import json


def parse_terminal_output(file_path: str) -> Dict:
    """
    解析终端输出文件，提取关键统计信息
    
    Args:
        file_path: 终端输出文件路径
        
    Returns:
        包含解析结果的字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {
        'papers': [],
        'relationships': [],
        'entities': set(),
        'relation_types': Counter(),
        'paper_stats': {},
        'final_stats': {}
    }
    
    # 提取每篇论文的统计信息
    paper_pattern = r'>>> \[(\d+)\] Paper Summary:\s*\n\s*- Claims extracted: (\d+)\s*\n\s*- Claims grounded: (\d+)\s*\n\s*- Claims validated: (\d+)\s*\n\s*- Claims written: (\d+)'
    paper_matches = re.findall(paper_pattern, content)
    
    for match in paper_matches:
        paper_num, extracted, grounded, validated, written = match
        results['paper_stats'][int(paper_num)] = {
            'extracted': int(extracted),
            'grounded': int(grounded),
            'validated': int(validated),
            'written': int(written)
        }
    
    # 提取写入的关系三元组
    relation_pattern = r'\[Neo4jGraphStore\] ✓ Written: (.+?) -\[(.+?)\]-> (.+?)\n'
    relation_matches = re.findall(relation_pattern, content)
    
    for subject, rel_type, obj in relation_matches:
        results['relationships'].append({
            'subject': subject.strip(),
            'relation': rel_type.strip(),
            'object': obj.strip()[:40]  # 截断过长的对象
        })
        results['entities'].add(subject.strip())
        results['entities'].add(obj.strip()[:40])
        results['relation_types'][rel_type.strip()] += 1
    
    # 提取最终统计信息
    final_stats_pattern = r'Final Graph Statistics.*?\n.*?Stats: (\d+) nodes, (\d+) relationships'
    final_match = re.search(final_stats_pattern, content, re.DOTALL)
    if final_match:
        results['final_stats'] = {
            'nodes': int(final_match.group(1)),
            'relationships': int(final_match.group(2))
        }
    
    # 提取总体统计
    overall_pattern = r'Papers processed: (\d+)\s*\nTotal claims extracted: (\d+)\s*\nTotal claims grounded: (\d+)\s*\nTotal claims validated: (\d+)\s*\nTotal claims written to graph: (\d+)'
    overall_match = re.search(overall_pattern, content)
    if overall_match:
        results['overall'] = {
            'papers_processed': int(overall_match.group(1)),
            'total_extracted': int(overall_match.group(2)),
            'total_grounded': int(overall_match.group(3)),
            'total_validated': int(overall_match.group(4)),
            'total_written': int(overall_match.group(5))
        }
    
    results['entities'] = list(results['entities'])
    return results


def analyze_pipeline_efficiency(results: Dict) -> Dict:
    """
    分析流水线效率
    
    Returns:
        效率分析结果
    """
    if 'overall' not in results:
        return {}
    
    overall = results['overall']
    
    efficiency = {
        'extraction_rate': 1.0,  # 提取率始终为100%
        'grounding_rate': overall['total_grounded'] / overall['total_extracted'] if overall['total_extracted'] > 0 else 0,
        'validation_rate': overall['total_validated'] / overall['total_grounded'] if overall['total_grounded'] > 0 else 0,
        'writing_rate': overall['total_written'] / overall['total_validated'] if overall['total_validated'] > 0 else 0,
        'overall_success_rate': overall['total_written'] / overall['total_extracted'] if overall['total_extracted'] > 0 else 0
    }
    
    return efficiency


def analyze_relation_distribution(results: Dict) -> Dict:
    """
    分析关系类型分布
    
    Returns:
        关系类型分析结果
    """
    relation_types = results['relation_types']
    total = sum(relation_types.values())
    
    distribution = {
        'total_relations': total,
        'unique_relation_types': len(relation_types),
        'top_10_relations': relation_types.most_common(10),
        'relation_diversity': len(relation_types) / total if total > 0 else 0
    }
    
    return distribution


def analyze_entity_coverage(results: Dict) -> Dict:
    """
    分析实体覆盖情况
    
    Returns:
        实体分析结果
    """
    entities = results['entities']
    relationships = results['relationships']
    
    # 统计实体出现频率
    entity_freq = Counter()
    for rel in relationships:
        entity_freq[rel['subject']] += 1
        entity_freq[rel['object']] += 1
    
    coverage = {
        'total_unique_entities': len(entities),
        'total_relationships': len(relationships),
        'avg_relationships_per_entity': len(relationships) * 2 / len(entities) if entities else 0,
        'top_10_entities': entity_freq.most_common(10),
        'entities_with_single_relation': sum(1 for count in entity_freq.values() if count == 1),
        'entities_with_multiple_relations': sum(1 for count in entity_freq.values() if count > 1)
    }
    
    return coverage


def analyze_paper_contribution(results: Dict) -> Dict:
    """
    分析每篇论文的贡献
    
    Returns:
        论文贡献分析结果
    """
    paper_stats = results['paper_stats']
    
    contributions = {}
    for paper_num, stats in paper_stats.items():
        contributions[paper_num] = {
            'claims_per_paper': stats['written'],
            'validation_rate': stats['validated'] / stats['extracted'] if stats['extracted'] > 0 else 0,
            'contribution_percentage': stats['written'] / sum(s['written'] for s in paper_stats.values()) * 100 if paper_stats else 0
        }
    
    return contributions


def generate_analysis_report(results: Dict) -> str:
    """
    生成完整的分析报告
    
    Returns:
        格式化的分析报告字符串
    """
    report = []
    report.append("=" * 80)
    report.append("知识图谱构建实验结果分析报告")
    report.append("=" * 80)
    report.append("")
    
    # 1. 总体统计
    if 'overall' in results:
        overall = results['overall']
        report.append("【1. 总体统计】")
        report.append(f"  处理论文数量: {overall['papers_processed']}")
        report.append(f"  提取的知识声明: {overall['total_extracted']}")
        report.append(f"  语义对齐的知识声明: {overall['total_grounded']}")
        report.append(f"  验证通过的知识声明: {overall['total_validated']}")
        report.append(f"  写入图谱的知识声明: {overall['total_written']}")
        report.append("")
    
    # 2. 流水线效率分析
    efficiency = analyze_pipeline_efficiency(results)
    if efficiency:
        report.append("【2. 流水线效率分析】")
        report.append(f"  提取率: {efficiency['extraction_rate']:.2%}")
        report.append(f"  对齐率: {efficiency['grounding_rate']:.2%}")
        report.append(f"  验证通过率: {efficiency['validation_rate']:.2%}")
        report.append(f"  写入成功率: {efficiency['writing_rate']:.2%}")
        report.append(f"  整体成功率: {efficiency['overall_success_rate']:.2%}")
        report.append("")
    
    # 3. 图谱规模
    if 'final_stats' in results and results['final_stats']:
        final_stats = results['final_stats']
        report.append("【3. 知识图谱规模】")
        report.append(f"  节点总数: {final_stats['nodes']}")
        report.append(f"  关系总数: {final_stats['relationships']}")
        report.append(f"  平均每个节点的关系数: {final_stats['relationships'] / final_stats['nodes']:.2f}" if final_stats['nodes'] > 0 else "  平均每个节点的关系数: N/A")
        report.append("")
    
    # 4. 关系类型分析
    relation_dist = analyze_relation_distribution(results)
    if relation_dist:
        report.append("【4. 关系类型分析】")
        report.append(f"  关系总数: {relation_dist['total_relations']}")
        report.append(f"  唯一关系类型数: {relation_dist['unique_relation_types']}")
        report.append(f"  关系多样性指数: {relation_dist['relation_diversity']:.4f}")
        report.append("  前10个最常见的关系类型:")
        for rel_type, count in relation_dist['top_10_relations']:
            percentage = count / relation_dist['total_relations'] * 100
            report.append(f"    - {rel_type}: {count} ({percentage:.2f}%)")
        report.append("")
    
    # 5. 实体覆盖分析
    entity_coverage = analyze_entity_coverage(results)
    if entity_coverage:
        report.append("【5. 实体覆盖分析】")
        report.append(f"  唯一实体总数: {entity_coverage['total_unique_entities']}")
        report.append(f"  关系总数: {entity_coverage['total_relationships']}")
        report.append(f"  平均每个实体的关系数: {entity_coverage['avg_relationships_per_entity']:.2f}")
        report.append(f"  仅有一个关系的实体数: {entity_coverage['entities_with_single_relation']}")
        report.append(f"  有多个关系的实体数: {entity_coverage['entities_with_multiple_relations']}")
        report.append("  前10个最活跃的实体:")
        for entity, count in entity_coverage['top_10_entities']:
            report.append(f"    - {entity[:50]}: {count} 个关系")
        report.append("")
    
    # 6. 论文贡献分析
    paper_contrib = analyze_paper_contribution(results)
    if paper_contrib:
        report.append("【6. 各论文贡献分析】")
        for paper_num in sorted(paper_contrib.keys()):
            contrib = paper_contrib[paper_num]
            report.append(f"  论文 {paper_num}:")
            report.append(f"    - 贡献的知识声明数: {contrib['claims_per_paper']}")
            report.append(f"    - 验证通过率: {contrib['validation_rate']:.2%}")
            report.append(f"    - 贡献占比: {contrib['contribution_percentage']:.2f}%")
        report.append("")
    
    # 7. 关键发现和建议
    report.append("【7. 关键发现和建议】")
    
    if efficiency:
        if efficiency['validation_rate'] < 0.9:
            report.append("  ⚠️  验证通过率较低，建议检查验证逻辑或数据质量")
        if efficiency['overall_success_rate'] < 0.8:
            report.append("  ⚠️  整体成功率有待提升，建议优化流水线各环节")
    
    if relation_dist and relation_dist['relation_diversity'] < 0.1:
        report.append("  ℹ️  关系类型集中度较高，建议扩展关系类型定义")
    
    if entity_coverage:
        if entity_coverage['entities_with_single_relation'] > entity_coverage['entities_with_multiple_relations']:
            report.append("  ℹ️  大部分实体仅有一个关系，图谱连接性可以进一步提升")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """主函数：分析终端输出结果"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python analyze_results.py <终端输出文件路径>")
        print("示例: python analyze_results.py terminal_output.txt")
        return
    
    file_path = sys.argv[1]
    
    try:
        print("正在解析终端输出...")
        results = parse_terminal_output(file_path)
        
        print("正在生成分析报告...")
        report = generate_analysis_report(results)
        
        print("\n" + report)
        
        # 保存报告到文件
        output_file = file_path.replace('.txt', '_analysis.txt').replace('.log', '_analysis.txt')
        if output_file == file_path:
            output_file = file_path + '_analysis.txt'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n分析报告已保存到: {output_file}")
        
        # 保存JSON格式的详细数据
        json_file = output_file.replace('.txt', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            # 转换set和Counter为可序列化格式
            json_results = {
                'overall': results.get('overall', {}),
                'final_stats': results.get('final_stats', {}),
                'paper_stats': results.get('paper_stats', {}),
                'relation_types': dict(results.get('relation_types', {})),
                'total_entities': len(results.get('entities', [])),
                'total_relationships': len(results.get('relationships', []))
            }
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"详细数据已保存到: {json_file}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
