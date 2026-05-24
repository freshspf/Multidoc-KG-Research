#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复现有图谱中的本体标记
为现有节点添加 is_concept 和 Concept 标签
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from core.neo4j_store import Neo4jGraphStore


def fix_ontology_labels():
    """为现有本体节点添加标记"""
    
    print("=" * 60)
    print("本体标记修复工具")
    print("=" * 60)
    
    # 连接数据库
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")
    
    print(f"连接 Neo4j: {neo4j_uri}")
    graph_store = Neo4jGraphStore(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password
    )
    
    # 使用原生 session 而不是 graph_store.query，以便处理多语句事务
    with graph_store.driver.session() as session:
        try:
            # 1. 找出可能的本体关系（基于关系类型）
            print("\n[1/4] 正在分析现有关系...")
            relations_query = """
            MATCH (s)-[r]->(o)
            RETURN DISTINCT type(r) as rel_type, count(r) as count
            ORDER BY count DESC
            """
            result = session.run(relations_query)
            relations = [{"rel_type": r["rel_type"], "count": r["count"]} for r in result]
            
            print("发现的关系类型:")
            for r in relations:
                rel_type = r['rel_type']
                count = r['count']
                # 标记可能的本体关系
                ontology_marker = " 🔷" if rel_type in [
                    'SUB_CLASS_OF', 'INSTANCE_OF', '为', '藏', '主', '生', '克', 
                    '合', '开窍于', '其华在', '走', '入', '伤', '胜', '畏', '恶'
                ] else ""
                print(f"  - {rel_type}: {count}{ontology_marker}")
            
            # 2. 为本体概念节点添加标记（基于关系类型）
            print("\n[2/4] 正在为概念节点添加标记...")
            
            # 先添加 is_concept 属性
            mark_concepts_query = """
            MATCH (n)
            WHERE n.is_concept IS NULL
            SET n.is_concept = false
            RETURN count(n) as marked_count
            """
            result = session.run(mark_concepts_query)
            record = result.single()
            print(f"  初始化 is_concept 属性: {record['marked_count'] if record else 0} 个节点")
            
            # 标记概念节点（出现在本体关系中的节点）
            mark_ontology_nodes_query = """
            MATCH (n)-[r]-()
            WHERE type(r) IN [
                'SUB_CLASS_OF', 'INSTANCE_OF', '为', '藏', '主', '生', '克',
                '合', '开窍于', '其华在', '走', '入', '伤', '胜', '畏', '恶'
            ]
            SET n.is_concept = true
            SET n:Concept
            RETURN count(DISTINCT n) as marked_count
            """
            result = session.run(mark_ontology_nodes_query)
            record = result.single()
            concept_count = record['marked_count'] if record else 0
            print(f"  标记概念节点: {concept_count}")
            
            # 3. 为本体关系添加标记
            print("\n[3/4] 正在为本体关系添加标记...")
            
            # 先添加 is_ontology 属性
            mark_rels_init_query = """
            MATCH ()-[r]->()
            WHERE r.is_ontology IS NULL
            SET r.is_ontology = false
            RETURN count(r) as marked_count
            """
            result = session.run(mark_rels_init_query)
            record = result.single()
            print(f"  初始化 is_ontology 属性: {record['marked_count'] if record else 0} 条关系")
            
            # 标记本体关系
            mark_rels_query = """
            MATCH ()-[r]->()
            WHERE type(r) IN [
                'SUB_CLASS_OF', 'INSTANCE_OF', '为', '藏', '主', '生', '克',
                '合', '开窍于', '其华在', '走', '入', '伤', '胜', '畏', '恶'
            ]
            SET r.is_ontology = true
            RETURN count(r) as marked_count
            """
            result = session.run(mark_rels_query)
            record = result.single()
            ontology_rel_count = record['marked_count'] if record else 0
            print(f"  标记本体关系: {ontology_rel_count}")
            
            # 4. 验证结果
            print("\n[4/4] 正在统计结果...")
            
            # 总节点数
            result = session.run("MATCH (n) RETURN count(n) as count")
            total_nodes = result.single()["count"]
            
            # 概念节点数
            result = session.run("""
                MATCH (n) 
                WHERE n.is_concept = true OR 'Concept' IN labels(n)
                RETURN count(n) as count
            """)
            concept_nodes = result.single()["count"]
            
            # 实体节点数 = 总节点 - 概念节点
            entity_nodes = total_nodes - concept_nodes
            
            # 总关系数
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            total_rels = result.single()["count"]
            
            # 本体关系数
            result = session.run("""
                MATCH ()-[r]->() 
                WHERE r.is_ontology = true OR type(r) IN ['SUB_CLASS_OF', 'INSTANCE_OF']
                RETURN count(r) as count
            """)
            ontology_rels = result.single()["count"]
            
            # 实例关系数
            instance_rels = total_rels - ontology_rels
            
            print("\n修复后统计:")
            print(f"  总节点: {total_nodes}")
            print(f"  概念节点: {concept_nodes}")
            print(f"  实体节点: {entity_nodes}")
            print(f"  总关系: {total_rels}")
            print(f"  本体关系: {ontology_rels}")
            print(f"  实例关系: {instance_rels}")
            
            # 5. 展示一些示例
            print("\n📊 概念节点示例:")
            result = session.run("""
                MATCH (n)
                WHERE n.is_concept = true
                RETURN n.name as name, labels(n) as labels
                LIMIT 10
            """)
            for i, record in enumerate(result, 1):
                print(f"  {i}. {record['name']} - 标签: {record['labels']}")
            
            print("\n📊 本体关系示例:")
            result = session.run("""
                MATCH (s)-[r]->(o)
                WHERE r.is_ontology = true
                RETURN s.name as source, type(r) as rel, o.name as target
                LIMIT 10
            """)
            for i, record in enumerate(result, 1):
                print(f"  {i}. {record['source']} -[{record['rel']}]-> {record['target']}")
            
        except Exception as e:
            print(f"\n❌ 执行过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            graph_store.close()
            print("\n[Neo4jGraphStore] Connection closed")


def verify_labels():
    """验证标记结果"""
    print("\n" + "=" * 60)
    print("验证标记结果")
    print("=" * 60)
    
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")
    
    graph_store = Neo4jGraphStore(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password
    )
    
    with graph_store.driver.session() as session:
        # 检查 is_concept 属性
        result = session.run("""
            MATCH (n)
            RETURN 
                count(n) as total_nodes,
                sum(CASE WHEN n.is_concept = true THEN 1 ELSE 0 END) as concept_nodes,
                sum(CASE WHEN n.is_concept IS NULL THEN 1 ELSE 0 END) as null_concept
        """)
        stats = result.single()
        print(f"\n节点统计:")
        print(f"  总节点: {stats['total_nodes']}")
        print(f"  概念节点: {stats['concept_nodes']}")
        print(f"  未标记节点: {stats['null_concept']}")
        
        # 检查 is_ontology 属性
        result = session.run("""
            MATCH ()-[r]->()
            RETURN 
                count(r) as total_rels,
                sum(CASE WHEN r.is_ontology = true THEN 1 ELSE 0 END) as ontology_rels,
                sum(CASE WHEN r.is_ontology IS NULL THEN 1 ELSE 0 END) as null_rels
        """)
        stats = result.single()
        print(f"\n关系统计:")
        print(f"  总关系: {stats['total_rels']}")
        print(f"  本体关系: {stats['ontology_rels']}")
        print(f"  未标记关系: {stats['null_rels']}")
    
    graph_store.close()


if __name__ == "__main__":
    fix_ontology_labels()
    verify_labels()