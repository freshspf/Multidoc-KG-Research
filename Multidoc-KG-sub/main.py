"""
主程序入口，使用优化的并行处理和批量验证。
流水线架构，各阶段并发运行以实现最大吞吐量。
"""
import os
import argparse
import time
import csv
import json
import queue
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

from schema import Paper, KnowledgeClaim, SubdomainAssignment
from core.logger import setup_logger
from core.llm_client import LLMClient
from core.neo4j_store import Neo4jGraphStore
from core.vector_store import VectorStore, MockVectorStore
from core.timing import TimingStats, timed_step
from agents.extraction import ExtractionAgent
from agents.grounding import SemanticGroundingAgent
from agents.subdomain import SubdomainClassifierAgent
from agents.subdomain_refinement import SubdomainHierarchyRefinementAgent
from agents.validation import KnowledgeValidationAgent
from agents.evolution import KnowledgeEvolutionAgent
from data_loader import PaperDataLoader

# 初始化日志
logger = setup_logger(__name__)


def iter_paper_batches(papers: List[Paper], batch_size: int) -> List[List[Paper]]:
    """Split papers into stable batches."""
    if batch_size <= 0:
        batch_size = len(papers) or 1
    return [papers[idx: idx + batch_size] for idx in range(0, len(papers), batch_size)]


def export_subdomain_assignments_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Export subdomain assignment results to CSV."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "paper_id",
        "paper_title",
        "subdomain",
        "parent_domain",
        "status",
        "is_new_subdomain",
        "batch_id",
        "taxonomy_version",
        "reason",
        "confidence",
        "new_relations",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "paper_id": row.get("paper_id", ""),
                    "paper_title": row.get("paper_title", ""),
                    "subdomain": row.get("subdomain", ""),
                    "parent_domain": row.get("parent_domain", ""),
                    "status": row.get("status", ""),
                    "is_new_subdomain": row.get("is_new_subdomain", ""),
                    "batch_id": row.get("batch_id", ""),
                    "taxonomy_version": row.get("taxonomy_version", ""),
                    "reason": row.get("reason", ""),
                    "confidence": row.get("confidence", ""),
                    "new_relations": json.dumps(row.get("new_relations", ""), ensure_ascii=False),
                }
            )


def export_extraction_claims_csv(claims: List[KnowledgeClaim], papers: List[Paper], output_path: str) -> None:
    """Export extracted claims to CSV for inspection."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    paper_lookup = {paper.id: paper for paper in papers}
    fieldnames = [
        "paper_id",
        "paper_title",
        "subdomain",
        "parent_domain",
        "claim_type",
        "subject",
        "relation",
        "object",
        "confidence",
        "section_title",
        "evidence",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for claim in claims:
            paper = paper_lookup.get(claim.source_paper_id)
            metadata = claim.metadata or {}
            writer.writerow(
                {
                    "paper_id": claim.source_paper_id,
                    "paper_title": paper.title if paper else "",
                    "subdomain": metadata.get("paper_subdomain", ""),
                    "parent_domain": metadata.get("paper_parent_domain", ""),
                    "claim_type": claim.claim_type or "",
                    "subject": claim.subject,
                    "relation": claim.relation,
                    "object": claim.object,
                    "confidence": claim.confidence,
                    "section_title": metadata.get("section_title", ""),
                    "evidence": claim.evidence,
                }
            )


def export_subdomain_refinement_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Export subdomain refinement decisions to CSV."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "candidate",
        "action",
        "target_subdomain",
        "parent_domain",
        "reason",
        "applied_taxonomy_version",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def persist_subdomain_assignments(graph_store: Neo4jGraphStore, papers: List[Paper]) -> None:
    """Persist paper/subdomain assignments to Neo4j before downstream claim writing."""
    persisted = 0

    for paper in papers:
        metadata = paper.metadata or {}
        assignment_payload = metadata.get("subdomain_assignment")
        if not assignment_payload:
            continue

        try:
            assignment = (
                assignment_payload
                if isinstance(assignment_payload, SubdomainAssignment)
                else SubdomainAssignment(**assignment_payload)
            )
            graph_store.write_subdomain_assignment(
                paper=paper,
                assignment=assignment,
            )
            persisted += 1
        except Exception as e:
            logger.error(f"写入论文子领域失败 {paper.id}: {e}", exc_info=True)

    logger.info(f"已写入 {persisted} 篇论文的子领域信息到 Neo4j")


def run_subdomain_refinement(
    graph_store: Neo4jGraphStore,
    refinement_agent: SubdomainHierarchyRefinementAgent,
    batch_id: Optional[str] = None,
    current_version: int = 1,
) -> tuple[List[Dict[str, Any]], int]:
    """Run one round of candidate refinement and apply decisions to Neo4j."""
    confirmed_subdomains = graph_store.get_confirmed_subdomains()
    candidate_subdomains = graph_store.get_subdomain_candidates(batch_id=batch_id)
    if not candidate_subdomains:
        return [], current_version

    decisions = refinement_agent.process(confirmed_subdomains, candidate_subdomains)
    if not decisions:
        return [], current_version

    next_version = current_version + 1
    applied_rows: List[Dict[str, Any]] = []

    for decision in decisions:
        candidate = str(decision.get("candidate", "")).strip()
        action = str(decision.get("action", "")).strip()
        target_subdomain = str(decision.get("target_subdomain", "")).strip()
        parent_domain = str(decision.get("parent_domain", "")).strip() or "biomedicine"
        reason = str(decision.get("reason", "")).strip()

        if not candidate or not action or not target_subdomain:
            continue

        if action == "merge":
            graph_store.merge_candidate_into_subdomain(
                candidate_name=candidate,
                target_name=target_subdomain,
                taxonomy_version=next_version,
                reason=reason,
            )
        else:
            graph_store.promote_candidate_subdomain(
                candidate_name=candidate,
                parent_name=parent_domain,
                taxonomy_version=next_version,
                reason=reason,
            )

        applied_rows.append(
            {
                "candidate": candidate,
                "action": action,
                "target_subdomain": target_subdomain,
                "parent_domain": parent_domain,
                "reason": reason,
                "applied_taxonomy_version": next_version,
            }
        )

    return applied_rows, (next_version if applied_rows else current_version)


class PipelineOrchestrator:
    """
    流水线并行处理器
    Extraction/Grounding/Validation 各运行在独立线程，通过 Queue 传递数据
    """
    
    def __init__(self,
                 extraction_agent: ExtractionAgent,
                 grounding_agent: SemanticGroundingAgent,
                 validation_agent: KnowledgeValidationAgent,
                 evolution_agent: KnowledgeEvolutionAgent,
                 timing_stats: Optional[TimingStats] = None):
        """
        初始化流水线处理器
        
        Args:
            extraction_agent: 提取代理
            grounding_agent: 接地代理
            validation_agent: 验证代理
            evolution_agent: 演化代理
            timing_stats: 可选的计时统计实例
        """
        self.extraction_agent = extraction_agent
        self.grounding_agent = grounding_agent
        self.validation_agent = validation_agent
        self.evolution_agent = evolution_agent
        self.timing_stats = timing_stats
        
        # 线程间通信队列
        self.q_grounding = queue.Queue()
        self.q_validation = queue.Queue()
        self.q_evolution = queue.Queue()
        
        # 哨兵值，用于通知线程结束
        self.DONE = object()
        
        # 线程锁，保护 Neo4j 写操作
        self.evolution_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'papers_processed': 0,
            'total_claims': 0,
            'total_grounded': 0,
            'total_validated': 0,
            'total_written': 0,
            'processing_times': []
        }
        
        logger.info("[PipelineOrchestrator] 初始化完成，使用流水线并行")
    
    def _extraction_stage(self, papers: List[Paper]):
        """
        Extraction 阶段线程
        读取论文，提取声明，送入 grounding 队列
        """
        for idx, paper in enumerate(papers, 1):
            logger.info(f">>> [{idx}/{len(papers)}] 提取: {paper.id}")
            try:
                if self.timing_stats:
                    claims = timed_step(
                        self.timing_stats, idx, "提取",
                        self.extraction_agent.process, paper
                    )
                    elapsed = self.timing_stats._paper_timings.get(idx, {}).get("提取", 0)
                else:
                    start_time = time.time()
                    claims = self.extraction_agent.process(paper)
                    elapsed = time.time() - start_time
                
                logger.info(f"提取: {len(claims)} 个声明，耗时 {elapsed:.2f}秒")
                
                # 早退调度 - 无声明直接跳过后续阶段
                if not claims:
                    logger.warning(f"未提取到声明，跳过论文 {paper.id}")
                    continue
                    
                self.q_grounding.put((idx, paper, claims, elapsed))
                
            except Exception as e:
                logger.error(f"提取论文 {paper.id} 时出错: {e}", exc_info=True)
        
        # 发送结束信号
        self.q_grounding.put(self.DONE)
        logger.info("[提取阶段] 完成")
    
    def _grounding_stage(self):
        """
        Grounding 阶段线程
        从队列接收声明，进行实体解析，送入 validation 队列
        """
        while True:
            item = self.q_grounding.get()
            if item is self.DONE:
                self.q_validation.put(self.DONE)
                break
                
            idx, paper, claims, extraction_time = item
            logger.info(f">>> [{idx}] 接地: {paper.id}")
            
            try:
                if self.timing_stats:
                    grounded_claims = timed_step(
                        self.timing_stats, idx, "接地",
                        self.grounding_agent.process, claims
                    )
                    elapsed = self.timing_stats._paper_timings.get(idx, {}).get("接地", 0)
                else:
                    start_time = time.time()
                    grounded_claims = self.grounding_agent.process(claims)
                    elapsed = time.time() - start_time
                
                logger.info(f"接地: {len(grounded_claims)} 个声明，耗时 {elapsed:.2f}秒")
                
                # 早退调度 - 无接地声明跳过后续阶段
                if not grounded_claims:
                    logger.warning(f"无接地声明，跳过论文 {paper.id}")
                    continue
                    
                self.q_validation.put((idx, paper, claims, grounded_claims, extraction_time, elapsed))
                
            except Exception as e:
                logger.error(f"接地论文 {paper.id} 时出错: {e}", exc_info=True)
        
        logger.info("[接地阶段] 完成")
    
    def _validation_stage(self):
        """
        Validation 阶段线程
        从队列接收接地声明，进行验证，送入 evolution 队列
        """
        while True:
            item = self.q_validation.get()
            if item is self.DONE:
                self.q_evolution.put(self.DONE)
                break
                
            idx, paper, claims, grounded_claims, extraction_time, grounding_time = item
            logger.info(f">>> [{idx}] 验证: {paper.id}")
            
            try:
                if self.timing_stats:
                    validated_claims = timed_step(
                        self.timing_stats, idx, "验证",
                        self.validation_agent.process, grounded_claims
                    )
                    elapsed = self.timing_stats._paper_timings.get(idx, {}).get("验证", 0)
                else:
                    start_time = time.time()
                    validated_claims = self.validation_agent.process(grounded_claims)
                    elapsed = time.time() - start_time
                
                num_validated = sum(1 for c in validated_claims if c.status.value == 'validated')
                logger.info(f"验证: {num_validated}/{len(validated_claims)} 个通过，耗时 {elapsed:.2f}秒")
                
                self.q_evolution.put((idx, paper, claims, grounded_claims, validated_claims,
                                       extraction_time, grounding_time, elapsed, num_validated))
                
            except Exception as e:
                logger.error(f"验证论文 {paper.id} 时出错: {e}", exc_info=True)
        
        logger.info("[验证阶段] 完成")
    
    def _evolution_stage(self):
        """
        Evolution 阶段（在主线程执行）
        从队列接收验证通过的声明，写入图数据库
        """
        paper_results = []
        
        while True:
            item = self.q_evolution.get()
            if item is self.DONE:
                break
                
            (idx, paper, claims, grounded_claims, validated_claims,
             extraction_time, grounding_time, validation_time, num_validated) = item
            
            logger.info(f">>> [{idx}] 演化: {paper.id}")
            
            try:
                if self.timing_stats:
                    with self.evolution_lock:
                        written_ids = timed_step(
                            self.timing_stats, idx, "演化",
                            self.evolution_agent.process, validated_claims
                        )
                    elapsed = self.timing_stats._paper_timings.get(idx, {}).get("演化", 0)
                else:
                    start_time = time.time()
                    with self.evolution_lock:
                        written_ids = self.evolution_agent.process(validated_claims)
                    elapsed = time.time() - start_time
                
                logger.info(f"演化: {len(written_ids)} 个写入，耗时 {elapsed:.2f}秒")
                
                total_time = extraction_time + grounding_time + validation_time + elapsed
                
                result = {
                    'paper_id': paper.id,
                    'paper_title': paper.title,
                    'claims_extracted': len(claims),
                    'claims_grounded': len(grounded_claims),
                    'claims_validated': num_validated,
                    'claims_written': len(written_ids),
                    'extraction_time': extraction_time,
                    'grounding_time': grounding_time,
                    'validation_time': validation_time,
                    'evolution_time': elapsed,
                    'total_time': total_time,
                    'success': True
                }
                
                # 更新统计
                self.stats['papers_processed'] += 1
                self.stats['total_claims'] += len(claims)
                self.stats['total_grounded'] += len(grounded_claims)
                self.stats['total_validated'] += num_validated
                self.stats['total_written'] += len(written_ids)
                self.stats['processing_times'].append(total_time)
                
                paper_results.append(result)
                
                logger.info(f">>> [{idx}] 论文总结:")
                logger.info(f"  - 提取声明: {len(claims)}")
                logger.info(f"  - 接地声明: {len(grounded_claims)}")
                logger.info(f"  - 验证通过: {num_validated}")
                logger.info(f"  - 写入声明: {len(written_ids)}")
                logger.info(f"  - 总耗时:   {total_time:.2f}秒")
                
            except Exception as e:
                logger.error(f"演化论文 {paper.id} 时出错: {e}", exc_info=True)
                paper_results.append({
                    'paper_id': paper.id,
                    'paper_title': paper.title,
                    'success': False,
                    'error': str(e)
                })
        
        return paper_results
    
    def process_papers(self, papers: List[Paper]) -> List[Dict[str, Any]]:
        """
        启动流水线，处理所有论文
        
        Args:
            papers: 论文列表
            
        Returns:
            处理结果列表
        """
        logger.info("=" * 60)
        logger.info(f"开始流水线并行处理 {len(papers)} 篇论文")
        logger.info("=" * 60)
        
        # 启动三个工作线程
        threads = [
            threading.Thread(target=self._extraction_stage, args=(papers,), name="提取"),
            threading.Thread(target=self._grounding_stage, name="接地"),
            threading.Thread(target=self._validation_stage, name="验证"),
        ]
        
        for t in threads:
            t.daemon = True
            t.start()
        
        # Evolution 在主线程执行
        results = self._evolution_stage()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        return results


def main(data_dir: str = "data/papers", 
         max_papers: int = None,
         batch_size: int = 50,
         max_workers: int = 16,
         validation_batch_size: int = 20,
         use_cache: bool = True,
         clear_db: bool = False,
         timing_report: str = None,
         neo4j_uri: str = None,
         neo4j_user: str = None,
         neo4j_password: str = None,
         skip_validation: bool = False,
         skip_subdomain: bool = False,
         subdomain_report: str = "reports/subdomain_assignments.csv",
         subdomain_only: bool = False,
         subdomain_graph_only: bool = False,
         refine_subdomains_only: bool = False,
         refinement_report: str = "reports/subdomain_refinement_decisions.csv",
         extraction_only: bool = False,
         extraction_report: str = "reports/extraction_claims.csv",
         vector_model_name: str = "BAAI/bge-m3"):
    """
    主工作流编排，使用流水线并行。
    
    Args:
        data_dir: 包含 JSON 论文文件的目录
        max_papers: 要处理的最大论文数量
        batch_size: 子领域分类批次大小
        max_workers: 验证阶段的并行工作线程数
        validation_batch_size: 验证批处理大小
        use_cache: 是否使用验证缓存
        clear_db: 是否在开始前清空 Neo4j 数据库
        timing_report: 保存计时报告的 JSON 文件路径
        neo4j_uri: Neo4j 连接 URI
        neo4j_user: Neo4j 用户名
        neo4j_password: Neo4j 密码
        skip_validation: 跳过 LLM 验证，所有接地声明直接写入
        skip_subdomain: 跳过子领域分类
        subdomain_report: 子领域分类结果 CSV 导出路径
        subdomain_only: 仅运行子领域分类并导出结果，不进入后续流水线
        subdomain_graph_only: 仅运行子领域分类并写入 Neo4j，不进入后续流水线
        refine_subdomains_only: 仅运行子领域 candidate refinement，不处理论文
        refinement_report: 子领域 refinement 决策 CSV 导出路径
        extraction_only: 仅运行 Extraction 并导出结果，不进入后续流水线
        extraction_report: Extraction 结果 CSV 导出路径
        vector_model_name: Grounding 使用的向量模型名称
    """
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("多论文知识图谱构建框架（流水线并行）")
    logger.info("=" * 60)
    
    # 初始化计时统计
    timing_stats = TimingStats()
    
    # 初始化依赖
    logger.info("初始化依赖...")
    llm_client = LLMClient(model_name="deepseek-chat")

    use_graph_hierarchy_for_subdomain = refine_subdomains_only or subdomain_graph_only or (not subdomain_only and not extraction_only)
    graph_store: Optional[Neo4jGraphStore] = None

    if use_graph_hierarchy_for_subdomain:
        neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password123")

        graph_store = Neo4jGraphStore(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )

        if clear_db:
            logger.warning("按请求清空数据库（子领域分类前）...")
            graph_store.clear_all()
            logger.info("数据库已清空")

        graph_store.ensure_subdomain_root("Biomedicine")

    if refine_subdomains_only:
        if graph_store is None:
            raise RuntimeError("refine_subdomains_only 模式需要可用的 Neo4jGraphStore")

        refinement_agent = SubdomainHierarchyRefinementAgent(
            llm_client=llm_client,
            config_path="config/subdomain_config.yaml",
        )
        current_version = graph_store.get_current_taxonomy_version()
        applied_rows, updated_version = run_subdomain_refinement(
            graph_store=graph_store,
            refinement_agent=refinement_agent,
            batch_id=None,
            current_version=current_version,
        )

        export_subdomain_refinement_csv(applied_rows, refinement_report)
        logger.info(f"子领域 refinement 决策已导出: {refinement_report}")
        logger.info(f"Refinement 决策数: {len(applied_rows)}")
        logger.info(f"Taxonomy version: {current_version} -> {updated_version}")
        graph_store.close()
        return

    # 加载论文
    logger.info("从医学文献 JSON 文件加载论文...")
    data_loader = PaperDataLoader(data_dir=data_dir)
    papers = data_loader.load_all_papers()
    
    if not papers:
        logger.error("未加载到论文，退出...")
        return
    
    # 限制论文数量
    if max_papers is not None and max_papers > 0:
        papers = papers[:max_papers]
        logger.info(f"处理前 {len(papers)} 篇论文")
    
    # 初始化代理
    logger.info("初始化代理...")
    subdomain_agent = SubdomainClassifierAgent(
        llm_client=llm_client,
        config_path="config/subdomain_config.yaml",
        hierarchy_provider=graph_store,
    )
    refinement_agent = (
        SubdomainHierarchyRefinementAgent(
            llm_client=llm_client,
            config_path="config/subdomain_config.yaml",
        )
        if graph_store is not None
        else None
    )
    subdomain_rows: List[Dict[str, Any]] = []
    refinement_rows: List[Dict[str, Any]] = []
    taxonomy_version = graph_store.get_current_taxonomy_version() if graph_store is not None else 1

    if skip_subdomain:
        logger.info("按请求跳过子领域分类")
    else:
        logger.info("=" * 60)
        logger.info("子领域分类（Extraction 前置）")
        logger.info("=" * 60)
        batches = iter_paper_batches(papers, batch_size)
        processed_count = 0
        for batch_idx, batch in enumerate(batches, 1):
            batch_id = f"subdomain_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{batch_idx:03d}"
            hierarchy_snapshot = subdomain_agent.get_hierarchy_snapshot() if use_graph_hierarchy_for_subdomain else []
            logger.info(
                f"[Batch {batch_idx}/{len(batches)}] 开始子领域分类: "
                f"{len(batch)} 篇论文, hierarchy_edges={len(hierarchy_snapshot)}, batch_id={batch_id}"
            )

            for paper in batch:
                processed_count += 1
                try:
                    assignment = subdomain_agent.process(
                        paper,
                        hierarchy_override=hierarchy_snapshot,
                        taxonomy_version=taxonomy_version,
                        batch_id=batch_id,
                    )
                    if paper.metadata is None:
                        paper.metadata = {}
                    paper.metadata["subdomain"] = assignment.subdomain
                    paper.metadata["parent_domain"] = assignment.parent_domain
                    paper.metadata["subdomain_reason"] = assignment.reason
                    paper.metadata["subdomain_confidence"] = assignment.confidence
                    paper.metadata["subdomain_new_relations"] = assignment.new_relations
                    paper.metadata["subdomain_assignment"] = assignment.to_dict()
                    paper.metadata["subdomain_status"] = assignment.status
                    paper.metadata["subdomain_batch_id"] = assignment.batch_id
                    paper.metadata["taxonomy_version"] = assignment.taxonomy_version
                    subdomain_rows.append(
                        {
                            "paper_id": paper.id,
                            "paper_title": paper.title,
                            "subdomain": assignment.subdomain,
                            "parent_domain": assignment.parent_domain,
                            "status": assignment.status,
                            "is_new_subdomain": assignment.is_new_subdomain,
                            "batch_id": assignment.batch_id,
                            "taxonomy_version": assignment.taxonomy_version,
                            "reason": assignment.reason,
                            "confidence": assignment.confidence,
                            "new_relations": assignment.new_relations,
                        }
                    )
                    logger.info(
                        f"[{processed_count}/{len(papers)}] {paper.id}: "
                        f"{assignment.subdomain} -> {assignment.parent_domain} "
                        f"(status={assignment.status})"
                    )
                except Exception as e:
                    logger.error(f"子领域分类失败 {paper.id}: {e}", exc_info=True)

            if graph_store is not None and refinement_agent is not None:
                logger.info(f"[Batch {batch_idx}/{len(batches)}] 写入候选子领域并自动执行 refinement")
                persist_subdomain_assignments(graph_store, batch)
                batch_refinement_rows, taxonomy_version = run_subdomain_refinement(
                    graph_store=graph_store,
                    refinement_agent=refinement_agent,
                    batch_id=batch_id,
                    current_version=taxonomy_version,
                )
                refinement_rows.extend(batch_refinement_rows)

                if batch_refinement_rows:
                    decision_map = {
                        str(row.get("candidate", "")).strip().lower(): row for row in batch_refinement_rows
                    }
                    for paper in batch:
                        metadata = paper.metadata or {}
                        candidate_name = str(metadata.get("subdomain", "")).strip()
                        if not candidate_name:
                            continue
                        decision = decision_map.get(candidate_name.lower())
                        if not decision:
                            continue

                        action = str(decision.get("action", "")).strip()
                        target_subdomain = str(decision.get("target_subdomain", "")).strip()
                        parent_domain = str(decision.get("parent_domain", "")).strip() or "biomedicine"

                        if action == "merge":
                            metadata["subdomain"] = target_subdomain
                            metadata["subdomain_status"] = "confirmed"
                        else:
                            metadata["subdomain"] = target_subdomain or candidate_name
                            metadata["parent_domain"] = parent_domain
                            metadata["subdomain_status"] = "confirmed"

                        metadata["taxonomy_version"] = taxonomy_version
                        assignment_payload = metadata.get("subdomain_assignment", {})
                        if isinstance(assignment_payload, dict):
                            assignment_payload["subdomain"] = metadata["subdomain"]
                            assignment_payload["parent_domain"] = metadata.get("parent_domain", parent_domain)
                            assignment_payload["status"] = "confirmed"
                            assignment_payload["is_new_subdomain"] = False
                            assignment_payload["taxonomy_version"] = taxonomy_version
                            metadata["subdomain_assignment"] = assignment_payload

                    for row in subdomain_rows:
                        if row.get("batch_id") != batch_id:
                            continue
                        decision = decision_map.get(str(row.get("subdomain", "")).strip().lower())
                        if not decision:
                            continue
                        row["subdomain"] = str(decision.get("target_subdomain", row.get("subdomain", ""))).strip()
                        row["parent_domain"] = str(decision.get("parent_domain", row.get("parent_domain", ""))).strip() or row.get("parent_domain", "")
                        row["status"] = "confirmed"
                        row["is_new_subdomain"] = False
                        row["taxonomy_version"] = taxonomy_version

        if subdomain_rows:
            export_subdomain_assignments_csv(subdomain_rows, subdomain_report)
            logger.info(f"子领域分类结果已导出: {subdomain_report}")
        if refinement_rows:
            export_subdomain_refinement_csv(refinement_rows, refinement_report)
            logger.info(f"子领域 refinement 决策已导出: {refinement_report}")

    if subdomain_only:
        logger.info("=" * 60)
        logger.info("子领域分类已完成，按请求跳过后续流水线")
        logger.info("=" * 60)
        return

    if subdomain_graph_only:
        if graph_store is None:
            raise RuntimeError("subdomain_graph_only 模式需要可用的 Neo4jGraphStore")

        graph_stats = graph_store.get_stats()
        hierarchy = graph_store.get_subdomain_hierarchy()
        logger.info("=" * 60)
        logger.info("子领域入图验证模式已完成，按请求跳过后续流水线")
        logger.info(f"当前图数据库节点数: {graph_stats['nodes']}")
        logger.info(f"当前图数据库关系数: {graph_stats['relationships']}")
        logger.info(f"当前子领域层级边数: {len(hierarchy)}")
        logger.info("=" * 60)
        graph_store.close()
        return

    extraction_agent = ExtractionAgent(
        llm_client=llm_client,
        config_path="config/extraction_config.yaml",
    )

    if extraction_only:
        logger.info("=" * 60)
        logger.info("Extraction Only 模式")
        logger.info("=" * 60)
        extracted_claims: List[KnowledgeClaim] = []
        for idx, paper in enumerate(papers, 1):
            logger.info(f">>> [{idx}/{len(papers)}] 提取: {paper.id}")
            try:
                claims = extraction_agent.process(paper)
                extracted_claims.extend(claims)
                logger.info(f"提取完成: {paper.id} -> {len(claims)} 条")
            except Exception as e:
                logger.error(f"Extraction 失败 {paper.id}: {e}", exc_info=True)

        export_extraction_claims_csv(extracted_claims, papers, extraction_report)
        logger.info(f"Extraction 结果已导出: {extraction_report}")
        logger.info(f"Extraction 总声明数: {len(extracted_claims)}")
        return

    # 初始化 Neo4j（如果前面尚未初始化）
    if graph_store is None:
        neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password123")

        graph_store = Neo4jGraphStore(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )
    
    if not skip_subdomain and subdomain_rows:
        logger.info("将论文子领域信息写入 Neo4j...")
        persist_subdomain_assignments(graph_store, papers)
    
    try:
        vector_store = VectorStore(model_name=vector_model_name)
    except Exception as e:
        logger.warning(f"向量模型初始化失败，退回 MockVectorStore: {e}")
        vector_store = MockVectorStore()

    grounding_agent = SemanticGroundingAgent(
        llm_client=llm_client,
        vector_store=vector_store
    )
    
    validation_agent = KnowledgeValidationAgent(
        llm_client=llm_client,
        graph_store=graph_store,
        use_cache=use_cache,
        batch_size=validation_batch_size,
        max_workers=max_workers,
        enable_parallel=True,
        skip_validation=skip_validation
    )
    
    evolution_agent = KnowledgeEvolutionAgent(graph_store=graph_store)
    
    # 使用流水线并行处理
    logger.info("=" * 60)
    logger.info("使用流水线并行（提取|接地|验证 并发运行）")
    logger.info("=" * 60)
    
    orchestrator = PipelineOrchestrator(
        extraction_agent=extraction_agent,
        grounding_agent=grounding_agent,
        validation_agent=validation_agent,
        evolution_agent=evolution_agent,
        timing_stats=timing_stats
    )
    
    results = orchestrator.process_papers(papers)
    processor_stats = orchestrator.stats
    
    # 获取最终统计
    total_time = time.time() - start_time
    
    # 输出计时报告
    logger.info("=" * 60)
    timing_stats.format_report(logger=logger)
    
    # 保存 JSON 报告
    if timing_report:
        timing_stats.save_json(timing_report)
        logger.info(f"计时报告已保存至: {timing_report}")
    
    # 打印汇总
    logger.info("=" * 60)
    logger.info("流水线汇总")
    logger.info("=" * 60)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    skipped = [r for r in results if r.get('skipped', False)]
    
    logger.info(f"处理的论文数: {len(successful)}/{len(papers)}")
    logger.info(f"失败的论文: {len(failed)}")
    logger.info(f"跳过的论文（无声明）: {len(skipped)}")
    
    if successful:
        avg_time = sum(r['total_time'] for r in successful) / len(successful)
        logger.info(f"每篇成功论文平均耗时: {avg_time:.2f}秒")
        
        logger.info(f"总提取声明数: {processor_stats['total_claims']}")
        logger.info(f"总接地声明数: {processor_stats['total_grounded']}")
        logger.info(f"总验证通过数: {processor_stats['total_validated']}")
        logger.info(f"总写入声明数: {processor_stats['total_written']}")
    
    # 获取图数据库统计
    graph_stats = graph_store.get_stats()
    logger.info(f"图数据库:")
    logger.info(f"  - 节点数: {graph_stats['nodes']}")
    logger.info(f"  - 关系数: {graph_stats['relationships']}")
    
    # 验证代理统计
    validation_stats = validation_agent.get_stats()
    logger.info(f"验证:")
    logger.info(f"  - API 调用次数: {validation_stats.get('api_calls', 0)}")
    logger.info(f"  - 缓存命中数: {validation_stats.get('cache_hits', 0)}")
    logger.info(f"  - 缓存命中率: {validation_stats.get('cache_hit_rate', 0):.2%}")
    
    logger.info("=" * 60)
    logger.info(f"流水线总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    logger.info("=" * 60)
    
    # 关闭连接
    graph_store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多论文医学知识图谱构建（流水线并行）")
    parser.add_argument("--data-dir", type=str, default="data/papers", 
                        help="包含医学文献 JSON 文件的目录")
    parser.add_argument("--max-papers", type=int, default=None,
                        help="要处理的最大论文数量")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="子领域分类批次大小")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="验证阶段的并行工作线程数")
    parser.add_argument("--validation-batch-size", type=int, default=20,
                        help="验证批处理大小")
    parser.add_argument("--no-cache", action="store_true",
                        help="禁用验证缓存")
    parser.add_argument("--skip-validation", action="store_true",
                        help="跳过 LLM 验证，所有接地声明直接写入图数据库")
    parser.add_argument("--skip-subdomain", action="store_true",
                        help="跳过子领域分类")
    parser.add_argument("--subdomain-report", type=str, default="reports/subdomain_assignments.csv",
                        help="子领域分类结果 CSV 导出路径")
    parser.add_argument("--subdomain-only", action="store_true",
                        help="仅运行子领域分类并导出结果，不进入后续流水线")
    parser.add_argument("--subdomain-graph-only", action="store_true",
                        help="仅运行子领域分类并写入 Neo4j，不进入后续流水线")
    parser.add_argument("--refine-subdomains-only", action="store_true",
                        help="仅运行子领域 candidate refinement，不处理论文")
    parser.add_argument("--refinement-report", type=str, default="reports/subdomain_refinement_decisions.csv",
                        help="子领域 refinement 决策 CSV 导出路径")
    parser.add_argument("--extraction-only", action="store_true",
                        help="仅运行 Extraction 并导出结果，不进入后续流水线")
    parser.add_argument("--extraction-report", type=str, default="reports/extraction_claims.csv",
                        help="Extraction 结果 CSV 导出路径")
    parser.add_argument("--vector-model", type=str, default="BAAI/bge-m3",
                        help="Grounding 使用的向量模型名称")
    parser.add_argument("--clear-db", action="store_true",
                        help="开始前清空 Neo4j 数据库")
    parser.add_argument("--timing-report", type=str, default=None,
                        help="保存计时报告的 JSON 文件路径（如 logs/timing.json）")
    parser.add_argument("--neo4j-uri", type=str, default=None,
                        help="Neo4j 连接 URI")
    parser.add_argument("--neo4j-user", type=str, default=None,
                        help="Neo4j 用户名")
    parser.add_argument("--neo4j-password", type=str, default=None,
                        help="Neo4j 密码")
    
    args = parser.parse_args()
    
    try:
        main(
            data_dir=args.data_dir,
            max_papers=args.max_papers,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            validation_batch_size=args.validation_batch_size,
            use_cache=not args.no_cache,
            clear_db=args.clear_db,
            timing_report=args.timing_report,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            skip_validation=args.skip_validation,
            skip_subdomain=args.skip_subdomain,
            subdomain_report=args.subdomain_report,
            subdomain_only=args.subdomain_only,
            subdomain_graph_only=args.subdomain_graph_only,
            refine_subdomains_only=args.refine_subdomains_only,
            refinement_report=args.refinement_report,
            extraction_only=args.extraction_only,
            extraction_report=args.extraction_report,
            vector_model_name=args.vector_model
        )
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)
