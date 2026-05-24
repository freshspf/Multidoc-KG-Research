"""
Main orchestration entry point for Multi-Paper Knowledge Graph Construction Framework.
"""
import os
import queue
import threading
from schema import Paper
from core.logger import setup_logger
from core.llm_client import LLMClient
from core.neo4j_store import Neo4jGraphStore
from core.vector_store import VectorStore
from core.timing import TimingStats, timed_step
from agents.extraction import ExtractionAgent
from agents.grounding import SemanticGroundingAgent
from agents.validation import KnowledgeValidationAgent
from agents.evolution import KnowledgeEvolutionAgent
from data_loader import PaperDataLoader

# Initialize logger at the very beginning
logger = setup_logger(__name__)

def main(data_dir: str = "data/preprocess/data", max_papers: int = None,
         neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None,
         timing_report: str = None):
    """
    Main workflow orchestration.

    Args:
        data_dir: Directory containing JSON paper files
        max_papers: Maximum number of papers to process (None = all)
        timing_report: Optional path to save timing report as JSON
    """
    logger.info("=" * 60)
    logger.info("Multi-Paper Knowledge Graph Construction Framework")
    logger.info("=" * 60)
    
    # Load papers from JSON files
    logger.info("Loading papers from JSON files...")
    data_loader = PaperDataLoader(data_dir=data_dir)
    papers = data_loader.load_all_papers()
    
    if not papers:
        logger.error("No papers loaded. Exiting...")
        return
    
    # Limit number of papers if specified
    if max_papers is not None and max_papers > 0:
        papers = papers[:max_papers]
        logger.info(f"Processing first {len(papers)} papers")
    
    # Initialize dependencies
    logger.info("Initializing dependencies...")
    llm_client = LLMClient(model_name="gpt-4o")
    
    # Initialize Neo4j Graph Store
    # 优先级：函数参数 > 环境变量 > 默认值
    neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password123")
    
    graph_store = Neo4jGraphStore(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password
    )
    
    vector_store = VectorStore(model_name='all-MiniLM-L6-v2')
    
    # Initialize all 4 Agents with dependency injection
    logger.info("Initializing agents...")
    extraction_agent = ExtractionAgent(llm_client=llm_client)
    grounding_agent = SemanticGroundingAgent(
        llm_client=llm_client,
        vector_store=vector_store
    )
    validation_agent = KnowledgeValidationAgent(
        llm_client=llm_client,
        graph_store=graph_store
    )
    evolution_agent = KnowledgeEvolutionAgent(graph_store=graph_store)
    
    # Process each paper through the pipeline
    logger.info("=" * 60)
    logger.info("Starting Multi-Paper Processing Pipeline")
    logger.info("=" * 60)

    total_claims = 0
    total_grounded = 0
    total_validated = 0
    total_written = 0
    timing_stats = TimingStats()
    evolution_lock = threading.Lock()  # 保护 Neo4j 并发写

    DONE = object()  # 流水线哨兵值
    q_grounding  = queue.Queue()
    q_validation = queue.Queue()
    q_evolution  = queue.Queue()

    # --- 流水线各阶段线程 ---

    def extraction_stage():
        for idx, paper in enumerate(papers, 1):
            logger.info(f">>> [{idx}/{len(papers)}] Extraction: {paper.id}")
            try:
                claims = timed_step(timing_stats, idx, "Extraction", extraction_agent.process, paper)
                # 早退调度：无 claims 直接跳过后续阶段
                if not claims:
                    logger.warning(f">>> [{idx}] No claims extracted, skipping paper")
                    continue
                q_grounding.put((idx, paper, claims))
            except Exception as e:
                logger.error(f"Extraction error for {paper.id}: {e}", exc_info=True)
        q_grounding.put(DONE)

    def grounding_stage():
        while True:
            item = q_grounding.get()
            if item is DONE:
                q_validation.put(DONE)
                break
            idx, paper, claims = item
            logger.info(f">>> [{idx}] Grounding: {paper.id}")
            try:
                grounded = timed_step(timing_stats, idx, "Grounding", grounding_agent.process, claims)
                # 早退调度：无 grounded claims 跳过
                if not grounded:
                    logger.warning(f">>> [{idx}] No grounded claims, skipping paper")
                    continue
                q_validation.put((idx, paper, claims, grounded))
            except Exception as e:
                logger.error(f"Grounding error for {paper.id}: {e}", exc_info=True)

    def validation_stage():
        while True:
            item = q_validation.get()
            if item is DONE:
                q_evolution.put(DONE)
                break
            idx, paper, claims, grounded = item
            logger.info(f">>> [{idx}] Validation: {paper.id}")
            try:
                validated = timed_step(timing_stats, idx, "Validation", validation_agent.process, grounded)
                q_evolution.put((idx, paper, claims, grounded, validated))
            except Exception as e:
                logger.error(f"Validation error for {paper.id}: {e}", exc_info=True)

    # --- 启动流水线线程 ---
    threads = [
        threading.Thread(target=extraction_stage, name="Extraction", daemon=True),
        threading.Thread(target=grounding_stage,  name="Grounding",  daemon=True),
        threading.Thread(target=validation_stage, name="Validation",  daemon=True),
    ]
    for t in threads:
        t.start()

    # Evolution 在主线程串行执行（保护 Neo4j 写安全）
    while True:
        item = q_evolution.get()
        if item is DONE:
            break
        idx, paper, claims, grounded, validated = item
        logger.info(f">>> [{idx}] Evolution: {paper.id}")
        try:
            with evolution_lock:
                written_ids = timed_step(timing_stats, idx, "Evolution", evolution_agent.process, validated)
            num_validated = sum(1 for c in validated if c.status.value == 'validated')
            total_claims    += len(claims)
            total_grounded  += len(grounded)
            total_validated += num_validated
            total_written   += len(written_ids)
            logger.info(f">>> [{idx}] Paper Summary:")
            logger.info(f"  - Claims extracted: {len(claims)}")
            logger.info(f"  - Claims grounded:  {len(grounded)}")
            logger.info(f"  - Claims validated: {num_validated}")
            logger.info(f"  - Claims written:   {len(written_ids)}")
        except Exception as e:
            logger.error(f"Evolution error for {paper.id}: {e}", exc_info=True)

    for t in threads:
        t.join()
    
    # Get final graph statistics
    logger.info("=" * 60)
    logger.info("Final Graph Statistics")
    logger.info("=" * 60)
    stats = graph_store.get_stats()

    # Output agent timing report
    timing_stats.format_report(logger=logger)
    if timing_report:
        timing_stats.save_json(timing_report)
        logger.info(f"Timing report saved to: {timing_report}")

    # Print overall summary
    logger.info("=" * 60)
    logger.info("Overall Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"Papers processed: {len(papers)}")
    logger.info(f"Total claims extracted: {total_claims}")
    logger.info(f"Total claims grounded: {total_grounded}")
    logger.info(f"Total claims validated: {total_validated}")
    logger.info(f"Total claims written to graph: {total_written}")
    logger.info(f"Graph Database:")
    logger.info(f"  - Nodes: {stats['nodes']}")
    logger.info(f"  - Relationships: {stats['relationships']}")
    
    logger.info("=" * 60)
    logger.info("=== Workflow Finished ===")
    logger.info("=" * 60)
    
    # Close database connection
    graph_store.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Paper Knowledge Graph Construction")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/preprocess/data",
        help="Directory containing JSON paper files (default: data/preprocess/data)"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to process (default: all)"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=None,
        help="Neo4j connection URI (default: from env or bolt://localhost:7687)"
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default=None,
        help="Neo4j username (default: from env or neo4j)"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=None,
        help="Neo4j password (default: from env or password123)"
    )
    parser.add_argument(
        "--timing-report",
        type=str,
        default=None,
        help="Path to save agent timing report as JSON (e.g., logs/timing_YYYYMMDD_HHMMSS.json)"
    )
    args = parser.parse_args()
    
    try:
        main(
            data_dir=args.data_dir,
            max_papers=args.max_papers,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            timing_report=args.timing_report
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
