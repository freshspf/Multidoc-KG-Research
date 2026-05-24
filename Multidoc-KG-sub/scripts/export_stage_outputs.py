"""
Export per-stage outputs for a small biomedical paper review set.

This script is intended for advisor-facing inspection. It exports:
- preprocess snapshots (processed/cleaned JSON + summaries)
- loaded paper view
- subdomain classification / refinement results
- extraction outputs
- grounding outputs
- validation outputs
- evolution write results
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.evolution import KnowledgeEvolutionAgent
from agents.extraction import ExtractionAgent
from agents.grounding import SemanticGroundingAgent
from agents.subdomain import SubdomainClassifierAgent
from agents.subdomain_refinement import SubdomainHierarchyRefinementAgent
from agents.validation import KnowledgeValidationAgent
from core.llm_client import LLMClient
from core.neo4j_store import Neo4jGraphStore
from core.vector_store import MockVectorStore, VectorStore
from data_loader import PaperDataLoader
from main import iter_paper_batches, persist_subdomain_assignments, run_subdomain_refinement
from schema import KnowledgeClaim, Paper


DEFAULT_PAPER_IDS = [
    "39669840",
    "39671082",
    "39677122",
    "39677775",
    "39764068",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def load_section_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        sections = payload.get("sections", [])
        if isinstance(sections, list):
            return [item for item in sections if isinstance(item, dict)]
    return []


def section_summary(sections: List[Dict[str, Any]], preview_chars: int = 300) -> Dict[str, Any]:
    items = []
    for section in sections:
        metadata = section.get("metadata", {}) or {}
        text = str(section.get("text", "") or "").strip()
        items.append(
            {
                "id": section.get("id", ""),
                "section_title": metadata.get("section_title") or section.get("title") or "",
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "cleaned": metadata.get("cleaned"),
                "tail_truncated": metadata.get("tail_truncated"),
                "text_length": len(text),
                "preview": text[:preview_chars],
            }
        )
    return {
        "section_count": len(items),
        "sections": items,
    }


def serialize_claims(claims: Iterable[KnowledgeClaim]) -> List[Dict[str, Any]]:
    return [claim.dict() for claim in claims]


def export_preprocess_views(
    paper_id: str,
    processed_dir: Path,
    cleaned_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    processed_path = processed_dir / f"{paper_id}.json"
    cleaned_path = cleaned_dir / f"{paper_id}.json"

    processed_sections = load_section_json(processed_path)
    cleaned_sections = load_section_json(cleaned_path)

    copy_if_exists(processed_path, output_dir / "01_processed_sections.json")
    copy_if_exists(cleaned_path, output_dir / "02_cleaned_sections.json")

    summary = {
        "paper_id": paper_id,
        "processed": section_summary(processed_sections),
        "cleaned": section_summary(cleaned_sections),
    }
    write_json(output_dir / "00_preprocess_summary.json", summary)
    return summary


def export_loaded_paper_view(paper: Paper, output_dir: Path) -> None:
    metadata = paper.metadata or {}
    payload = {
        "paper_id": paper.id,
        "title": paper.title,
        "abstract": paper.get_abstract(),
        "keywords": paper.get_keywords(),
        "classification_text": paper.build_classification_text(),
        "section_count": metadata.get("section_count"),
        "sections": metadata.get("sections", []),
        "content_preview": paper.content[:3000],
    }
    write_json(output_dir / "03_loaded_paper.json", payload)


def export_index_markdown(
    path: Path,
    paper_ids: List[str],
    summary_rows: List[Dict[str, Any]],
    batch_size: int,
    vector_model: str,
) -> None:
    lines = [
        "# 五篇 PubMed 论文分阶段导出",
        "",
        f"- 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 论文数量：{len(paper_ids)}",
        f"- 论文 ID：{', '.join(paper_ids)}",
        f"- 子领域批次大小：{batch_size}",
        f"- Grounding 向量模型：{vector_model}",
        "",
        "## 每篇论文结果目录",
        "",
    ]

    for row in summary_rows:
        lines.append(
            f"- {row['paper_id']} | subdomain={row.get('subdomain','')} | "
            f"extracted={row.get('claims_extracted', 0)} | grounded={row.get('claims_grounded', 0)} | "
            f"validated={row.get('claims_validated', 0)} | written={row.get('claims_written', 0)}"
        )

    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "paper_id",
        "paper_title",
        "subdomain",
        "parent_domain",
        "subdomain_status",
        "taxonomy_version",
        "claims_extracted",
        "claims_grounded",
        "claims_validated",
        "claims_written",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Export stage outputs for five biomedical papers.")
    parser.add_argument("--paper-ids", nargs="+", default=DEFAULT_PAPER_IDS, help="Paper IDs to export")
    parser.add_argument("--processed-dir", default="data/processed_papers", help="Processed paper JSON directory")
    parser.add_argument("--cleaned-dir", default="data/cleaned_papers", help="Cleaned paper JSON directory")
    parser.add_argument(
        "--output-dir",
        default=f"reports/stage_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output directory",
    )
    parser.add_argument("--batch-size", type=int, default=5, help="Subdomain classification batch size")
    parser.add_argument("--vector-model", default="BAAI/bge-m3", help="Grounding vector model")
    parser.add_argument("--clear-db", action="store_true", help="Clear Neo4j before the export run")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "password123"))
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    ensure_dir(output_root)

    processed_dir = Path(args.processed_dir)
    cleaned_dir = Path(args.cleaned_dir)
    paper_ids = list(dict.fromkeys(args.paper_ids))

    llm_client = LLMClient(model_name="deepseek-chat")
    graph_store = Neo4jGraphStore(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )
    if args.clear_db:
        graph_store.clear_all()
    graph_store.ensure_subdomain_root("Biomedicine")

    data_loader = PaperDataLoader(str(cleaned_dir))
    papers: List[Paper] = [
        data_loader.load_paper_from_json(f"{paper_id}.json")
        for paper_id in paper_ids
    ]

    selected_rows: List[Dict[str, Any]] = []
    for paper in papers:
        paper_dir = output_root / paper.id
        ensure_dir(paper_dir)
        export_preprocess_views(paper.id, processed_dir, cleaned_dir, paper_dir)
        export_loaded_paper_view(paper, paper_dir)

    subdomain_agent = SubdomainClassifierAgent(
        llm_client=llm_client,
        config_path="config/subdomain_config.yaml",
        hierarchy_provider=graph_store,
    )
    refinement_agent = SubdomainHierarchyRefinementAgent(
        llm_client=llm_client,
        config_path="config/subdomain_config.yaml",
    )

    taxonomy_version = graph_store.get_current_taxonomy_version()
    all_refinement_rows: List[Dict[str, Any]] = []

    batches = iter_paper_batches(papers, args.batch_size)
    for batch_idx, batch in enumerate(batches, 1):
        batch_id = f"stage_export_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{batch_idx:03d}"
        hierarchy_snapshot = subdomain_agent.get_hierarchy_snapshot()

        for paper in batch:
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

            paper_dir = output_root / paper.id
            write_json(paper_dir / "04_subdomain_candidate.json", assignment.to_dict())

        persist_subdomain_assignments(graph_store, batch)
        batch_refinement_rows, taxonomy_version = run_subdomain_refinement(
            graph_store=graph_store,
            refinement_agent=refinement_agent,
            batch_id=batch_id,
            current_version=taxonomy_version,
        )
        all_refinement_rows.extend(batch_refinement_rows)

        decision_map = {
            str(row.get("candidate", "")).strip().lower(): row
            for row in batch_refinement_rows
        }
        for paper in batch:
            metadata = paper.metadata or {}
            candidate_name = str(metadata.get("subdomain", "")).strip()
            if not candidate_name:
                continue
            decision = decision_map.get(candidate_name.lower())
            if not decision:
                final_payload = metadata.get("subdomain_assignment", {})
            else:
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
                final_payload = metadata.get("subdomain_assignment", {})

            paper_dir = output_root / paper.id
            write_json(paper_dir / "05_subdomain_final.json", final_payload)

    write_json(output_root / "subdomain_refinement_decisions.json", all_refinement_rows)

    try:
        vector_store = VectorStore(model_name=args.vector_model)
    except Exception as exc:
        print(f"[export_stage_outputs] 向量模型初始化失败，退回 MockVectorStore: {exc}")
        vector_store = MockVectorStore()

    extraction_agent = ExtractionAgent(
        llm_client=llm_client,
        config_path="config/extraction_config.yaml",
    )
    grounding_agent = SemanticGroundingAgent(
        llm_client=llm_client,
        vector_store=vector_store,
    )
    validation_agent = KnowledgeValidationAgent(
        llm_client=llm_client,
        graph_store=graph_store,
        use_cache=True,
        batch_size=20,
        max_workers=8,
        enable_parallel=True,
        skip_validation=False,
    )
    evolution_agent = KnowledgeEvolutionAgent(graph_store=graph_store)

    for paper in papers:
        paper_dir = output_root / paper.id

        extracted_claims = extraction_agent.process(paper)
        write_json(paper_dir / "06_extraction_claims.json", serialize_claims(extracted_claims))

        grounded_claims = grounding_agent.process(deepcopy(extracted_claims))
        write_json(paper_dir / "07_grounding_claims.json", serialize_claims(grounded_claims))

        validated_claims = validation_agent.process(deepcopy(grounded_claims))
        write_json(paper_dir / "08_validation_claims.json", serialize_claims(validated_claims))

        written_ids = evolution_agent.process(validated_claims)
        write_json(
            paper_dir / "09_evolution_result.json",
            {
                "paper_id": paper.id,
                "written_claim_ids": written_ids,
                "graph_stats_after_write": graph_store.get_stats(),
            },
        )

        selected_rows.append(
            {
                "paper_id": paper.id,
                "paper_title": paper.title,
                "subdomain": str(paper.metadata.get("subdomain", "")),
                "parent_domain": str(paper.metadata.get("parent_domain", "")),
                "subdomain_status": str(paper.metadata.get("subdomain_status", "")),
                "taxonomy_version": paper.metadata.get("taxonomy_version", ""),
                "claims_extracted": len(extracted_claims),
                "claims_grounded": len(grounded_claims),
                "claims_validated": sum(1 for claim in validated_claims if claim.status.value == "validated"),
                "claims_written": len(written_ids),
            }
        )

    export_summary_csv(output_root / "stage_summary.csv", selected_rows)
    export_index_markdown(
        output_root / "README.md",
        paper_ids=paper_ids,
        summary_rows=selected_rows,
        batch_size=args.batch_size,
        vector_model=args.vector_model,
    )
    write_json(output_root / "final_graph_stats.json", graph_store.get_stats())
    graph_store.close()


if __name__ == "__main__":
    main()
