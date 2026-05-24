#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster and Select Representative Relations from Mentor Task CSV

This script uses LLM to:
1. Extract all unique relations from the CSV
2. Cluster similar relations by semantic meaning
3. Select 20-30 most representative relations
4. Convert to RDF format (URI style)
5. Generate detailed report

Input: outputs/mentor_50papers.csv
Output: mentor_task/representative_relations.csv + report
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Z0-9_]+):-([^}]+)\}")
_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env_vars(text: str) -> str:
    def repl_default(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        return os.getenv(name, default)

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        return os.getenv(name, "")

    text = _ENV_DEFAULT_RE.sub(repl_default, text)
    text = _ENV_RE.sub(repl, text)
    return text


def _load_config_compat(config_path: str) -> Dict[str, Any]:
    content = Path(config_path).read_text(encoding="utf-8")
    content = _expand_env_vars(content)

    try:
        import yaml  # type: ignore
        return yaml.safe_load(content) or {}
    except Exception:
        cfg: Dict[str, Any] = {"api": {}, "extractor": {}}
        current = None
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*:\s*$", line):
                current = line[:-1].strip()
                continue
            m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+?)\s*$", line)
            if not m or current not in {"api", "extractor"}:
                continue
            key, val = m.group(1), m.group(2)
            val = val.split("#", 1)[0].strip()
            val = val.strip('"').strip("'")
            if re.fullmatch(r"\d+(\.\d+)?", val):
                num: Any = float(val) if "." in val else int(val)
                cfg[current][key] = num
            else:
                cfg[current][key] = val
        return cfg


def _validate_environment_min(config: Dict[str, Any]) -> List[str]:
    api_config = config.get("api", {}) if config else {}
    api_key = api_config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    missing = []
    if not api_key:
        missing.append("OPENAI_API_KEY (or api.openai_api_key in config)")
    return missing


def _load_dotenv_if_present(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str
    source_paper: str
    source_section: str
    source_chunk: str
    context_text: str

    def __str__(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.object}"

    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


@dataclass(frozen=True)
class OpenAICompatClient:
    api_key: str
    base_url: str
    timeout_s: float

    def chat_completions_create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return (
            (((payload.get("choices") or [{}])[0]).get("message") or {}).get("content")
            or ""
        )


def _openai_client_from_config(config: Dict[str, Any]) -> OpenAICompatClient:
    api_config = config.get("api", {}) if config else {}
    api_key = api_config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    base_url = api_config.get("openai_base_url") or os.getenv("OPENAI_BASE_URL")
    timeout = float(api_config.get("timeout", 600.0))
    if not base_url:
        base_url = "https://api.openai.com/v1"
    return OpenAICompatClient(
        api_key=str(api_key), base_url=str(base_url), timeout_s=timeout
    )


def _call_llm(
    client: OpenAICompatClient,
    model: str,
    temperature: float,
    max_tokens: int,
    prompt: str,
) -> str:
    return client.chat_completions_create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in biomedical knowledge graph and ontology design.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _extract_json(text: str, debug_path: Optional[Path] = None) -> Dict[str, Any]:
    """Extract JSON from LLM response with multiple fallback strategies.

    Args:
        text: Raw LLM response
        debug_path: If provided, save raw response for debugging

    Returns:
        Parsed JSON dict

    Raises:
        ValueError: If no valid JSON found
    """
    text = text.strip()

    # Save raw response for debugging if path provided
    if debug_path:
        debug_path.write_text(text, encoding="utf-8")

    # Try raw JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try fenced code block ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Try to find complete JSON object by matching braces
    # This handles nested objects correctly
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx >= 0:
                json_str = text[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except Exception:
                    pass

    # Last resort: try to fix common JSON errors
    # Remove trailing commas, fix quotes, etc.
    try:
        # Remove common markdown artifacts
        cleaned = re.sub(r'```[a-z]*\n?', '', text)
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except Exception:
        pass

    raise ValueError(f"Could not extract valid JSON from LLM response. Saved to: {debug_path if debug_path else 'N/A'}")


def parse_location_from_snippet(snippet: str) -> Tuple[str, str]:
    """Extract source_section and source_chunk from snippet string.

    Snippet format: "位置: 前言 / chunk 0 | 证据: ..."
    Returns: (source_section, source_chunk)
    """
    section = "Unknown"
    chunk = "0"

    # Extract section
    m = re.search(r"位置:\s*([^/|]+)", snippet)
    if m:
        section = m.group(1).strip()

    # Extract chunk number
    m = re.search(r"chunk\s*(\d+)", snippet)
    if m:
        chunk = m.group(1).strip()

    return (section, chunk)


def parse_context_from_snippet(snippet: str) -> str:
    """Extract evidence text from snippet string.

    Snippet format: "位置: ... | 证据: ..."
    Returns: evidence text
    """
    m = re.search(r"证据:\s*(.+)", snippet)
    if m:
        return m.group(1).strip()
    return snippet


def load_relations_from_csv(csv_path: Path) -> List[Relation]:
    """Load relations from mentor CSV file."""
    relations = []

    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kind = row.get("类型", "").strip()
            if kind != "关系":
                continue

            name = row.get("名称", "").strip()
            source_paper = row.get("出处", "").strip()
            snippet = row.get("片段", "").strip()

            if not name or not source_paper:
                continue

            # Parse "Subject | predicate | Object"
            parts = [p.strip() for p in name.split("|")]
            if len(parts) != 3:
                continue

            subject, predicate, obj = parts

            # Parse metadata from snippet
            section, chunk = parse_location_from_snippet(snippet)
            context = parse_context_from_snippet(snippet)

            relations.append(
                Relation(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    source_paper=source_paper,
                    source_section=section,
                    source_chunk=chunk,
                    context_text=context,
                )
            )

    return relations


def get_unique_relations_with_frequency(
    relations: List[Relation],
) -> Counter:
    """Get frequency count of unique relations."""
    return Counter(rel.to_tuple() for rel in relations)


def build_clustering_prompt(
    unique_relations: List[Tuple[str, str, str]], target_clusters: int
) -> str:
    """Build prompt for LLM clustering task."""

    # Format relations for display
    relation_list = "\n".join(
        [
            f"{i+1}. {subj} | {pred} | {obj}"
            for i, (subj, pred, obj) in enumerate(unique_relations[:150])
        ]
    )

    return f"""You are a biomedical ontology expert. Your task is to cluster the following relations into {target_clusters} semantically meaningful groups.

**Clustering Guidelines:**

1. **Semantic Similarity**: Group relations that express similar semantic meanings
   - Example: "inhibits", "suppresses", "blocks" → Negative Regulation
   - Example: "treats", "therapy", "management" → Treatment

2. **Biomedical Domain**: Consider common biomedical relation types:
   - Causation (causes, leads to, induces)
   - Regulation (increases, decreases, regulates, inhibits)
   - Treatment (treats, targets, indicated_for)
   - Association (associated_with, correlated_with)
   - Measurement (has_measurement, has_clinical_feature)
   - Risk (risk_factor_for, protective_factor_for)

3. **Cluster Characteristics**:
   - Each cluster should have a clear, descriptive name
   - Provide a brief explanation of what the cluster represents
   - Aim for {target_clusters} clusters total
   - Ensure balanced distribution (avoid 1-2 huge clusters)

4. **Output Format**:
   - Output ONLY valid JSON (no explanations)
   - Use this exact schema:

{{
  "clusters": [
    {{
      "name": "Cluster Name (e.g., 'Positive Regulation')",
      "description": "Brief description of what this cluster represents",
      "relations": [
        {{"subject": "...", "predicate": "...", "object": "..."}},
        ...
      ]
    }}
  ]
}}

**Relations to Cluster:**

{relation_list}

{f"Note: Showing first 150 of {len(unique_relations)} relations (highest frequency). Focus on clustering these into meaningful groups." if len(unique_relations) > 150 else ""}

Remember: Output ONLY the JSON, no additional text.
"""


def build_selection_prompt(
    clusters: List[Dict], frequency_counter: Counter, target_n: int
) -> str:
    """Build prompt for selecting representative relations from clusters."""

    # Build cluster descriptions with frequencies
    cluster_descs = []
    for i, cluster in enumerate(clusters):
        cluster_name = cluster.get("name", f"Cluster {i+1}")
        description = cluster.get("description", "")
        relations = cluster.get("relations", [])

        # Add frequency information
        relations_with_freq = []
        for rel in relations:
            tuple_key = (rel["subject"], rel["predicate"], rel["object"])
            freq = frequency_counter.get(tuple_key, 0)
            relations_with_freq.append({**rel, "frequency": freq})

        # Sort by frequency
        relations_with_freq.sort(key=lambda x: x["frequency"], reverse=True)

        rel_list = "\n      ".join(
            [
                f"- {r['subject']} | {r['predicate']} | {r['object']} (freq: {r['frequency']})"
                for r in relations_with_freq[:10]
            ]
        )

        cluster_descs.append(
            f"""
    Cluster {i+1}: {cluster_name}
    Description: {description}
    Relations (top 10 by frequency):
      {rel_list}
"""
        )

    clusters_text = "\n".join(cluster_descs)

    return f"""You are a biomedical knowledge graph curator. Your task is to select the {target_n} most representative relations from the clusters below.

**Selection Criteria:**

1. **Frequency**: Prioritize relations that appear more frequently in the literature
2. **Semantic Importance**: Choose relations that are fundamental to biomedical knowledge
3. **Domain Coverage**: Ensure representation across different biomedical domains (diseases, drugs, mechanisms, measurements)
4. **Cluster Balance**: Select 1-2 representatives from each cluster, prioritizing larger clusters

5. **Selection Strategy**:
   - For each cluster, select the most frequently occurring relation
   - For larger clusters (10+ relations), select 2 representatives
   - For smaller clusters, select 1 representative
   - Ensure total selected ≈ {target_n} relations

**Output Format:**

Output ONLY valid JSON with this schema:

{{
  "selected_relations": [
    {{
      "subject": "...",
      "predicate": "...",
      "object": "...",
      "cluster": "Cluster Name",
      "rationale": "Brief explanation of why this relation was selected"
    }}
  ]
}}

**Clusters:**

{clusters_text}

Remember: Output ONLY the JSON, no additional text.
"""


def to_uri(s: str, keep_chinese: bool = True) -> str:
    """Convert string to RDF URI format.

    Rules:
    1. Clean special characters (keep alphanumeric, underscore, Chinese)
    2. Replace spaces with underscores
    3. Capitalize first letter of each word
    4. Add ':' prefix

    Examples:
        "Hypertension Diagnosis" → ":Hypertension_Diagnosis"
        "increases" → ":increases"
        "OBPM Monitoring" → ":OBPM_Monitoring"
    """
    if not s:
        return ":Unknown"

    # Split by spaces and underscores
    words = re.split(r"[\s_]+", s.strip())

    cleaned_words = []
    for word in words:
        if not word:
            continue
        # Keep alphanumeric, Chinese characters, and basic punctuation
        if keep_chinese:
            # Keep Chinese characters and alphanumeric
            cleaned = re.sub(r"[^\w\u4e00-\u9fff]", "", word)
        else:
            # Only keep alphanumeric
            cleaned = re.sub(r"[^\w]", "", word)

        if cleaned:
            # Capitalize first letter
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
            cleaned_words.append(cleaned)

    if not cleaned_words:
        return ":Unknown"

    uri = ":" + "_".join(cleaned_words)
    return uri


def find_best_context_for_relation(
    target_relation: Tuple[str, str, str], all_relations: List[Relation]
) -> Relation:
    """Find the best context (evidence snippet) for a given relation.

    Priority:
    1. Longest context_text (usually more complete)
    2. From Abstract/Introduction sections
    3. First occurrence if ties
    """
    matching = [r for r in all_relations if r.to_tuple() == target_relation]

    if not matching:
        return None

    if len(matching) == 1:
        return matching[0]

    # Sort by context length (descending), then by section priority
    section_priority = {"摘要": 0, "Abstract": 0, "前言": 1, "Introduction": 1, "引言": 1}

    def sort_key(rel: Relation):
        priority = section_priority.get(rel.source_section, 99)
        return (-len(rel.context_text), priority)

    matching.sort(key=sort_key)
    return matching[0]


def generate_markdown_report(
    selected_relations: List[Dict[str, Any]],
    clusters: List[Dict],
    stats: Dict[str, Any],
    output_path: Path,
):
    """Generate detailed markdown report."""

    lines = [
        "# 代表性关系选择报告\n",
        "## 1. 执行概要\n",
        f"- **执行时间**: {stats.get('timestamp', 'N/A')}",
        f"- **原始关系总数**: {stats.get('total_relations', 0)}",
        f"- **独特关系数**: {stats.get('unique_relations', 0)}",
        f"- **聚类数量**: {len(clusters)}",
        f"- **选中关系数**: {len(selected_relations)}\n",
        "## 2. 聚类结果\n",
    ]

    for i, cluster in enumerate(clusters):
        lines.append(f"\n### 聚类 {i+1}: {cluster.get('name', 'Unnamed')}")
        lines.append(f"\n**描述**: {cluster.get('description', 'No description')}\n")
        lines.append(f"**包含关系数**: {len(cluster.get('relations', []))}\n")

        rels = cluster.get("relations", [])
        if rels:
            lines.append("**关系列表** (前10个):\n")
            for rel in rels[:10]:
                lines.append(
                    f"- {rel['subject']} | {rel['predicate']} | {rel['object']}"
                )
            lines.append("")

    lines.append("\n## 3. 选中的代表性关系\n\n")

    for i, sel_rel in enumerate(selected_relations):
        lines.append(f"### {i+1}. {sel_rel['subject']} | {sel_rel['predicate']} | {sel_rel['object']}")
        lines.append(f"\n- **所属聚类**: {sel_rel.get('cluster', 'Unknown')}")
        lines.append(f"- **选择理由**: {sel_rel.get('rationale', 'N/A')}")
        lines.append(f"- **URI格式**: `: {sel_rel['subject']} | : {sel_rel['predicate']} | : {sel_rel['object']}`")
        if "frequency" in sel_rel:
            lines.append(f"- **出现频率**: {sel_rel['frequency']}")
        if "context" in sel_rel:
            lines.append(f"- **上下文**: {sel_rel['context'][:100]}...")
        lines.append("")

    lines.append("\n## 4. URI 映射表\n\n")
    lines.append("| 原始文本 | URI 格式 |")
    lines.append("|---------|----------|")

    # Collect all unique entities
    entities = set()
    for sel_rel in selected_relations:
        entities.add(sel_rel["subject"])
        entities.add(sel_rel["predicate"])
        entities.add(sel_rel["object"])

    for entity in sorted(entities):
        uri = to_uri(entity)
        lines.append(f"| {entity} | {uri} |")

    lines.append("\n## 5. 统计信息\n\n")
    lines.append(f"- 平均每个聚类的关系数: {stats.get('avg_relations_per_cluster', 0):.1f}")
    lines.append(f"- 最大聚类大小: {stats.get('max_cluster_size', 0)}")
    lines.append(f"- 最小聚类大小: {stats.get('min_cluster_size', 0)}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report generated: {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cluster and select representative relations using LLM"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--input",
        default="mentor_task/mentor_50papers.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        default="mentor_task/representative_relations.csv",
        help="Output CSV path (RDF format)",
    )
    parser.add_argument(
        "--report",
        default="mentor_task/representative_relations_report.md",
        help="Report output path",
    )
    parser.add_argument(
        "--n-relations",
        type=int,
        default=30,
        help="Target number of representative relations (default: 30)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=25,
        help="Target number of clusters (default: 25)",
    )
    parser.add_argument(
        "--keep-chinese",
        action="store_true",
        default=True,
        help="Keep Chinese characters in URIs (default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call API; print prompts only",
    )
    args = parser.parse_args(argv)

    _load_dotenv_if_present(Path(".env"))
    config = _load_config_compat(args.config)
    missing = _validate_environment_min(config)
    if missing and not args.dry_run:
        raise SystemExit(f"Missing API configuration: {', '.join(missing)}")

    # Use mentor_task specific configuration
    mentor_cfg = config.get("mentor_task", {})
    model = mentor_cfg.get("model", "deepseek-chat")
    temperature = float(mentor_cfg.get("temperature", 0.1))
    # Get token limits from config
    max_tokens = int(mentor_cfg.get("max_tokens", 8000))
    cluster_max_tokens = int(mentor_cfg.get("cluster_max_tokens", 12000))
    selection_max_tokens = int(mentor_cfg.get("selection_max_tokens", 10000))

    print(f"Loading relations from: {args.input}")
    relations = load_relations_from_csv(Path(args.input))
    print(f"Loaded {len(relations)} relations")

    # Get unique relations with frequency
    freq_counter = get_unique_relations_with_frequency(relations)
    unique_relations = list(freq_counter.keys())
    print(f"Found {len(unique_relations)} unique relations")

    # Sort by frequency and take top N for clustering
    unique_relations.sort(key=lambda x: freq_counter[x], reverse=True)
    top_relations = unique_relations[:150]

    print(f"\nUsing top {len(top_relations)} relations for clustering")

    client = None if args.dry_run else _openai_client_from_config(config)

    # Stage 1: Clustering
    print(f"\n{'='*60}")
    print("STAGE 1: Clustering relations")
    print(f"{'='*60}")

    cluster_prompt = build_clustering_prompt(top_relations, args.n_clusters)

    if args.dry_run:
        print("\n[DRY RUN] Clustering Prompt:")
        print("=" * 60)
        print(cluster_prompt[:3000])
        print("...")
        print("=" * 60)
        return 0

    print("Sending clustering request to LLM...")
    cluster_response = _call_llm(
        client=client,
        model=model,
        temperature=temperature,
        max_tokens=cluster_max_tokens,
        prompt=cluster_prompt,
    )

    print("Parsing clustering response...")
    # Save response for debugging
    debug_dir = Path("mentor_task/debug_output")
    debug_dir.mkdir(parents=True, exist_ok=True)
    cluster_debug_path = debug_dir / "cluster_response.txt"

    try:
        cluster_result = _extract_json(cluster_response, debug_path=cluster_debug_path)
        clusters = cluster_result.get("clusters", [])
        print(f"LLM created {len(clusters)} clusters")
    except ValueError as e:
        print(f"\n❌ Error parsing clustering response: {e}")
        print(f"Raw response saved to: {cluster_debug_path}")
        print(f"Response length: {len(cluster_response)} characters")
        print(f"\nFirst 500 characters of response:\n{cluster_response[:500]}")
        return 1

    # Stage 2: Representative selection
    print(f"\n{'='*60}")
    print("STAGE 2: Selecting representative relations")
    print(f"{'='*60}")

    selection_prompt = build_selection_prompt(clusters, freq_counter, args.n_relations)

    print("Sending selection request to LLM...")
    selection_response = _call_llm(
        client=client,
        model=model,
        temperature=temperature,
        max_tokens=selection_max_tokens,
        prompt=selection_prompt,
    )

    print("Parsing selection response...")
    # Save response for debugging
    selection_debug_path = debug_dir / "selection_response.txt"

    try:
        selection_result = _extract_json(selection_response, debug_path=selection_debug_path)
        selected = selection_result.get("selected_relations", [])
        print(f"LLM selected {len(selected)} representative relations")
    except ValueError as e:
        print(f"\n❌ Error parsing selection response: {e}")
        print(f"Raw response saved to: {selection_debug_path}")
        print(f"Response length: {len(selection_response)} characters")
        print(f"\nFirst 500 characters of response:\n{selection_response[:500]}")
        return 1

    # Prepare output data
    print(f"\n{'='*60}")
    print("Generating output files")
    print(f"{'='*60}")

    output_csv = Path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["subject", "predicate", "object", "source_section", "source_chunk", "context_text", "出处"]
        )

        for sel in selected:
            subj = sel["subject"]
            pred = sel["predicate"]
            obj = sel["object"]

            # Find best context
            rel_tuple = (subj, pred, obj)
            best_rel = find_best_context_for_relation(rel_tuple, relations)

            if best_rel:
                writer.writerow(
                    [
                        to_uri(subj, args.keep_chinese),
                        to_uri(pred, args.keep_chinese),
                        to_uri(obj, args.keep_chinese),
                        best_rel.source_section,
                        best_rel.source_chunk,
                        best_rel.context_text,
                        best_rel.source_paper,
                    ]
                )
            else:
                # Fallback if context not found
                writer.writerow(
                    [
                        to_uri(subj, args.keep_chinese),
                        to_uri(pred, args.keep_chinese),
                        to_uri(obj, args.keep_chinese),
                        "",
                        "",
                        "",
                        "",
                    ]
                )

            # Add frequency to selected dict for report
            sel["frequency"] = freq_counter.get(rel_tuple, 0)
            if best_rel:
                sel["context"] = best_rel.context_text

    print(f"CSV saved to: {output_csv}")

    # Generate report
    from datetime import datetime

    stats = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_relations": len(relations),
        "unique_relations": len(unique_relations),
        "avg_relations_per_cluster": sum(len(c.get("relations", [])) for c in clusters) / len(clusters) if clusters else 0,
        "max_cluster_size": max((len(c.get("relations", [])) for c in clusters), default=0),
        "min_cluster_size": min((len(c.get("relations", [])) for c in clusters), default=0),
    }

    generate_markdown_report(selected, clusters, stats, Path(args.report))

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print(f"Output CSV: {output_csv}")
    print(f"Report: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
