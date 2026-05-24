#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mentor task helper (moved to `mentor_task/`):
Randomly sample N processed PubMed papers and ask an LLM to extract *important*
concepts and relations with evidence snippets and approximate locations, then
export as a single CSV with 4 columns:
  类型（概念/关系）, 名称, 出处（paper文件名）, 片段（含位置：摘要/前言/等）

Input format: `data/processed_data/*.json`
Each JSON is a list of chunks like:
  { "id": "...", "metadata": {"section_title": "...", "page_start": 0, "page_end": 1}, "text": "..." }

Notes:
- Network calls are required (LLM API).
- Default behavior prioritizes Abstract/Introduction (and optionally Discussion/Conclusion) to reduce tokens,
  but you can pass --all-sections to include all chunks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
        # Minimal YAML parsing for this repo's config.yaml (api/extractor only).
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
class Chunk:
    index: int
    section_id: str
    section_title: str
    page_start: Optional[int]
    page_end: Optional[int]
    text: str


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _is_abstract(title: str) -> bool:
    t = (title or "").strip().lower()
    return bool(re.search(r"\babstract\b", t)) or t in {"摘要", "中文摘要", "摘要：", "abstract:"}


def _is_intro(title: str) -> bool:
    t = (title or "").strip().lower()
    return bool(re.search(r"\bintroduction\b", t)) or t in {"引言", "前言", "背景"}


def _is_discussion_or_conclusion(title: str) -> bool:
    t = (title or "").strip().lower()
    return bool(re.search(r"\bdiscussion\b|\bconclusion\b", t)) or t in {"讨论", "结论", "总结"}


def _load_chunks(processed_json_path: Path) -> List[Chunk]:
    raw = json.loads(processed_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list JSON in {processed_json_path}")

    chunks: List[Chunk] = []
    for idx, item in enumerate(raw):
        meta = item.get("metadata") or {}
        chunks.append(
            Chunk(
                index=idx,
                section_id=str(item.get("id") or f"chunk_{idx}"),
                section_title=str(meta.get("section_title") or ""),
                page_start=meta.get("page_start"),
                page_end=meta.get("page_end"),
                text=str(item.get("text") or item.get("content") or ""),
            )
        )
    return chunks


def _select_chunks(chunks: Sequence[Chunk], all_sections: bool) -> List[Chunk]:
    if all_sections:
        return [c for c in chunks if _norm(c.text)]

    picked: List[Chunk] = []
    for c in chunks:
        if _is_abstract(c.section_title) or _is_intro(c.section_title):
            if _norm(c.text):
                picked.append(c)

    # If nothing matched, take the first 2 non-empty chunks as a fallback.
    if not picked:
        for c in chunks:
            if _norm(c.text):
                picked.append(c)
            if len(picked) >= 2:
                break

    # Also include discussion/conclusion if present (often has key claims/limitations).
    for c in chunks:
        if _is_discussion_or_conclusion(c.section_title):
            if _norm(c.text) and c not in picked:
                picked.append(c)

    return picked


def _format_location(c: Chunk) -> str:
    parts = []
    if c.section_title:
        # Favor mentor wording for early sections.
        if _is_abstract(c.section_title):
            parts.append("摘要")
        elif _is_intro(c.section_title):
            parts.append("前言")
        else:
            parts.append(c.section_title)
    else:
        parts.append(c.section_id)

    parts.append(f"chunk {c.index}")
    if c.page_start is not None:
        if c.page_end is not None and c.page_end != c.page_start:
            parts.append(f"pages {int(c.page_start)+1}-{int(c.page_end)+1}")
        else:
            parts.append(f"page {int(c.page_start)+1}")
    return " / ".join(parts)


def _build_prompt(chunks: Sequence[Chunk]) -> str:
    # Keep the prompt concise but strict about output schema.
    chunk_blocks = []
    for c in chunks:
        text = c.text.strip()
        chunk_blocks.append(
            f"[CHUNK {c.index}] id={c.section_id} title={c.section_title!r} loc={_format_location(c)}\n{text}"
        )

    return "\n\n".join(
        [
            "You are a biomedical literature information extraction assistant. Task: extract important concepts and relations from the given paper snippets, and provide verifiable evidence spans with approximate locations.",
            "",
            "Extraction rules (must follow):",
            "1) Output JSON only (no explanations or extra text).",
            "2) Extract only IMPORTANT concepts/relations. Prioritize: diseases/conditions, symptoms, drugs/contrast agents, mechanisms/pathways, key experiments/methods, main findings/conclusions, major safety/adverse events, key doses/timepoints, key measurements/biomarkers and outcomes.",
            "3) Every item must include a short verbatim evidence snippet (<=200 chars) and a location label (section + chunk).",
            "4) Relation naming MUST be a concise label (single word or short multi-word), not a full sentence. Prefer lower_snake_case or single-token labels, e.g., has_good_friend, associated_with, increases, decreases, weight, height, dose, age, mortality, survival, efficacy.",
            "5) Relation format: use a triple string with a clear delimiter: \"Subject | relation_label | Object\". Example: \"Aspirin | reduces | platelet aggregation\"; \"Patient | weight | 70 kg\".",
            "6) Avoid low-value items (copyright/permissions, generic background, unrelated boilerplate).",
            "",
            "Output JSON schema (keys must be in Chinese):",
            "{",
            '  "items": [',
            "    {",
            '      "类型": "概念" | "关系",',
            '      "名称": "concept name OR relation triple string",',
            '      "片段": "位置: ... | 证据: ..."',
            "    }",
            "  ]",
            "}",
            "",
            "Paper snippets:",
            *chunk_blocks,
        ]
    )


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Try raw JSON first.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try fenced block.
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return json.loads(m.group(1))

    # Try first {...} block.
    m2 = re.search(r"(\{.*\})", text, flags=re.S)
    if m2:
        return json.loads(m2.group(1))

    raise ValueError("Model output is not valid JSON.")


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
    return OpenAICompatClient(api_key=str(api_key), base_url=str(base_url), timeout_s=timeout)


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
            {"role": "system", "content": "You are a careful biomedical information extraction assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _iter_processed_files(input_dir: Path) -> List[Path]:
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No *.json files found in {input_dir}")
    return files


def _sample_files(files: Sequence[Path], n: int, seed: int) -> List[Path]:
    if n > len(files):
        raise SystemExit(f"Requested n={n}, but only {len(files)} files available.")
    rnd = random.Random(seed)
    return rnd.sample(list(files), n)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract important concepts/relations into mentor CSV.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input-dir", default="data/processed_data", help="Processed JSON directory")
    parser.add_argument("--out", default="mentor_task/mentor_50papers.csv", help="Output CSV path (default: mentor_task/mentor_50papers.csv)")
    parser.add_argument("--n", type=int, default=50, help="Number of papers to sample")
    parser.add_argument("--seed", type=int, default=42, help="Sampling RNG seed")
    parser.add_argument("--all-sections", action="store_true", help="Include all chunks (more tokens/cost)")
    parser.add_argument("--per-paper-cap", type=int, default=40, help="Max items to keep per paper")
    parser.add_argument("--dry-run", action="store_true", help="Do not call API; print prompt for first sampled paper")
    args = parser.parse_args(argv)

    # Best-effort: load .env without requiring python-dotenv.
    _load_dotenv_if_present(Path(".env"))

    config = _load_config_compat(args.config)
    missing_required = _validate_environment_min(config)
    if missing_required:
        raise SystemExit(f"Missing required API configuration: {', '.join(missing_required)}")

    # Use mentor_task specific configuration
    mentor_cfg = config.get("mentor_task", {})
    model = mentor_cfg.get("model", "deepseek-chat")
    temperature = float(mentor_cfg.get("temperature", 0.1))
    max_tokens = int(mentor_cfg.get("max_tokens", 1200))

    processed_files = _iter_processed_files(Path(args.input_dir))
    sampled = _sample_files(processed_files, args.n, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = None if args.dry_run else _openai_client_from_config(config)

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["类型", "名称", "出处", "片段"])

        for i, p in enumerate(sampled):
            paper_id = p.stem  # e.g. paper_4_39906526
            chunks = _load_chunks(p)
            selected = _select_chunks(chunks, all_sections=args.all_sections)
            prompt = _build_prompt(selected)

            if args.dry_run:
                print(f"DRY RUN paper={paper_id} chunks={len(selected)}")
                print(prompt[:6000])
                return 0

            assert client is not None
            content = _call_llm(
                client=client,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt=prompt,
            )
            parsed = _extract_json(content)
            items = parsed.get("items") or []
            if not isinstance(items, list):
                items = []

            kept = 0
            for item in items:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("类型") or "").strip()
                name = str(item.get("名称") or "").strip()
                snippet = str(item.get("片段") or "").strip()
                if kind not in {"概念", "关系"} or not name or not snippet:
                    continue
                writer.writerow([kind, name, paper_id, snippet])
                kept += 1
                if kept >= args.per_paper_cap:
                    break

            print(f"[{i+1}/{len(sampled)}] wrote {kept} items from {paper_id}")

    print(f"Done. CSV saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
