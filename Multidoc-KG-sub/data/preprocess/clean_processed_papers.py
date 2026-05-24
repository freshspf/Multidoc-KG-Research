#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process flattened PDF JSON chunks into cleaner biomedical-paper text.

Usage:
    python data/preprocess/clean_processed_papers.py
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from bookmark_based_splitter import BookmarkBasedSplitter


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


FRONT_MATTER_PATTERNS = (
    r"^running title\b",
    r"^cite this\b",
    r"^open access\b",
    r"^received\b",
    r"^accepted\b",
    r"^published\b",
    r"^available online\b",
    r"^associate editor\b",
    r"^how to cite this article\b",
    r"^\*?correspondence\b",
    r"^corresponding author\b",
    r"^affiliations?\b",
    r"^authors?' contributions?\b",
    r"^author contributions?\b",
    r"^equal contribution\b",
    r"^contributed equally to this work\b",
    r"^license and terms\b",
    r"^copyright\b",
)

CAPTION_START_PATTERNS = (
    r"^(fig(?:ure)?\.?)\s*\d+",
    r"^(table)\s*\d+",
    r"^(scheme)\s*\d+",
)


class ProcessedPaperCleaner:
    def __init__(self, splitter: BookmarkBasedSplitter):
        self.splitter = splitter

    def _load_records(self, path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []

    def _strip_inline_noise(self, line: str) -> str:
        line = re.sub(r"https?://doi\.org/\S+", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\bdoi\s*:?\s*\S+", "", line, flags=re.IGNORECASE)
        line = re.sub(r"©\s*the author\(s\).*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"creative\s+commons.*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"license and terms:.*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\s{2,}", " ", line)
        return line.strip(" -;\t")

    def _looks_like_body_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        heading, is_tail, _ = self.splitter._parse_heading_line(stripped)
        if heading and not is_tail:
            return True

        if re.match(r"^\d+(?:\.\d+){0,3}\s+[A-Z]", stripped):
            return True

        words = stripped.split()
        if len(words) >= 8 and re.search(r"[A-Za-z]{3,}", stripped):
            return True

        return False

    def _looks_like_author_name_line(self, line: str) -> bool:
        stripped = line.strip().replace("‑", "-").replace("–", "-")
        tokens = [token for token in re.split(r"\s+", stripped) if token]
        if not 2 <= len(tokens) <= 6:
            return False

        cleaned_tokens = []
        for token in tokens:
            token = re.sub(r"^[\*\d,.;:()]+|[\*\d,.;:()]+$", "", token)
            if token.lower() in {"and", "&"}:
                continue
            cleaned_tokens.append(token)

        if len(cleaned_tokens) < 2:
            return False

        valid = 0
        for token in cleaned_tokens:
            if re.match(r"^[A-Z][A-Za-z'\-]+$", token):
                valid += 1
            elif re.match(r"^[A-Z]\.$", token):
                valid += 1

        return valid >= max(2, len(cleaned_tokens) - 1)

    def _is_front_matter_line(self, line: str) -> bool:
        stripped = line.strip()
        lowered = stripped.lower()

        if any(re.match(pattern, stripped, flags=re.IGNORECASE) for pattern in FRONT_MATTER_PATTERNS):
            return True
        if "creativecommons" in lowered:
            return True
        if "all rights reserved" in lowered:
            return True
        if "licensee" in lowered:
            return True
        if "contributed equally to this work" in lowered:
            return True
        if "mail stop" in lowered:
            return True
        if "fax:" in lowered:
            return True
        if "tel:" in lowered:
            return True
        if self._looks_like_author_name_line(stripped):
            return True
        if "@" in stripped and len(stripped) < 180:
            return True
        if re.search(r"\b(university|department|faculty|hospital|institute|school of)\b", lowered) and len(stripped.split()) <= 20:
            return True
        if re.match(r"^[A-Z][A-Za-z.\-']+(?:,\s*[A-Z][A-Za-z.\-']+)?(?:\s+\d+[\*,]*)?$", stripped):
            return True
        return False

    def _is_caption_start(self, line: str) -> bool:
        stripped = line.strip()
        if any(re.match(pattern, stripped, flags=re.IGNORECASE) for pattern in CAPTION_START_PATTERNS):
            return True
        if re.match(r"^\([A-Z]\)\s", stripped):
            return True
        return False

    def _clean_lines(self, lines: List[str], section_title: str) -> List[str]:
        cleaned_lines: List[str] = []
        skipping_caption = False
        skipping_front_block = False
        in_keyword_block = False
        keyword_continuations = 0

        for raw_line in lines:
            line = self._strip_inline_noise(raw_line)
            if not line:
                continue

            heading, is_tail, _ = self.splitter._parse_heading_line(line)
            if heading and is_tail:
                break

            if section_title == "Abstract":
                if re.match(r"^keywords?\b", line, flags=re.IGNORECASE):
                    cleaned_lines.append(re.sub(r"\s{2,}", " ", line))
                    in_keyword_block = True
                    keyword_continuations = 0
                    continue

                if in_keyword_block:
                    if self._is_front_matter_line(line) or self._is_caption_start(line):
                        break
                    if len(line.split()) <= 12 and keyword_continuations < 2:
                        cleaned_lines.append(line)
                        keyword_continuations += 1
                        continue
                    break

            if self._is_caption_start(line):
                skipping_caption = True
                continue

            if skipping_caption:
                if self._looks_like_body_line(line) or (heading and not is_tail):
                    skipping_caption = False
                else:
                    continue

            if self._is_front_matter_line(line):
                skipping_front_block = True
                continue

            if skipping_front_block:
                if self._looks_like_body_line(line):
                    skipping_front_block = False
                else:
                    continue

            cleaned_lines.append(line)

        if section_title in {"Abstract", "Introduction", "Background", "Chunk 1"}:
            while cleaned_lines and self._is_front_matter_line(cleaned_lines[0]):
                cleaned_lines.pop(0)

        return cleaned_lines

    def _clean_record(
        self,
        record: Dict[str, Any],
        repeated_boundary_lines: set[str],
    ) -> Dict[str, Any] | None:
        metadata = dict(record.get("metadata") or {})
        original_title = str(metadata.get("section_title") or "").strip()
        title_heading, is_tail_title, _ = self.splitter._parse_heading_line(original_title)
        if is_tail_title:
            return None

        cleaned_text = self.splitter.clean_text_block(str(record.get("text") or ""), repeated_boundary_lines)
        cleaned_text, has_tail = self.splitter._truncate_tail_matter(cleaned_text)
        if not cleaned_text:
            return None

        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        if not lines:
            return None

        heading, is_tail, remainder = self.splitter._parse_heading_line(lines[0])
        if is_tail:
            return None

        if heading and (not original_title or original_title.lower().startswith("chunk")):
            metadata["section_title"] = heading
            lines = ([remainder] if remainder else []) + lines[1:]
        elif title_heading:
            metadata["section_title"] = title_heading

        section_title = str(metadata.get("section_title") or original_title or "").strip()
        lines = self._clean_lines(lines, section_title)
        if not lines:
            return None

        cleaned_text, has_tail_after_filter = self.splitter._truncate_tail_matter("\n".join(lines))
        has_tail = has_tail or has_tail_after_filter
        if not cleaned_text:
            return None

        merged_lines = self.splitter._merge_wrapped_lines([line.strip() for line in cleaned_text.splitlines() if line.strip()])
        cleaned_text = "\n".join(merged_lines).strip()
        if len(cleaned_text.split()) < 20:
            return None

        if (
            section_title.lower().startswith("chunk")
            and int(metadata.get("page_start") or 1) == 1
            and len(cleaned_text.split()) < 180
        ):
            body_line_count = sum(1 for line in merged_lines if self._looks_like_body_line(line))
            if body_line_count < 2:
                return None

        metadata["cleaned"] = True
        metadata["tail_truncated"] = has_tail

        cleaned_record = {
            "id": record.get("id"),
            "metadata": metadata,
            "text": cleaned_text,
        }
        return cleaned_record

    def clean_file(self, input_path: Path, output_path: Path) -> Dict[str, int]:
        records = self._load_records(input_path)
        repeated_boundary_lines = self.splitter._identify_repeated_boundary_lines(
            [str(record.get("text") or "") for record in records]
        )

        cleaned_records: List[Dict[str, Any]] = []
        dropped = 0

        for record in records:
            cleaned_record = self._clean_record(record, repeated_boundary_lines)
            if cleaned_record is None:
                dropped += 1
                continue
            cleaned_records.append(cleaned_record)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(cleaned_records, file, indent=2, ensure_ascii=False)

        return {
            "input_records": len(records),
            "output_records": len(cleaned_records),
            "dropped_records": dropped,
        }

    def clean_folder(self, input_folder: Path, output_folder: Path) -> None:
        output_folder.mkdir(parents=True, exist_ok=True)
        json_files = sorted(input_folder.glob("*.json"))
        logger.info("Found %s JSON files in %s", len(json_files), input_folder)

        total_input = 0
        total_output = 0
        total_dropped = 0

        for index, input_path in enumerate(json_files, start=1):
            output_path = output_folder / input_path.name
            stats = self.clean_file(input_path, output_path)
            total_input += stats["input_records"]
            total_output += stats["output_records"]
            total_dropped += stats["dropped_records"]
            logger.info(
                "Cleaned %s/%s: %s -> %s records",
                index,
                len(json_files),
                input_path.name,
                stats["output_records"],
            )

        logger.info("Finished cleaning processed papers")
        logger.info("Input records: %s", total_input)
        logger.info("Output records: %s", total_output)
        logger.info("Dropped records: %s", total_dropped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean flattened PDF chunk JSON files")
    parser.add_argument("--input", type=str, default="data/processed_papers", help="Input JSON folder")
    parser.add_argument("--output", type=str, default="data/cleaned_papers", help="Output JSON folder")
    parser.add_argument("--max-chunk-size", type=int, default=5000, help="Chunk size used by helper cleaner")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    input_folder = project_root / args.input
    output_folder = project_root / args.output

    if not input_folder.exists():
        raise SystemExit(f"Input folder does not exist: {input_folder}")

    splitter = BookmarkBasedSplitter(max_chunk_size=args.max_chunk_size)
    cleaner = ProcessedPaperCleaner(splitter)
    cleaner.clean_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
