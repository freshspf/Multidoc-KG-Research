#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform mentor CSV output to triple format.

Input: CSV with columns (类型, 名称, 出处, 片段)
Output: CSV with columns (subject, predicate, object, source_section, source_chunk, context_text)

Only processes rows where 类型 == "关系"
"""

import csv
import re
from pathlib import Path


def extract_location_info(location_str: str) -> tuple[str, str]:
    """
    Extract section and chunk info from location string.

    Examples:
    - "位置: 前言 / chunk 0 | 证据: ..." -> ("前言", "0")
    - "位置: Results and Discussion / chunk 7 | 证据: ..." -> ("Results and Discussion", "7")
    - "位置: Conclusion / chunk 12 | 证据: ..." -> ("Conclusion", "12")
    """
    # Extract the location part before "|"
    location_part = location_str.split("|")[0].strip()
    # Remove "位置:" prefix
    location_part = location_part.replace("位置:", "").strip()

    # Extract section and chunk
    if "/" in location_part:
        section_part, chunk_part = location_part.split("/", 1)
        section = section_part.strip()
        # Extract chunk number
        chunk_match = re.search(r"chunk\s+(\d+)", chunk_part.strip())
        chunk_num = chunk_match.group(1) if chunk_match else ""
        return section, chunk_num
    else:
        # Fallback: try to find chunk number
        chunk_match = re.search(r"chunk\s+(\d+)", location_part)
        if chunk_match:
            # Extract section from remaining text
            section = location_part[:chunk_match.start()].strip()
            return section, chunk_match.group(1)
        return location_part, ""


def extract_evidence(snippet_str: str) -> str:
    """
    Extract evidence text from snippet string.

    Examples:
    - "位置: 前言 / chunk 0 | 证据: Relugolix has higher biosafety..." -> "Relugolix has higher biosafety..."
    """
    # Extract the evidence part after "|"
    if "|" in snippet_str:
        evidence_part = snippet_str.split("|", 1)[1].strip()
        # Remove "证据:" prefix
        evidence = evidence_part.replace("证据:", "").strip()
        return evidence
    # Fallback: return original
    return snippet_str


def parse_triple(triple_str: str) -> tuple[str, str, str]:
    """
    Parse triple string "Subject | predicate | Object" into components.

    Examples:
    - "Relugolix | low_oral_bioavailability | poor water solubility"
      -> ("Relugolix", "low_oral_bioavailability", "poor water solubility")
    - "IGF2BP3 | 表达上调 | 子宫内膜癌组织"
      -> ("IGF2BP3", "表达上调", "子宫内膜癌组织")
    """
    parts = triple_str.split("|")
    if len(parts) == 3:
        subject = parts[0].strip()
        predicate = parts[1].strip()
        obj = parts[2].strip()
        return subject, predicate, obj
    elif len(parts) > 3:
        # Handle cases with extra | in the object
        subject = parts[0].strip()
        predicate = parts[1].strip()
        obj = "|".join(parts[2:]).strip()
        return subject, predicate, obj
    else:
        # Return as-is if format is unexpected
        return triple_str, "", ""


def transform_csv(input_path: Path, output_path: Path) -> dict:
    """
    Transform mentor CSV to triple format.

    Returns statistics about the transformation.
    """
    total_rows = 0
    relation_rows = 0
    skipped_rows = 0
    parse_errors = 0

    with input_path.open("r", encoding="utf-8-sig") as infile, \
         output_path.open("w", encoding="utf-8-sig", newline="") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write header
        writer.writerow([
            "subject", "predicate", "object",
            "source_section", "source_chunk", "context_text"
        ])

        # Skip header row
        next(reader)

        for row in reader:
            total_rows += 1

            if len(row) < 4:
                skipped_rows += 1
                continue

            kind, name, source, snippet = row[0], row[1], row[2], row[3]

            # Only process relations
            if kind != "关系":
                continue

            relation_rows += 1

            # Parse triple
            subject, predicate, obj = parse_triple(name)

            # Check for parse errors
            if not predicate or not obj:
                parse_errors += 1
                print(f"Warning: Could not parse triple: {name}")

            # Extract location info
            section, chunk = extract_location_info(snippet)

            # Extract evidence
            context = extract_evidence(snippet)

            # Write transformed row
            writer.writerow([
                subject, predicate, obj,
                section, chunk, context
            ])

    return {
        "total_rows": total_rows,
        "relation_rows": relation_rows,
        "skipped_rows": skipped_rows,
        "parse_errors": parse_errors,
        "output_rows": relation_rows - parse_errors
    }


def main():
    input_csv = Path("outputs/mentor_50papers.csv")
    output_csv = Path("mentor_task/mentor_50papers_triples.csv")

    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        return 1

    print(f"Transforming {input_csv} -> {output_csv}")
    stats = transform_csv(input_csv, output_csv)

    print("\n=== Transformation Statistics ===")
    print(f"Total input rows: {stats['total_rows']}")
    print(f"Relation rows: {stats['relation_rows']}")
    print(f"Skipped rows (malformed): {stats['skipped_rows']}")
    print(f"Parse errors: {stats['parse_errors']}")
    print(f"Output rows: {stats['output_rows']}")
    print(f"\n✅ Output saved to: {output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
