#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate final analysis report from normalized triples.
"""

import csv
from collections import Counter
from pathlib import Path


def load_normalized_triples(csv_path: Path) -> list[dict]:
    """Load normalized triples."""
    triples = []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            triples.append(row)
    return triples


def analyze_predicates(triples: list[dict]) -> dict:
    """Analyze predicate distribution."""
    predicates = [t["predicate"] for t in triples]
    predicate_counts = Counter(predicates)

    # Group by category
    categories = {
        "Effect Modification": ["increases", "improves", "enhances", "reduces", "inhibits", "blocks"],
        "Association": ["associated_with", "correlated_with"],
        "Causation": ["causes", "leads_to", "induces", "mediated_by", "regulates"],
        "Treatment": ["treats", "targets", "indicated_for", "contraindicated_with", "adverse_event_of"],
        "Risk": ["risk_factor_for", "protective_factor_for", "prognostic_marker_for"],
        "Measurement": ["has_measurement", "has_clinical_feature", "diagnosed_by"],
        "Other": []
    }

    categorized = {}
    for cat, rels in categories.items():
        categorized[cat] = sum(predicate_counts[r] for r in rels if r in predicate_counts)

    # Add uncategorized to "Other"
    categorized["Other"] = len(triples) - sum(categorized.values())

    return {
        "total": len(triples),
        "unique_predicates": len(predicate_counts),
        "predicate_counts": predicate_counts,
        "categorized": categorized
    }


def print_report(stats: dict):
    """Print formatted report."""

    print("=" * 80)
    print("FINAL RELATION ANALYSIS REPORT")
    print("=" * 80)

    print(f"\nTotal Relations: {stats['total']}")
    print(f"Unique Predicates: {stats['unique_predicates']}")

    print("\n" + "=" * 80)
    print("RELATION DISTRIBUTION BY CATEGORY:")
    print("=" * 80)

    for cat, count in sorted(stats["categorized"].items(), key=lambda x: -x[1]):
        pct = count / stats["total"] * 100
        print(f"{cat:20s} | {count:4d} ({pct:5.1f}%)")

    print("\n" + "=" * 80)
    print("TOP 40 MOST COMMON PREDICATES:")
    print("=" * 80)

    for pred, count in stats["predicate_counts"].most_common(40):
        pct = count / stats["total"] * 100
        print(f"{count:3d} ({pct:4.1f}%) | :{pred}")

    print("\n" + "=" * 80)
    print("CHINESE PREDICATES (需要人工审核):")
    print("=" * 80)

    chinese_preds = [(p, c) for p, c in stats["predicate_counts"].items()
                     if any(ord(ch) > 127 for ch in p)]

    if chinese_preds:
        for pred, count in sorted(chinese_preds, key=lambda x: -x[1]):
            print(f"{count:3d} | {pred}")
    else:
        print("(None)")

    print("\n" + "=" * 80)


def main():
    csv_path = Path("mentor_task/mentor_50papers_normalized.csv")

    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return 1

    # Load and analyze
    triples = load_normalized_triples(csv_path)
    stats = analyze_predicates(triples)

    # Print report
    print_report(stats)

    print(f"\n✅ Analysis complete!")
    print(f"📊 Data source: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
