#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze relations from mentor_50papers_triples.csv
- Count frequency of each predicate
- Group similar relations
- Normalize Chinese to English
"""

import csv
from collections import defaultdict, Counter
from pathlib import Path


def load_triples(csv_path: Path) -> list[dict]:
    """Load triples from CSV."""
    triples = []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            triples.append({
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "source_section": row["source_section"],
                "source_chunk": row["source_chunk"],
                "context_text": row["context_text"]
            })
    return triples


def analyze_predicates(triples: list[dict]) -> dict:
    """Analyze predicate frequencies."""
    predicate_counts = Counter(t["predicate"] for t in triples)
    return predicate_counts


def group_similar_relations(predicate_counts: Counter) -> dict:
    """
    Group similar relations into standard categories.

    Returns: {
        "standard_relation": {
            "variants": {"variant1": count1, "variant2": count2, ...},
            "total": total_count,
            "category": "category_name"
        }
    }
    """
    groups = {
        # Positive Effects
        "increases": {
            "variants": {},
            "total": 0,
            "category": "Effect Modification"
        },
        "improves": {
            "variants": {},
            "total": 0,
            "category": "Effect Modification"
        },
        "enhances": {
            "variants": {},
            "total": 0,
            "category": "Effect Modification"
        },

        # Negative Effects
        "reduces": {
            "variants": {},
            "total": 0,
            "category": "Effect Modification"
        },
        "inhibits": {
            "variants": {},
            "total": 0,
            "category": "Effect Modification"
        },
        "blocks": {
            "variants": {},
            "total": 0,
            "category": "Effect Modification"
        },

        # Association
        "associated_with": {
            "variants": {},
            "total": 0,
            "category": "Association"
        },
        "correlated_with": {
            "variants": {},
            "total": 0,
            "category": "Association"
        },

        # Causation
        "causes": {
            "variants": {},
            "total": 0,
            "category": "Causation"
        },
        "leads_to": {
            "variants": {},
            "total": 0,
            "category": "Causation"
        },
        "induces": {
            "variants": {},
            "total": 0,
            "category": "Causation"
        },

        # Treatment
        "treats": {
            "variants": {},
            "total": 0,
            "category": "Treatment"
        },
        "targets": {
            "variants": {},
            "total": 0,
            "category": "Treatment"
        },
        "adverse_event_of": {
            "variants": {},
            "total": 0,
            "category": "Treatment"
        },

        # Risk
        "risk_factor_for": {
            "variants": {},
            "total": 0,
            "category": "Risk"
        },
        "protective_factor_for": {
            "variants": {},
            "total": 0,
            "category": "Risk"
        },

        # Measurement
        "has_measurement": {
            "variants": {},
            "total": 0,
            "category": "Measurement"
        },

        # Other
        "other": {
            "variants": {},
            "total": 0,
            "category": "Other"
        }
    }

    # Manual mapping for common variants
    mapping_rules = {
        # Increases variants
        "increases": ["increases", "increase", "increased", "elevates", "elevated", "upregulates", "upregulated"],

        # Improves variants
        "improves": ["improves", "improved", "improvement", "better", "ameliorates"],

        # Enhances variants
        "enhances": ["enhances", "enhanced", "enhancement", "boosts", "boosted", "potentiates"],

        # Reduces variants
        "reduces": ["reduces", "reduced", "reduction", "decreases", "decreased", "decrease", "lowers", "lowered", "downregulates", "downregulated"],

        # Inhibits variants
        "inhibits": ["inhibits", "inhibited", "inhibition", "suppresses", "suppressed", "suppression", "抑制", "抑制生长"],

        # Blocks variants
        "blocks": ["blocks", "blocked", "blockade", "prevents", "prevented"],

        # Associated_with variants
        "associated_with": ["associated_with", "association", "related_to", "linked_to", "connection", "相关", "关联", "相关因素", "具有"],

        # Correlated_with variants
        "correlated_with": ["correlated_with", "correlation", "correlates"],

        # Causes variants
        "causes": ["causes", "caused", "causality", "etiology", "导致", "引起"],

        # Leads_to variants
        "leads_to": ["leads_to", "lead_to", "resulted_in", "results_in", "consequence"],

        # Induces variants
        "induces": ["induces", "induced", "induction", "triggers", "triggered", "诱导", "激活"],

        # Treats variants
        "treats": ["treats", "treated", "treatment", "therapy", "therapeutic", "manages", "management", "治疗"],

        # Targets variants
        "targets": ["targets", "targeted", "targeting", "binds", "binding", "靶向", "结合"],

        # Adverse_event variants
        "adverse_event_of": ["adverse_event_of", "adverse_event", "side_effect", "side_effect_of", "toxicity", "toxic"],

        # Risk_factor variants
        "risk_factor_for": ["risk_factor_for", "risk", "risky", "增加风险", "increases_risk_of", "increased_risk"],

        # Protective_factor variants
        "protective_factor_for": ["protective_factor_for", "protective", "protection", "protects", "preventive"],

        # Has_measurement variants
        "has_measurement": ["has_measurement", "measurement", "measured", "level", "value", "concentration"],
    }

    # Group predicates
    for predicate, count in predicate_counts.items():
        pred_lower = predicate.lower().strip()

        # Find matching group
        matched = False
        for standard, variants in mapping_rules.items():
            if pred_lower in variants:
                groups[standard]["variants"][predicate] = count
                groups[standard]["total"] += count
                matched = True
                break

        if not matched:
            groups["other"]["variants"][predicate] = count
            groups["other"]["total"] += count

    return groups


def print_analysis(predicate_counts: Counter, groups: dict):
    """Print analysis results."""

    print("=" * 80)
    print("RELATION FREQUENCY ANALYSIS")
    print("=" * 80)
    print(f"\nTotal unique predicates: {len(predicate_counts)}")
    print(f"Total relations: {sum(predicate_counts.values())}\n")

    print("-" * 80)
    print("TOP 20 MOST COMMON PREDICATES:")
    print("-" * 80)
    for pred, count in predicate_counts.most_common(20):
        print(f"{count:3d} | {pred}")

    print("\n" + "=" * 80)
    print("GROUPED BY STANDARD RELATIONS:")
    print("=" * 80)

    # Sort by total count
    sorted_groups = sorted(groups.items(), key=lambda x: x[1]["total"], reverse=True)

    for standard, info in sorted_groups:
        if info["total"] == 0:
            continue

        print(f"\n【{info['category']}】 :{standard} (Total: {info['total']})")
        print("-" * 60)
        for variant, count in sorted(info["variants"].items(), key=lambda x: -x[1]):
            print(f"  {count:3d} | {variant}")

    print("\n" + "=" * 80)
    print("CHINESE PREDICATES (需要翻译):")
    print("=" * 80)

    chinese_predicates = {k: v for k, v in predicate_counts.items() if any(
        ord(c) > 127 for c in k  # Contains non-ASCII characters
    )}

    if chinese_predicates:
        for pred, count in sorted(chinese_predicates.items(), key=lambda x: -x[1]):
            print(f"{count:3d} | {pred}")
    else:
        print("(None found)")

    print("\n" + "=" * 80)


def main():
    csv_path = Path("mentor_task/mentor_50papers_triples.csv")

    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return 1

    # Load data
    print("Loading triples...")
    triples = load_triples(csv_path)
    print(f"Loaded {len(triples)} triples\n")

    # Analyze predicates
    predicate_counts = analyze_predicates(triples)

    # Group similar relations
    groups = group_similar_relations(predicate_counts)

    # Print results
    print_analysis(predicate_counts, groups)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
