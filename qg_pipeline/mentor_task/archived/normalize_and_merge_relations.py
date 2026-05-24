#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalize and merge relations from mentor_50papers_triples.csv

Steps:
1. Load original triples
2. Normalize predicates to English standard forms
3. Merge similar relations
4. Output normalized CSV
"""

import csv
from collections import Counter
from pathlib import Path


# Comprehensive mapping of all variants to standard English relations
PREDICATE_MAPPING = {
    # === Effect Modification (Positive) ===
    "increases": ["increases", "increase", "increased", "elevates", "elevated",
                  "raises", "raised", "upregulates", "upregulated", "upregulation",
                  "增加", "promotes", "促进", "上调", "enhances", "enhanced"],

    "improves": ["improves", "improved", "improvement", "better", "ameliorates",
                 "改善", "改善"],

    "enhances": ["enhances", "enhanced", "enhancement", "boosts", "boosted",
                 "potentiates", "potentiation", "augments", "synergizes", "增强"],

    # === Effect Modification (Negative) ===
    "reduces": ["reduces", "reduced", "reduction", "decreases", "decreased",
                "decrease", "lowers", "lowered", "downregulates", "downregulated",
                "downregulation", "减少", "降低", "reduce"],

    "inhibits": ["inhibits", "inhibited", "inhibition", "suppresses", "suppressed",
                 "suppression", "抑制", "抑制生长"],

    "blocks": ["blocks", "blocked", "blockade", "prevents", "prevented"],

    # === Association ===
    "associated_with": ["associated_with", "association", "related_to", "linked_to",
                        "connection", "相关因素", "关联", "相关", "具有", "correlates_with",
                        "correlated_with", "positively_correlated_with", "causally_associated_with"],

    # === Causation ===
    "causes": ["causes", "caused", "causality", "etiology", "导致", "引起", "cause"],

    "leads_to": ["leads_to", "lead_to", "resulted_in", "results_in", "consequence"],

    "induces": ["induces", "induced", "induction", "triggers", "triggered",
                "elicits", "诱导", "激活", "引发"],

    "mediated_by": ["mediated_by", "mediation", "mechanism", "through", "via"],

    "regulates": ["regulates", "regulated", "regulation", "modulates", "modulated",
                  "modulation", "controls", "controlled", "调控", "调节生长", "影响"],

    # === Treatment ===
    "treats": ["treats", "treated", "treatment", "therapy", "therapeutic",
               "manages", "management", "alleviates", "relieves", "治疗"],

    "targets": ["targets", "targeted", "targeting", "binds", "binding",
                "靶向", "结合", "interact_with"],

    "indicated_for": ["indicated_for", "indication", "approved_for", "approved",
                      "usage", "用于", "effective_for"],

    "contraindicated_with": ["contraindicated_with", "contraindication", "contraindicates"],

    "adverse_event_of": ["adverse_event_of", "adverse_event", "side_effect",
                         "side_effect_of", "toxicity", "toxic"],

    # === Risk & Prognostic ===
    "risk_factor_for": ["risk_factor_for", "risk", "risky", "增加风险",
                        "increases_risk_of", "increased_risk",
                        "independent_risk_factor_for"],

    "protective_factor_for": ["protective_factor_for", "protective", "protection",
                              "protects", "preventive", "protective_against",
                              "protective_factor_against"],

    "prognostic_marker_for": ["prognostic_marker_for", "prognostic", "prognosis",
                              "predictor", "predicts", "predictive"],

    # === Pharmacokinetic ===
    "has_pharmacokinetic_property": ["has_pharmacokinetic_property", "pharmacokinetic",
                                      "pk", "bioavailability", "half_life", "clearance",
                                      "plasma_stability", "plasmatic_protein_binding"],

    "administered_via": ["administered_via", "administration", "route", "delivery"],

    "has_dose_response": ["has_dose_response", "dose_response", "dose_dependent",
                          "ed50", "ld50", "therapeutic_window", "剂量"],

    # === Diagnostic & Measurement ===
    "diagnosed_by": ["diagnosed_by", "diagnosis", "diagnostic", "detected_by",
                     "identified_by", "identifies", "识别"],

    "has_measurement": ["has_measurement", "measurement", "measured", "level", "value",
                        "concentration", "incidence", "IC50", "平均粒径", "Zeta电位",
                        "包封率_DOX", "包封率_CAM", "logP", "radiochemical_yield",
                        "mean_change_in_SGRQ", "mean_change_in_6MWD", "usability_score",
                        "5-year survival rate", "median_PFS", "clinical_benefit_rate",
                        "粒径", "准确率"],

    "has_clinical_feature": ["has_clinical_feature", "clinical_feature", "symptom", "sign",
                             "manifestation", "presentation", "has_symptom",
                             "常见症状", "显著差异"],

    # === Other relations (needs manual review) ===
    "located_in": ["located_in", "location", "位置"],

    "composed_of": ["composed_of", "contains", "includes"],

    "derived_from": ["derived_from", "source", "来源"],

    "characterized_by": ["characterized_by", "features", "特点", "表现出"],

    "affects": ["affects", "impacts", "influences", "影响", "主要影响"],

    "requires": ["requires", "需要"],

    "enables": ["enables", "allows", "允许"],

    "is_a": ["is_a", "is", "type_of", "是", "作为"],

    "has_outcome": ["has_outcome", "results_in", "outcome"],

    "involved_in": ["involved_in", "participates_in", "参与", "涉及"],

    "responsible_for": ["responsible_for", "负责"],

    "determines": ["determines", "决定"],

    "expressed_as": ["expressed_as", "表达"],

    "changes": ["changes", "改变"],

    "provides": ["provides", "提供"],

    " achieves": ["achieves", "实现"],

    "superior_to": ["superior_to", "better_than", "优于", "has_advantage_over"],

    "effective_regardless_of": ["effective_regardless_of"],

    "used_as": ["used_as", "applied_to", "应用"],

    "safe_for": ["safe_for", "safe_option_for", "safe_alternative_for"],

    "prevalence_of": ["prevalence_of", "更高患病率", "更常见"],

    "disproportionately_affect": ["disproportionately_affect"],

    "exacerbated_by": ["exacerbated_by"],

    "stimulates": ["stimulates"],

    "useful_for": ["useful_for", "has_utility_in", "beneficial_for", "have_potential_in"],

    "calculation_formula": ["calculation_formula"],

    "have_lower_expression": ["have_lower_expression"],

    "have_higher_expression": ["have_higher_expression", "高表达", "表达上调"],

    "encodes": ["encodes"],

    "not_correlated_with": ["not_correlated_with"],

    "no_significant_difference": ["no_significant_difference", "no_total_causal_effect_on"],

    "stable_in": ["stable_in", "稳定性"],

    "unstable_in": ["unstable_in"],

    "uptake_by": ["uptake_by", "tumor_uptake", "tumor_uptake_in"],

    "formation": ["formation"],

    "chelator_for": ["chelator_for"],

    "influenced_by": ["influenced_by"],

    "extends": ["extends", "prolongs"],

    "more_sensitive_to": ["more_sensitive_to", "are_sensitive_to"],

    "more_susceptible_to": ["more_susceptible_to"],

    "enriched_in": ["enriched_in"],

    "reverses": ["reverses"],

    "produces": ["produces"],

    "processes": ["processes"],

    "synthesizes_and_secretes": ["synthesizes_and_secretes"],

    "mixes_and_secretes": ["mixes_and_secretes"],

    "exhibits": ["exhibits"],

    "accelerates": ["accelerates"],

    "addresses_problem": ["addresses_problem"],

    "shifts_differentiation_towards": ["shifts_differentiation_towards"],

    "modulates": ["modulates"],

    "compared_to": ["compared_to"],

    "lower_prevalence_of": ["lower_prevalence_of"],

    "longer_median_time_to": ["longer_median_time_to"],

    "mediated_negative_effect_on": ["mediated_negative_effect_on"],

    "driven_by": ["driven_by"],

    "excludes": ["excludes"],

    "does_not_induce": ["does_not_induce", "不消除"],

    "symptom_onset": ["symptom_onset"],

    "management_includes": ["management_includes"],

    "involves_mechanism": ["involves_mechanism"],

    "has_pathological_feature": ["has_pathological_feature"],

    "mechanism_of_action": ["mechanism_of_action", "mechanism_of"],

    "is_potential_therapy_for": ["is_potential_therapy_for", "有前景", "可能减轻", "潜在克服"],

    "predictor_of": ["predictor_of"],

    "best_model_for": ["best_model_for"],

    "disrupts": ["disrupts"],

    "是_靶点": ["targets", "是_靶点"],

    "表达_变化_因_癌症类型": ["varies_by_cancer_type"],

    "is_mnemonic_for": ["is_mnemonic_for"],

    "治疗不依从性比例": ["treatment_non_compliance_rate"],

    "challenge": ["challenge"],

    "对碳青霉烯耐药率": ["carbapenem_resistance_rate"],

    "对碳青霉烯耐药率高": ["high_carbapenem_resistance_rate"],

    "中位住院时间": ["median_hospital_stay"],

    "中位机械通气时间": ["median_ventilation_duration"],

    "30天再入院率": ["readmission_rate_30days"],

    "住院率": ["hospitalization_rate"],

    "恢复率": ["recovery_rate"],

    "low_oral_bioavailability": ["has_property", "low_oral_bioavailability"],

    "first_line_drug_for": ["indicated_for", "first_line_drug_for"],

    "降低水平": ["reduces_level", "降低水平"],

    "相互作用": ["interacts_with", "相互作用"],

    "削弱效应": ["weakens", "削弱效应"],

    "对...至关重要": ["essential_for", "对...至关重要"],

    "是主要病原体": ["is_major_pathogen", "是主要病原体"],

    "最常见原因": ["most_common_cause", "最常见原因"],

    "caused_by": ["caused_by"],
}


def build_reverse_mapping() -> dict:
    """Build reverse mapping from variant to standard."""
    reverse_map = {}
    for standard, variants in PREDICATE_MAPPING.items():
        for variant in variants:
            reverse_map[variant.lower().strip()] = standard

    # Handle edge case: empty predicate
    reverse_map[""] = "unknown"
    reverse_map[" "] = "unknown"

    return reverse_map


def normalize_predicate(predicate: str, reverse_map: dict) -> str:
    """Normalize a predicate to standard form."""
    if not predicate:
        return "unknown"

    pred_lower = predicate.lower().strip()

    # Look up in mapping
    if pred_lower in reverse_map:
        return reverse_map[pred_lower]

    # Fuzzy match: check if it's contained in any variant
    for variant, standard in reverse_map.items():
        if pred_lower in variant or variant in pred_lower:
            return standard

    # Not found, return original (will be reviewed)
    return predicate


def load_triples(csv_path: Path) -> list[dict]:
    """Load triples from CSV."""
    triples = []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            triples.append(row)
    return triples


def normalize_triples(triples: list[dict], reverse_map: dict) -> tuple[list[dict], dict]:
    """Normalize all triples and return statistics."""

    normalized_triples = []
    stats = {
        "total": len(triples),
        "normalized": 0,
        "unchanged": 0,
        "unknown": 0,
        "predicate_distribution": Counter()
    }

    for triple in triples:
        original_predicate = triple["predicate"]
        normalized_predicate = normalize_predicate(original_predicate, reverse_map)

        # Track statistics
        if normalized_predicate == "unknown":
            stats["unknown"] += 1
        elif normalized_predicate.lower() == original_predicate.lower():
            stats["unchanged"] += 1
        else:
            stats["normalized"] += 1

        stats["predicate_distribution"][normalized_predicate] += 1

        # Create normalized triple
        normalized_triple = {
            "subject": triple["subject"],
            "predicate": normalized_predicate,
            "object": triple["object"],
            "source_section": triple["source_section"],
            "source_chunk": triple["source_chunk"],
            "context_text": triple["context_text"],
            "_original_predicate": original_predicate  # Keep for reference
        }

        normalized_triples.append(normalized_triple)

    return normalized_triples, stats


def write_normalized_csv(triples: list[dict], output_path: Path):
    """Write normalized triples to CSV."""

    fieldnames = ["subject", "predicate", "object", "source_section",
                  "source_chunk", "context_text"]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for triple in triples:
            # Write without the _original_predicate field
            row = {k: triple[k] for k in fieldnames}
            writer.writerow(row)


def write_unknown_mapping(triples: list[dict], output_path: Path):
    """Write unknown predicates to a separate file for review."""

    unknown_predicates = set()
    for triple in triples:
        if triple["predicate"] == "unknown":
            original = triple["_original_predicate"]
            if original:
                unknown_predicates.add(original)

    with output_path.open("w", encoding="utf-8-sig") as f:
        f.write("# Unknown Predicates (need manual mapping)\n")
        f.write("# Count: {}\n\n".format(len(unknown_predicates)))
        for pred in sorted(unknown_predicates):
            f.write(f"{pred}\n")


def print_statistics(stats: dict):
    """Print normalization statistics."""

    print("=" * 80)
    print("NORMALIZATION STATISTICS")
    print("=" * 80)
    print(f"\nTotal triples: {stats['total']}")
    print(f"Normalized: {stats['normalized']} ({stats['normalized']/stats['total']*100:.1f}%)")
    print(f"Unchanged: {stats['unchanged']} ({stats['unchanged']/stats['total']*100:.1f}%)")
    print(f"Unknown: {stats['unknown']} ({stats['unknown']/stats['total']*100:.1f}%)")

    print("\n" + "=" * 80)
    print("TOP 30 NORMALIZED PREDICATES:")
    print("=" * 80)
    for pred, count in stats["predicate_distribution"].most_common(30):
        print(f"{count:3d} | :{pred}")

    print("\n" + "=" * 80)


def main():
    input_csv = Path("mentor_task/mentor_50papers_triples.csv")
    output_csv = Path("mentor_task/mentor_50papers_normalized.csv")
    unknown_file = Path("mentor_task/unknown_predicates.txt")

    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        return 1

    # Build reverse mapping
    print("Building predicate mapping...")
    reverse_map = build_reverse_mapping()
    print(f"Loaded {len(reverse_map)} variant mappings\n")

    # Load triples
    print("Loading triples...")
    triples = load_triples(input_csv)
    print(f"Loaded {len(triples)} triples\n")

    # Normalize
    print("Normalizing predicates...")
    normalized_triples, stats = normalize_triples(triples, reverse_map)

    # Write output
    print("Writing output...")
    write_normalized_csv(normalized_triples, output_csv)
    write_unknown_mapping(normalized_triples, unknown_file)

    # Print statistics
    print_statistics(stats)

    print(f"\n✅ Normalized triples saved to: {output_csv}")
    print(f"📝 Unknown predicates saved to: {unknown_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
