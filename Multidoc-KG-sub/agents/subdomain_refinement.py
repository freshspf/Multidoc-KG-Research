"""
Batch refinement agent for candidate biomedical subdomains.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.llm_client import LLMClient


class SubdomainHierarchyRefinementAgent:
    """Refine candidate subdomains into a confirmed hierarchy."""

    def __init__(self, llm_client: LLMClient, config_path: Optional[str] = None):
        self.llm_client = llm_client
        self.config = self._load_config(config_path)
        self.root_domain = self._normalize_label(self.config.get("root_domain", "Biomedicine"))
        self.max_candidates = int(self.config.get("refinement_max_candidates", 50))
        self.max_confirmed = int(self.config.get("refinement_max_confirmed", 200))
        self.biomedical_anchor_terms = [
            "cancer", "carcinoma", "tumor", "oncology", "lymph", "hepat",
            "breast", "germ cell", "rna sequencing", "genomic", "biomarker",
            "diagnostic", "therapy", "therapeutic", "immunology", "pathology",
            "disease", "syndrome", "infection", "clinical", "medicine",
        ]
        self.specialty_anchor_terms = [
            "cancer", "carcinoma", "tumor", "oncology", "breast", "hepat",
            "germ cell", "lymph", "disease", "syndrome", "infection",
            "therapy", "therapeutic", "diagnostic", "pathology", "medicine",
        ]
        self.force_promote_terms = [
            "carcinoma", "tumor", "tumors", "oncology", "lymph", "hepat",
            "breast cancer", "germ cell", "sarcoma", "lymphoma", "leukemia",
            "melanoma", "disease", "syndrome", "infection",
        ]
        self.generic_discouraged_terms = [
            "research", "study", "analysis", "assessment", "management",
            "practice", "workflow", "platform", "approach",
        ]
        self.force_merge_phrases = [
            "shared decision making",
            "decision making",
            "rna sequencing analysis",
            "drug delivery systems",
        ]

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        default_config: Dict[str, Any] = {
            "root_domain": "Biomedicine",
            "refinement_max_candidates": 50,
            "refinement_max_confirmed": 200,
        }

        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = yaml.safe_load(f) or {}
            default_config.update(loaded_config)
        return default_config

    def _normalize_label(self, value: str) -> str:
        value = (value or "").replace("_", " ").strip()
        value = re.sub(r"\s+", " ", value)
        return value.strip(" .;:,").lower()

    def _build_prompt(
        self,
        confirmed_subdomains: List[Dict[str, Any]],
        candidate_subdomains: List[Dict[str, Any]],
    ) -> str:
        confirmed_lines = []
        for item in confirmed_subdomains[: self.max_confirmed]:
            name = str(item.get("name", "")).strip()
            parent = str(item.get("parent_name", "") or self.root_domain).strip()
            if name:
                confirmed_lines.append(f"- {name} -> {parent}")

        candidate_lines = []
        for item in candidate_subdomains[: self.max_candidates]:
            name = str(item.get("name", "")).strip()
            suggested_parent = str(item.get("suggested_parent", "") or self.root_domain).strip()
            paper_count = item.get("paper_count", 0)
            avg_confidence = item.get("avg_confidence", 0)
            samples = item.get("sample_papers", [])
            candidate_lines.append(
                f"- {name} | suggested_parent={suggested_parent} | paper_count={paper_count} "
                f"| avg_confidence={avg_confidence} | sample_papers={samples}"
            )

        return f"""You are an ontology engineering expert refining a biomedical taxonomy.

Your task is to review candidate subdomains and decide whether each one should:
1. merge into an existing confirmed subdomain, or
2. be promoted as a new confirmed subdomain under the most appropriate confirmed parent.

Rules:
- Prefer merging if a candidate is semantically equivalent to an existing confirmed subdomain.
- Prefer merging when a candidate looks like a broad research topic, method phrase, workflow, or healthcare practice label rather than a stable biomedical subdomain.
- Promote only when the candidate represents a meaningful, reusable biomedical subdomain with clear medical semantics.
- Prefer promoting explicit disease names, tumor types, carcinoma subtypes, and clinically established biomedical specialties.
- Use the closest confirmed parent possible; use {self.root_domain} only if no more specific parent exists.
- Return one decision for every candidate.
- `action` must be either `merge` or `promote`.
- For `merge`, `target_subdomain` must be an existing confirmed subdomain.
- For `promote`, `target_subdomain` should usually match the candidate name unless normalization is needed.
- Avoid promoting labels that are generic process terms such as "research", "analysis", "management", or "shared decision making" unless they are clearly anchored to a stable biomedical specialty.
- Keep reasons short.

[Confirmed Subdomains]
{chr(10).join(confirmed_lines) or f"- {self.root_domain} -> null"}

[Candidate Subdomains]
{chr(10).join(candidate_lines)}

Return strict JSON:
{{
  "decisions": [
    {{
      "candidate": "...",
      "action": "merge" or "promote",
      "target_subdomain": "...",
      "parent_domain": "...",
      "reason": "..."
    }}
  ]
}}
"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def _fallback_decisions(
        self,
        confirmed_subdomains: List[Dict[str, Any]],
        candidate_subdomains: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        confirmed_lookup = {
            self._normalize_label(str(item.get("name", ""))): str(item.get("name", "")).strip()
            for item in confirmed_subdomains
            if str(item.get("name", "")).strip()
        }

        decisions: List[Dict[str, str]] = []
        for candidate in candidate_subdomains:
            candidate_name = str(candidate.get("name", "")).strip()
            normalized = self._normalize_label(candidate_name)
            if normalized in confirmed_lookup:
                decisions.append(
                    {
                        "candidate": candidate_name,
                        "action": "merge",
                        "target_subdomain": confirmed_lookup[normalized],
                        "parent_domain": str(candidate.get("suggested_parent", "") or self.root_domain),
                        "reason": "Fallback merge due to exact normalized match with confirmed subdomain.",
                    }
                )
            else:
                decisions.append(
                    {
                        "candidate": candidate_name,
                        "action": "promote",
                        "target_subdomain": candidate_name,
                        "parent_domain": str(candidate.get("suggested_parent", "") or self.root_domain),
                        "reason": "Fallback promotion because no matching confirmed subdomain exists.",
                    }
                )
        return decisions

    def _looks_biomedical_enough(self, label: str) -> bool:
        normalized = self._normalize_label(label)
        if not normalized:
            return False
        return any(term in normalized for term in self.biomedical_anchor_terms)

    def _has_strong_specialty_anchor(self, label: str) -> bool:
        normalized = self._normalize_label(label)
        if not normalized:
            return False
        return any(term in normalized for term in self.specialty_anchor_terms)

    def _looks_too_generic(self, label: str) -> bool:
        normalized = self._normalize_label(label)
        if not normalized:
            return True
        return any(term in normalized for term in self.generic_discouraged_terms)

    def _should_force_promote(self, label: str) -> bool:
        normalized = self._normalize_label(label)
        if not normalized:
            return False
        return any(term in normalized for term in self.force_promote_terms)

    def _should_force_merge(self, label: str) -> bool:
        normalized = self._normalize_label(label)
        if not normalized:
            return True
        if any(phrase in normalized for phrase in self.force_merge_phrases):
            return True
        if self._looks_too_generic(normalized) and not self._has_strong_specialty_anchor(normalized):
            return True
        return False

    def _pick_merge_target(
        self,
        candidate: str,
        confirmed_subdomains: List[Dict[str, Any]],
    ) -> Optional[str]:
        candidate_norm = self._normalize_label(candidate)
        candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate_norm))
        best_match: Optional[str] = None
        best_score = 0.0

        for item in confirmed_subdomains:
            name = str(item.get("name", "")).strip()
            name_norm = self._normalize_label(name)
            if not name_norm or name_norm == self.root_domain:
                continue

            if name_norm == candidate_norm:
                return name

            name_tokens = set(re.findall(r"[a-z0-9]+", name_norm))
            if not candidate_tokens or not name_tokens:
                continue

            overlap = len(candidate_tokens & name_tokens)
            union = len(candidate_tokens | name_tokens)
            score = overlap / union if union else 0.0
            if candidate_norm in name_norm or name_norm in candidate_norm:
                score = max(score, 0.8)

            if score > best_score:
                best_score = score
                best_match = name

        if best_match and best_score >= 0.34:
            return best_match
        return None

    def _post_filter_decisions(
        self,
        decisions: List[Dict[str, str]],
        confirmed_subdomains: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        filtered: List[Dict[str, str]] = []
        for decision in decisions:
            candidate = str(decision.get("candidate", "")).strip()
            action = str(decision.get("action", "")).strip().lower()
            target = str(decision.get("target_subdomain", "")).strip()
            parent = str(decision.get("parent_domain", "")).strip() or self.root_domain
            reason = str(decision.get("reason", "")).strip()

            if not candidate or action not in {"merge", "promote"} or not target:
                continue

            if action == "promote":
                if self._should_force_promote(candidate):
                    filtered.append(
                        {
                            "candidate": candidate,
                            "action": "promote",
                            "target_subdomain": target,
                            "parent_domain": parent,
                            "reason": reason or "Post-filter preserved explicit disease or specialty label.",
                        }
                    )
                    continue

                if self._should_force_merge(candidate):
                    merge_target = self._pick_merge_target(candidate, confirmed_subdomains) or self.root_domain
                    filtered.append(
                        {
                            "candidate": candidate,
                            "action": "merge",
                            "target_subdomain": merge_target,
                            "parent_domain": parent,
                            "reason": "Post-filter forced merge for generic or non-specialty label.",
                        }
                    )
                    continue

                if self._looks_too_generic(candidate) and not self._looks_biomedical_enough(candidate):
                    merge_target = self._pick_merge_target(candidate, confirmed_subdomains) or self.root_domain
                    filtered.append(
                        {
                            "candidate": candidate,
                            "action": "merge",
                            "target_subdomain": merge_target,
                            "parent_domain": parent,
                            "reason": "Post-filter downgraded generic non-specific label to merge.",
                        }
                    )
                    continue

                if self._looks_too_generic(candidate) and self._looks_biomedical_enough(candidate):
                    merge_target = self._pick_merge_target(candidate, confirmed_subdomains)
                    if merge_target:
                        filtered.append(
                            {
                                "candidate": candidate,
                                "action": "merge",
                                "target_subdomain": merge_target,
                                "parent_domain": parent,
                                "reason": "Post-filter preferred merge for generic biomedical phrase.",
                            }
                    )
                    continue

            if action == "merge" and self._should_force_promote(candidate):
                filtered.append(
                    {
                        "candidate": candidate,
                        "action": "promote",
                        "target_subdomain": candidate,
                        "parent_domain": parent or self.root_domain,
                        "reason": "Post-filter upgraded explicit disease or specialty label to promote.",
                    }
                )
                continue

            filtered.append(
                {
                    "candidate": candidate,
                    "action": action,
                    "target_subdomain": target,
                    "parent_domain": parent,
                    "reason": reason,
                }
            )

        return filtered

    def process(
        self,
        confirmed_subdomains: List[Dict[str, Any]],
        candidate_subdomains: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        if not candidate_subdomains:
            return []

        prompt = self._build_prompt(confirmed_subdomains, candidate_subdomains)
        system_prompt = "You refine biomedical taxonomy candidates into confirmed hierarchy decisions."

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                json_mode=True,
                max_tokens=2200,
            )
            payload = self._parse_response(response)
            decisions_raw = payload.get("decisions", [])
            decisions: List[Dict[str, str]] = []
            for item in decisions_raw:
                if not isinstance(item, dict):
                    continue
                candidate = str(item.get("candidate", "")).strip()
                action = self._normalize_label(str(item.get("action", "")))
                target_subdomain = str(item.get("target_subdomain", "")).strip()
                parent_domain = str(item.get("parent_domain", "") or self.root_domain).strip()
                reason = str(item.get("reason", "")).strip()
                if candidate and action in {"merge", "promote"} and target_subdomain:
                    decisions.append(
                        {
                            "candidate": candidate,
                            "action": action,
                            "target_subdomain": target_subdomain,
                            "parent_domain": parent_domain,
                            "reason": reason,
                        }
                    )
            decisions = self._post_filter_decisions(decisions, confirmed_subdomains)
            if not decisions:
                raise ValueError("No valid refinement decisions returned")
            return decisions
        except Exception as e:
            print(f"[SubdomainHierarchyRefinementAgent] Refinement failed, using fallback: {e}")
            fallback = self._fallback_decisions(confirmed_subdomains, candidate_subdomains)
            return self._post_filter_decisions(fallback, confirmed_subdomains)
