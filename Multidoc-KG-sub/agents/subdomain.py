"""
Biomedical literature subdomain classifier with hierarchy-aware prompting.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from core.llm_client import LLMClient
from schema import Paper, SubdomainAssignment


class SubdomainClassifierAgent:
    """Assign a biomedical subdomain to each paper using the current hierarchy."""

    def __init__(
        self,
        llm_client: LLMClient,
        config_path: Optional[str] = None,
        hierarchy_provider: Optional[Any] = None,
    ):
        self.llm_client = llm_client
        self.hierarchy_provider = hierarchy_provider
        self.config = self._load_config(config_path)
        self.root_domain = self._normalize_label(self.config.get("root_domain", "Biomedicine"))
        self.max_hierarchy_edges = int(self.config.get("max_hierarchy_edges", 200))
        self.allow_new_subdomain = bool(self.config.get("allow_new_subdomain", True))
        self.reuse_similarity_threshold = float(self.config.get("reuse_similarity_threshold", 0.72))
        self.keyword_rules = self.config.get("keyword_rules", {})
        print(
            f"[SubdomainClassifierAgent] 初始化完成，root_domain={self.root_domain}，"
            f"allow_new_subdomain={self.allow_new_subdomain}"
        )

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        default_config: Dict[str, Any] = {
            "max_input_chars": 6000,
            "root_domain": "Biomedicine",
            "max_hierarchy_edges": 200,
            "allow_new_subdomain": True,
            "reuse_similarity_threshold": 0.72,
            "keyword_rules": {},
        }

        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = yaml.safe_load(f) or {}
            default_config.update(loaded_config)

        return default_config

    def _normalize_label(self, value: str) -> str:
        value = (value or "").replace("_", " ").strip()
        value = re.sub(r"\s+", " ", value)
        value = value.strip(" .;:,")
        return value.lower()

    def _tokenize_label(self, value: str) -> List[str]:
        normalized = self._normalize_label(value)
        return re.findall(r"[a-z0-9]+", normalized)

    def _get_existing_hierarchy(self) -> List[Dict[str, Any]]:
        if not self.hierarchy_provider or not hasattr(self.hierarchy_provider, "get_subdomain_hierarchy"):
            return []

        try:
            hierarchy = self.hierarchy_provider.get_subdomain_hierarchy()
            if not isinstance(hierarchy, list):
                return []
            return hierarchy[: self.max_hierarchy_edges]
        except Exception as e:
            print(f"[SubdomainClassifierAgent] 读取现有子领域层级失败: {e}")
            return []

    def get_hierarchy_snapshot(self) -> List[Dict[str, Any]]:
        """Return a hierarchy snapshot to freeze classification within one batch."""
        return self._get_existing_hierarchy()

    def _extract_known_labels(self, hierarchy: Sequence[Dict[str, Any]]) -> List[str]:
        known: List[str] = [self.root_domain]
        for item in hierarchy:
            child = self._normalize_label(str(item.get("child", "") or item.get("subject", "")))
            parent = self._normalize_label(str(item.get("parent", "") or item.get("object", "")))
            if child:
                known.append(child)
            if parent:
                known.append(parent)

        deduped: List[str] = []
        seen = set()
        for name in known:
            if name and name not in seen:
                deduped.append(name)
                seen.add(name)
        return deduped

    def _build_parent_lookup(self, hierarchy: Sequence[Dict[str, Any]]) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for item in hierarchy:
            child = self._normalize_label(str(item.get("child", "") or item.get("subject", "")))
            parent = self._normalize_label(str(item.get("parent", "") or item.get("object", "")))
            if child and parent and child not in lookup:
                lookup[child] = parent
        return lookup

    def _label_similarity(self, candidate: str, existing: str) -> float:
        candidate_norm = self._normalize_label(candidate)
        existing_norm = self._normalize_label(existing)
        if not candidate_norm or not existing_norm:
            return 0.0
        if candidate_norm == existing_norm:
            return 1.0
        if candidate_norm in existing_norm or existing_norm in candidate_norm:
            shorter = min(len(candidate_norm), len(existing_norm))
            longer = max(len(candidate_norm), len(existing_norm))
            if longer > 0:
                return shorter / longer

        candidate_tokens = set(self._tokenize_label(candidate_norm))
        existing_tokens = set(self._tokenize_label(existing_norm))
        if not candidate_tokens or not existing_tokens:
            return 0.0

        overlap = len(candidate_tokens & existing_tokens)
        union = len(candidate_tokens | existing_tokens)
        jaccard = overlap / union if union else 0.0
        containment = overlap / min(len(candidate_tokens), len(existing_tokens))
        return max(jaccard, containment)

    def _resolve_existing_subdomain(
        self,
        candidate: str,
        known_subdomains: Sequence[str],
    ) -> Optional[str]:
        candidate_norm = self._normalize_label(candidate)
        best_match: Optional[str] = None
        best_score = 0.0

        for existing in known_subdomains:
            existing_norm = self._normalize_label(existing)
            if not existing_norm or existing_norm == self.root_domain:
                continue
            score = self._label_similarity(candidate_norm, existing_norm)
            if score > best_score:
                best_score = score
                best_match = existing_norm

        if best_match and best_score >= self.reuse_similarity_threshold:
            return best_match
        return None

    def _resolve_existing_parent(
        self,
        subdomain: str,
        requested_parent: str,
        known_subdomains: Sequence[str],
    ) -> str:
        parent_norm = self._normalize_label(requested_parent)
        subdomain_norm = self._normalize_label(subdomain)

        if parent_norm and parent_norm != self.root_domain:
            reused_parent = self._resolve_existing_subdomain(parent_norm, known_subdomains)
            if reused_parent and reused_parent != subdomain_norm:
                return reused_parent

        best_parent = self.root_domain
        best_score = 0.0

        for existing in known_subdomains:
            existing_norm = self._normalize_label(existing)
            if not existing_norm or existing_norm in {self.root_domain, subdomain_norm}:
                continue

            score = self._label_similarity(subdomain_norm, existing_norm)
            if score <= 0.0:
                continue

            if subdomain_norm.startswith(existing_norm) or existing_norm in subdomain_norm:
                score = max(score, 0.78)

            if score > best_score:
                best_score = score
                best_parent = existing_norm

        if best_score >= self.reuse_similarity_threshold:
            return best_parent

        return self.root_domain if not parent_norm else parent_norm

    def _format_hierarchy_for_prompt(self, hierarchy: Sequence[Dict[str, Any]]) -> str:
        if not hierarchy:
            return f"{self.root_domain}, rdfs:subClassOf, null, null, system"

        lines: List[str] = []
        for item in hierarchy[: self.max_hierarchy_edges]:
            child = str(item.get("child", "") or item.get("subject", "")).strip()
            parent = str(item.get("parent", "") or item.get("object", "")).strip()
            source = str(item.get("source", "") or item.get("evidence", "") or "graph").strip() or "graph"
            if child and parent:
                lines.append(f"{child}, rdfs:subClassOf, {parent}, null, {source}")

        if not lines:
            return f"{self.root_domain}, rdfs:subClassOf, null, null, system"
        return "\n".join(lines)

    def _build_prompt(self, paper: Paper, hierarchy: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
        classification_text = paper.build_classification_text()
        max_input_chars = int(self.config.get("max_input_chars", 6000))
        if len(classification_text) > max_input_chars:
            classification_text = classification_text[:max_input_chars].rstrip() + "..."

        hierarchy_text = self._format_hierarchy_for_prompt(hierarchy)
        allow_new = "Yes" if self.allow_new_subdomain else "No"

        system_prompt = f"""You are an experienced ontology engineering expert.

The given text is in the domain of [{self.root_domain}].
Your task is to:
1. Identify the biomedical subdomain that most accurately reflects the text's content, and provide a concise and commonly used name for that subdomain.
2. Check whether this subdomain already exists in the given hierarchy.
3. If it already exists, keep new_relations empty.
4. If it does not exist, determine its correct position by identifying its most appropriate parent concept from the existing hierarchy and any relevant child concepts from the existing hierarchy, then output all new subclass_of relationships needed to integrate this subdomain.

Requirements:
- Prefer title, abstract, and keywords as the primary signal.
- Use the hierarchy as the first reference, not a fixed candidate list.
- The subdomain should be neither too broad nor too paper-specific.
- If an existing hierarchy concept is semantically equivalent or very close, reuse the existing concept name instead of creating a new one.
- When proposing a new subdomain, choose the closest existing parent concept available in the hierarchy. Use the root domain only when no more specific parent is appropriate.
- If allow_new_subdomain is No, you must choose an existing hierarchy concept.
- parent_domain should be the closest broader concept for the chosen subdomain.
- Use relation label exactly `subclass_of`.
- Keep reason short and evidence-based.

Return strict JSON with:
{{
  "subdomain": "...",
  "parent_domain": "...",
  "reason": "...",
  "confidence": 0.0,
  "new_relations": [
    {{"subject": "...", "relation": "subclass_of", "object": "..."}}
  ]
}}"""

        user_prompt = f"""[Top Domain] {self.root_domain}
[Allow New Subdomain] {allow_new}
[Given Hierarchy]
{hierarchy_text}

[Given Text]
{classification_text}
"""
        return system_prompt, user_prompt

    def _coerce_relations(
        self,
        subdomain: str,
        parent_domain: str,
        raw_relations: Any,
        known_subdomains: Sequence[str],
    ) -> List[Dict[str, str]]:
        relations: List[Dict[str, str]] = []
        known = {self._normalize_label(item) for item in known_subdomains if item}

        if isinstance(raw_relations, list):
            for item in raw_relations:
                if not isinstance(item, dict):
                    continue
                subject = self._normalize_label(str(item.get("subject", "")))
                relation = self._normalize_label(str(item.get("relation", "")))
                obj = self._normalize_label(str(item.get("object", "")))
                if subject and relation == "subclass_of" and obj and subject != obj:
                    candidate = {"subject": subject, "relation": relation, "object": obj}
                    if candidate not in relations:
                        relations.append(candidate)

        if subdomain in known:
            return relations

        if subdomain and parent_domain and subdomain != parent_domain:
            default_relation = {
                "subject": subdomain,
                "relation": "subclass_of",
                "object": parent_domain,
            }
            if default_relation not in relations:
                relations.insert(0, default_relation)

        return relations

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def _keyword_fallback(
        self,
        paper: Paper,
        known_subdomains: Sequence[str],
        parent_lookup: Dict[str, str],
        taxonomy_version: int,
        batch_id: str,
    ) -> SubdomainAssignment:
        text = self._normalize_label(paper.build_classification_text())
        known = {self._normalize_label(item) for item in known_subdomains if item}

        best_subdomain = "general clinical medicine"
        best_parent = self.root_domain
        best_score = 0

        for parent_domain, rules in self.keyword_rules.items():
            if not isinstance(rules, dict):
                continue
            for subdomain, keywords in rules.items():
                if not isinstance(keywords, list):
                    continue
                score = sum(1 for keyword in keywords if self._normalize_label(str(keyword)) in text)
                if score > best_score:
                    best_score = score
                    best_subdomain = self._normalize_label(subdomain)
                    best_parent = self._normalize_label(parent_domain)

        if best_score == 0:
            if "cancer" in text or "tumor" in text or "carcinoma" in text:
                best_subdomain = "cancer biology and therapy"
                best_parent = "oncology"
            elif "pregnan" in text or "maternal" in text or "fetal" in text:
                best_subdomain = "maternal-fetal medicine"
                best_parent = "obstetrics and gynecology"
            elif "infection" in text or "virus" in text or "bacterial" in text:
                best_subdomain = "clinical infectious diseases"
                best_parent = "infectious diseases"

        if not self.allow_new_subdomain and best_subdomain not in known:
            best_subdomain = best_parent if best_parent in known else self.root_domain

        reused_subdomain = self._resolve_existing_subdomain(best_subdomain, known_subdomains)
        if reused_subdomain:
            best_subdomain = reused_subdomain
            best_parent = parent_lookup.get(reused_subdomain, self.root_domain)
        else:
            best_parent = self._resolve_existing_parent(
                best_subdomain,
                best_parent,
                known_subdomains,
            )

        return SubdomainAssignment(
            subdomain=best_subdomain,
            parent_domain=best_parent,
            reason="Fallback assignment from paper summary and lightweight keyword heuristics.",
            confidence=0.4 if best_score > 0 else 0.25,
            status="confirmed" if best_subdomain in known else "candidate",
            is_new_subdomain=best_subdomain not in known,
            batch_id=batch_id,
            taxonomy_version=taxonomy_version,
            new_relations=self._coerce_relations(best_subdomain, best_parent, [], known_subdomains),
        )

    def process(
        self,
        paper: Paper,
        hierarchy_override: Optional[Sequence[Dict[str, Any]]] = None,
        taxonomy_version: int = 1,
        batch_id: str = "",
    ) -> SubdomainAssignment:
        print(f"[SubdomainClassifierAgent] 分类论文: {paper.id} - {paper.title[:80]}")
        hierarchy = list(hierarchy_override) if hierarchy_override is not None else self._get_existing_hierarchy()
        known_subdomains = self._extract_known_labels(hierarchy)
        parent_lookup = self._build_parent_lookup(hierarchy)
        system_prompt, user_prompt = self._build_prompt(paper, hierarchy)

        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
                max_tokens=1200,
            )
            payload = self._parse_response(response)

            subdomain = self._normalize_label(str(payload.get("subdomain", "")))
            parent_domain = self._normalize_label(str(payload.get("parent_domain", "")))
            reason = str(payload.get("reason", "")).strip()

            confidence_raw = payload.get("confidence", 0.0)
            try:
                confidence = max(0.0, min(1.0, float(confidence_raw)))
            except (TypeError, ValueError):
                confidence = 0.0

            if not subdomain:
                raise ValueError("Missing subdomain in classifier response")

            if not parent_domain:
                parent_domain = self.root_domain

            reused_subdomain = self._resolve_existing_subdomain(subdomain, known_subdomains)
            if reused_subdomain:
                subdomain = reused_subdomain
                parent_domain = parent_lookup.get(reused_subdomain, parent_domain or self.root_domain)
            else:
                parent_domain = self._resolve_existing_parent(
                    subdomain,
                    parent_domain,
                    known_subdomains,
                )

            if not self.allow_new_subdomain and subdomain not in set(known_subdomains):
                raise ValueError(f"New subdomain not allowed: {subdomain}")

            assignment = SubdomainAssignment(
                subdomain=subdomain,
                parent_domain=parent_domain,
                reason=reason,
                confidence=confidence,
                status="confirmed" if subdomain in set(known_subdomains) else "candidate",
                is_new_subdomain=subdomain not in set(known_subdomains),
                batch_id=batch_id,
                taxonomy_version=taxonomy_version,
                new_relations=self._coerce_relations(
                    subdomain,
                    parent_domain,
                    payload.get("new_relations", []),
                    known_subdomains,
                ),
            )
            print(
                f"[SubdomainClassifierAgent] 分类完成: {paper.id} -> "
                f"{assignment.subdomain} ({assignment.parent_domain})"
            )
            return assignment
        except Exception as e:
            print(f"[SubdomainClassifierAgent] 分类失败，回退轻量规则: {e}")
            assignment = self._keyword_fallback(
                paper,
                known_subdomains,
                parent_lookup,
                taxonomy_version,
                batch_id,
            )
            print(
                f"[SubdomainClassifierAgent] 回退完成: {paper.id} -> "
                f"{assignment.subdomain} ({assignment.parent_domain})"
            )
            return assignment
