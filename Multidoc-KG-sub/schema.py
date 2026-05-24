"""
Pydantic models for Multi-Paper Knowledge Graph Construction Framework.
"""
from enum import Enum
import re
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    """Knowledge claim processing status."""
    EXTRACTED = "extracted"
    GROUNDED = "grounded"
    VALIDATED = "validated"
    REJECTED = "rejected"


class Paper(BaseModel):
    """Academic paper model."""
    id: str = Field(..., description="Unique paper identifier")
    title: str = Field(..., description="Paper title")
    content: str = Field(..., description="Full paper content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def get_abstract(self) -> str:
        """Return the abstract text stored in metadata, if available."""
        abstract = self.metadata.get("abstract", "")
        return abstract.strip() if isinstance(abstract, str) else ""

    def get_keywords(self) -> List[str]:
        """Return normalized keyword strings stored in metadata."""
        keywords = self.metadata.get("keywords", [])

        if isinstance(keywords, str):
            keywords = [item.strip() for item in keywords.split(",") if item.strip()]
        elif isinstance(keywords, list):
            keywords = [str(item).strip() for item in keywords if str(item).strip()]
        else:
            keywords = []

        return keywords

    def _looks_like_identifier_title(self) -> bool:
        """Return True when the title is effectively just an ID/PMID placeholder."""
        title = self.title.strip()
        if not title:
            return True

        normalized = re.sub(r"[\s\-_:/]+", "", title)
        return bool(re.fullmatch(r"[A-Za-z]*\d{5,}[A-Za-z]*", normalized))

    def _build_content_summary(self, max_sections: int = 2, chars_per_section: int = 700) -> str:
        """Build a short fallback summary from the first substantive content sections."""
        content = (self.content or "").strip()
        if not content:
            return ""

        ignored_sections = {
            "references",
            "acknowledgements",
            "acknowledgments",
            "funding",
            "conflict of interest",
            "declarations",
            "author contributions",
            "supporting information",
            "supplementary information",
            "data availability",
        }

        parts: List[str] = []
        chunks = re.split(r"(?m)^##\s+(.+?)\n+", content)

        if len(chunks) >= 3:
            for idx in range(1, len(chunks), 2):
                heading = re.sub(r"\s+", " ", chunks[idx]).strip()
                body = re.sub(r"\s+", " ", chunks[idx + 1]).strip() if idx + 1 < len(chunks) else ""
                heading_lower = heading.lower()

                if not body or len(body) < 80:
                    continue
                if heading_lower in ignored_sections:
                    continue

                parts.append(f"{heading}: {body[:chars_per_section].strip()}")
                if len(parts) >= max_sections:
                    break

        if not parts:
            plain = re.sub(r"\s+", " ", content)
            return plain[: max_sections * chars_per_section].strip()

        return "\n".join(parts)

    def build_classification_text(self) -> str:
        """
        Build a compact literature summary for tasks like subdomain assignment.
        Uses title + abstract + keywords as the preferred signal and
        appends any existing subdomain assignment for downstream prompts.
        """
        parts = [f"Title: {self.title.strip()}"]

        abstract = self.get_abstract()
        if abstract:
            parts.append(f"Abstract: {abstract}")

        keywords = self.get_keywords()
        if keywords:
            parts.append(f"Keywords: {', '.join(keywords)}")

        if not abstract and not keywords:
            content_summary = self._build_content_summary()
            if content_summary:
                label = "Content summary"
                if self._looks_like_identifier_title():
                    label = "Content summary (fallback because title is identifier-like)"
                parts.append(f"{label}: {content_summary}")

        subdomain = str(self.metadata.get("subdomain", "")).strip()
        parent_domain = str(self.metadata.get("parent_domain", "")).strip()
        if subdomain:
            parts.append(f"Assigned subdomain: {subdomain}")
        if parent_domain:
            parts.append(f"Parent domain: {parent_domain}")

        return "\n".join(parts)


class SubdomainAssignment(BaseModel):
    """Assigned biomedical subdomain for a paper."""
    subdomain: str = Field(..., description="Reasonably specific biomedical subdomain")
    parent_domain: str = Field(..., description="Broader parent biomedical domain")
    reason: str = Field(default="", description="Short rationale for the assignment")
    confidence: float = Field(default=0.0, description="Model confidence between 0 and 1")
    status: str = Field(default="confirmed", description="confirmed or candidate")
    is_new_subdomain: bool = Field(default=False, description="Whether this assignment proposes a new subdomain")
    batch_id: str = Field(default="", description="Batch identifier for the assignment run")
    taxonomy_version: int = Field(default=1, description="Taxonomy version used for this assignment")
    new_relations: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Optional ontology relations such as subdomain subclass_of parent_domain",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize compatibly across Pydantic versions."""
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


class KnowledgeClaim(BaseModel):
    """Knowledge claim extracted from papers."""
    subject: str = Field(..., description="Subject entity of the claim")
    relation: str = Field(..., description="Relation/predicate between subject and object")
    object: str = Field(..., description="Object entity of the claim")
    evidence: str = Field(..., description="Text evidence supporting this claim")
    source_paper_id: str = Field(..., description="ID of the source paper")
    status: ClaimStatus = Field(default=ClaimStatus.EXTRACTED, description="Current processing status")
    grounded_ids: Optional[List[str]] = Field(default=None, description="IDs of grounded entities in the graph")
    subject_id: Optional[str] = Field(default=None, description="Grounded ID for subject entity")
    object_id: Optional[str] = Field(default=None, description="Grounded ID for object entity")
    validation_type: Optional[str] = Field(default=None, description="Validation type: support/conflict/new")
    claim_type: Optional[str] = Field(default=None, description="Type of claim: ontology or instance")
    confidence: float = Field(default=1.0, description="Confidence score of the claim")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    def dict(self):
        """Convert to dictionary."""
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "evidence": self.evidence,
            "source_paper_id": self.source_paper_id,
            "status": self.status.value,
            "grounded_ids": self.grounded_ids,
            "subject_id": self.subject_id,
            "object_id": self.object_id,
            "validation_type": self.validation_type,
            "claim_type": self.claim_type,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
