"""
Pydantic models for Multi-Paper Knowledge Graph Construction Framework.
"""
from enum import Enum
from typing import Optional, List
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