from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .enums import DocumentFamily, PageRole, TemplateStatus


class PageFingerprint(BaseModel):
    page_index: int
    visual_hash: int
    visual_hash_hex: str
    role: Optional[PageRole] = None
    stable_anchor_signature: Dict[str, Any] = Field(default_factory=dict)


class DocumentFingerprint(BaseModel):
    fingerprint_hash: str
    page_fingerprints: List[PageFingerprint]
    layout_template: Dict[str, Any]
    page_count: int


class FingerprintRecord(BaseModel):
    hash: str
    layout_template: Dict[str, Any]
    template_family_id: Optional[str] = None
    provider_name: Optional[str] = None
    country_code: Optional[str] = None
    page_fingerprints: List[PageFingerprint] = Field(default_factory=list)
    confidence: float = 0.0
    apply_count: int = 0
    reject_count: int = 0
    gt_apply_count: int = 0
    gt_reject_count: int = 0
    gt_confidence: float = 0.0
    status: TemplateStatus = TemplateStatus.provisional
    version: str = "v3"
    created_at: str
    last_used: Optional[str] = None


class TemplateFamilyRecord(BaseModel):
    template_family_id: str
    provider_name: Optional[str] = None
    country_code: Optional[str] = None
    document_family: DocumentFamily = DocumentFamily.unknown
    stable_anchor_regions: Dict[str, Any] = Field(default_factory=dict)
    anchor_summary: Dict[str, Any] = Field(default_factory=dict)
    page_role_expectations: List[PageRole] = Field(default_factory=list)
    summary_area_anchors: Dict[str, Any] = Field(default_factory=dict)
    variable_region_masks: List[Dict[str, Any]] = Field(default_factory=list)
    extraction_profile: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    apply_count: int = 0
    reject_count: int = 0
    gt_apply_count: int = 0
    gt_reject_count: int = 0
    gt_confidence: float = 0.0
    status: TemplateStatus = TemplateStatus.provisional
    created_at: str
    updated_at: Optional[str] = None


class TemplateFamilyExample(BaseModel):
    id: Optional[int] = None
    template_family_id: str
    fingerprint_hash: Optional[str] = None
    invoice_path: Optional[str] = None
    example_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str


class TemplateFamilyVersion(BaseModel):
    id: Optional[int] = None
    template_family_id: str
    version: int
    family_snapshot: Dict[str, Any]
    change_reason: Optional[str] = None
    created_at: str
