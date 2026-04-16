from typing import List, Optional

from pydantic import BaseModel, Field

from .enums import DocumentFamily, ExtractionStrategy, SourceFormat


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class ProviderMatch(BaseModel):
    provider_name: str
    confidence: float
    matched_on: List[str] = Field(default_factory=list)
    country_code: Optional[str] = None


class DocumentContext(BaseModel):
    source_format: SourceFormat = SourceFormat.unknown
    document_family: DocumentFamily = DocumentFamily.unknown
    country_code: Optional[str] = None
    language_code: Optional[str] = None
    currency_code: Optional[str] = None
    provider_match: Optional[ProviderMatch] = None
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.fail
