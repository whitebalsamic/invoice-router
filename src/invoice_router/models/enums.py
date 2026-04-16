from enum import Enum


class Route(str, Enum):
    APPLY = "APPLY"
    DISCOVERY = "DISCOVERY"
    REJECTED = "REJECTED"
    FAIL = "FAIL"


class SourceFormat(str, Enum):
    pdf_text = "pdf_text"
    pdf_scanned = "pdf_scanned"
    image = "image"
    unknown = "unknown"


class DocumentFamily(str, Enum):
    invoice = "invoice"
    estimate = "estimate"
    statement = "statement"
    attachment = "attachment"
    unknown = "unknown"


class ExtractionStrategy(str, Enum):
    provider_template = "provider_template"
    native_pdf = "native_pdf"
    ocr_structured = "ocr_structured"
    discovery_fallback = "discovery_fallback"
    fail = "fail"


class TemplateStatus(str, Enum):
    provisional = "provisional"
    established = "established"
    degraded = "degraded"
    retired = "retired"


class PageRole(str, Enum):
    header_page = "header_page"
    line_item_page = "line_item_page"
    summary_page = "summary_page"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
