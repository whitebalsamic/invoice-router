from pathlib import Path
from typing import List, Optional, Tuple

from ...models import DocumentContext, DocumentFamily, ExtractionStrategy, SourceFormat
from .country_rules import infer_country_and_currency


def _detect_source_format(invoice_path: str) -> SourceFormat:
    suffix = Path(invoice_path).suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz

            with fitz.open(invoice_path) as doc:
                text = "\n".join(doc[index].get_text("text") for index in range(min(len(doc), 3)))
            return SourceFormat.pdf_text if text.strip() else SourceFormat.pdf_scanned
        except Exception:
            return SourceFormat.pdf_scanned
    if suffix in {".png", ".jpg", ".jpeg"}:
        return SourceFormat.image
    return SourceFormat.unknown


def _infer_document_family(tokens: List[str]) -> DocumentFamily:
    joined = " ".join(tokens).lower()
    if "estimate" in joined or "treatment plan" in joined:
        return DocumentFamily.estimate
    if "statement" in joined or "account summary" in joined:
        return DocumentFamily.statement
    if "invoice" in joined or "receipt" in joined or "bill to" in joined:
        return DocumentFamily.invoice
    return DocumentFamily.unknown


def build_document_context(
    invoice_path: str,
    ocr_results_per_page: List[List[Tuple[str, int, int, int, int]]],
    *,
    source_format: Optional[SourceFormat] = None,
):
    tokens = [text for page in ocr_results_per_page for text, *_ in page if text.strip()]
    resolved_source_format = source_format or _detect_source_format(invoice_path)
    document_family = _infer_document_family(tokens)
    country_code, currency_code = infer_country_and_currency(tokens)

    strategy = ExtractionStrategy.ocr_structured
    if resolved_source_format == SourceFormat.pdf_text:
        strategy = ExtractionStrategy.native_pdf

    return DocumentContext(
        source_format=resolved_source_format,
        document_family=document_family,
        country_code=country_code,
        currency_code=currency_code,
        extraction_strategy=strategy,
    )
