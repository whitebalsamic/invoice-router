import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_key_value_pairs(lines: List[str]) -> Dict[str, str]:
    extracted: Dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = _clean_text(key)
        value = _clean_text(value)
        if key and value and key not in extracted:
            extracted[key] = value
    return extracted


def _extract_invoice_like_fields(lines: List[str]) -> Dict[str, str]:
    extracted: Dict[str, str] = {}
    patterns = {
        "invoice_number": re.compile(
            r"\b(invoice|receipt)\s*(no|number|#)\b[:\s-]*(.+)", re.IGNORECASE
        ),
        "invoice_date": re.compile(r"\b(date|invoice date)\b[:\s-]*(.+)", re.IGNORECASE),
        "total": re.compile(
            r"\b(total|amount due|balance due)\b[:\s-]*([$A-Z0-9., -]+)", re.IGNORECASE
        ),
        "subtotal": re.compile(r"\b(subtotal)\b[:\s-]*([$A-Z0-9., -]+)", re.IGNORECASE),
        "tax": re.compile(r"\b(tax|gst|hst|pst|vat)\b[:\s-]*([$A-Z0-9., -]+)", re.IGNORECASE),
    }

    for line in lines:
        for field_name, pattern in patterns.items():
            match = pattern.search(line)
            if match and field_name not in extracted:
                extracted[field_name] = _clean_text(match.group(match.lastindex or 0))
    return extracted


def _looks_like_money(token: str) -> bool:
    return bool(re.fullmatch(r"[$€£]?\d[\d,]*\.?\d{0,2}", token))


def _looks_like_qty(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", token))


def _extract_line_items(lines: List[str]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for line in lines:
        compact = _clean_text(line)
        if not compact or ":" in compact:
            continue

        tokens = compact.split()
        if len(tokens) < 3:
            continue

        tail_amounts: List[str] = []
        for token in reversed(tokens):
            if _looks_like_money(token) or _looks_like_qty(token):
                tail_amounts.append(token)
            else:
                break

        if len(tail_amounts) < 2:
            continue

        tail_amounts.reverse()
        description_tokens = tokens[: len(tokens) - len(tail_amounts)]
        if len(description_tokens) < 1:
            continue

        quantity = tail_amounts[0]
        amount = tail_amounts[-1]
        unit_price = tail_amounts[-2] if len(tail_amounts) >= 3 else None

        description = " ".join(description_tokens).strip()
        lower_description = description.lower()
        if any(
            marker in lower_description
            for marker in ("subtotal", "total", "tax", "balance due", "amount due")
        ):
            continue

        items.append(
            {
                "description": description,
                "quantity": quantity,
                **({"unit_price": unit_price} if unit_price is not None else {}),
                "amount": amount,
            }
        )

    return items


def load_pdf_page_texts(invoice_path: str, max_pages: int = 3) -> List[str]:
    """Read raw page text from the first `max_pages` pages of a PDF."""
    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyMuPDF is required for native PDF extraction") from exc

    path = Path(invoice_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Native PDF extraction expects a PDF, got: {invoice_path}")

    with fitz.open(invoice_path) as doc:
        return [doc[index].get_text("text") for index in range(min(len(doc), max_pages))]


def extract_from_text_pdf(
    invoice_path: str,
    max_pages: int = 3,
    *,
    page_texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract simple structured data from a text PDF without OCR."""
    page_texts = (
        list(page_texts)
        if page_texts is not None
        else load_pdf_page_texts(invoice_path, max_pages=max_pages)
    )
    lines: List[str] = []
    for text in page_texts:
        lines.extend([line.strip() for line in text.splitlines() if line.strip()])

    extracted = {
        "raw_text": "\n\n".join(page_texts).strip(),
        "page_count": len(page_texts),
    }
    extracted.update(_extract_key_value_pairs(lines))
    extracted.update(_extract_invoice_like_fields(lines))
    line_items = _extract_line_items(lines)
    if line_items:
        extracted["line_items"] = line_items
    return extracted
