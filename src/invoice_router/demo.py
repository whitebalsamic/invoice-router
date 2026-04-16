from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class DemoInvoiceSpec:
    invoice_number: str = "INV-2026-001"
    invoice_date: str = "2026-04-16"
    provider_name: str = "Acme Veterinary Clinic"
    customer_name: str = "Jamie Rivera"
    pet_name: str = "Mochi"


DEFAULT_DEMO_SPEC = DemoInvoiceSpec()
DEFAULT_DEMO_DATASET = "demo-native-pdf"


def write_demo_invoice_pdf(pdf_path: str | Path, spec: DemoInvoiceSpec = DEFAULT_DEMO_SPEC) -> Path:
    pdf_path = Path(pdf_path).expanduser()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    page = doc.new_page()
    lines = [
        spec.provider_name,
        "Invoice",
        f"Invoice Number: {spec.invoice_number}",
        f"Date: {spec.invoice_date}",
        f"Bill To: {spec.customer_name}",
        f"Patient: {spec.pet_name}",
        "Office Visit Consult 1 75.00",
        "Lab Test 1 25.00",
        "Total: $100.00",
    ]
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=12)
        y += 18
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def create_demo_dataset(
    workspace: str | Path, spec: DemoInvoiceSpec = DEFAULT_DEMO_SPEC
) -> tuple[Path, Path]:
    workspace = Path(workspace).expanduser()
    dataset_dir = workspace / DEFAULT_DEMO_DATASET
    invoice_path = write_demo_invoice_pdf(dataset_dir / "demo_invoice.pdf", spec=spec)
    return dataset_dir, invoice_path
