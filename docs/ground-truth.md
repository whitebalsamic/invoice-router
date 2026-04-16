# Ground Truth and Data Prep

`invoice-router` can process invoice files without ground truth (GT), but GT sidecars make the
results much more useful for validation, benchmarking, and discovery-oriented routing analysis.

This guide explains how to lay out your dataset, how GT files are paired with invoices, and which
JSON format to use when preparing your own data.

## Folder layout

Each invoice file may have an optional GT JSON sidecar with the same filename stem.

```text
my-dataset/
  invoice-001.pdf
  invoice-001.json
  invoice-002.jpg
  invoice-002.json
  invoice-003.pdf
```

The runtime pairs files by stem:

- `invoice-001.pdf` <-> `invoice-001.json`
- `invoice-002.jpg` <-> `invoice-002.json`

Supported invoice inputs include PDF and common image formats such as PNG and JPEG.

## First-time workflow

For a first pass, a practical workflow is:

1. Put your invoices in one dataset directory.
2. Run `invoice-router process /path/to/dataset` to confirm the files are readable.
3. Add GT sidecars incrementally for the invoices you want to validate or benchmark.
4. Move toward canonical `gt-v2` JSON as your dataset matures.

If a GT file is missing, processing still runs. You simply do not get GT-backed validation or
benchmarking for that invoice.

## Canonical GT format

For new datasets, author GT in the canonical `gt-v2` format.

The authoritative schema is:

- [ground-truth-v2.schema.json](../schemas/ground-truth-v2.schema.json)

At a high level, a `gt-v2` file contains:

- `schemaVersion`: should start with `gt-v2`
- `document`: document-level text fields such as invoice number and seller name
- `summary`: summary numeric fields such as subtotal and total amount
- `lineItems`: array of annotated line items

Each annotated field includes:

- `status`: one of `present`, `absent`, `unclear`, or `derived`
- `value`: optional canonical value
- `raw`: optional source value before normalization
- `evidence`: optional human-readable note or source snippet
- `confidence`: optional score between `0` and `1`

### Minimal `gt-v2` example

```json
{
  "schemaVersion": "gt-v2",
  "document": {
    "invoiceNumber": {
      "status": "present",
      "value": "INV-100"
    },
    "invoiceDate": {
      "status": "present",
      "value": "2026-04-01"
    },
    "sellerName": {
      "status": "present",
      "value": "Acme Ltd"
    }
  },
  "summary": {
    "totalAmount": {
      "status": "present",
      "value": 42.5
    }
  },
  "lineItems": [
    {
      "index": 1,
      "description": {
        "status": "present",
        "value": "Widget"
      },
      "amount": {
        "status": "present",
        "value": 42.5
      }
    }
  ]
}
```

## Discovery-ready minimum

The pipeline treats GT as discovery-ready when it contains:

- `totalAmount`
- and either `invoiceNumber` or `invoiceDate`

That is the minimum useful GT for discovery-oriented evaluation. Richer fields still help with
validation, benchmarking, and line-item analysis.

## Simple JSON compatibility

If you are getting started quickly, a simpler flat JSON sidecar can also be used. The loader can
normalize legacy-style JSON when it contains recognized invoice fields or sections.

### Minimal simple JSON example

```json
{
  "invoiceNumber": "INV-100",
  "invoiceDate": "2026-04-01",
  "sellerName": "Acme Ltd",
  "totalAmount": "42.50",
  "lineItems": [
    {
      "description": "Widget",
      "amount": "42.50"
    }
  ]
}
```

This compatibility path is useful for basic onboarding, but it has important limitations:

- It is not the recommended authoring format for new datasets.
- It cannot express field status such as `absent`, `unclear`, or `derived`.
- It cannot carry structured evidence, confidence, or raw-value metadata in the canonical way.
- Only recognized field names and aliases are normalized; unrelated keys may be ignored.
- If the JSON does not contain recognized invoice fields or sections, it may be ignored rather than
  treated as valid GT.
- Richer benchmarking, annotation, and review workflows should use `gt-v2`.

In short: simple JSON is a useful compatibility bridge, but `gt-v2` is the format you should
standardize on.

## GT sync for copied or subset datasets

If your working dataset is a copy or subset of a larger source-of-truth dataset, you can audit or
refresh sidecar files with:

```bash
invoice-router sync-ground-truth /path/to/dataset --check
invoice-router sync-ground-truth /path/to/dataset
```

Use `--check` to see whether local GT files are in sync. Run without `--check` to copy or update
GT sidecars from the source-of-truth directory.

This is mainly useful when:

- you maintain a small working subset of a larger dataset
- you copy invoice files into a benchmark dataset
- GT is curated in one place and consumed in another

It is not required for a first-time local run with a self-contained dataset.
