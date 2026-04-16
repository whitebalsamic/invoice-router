# Architecture

`invoice-router` is designed to make document routing explicit before extraction begins.

## Design goals

- prefer deterministic routing over opaque guesswork
- keep local execution reproducible
- preserve enough metadata to compare runs over time
- support both native-text PDFs and OCR-heavy image inputs

## High-level flow

```text
[Input invoice]
        |
        v
[Normalize pages]
        |
        v
[OCR and fingerprint pages]
        |
        v
[Build document context]
        |
        v
{Fingerprint match?} -- Yes --> [Reuse routed extraction] ---+
        |                                                    |
        No                                                   v
        v                                           [Normalize fields]
{Template family match?} -- Yes --> [Apply family profile] ---+
        |
        No
        v
[Heuristic discovery] ----------------------------------------+
                                                             |
                                                             v
                                               [Validate and reconcile]
                                                             |
                                                             v
                                       [Persist results and analysis history]
```

## Core differentiator

The key idea is that invoice layouts are not treated as unrelated OCR blobs. The runtime fingerprints page structure, groups related layouts into template families, and uses those signals to choose a stable extraction strategy before attempting field recovery.

That makes the system easier to reason about when you want:

- repeatable extraction behavior
- explainable regressions
- benchmarkable improvements over time

See the [glossary](glossary.md) for the project-specific terminology used by the CLI and reports.
