# Production Workflow

`invoice-router` is meant to be run against private invoice datasets without forking the repo by default.

The lightweight production model is:

- keep this repo as the shared engine
- keep live invoices, GT, outputs, and runtime databases outside the repo
- keep production-specific config and family-profile patches in one private folder outside the repo
- only commit code here when the change is generic and reusable

## Recommended layout

```text
~/Desktop/invoice-router/                       # shared code repo
~/invoice-router-private/
  config.production.yaml
  family-profiles/
  run-production.sh
  notes.md
~/data/
  datasets/
  output/
  temp/
```

## Important config rule

`--config` points to one complete YAML file. It is not merged with the packaged defaults.

That means your private `config.production.yaml` should start from the example in
[`samples/config.production.example.yaml`](samples/config.production.example.yaml) and then be edited in place.

## First setup

1. Create a private folder outside the repo.
2. Copy [`samples/config.production.example.yaml`](samples/config.production.example.yaml) to `~/invoice-router-private/config.production.yaml`.
3. Fill in provider aliases and any production-specific thresholds you want to override.
4. Keep runtime environment values such as `DATABASE_URL`, `ANALYSIS_DATABASE_URL`, `DATASET_ROOT`, `OUTPUT_DIR`, and `TEMP_DIR` outside the repo too.
5. Run production through one small wrapper script so the command stays boring and repeatable.

Example `run-production.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

export DATABASE_URL="postgresql://..."
export ANALYSIS_DATABASE_URL="postgresql://..."
export DATASET_ROOT="$HOME/data/datasets"
export OUTPUT_DIR="$HOME/data/output"
export TEMP_DIR="$HOME/data/temp"

invoice-router \
  --config "$HOME/invoice-router-private/config.production.yaml" \
  process "$1"
```

## Day-to-day workflow

1. Run production with the private config.
2. Review output and note repeated failures or unstable families.
3. If a layout issue is specific to one family, add a small JSON patch under `~/invoice-router-private/family-profiles/`.
4. Apply that patch with `invoice-router family-update ... --profile-json ...`.
5. If the improvement is clearly general, implement and commit it in this repo instead of keeping it private.

Example family patch application:

```bash
invoice-router family-update FAMILY_123 \
  --profile-json "$HOME/invoice-router-private/family-profiles/FAMILY_123.json" \
  --reason production_tuning
```

## Safety rail

The private wrapper should fail fast if any sensitive path resolves inside the Git repo.

At minimum, treat these as private-only locations:

- the private folder itself
- `config.production.yaml`
- `family-profiles/`
- `DATASET_ROOT`
- `OUTPUT_DIR`
- `TEMP_DIR`
- any explicit dataset path passed on the command line

This is intentionally stricter than `.gitignore`. The goal is to reduce the chance that private data or private tuning ever gets near Git history, Docker build context, archive bundles, or accidental `git add .`.

## What belongs where

Shared repo:

- reusable routing, extraction, normalization, and validation improvements
- test coverage for behavior we want to support for everyone
- public documentation and examples

Private folder:

- `config.production.yaml`
- family-profile JSON patches
- wrapper scripts and operational notes
- private provider aliases or environment-specific tuning

Outside both repos:

- live invoices
- GT sidecars containing private data
- outputs, temp files, and runtime databases

## Decision rule

Use the simplest option that matches the problem:

- one odd invoice once: note it, do not code it yet
- repeated issue for one supplier family: private family-profile patch or config tweak
- repeated issue across several suppliers or datasets: promote it into the shared codebase

## When to add more machinery

Do not add a fork, plugin system, or second private code repo yet.

Add more structure only if one of these starts to hurt:

- family-profile changes need to be rebuilt often from scratch
- DB-held family tuning becomes hard to track or review
- you need truly private Python logic that cannot be expressed as config or profile patches

If that happens, the next step should be small: add export/import for family profiles before introducing a fuller extension mechanism.
