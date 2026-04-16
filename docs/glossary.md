# Glossary

## Fingerprint

A structural signature for a document or page, used to recognize layout similarity across runs.

## Template family

A group of related fingerprints that represent one invoice layout family over time.

## Apply

A route where the runtime has enough confidence to reuse a known extraction path directly.

## Discovery

A route where the runtime needs to infer structure and field positions rather than reusing a known template decision.

## Healing

A retry pass that revisits prior failed results when newer trust signals indicate a known path can now be safely reused.

## Analysis run

A persisted summary of a processing or benchmark execution, used for later comparison and regression tracking.
