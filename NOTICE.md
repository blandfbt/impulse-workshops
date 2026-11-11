# Notices

## Dataset

**Synthetic Vehicle Fleet Telemetry** — https://huggingface.co/datasets/renumics/synthetic-vehicle-fleet-telemetry
(revision `017ec8f02e0753d8b092cf9229e1372d92a7b635`).

Published by [Renumics](https://renumics.com/) under the
[Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)
license. Based on joint work between Volkswagen AG Group Innovation, IPG Automotive and Renumics.

This repository does **not** contain any data from the dataset. The instructor setup downloads it into
the instructor's own Unity Catalog volume. Derived tables (bronze/silver/gold) created by these notebooks
are adaptations of the dataset and remain subject to CC BY-SA 4.0.

## Impulse

[databrickslabs/impulse](https://github.com/databrickslabs/impulse) is a Databricks Labs project, provided
for exploration only, without formal support or SLAs. It is not vendored here: the notebooks import it from
a Databricks Git folder clone created by the instructor. The Genie Code skills used in workshop 1 are part
of that repository.

## Approach

The bronze → silver mapping follows the Renumics ingest used by Impulse's own performance benchmark
(`tests/e2e/benchmark/ingest` in the Impulse repository).
