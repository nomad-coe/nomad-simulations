# Copilot Instructions: maintaining the runschema → nomad-simulations mapping

## Overview

`docs/reference/runschema_mapping.md` maps every attribute of the legacy **runschema**
(`Run` / `System` / `Method` / `Calculation`) onto this package's modern **nomad-simulations**
schema (`Simulation` → `Program` / `ModelSystem` / `ModelMethod` / `Outputs`). It lets the legacy
parser coverage reports (which target runschema) be compared against the new simulation parsers
(which target nomad-simulations), and its `Unmapped` rows are a running coverage-gap audit of this
schema.

Keep it in sync as the schema evolves: a new attribute that fills a missing concept flips an
`Unmapped`/`Partial` row to `Mapped`; a renamed or restructured section changes a target path. The
mapping is documentation that tracks the schema — not a one-off artifact.

## Format

The format is defined by the document itself — its intro explains the direction and the
`Mapped` / `Partial` / `Unmapped` status semantics, and each `##` section carries the four-column
table (`runschema path | Status | nomad-simulations target | Notes`) and a coverage summary. Do not
restate the format here; read the doc. When editing, preserve its conventions: keep the top-of-file
`generated-by` comment (refresh the date), recompute the edited section's `**Summary:**` counts and
`mapped%`, and update the overall counts in the intro.

## Maintenance recipe

1. Identify the affected runschema area and read its source. runschema is an installed dependency;
   locate it with `python -c "import runschema, os; print(os.path.dirname(runschema.__file__))"`
   (`run.py`, `system.py`, `method.py`, `calculation.py`). Enumerate every `Quantity` / `SubSection`
   as a dotted path rooted at the lowercase section name (e.g. `method.k_mesh.grid`,
   `calculation.energy.total.value`), including `x_*` code-specific quantities.
2. For each legacy path, find the modern equivalent by reading
   `src/nomad_simulations/schema_packages/` (`general.py`, `model_system.py`, `model_method.py`,
   `numerical_settings.py`, `basis_set.py`, `atoms_state.py`, `outputs.py`, `properties/*.py`,
   `force_field.py`). Set the status and target, with a Notes reason for anything not 1:1.
3. Recompute the section summary and the overall intro counts.

## Anti-hallucination (review-based, no validator)

Correctness is a review responsibility — self-check every edited row against both sources:
1. Every left path must resolve to a real `Quantity` / `SubSection` in the installed `runschema`.
2. Every cited target must resolve to a real attribute in `src/nomad_simulations/schema_packages/`
   — an assigned `Quantity` / `SubSection`, an inherited attribute (some live on nomad-core
   `BaseSection`, e.g. `datetime`, `lab_id`), or a `@property`. Confirm with `grep`.
3. Do not invent equivalents to inflate coverage; an honest `Unmapped` with a reason is the point.
