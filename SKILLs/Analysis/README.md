# Analysis/

Guides for **diagnosing galfitS outputs after a run** — what to check in the
`.gssummary` / `.params` / fit images before trusting a result or feeding it
into the next fitting stage. One file per topic; keep each file short and
checklist-style.

## When to use

After **every** fit, walk the checklist below before reporting results to the
user or writing the next stage's lyric. These checks are cheap; skipping them
propagates silent problems into later stages.

## Checklist index

| # | Check | File |
|---|-------|------|
| 1 | Boundary hits: did any free parameter land on its min/max? | `BOUNDARY.md` |
| 2 | Was `--readsummary` used (here or upstream)? Bounds were tightened | `BOUNDARY.md` |
| 3 | ES verbose fitness ≠ gssummary chisq? (~2× is by design) | `FITNESS_VS_CHISQ.md` |

## Notes

- The gssummary header records the run setup (`# command line:`,
  `# input options:` with `readsummary` / `readpar` / `priorpath` /
  non-default flags). Always read the header first — it tells you which
  checks apply. Older summaries (written before this feature) lack the
  block; treat their history as unknown.
- Companion docs: `../Basics/PARAM_GUIDE.md` (config keys),
  `../Basics/NAME_FORMAT.md` (parameter names in the summary),
  `../Basics/QA.md` (runtime errors).
