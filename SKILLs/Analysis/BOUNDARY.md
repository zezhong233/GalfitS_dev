# Boundary checks on a galfitS result

> Verified against `src/galfits/galfitS.py` (summary writing + `--readsummary`
> handling) and the LRD_RUBINS-CEERS-49140 example runs.

## Check 1 — did any free parameter hit its bound?

A free parameter whose `best_value` sits on (or within ~1 `step` of) its lyric
`min`/`max` means the fit wanted to go further. Two common physical readings:

- **Bounds too narrow** for an honest parameter (e.g. an elongated disk
  pinned at `axrat = 0.1`).
- **A component soaking up background/flux it should not own.** Classic
  signature: `Re` growing to the fitting-area half-size (`Ix8)`), especially
  in `images - photometry` mode where per-band `logNorm_{p}_{band}` are free
  and a large low-surface-brightness component costs nothing.

How to check: for every `pname` under `# free parameters:` in the
`.gssummary`, compare `best_value` with the 5-list `[value, min, max, step,
vary]` of the matching key in the lyric (`../Basics/PARAM_GUIDE.md` for
key → parameter mapping). Flag if `|best - min| ≲ step` or `|best - max| ≲ step`.

Observed example (`obj49140_image-phot` run, 2025-10-29):

| parameter | best | bound hit | reading |
|---|---|---|---|
| `disk_axrat` | 0.1000 | min = 0.1 | disk wants to be flatter than allowed |
| `obj1_Re` | 1.0200 | max = 1.02 = `Ia8)` half-size | companion likely soaking background |

Actions (pick deliberately, never silently):
1. Widen the bound and rerun.
2. Fix the parameter at a justified value (state why).
3. Reconsider the model — extra component, better mask, sky level.

## Check 2 — `--readsummary` propagates and *shrinks* boundaries

Mechanism (`galfitS.py`, `readsummary` block): each listed parameter is set to
`value = best_value` with

```
newmin = max(lyric_min, best - 10*brute_step)
newmax = min(lyric_max, best + 10*brute_step)
```

Consequences:

- **A boundary hit gets baked in.** Example: `axrat` best = 0.1 at min,
  `step = 0.02` → new bounds `[0.1, 0.3]`; the parameter starts exactly on
  the corner it already wanted to cross.
- **The stage-1 lyric's `step` is the stage-2 constraint half-width**
  (bounds = best ± 10·step). Do not leave `step` at arbitrary values in any
  lyric whose summary will feed a later stage.
- `--readsummary` never *widens* beyond the new lyric's `[min, max]` — so the
  stage-2 lyric is the place to repair bounds.

Before writing a stage-N lyric that reads a stage-(N−1) summary:

1. Run Check 1 on that summary.
2. For every hit, either fix the parameter (with justification) or set wider
   `[min, max]` in the new lyric — do not rely on `--readsummary` alone.
3. Check the summary header: `# input options:` → `readsummary: ...` tells
   you this summary is itself a re-seed of an earlier run, so its bounds were
   already tightened once; inspect the earlier summary too if in doubt.
   (Summaries written before the header feature lack this block — treat
   their provenance as unknown.)
