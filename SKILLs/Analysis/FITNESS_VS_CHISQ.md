# ES fitness vs. gssummary `chisq`

> Verified against `src/galfits/gsfit.py` and `src/galfits/galfitS.py`
> (2025-10-29). Rule: **both numbers are correct — they are different
> quantities.** fitness ≈ chisq/2, never exactly chisq/2.

## Definitions

| Quantity | Definition | Where it appears |
|---|---|---|
| ES fitness | `0.5 * sum(r^2)` **+ prior penalties** at the best solution | verbose progress bar ("Best Fitness"), and `# best fitness:` in the gssummary |
| `# chisq:` | `sum(r^2)` at the best solution — no 0.5 factor, **no prior terms** | gssummary header (`gsfit.py: update_parameter`) |

Prior penalties that enter the fitness (but never `# chisq`):

- Gaussian priors from the `.prior` file's `GP)` list — only when
  `--baysian` is passed.
- Cosmic-SFH prior (`SFH_Tx)` in the `.prior` file → `_ftp_penalty`) —
  applied by `evolution_strategies*` whenever the prior is loaded,
  **independent of `--baysian`**.

## Why the ratio is not exactly 2

1. Prior penalties above (only move fitness up).
2. The progress bar prints the **current generation's** best fitness
   (`gsfit.py`: `current_fitness = min of this generation`), while the
   reported `# best fitness:` is the **historical best**
   (`state.best_solution`). CMA-ES is not elitist, so the last printed
   value can be worse than the final reported one.

Quick sanity check after an ES run:

```
0.5 * chisq  <=  best fitness        (equality iff no priors were active)
```

If `best fitness < 0.5 * chisq`, something is wrong (e.g. the summary's
chisq was evaluated at different parameters — report it).

## Fixed inconsistency (2025-10-29)

`evolution_strategies_theta` (the CLI `ES` path) always included the
cosmic-SFH penalty, but `evolution_strategies` (logit-space) and
`evolution_strategies_multi` (`--multistart`) silently dropped it in both
bayesian and non-bayesian branches — so a run with `--multistart` and an
`SFH_T` prior optimized a different objective than a plain `ES` run. This
was fixed by adding `_ftp_penalty` to all four branches; all ES variants
now share the theta-variant objective.
