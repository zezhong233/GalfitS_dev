---
name: galfits
description: |
  Skill for GalfitS — a JAX-accelerated galaxy image + SED fitting package.
  Use this skill whenever the user asks about running galfitS, writing or
  debugging a galfitS config file, choosing a fitter class (sedfitter,
  imagefitter_phot, imagefitter_SED_allconst — all subclass `gsfitter`) or
  fit method (optimizer, evolution_strategies, evolution_strategies_theta,
  nested_sampling, jnesty_nested_sampling), calling any function in the
  `galfits` package, or decoding galfitS errors.
  Coverage: 15 modules in src/galfits; 4 fitter classes (1 base + 3
  concrete); 12 fit methods, 5 reachable via --fit_method. Note:
  `imagefitter_SED` and `flowmc` are ghosts — docstring/CLI-only, no code.
---

# GalfitS Skill

GalfitS fits parametric galaxy/AGN/star models simultaneously to
multi-band images and spectra, using JAX for GPU-accelerated
gradient-based fitting. This skill helps anyone who runs `galfitS.py`,
edits config files, or calls `galfits.*` functions from Python.

---

## When to invoke this skill

Trigger when the request involves **any** of:

- Installing GalfitS or its external dependencies (jax, optax, jaxopt,
  evosax, dynesty, jnesty, lmfit, astropy, photutils, reproject).
- Running the CLI (`galfitS.py <config> --workplace ... --fit_method ...`)
  or writing/editing a galfitS **config file**.
- Choosing or debugging a fitter: `sedfitter`, `imagefitter_phot`,
  `imagefitter_SED_allconst`.(Other fitter or method will be added soon.)
- Choosing or debugging a fit method: `optimizer`, `evolution_strategies`,
  `evolution_strategies_theta`, `nested_sampling` (dynesty),
  `jnesty_nested_sampling`.
- Calling, explaining, or debugging **any function** in `galfits.*`.
- Decoding an error message that mentions galfits modules
  (`gsfit.py`, `gsutils.py`, `images.py`, `galaxy.py`, ...).

Do **not** trigger for general Python/JAX/astropy questions that don't
touch this package.

---

## Knowledge base layout

```
skills/
├── SKILL.md            ← you are here: WHEN to use the skill, HOW to route
├── todo.md             ← what's left to fill in (single source of truth)
├── paths.toml          ← machine-specific paths (edit per machine)
├── scripts/
│   ├── scan_source.py  ← AST scanner that rebuilds reference/
│   └── refresh.sh      ← scan → diff → rebuild reference/
├── reference/          ← AUTO-GENERATED from source, do not hand-edit
│   ├── package_overview.md
│   ├── cli.md              (every galfitS.py CLI flag, from source)
│   └── functions/<module>.md
└── guides/             ← HAND-MAINTAINED teaching material
    ├── installation.md
    ├── gotchas.md          ← "this will bite you" list
    ├── parameter_tuning.md ← what every magic number means
    ├── task_templates.md   ← "I want to do X" → snippet
    └── troubleshooting.md  ← error keyword → cause → fix
```

**Rule of thumb:** `reference/` answers "what is X"; `guides/` answers
"how do I actually do Y / why does Z fail".

---

## How to route a user question

| User says… | First file to read |
|---|---|
| "How do I install GalfitS?" | `guides/installation.md` |
| "What does function `foo` do?" | `reference/functions/<module>.md` |
| "Which CLI flags exist / what does `--nlive` do?" | `reference/cli.md` then `guides/parameter_tuning.md` |
| "Which fitter / fit_method should I use?" | `reference/functions/gsfit.md` then `guides/task_templates.md` |
| "What value should parameter Y be?" | `guides/parameter_tuning.md` |
| "I got this error: …" | `guides/troubleshooting.md` then `guides/gotchas.md` |
| "How do I fit <task>?" | `guides/task_templates.md` |
| "Why does X behave like Y?" | `guides/gotchas.md` |
| "What config keys exist (R/I/M/S…)?" | `reference/functions/gsutils.md` (`parse_config_file`, `read_config_file`) |

---

## Hard rules (always follow)

1. **Default = snippet, not full script.** Show the 5–15 key lines (or the
   one CLI command). Only produce a full runnable file when the user
   explicitly asks for one.
2. **No hardcoded machine paths in examples.** Route through
   `paths.toml` / user config. Never copy `/home/zhongyi/...` into answers.
3. **English in artifacts.** Chat may mirror the user's language; files are
   English.
4. **Cite the source location** when explaining: `function`
   (`reference/functions/<module>.md`, line N).
5. **Flag GPU needs.** GalfitS fits run on JAX; GPU (CUDA jax) is the
   normal mode (`--device gpu` is the CLI default). Warn before any
   long-running fit command.
6. **Never invent** install steps, gotchas, or parameter advice. If it is
   not in `guides/`, say so and point at `todo.md`.
7. **Trust code over comments.** This repo is in flux and many docstrings
   are stale AI-generated text. Verify claims against the source, and say
   when a docstring looks wrong.

---

## Refreshing the reference after a source update

```bash
bash skills/scripts/refresh.sh    # scan → diff → rebuild reference/
```

`guides/` is never touched by the refresh. If the diff shows removed/renamed
functions, `grep -r "<old_name>" skills/guides/` and update the mentions.

---

## Out of scope

- General programming help not touching this package.
- Modifying the package source (that is dev work, not skill work).
- Running the full pipeline for the user — we explain, the user runs.
- Anything under `TRASH/` — never read it.
