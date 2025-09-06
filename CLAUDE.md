# CLAUDE.md — Project Assistant Guide

**Project:** Reproducible RL Template
**Assistant:** Claude (Anthropic)

This file tells Claude exactly how to help on this repo. It defines goals, guardrails, code style, tasks Claude can perform, and the workflow for proposing changes. Keep this document in sync as the project evolves.

---

## 1) Project Goals & Scope

* **Goal:** Provide a clean, reproducible template for deep RL experiments (and baselines) with deterministic runs, clear configs, and one-command training/evaluation.
* **Primary language:** Python 3.11+
* **Secondary tools:** Bash for scripts; optional TypeScript/JS only for dashboard or minimal UI.
* **Framework assumptions:** PyTorch + Gymnasium (or PettingZoo for multi‑agent), Hydra for configs, Weights & Biases or MLflow for tracking (toggle via config).
* **Core pillars:** Reproducibility • Clarity • Small, documented modules • Fast iteration • Automated testing.

Out of scope for Claude unless asked explicitly: cloud infra, monolithic refactors, adding heavyweight dependencies, or changing licenses.

---

## 2) Folder & File Conventions

```
.
├── src/
│   ├── rrl/                   # Package root (Reproducible RL)
│   │   ├── agents/            # PPO, DQN, etc.
│   │   ├── envs/              # Thin wrappers around Gym envs
│   │   ├── utils/             # seed utils, buffers, schedulers, logging
│   │   ├── train.py           # Train entrypoint (Hydra-aware)
│   │   └── eval.py            # Eval entrypoint
├── configs/
│   ├── defaults.yaml          # Base config (Hydra composition)
│   ├── agent/                 # Agent-specific configs
│   ├── env/                   # Environment configs
│   └── run/                   # Run presets (debug, sweep, paper)
├── experiments/               # Outputs (git‑ignored): runs, checkpoints
├── scripts/                   # CLI helpers, setup, lint, test, sweep
├── tests/                     # pytest unit/integration tests
├── notebooks/                 # Lightweight EDA; never source of truth
├── .pre-commit-config.yaml    # Formatting/lint hooks
├── LICENSE
└── CLAUDE.md                  # You are here
```

If structure differs, read `README.md` and inline module docs, then adapt suggestions accordingly.

---

## 3) How Claude Should Work Here

**Mindset:** propose the *smallest correct change* that improves reproducibility, clarity, or safety.

1. **Before writing code**

   * Read related config(s) in `configs/` and module docstrings.
   * Confirm interfaces and expected types. Keep public APIs stable.
   * Prefer extension points over rewrites.

2. **When changing code**

   * Maintain determinism: set `torch`, `numpy`, and Python seeds; control CUDA flags.
   * Keep batch shapes and dtypes explicit; avoid silent casts.
   * Add/extend tests for any bugfix or feature.
   * Update docs and configs if behavior changes.

3. **When unsure**

   * Open a Draft PR with options A/B if trade‑offs exist. Summarize pros/cons and measured impact.

---

## 4) Guardrails

* **Do not** commit secrets, API keys, or large datasets. Use env vars and `.env.example`.
* **Do not** change training defaults in `configs/defaults.yaml` without explaining measured benefits.
* **Do not** introduce non‑deterministic ops without justification and a toggle.
* **Do not** create circular imports or hidden global state.
* **Licensing:** keep code under existing project license; attribute new third‑party code.

---

## 5) Coding Standards

**Python**

* Use **Black** (line length 100) + **ruff** for linting. Type‑hint everything.
* Public functions and modules require docstrings (Google or NumPy style) with `Args/Returns/Raises`.
* Separate pure logic from I/O. Keep `train.py` thin; heavy logic lives in `rrl/` modules.
* Prefer dataclasses for config‑like objects.
* Keep functions <60 lines where feasible; factor helpers.

**Testing (pytest)**

* Each bug fix needs a regression test.
* Add unit tests for utilities and small modules; add one integration test per agent.
* Use small deterministic fixtures (`CartPole-v1`, tiny nets) to keep tests <5s.

**Docs**

* Update README snippets and inline examples when changing CLI or config keys.
* Keep notebooks illustrative only; never authoritative.

---

## 6) Reproducibility Checklist

* [ ] Set `PYTHONHASHSEED` and standardized seeding in `rrl/utils/seed.py`.
* [ ] Configure PyTorch deterministic flags (warn if they slow kernels).
* [ ] Log **all** hyperparameters and hashes of source files.
* [ ] Save training/eval configs alongside checkpoints.
* [ ] Record library versions, CUDA version, and GPU model.
* [ ] Provide a minimal `scripts/run_debug.sh` to validate end‑to‑end.

---

## 7) Configuration (Hydra)

* Centralize defaults in `configs/defaults.yaml` and compose per‑run presets under `configs/run/`.
* All magic numbers must live in configs, not code.
* Boolean toggles: `use_wandb`, `deterministic`, `amp`, `clip_grad`, `torch_compile`.
* Allow overrides via CLI: e.g., `python -m rrl.train agent=ppo env=cartpole run=debug`.

---

## 8) Logging & Tracking

* Logger interface in `rrl/utils/logging.py` with pluggable backends:

  * `ConsoleLogger` (default, minimal)
  * `WandbLogger` or `MLflowLogger` if enabled via config
* Always log: episode returns, lengths, loss components, LR, grad norms, wall‑clock.
* Summaries written to `experiments/<run_id>/` with a `manifest.json`.

---

## 9) Performance & Safety

* Include gradient clipping and NaN/inf guards.
* Use mixed precision (`amp`) behind a config flag; log numerical instabilities.
* Keep rollout collection separate from learner to support A2C/PPO/IMPALA variants later.
* Provide a `--max-steps` and `--time-limit` to guarantee bounded runs.

---

## 10) CLI Contracts

**Training**

```
python -m rrl.train agent=<ppo|dqn> env=<cartpole|...> run=<debug|paper> \
    seed=42 total_steps=100_000 deterministic=true use_wandb=false
```

**Evaluation**

```
python -m rrl.eval checkpoint=experiments/<id>/ckpt.pt episodes=10 render=false
```

Both commands must accept `+overrides` for any Hydra key.

---

## 11) What Claude Can Safely Do

* Create small utilities (e.g., replay buffer, GAE lambda, schedulers) with tests.
* Add docstrings, comments, type hints, and examples.
* Improve error messages and input validation.
* Write/extend unit tests and GitHub Actions for lint/test.
* Draft scripts: `scripts/setup_dev.sh`, `scripts/run_debug.sh`, `scripts/format.sh`.
* Optimize hot spots with clear benchmarks (document micro/macro speedups).

**Ask for review before:** changing training loops, switching core dependencies, or altering config schemas.

---

## 12) PR & Commit Guidelines

* Branch naming: `feat/…`, `fix/…`, `chore/…`, `docs/…`, `test/…`.
* Conventional commits (preferred): `feat(agent): add PPO entropy bonus sched`.
* PR template must include: intent, approach, tests, reproducibility notes, and screenshots/metrics.
* Keep PRs < 400 LOC when possible.

---

## 13) Local Dev Environment

* Use `uv` or `pip-tools` for locked deps; include `requirements.in`/`requirements.txt` or `pyproject.toml`.
* Provide a minimal `Makefile`/`justfile` with:

```
just setup   # create venv, install deps, pre-commit
just format  # ruff + black
just test    # pytest -q
just debug   # tiny deterministic run
```

* Pre-commit must run ruff, black, end‑of‑file‑fixer, and trailing whitespace.

---

## 14) Data & Checkpoints

* Datasets and environment assets are **not** committed. Use `data/` (git‑ignored).
* Checkpoints and logs live under `experiments/` (git‑ignored). Always include a `metadata.json` with: config, git SHA, env info, and metric summary.

---

## 15) Issue Labels (Suggested)

* `good first issue`, `help wanted`, `repro`, `perf`, `api`, `tests`, `docs`, `tech debt`, `blocked`.

---

## 16) Ready‑To‑Use Prompts for Claude

**Tidy a module**

> Improve type hints and docstrings for `rrl/utils/seed.py`. Add a pytest that verifies deterministic behavior across two runs with identical seeds.

**Add a feature (small)**

> Implement gradient‑norm logging (L2) per step in the training loop and surface a `clip_grad=true` option with a configurable `clip_grad_norm`. Include tests for both paths.

**Bug reproduction**

> Create a failing test that reproduces issue #123 (diverging loss on CartPole with AMP=true). Keep the test under 3s runtime.

**Benchmark scaffold**

> Add `scripts/bench_rollout.sh` to time 100 rollout steps (CPU/GPU). Print mean/std, env name, and PyTorch/CUDA versions.

---

## 17) Review Checklist for Claude

Before marking work ready for review, verify:

* [ ] Deterministic seed path and logged environment info
* [ ] New/changed behavior covered by tests
* [ ] Config keys documented and discoverable via `--help` or README
* [ ] No new warnings or lint errors; pre‑commit passes
* [ ] Runtime on `run=debug` unchanged or improved

---

## 18) Contact & Decisions

If a decision is ambiguous (algorithmic choice, API shape), prefer:

1. The simpler, more composable design
2. Minimal public API surface
3. Measured evidence (micro‑benchmarks, ablations), included in PR description

If in doubt, open a Draft PR with a brief design note.

---

### Please do not use emojis within the code

**End of CLAUDE.md**
