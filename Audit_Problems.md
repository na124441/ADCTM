# ADCTM Audit: Problems Catalog (Audit_Problems.md)

This document serves as the master register of all material engineering, ML/RL, physical/scientific, architectural, security, evaluation, and production issues uncovered during the adversarial audit of the **ADCTM (Autonomous Data Centre Thermal Management)** repository.

---

## Summary Matrix

| ID | Severity | Category | Subsystem | Brief Summary |
|---|---|---|---|---|
| **C1** | ✅ RESOLVED | ML / Claims | `README.md`, `baselines/` | Real empirical baselines trained and verified (PPO, PID, Rule-Based, LLM, Zero) |
| **C2** | ✅ RESOLVED | Security | `.env`, `.gitignore` | Token untracked from git, sanitized with template, .gitignore & .dockerignore configured |
| **C3** | ✅ RESOLVED | Evaluation / API | `core/env.py`, `core/models.py` | ResetPayload enforces canonical task whitelist, rejects arbitrary injected configs |
| **C4** | ✅ RESOLVED | Evaluation | `core/simulator.py`, `core/env.py` | /score rejects zero-step sessions with HTTP 400; get_score returns 0.0 fallback |
| **H1** | 🟠 HIGH | RL / Reward | `reward/reward_fn.py` | Jitter penalty bypassed globally if any single zone is near safety threshold |
| **H2** | 🟠 HIGH | Evaluation | `grader/evaluator.py`, `metrics.py` | Degenerate zero-action policy gets 30% baseline score for free (perfect energy & jitter) |
| **H3** | 🟠 HIGH | Physics | `dynamics/thermal_model.py` | No physical upper temperature bound (temperatures can rise infinitely past destruction) |
| **H4** | 🟠 HIGH | Physics | `dynamics/thermal_model.py` | No inter-zone thermal diffusion/coupling (effective N isolated 1-zone problems) |
| **H5** | 🟠 HIGH | Server / API | `core/env.py` | `/simulate` endpoint crashes (`AttributeError: 'dict' object has no attribute 'model_dump'`) |
| **H6** | 🟠 HIGH | Benchmark | `tasks/hard.json` | Hard task starts already in thermal safety violation at step 0 |
| **M1** | 🟡 MEDIUM | Systems | `core/env.py` | Global `CURRENT_SESSION` allows no multi-client or multi-agent state isolation |
| **M2** | 🟡 MEDIUM | MDP Formulation | `core/models.py` | Observation lacks goal info (`target_temperature`, `safe_temperature`) needed for Markov control |
| **M3** | 🟡 MEDIUM | Evaluation | Benchmark scripts | Single fixed seed evaluation per task with zero variance or confidence intervals reported |
| **M4** | 🟡 MEDIUM | Dependencies | `pyproject.toml`, `requirements.txt` | Mismatched dependency specs (`openenv` vs `openenv-core`) and unpinned versions |
| **M5** | 🟡 MEDIUM | Analysis | `analysis/trend_predictor.py` | Velocity calculation cancels intermediate values; purely linear endpoint extrapolation |
| **M6** | 🟡 MEDIUM | Inference | `inference/parser.py`, `inference.py` | Brittle JSON string parsing via substring find; silent fallback to 0.3 masking errors |
| **M7** | 🟡 MEDIUM | Docker / Infra | `Dockerfile` | Missing `.dockerignore`, missing container health check, unnecessary tools in final image |

---

## Detailed Problem Statements

### 🔴 CRITICAL FINDINGS

#### Problem ID: C1 — Phantom Baselines Claimed in README
- **Severity**: 🔴 CRITICAL
- **Subsystem**: `README.md` (Lines 277–294), `baselines/`
- **Problem**: The README claims 4 benchmarked baseline approaches with precise performance figures:
  - Rule-Based: `0.45`
  - PID: `0.62`
  - LLM Agent: `0.68`
  - RL (PPO): `0.81`
  **None of these baselines exist in the codebase.** There is no PID controller, no RL training loop, no PPO/SAC/DDPG policy network, no training config, and no baseline comparison runner.
- **Why it matters**: In an interview or code audit, claiming empirical benchmarks for models that do not exist destroys technical credibility and invites accusations of fabricated results.
- **Concrete Failure Scenario**: Interviewer asks: *"Show me your PPO hyperparameter tuning and training convergence curves."* → There is zero RL training code to show.

---

#### Problem ID: C2 — Active HuggingFace API Secret Committed to Git
- **Severity**: 🔴 CRITICAL
- **Subsystem**: `.env` (Line 3), `.gitignore`
- **Problem**: `.env` contains a live secret: `HF_TOKEN=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`. This file is tracked in version control and pushed to the public repository.
- **Why it matters**: Severe security liability. Public secret leakage demonstrates lack of production-grade credential hygiene.
- **Concrete Failure Scenario**: Automated token sniffers scrape the repository and abuse the token, or an interviewer inspects `.env` and flags a fundamental security violation.

---

#### Problem ID: C3 — `/reset` Accepts Arbitrary TaskConfig Permitting Evaluation Gaming
- **Severity**: 🔴 CRITICAL
- **Subsystem**: `core/env.py` (Lines 128–132), `core/simulator.py` (Lines 57–61)
- **Problem**: The `/reset` endpoint accepts arbitrary dictionary payloads. When passed a custom payload without `task_name`, it executes `SimulationSession.from_dict(config_payload)` directly.
- **Why it matters**: An agent or user can inject a trivial config (e.g. `safe_temperature=9999`, `max_steps=1`) over the network, step once, and retrieve a score of 1.0 from `/score`. The evaluation boundary has zero integrity.
- **Concrete Failure Scenario**: 
  ```bash
  curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"num_zones":1,"initial_temperatures":[20],"initial_workloads":[0],"ambient_temperature":20,"safe_temperature":9999,"max_steps":1,"target_temperature":20,"seed":42}'
  curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"cooling":[0.0]}'
  curl -X GET http://localhost:7860/score
  # Returns: {"total": 1.0, "score": 1.0, "metrics": {"safety": 1.0, ...}}
  ```

---

#### Problem ID: C4 — `get_score()` Returns Perfect 1.0 Score on Zero Steps
- **Severity**: 🔴 CRITICAL
- **Subsystem**: `core/simulator.py` (Lines 129–134)
- **Problem**: In `SimulationSession.get_score()`:
  ```python
  if len(self.history_actions) == 0:
      return {
          "total": 1.0,
          "score": 1.0,
          "metrics": {"safety": 1.0, "precision": 1.0, "efficiency": 1.0, "smoothness": 1.0}
      }
  ```
- **Why it matters**: Calling `/score` immediately after `/reset` without executing any action yields an instant 100% score across all 4 metrics.
- **Concrete Failure Scenario**: Any benchmarking harness querying `/score` on uninitialized or aborted rollouts records a false perfect evaluation score.

---

### 🟠 HIGH SEVERITY FINDINGS

#### Problem ID: H1 — Global Jitter Penalty Bypass Exploit
- **Severity**: 🟠 HIGH
- **Subsystem**: `reward/reward_fn.py` (Lines 37–38)
- **Problem**: 
  ```python
  if np.max(temps) >= config.safe_temperature - config.jitter_bypass_threshold:
      jitter = 0.0
  ```
  If **any single zone** reaches the threshold, the jitter penalty is zeroed out for the **entire datacenter**.
- **Why it matters**: Classic reward hacking vector. A reinforcement learning agent can learn to keep one sacrificial zone permanently hot, granting zero jitter cost across all other zones and enabling erratic, hardware-damaging bang-bang control.

---

#### Problem ID: H2 — Evaluation Metric Heavily Rewards Total Inaction
- **Severity**: 🟠 HIGH
- **Subsystem**: `grader/evaluator.py` (Lines 22–23), `grader/metrics.py` (Line 40)
- **Problem**: 
  ```python
  energy_score = 1.0 - energy  # where energy is mean cooling in [0, 1]
  jitter_score = 1.0 - jitter  # where jitter is mean step diff
  ```
  A completely dead agent applying `cooling = [0.0] * N` automatically scores $1.0$ in Energy ($w=0.2$) and $1.0$ in Smoothness ($w=0.1$).
- **Why it matters**: Doing nothing starts with a guaranteed $0.30$ (30%) score floor. If an easy task stays below safe limits for several initial steps, a broken or zero-action policy gets an artificially inflated passing grade.

---

#### Problem ID: H3 — Absence of Physical Upper Temperature Bound
- **Severity**: 🟠 HIGH
- **Subsystem**: `dynamics/thermal_model.py` (Lines 35–41)
- **Problem**: While temperatures are floored at ambient via `np.maximum(..., ambient_temp)`, there is no upper ceiling or hardware burnout cutoff. Temperatures can mathematically rise to $150^\circ\text{C}+$ without triggering equipment shutdown, catastrophic failure, or termination.
- **Why it matters**: Contradicts the README claim of "high-fidelity industrial realism". Real silicon begins throttling at ~85°C and suffers thermal shutdown/destruction at ~100-105°C.

---

#### Problem ID: H4 — Zero Inter-Zone Spatial Thermal Diffusion
- **Severity**: 🟠 HIGH
- **Subsystem**: `dynamics/thermal_model.py` (Line 35)
- **Problem**: 
  ```python
  delta_t = ALPHA * workloads - cooling_effect * cooling + GAMMA * (ambient_temp - temperatures)
  ```
  All operations are purely 1D element-wise arrays. There is no conductive or convective heat transfer between zone i and zone i ± 1.
- **Why it matters**: The simulation is physically N completely independent, uncoupled 1-zone thermal problems run in parallel. It does not require multi-agent coordination or spatial reasoning.

---

#### Problem ID: H5 — `/simulate` Endpoint Fails with AttributeError
- **Severity**: 🟠 HIGH
- **Subsystem**: `core/env.py` (Lines 206–207)
- **Problem**: 
  ```python
  initial_observation = reset(task_name=task_name)
  session_config = initial_observation.model_dump()
  ```
  `reset()` returns a Python `dict` (via `session.observation.model_dump()` at line 147). Calling `.model_dump()` on an existing dict raises `AttributeError: 'dict' object has no attribute 'model_dump'`.
- **Why it matters**: Calling `POST /simulate` immediately returns HTTP 500. A primary documented endpoint is broken in production.

---

#### Problem ID: H6 — Hard Benchmark Scenario Starts in Immediate Violation
- **Severity**: 🟠 HIGH
- **Subsystem**: `tasks/hard.json`
- **Problem**: In `hard.json`, `safe_temperature` is set to `70.5`, but `initial_temperatures` are `[69.0, 70.0, 70.5, 69.5, 71.0, 69.5, 70.5, 69.0]`. Zone 4 is at 71.0°C (> 70.5°C) before step 0 has even executed.
- **Why it matters**: It is mathematically impossible for any agent to achieve a 100% safety score on `hard.json` because step 0 is in violation before any cooling command can be issued.

---

### 🟡 MEDIUM SEVERITY FINDINGS

#### Problem ID: M1 — Global `CURRENT_SESSION` Without Multi-Tenant State Isolation
- **Severity**: 🟡 MEDIUM
- **Subsystem**: `core/env.py` (Line 46)
- **Problem**: A single global variable `CURRENT_SESSION` holds the environment instance. While `threading.Lock` serializes steps, two simultaneous client sessions will overwrite each other's state and interfere with ongoing rollouts.

---

#### Problem ID: M2 — Observation Space Lacks Goal Parameters (Non-Markovian from Observation Alone)
- **Severity**: 🟡 MEDIUM
- **Subsystem**: `core/models.py` (Lines 10–20)
- **Problem**: The `Observation` model exposes temperatures, workloads, cooling, ambient temp, and step. It does **not** include `safe_temperature` or `target_temperature`. The agent must read local JSON task configuration files out-of-band to know its goal.

---

#### Problem ID: M3 — Lack of Multi-Seed Statistical Evaluation
- **Severity**: 🟡 MEDIUM
- **Subsystem**: Evaluation workflows, `sample_run.py`
- **Problem**: Benchmarks evaluate exactly one hardcoded seed per task (`101`, `202`, `303`). There is no measurement of policy variance, confidence intervals, or performance across randomized seeds.

---

#### Problem ID: M4 — Dependency Specification Conflicts and Missing Locks
- **Severity**: 🟡 MEDIUM
- **Subsystem**: `pyproject.toml`, `requirements.txt`
- **Problem**: `requirements.txt` specifies `openenv-core`, while `pyproject.toml` specifies `openenv`. Version numbers are loosely bounded (`>=`), and `pytest` is omitted from `pyproject.toml`.

---

#### Problem ID: M5 — Oversimplified Flawed Trend Prediction Mathematics
- **Severity**: 🟡 MEDIUM
- **Subsystem**: `analysis/trend_predictor.py` (Lines 14–20)
- **Problem**: In `predict_thermal_future`, summing adjacent differences algebraically cancels all intermediate points, reducing velocity to simply (T_last - T_first) / k. The predictor is blind to intermediate trajectory curvature.

---

#### Problem ID: M6 — Brittle LLM Output Parsing and Silent Failure Masking
- **Severity**: 🟡 MEDIUM
- **Subsystem**: `inference/parser.py`, `inference.py` (Lines 29–42)
- **Problem**: Uses string `.find("{")` and `.rfind("}")`. If JSON decoding fails, it silently defaults to a fixed action of `[0.3] * num_zones` without logging or alerting, obscuring systematic model prompt compliance failures.

---

#### Problem ID: M7 — Suboptimal Docker Image Hygiene
- **Severity**: 🟡 MEDIUM
- **Subsystem**: `Dockerfile`
- **Problem**: Missing `.dockerignore` means `.git/`, `.pytest_cache/`, and virtualenv directories are baked into images. No `HEALTHCHECK` directive exists, and build tools (`build-essential`) remain in the final slim image layer.
