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
| **H1** | ✅ RESOLVED | RL / Reward | `reward/reward_fn.py` | Jitter penalty exemption vectorized per-zone; prevents sacrificial zone reward hacking |
| **H2** | ✅ RESOLVED | Evaluation | `grader/evaluator.py` | Safety-gated efficiency & smoothness; zero-cooling penalty properly enforced |
| **H3** | ✅ RESOLVED | Physics | `dynamics/thermal_model.py` | Physical 105°C silicon junction failure bound enforced via np.clip |
| **H4** | ✅ RESOLVED | Physics | `dynamics/thermal_model.py` | 1D discrete Laplacian spatial thermal diffusion with Neumann boundaries |
| **H5** | ✅ RESOLVED | Server / API | `core/env.py` | /simulate endpoint dict handling and direct session access fixed |
| **H6** | ✅ RESOLVED | Benchmark | `tasks/hard.json` | Hard task initial temperatures adjusted to start within safe headroom (<70.5°C) |
| **M1** | ✅ RESOLVED | Systems | `core/env.py` | Multi-tenant session isolation with fine-grained per-session threading locks |
| **M2** | ✅ RESOLVED | MDP Formulation | `core/models.py` | Goal parameters (target_temp, safe_temp) included in Observation model |
| **M3** | ✅ RESOLVED | Evaluation | Benchmark scripts | Statistical multi-seed evaluation with CLI flags (--seeds, --num-seeds) |
| **M4** | ✅ RESOLVED | Dependencies | `pyproject.toml`, `requirements.txt` | Harmonized openenv-core, added [project.optional-dependencies] test |
| **M5** | ✅ RESOLVED | Analysis | `analysis/trend_predictor.py` | Closed-form OLS rolling linear regression sensitive to trajectory curvature |
| **M6** | ✅ RESOLVED | Inference | `inference/parser.py`, `inference.py` | Robust markdown fence stripping, regex extraction, diagnostic error logging |
| **M7** | ✅ RESOLVED | Docker / Infra | `Dockerfile`, `.dockerignore` | HEALTHCHECK curl probe added, .dockerignore configured, apt cache cleaned |


---

## Detailed Problem Statements

### 🔴 CRITICAL FINDINGS

#### Problem ID: C1 — Phantom Baselines Claimed in README ✅ RESOLVED
- **Severity**: 🔴 CRITICAL
- **Subsystem**: `README.md` (Lines 277–294), `baselines/`
- **Status**: ✅ RESOLVED — Built Gymnasium environment (`core/gym_env.py`), trained PPO RL models for all three difficulty tiers (`models/ppo_{easy,medium,hard}.zip`), implemented classical controllers (`baselines/classical.py`), RL inference agent (`baselines/rl_agent.py`), full multi-seed benchmark runner (`run_benchmark.py`), and baseline unit tests (`tests/test_baselines.py`).
- **Problem**: The README claims 4 benchmarked baseline approaches with precise performance figures:
  - Rule-Based: `0.45`
  - PID: `0.62`
  - LLM Agent: `0.68`
  - RL (PPO): `0.81`
  **None of these baselines exist in the codebase.** There is no PID controller, no RL training loop, no PPO/SAC/DDPG policy network, no training config, and no baseline comparison runner.
- **Why it matters**: In an interview or code audit, claiming empirical benchmarks for models that do not exist destroys technical credibility and invites accusations of fabricated results.
- **Concrete Failure Scenario**: Interviewer asks: *"Show me your PPO hyperparameter tuning and training convergence curves."* → There is zero RL training code to show.

---

#### Problem ID: C2 — Active HuggingFace API Secret Committed to Git ✅ RESOLVED
- **Severity**: 🔴 CRITICAL
- **Subsystem**: `.env` (Line 3), `.gitignore`
- **Status**: ✅ RESOLVED — Removed live token from `.env`, untracked `.env` from git tracking, created `.env.example` template, and ensured comprehensive git and docker ignore patterns.
- **Problem**: `.env` contains a live secret: `HF_TOKEN=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`. This file is tracked in version control and pushed to the public repository.
- **Why it matters**: Severe security liability. Public secret leakage demonstrates lack of production-grade credential hygiene.
- **Concrete Failure Scenario**: Automated token sniffers scrape the repository and abuse the token, or an interviewer inspects `.env` and flags a fundamental security violation.

---

#### Problem ID: C3 — `/reset` Accepts Arbitrary TaskConfig Permitting Evaluation Gaming ✅ RESOLVED
- **Severity**: 🔴 CRITICAL
- **Subsystem**: `core/env.py` (Lines 128–132), `core/simulator.py` (Lines 57–61)
- **Status**: ✅ RESOLVED — Defined strict Pydantic model `ResetPayload(task_name=..., seed=..., extra="forbid")` whitelist. Rejected non-canonical task names and custom arbitrary dictionaries with HTTP 400. Verified via `test_reset_rejects_injected_configuration` and `test_reset_rejects_arbitrary_task_names` in `tests/test_api_security.py`.
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

#### Problem ID: C4 — `get_score()` Returns Perfect 1.0 Score on Zero Steps ✅ RESOLVED
- **Severity**: 🔴 CRITICAL
- **Subsystem**: `core/simulator.py` (Lines 129–134)
- **Status**: ✅ RESOLVED — Modified `SimulationSession.get_score()` to return `0.0` score when `len(history_actions) == 0`. Added guard in `GET /score` endpoint returning HTTP 400 if queried on a 0-step session. Verified via `test_score_rejects_zero_step_evaluation` in `tests/test_api_security.py`.
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

#### Problem ID: H1 — Global Jitter Penalty Bypass Exploit ✅ RESOLVED
- **Severity**: 🟠 HIGH
- **Subsystem**: `reward/reward_fn.py` (Lines 37–38)
- **Status**: ✅ RESOLVED — Vectorized jitter exemption per-zone (`danger_mask = temps >= danger_threshold; zone_jitters = np.where(danger_mask, 0.0, zone_jitters)`). Confirmed via `test_compute_reward_jitter_bypass_is_per_zone` in `tests/reward/test_reward_fn.py`.
- **Problem**: 
  ```python
  if np.max(temps) >= config.safe_temperature - config.jitter_bypass_threshold:
      jitter = 0.0
  ```
  If **any single zone** reaches the threshold, the jitter penalty is zeroed out for the **entire datacenter**.
- **Why it matters**: Classic reward hacking vector. A reinforcement learning agent can learn to keep one sacrificial zone permanently hot, granting zero jitter cost across all other zones and enabling erratic, hardware-damaging bang-bang control.

---

#### Problem ID: H2 — Evaluation Metric Heavily Rewards Total Inaction ✅ RESOLVED
- **Severity**: 🟠 HIGH
- **Subsystem**: `grader/evaluator.py` (Lines 22–23), `grader/metrics.py` (Line 40)
- **Status**: ✅ RESOLVED — Refactored `evaluate_trajectory` in `grader/evaluator.py` so efficiency and jitter scores are strictly safety-gated: `(1.0 - metric) * sqrt(max(0.0, safety))`. Inaction causing overheating yields a failing grade. Verified via `test_zero_cooling_does_not_get_unearned_efficiency_bonus` in `tests/grader/test_evaluator.py` and `test_zero_policy_is_weak_on_active_tasks` in `tests/test_submission_readiness.py`.
- **Problem**: 
  ```python
  energy_score = 1.0 - energy  # where energy is mean cooling in [0, 1]
  jitter_score = 1.0 - jitter  # where jitter is mean step diff
  ```
  A completely dead agent applying `cooling = [0.0] * N` automatically scores $1.0$ in Energy ($w=0.2$) and $1.0$ in Smoothness ($w=0.1$).
- **Why it matters**: Doing nothing starts with a guaranteed $0.30$ (30%) score floor. If an easy task stays below safe limits for several initial steps, a broken or zero-action policy gets an artificially inflated passing grade.

---

#### Problem ID: H3 — Absence of Physical Upper Temperature Bound ✅ RESOLVED
- **Severity**: 🟠 HIGH
- **Subsystem**: `dynamics/thermal_model.py` (Lines 35–41)
- **Status**: ✅ RESOLVED — Defined `MAX_PHYSICAL_TEMPERATURE = 105.0` in `config/constants.py` (representing silicon thermal junction breakdown $T_{\text{jmax}}$). Clamped `next_temperatures` via `np.clip(temperatures + delta_t, ambient_temp, MAX_PHYSICAL_TEMPERATURE)` in `dynamics/thermal_model.py`. Verified via unit test `test_apply_transition_enforces_upper_temperature_bound`.
- **Problem**: While temperatures are floored at ambient via `np.maximum(..., ambient_temp)`, there is no upper ceiling or hardware burnout cutoff. Temperatures can mathematically rise to $150^\circ\text{C}+$ without triggering equipment shutdown, catastrophic failure, or termination.
- **Why it matters**: Contradicts the README claim of "high-fidelity industrial realism". Real silicon begins throttling at ~85°C and suffers thermal shutdown/destruction at ~100-105°C.

---


#### Problem ID: H4 — Zero Inter-Zone Spatial Thermal Diffusion ✅ RESOLVED
- **Severity**: 🟠 HIGH
- **Subsystem**: `dynamics/thermal_model.py` (Line 35)
- **Status**: ✅ RESOLVED — Defined `KAPPA_DIFFUSION = 0.05` in `config/constants.py`. Implemented discrete 1D Laplacian thermal diffusion (`diffusion = KAPPA_DIFFUSION * [T_{i-1} + T_{i+1} - 2*T_i]`) with Neumann boundary conditions in `dynamics/thermal_model.py`. Verified via unit test `test_apply_transition_diffuses_heat_between_adjacent_zones`.
- **Problem**: 
  ```python
  delta_t = ALPHA * workloads - cooling_effect * cooling + GAMMA * (ambient_temp - temperatures)
  ```
  All operations are purely 1D element-wise arrays. There is no conductive or convective heat transfer between zone i and zone i ± 1.
- **Why it matters**: The simulation is physically N completely independent, uncoupled 1-zone thermal problems run in parallel. It does not require multi-agent coordination or spatial reasoning.

---


#### Problem ID: H5 — `/simulate` Endpoint Fails with AttributeError ✅ RESOLVED
- **Severity**: 🟠 HIGH
- **Subsystem**: `core/env.py` (Lines 206–270)
- **Status**: ✅ RESOLVED — Refactored `/simulate` in `core/env.py` to check `isinstance(initial_observation, dict)`, properly extract `CURRENT_SESSION.config` under `env_lock`, and handle default query parameters without raising parameter validation errors. Added integration test `test_simulate_endpoint_executes_successfully` in `tests/test_api_security.py`.
- **Problem**: 
  ```python
  initial_observation = reset(task_name=task_name)
  session_config = initial_observation.model_dump()
  ```
  `reset()` returns a Python `dict` (via `session.observation.model_dump()` at line 147). Calling `.model_dump()` on an existing dict raises `AttributeError: 'dict' object has no attribute 'model_dump'`.
- **Why it matters**: Calling `POST /simulate` immediately returns HTTP 500. A primary documented endpoint is broken in production.

---


#### Problem ID: H6 — Hard Benchmark Scenario Starts in Immediate Violation ✅ RESOLVED
- **Severity**: 🟠 HIGH
- **Subsystem**: `tasks/hard.json`
- **Status**: ✅ RESOLVED — Adjusted `initial_temperatures` in `tasks/hard.json` so the maximum initial temperature is $70.4^\circ\text{C}$ ($< 70.5^\circ\text{C}$ safe limit). Preserves extreme stress (0.1°C headroom under 0.98 workload) while eliminating the unpreventable pre-step violation. Verified via invariant test `test_canonical_tasks_start_in_non_violating_state` in `tests/test_submission_readiness.py`.
- **Problem**: In `hard.json`, `safe_temperature` is set to `70.5`, but `initial_temperatures` are `[69.0, 70.0, 70.5, 69.5, 71.0, 69.5, 70.5, 69.0]`. Zone 4 is at 71.0°C (> 70.5°C) before step 0 has even executed.
- **Why it matters**: It is mathematically impossible for any agent to achieve a 100% safety score on `hard.json` because step 0 is in violation before any cooling command can be issued.

---


### 🟡 MEDIUM SEVERITY FINDINGS

#### Problem ID: M1 — Global `CURRENT_SESSION` Without Multi-Tenant State Isolation ✅ RESOLVED
- **Severity**: 🟡 MEDIUM
- **Subsystem**: `core/env.py` (Line 46)
- **Status**: ✅ RESOLVED — Implemented `ACTIVE_SESSIONS: Dict[str, SimulationSession]` registry with per-session fine-grained locks and global registry lock. Maintained `DEFAULT_SESSION_ID = "default"` for backward compatibility. Added session resolution via query parameters and headers across all endpoints. Verified via concurrency and isolation test `test_multi_tenant_session_isolation` in `tests/test_api_security.py`.
- **Problem**: A single global variable `CURRENT_SESSION` holds the environment instance. While `threading.Lock` serializes steps, two simultaneous client sessions will overwrite each other's state and interfere with ongoing rollouts.
- **Why it matters**: Multiple clients or parallel benchmark evaluation threads would corrupt each other's rollout trajectories.

---


#### Problem ID: M2 — Observation Space Lacks Goal Parameters (Non-Markovian from Observation Alone) ✅ RESOLVED
- **Severity**: 🟡 MEDIUM
- **Subsystem**: `core/models.py` (Lines 10–20), `core/simulator.py`, `dynamics/thermal_model.py`
- **Status**: ✅ RESOLVED — Added optional goal fields `target_temperature` and `safe_temperature` to `Observation` in `core/models.py`. Populated them automatically from active `TaskConfig` during session initialization and step transitions. Verified in `test_api_exposes_reset_step_state_for_root_submission` in `tests/test_submission_readiness.py`.
- **Problem**: The `Observation` model exposes temperatures, workloads, cooling, ambient temp, and step. It does **not** include `safe_temperature` or `target_temperature`. The agent must read local JSON task configuration files out-of-band to know its goal.
- **Why it matters**: A remote agent communicating with the server over HTTP cannot know what temperature to regulate toward or what safe boundary to avoid without out-of-band file access.

---


#### Problem ID: M3 — Lack of Multi-Seed Statistical Evaluation ✅ RESOLVED
- **Severity**: 🟡 MEDIUM
- **Subsystem**: Evaluation workflows, `sample_run.py`, `run_benchmark.py`
- **Status**: ✅ RESOLVED — Created `run_benchmark.py` running multi-seed statistical evaluation (mean ± std) across 5 control paradigms. Extended `sample_run.py` with CLI flags (`--seeds`, `--num-seeds`, `--task`) enabling multi-seed statistical rollouts over HTTP. Added unit test `test_multi_seed_evaluation_computes_statistics` in `tests/test_baselines.py`.
- **Problem**: Benchmarks evaluate exactly one hardcoded seed per task (`101`, `202`, `303`). There is no measurement of policy variance, confidence intervals, or performance across randomized seeds.
- **Why it matters**: A single deterministic seed cannot measure controller robustness or volatility resilience, risking overfitting or lucky rollouts.

---


#### Problem ID: M4 — Dependency Specification Conflicts and Missing Locks ✅ RESOLVED
- **Severity**: 🟡 MEDIUM
- **Subsystem**: `pyproject.toml`, `requirements.txt`
- **Status**: ✅ RESOLVED — Harmonized `openenv-core` and `ollama` across `pyproject.toml` and `requirements.txt`. Added `[project.optional-dependencies] test = ["pytest>=8.0.0", "pytest-cov"]` to `pyproject.toml`.
- **Problem**: `requirements.txt` specifies `openenv-core`, while `pyproject.toml` specifies `openenv`. Version numbers are loosely bounded (`>=`), and `pytest` is omitted from `pyproject.toml`.
- **Why it matters**: Inconsistent package definitions break packaging, CI pipelines, and standard pip installs across different container environments.

---


#### Problem ID: M5 — Oversimplified Flawed Trend Prediction Mathematics ✅ RESOLVED
- **Severity**: 🟡 MEDIUM
- **Subsystem**: `analysis/trend_predictor.py` (Lines 14–20)
- **Status**: ✅ RESOLVED — Replaced telescoping delta summation with closed-form Ordinary Least Squares (OLS) linear regression across the rolling window. Evaluates true trajectory slope while accounting for intermediate curvature and acceleration. Verified via unit test `test_predict_thermal_future_curvature_sensitivity` in `tests/analysis/test_trend_predictor.py`.
- **Problem**: In `predict_thermal_future`, summing adjacent differences algebraically cancels all intermediate points, reducing velocity to simply (T_last - T_first) / k. The predictor is blind to intermediate trajectory curvature.
- **Why it matters**: A straight-chord extrapolation fails to detect rapid exponential runaway or decelerating plateaus, giving false safety senses or premature alarms.

---


#### Problem ID: M6 — Brittle LLM Output Parsing and Silent Failure Masking ✅ RESOLVED
- **Severity**: 🟡 MEDIUM
- **Subsystem**: `inference/parser.py`, `inference.py` (Lines 29–42)
- **Status**: ✅ RESOLVED — Upgraded `parse_llm_response` in `inference/parser.py` with markdown code fence extraction (` ```json `), outermost JSON boundary isolation, and regex fallback for `"cooling": [...]`. Implemented `parse_action_safe` returning both action and diagnostic error messages. Integrated into `inference.py` to record parser diagnostics in step logs instead of silently masking compliance failures. Added unit test `test_llm_parser_handles_markdown_blocks_and_logs_errors` in `tests/test_submission_readiness.py`.
- **Problem**: Uses string `.find("{")` and `.rfind("}")`. If JSON decoding fails, it silently defaults to a fixed action of `[0.3] * num_zones` without logging or alerting, obscuring systematic model prompt compliance failures.


---

#### Problem ID: M7 — Suboptimal Docker Image Hygiene ✅ RESOLVED
- **Severity**: 🟡 MEDIUM
- **Subsystem**: `Dockerfile`, `.dockerignore`
- **Status**: ✅ RESOLVED — Configured `.dockerignore` to cleanly ignore `.git`, `.pytest_cache`, `.venv`, `.env`, and intermediate build artifacts. Added `HEALTHCHECK` directive in `Dockerfile` verifying `GET /` on port 7860 via curl (`HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:7860/ || exit 1`). Cleaned apt lists to minimize container image footprint. Verified via `test_dockerfile_matches_server_entrypoint` in `tests/test_submission_readiness.py`.
- **Problem**: Missing `.dockerignore` means `.git/`, `.pytest_cache/`, and virtualenv directories are baked into images. No `HEALTHCHECK` directive exists, and build tools (`build-essential`) remain in the final slim image layer.

