# ADCTM Audit: Remediation Plans (Audit_Fixes_Plans.md)

This document tracks the detailed engineering implementation plans for addressing each problem recorded in `Audit_Problems.md`. As we progress through fixing issues, actionable remediation designs, file modifications, code snippets, and verification procedures are appended here.

---

## Remediation Index

| Problem ID | Problem Description | Plan Status |
|---|---|---|
| **C1** | Phantom Baselines Claimed in README | ✅ **Completed & Verified** |
| **C2** | Active HuggingFace API Secret Committed to Git | ✅ **Completed & Verified** |
| **C3** | `/reset` Accepts Arbitrary TaskConfig Permitting Evaluation Gaming | ✅ **Completed & Verified** |
| **C4** | `get_score()` Returns Perfect 1.0 Score on Zero Steps | ✅ **Completed & Verified** |
| **H1** | Global Jitter Penalty Bypass Exploit | ✅ **Completed & Verified** |
| **H2** | Evaluation Metric Heavily Rewards Total Inaction | ✅ **Completed & Verified** |
| **H3** | Absence of Physical Upper Temperature Bound | 📝 **Drafted (Below)** |
| **H4** | Zero Inter-Zone Spatial Thermal Diffusion | ⏳ Pending |
| **H5** | `/simulate` Endpoint Fails with AttributeError | ⏳ Pending |
| **H6** | Hard Benchmark Scenario Starts in Immediate Violation | ⏳ Pending |
| **M1**–**M7** | Medium Severity Architectural / Systems Polish | ⏳ Pending |

---

## 🛠️ Implementation Plans

### 1. Fix Plan for C1: Phantom Baselines (RL, PID, Rule-Based)

#### Overview
README lines 277–294 currently claim performance scores for Rule-Based (0.45), PID (0.62), LLM Agent (0.68), and RL/PPO (0.81), but none of these controllers or training routines exist in the codebase. To make this flagship-grade, we implement legitimate classical controllers, wrap the simulator into a standard Gymnasium environment, train an RL agent (PPO), and execute a unified benchmark across random seeds to replace the phantom table with real empirical data.

#### Architecture of Solution
```
                    ┌───────────────────────────────┐
                    │      core/simulator.py        │
                    │   (Fast In-Process Physics)   │
                    └──────────────┬────────────────┘
                                   │
                    ┌──────────────▼────────────────┐
                    │       core/gym_env.py         │
                    │   (Gymnasium Standard Env)    │
                    └──────┬─────────────────┬──────┘
                           │                 │
            ┌──────────────▼──────┐   ┌──────▼──────────────┐
            │   Classical Models  │   │     RL Training     │
            │  - baselines/pid.py │   │   - train_rl.py     │
            │  - baselines/rule.py│   │     (PPO via SB3)   │
            └──────────────┬──────┘   └──────┬──────────────┘
                           │                 │
                           └────────┬────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │       run_benchmark.py        │
                    │   (Multi-Seed Evaluator)      │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                      Empirical Data in README.md
```

#### Detailed File Changes

##### 1. `core/gym_env.py` (NEW)
- Subclass `gymnasium.Env`.
- Expose:
  - `observation_space`: `Box(low=0.0, high=1.0, shape=(num_zones * 3 + 2,), dtype=np.float32)` (normalized temperatures, workloads, previous cooling, ambient temp, normalized time step).
  - `action_space`: `Box(low=0.0, high=1.0, shape=(num_zones,), dtype=np.float32)`.
- Methods:
  - `reset(seed=None, options=None)`: In-process initialization of `SimulationSession`.
  - `step(action)`: Translates continuous `action` to `Action(cooling=...)`, calls session step, returns `(obs, reward, terminated, truncated, info)`.

##### 2. `baselines/rule_based.py` (NEW)
- Multi-zone threshold reactive controller.
- If $T_i > T_{\text{safe}} - 3.0$: high cooling ($0.8\text{--}1.0$).
- If $T_i < T_{\text{target}}$: eco-cooling ($0.1\text{--}0.2$).
- Otherwise: proportional ramp.

##### 3. `baselines/pid.py` (NEW)
- Decoupled per-zone PID controller with anti-windup clamping.
- Error signal: $e_i(t) = T_i(t) - T_{\text{target}}$.
- Control signal: $u_i(t) = \text{clip}(K_p e_i(t) + K_i \int e_i dt + K_d \frac{de_i}{dt}, 0.0, 1.0)$.

##### 4. `train_rl.py` & `baselines/rl_agent.py` (NEW)
- Use Stable-Baselines3 (or clean lightweight PPO in PyTorch) to train a continuous policy network (MLP: 64x64 or 128x128).
- Train PPO on the vectorized `core/gym_env.py` environment across task scenarios.
- Save weights to `models/ppo_adctm.zip`.
- `RLAgent` loads trained policy and outputs actions during benchmarking.

##### 5. `run_benchmark.py` (NEW)
- Evaluates:
  1. `Zero Cooling Policy`
  2. `Rule-Based Controller`
  3. `PID Controller`
  4. `LLM Agent / Fallback`
  5. `Trained PPO Policy`
- Runs across all three tasks (`easy`, `medium`, `hard`) over 5 distinct random seeds.
- Records mean and standard deviation for Safety, Target Error, Energy, Jitter, and Final Score.
- Prints a clean Markdown summary table ready to paste directly into `README.md`.

##### 6. `README.md` (MODIFY)
- Replace lines 277–294 with the genuine empirical table generated by `run_benchmark.py`.
- Add exact command line reproduction instructions:
  ```bash
  python run_benchmark.py --episodes 10
  ```

#### Verification & Testing
1. `pytest tests/test_gym_env.py`: Test Gymnasium environment conformance using `gymnasium.utils.env_checker.check_env`.
2. `pytest tests/test_baselines.py`: Validate action shapes, ranges `[0.0, 1.0]`, and state stability for Rule-Based, PID, and PPO.
3. Run `python run_benchmark.py`: Confirm reproducible output table without exceptions.

---

### 2. Fix Plan for C2: Active API Token Committed to Git

#### Overview
Line 3 of `.env` contains an active personal access token: `HF_TOKEN=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`. Furthermore, `.env` is actively tracked in git index (`git ls-files .env` shows `.env`), and `.gitignore` contains unresolved git merge conflict markers (`<<<<<<< HEAD`, `=======`, `>>>>>>>`). Additionally, the absence of `.dockerignore` causes `.env` and sensitive files to be copied into production Docker images via `COPY . .`.

#### Architecture of Solution
```
        [Unsafe: Tracked Secret in Git & Image]
           .env (with live HF_TOKEN) 
               ├──> Tracked in Git
               └──> Baked into Docker Image via "COPY . ."

                         │
                         ▼  REMEDIATION
        [Safe: Zero Secrets in Tree, Template Config]
           1. Remove .env from git tracking (git rm --cached .env)
           2. Create .env.example with dummy placeholders
           3. Sanitize local .env (strip token to template/environment variable)
           4. Fix merge conflicts in .gitignore ensuring .env is ignored
           5. Create .dockerignore excluding .env, .git, models, and caches
           6. User action: Revoke/regenerate the exposed token on HuggingFace
```

#### Detailed File Changes

##### 1. `.env.example` (NEW)
Create a safe template showing required environment variables:
```dotenv
# API Configuration for LLM Inference
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=google/gemma-4-31B-it
HF_TOKEN=your_huggingface_token_here
IMAGE_NAME=""
```

##### 2. `.env` (SANITIZE)
Replace the active token in `.env` with a placeholder:
```dotenv
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=google/gemma-4-31B-it
HF_TOKEN=your_huggingface_token_here
IMAGE_NAME=""
```

##### 3. Git Index Untracking
Execute:
```bash
git rm --cached .env
```
This leaves the file on disk locally for development, but removes it from the git tracking index.

##### 4. `.gitignore` (CLEAN & FIX)
- Remove the unresolved merge conflict markers at lines 1–7 and lines 215–216.
- Ensure `.env`, `.env.*`, and sensitive keys are explicitly listed.
- Also ignore `models/*.zip` (large model binaries) and `.pytest_cache/`.

##### 5. `.dockerignore` (NEW)
Create a `.dockerignore` file to ensure secrets, git history, and bloated caches are never packaged into container images:
```dockerignore
.git
.gitignore
.env
.env.*
__pycache__
*.pyc
.pytest_cache
.venv
venv
models/*.zip
*.log
```

##### 6. User Security Action (CRITICAL)
- The token `hf_VFyWpnqu...` has already been exposed to the git commit history. The user must immediately navigate to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and **delete / revoke** this token.
- Guidance will be provided for optional history rewriting (`git filter-repo` or BFG) if preparing the repo for public presentation.

#### Verification & Testing
1. `git ls-files .env`: Must return empty (not tracked).
2. `git status`: Confirm `.env` is untracked and ignored by `.gitignore`.
3. `grep -rn "hf_VFy" .`: Confirm no file in working tree contains the exposed secret.
4. `test_submission_readiness.py`: Ensure test suite still imports environment cleanly using fallback tokens.

---

### 3. Fix Plan for C3: Evaluation Gaming via Injected TaskConfig in `/reset`

#### Overview
In `core/env.py` (lines 128–132), the `/reset` endpoint accepts an arbitrary dictionary payload:
```python
if config_payload is None:
    session = SimulationSession.from_task_name(selected_task or "easy")
else:
    session = SimulationSession.from_dict(config_payload)
```
This enables an agent or attacker to submit a custom configuration (e.g. `{"safe_temperature": 9999, "max_steps": 1, ...}`) over the network, step once, and retrieve a fabricated 100% score from `/score`. The evaluation boundary has zero integrity.

#### Architecture of Solution
```
[Insecure: Open Config Injection]
Client POST /reset {"safe_temperature": 9999, ...} ──> SimulationSession.from_dict(...) ──> Score = 1.0!

                                   │
                                   ▼  REMEDIATION
[Secure: Strict Whitelisted Canonical Tasks]
Client POST /reset {"task_name": "easy", "seed": 42}
       │
       ▼ Validate against canonical tasks: {"easy", "medium", "hard"}
       ├── If invalid task_name ────────> HTTP 400 "Invalid task_name. Allowed: ['easy', 'medium', 'hard']"
       ├── If arbitrary config keys ────> HTTP 422 "Arbitrary configuration injection is forbidden on evaluation server."
       └── If valid task & optional seed ──> Load canonical tasks/<task_name>.json safely with optional seed
```

#### Detailed File Changes

##### 1. `core/models.py` (MODIFY)
Introduce a strict, validated `ResetPayload` Pydantic model:
```python
class ResetPayload(BaseModel):
    """
    Strict payload for resetting the environment.
    Only permits specifying a canonical task tier and an optional seed.
    Rejects arbitrary physics or threshold overrides.
    """
    task_name: str = Field("easy", description="Benchmark tier: easy, medium, or hard")
    seed: Optional[int] = Field(None, description="Optional episode RNG seed")

    model_config = ConfigDict(extra="forbid")

    @field_validator("task_name")
    @classmethod
    def validate_canonical_task(cls, v: str) -> str:
        clean = v.replace(".json", "").lower()
        if clean not in {"easy", "medium", "hard"}:
            raise ValueError(f"Task '{v}' is not a valid benchmark task. Allowed: ['easy', 'medium', 'hard']")
        return clean
```

##### 2. `core/env.py` (MODIFY lines 98–148)
- Update `/reset` to accept `payload: Optional[ResetPayload]` or query parameters `task_name: Optional[str]`, `seed: Optional[int]`.
- Enforce that sessions are **only** instantiated via canonical task files (`SimulationSession.from_task_name(...)`), with an optional random `seed`.
- Completely delete the insecure `SimulationSession.from_dict(config_payload)` route from the API endpoint.
- If a client provides custom configuration parameters other than `task_name` and `seed`, return HTTP 422.

##### 3. `core/simulator.py` (MODIFY)
- Track `task_name` and `is_canonical: bool` in `SimulationSession`.
- Ensure `get_score()` verifies `self.is_canonical is True`.

#### Verification & Testing
1. **Legitimate Task Reset**: Test `POST /reset` with `{"task_name": "easy"}` and query parameter `?task_name=medium`. Must succeed (HTTP 200).
2. **Rejection of Injected Configuration**: Test `POST /reset` with `{"safe_temperature": 9999}` or `{"max_steps": 1}`. Must return HTTP 422 Unprocessable Entity.
3. **Rejection of Unknown Tasks**: Test `POST /reset` with `{"task_name": "malicious_task"}`. Must return HTTP 422/400.
4. Add automated test `tests/test_api_security.py` verifying these security constraints.

---

### 4. Fix Plan for C4: `get_score()` Returns Perfect 1.0 Score on Zero Steps

#### Overview
In `core/simulator.py` (lines 140–145):
```python
if len(self.history_actions) == 0:
    return {
        "total": 1.0,
        "score": 1.0,
        "metrics": {"safety": 1.0, "precision": 1.0, "efficiency": 1.0, "smoothness": 1.0}
    }
```
If a client or automated grader resets the environment and immediately calls `/score` without executing any steps, the environment awards a perfect $1.0$ (100%) score across all 4 metrics. An aborted or non-functioning agent appears as a perfect controller.

#### Architecture of Solution
```
[Flawed Logic: Immediate 100% Score on Aborted / Zero Rollout]
POST /reset ──> GET /score ──> Returns 1.0 (Safety: 1.0, Precision: 1.0, Energy: 1.0, Smoothness: 1.0)!

                                   │
                                   ▼  REMEDIATION
[Defensible Engineering: Zero Rollout Cannot Be Scored]
Option 1: Explicit HTTP 400 rejection if episode has 0 steps.
Option 2: Return total=0.0 with 0.0 metrics and an explicit status flag "no_steps_executed".

To conform with clean API standards and OpenEnv contracts:
- In `SimulationSession.get_score()`:
    if len(self.history_actions) == 0:
        return {
            "total": 0.0,
            "score": 0.0,
            "metrics": {"safety": 0.0, "precision": 0.0, "efficiency": 0.0, "smoothness": 0.0},
            "status": "uninitialized"
        }
- In `core/env.py` `/score` endpoint:
    If `len(CURRENT_SESSION.history_actions) == 0`:
        raise HTTPException(
            status_code=400, 
            detail="Cannot compute evaluation score: no simulation steps have been executed. Step the environment at least once before requesting /score."
        )
```

#### Detailed File Changes

##### 1. `core/simulator.py` (MODIFY lines 140–145)
Update `get_score()` fallback to assign `0.0` instead of `1.0`:
```python
if len(self.history_actions) == 0:
    return {
        "total": 0.0,
        "score": 0.0,
        "metrics": {"safety": 0.0, "precision": 0.0, "efficiency": 0.0, "smoothness": 0.0},
        "status": "no_steps_executed"
    }
```

##### 2. `core/env.py` (MODIFY line 190 `/score` endpoint)
Ensure `/score` verifies that steps were executed:
```python
@app.get("/score")
def get_score() -> Dict[str, Any]:
    """
    Computes and returns the evaluation score for the active simulation session.
    Requires at least one step to have been executed.
    """
    _ensure_initialized()
    with env_lock:
        if len(CURRENT_SESSION.history_actions) == 0:
            raise HTTPException(
                status_code=400,
                detail="Cannot score an unexecuted session. Take at least one step before requesting /score."
            )
        return CURRENT_SESSION.get_score()
```

#### Verification & Testing
1. In `tests/test_api_security.py`:
   - Add test `test_score_rejects_zero_step_evaluation`: Reset environment, immediately query `GET /score`, assert HTTP 400 Bad Request.
   - Step once with valid action, query `GET /score`, assert HTTP 200 and valid numeric score breakdown.
2. Run pytest suite to ensure no regressions.

---

### 5. Fix Plan for H1: Global Jitter Penalty Bypass Exploit

#### Overview
In `reward/reward_fn.py` (lines 34–39):
```python
if prev_obs.time_step > 0:
    prev_cool = np.array(prev_obs.cooling, dtype=float)
    jitter = float(np.abs(cool - prev_cool).sum())
    if np.max(temps) >= config.safe_temperature - config.jitter_bypass_threshold:
        jitter = 0.0
else:
    jitter = 0.0
```
If **any single zone** exceeds `safe_temperature - jitter_bypass_threshold`, the entire jitter penalty is bypassed globally ($0.0$) across **all zones**.
An intelligent RL agent can exploit this by intentionally letting a single sacrificial zone run hot (just above the threshold), which completely disables the jitter penalty across all remaining zones. The agent can then oscillate cooling wildly on the other zones without any penalty, defeating the engineering purpose of the smoothness regularization.

#### Architecture of Solution
```
[Exploitable Global Bypass]
Zone 1: 84°C (>= 83°C threshold)  ──> Trigger Global Bypass
Zone 2: 60°C (Normal)             ──> Jitter penalty = 0.0 (Can oscillate wildly!)
Zone 3: 60°C (Normal)             ──> Jitter penalty = 0.0 (Can oscillate wildly!)

                               │
                               ▼  REMEDIATION
[Per-Zone Independent Safety Jitter Exemption]
Calculate jitter independently for each zone i:
    diff_i = |cool_i - prev_cool_i|
    if temp_i >= safe_temp - threshold:
        # Zone i is in danger: zero out jitter ONLY for zone i so emergency cooling can actuate freely
        jitter_i = 0.0
    else:
        # Zone i is in nominal state: smooth control is strictly enforced
        jitter_i = diff_i

Total jitter penalty = sum(jitter_i)
```

#### Detailed File Changes

##### 1. `reward/reward_fn.py` (MODIFY lines 34–40)
Replace the scalar `np.max(temps)` global check with a vectorized per-zone mask:
```python
    if prev_obs.time_step > 0:
        prev_cool = np.array(prev_obs.cooling, dtype=float)
        zone_jitters = np.abs(cool - prev_cool)
        # Bypassed only for zones that are individually near or exceeding the safety threshold
        danger_threshold = config.safe_temperature - config.jitter_bypass_threshold
        danger_mask = temps >= danger_threshold
        zone_jitters = np.where(danger_mask, 0.0, zone_jitters)
        jitter = float(zone_jitters.sum())
    else:
        jitter = 0.0
```

##### 2. `tests/reward/test_reward_fn.py` (UPDATE & ADD)
- Update `test_compute_reward_bypasses_jitter_near_safe_limit` to verify that when all zones are hot, jitter is zeroed.
- Add `test_compute_reward_jitter_bypass_is_per_zone`:
  - Construct a scenario where Zone 0 is hot ($\ge 83^\circ\text{C}$), while Zone 1 is cool ($60^\circ\text{C}$).
  - Change cooling by $0.5$ on both zones.
  - Assert that Zone 0 pays $0.0$ jitter, but Zone 1 pays $0.5$ jitter, confirming that hot zones do not grant immunity to cool zones.

#### Verification & Testing
1. Run `pytest tests/reward/test_reward_fn.py -v`.
2. Confirm per-zone jitter isolation prevents sacrificial zone reward hacking.
3. Run full regression suite to ensure no unexpected breaking changes.

---

### 6. Fix Plan for H2: Evaluation Metric Heavily Rewards Total Inaction

#### Overview
In `grader/evaluator.py` (lines 22–23) and `grader/metrics.py`:
```python
energy_score = 1.0 - avg_energy  # avg_energy in [0, 1]
jitter_score = 1.0 - avg_jitter  # avg_jitter in [0, 1]

final_score = 0.4 * safety + 0.3 * target_score + 0.2 * energy_score + 0.1 * jitter_score
```
A completely dead agent applying `cooling = [0.0] * N` automatically scores `1.0` in Energy ($w=0.2$) and `1.0` in Smoothness ($w=0.1$). This establishes an artificial $0.30$ (30%) score floor for an inactive or crashed controller. In fact, if the thermal trajectory starts slightly below safe limits (as in `easy.json`), a do-nothing policy gets a substantial passing score ($\sim 0.535$ on Easy) despite letting the entire datacenter overheat.

#### Architecture of Solution
```
[Flawed Metric: Energy & Smoothness Awarded Regardless of Safety]
Inaction (Cooling = 0.0) ──> All Servers Overheat!
                      ──> BUT: Energy Score = 1.0 (20%)
                      ──> AND: Jitter Score = 1.0 (10%)
                      ──> Total Score >= 0.30 - 0.53 (Free Points for Hardware Damage!)

                               │
                               ▼  REMEDIATION
[Safety-Gated Efficiency & Penalty for Catastrophic Overheating]
In an industrial data center, efficiency and smoothness only matter if the system operates SAFELY.
Saving electricity while frying servers is not efficient; it is catastrophic infrastructure failure.

Formulation:
1. Gated Efficiency:
   energy_score is scaled by thermal health. If safety is violated, the energy score is penalized proportionally:
   effective_energy_score = (1.0 - avg_energy) * (safety_ratio ** 0.5)
   
2. Gated Smoothness:
   jitter_score = (1.0 - avg_jitter) * (safety_ratio ** 0.5)

3. Consequence:
   - For an agent maintaining 100% safety (safety_ratio = 1.0):
     Both energy_score and jitter_score remain completely unchanged from the original formulation.
   - For a dead agent that lets servers burn (safety_ratio -> 0):
     Both energy and jitter scores decay to 0, eliminating the unearned 30% bonus.
```

#### Detailed File Changes

##### 1. `grader/evaluator.py` (MODIFY lines 20–40)
Update component score calculation to gate efficiency and smoothness on safety integrity:
```python
    # Normalize components to [0,1]
    raw_energy_score = 1.0 - energy
    raw_jitter_score = 1.0 - jitter
    target_score = max(0.0, min(1.0, 1.0 - target_error))

    # Safety gating: in industrial control, energy efficiency and smoothness only count
    # if safety constraints are respected. Burning servers with 0 cooling is not "efficient".
    safety_factor = float(np.sqrt(max(0.0, safety)))
    energy_score = raw_energy_score * safety_factor
    jitter_score = raw_jitter_score * safety_factor

    # Weighted sum
    w_safety = 0.4
    w_target = 0.3
    w_energy = 0.2
    w_jitter = 0.1

    final_score = (
        w_safety * safety +
        w_target * target_score +
        w_energy * energy_score +
        w_jitter * jitter_score
    )
```

##### 2. `tests/grader/test_evaluator.py` & `tests/test_submission_readiness.py`
- Verify that a nominal controller achieving safety = 1.0 receives identical scores.
- Verify that a zero policy (`cooling=0.0`) on tasks where overheating occurs drops significantly in score.
- Tighten test thresholds in `test_zero_policy_is_weak_on_active_tasks` to reflect that inaction is penalized properly.

#### Verification & Testing
1. Run `run_benchmark.py`: Confirm `Zero (Passive)` overall score drops from ~0.44 down to realistic failing range (~0.15–0.25).
2. Confirm nominal controllers (`Rule-Based`, `PID`, `LLM`, `PPO`) maintaining high safety scores are unaffected or accurately evaluated.
3. Run full regression test suite.

---

### 7. Fix Plan for H3: Absence of Physical Upper Temperature Bound

#### Overview
In `dynamics/thermal_model.py` (lines 35–41):
```python
delta_t = ALPHA * workloads - cooling_effect * cooling + GAMMA * (ambient_temp - temperatures)
next_temperatures = np.maximum(temperatures + delta_t, ambient_temp - EPSILON)
next_temperatures = np.maximum(next_temperatures, ambient_temp)
```
While temperatures are clamped at a lower bound (`ambient_temp`), there is **no physical upper ceiling or hardware burnout/throttling model**. Under sustained high workload with zero cooling, temperature mathematically rises past $120^\circ\text{C}\text{--}150^\circ\text{C}+$ without triggering equipment damage, thermal throttling, or an emergency cutoff. In real server hardware, semiconductor silicon throttles at $\sim 85^\circ\text{C}$ and suffers irreversible thermal runaway / shutdown around $100\text{--}105^\circ\text{C}$. Allowing temperatures to climb infinitely contradicts the README's claim of "industrial realism and high-fidelity thermodynamics."

#### Architecture of Solution
```
[Unbounded Linear Spike]
Workload = 1.0, Cooling = 0.0 ──> Temperature rises to 120°C, 150°C, 200°C... (Unphysical!)

                               │
                               ▼  REMEDIATION
[Realistic Thermodynamic Ceiling & Hardware Burnout Model]
1. Define PHYSICAL_MAX_TEMP = 105.0°C (Silicium thermal junction breakdown / T_jmax).
2. Clamping:
   next_temperatures = np.clip(temperatures + delta_t, ambient_temp, PHYSICAL_MAX_TEMP)
3. Emergency Meltdown Info Flag:
   If any zone reaches PHYSICAL_MAX_TEMP, register an info flag:
   `"hardware_meltdown": True` or `"thermal_runaway": True`.
4. Consistent Configuration:
   Expose `max_temperature: float = 105.0` in `config/constants.py` and optionally in `TaskConfig` for configurability.
```

#### Detailed File Changes

##### 1. `config/constants.py` (MODIFY)
Add physical boundary constants:
```python
ALPHA = 7.5
BETA = 8.0
GAMMA = 0.1
MAX_PHYSICAL_TEMPERATURE = 105.0  # Silicon junction failure threshold (°C)
```

##### 2. `dynamics/thermal_model.py` (MODIFY lines 35–45)
Clamp next temperatures between `ambient_temp` and `MAX_PHYSICAL_TEMPERATURE`:
```python
    delta_t = ALPHA * workloads - cooling_effect * cooling + GAMMA * (ambient_temp - temperatures)
    
    # Floor at ambient temperature and ceiling at maximum physical temperature
    next_temperatures = np.clip(temperatures + delta_t, ambient_temp, MAX_PHYSICAL_TEMPERATURE)
```

##### 3. `tests/dynamics/test_thermal_model.py` (ADD)
Add a test `test_apply_transition_enforces_upper_temperature_bound`:
- Feed in initial temperatures of $104^\circ\text{C}$, workload $1.0$, cooling $0.0$.
- Assert that resulting temperatures do not exceed $105.0^\circ\text{C}$.

#### Verification & Testing
1. Run `pytest tests/dynamics/test_thermal_model.py -v`.
2. Confirm temperature never exceeds $105^\circ\text{C}$ under worst-case adversarial inputs (workload=1.0, cooling=0.0 over 100 steps).
3. Run full test suite to guarantee zero regression on existing tasks.

**Status**: ✅ **Completed & Verified**
- Clamped maximum temperature to `MAX_PHYSICAL_TEMPERATURE = 105.0` in `config/constants.py` and `dynamics/thermal_model.py`.
- Added unit test in `tests/dynamics/test_thermal_model.py`.
- Full regression suite passed (34/34 tests passing).

---

### 8. Fix Plan for H4: Zero Inter-Zone Spatial Thermal Diffusion

#### Overview
In `dynamics/thermal_model.py` (line 35):
```python
delta_t = ALPHA * workloads - cooling_effect * cooling + GAMMA * (ambient_temp - temperatures)
```
Every term in `delta_t` is evaluated purely element-wise across the zones. There is zero heat conduction, convection, or thermal coupling between adjacent physical zones ($i-1, i, i+1$). 

In a real physical data center server rack row:
- Heat naturally dissipates from hotter server zones to cooler neighboring zones via thermal conduction through server chassis and air convection in the hot/cold aisles:
$$\frac{dT_i}{dt} = \alpha W_i - \beta C_i + \gamma (T_{\text{ambient}} - T_i) + \kappa (T_{i-1} - 2T_i + T_{i+1})$$
where $\kappa$ is the inter-zone thermal diffusion coefficient.
- Without this term, the simulation is merely $N$ completely independent, uncoupled single-zone problems running in a parallel loop. The controller does not need to learn spatial trade-offs, heat spreading, or cooperative zone cooling.

#### Architecture of Solution
```
[Before: Independent 1D Zones]
Zone 0 (95°C)  |  Zone 1 (30°C)  |  Zone 2 (95°C)
      │                 │                 │
   No heat           No heat           No heat
  diffusion         diffusion         diffusion

                               │
                               ▼  REMEDIATION
[After: Spatial Thermal Diffusion]
Zone 0 (95°C)  <──heat──>  Zone 1 (30°C)  <──heat──>  Zone 2 (95°C)
Heat diffuses from hotter zones into cooler adjacent zones:
diff_i = KAPPA_DIFFUSION * ((T_{i-1} - T_i) + (T_{i+1} - T_i))
```

1. **Diffusion Constant**:
   Define `KAPPA_DIFFUSION = 0.05` in `config/constants.py` (physically realistic moderate coupling between adjacent rack bays).
2. **Boundary Conditions**:
   For linear zone arrays ($i = 0, \dots, N-1$), use standard Neumann / isolated boundary conditions (outer end walls):
   - For $i=0$: neighbor is only $i=1$. Diffusion term: $\kappa (T_1 - T_0)$.
   - For $i=N-1$: neighbor is only $i=N-2$. Diffusion term: $\kappa (T_{N-2} - T_{N-1})$.
   - For $0 < i < N-1$: neighbors are $i-1$ and $i+1$. Diffusion term: $\kappa (T_{i-1} + T_{i+1} - 2T_i)$.
   - In vectorized NumPy:
     ```python
     diffusion = np.zeros_like(temperatures)
     if len(temperatures) > 1:
         diffusion[0] = temperatures[1] - temperatures[0]
         diffusion[-1] = temperatures[-2] - temperatures[-1]
         if len(temperatures) > 2:
             diffusion[1:-1] = temperatures[:-2] + temperatures[2:] - 2.0 * temperatures[1:-1]
         diffusion = KAPPA_DIFFUSION * diffusion
     ```
3. **Integration into Physics**:
   ```python
   delta_t = (
       ALPHA * workloads 
       - cooling_effect * cooling 
       + GAMMA * (ambient_temp - temperatures)
       + diffusion
   )
   ```

#### Detailed File Changes

##### 1. `config/constants.py` (MODIFY)
Add inter-zone diffusion coefficient:
```python
KAPPA_DIFFUSION = 0.05  # Inter-zone thermal conductivity / diffusion coupling
```

##### 2. `dynamics/thermal_model.py` (MODIFY)
Compute the discrete spatial Laplacian diffusion vector and incorporate it into `delta_t`.

##### 3. `tests/dynamics/test_thermal_model.py` (ADD)
Add tests:
- `test_apply_transition_diffuses_heat_between_adjacent_zones`:
  - Setup 3 zones: Zone 0 at 90°C, Zone 1 at 30°C, Zone 2 at 30°C, all workloads 0, cooling 0, ambient 30°C.
  - Assert that Zone 0 cools down faster due to heat leaking into Zone 1, and Zone 1 heats up above ambient purely due to diffusion from Zone 0.
  - Assert that Zone 2 (not adjacent to Zone 0) remains cooler than Zone 1.

#### Verification & Testing
1. Run `pytest tests/dynamics/test_thermal_model.py -v`.
2. Run full regression test suite (`tests/test_submission_readiness.py`, `tests/test_baselines.py`, `tests/test_api_security.py`, etc.).
3. Run `python run_benchmark.py` to ensure benchmark controllers adapt cleanly to realistic multi-zone spatial coupling.

**Status**: ✅ **Completed & Verified**
- Defined `KAPPA_DIFFUSION = 0.05` in `config/constants.py`.
- Integrated 1D discrete Laplacian thermal diffusion with Neumann boundaries into `dynamics/thermal_model.py`.
- Added unit test `test_apply_transition_diffuses_heat_between_adjacent_zones` in `tests/dynamics/test_thermal_model.py`.
- Full regression suite passed (35/35 tests passing).

---

### 9. Fix Plan for H5: `/simulate` Endpoint Fails with AttributeError

#### Overview
In `core/env.py` (lines 214–215):
```python
# Directly call the reset function
initial_observation = reset(task_name=task_name)
session_config = initial_observation.model_dump() # Convert Observation to dict for consistency
```
`reset()` in `core/env.py` returns `session.observation.model_dump()`, which is already a Python `dict`.
Calling `.model_dump()` on a `dict` raises `AttributeError: 'dict' object has no attribute 'model_dump'`, causing `POST /simulate` to crash with HTTP 400/500 immediately.

Furthermore:
In `core/env.py` lines 228–230:
```python
temp_session = SimulationSession.from_task_name(task_name)
num_zones = temp_session.config.num_zones
```
`CURRENT_SESSION` is already instantiated by `reset()`, with its configuration and number of zones cleanly accessible in `CURRENT_SESSION.config`. Re-instantiating a disconnected `temp_session` is redundant and can cause drift if seeds or parameters vary.

#### Architecture of Solution
```
[Before: Broken .model_dump() on Dict]
Client POST /simulate {"task_name": "easy"}
  └── reset(task_name=task_name) ──> returns dict
        └── dict.model_dump() ──> AttributeError CRASH (HTTP 400/500)

                               │
                               ▼  REMEDIATION
[After: Clean Dict Handling & Direct Session Access]
Client POST /simulate {"task_name": "easy"}
  └── initial_obs = reset(task_name=task_name)  # already dict
  └── num_zones = len(initial_obs["temperatures"])
  └── Step rollout with valid action payload
  └── evaluate_trajectory with CURRENT_SESSION.config
  └── Returns clean simulation summary JSON (HTTP 200)
```

#### Detailed File Changes

##### 1. `core/env.py` (MODIFY lines 206–270)
Refactor `simulate`:
```python
@app.post("/simulate")
def simulate(task_name: str = "easy", cooling_level: float = 0.4) -> Dict[str, Any]:
    """
    Runs a full simulation from start to finish using a fixed cooling policy.
    Returns the final trajectory grade and performance metrics.
    """
    try:
        initial_observation = reset(task_name=task_name)
        if isinstance(initial_observation, dict):
            obs_dict = initial_observation
        else:
            obs_dict = initial_observation.model_dump()
            
        with env_lock:
            config = CURRENT_SESSION.config
            num_zones = config.num_zones

    except HTTPException as exc:
        raise HTTPException(status_code=exc.status_code, detail=f"Error resetting environment: {exc.detail}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error loading task configuration: {str(exc)}")

    observations = [obs_dict]
    actions = []
    total_reward = 0.0
    done = False
    
    while not done:
        action = {"cooling": [cooling_level] * num_zones}
        try:
            step_result = step(action)
        except HTTPException as exc:
            raise HTTPException(status_code=exc.status_code, detail=f"Error stepping environment: {exc.detail}")

        observations.append(step_result["observation"])
        actions.append(action)
        total_reward += step_result["reward"]["value"]
        done = step_result["done"]

    score = evaluate_trajectory(observations, actions, config)

    return {
        "task": task_name,
        "steps": len(actions),
        "total_reward": total_reward,
        "score": score,
        "status": "completed"
    }
```

##### 2. `tests/test_api_security.py` (ADD)
Add integration test `test_simulate_endpoint_executes_successfully`:
- Call `client.post("/simulate?task_name=easy&cooling_level=0.5")`.
- Assert response status is 200.
- Assert response contains `{"task": "easy", "status": "completed"}` and valid numerical `"score"`.

#### Verification & Testing
1. Run `pytest tests/test_api_security.py -v`.
2. Run full regression test suite.

**Status**: ✅ **Completed & Verified**
- Refactored `simulate()` in `core/env.py` to handle `initial_observation` without dict attribute error.
- Direct access to `CURRENT_SESSION.config` under `env_lock`.
- Handled query defaults robustly without parameter validation rejections.
- Added unit test `test_simulate_endpoint_executes_successfully` in `tests/test_api_security.py`.
- Full regression suite passed (36/36 tests passing).

---

### 10. Fix Plan for H6: Hard Benchmark Scenario Starts in Immediate Violation

#### Overview
In `tasks/hard.json`:
```json
{
  "num_zones": 8,
  "initial_temperatures": [69.0, 70.0, 70.5, 69.5, 71.0, 69.5, 70.5, 69.0],
  "safe_temperature": 70.5,
  ...
}
```
Notice Zone 4 is initialized to $71.0^\circ\text{C}$, and Zones 2 and 6 are at $70.5^\circ\text{C}$, while `safe_temperature` is $70.5^\circ\text{C}$.
In `grader/metrics.py`:
```python
violations = sum(1 for obs in observations if any(t > config.safe_temperature for t in obs["temperatures"]))
```
Because the initial observation at step 0 already has Zone 4 at $71.0^\circ\text{C}$, `violations` is guaranteed to be $\ge 1$ from the moment of initialization, before any controller or policy has had a single step to act!
This makes it **mathematically impossible** for any agent (human, classical, or RL) to achieve 100% safety on `hard.json`.

#### Architecture of Solution
```
[Unfair Initial Violation]
hard.json: safe_temperature = 70.5°C, initial_temperatures = [..., 71.0, ...]
  └── Step 0 observation: Zone 4 = 71.0°C > 70.5°C
        └── 1 violation recorded BEFORE step 1!
        └── Safety score can never reach 1.0.

                               │
                               ▼  REMEDIATION
[Fair Critical Stress: Extreme Closeness to Safe Threshold Without Pre-Action Violation]
hard.json: safe_temperature = 70.5°C
Adjust initial temperatures so maximum initial temperature is 70.4°C:
"initial_temperatures": [69.0, 70.0, 70.3, 69.5, 70.4, 69.5, 70.2, 69.0]
  └── Maximum initial temperature is 70.4°C (< 70.5°C)
  └── Extreme stress: 0.1°C margin from failure under 0.98 initial workload!
  └── A fast, intelligent controller can immediately apply cooling=1.0 and prevent violation.
  └── A 100% safety score is now achievable by a perfect controller.
```

#### Detailed File Changes

##### 1. `tasks/hard.json` (MODIFY lines 3)
Change:
```json
"initial_temperatures": [69.0, 70.0, 70.3, 69.5, 70.4, 69.5, 70.2, 69.0],
```

##### 2. `tests/test_submission_readiness.py` (ADD)
Add test `test_canonical_tasks_start_in_non_violating_state`:
- Iterate over all canonical tasks (`easy.json`, `medium.json`, `hard.json`).
- Verify that for every zone $z$, $T_{z, 0} \le T_{\text{safe}}$.
- Guarantee that no task starts in a pre-step unpreventable violation.

#### Verification & Testing
1. Run `pytest tests/test_submission_readiness.py -v`.
2. Run full regression test suite (`tests/test_submission_readiness.py`, `tests/test_baselines.py`, `tests/test_api_security.py`, `tests/reward/test_reward_fn.py`, `tests/grader/test_evaluator.py`, `tests/dynamics/test_thermal_model.py`).
3. Run `python run_benchmark.py` and inspect hard task performance across controllers.

**Status**: ✅ **Completed & Verified**
- Adjusted `tasks/hard.json` initial temperatures to peak at $70.4^\circ\text{C}$ ($< 70.5^\circ\text{C}$).
- Added invariant test `test_canonical_tasks_start_in_non_violating_state` in `tests/test_submission_readiness.py`.
- Full regression suite passed (37/37 tests passing). All 6 High Severity issues (H1–H6) are now completely resolved.

---

### 11. Fix Plan for M1: Global `CURRENT_SESSION` Without Multi-Tenant State Isolation

#### Overview
In `core/env.py` (line 46):
```python
CURRENT_SESSION: Optional[SimulationSession] = None
env_lock = threading.Lock()
```
The server relies on a single global variable `CURRENT_SESSION`. While `env_lock` prevents thread collision during the execution of a single step, any incoming `/reset` call overwrites `CURRENT_SESSION` globally.
If multiple agents or benchmark runners connect simultaneously (e.g. concurrent evaluation threads, parallel inference runners, or multi-tenant benchmarking):
- Client A resets to `hard`.
- Client B resets to `easy`.
- Client A steps, thinking it is running `hard`, but its action is applied to `easy`!

#### Architecture of Solution
```
[Single Shared Global Session]
Client A ──> POST /reset (hard) ──> CURRENT_SESSION = hard
Client B ──> POST /reset (easy) ──> CURRENT_SESSION = easy (Overwrites Client A!)
Client A ──> POST /step ──────────> Steps Client B's session! State corrupted!

                               │
                               ▼  REMEDIATION
[Multi-Tenant Session Registry with Backward Compatibility]
ACTIVE_SESSIONS: Dict[str, SimulationSession] = {}
DEFAULT_SESSION_ID = "default"

1. POST /reset:
   - Accepts optional header / param `X-Session-ID` or `session_id`.
   - If not provided, defaults to DEFAULT_SESSION_ID (100% backward compatible with single-client root submission & test suite).
   - Generates/assigns session_id, stores in `ACTIVE_SESSIONS[session_id]`.
   - Returns observation dict with `session_id` included in response or headers.

2. POST /step, GET /state, GET /score:
   - Looks up session by `session_id` (from query param, header `X-Session-ID`, or defaults to DEFAULT_SESSION_ID).
   - Operates on isolated `SimulationSession` instance.
   - Raises HTTP 404/400 if specific `session_id` does not exist.

3. Concurrency Protection:
   - Fine-grained per-session lock or dictionary lock to allow concurrent independent simulations.
```

#### Detailed File Changes

##### 1. `core/env.py` (MODIFY)
- Implement `SESSIONS: Dict[str, SimulationSession] = {}` and `session_locks: Dict[str, threading.Lock] = {}`.
- Maintain `DEFAULT_SESSION_ID = "default"` for zero-configuration compatibility with existing scripts (`sample_run.py`, `tests/test_submission_readiness.py`).
- Update `_get_session(session_id: str)` to safely retrieve active sessions.
- In `/reset`, `/step`, `/state`, `/score`, `/simulate`, accept optional `session_id: Optional[str] = Query(None)` and header `x_session_id: Optional[str] = Header(None)`.
- If `session_id` is provided, use it; otherwise fallback to `DEFAULT_SESSION_ID`.

##### 2. `tests/test_api_security.py` (ADD)
Add concurrency and isolation test `test_multi_tenant_session_isolation`:
- Reset Session 1 to `easy` with ID `"client-1"`.
- Reset Session 2 to `hard` with ID `"client-2"`.
- Step Session 1 with 3-zone cooling action `[0.3, 0.3, 0.3]`.
- Step Session 2 with 8-zone cooling action `[0.8] * 8`.
- Verify Session 1 has `num_zones == 3` and step counter 1, completely unaffected by Session 2 (`num_zones == 8`).

#### Verification & Testing
1. Run `pytest tests/test_api_security.py -v`.
2. Run full regression test suite (`37/37+` tests).

**Status**: ✅ **Completed & Verified**
- Implemented `ACTIVE_SESSIONS` multi-tenant dictionary registry with per-session thread locks in `core/env.py`.
- Preserved single-tenant fallback to `DEFAULT_SESSION_ID = "default"`.
- Added multi-tenant concurrent state isolation unit test `test_multi_tenant_session_isolation` in `tests/test_api_security.py`.
- Full regression suite passed (38/38 tests passing).

---

### 12. Fix Plan for M2: Observation Space Lacks Goal Parameters (Non-Markovian State)

#### Overview
In `core/models.py`:
```python
class Observation(BaseModel):
    temperatures: List[float] = Field(..., min_length=1)  # °C per zone
    workloads: List[float] = Field(..., min_length=1)     # normalized workload index [0,1] per zone
    cooling: List[float] = Field(..., min_length=1)       # last applied cooling level tracking buffer
    ambient_temp: float = Field(..., description="Ambient temperature in Celsius")
    time_step: int = Field(..., ge=0)
```
The raw HTTP / API observation model does **not** include `target_temperature` or `safe_temperature`. 
An autonomous RL agent, LLM agent, or remote client receiving this observation payload does not know:
1. What temperature it is supposed to regulate toward (`target_temperature`).
2. What critical threshold triggers equipment destruction (`safe_temperature`).

To know the goals, an agent is currently forced to read local benchmark JSON files from disk out-of-band. For a truly autonomous OpenEnv environment running remotely over HTTP, this makes the observation non-Markovian and breaks remote agent autonomy.

#### Architecture of Solution
```
[Before: Blind Agent]
Observation ──> {temps, workloads, cooling, ambient, step}
                  └── Missing: target_temperature? safe_temperature?
                  └── Agent must peek into disk JSON config!

                               │
                               ▼  REMEDIATION
[After: Goal-Conditioned Markovian Observation with Backward Compatibility]
Observation ──> {
  temps, workloads, cooling, ambient, step,
  target_temperature: Optional[float] = None,  # Backward compatible default
  safe_temperature: Optional[float] = None     # Backward compatible default
}
1. In `core/models.py`: Add `target_temperature: Optional[float] = None` and `safe_temperature: Optional[float] = None`.
2. In `core/simulator.py`: Populate `target_temperature=config.target_temperature` and `safe_temperature=config.safe_temperature` during session initialization and step transitions.
3. In `dynamics/thermal_model.py`: Forward `target_temperature` and `safe_temperature` from the previous observation or task config to the next observation.
4. Compatibility: Since default values are `None`, existing test fixtures creating synthetic `Observation(...)` without these keys continue to pass with 0 breakages.
```

#### Detailed File Changes

##### 1. `core/models.py` (MODIFY)
Add optional goal fields to `Observation`:
```python
class Observation(BaseModel):
    temperatures: List[float] = Field(..., min_length=1)  # °C per zone
    workloads: List[float] = Field(..., min_length=1)     # normalized workload index [0,1] per zone
    cooling: List[float] = Field(..., min_length=1)       # last applied cooling level tracking buffer
    ambient_temp: float = Field(..., description="Ambient temperature in Celsius")
    time_step: int = Field(..., ge=0)
    target_temperature: Optional[float] = Field(None, description="Regulation setpoint goal (°C)")
    safe_temperature: Optional[float] = Field(None, description="Critical upper safety threshold (°C)")
```

##### 2. `core/simulator.py` (MODIFY)
Pass `target_temperature=config.target_temperature` and `safe_temperature=config.safe_temperature` in `SimulationSession.__init__`.

##### 3. `dynamics/thermal_model.py` (MODIFY)
Propagate `target_temperature=config.target_temperature` and `safe_temperature=config.safe_temperature` in `apply_transition`.

##### 4. `tests/test_submission_readiness.py` (ADD)
Verify that `/reset` and `/step` return observations containing `target_temperature` and `safe_temperature`.

#### Verification & Testing
1. Run `pytest tests/test_submission_readiness.py -v`.
2. Run full regression suite (`tests/test_baselines.py`, `tests/test_api_security.py`, `tests/reward/test_reward_fn.py`, `tests/grader/test_evaluator.py`, `tests/dynamics/test_thermal_model.py`).
3. Confirm all baseline agents continue to function smoothly.

**Status**: ✅ **Completed & Verified**
- Added `target_temperature` and `safe_temperature` to `Observation` in `core/models.py`.
- Populated goal parameters across `core/simulator.py` and `dynamics/thermal_model.py`.
- Verified in `tests/test_submission_readiness.py`.
- Full regression suite passed (38/38 tests passing).

---

### 13. Fix Plan for M3: Lack of Multi-Seed Statistical Evaluation

#### Overview
In standard evaluation scripts (`sample_run.py`), evaluation is performed on exactly one fixed seed per tier (`101`, `202`, `303`).
A single deterministic seed cannot measure:
1. Controller robustness against volatility.
2. Performance variance / standard deviations across initial temperature perturbations.
3. Statistically rigorous confidence intervals.

In C1, we built `run_benchmark.py` which evaluates across multiple seeds (`[42, 101, 202, 303, 404]`). To complete remediation of M3:
1. Extend `sample_run.py` to support multi-seed evaluation via CLI argument `--seeds` (e.g. `--seeds 42 101 202` or `--num-seeds 5`).
2. Aggregate mean and standard deviation scores in the terminal Rich table summary.
3. Add a dedicated test in `tests/test_baselines.py` verifying that multi-seed evaluation produces accurate statistics.

#### Architecture of Solution
```
[Single Seed Evaluation]
sample_run.py ──> Seed 101 only ──> Single point score (High variance / risk of overfit)

                               │
                               ▼  REMEDIATION
[Statistical Multi-Seed Evaluation]
sample_run.py --seeds 42 101 202 303 404 (or --num-seeds 3)
  └── Evaluates each tier across N distinct seeds via POST /reset {"seed": s}
  └── Computes Mean Score ± Std Dev
  └── Renders statistical summary table in Rich console
```

#### Detailed File Changes

##### 1. `sample_run.py` (MODIFY)
- Add `argparse` support to `main()` with `--seeds` (list of ints) and `--num-seeds` (int).
- If `--seeds` or `--num-seeds` is specified, execute each task over the seed set, computing `mean` and `std`.
- Update `print_final_summary` to optionally display `mean ± std`.

##### 2. `tests/test_baselines.py` (ADD)
Add test `test_multi_seed_evaluation_computes_statistics`:
- Call `evaluate_agent_on_task(RuleBasedController(), "easy", seeds=[42, 101, 202])`.
- Assert output contains `mean`, `std`, and length of seeds equals 3.

#### Verification & Testing
1. Run `pytest tests/test_baselines.py -v`.
2. Run `python sample_run.py --help` to confirm CLI flags.
3. Run full regression test suite.

**Status**: ✅ **Completed & Verified**
- Extended `sample_run.py` with multi-seed CLI parsing (`--seeds`, `--num-seeds`, `--task`) and mean/std stats computation.
- Added unit test `test_multi_seed_evaluation_computes_statistics` in `tests/test_baselines.py`.
- Full regression suite passed (39/39 tests passing).

---

### 14. Fix Plan for M4: Dependency Specification Conflicts and Missing Locks

#### Overview
1. `requirements.txt` specifies `openenv-core` and includes `pytest` and `ollama`.
2. `pyproject.toml` specifies `openenv` (conflicting with `openenv-core` in requirements.txt), omits `pytest`, omits `ollama`, and leaves versions loosely constrained (`numpy>=2.0.0`, `pydantic>=2.0.0`).
3. Furthermore, modern packaging standards recommend pinning exact or properly bounded ranges and optional test dependencies under `[project.optional-dependencies] test = [...]`.

#### Architecture of Solution
```
[Inconsistent Dependencies]
requirements.txt ──> openenv-core, pytest, ollama, loosely bounded versions
pyproject.toml   ──> openenv (mismatch!), missing pytest, missing optional-deps

                               │
                               ▼  REMEDIATION
[Harmonized, Clean Dependency Specifications]
1. Harmonize package names: Use `openenv-core` consistently across both files.
2. In `pyproject.toml`:
   - Add `[project.optional-dependencies]` with `test = ["pytest>=8.0.0"]`.
   - Include `ollama>=0.3.0` for local LLM evaluation parity with `requirements.txt`.
   - Harmonize `openenv-core>=0.1.0`.
3. In `requirements.txt`:
   - Ensure clean consistency with `pyproject.toml`.
```

#### Detailed File Changes

##### 1. `pyproject.toml` (MODIFY)
Harmonize dependencies:
```toml
dependencies = [
    "fastapi==0.112.0",
    "uvicorn[standard]==0.30.1",
    "pydantic>=2.0.0,<3.0.0",
    "numpy>=1.26.0",
    "openai>=1.0.0",
    "requests==2.32.3",
    "openenv-core",
    "python-dotenv>=1.0.0",
    "rich>=13.0.0",
    "ollama"
]

[project.optional-dependencies]
test = [
    "pytest>=8.0.0",
    "pytest-cov"
]
```

##### 2. `requirements.txt` (MODIFY)
Keep synchronized with `pyproject.toml`.

#### Verification & Testing
1. Run `pip check` or import checks.
2. Run `pytest tests/test_submission_readiness.py -v`.
3. Run full regression test suite.

**Status**: ✅ **Completed & Verified**
- Synchronized `pyproject.toml` dependencies with `openenv-core` and added `[project.optional-dependencies] test`.
- Full regression suite passed (39/39 tests passing).

---

### 15. Fix Plan for M5: Oversimplified Flawed Trend Prediction Mathematics

#### Overview
In `analysis/trend_predictor.py` (lines 13–20):
```python
# Calculate average velocity over the window
deltas = []
for i in range(1, window):
    prev = history[-(i+1)][z_idx]
    curr = history[-i][z_idx]
    deltas.append(curr - prev)
    
avg_velocity = sum(deltas) / len(deltas)
```
Notice what happens when summing adjacent differences of a discrete series:
$$\sum_{i=1}^{k} (x_i - x_{i-1}) = x_k - x_0$$
All intermediate temperature readings algebraically cancel out!
Dividing by $(k-1)$ reduces the "average velocity" purely to $\frac{x_k - x_0}{k-1}$, rendering the predictor mathematically equivalent to drawing a single chord between the first and last point of the window. Any non-linear thermal acceleration, sudden spike in the middle of the window, or recent deceleration is completely ignored.

#### Architecture of Solution
```
[Telescoping Cancellation: Chord Only]
History = [70, 75, 82, 84, 85]
Sum of deltas cancels intermediate points: (85 - 70) / 4 = 3.75
Blind to curvature or exponential runaway!

                               │
                               ▼  REMEDIATION
[Ordinary Least Squares (OLS) Linear Regression / Robust Trend Estimation]
Given window observations y = [y_0, y_1, ..., y_{k-1}] at time steps x = [0, 1, ..., k-1]:
Slope (velocity) via closed-form OLS:
  beta = Cov(x, y) / Var(x) = sum((x_i - x_mean) * (y_i - y_mean)) / sum((x_i - x_mean)^2)
1. Accounts for every intermediate observation with statistical weighting.
2. Outlier-resistant and captures real trajectory acceleration.
3. If slope > 0, project steps to critical threshold: (safe_temp - current_temp) / slope.
```

#### Detailed File Changes

##### 1. `analysis/trend_predictor.py` (MODIFY)
Implement OLS closed-form slope estimation across the rolling window:
```python
import numpy as np
from typing import List

def predict_thermal_future(
    temps: List[float], 
    history: List[List[float]], 
    safe_temp: float, 
    window: int = 5
) -> List[str]:
    """
    Predicts time-to-violation for each zone based on robust rolling linear regression.
    Uses OLS slope over the rolling history window to incorporate all intermediate points.
    """
    num_zones = len(temps)
    if len(history) < window:
        return ["Initializing..." for _ in temps]
        
    recent_history = np.array(history[-window:])  # shape: (window, num_zones)
    x = np.arange(window)
    x_mean = np.mean(x)
    x_var = np.sum((x - x_mean) ** 2)

    forecasts = []
    for z_idx in range(num_zones):
        current_temp = temps[z_idx]
        y = recent_history[:, z_idx]
        y_mean = np.mean(y)
        
        # Closed-form OLS slope (velocity in °C/step)
        velocity = np.sum((x - x_mean) * (y - y_mean)) / x_var
        
        if velocity <= 0.05:  # Flat or cooling
            forecasts.append("Stable/Cooling")
        else:
            steps_to_critical = (safe_temp - current_temp) / velocity
            if steps_to_critical < 0:
                forecasts.append("CRITICAL")
            elif steps_to_critical < 1.0:
                forecasts.append("< 1 step ⚠️")
            else:
                forecasts.append(f"~{int(np.ceil(steps_to_critical))} steps")
                
    return forecasts
```

##### 2. `tests/analysis/test_trend_predictor.py` (MODIFY)
Add curvature sensitivity test verifying that an accelerating trajectory yields a higher slope than a decelerating one having the same endpoints.

#### Verification & Testing
1. Run `pytest tests/analysis/test_trend_predictor.py -v`.
2. Run full regression test suite.








