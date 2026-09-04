# ADCTM Audit: Remediation Plans (Audit_Fixes_Plans.md)

This document tracks the detailed engineering implementation plans for addressing each problem recorded in `Audit_Problems.md`. As we progress through fixing issues, actionable remediation designs, file modifications, code snippets, and verification procedures are appended here.

---

## Remediation Index

| Problem ID | Problem Description | Plan Status |
|---|---|---|
| **C1** | Phantom Baselines Claimed in README | ✅ **Completed & Verified** |
| **C2** | Active HuggingFace API Secret Committed to Git | ✅ **Completed & Verified** |
| **C3** | `/reset` Accepts Arbitrary TaskConfig Permitting Evaluation Gaming | ✅ **Completed & Verified** |
| **C4** | `get_score()` Returns Perfect 1.0 Score on Zero Steps | ⏳ Pending |
| **H1** | Global Jitter Penalty Bypass Exploit | ⏳ Pending |
| **H2** | Evaluation Metric Heavily Rewards Total Inaction | ⏳ Pending |
| **H3** | Absence of Physical Upper Temperature Bound | ⏳ Pending |
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
