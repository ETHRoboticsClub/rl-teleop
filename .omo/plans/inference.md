# Cosmos/IDM `.pt` Inference via `rr-session`

## TL;DR
> **Summary**: Add v1 inference for Cosmos/IDM PyTorch `.pt`/`.pts` policies through the existing `rr-session <yaml>` runtime, `PolicyAgent` interface, `AgentNode` bus bridge, TUI/dashboard, and YAM arm configs. LeRobot/`lerobot-record` is explicitly deferred because the policy artifacts are not LeRobot-native and no rollout recording output is required.
> **Deliverables**:
> - TDD fake-policy test suite for checkpoint loading, observation mapping, action mapping, timing, and safety failures.
> - `CosmosIdmPolicyAgent` implementing the existing `PolicyAgent.act(obs)` contract.
> - Checkpoint inspection utilities that load/probe the Cosmos and IDM `.pt` files on CPU to discover available tensors, metadata, input/output shapes, and required mapping config before any robot session starts.
> - Config validation helpers for model path/device/mapping/normalization/safety settings.
> - Example `configs/yam/*cosmos_idm*_inference*.yaml` showing `.pt` deployment with existing arms, cameras, Viser, and TUI.
> - Deployment docs explaining `uv run rr-session <config>` and why LeRobot is out of v1 scope.
> **Effort**: Medium
> **Parallel**: YES - 3 waves
> **Critical Path**: Task 1 → Task 2 → Task 4 → Task 7 → Final Verification

## Context
### Original Request
- Add inference to this project.
- Investigate whether there already is a code path for inference.
- Reuse the same dashboard and as many teleop code paths/configs as possible, especially arms.
- Initial mention of `lerobot-record`/LR-compatible policies was revised after clarification: user `.pt` artifacts are produced from fine-tuned NVIDIA Cosmos plus custom IDM, so LeRobot is not required for v1.

### Interview Summary
- Use existing `rr-session <yaml>` deployment UX, not a new runtime/dashboard.
- Deploy `.pt`/`.pts` through YAML by setting an `AgentNode` `agent_class` and `agent_kwargs`.
- No rollout recording output required in v1.
- Test strategy: TDD adapter tests with fake policy fixtures first, then dry-run/sim verification before hardware.
- LeRobot/`lerobot-record` stays future optional only if actual LeRobot checkpoints or LeRobotDataset output become required.

### Metis Review (gaps addressed)
- Unknown `.pt` format, observation schema, action schema, normalizer, timing, device, and safety limits are not further user-blocking; they become explicit contract/config-validation tasks with fake fixtures.
- Guardrails added: no new dashboard, no LeRobot runtime ownership, no recording/export pipeline, no real hardware required for tests, fail-closed action safety.
- Acceptance criteria require executable commands and fake data, not manual operator confirmation.

## Work Objectives
### Core Objective
Enable Cosmos/IDM PyTorch policy inference inside the existing robots_realtime runtime by adding a tested `PolicyAgent` adapter that maps existing session observations to model inputs and maps model outputs to existing left/right arm command topics.

### Deliverables
1. `tests/agents/policy_learning/test_cosmos_idm_policy_agent.py` with fake checkpoint/model tests, including fake Cosmos and IDM `.pt` shape-inspection fixtures.
2. `robots_realtime/agents/policy_learning/cosmos_idm_policy_agent.py` with adapter implementation and checkpoint-probing helpers.
3. Config/dataclass or validation helpers inside the same module unless complexity justifies `cosmos_idm_config.py`.
4. Example config under `configs/yam/`, derived from existing ACT/OpenPI policy configs.
5. README or docs update documenting deployment with `uv run rr-session <config>`.
6. Optional `rr-session --validate` or equivalent config-validation path only if no existing non-starting validation path can support agent-executable integration QA.

### Definition of Done (verifiable conditions with commands)
- `uv run pytest tests/agents/policy_learning/test_cosmos_idm_policy_agent.py` passes.
- `uv run pytest tests/runtime/test_rr_session_cli.py` passes after any CLI/config-loader changes.
- `uv run pytest tests/agents/teleoperation/test_yam_leader_agent.py` passes to show teleop/arms behavior did not regress.
- `uv run ruff check robots_realtime/agents/policy_learning tests/agents/policy_learning tests/runtime` passes.
- Example config validation command added by this work passes, e.g. `uv run rr-session configs/yam/yam_bimanual_cosmos_idm_inference_fake.yaml --validate`.

### Must Have
- Existing `rr-session` remains the user-facing launcher.
- Adapter implements `PolicyAgent` from `robots_realtime/agents/agent.py:31`.
- Adapter inspects both configured Cosmos and IDM `.pt` files before first inference, records discovered tensor/module shapes, and rejects ambiguous or unsupported checkpoint contents before `Session.start()` can command hardware.
- Existing `AgentNode` YAML `agent_class`/`agent_kwargs` dynamic build path at `robots_realtime/runtime/agent_node.py:45` and `robots_realtime/runtime/agent_node.py:127` is reused.
- Existing multi-arm output format from `robots_realtime/runtime/agent_node.py:14` and fanout at `robots_realtime/runtime/agent_node.py:197` is used: `{"left": {"pos": ...}, "right": {"pos": ...}}`.
- Existing TUI/session lifecycle from `robots_realtime/rr_session_cli.py:121` and `robots_realtime/rr_session_cli.py:128` is reused.
- Example config uses `session.start_paused: true`, matching policy config safety pattern at `configs/yam/yam_bimanual_openpi_policy_autolab.yaml:21` and `configs/yam/yam_bimanual_act_policy_xdof_hq.yaml:31`.
- Robot nodes reuse existing YAM `robot_config` entries and `cmd_topic` fanout patterns from `configs/yam/yam_bimanual_act_policy_xdof_hq.yaml:86` through `configs/yam/yam_bimanual_act_policy_xdof_hq.yaml:108`.

### Must NOT Have
- No LeRobot runtime, `lerobot-record`, `lerobot-rollout`, or LeRobot robot/camera construction in v1.
- No LeRobotDataset or RR recording output changes in v1.
- No new dashboard or alternate session manager.
- No model-specific hardcoded station details in Python: camera serials, YAM config paths, startup poses, and command topics stay in YAML.
- No hardcoded Cosmos/IDM tensor shapes in Python without checkpoint inspection or explicit YAML override.
- No silent action clamping unless explicitly configured; default invalid action behavior is reject/fail closed.
- No real hardware requirement for automated tests.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: TDD with pytest fake policy fixtures before implementation.
- QA policy: Every task has agent-executed happy-path and failure-path scenarios.
- Evidence: terminal output captured in executor responses or files under `/tmp/rr-inference-evidence/task-{N}-{slug}.{ext}`. Do not write evidence under `.omo/`.

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. This plan uses fewer tasks per wave because the work has a narrow critical contract dependency.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: Task 1 (contract/TDD), Task 3 (config validation tests), Task 6 (docs skeleton) can start together after reading referenced files.
Wave 2: Task 2 (adapter implementation), Task 4 (AgentNode/session validation integration), Task 5 (example YAML) after Task 1 contract is accepted.
Wave 3: Task 7 (integration QA), Task 8 (safety hardening review/docs finalization) after implementation and config exist.

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 2, 4, 5, 7, 8.
- Task 2 blocks Tasks 4, 5, 7, 8.
- Task 3 blocks Task 4.
- Task 4 blocks Task 7.
- Task 5 blocks Task 7.
- Task 6 blocks Task 8.
- Task 7 blocks final verification.
- Task 8 blocks final verification.

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 3 tasks → quick, quick, writing
- Wave 2 → 3 tasks → unspecified-high, quick, quick
- Wave 3 → 2 tasks → unspecified-high, writing

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Define Cosmos/IDM checkpoint inspection and policy adapter contract with fake-policy TDD

  **What to do**: Create `tests/agents/policy_learning/test_cosmos_idm_policy_agent.py`. Add fake Cosmos `.pt` and fake IDM `.pt` fixtures that define the v1 supported contract before implementation: local PyTorch load path, CPU-only checkpoint inspection, discovered tensor/module metadata, device selection (`cpu`, `cuda`, `mps` with unavailable-device failure), required observation keys, optional image keys, normalizer config, action vector shape, and left/right split. Tests must assert the adapter constructor rejects unsupported checkpoint formats, missing paths, missing mapping keys, missing normalizer when `action_normalization: required`, unavailable device, and ambiguous left/right action splits.
  **Must NOT do**: Do not import real Cosmos/NVIDIA dependencies. Do not require real `.pt` files beyond tiny fake fixtures created in `tmp_path`. Do not start `rr-session` or hardware in this task.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: test-first contract in one test file with fake fixtures.
  - Skills: [`superpowers:test-driven-development`] - Use red tests before implementation.
  - Omitted: [`frontend-ui-ux`] - No UI work.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [2, 4, 5, 7, 8] | Blocked By: []

  **References**:
  - Pattern: `tests/agents/teleoperation/test_yam_leader_agent.py:61` - pytest style with fake robot fixtures and direct `agent.act()` assertions.
  - Pattern: `tests/runtime/test_rr_session_cli.py:84` - temp YAML config test style.
  - API/Type: `robots_realtime/agents/agent.py:31` - `PolicyAgent` contract to implement.
  - Pattern: `robots_realtime/agents/policy_learning/diffusion_policy_agent.py:29` - learned policy `act()` returns left/right `pos` commands.

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/agents/policy_learning/test_cosmos_idm_policy_agent.py -q` initially fails because `CosmosIdmPolicyAgent` does not exist or does not validate the contract.
  - [ ] Test file includes concrete fake Cosmos and IDM `.pt` files saved under `tmp_path`; inspection asserts discovered keys/shapes before fake inference runs.
  - [ ] Test file includes concrete fake observation: `left.joint_pos=np.zeros(7)`, `right.joint_pos=np.ones(7)`, and fake action tensor of shape `(14,)` split into two `(7,)` commands.
  - [ ] Test file includes failure cases for missing checkpoint path, unavailable device, missing observation key, wrong output shape `(13,)`, NaN output, and missing normalization config when required.

  **QA Scenarios**:
  ```
  Scenario: Fake policy contract is locked before implementation
    Tool: Bash
    Steps: Run `uv run pytest tests/agents/policy_learning/test_cosmos_idm_policy_agent.py -q` immediately after writing tests.
    Expected: Command fails only because implementation is missing/incomplete, not because fixtures or imports are broken.
    Evidence: /tmp/rr-inference-evidence/task-1-contract-red.txt

  Scenario: Unsupported checkpoint type is specified in tests
    Tool: Bash
    Steps: Run `uv run pytest tests/agents/policy_learning/test_cosmos_idm_policy_agent.py -q -k rejects_unsupported_checkpoint_type`.
    Expected: Test asserts a clear `ValueError` message containing `unsupported checkpoint`.
    Evidence: /tmp/rr-inference-evidence/task-1-unsupported-checkpoint.txt

  Scenario: Cosmos and IDM checkpoint shape probes are specified
    Tool: Bash
    Steps: Run `uv run pytest tests/agents/policy_learning/test_cosmos_idm_policy_agent.py -q -k "inspect_checkpoint_shapes or probe_checkpoint"`.
    Expected: Test asserts fake Cosmos and IDM `.pt` keys/shapes are discovered on CPU and stored in debug metadata.
    Evidence: /tmp/rr-inference-evidence/task-1-checkpoint-shape-probe.txt
  ```

  **Commit**: YES | Message: `add cosmos idm policy contract tests` | Files: [`tests/agents/policy_learning/test_cosmos_idm_policy_agent.py`]

- [ ] 2. Implement `CosmosIdmPolicyAgent` and checkpoint shape probing behind existing `PolicyAgent`

  **What to do**: Add `robots_realtime/agents/policy_learning/cosmos_idm_policy_agent.py`. Implement `CosmosIdmPolicyAgent(PolicyAgent)` with clear sub-functions: resolve configured `cosmos_checkpoint_path` and `idm_checkpoint_path`, inspect each `.pt` on CPU with safe PyTorch loading, summarize discovered keys/tensor shapes/module signatures, validate config against discovered shapes, load runtime model(s), map observations to model input, run inference under `torch.no_grad()`, map action to `{"left": {"pos": np.ndarray}, "right": {"pos": np.ndarray}}`, validate finite values/shape/bounds, expose optional `_chunk` and `_images` only if config supplies them. Support only local PyTorch/TorchScript-style callables needed by fake tests; if a raw `state_dict` requires an unavailable model class, fail with a clear error instructing the user to provide a loader module path in config.
  **Must NOT do**: Do not modify `AgentNode` for basic mapping. Do not add LeRobot imports. Do not silently guess joint order, gripper scaling, image preprocessing, normalization, or Cosmos/IDM tensor shapes.

  **Adapter Implementation Details**:
  - Add a small inspection result type, e.g. `CheckpointInfo(kind, path, top_level_type, keys, tensor_shapes, module_class, metadata)`.
  - Add `inspect_checkpoint(path, kind)` that loads with `map_location="cpu"`, never starts CUDA/MPS, and returns shape metadata for tensors nested in dict/list/module state dicts.
  - For actual runtime load, support explicit config fields:
    - `cosmos_checkpoint_path`: required path to the fine-tuned Cosmos `.pt`.
    - `idm_checkpoint_path`: required path to custom IDM `.pt` if IDM is a separate artifact; optional only if config says `idm: null` / `idm_enabled: false`.
    - `loader`: optional dotted `module:function` for project-specific model construction when `.pt` is a state dict rather than callable module.
    - `expected_input_shapes` and `expected_action_shape`: explicit overrides that must match inspected shapes or fail with a shape mismatch.
  - Setup sequence must be: validate paths → inspect Cosmos `.pt` → inspect IDM `.pt` → validate YAML mapping/expected shapes → load callable runtime model(s) → set debug metadata → allow `act()`.
  - `act()` must refuse to run if inspection failed, model load failed, or config/discovered shape mismatch remains unresolved.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: safety-critical adapter with validation and test-driven implementation.
  - Skills: [`superpowers:test-driven-development`] - Make Task 1 tests pass minimally.
  - Omitted: [`frontend-ui-ux`] - No UI work.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [4, 5, 7, 8] | Blocked By: [1]

  **References**:
  - API/Type: `robots_realtime/agents/agent.py:34` - required `act(obs)` method.
  - Pattern: `robots_realtime/agents/policy_learning/diffusion_policy_agent.py:29` - convert model output to left/right `pos` dict.
  - Bus contract: `robots_realtime/runtime/agent_node.py:180` - action publishing expects dict commands.
  - Bus contract: `robots_realtime/runtime/agent_node.py:216` - positions are converted to `np.float32` before publishing.

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/agents/policy_learning/test_cosmos_idm_policy_agent.py -q` passes.
  - [ ] `uv run ruff check robots_realtime/agents/policy_learning/cosmos_idm_policy_agent.py tests/agents/policy_learning/test_cosmos_idm_policy_agent.py` passes.
  - [ ] Adapter debug metadata includes inspected Cosmos and IDM paths, top-level types, and tensor shapes.
  - [ ] Adapter rejects mismatches between YAML `expected_*_shape` fields and inspected checkpoint/model output shapes before `act()` returns any command.
  - [ ] Adapter rejects NaN/Inf and wrong action shapes before returning commands to `AgentNode`.
  - [ ] Adapter returns exactly `left.pos.shape == (7,)` and `right.pos.shape == (7,)` for the fake bimanual action.

  **QA Scenarios**:
  ```
  Scenario: Fake `.pt` policy produces bimanual commands
    Tool: Bash
    Steps: Run `uv run pytest tests/agents/policy_learning/test_cosmos_idm_policy_agent.py -q -k "happy_path or maps"`.
    Expected: Tests pass and assert left/right command arrays match the fake action split exactly.
    Evidence: /tmp/rr-inference-evidence/task-2-happy-path.txt

  Scenario: Invalid policy output fails closed
    Tool: Bash
    Steps: Run `uv run pytest tests/agents/policy_learning/test_cosmos_idm_policy_agent.py -q -k "nan or wrong_output_shape"`.
    Expected: Tests pass and assert no arm command dict is returned for invalid output; clear exception is raised.
    Evidence: /tmp/rr-inference-evidence/task-2-invalid-output.txt

  Scenario: Checkpoint shape mismatch fails before command output
    Tool: Bash
    Steps: Run `uv run pytest tests/agents/policy_learning/test_cosmos_idm_policy_agent.py -q -k "shape_mismatch or checkpoint_info"`.
    Expected: Tests pass and assert explicit shape mismatch errors include checkpoint kind (`cosmos` or `idm`) and expected vs discovered shape.
    Evidence: /tmp/rr-inference-evidence/task-2-checkpoint-shape-mismatch.txt
  ```

  **Commit**: YES | Message: `add cosmos idm policy agent` | Files: [`robots_realtime/agents/policy_learning/cosmos_idm_policy_agent.py`, `tests/agents/policy_learning/test_cosmos_idm_policy_agent.py`]

- [ ] 3. Add config validation tests for `AgentNode` YAML integration

  **What to do**: Extend or add runtime tests that load a tiny temp YAML with an `AgentNode` pointing at `robots_realtime.agents.policy_learning.cosmos_idm_policy_agent:CosmosIdmPolicyAgent`. Verify `load_session()` passes `agent_kwargs`, `state_topics`, `image_topics`, `loop_mode`, and `poll_freq` without starting hardware. If current `load_session()` eagerly constructs agents only on node `setup()`, assert the node stores the class path and kwargs correctly. Add failing YAML cases for missing `agent_class`, missing `checkpoint_path`, and unknown node type only where applicable.
  **Must NOT do**: Do not start a session or instantiate real robot/camera nodes. Do not use absolute machine-specific checkpoint paths.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: focused YAML loader tests.
  - Skills: [`superpowers:test-driven-development`] - Lock behavior before integration tweaks.
  - Omitted: [`git-master`] - Commit handled by executor workflow as needed.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [4] | Blocked By: []

  **References**:
  - Pattern: `tests/runtime/test_rr_session_cli.py:84` - temp YAML config load test.
  - Config loader: `robots_realtime/runtime/config.py:93` - `load_session()` entry point.
  - Agent kwargs build: `robots_realtime/runtime/agent_node.py:229` - `AgentNode.build_kwargs()` handles YAML fields.

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/runtime/test_rr_session_cli.py -q` passes.
  - [ ] New/updated test asserts `agent_kwargs["checkpoint_path"] == "fake_policy.pt"` or an equivalent concrete fake path.
  - [ ] New/updated test asserts `loop_mode: fixed_rate` and `poll_freq: 30` survive config loading.

  **QA Scenarios**:
  ```
  Scenario: YAML config can express Cosmos/IDM policy node
    Tool: Bash
    Steps: Run `uv run pytest tests/runtime/test_rr_session_cli.py -q -k "cosmos or yaml"`.
    Expected: Config loads without starting session; AgentNode has expected class path, topics, and kwargs.
    Evidence: /tmp/rr-inference-evidence/task-3-yaml-config.txt

  Scenario: Missing required adapter config fails validation path
    Tool: Bash
    Steps: Run `uv run pytest tests/runtime/test_rr_session_cli.py -q -k "missing_checkpoint_path or cosmos"`.
    Expected: Test asserts clear error before runtime control loop starts.
    Evidence: /tmp/rr-inference-evidence/task-3-missing-config.txt
  ```

  **Commit**: YES | Message: `test cosmos idm session config` | Files: [`tests/runtime/test_rr_session_cli.py`]

- [ ] 4. Add non-starting config validation path for `rr-session` if needed

  **What to do**: If no existing command can validate an inference config without starting nodes, add `--validate` to `robots_realtime/rr_session_cli.py`. The flag must load YAML via existing `load_session()`, optionally call a lightweight validation method if added, print a deterministic success message, and exit `0` without `session.start()`. Add CLI tests using the existing monkeypatch pattern. If a non-starting validation path already exists by implementation time, use it and do not add a new flag.
  **Must NOT do**: Do not start hardware, camera, TUI, or session threads under `--validate`. Do not change normal `rr-session` behavior.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: small CLI/config validation addition if necessary.
  - Skills: [`superpowers:test-driven-development`] - Tests before CLI behavior.
  - Omitted: [`frontend-ui-ux`] - CLI-only.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [7] | Blocked By: [1, 2, 3]

  **References**:
  - CLI parser: `robots_realtime/rr_session_cli.py:39` - `argparse` command setup.
  - Session start point: `robots_realtime/rr_session_cli.py:121` - must not be reached under validate mode.
  - Test pattern: `tests/runtime/test_rr_session_cli.py:28` - monkeypatch CLI args and fake session.

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/runtime/test_rr_session_cli.py -q -k "validate or no_tui"` passes.
  - [ ] `uv run rr-session configs/yam/yam_bimanual_cosmos_idm_inference_fake.yaml --validate` exits `0` after Task 5 creates the config.
  - [ ] `--validate` output contains `Session config valid` and does not contain `Session running`.

  **QA Scenarios**:
  ```
  Scenario: Validate mode does not start session
    Tool: Bash
    Steps: Run `uv run pytest tests/runtime/test_rr_session_cli.py -q -k validate`.
    Expected: Fake session events do not include `start`, `wait`, or `stop`; command exits 0.
    Evidence: /tmp/rr-inference-evidence/task-4-validate-no-start.txt

  Scenario: Normal no-TUI path still starts/stops
    Tool: Bash
    Steps: Run `uv run pytest tests/runtime/test_rr_session_cli.py -q -k no_tui`.
    Expected: Existing event sequence remains `start`, `wait`, `stop`, `exit:0`.
    Evidence: /tmp/rr-inference-evidence/task-4-no-tui-regression.txt
  ```

  **Commit**: YES | Message: `add rr session config validation` | Files: [`robots_realtime/rr_session_cli.py`, `tests/runtime/test_rr_session_cli.py`]

- [ ] 5. Add fake-safe Cosmos/IDM inference YAML derived from existing policy configs

  **What to do**: Add `configs/yam/yam_bimanual_cosmos_idm_inference_fake.yaml`. Copy the structure from `yam_bimanual_act_policy_xdof_hq.yaml` or `yam_bimanual_openpi_policy_autolab.yaml`: `session.start_paused: true`, `record_on_unpause: false`, one `AgentNode`, two `RobotNode`s, camera nodes as needed, and one `ViserMonitorNode`. Change only the policy node name/class/kwargs and command topics. Use fake/sample `cosmos_checkpoint_path` and `idm_checkpoint_path` values documented as placeholders for validation tests; do not point to real private models. Include explicit `observation_mapping`, `action_mapping`, `expected_input_shapes`, `expected_action_shape`, `safety`, `device`, and `inference_mode` kwargs.
  **Must NOT do**: Do not duplicate robot configs. Do not change existing ACT/OpenPI configs. Do not enable recording output. Do not set `start_paused: false`.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: config-copy task with strict references.
  - Skills: [] - No specialized skill required.
  - Omitted: [`frontend-ui-ux`] - No UI design.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [7] | Blocked By: [1, 2]

  **References**:
  - Pattern: `configs/yam/yam_bimanual_act_policy_xdof_hq.yaml:35` - `AgentNode` for policy deployment.
  - Pattern: `configs/yam/yam_bimanual_act_policy_xdof_hq.yaml:76` - state/image topic mapping.
  - Pattern: `configs/yam/yam_bimanual_act_policy_xdof_hq.yaml:86` - left/right `RobotNode` command topic wiring.
  - Pattern: `configs/yam/yam_bimanual_act_policy_xdof_hq.yaml:131` - `ViserMonitorNode` reuse.

  **Acceptance Criteria**:
  - [ ] `uv run rr-session configs/yam/yam_bimanual_cosmos_idm_inference_fake.yaml --validate` exits `0` after Task 4.
  - [ ] Config contains `agent_class: robots_realtime.agents.policy_learning.cosmos_idm_policy_agent:CosmosIdmPolicyAgent`.
  - [ ] Config contains placeholder `cosmos_checkpoint_path`, placeholder `idm_checkpoint_path`, `expected_input_shapes`, and `expected_action_shape` fields with comments explaining they are validated against inspected `.pt` shapes.
  - [ ] Config contains `session.start_paused: true` and `session.record_on_unpause: false`.
  - [ ] Robot `cmd_topic` values are `cosmos_idm_policy/left_pos` and `cosmos_idm_policy/right_pos`.

  **QA Scenarios**:
  ```
  Scenario: Example config validates without hardware start
    Tool: Bash
    Steps: Run `uv run rr-session configs/yam/yam_bimanual_cosmos_idm_inference_fake.yaml --validate`.
    Expected: Exit 0 and print `Session config valid`.
    Evidence: /tmp/rr-inference-evidence/task-5-config-validate.txt

  Scenario: Example config preserves safety gate
    Tool: Bash
    Steps: Run `uv run python - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('configs/yam/yam_bimanual_cosmos_idm_inference_fake.yaml').read_text())
assert cfg['session']['start_paused'] is True
assert cfg['session']['record_on_unpause'] is False
PY`.
    Expected: Assertions pass.
    Evidence: /tmp/rr-inference-evidence/task-5-safety-yaml.txt
  ```

  **Commit**: YES | Message: `add cosmos idm inference config` | Files: [`configs/yam/yam_bimanual_cosmos_idm_inference_fake.yaml`]

- [ ] 6. Document `rr-session` inference deployment and v1 scope

  **What to do**: Update `README.md` or add `docs/inference.md` and link it from README. Explain that `rr-session` is the repo launcher for YAML sessions, show how to deploy `.pt` artifacts by editing the Cosmos/IDM YAML `cosmos_checkpoint_path` and `idm_checkpoint_path`, and state that the adapter inspects both checkpoint files to discover keys/shapes before session startup. State that v1 uses existing TUI/dashboard/arms/cameras. Include a short “Why not LeRobot in v1?” note: artifacts are Cosmos/IDM `.pt`, no LeRobotDataset output is required, and LeRobot can be future backend only.
  **Must NOT do**: Do not promise support for arbitrary `.pt` formats beyond the implemented loader contract. Do not include private checkpoint paths or hardware serials not already in configs.

  **Recommended Agent Profile**:
  - Category: `writing` - Reason: user-facing deployment docs.
  - Skills: [] - No specialized skill required.
  - Omitted: [`git-master`] - Commit handled separately.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [8] | Blocked By: []

  **References**:
  - Existing usage docs: `README.md` Quickstart shows `uv run rr-session configs/yam/yam_bimanual_yam_leader.yaml`.
  - CLI docs: `robots_realtime/rr_session_cli.py:45` - positional YAML session argument.
  - Config examples: `configs/yam/yam_bimanual_act_policy_xdof_hq.yaml:15` - policy checkpoint-style deployment comment.

  **Acceptance Criteria**:
  - [ ] Docs contain command `uv run rr-session configs/yam/yam_bimanual_cosmos_idm_inference_fake.yaml --validate`.
  - [ ] Docs contain command `uv run rr-session configs/yam/yam_bimanual_cosmos_idm_inference_fake.yaml` for actual launch.
  - [ ] Docs explain `.pt` supported contract, Cosmos/IDM checkpoint shape inspection, expected-shape YAML validation, and explicit unsupported cases.
  - [ ] Docs state LeRobot/recording output is out of v1 scope.

  **QA Scenarios**:
  ```
  Scenario: Docs include deployment command
    Tool: Bash
    Steps: Run `uv run python - <<'PY'
from pathlib import Path
text = Path('README.md').read_text()
docs = Path('docs')
if docs.exists():
    text += '\n' + '\n'.join(p.read_text() for p in docs.glob('*.md'))
cmd = 'uv run rr-session configs/yam/yam_bimanual_cosmos_idm_inference_fake.yaml'
assert cmd in text
assert cmd + ' --validate' in text
PY`.
    Expected: Command appears exactly once in deployment section and once with `--validate`.
    Evidence: /tmp/rr-inference-evidence/task-6-doc-command.txt

  Scenario: Docs prevent LeRobot scope creep
    Tool: Bash
    Steps: Run `uv run python - <<'PY'
from pathlib import Path
text = Path('README.md').read_text()
docs = Path('docs')
if docs.exists():
    text += '\n' + '\n'.join(p.read_text() for p in docs.glob('*.md'))
assert 'LeRobot' in text
assert 'out of v1 scope' in text
PY`.
    Expected: Docs explicitly say LeRobot runtime/recording is deferred unless actual LR checkpoints are needed.
    Evidence: /tmp/rr-inference-evidence/task-6-lerobot-scope.txt
  ```

  **Commit**: YES | Message: `document cosmos idm inference` | Files: [`README.md` or `docs/inference.md`]

- [ ] 7. Verify end-to-end fake inference through session/config path

  **What to do**: Add an integration test or validation command that creates/uses the fake checkpoint and validates the example YAML without starting hardware. If feasible with existing bus/test utilities, instantiate `AgentNode` directly with fake topics and assert one `step()` publishes `left_pos`/`right_pos`; otherwise keep this as a unit-level `AgentNode` test with monkeypatched publish/get_latest. Ensure slow inference timeout/stale observation behavior is tested according to adapter config.
  **Must NOT do**: Do not run real cameras/arms. Do not require user visual dashboard confirmation.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: integration of agent, node, YAML, and safety behavior.
  - Skills: [`superpowers:test-driven-development`] - Add integration regression tests.
  - Omitted: [`playwright`] - No browser UI interaction required.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [Final Verification] | Blocked By: [2, 4, 5]

  **References**:
  - Node step: `robots_realtime/runtime/agent_node.py:137` - constructs obs and calls `agent.act(obs)`.
  - Publish fanout: `robots_realtime/runtime/agent_node.py:207` - publishes per-arm `{key}_pos` topics.
  - Config loader: `robots_realtime/runtime/config.py:164` - instantiates nodes from YAML.

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/runtime/test_agent_node_cosmos_idm_integration.py -q` passes if a new file is added, or equivalent test selection passes if added elsewhere.
  - [ ] Test asserts published topics include `left_pos` and `right_pos` with `joint_pos` arrays of shape `(7,)`.
  - [ ] Test asserts missing/stale observations prevent command publication and raise/log a clear policy error.
  - [ ] `uv run rr-session configs/yam/yam_bimanual_cosmos_idm_inference_fake.yaml --validate` passes.

  **QA Scenarios**:
  ```
  Scenario: AgentNode publishes fake policy bimanual commands
    Tool: Bash
    Steps: Run `uv run pytest tests/runtime/test_agent_node_cosmos_idm_integration.py -q -k publishes`.
    Expected: Test captures `left_pos` and `right_pos` publications with expected fake action values.
    Evidence: /tmp/rr-inference-evidence/task-7-agentnode-publish.txt

  Scenario: Stale/missing observations fail closed
    Tool: Bash
    Steps: Run `uv run pytest tests/runtime/test_agent_node_cosmos_idm_integration.py -q -k "stale or missing"`.
    Expected: No command publication occurs; error message identifies missing/stale observation key.
    Evidence: /tmp/rr-inference-evidence/task-7-stale-observation.txt
  ```

  **Commit**: YES | Message: `verify cosmos idm session inference` | Files: [`tests/runtime/test_agent_node_cosmos_idm_integration.py`, related implementation files if needed]

- [ ] 8. Harden safety/debug surfaces and finalize docs

  **What to do**: Ensure adapter logs or exposes deterministic debug information at setup and per failure: checkpoint path, device, model input keys/shapes, image preprocessing expectations, state vector order, action vector split, latency, and rejection reason. Add tests for debug metadata if represented in code. Finalize docs with a hardware rollout checklist that is agent-executable up to validation and explicitly says real robot handoff requires operator procedures outside automated acceptance.
  **Must NOT do**: Do not add manual confirmation as acceptance criteria. Do not hide safety failures behind warnings while still sending commands.

  **Recommended Agent Profile**:
  - Category: `writing` - Reason: mostly docs plus small debug/test polish.
  - Skills: [] - No specialized skill required.
  - Omitted: [`frontend-ui-ux`] - No UI changes.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [Final Verification] | Blocked By: [2, 6, 7]

  **References**:
  - Existing policy config comments: `configs/yam/yam_bimanual_openpi_policy_autolab.yaml:93` - warns image preprocessing must match training augmentation.
  - Safety comment: `configs/yam/yam_bimanual_act_policy_xdof_hq.yaml:25` - start-paused rationale.
  - AgentNode optional debug image/chunk surfaces: `robots_realtime/runtime/agent_node.py:154` and `robots_realtime/runtime/agent_node.py:161`.

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/agents/policy_learning/test_cosmos_idm_policy_agent.py -q -k "debug or safety or latency"` passes.
  - [ ] Docs contain a checklist item for validating config before launch.
  - [ ] Docs contain explicit failure behavior: checkpoint shape mismatch, invalid output, missing observations, stale observations, and unavailable device fail closed.
  - [ ] `uv run ruff check robots_realtime/agents/policy_learning tests/agents/policy_learning tests/runtime` passes.

  **QA Scenarios**:
  ```
  Scenario: Debug metadata identifies model contract
    Tool: Bash
    Steps: Run `uv run pytest tests/agents/policy_learning/test_cosmos_idm_policy_agent.py -q -k debug`.
    Expected: Test asserts debug info includes checkpoint path, device, input keys/shapes, and action split.
    Evidence: /tmp/rr-inference-evidence/task-8-debug-metadata.txt

  Scenario: Docs list fail-closed behavior
    Tool: Bash
    Steps: Run `uv run python - <<'PY'
from pathlib import Path
text = Path('README.md').read_text()
docs = Path('docs')
if docs.exists():
    text += '\n' + '\n'.join(p.read_text() for p in docs.glob('*.md'))
for term in ['fail closed', 'NaN', 'stale observation', 'unavailable device', 'checkpoint shape mismatch']:
    assert term in text, term
PY`.
    Expected: All terms appear in the safety/failure section with concrete behavior.
    Evidence: /tmp/rr-inference-evidence/task-8-doc-safety.txt
  ```

  **Commit**: YES | Message: `harden cosmos idm inference safety` | Files: [`robots_realtime/agents/policy_learning/cosmos_idm_policy_agent.py`, tests, docs]

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE through agent-executed checks. Any rejection -> fix -> re-run the failed verification agent(s).
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Automated Runtime QA — unspecified-high (no real hardware required; execute fake/session validation paths)
- [ ] F4. Scope Fidelity Check — deep

  **F1 Required Checks**:
  - Verify implementation matches this plan's v1 scope and every TODO acceptance criterion is either completed or explicitly superseded by an equivalent check.
  - Verify no implementation task introduced LeRobot runtime ownership, new dashboard/session lifecycle, or recording/export output.
  - Output `APPROVE` or `REJECT` with exact file/task references.

  **F2 Required Checks**:
  - Run `uv run pytest tests/agents/policy_learning/test_cosmos_idm_policy_agent.py tests/runtime/test_rr_session_cli.py tests/runtime/test_agent_node_cosmos_idm_integration.py -q`.
  - Run `uv run ruff check robots_realtime/agents/policy_learning tests/agents/policy_learning tests/runtime`.
  - Inspect changed source for hidden hardware side effects at import/construction time.
  - Output `APPROVE` only if commands pass and no quality blockers remain.

  **F3 Required Checks**:
  - Run `uv run rr-session configs/yam/yam_bimanual_cosmos_idm_inference_fake.yaml --validate`.
  - Run the fake AgentNode publish test command from Task 7.
  - Confirm no real RobotNode, CameraNode, hardware device, browser, or TUI thread starts during validation tests.
  - Output `APPROVE` only if fake/session validation paths pass without hardware.

  **F4 Required Checks**:
  - Run `uv run python - <<'PY'
from pathlib import Path
changed = [p for p in Path('.').rglob('*') if p.is_file() and p.suffix in {'.py', '.yaml', '.yml', '.md'} and '.git' not in p.parts and '.omo' not in p.parts]
text = '\n'.join(p.read_text(errors='ignore') for p in changed)
for forbidden in ['lerobot-rollout', 'lerobot-record', 'LeRobotDataset']:
    assert forbidden not in text, forbidden
PY`.
  - Verify all new runtime operation still routes through `rr-session`, `PolicyAgent`, `AgentNode`, existing RobotNode, and existing Viser/TUI paths.
  - Output `APPROVE` only if scope fidelity holds.

## Commit Strategy
- Use multiple focused commits, following repo plain lowercase style:
  1. `add cosmos idm policy contract tests`
  2. `add cosmos idm policy agent`
  3. `test cosmos idm session config`
  4. `add rr session config validation` (only if needed)
  5. `add cosmos idm inference config`
  6. `document cosmos idm inference`
  7. `verify cosmos idm session inference`
  8. `harden cosmos idm inference safety`
- Commit only source/config/docs/test files. Never commit `.omo/` plan/draft/evidence artifacts; `.omo/` is ignored.

## Success Criteria
- User can deploy a supported Cosmos/IDM `.pt` by editing the example YAML `checkpoint_path` and running `uv run rr-session configs/yam/yam_bimanual_cosmos_idm_inference_fake.yaml`.
- Existing dashboard/TUI/session lifecycle is reused unchanged for runtime operation.
- Existing YAM arm configs and `RobotNode` command topics are reused.
- Fake-policy tests prove loading, mapping, action validation, timing, and fail-closed behavior before hardware.
- LeRobot and recording/export remain out of v1 scope.
