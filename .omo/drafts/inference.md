# Draft: Inference

## Requirements (confirmed)
- Add inference to this project.
- Investigate whether there already is a code path for inference.
- Use `lerobot-record` for inference so it can load LR-compatible policies.
- Reuse the same dashboard as teleop.
- Reuse as many teleop code paths/configs as possible, especially arms configuration.

## Technical Decisions
- Pending: whether to wrap/extend an existing inference path or add a new command/session mode.
- Pending: whether inference should be a separate CLI command, a config-driven session mode, or both.
- Pending: policy source interface details for LR-compatible policies.

## Research Findings
- README describes a unified pipeline for teleoperation, data collection, and policy deployment, but marks “Test + verify policy deploy pipeline” as TODO.
- README quickstart uses `uv run rr-session <config.yaml>` for teleop sessions and shows the existing Rich TUI/dashboard.
- No OpenSpec or Spec Kit directories found.
- Detailed code-path and test/config exploration is in progress via background research agents.
- Existing policy/inference-shaped path: `robots_realtime/agents/agent.py` defines the base `PolicyAgent`; `robots_realtime/agents/policy_learning/diffusion_policy_agent.py` shows learned-policy agent patterns with `load_model` and `act`.
- Existing runtime bridge: `robots_realtime/runtime/agent_node.py` wraps agents on the ZMQ bus and supports `subscriber_driven` loop mode, which research identified as likely reusable for inference.
- Existing session/config path: `robots_realtime/runtime/config.py`, `robots_realtime/runtime/session.py`, and `robots_realtime/rr_session_cli.py` load YAML session configs and run the dashboard/TUI.
- Existing recording path: `robots_realtime/runtime/recording.py` has `McapWriter` and `AsyncMp4Writer`; LeRobot dataset writing/format mapping appears missing.
- Existing reusable arms/config area: `configs/yam/` contains current robot/leader/sim configs to preserve rather than duplicate.
- Gap: no dedicated `lerobot-record`/LR-compatible inference CLI was found; likely plan needs either a wrapper command or session mode that reuses `rr-session` internals.

## Open Questions
- Should inference be launched through `rr-session` with an inference mode, through a new command, or by matching the `lerobot-record` UX directly?
- What exact inference success criteria matter most: real robot deployment, sim deployment, policy smoke-test only, recording during inference, or all of these?
- Should the initial plan include automated tests only, or test-driven implementation for any policy/config glue?

## Scope Boundaries
- INCLUDE: inference planning, LR-compatible policy loading through `lerobot-record`, teleop dashboard reuse, teleop arms/config reuse.
- EXCLUDE: training new policies unless explicitly requested.
