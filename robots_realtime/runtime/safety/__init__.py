"""Command-space safety guardrails for teleop and inference.

Guardrails sit on the command path in ``AgentNode`` (just before a joint command is
published) and bound what can reach the robot:

  - Bounding box (position clamp) — teleop + inference.
  - Speed limit (per-step joint delta) — inference, and the new teleop speed cap.
  - Cartesian workspace reject/hold — teleop (FK-based, no IK).

They are pure and deterministic (numpy in, numpy + event out) so they are unit-testable
without hardware, and they clamp rather than drop so teleop keeps publishing at cadence.

Existing lower-level safety (motor/joint-limit clipping in the robot drivers) remains
authoritative after these filters.
"""

from robots_realtime.runtime.safety.config import (
    ArmSafety,
    BoundingBoxConfig,
    CommandSource,
    SafetyConfig,
    SafetyConfigError,
    SpeedLimitConfig,
    build_safety_config,
)
from robots_realtime.runtime.safety.guardrails import (
    BoundingBoxGuardrail,
    ClampEvent,
    SpeedLimitGuardrail,
)

__all__ = [
    "ArmSafety",
    "BoundingBoxConfig",
    "BoundingBoxGuardrail",
    "ClampEvent",
    "CommandSource",
    "SafetyConfig",
    "SafetyConfigError",
    "SpeedLimitConfig",
    "SpeedLimitGuardrail",
    "build_safety_config",
]
