# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bash Env Environment Client."""

from __future__ import annotations

from typing import Any

# Support both in-repo and standalone imports
try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import BashAction, BashObservation, BashState
except ImportError:
    from models import BashAction, BashObservation, BashState

    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient


class BashEnv(EnvClient[BashAction, BashObservation, BashState]):
    """HTTP client for the Bash Env environment."""

    def _step_payload(self, action: BashAction) -> dict[str, Any]:
        return {
            "action_type": action.action_type,
            "command": action.command,
            "answer": action.answer,
            "timeout_s": action.timeout_s,
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[BashObservation]:
        obs_data = payload.get("observation", {})
        observation = BashObservation(
            instruction=obs_data.get("instruction", ""),
            output=obs_data.get("output", ""),
            success=obs_data.get("success", True),
            error=obs_data.get("error", ""),
            task_id=obs_data.get("task_id", ""),
            action_type=obs_data.get("action_type", ""),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> BashState:
        return BashState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            workdir=payload.get("workdir", ""),
            last_action_type=payload.get("last_action_type", ""),
            last_command=payload.get("last_command", ""),
            last_output=payload.get("last_output", ""),
        )
