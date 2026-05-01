# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Bash Env environment.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


class BashAction(Action):
    """Action for interacting with a task session."""

    action_type: Literal["exec", "submit", "close"] = Field(default="exec")
    command: str = Field(default="")
    answer: str = Field(default="")
    timeout_s: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_required_fields(self) -> "BashAction":
        if self.action_type == "exec" and not self.command:
            raise ValueError("command is required for action_type='exec'")
        return self


class BashObservation(Observation):
    """Observation returned from the Bash Env environment."""

    instruction: str = Field(default="")
    output: str = Field(default="")
    success: bool = Field(default=True)
    error: str = Field(default="")
    task_id: str = Field(default="")
    action_type: str = Field(default="")


class BashState(State):
    """Server-side state for a Bash Env session."""

    task_id: str = Field(default="")
    workdir: str = Field(default="")
    last_action_type: str = Field(default="")
    last_command: str = Field(default="")
    last_output: str = Field(default="")
