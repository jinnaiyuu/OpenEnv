# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bash Env environment server implementation (local execution)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import BashAction, BashObservation, BashState
except ImportError:
    from models import BashAction, BashObservation, BashState

_DEFAULT_TASKS_PATH = (
    Path(__file__).resolve().parent.parent / "tasks" / "tasks.jsonl"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _normalize_answer(text: str) -> str:
    return " ".join(text.strip().split()).casefold()


def _load_tasks(tasks_path: Path) -> dict[str, dict[str, str]]:
    if not tasks_path.exists():
        logger.error("BashEnv tasks file not found: %s", tasks_path)
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

    tasks: dict[str, dict[str, str]] = {}
    lines = tasks_path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.exception(
                "Invalid JSON in tasks file %s at line %s", tasks_path, idx
            )
            raise ValueError(
                f"Invalid JSON in tasks file {tasks_path} at line {idx}: {exc}"
            ) from exc

        task_id = data.get("task_id")
        instruction = data.get("instruction")
        expected_answer = data.get("expected_answer")

        if not task_id or not instruction or expected_answer is None:
            logger.error(
                "Invalid task entry in %s at line %s: %s",
                tasks_path,
                idx,
                data,
            )
            raise ValueError(
                "Each task must include task_id, instruction, and expected_answer"
            )

        tasks[str(task_id)] = {
            "instruction": str(instruction),
            "expected_answer": str(expected_answer),
        }

    if not tasks:
        logger.error("No tasks loaded from %s", tasks_path)
        raise ValueError(f"No tasks loaded from {tasks_path}")

    return tasks


class BashEnvironment(Environment[BashAction, BashObservation, BashState]):
    """OpenEnv wrapper for local shell tasks with submit-style scoring."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        tasks_path: str | None = None,
        output_dir: str | None = None,
        command_timeout_s: float | None = None,
    ) -> None:
        super().__init__()

        env_tasks_path = os.getenv("BASH_TASKS_PATH")
        self.tasks_path = Path(tasks_path or env_tasks_path or _DEFAULT_TASKS_PATH)
        self.output_dir = Path(
            output_dir or os.getenv("BASH_OUTPUT_DIR", "/tmp/bash_env_runs")
        )
        self.command_timeout_s = float(
            command_timeout_s
            if command_timeout_s is not None
            else os.getenv("BASH_COMMAND_TIMEOUT_S", "20.0")
        )

        logger.debug(
            "Initializing BashEnvironment (tasks_path=%s, output_dir=%s, timeout_s=%s, env_tasks_path=%s)",
            self.tasks_path,
            self.output_dir,
            self.command_timeout_s,
            env_tasks_path,
        )

        self._tasks = _load_tasks(self.tasks_path)

        self._state = BashState(episode_id=str(uuid4()), step_count=0)
        self._task_id: str | None = None
        self._instruction = ""
        self._expected_answer = ""
        self._workdir: Path | None = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> BashObservation:
        del seed

        task_id = kwargs.get("task_id") or kwargs.get("task_name")
        if not task_id:
            raise ValueError("Provide task_id to reset Bash Env.")

        task = self._tasks.get(str(task_id))
        if not task:
            raise KeyError(f"Unknown task_id: {task_id}")

        self._task_id = str(task_id)
        self._instruction = task["instruction"]
        self._expected_answer = task["expected_answer"]

        run_id = episode_id or uuid4().hex
        workdir = self.output_dir / f"{self._task_id}.{run_id}" / "workdir"
        workdir.mkdir(parents=True, exist_ok=True)
        self._workdir = workdir

        logger.info(
            "Reset BashEnvironment (task_id=%s, episode_id=%s, workdir=%s)",
            self._task_id,
            run_id,
            workdir,
        )

        self._state = BashState(
            episode_id=str(run_id),
            step_count=0,
            task_id=self._task_id,
            workdir=str(workdir),
            last_action_type="reset",
            last_command="",
            last_output="",
        )

        return BashObservation(
            instruction=self._instruction,
            output="",
            success=True,
            error="",
            task_id=self._task_id,
            action_type="reset",
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: BashAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> BashObservation:
        del kwargs

        if not isinstance(action, BashAction):
            raise TypeError(f"Expected BashAction, got {type(action)}")
        if self._workdir is None or self._task_id is None:
            raise RuntimeError("Bash Env not initialized. Call reset() first.")

        logger.debug(
            "Step BashEnvironment (task_id=%s, action_type=%s)",
            self._task_id,
            action.action_type,
        )

        self._state.step_count += 1
        self._state.last_action_type = action.action_type

        output = ""
        error = ""
        success = True
        reward: float | None = None
        done = False
        metadata: dict[str, Any] = {}

        try:
            if action.action_type == "exec":
                timeout_value = (
                    action.timeout_s
                    if action.timeout_s is not None
                    else (timeout_s or self.command_timeout_s)
                )
                logger.debug(
                    "Executing command (timeout_s=%s): %s",
                    timeout_value,
                    action.command,
                )
                result = subprocess.run(
                    ["bash", "-lc", action.command],
                    cwd=str(self._workdir),
                    capture_output=True,
                    text=True,
                    timeout=timeout_value,
                )
                output = (result.stdout or "") + (result.stderr or "")
                success = result.returncode == 0
                error = "" if success else (result.stderr or "")
                self._state.last_command = action.command

            elif action.action_type == "submit":
                logger.debug("Submitting answer: %s", action.answer)
                normalized_answer = _normalize_answer(action.answer)
                normalized_expected = _normalize_answer(self._expected_answer)
                reward = 1.0 if normalized_answer == normalized_expected else 0.0
                done = True
                metadata = {"correct": reward == 1.0}

            elif action.action_type == "close":
                logger.debug("Closing BashEnvironment session")
                done = True
                output = "Closed Bash Env environment."

            else:
                raise ValueError(f"Unsupported action_type: {action.action_type}")

        except Exception as exc:  # pragma: no cover
            success = False
            error = str(exc)
            logger.exception("BashEnvironment step failed: %s", exc)

        self._state.last_output = output

        return BashObservation(
            instruction=self._instruction,
            output=output,
            success=success,
            error=error,
            task_id=self._task_id,
            action_type=action.action_type,
            reward=reward,
            done=done,
            metadata=metadata,
        )

    @property
    def state(self) -> BashState:
        return self._state

    def close(self) -> None:
        logger.info("Closing BashEnvironment (task_id=%s)", self._task_id)
        self._workdir = None
        self._task_id = None
        self._instruction = ""
        self._expected_answer = ""
