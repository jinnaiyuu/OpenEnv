---
title: Bash Env Environment Server
emoji: ":computer:"
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - bash
---

# Bash Env Environment

Local terminal environment with task instructions and script-evaluated rewards.
Each episode provides an instruction and an empty working directory. The agent
uses shell commands to work, writes a solution script, then submits to run tests.

## Quick Start

```python
from bash_env import BashAction, BashEnv

env = BashEnv(base_url="http://localhost:8000")
result = env.reset(task_id="hello_world")
print(result.observation.instruction)

result = env.step(
  BashAction(
    action_type="exec",
    command="""
cat > solve.sh << 'SH'
#!/usr/bin/env bash
printf 'Hello, World!'
SH
""",
  )
)
print(result.observation.output)

result = env.step(BashAction(action_type="submit"))
print(result.reward, result.done)

env.close()
```

## Task Format (JSONL)

Each line in the tasks file defines one task. Prefer script-based evaluation
with `script_names` and `test_cases`:

```json
{
  "task_id": "hello_world",
  "instruction": "...",
  "script_names": ["solve.sh", "solve.py"],
  "test_cases": [{"input": "", "output": "Hello, World!"}]
}

Legacy string-matching tasks are still supported:

{"task_id": "hello_world", "instruction": "...", "expected_answer": "..."}
```

Configure the tasks file with `BASH_TASKS_PATH`.

## Actions

- `exec`: run a shell command in the episode workdir
- `submit`: run the solution script against hidden tests
- `close`: end the session

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BASH_TASKS_PATH` | envs/bash_env/tasks/tasks.jsonl | Path to JSONL task file |
| `BASH_OUTPUT_DIR` | /tmp/bash_env_runs | Base directory for per-episode workdirs |
| `BASH_COMMAND_TIMEOUT_S` | 20.0 | Default command timeout (seconds) |
