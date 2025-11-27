"""Configuration loading utilities for the Debug-Gym bug generation pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CUR_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = CUR_DIR / "debug_gym_buggen.yaml"


@dataclass(frozen=True)
class DebugGymSessionConfig:
    """Runtime configuration for FreeEnv, FreeAgent, and supporting tools."""

    llm_name: str | None
    tools: tuple[str | dict[str, Any], ...]
    env_terminal: str | dict[str, Any] | None
    env_workspace_dir: str
    env_instructions: str
    env_setup_commands: tuple[str, ...]
    env_terminal_kwargs: dict[str, Any]
    env_dir_tree_depth: int
    env_init_git: bool
    agent_config: dict[str, Any]
    issue_gen_config: Path


@dataclass(frozen=True)
class BuggenRuntimeConfig:
    """Execution parameters that previously came from CLI arguments."""

    images_file: Path
    output_file: Path
    logdir: Path
    run_id: str
    seed_per_image: int
    max_workers: int
    max_tries: int
    shuffle: bool
    validation_timeout: int | None
    max_fail_fraction: float


def _resolve_path(base: Path, maybe_path: str | None) -> str | None:
    """Resolve ``maybe_path`` relative to ``base`` when not absolute."""

    if maybe_path is None:
        return None
    path = Path(maybe_path)
    if not path.is_absolute():
        path = (base.parent / path).resolve()
    return str(path)


def load_pipeline_config(
    config_path: str | Path | None,
) -> tuple[DebugGymSessionConfig, BuggenRuntimeConfig]:
    """Load environment, agent, and runtime parameters from YAML configuration."""

    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Debug-Gym buggen config not found: {cfg_path}")

    config_data = yaml.safe_load(cfg_path.read_text()) or {}

    llm_name = config_data.get("llm") or config_data.get("model_name")
    if llm_name is not None:
        llm_name = str(llm_name).strip() or None

    tools_cfg = config_data.get("tools")
    if not tools_cfg:
        raise ValueError("Configuration must specify a non-empty 'tools' list.")
    if not isinstance(tools_cfg, (list, tuple)):
        raise ValueError("Configuration 'tools' must be a list of tool names or mappings.")

    normalized_tools: list[str | dict[str, Any]] = []
    for entry in tools_cfg:
        if isinstance(entry, str):
            tool_name = entry.strip()
            if tool_name:
                normalized_tools.append(tool_name)
            continue

        if isinstance(entry, dict):
            if len(entry) != 1:
                raise ValueError("Each tool mapping must contain exactly one tool name")
            name, options = next(iter(entry.items()))
            tool_name = str(name).strip()
            if not tool_name:
                raise ValueError("Tool name in mapping cannot be empty")
            if options is None:
                normalized_tools.append({tool_name: {}})
            else:
                if not isinstance(options, dict):
                    raise ValueError(
                        f"Configuration for tool '{tool_name}' must be a mapping of options"
                    )
                normalized_tools.append({tool_name: dict(options)})
            continue

        raise ValueError("Tool entries must be strings or single-key mappings")

    if not normalized_tools:
        raise ValueError(
            "Configuration produced an empty tool list after processing entries."
        )

    tools = tuple(normalized_tools)

    env_cfg = config_data.get("environment")
    if not isinstance(env_cfg, dict):
        raise ValueError("Configuration must include an 'environment' mapping.")

    terminal_cfg = env_cfg.get("terminal")
    if terminal_cfg is None:
        terminal: str | dict[str, Any] | None = "docker"
    elif isinstance(terminal_cfg, str):
        terminal = terminal_cfg.strip() or "docker"
    elif isinstance(terminal_cfg, dict):
        terminal = dict(terminal_cfg)
    else:
        raise ValueError(
            "environment.terminal must be a string, mapping, or omitted.",
        )

    workspace_cfg = env_cfg.get("workspace_dir")
    if workspace_cfg is None:
        workspace_dir = "/testbed"
    else:
        workspace_dir = str(workspace_cfg).strip() or "/testbed"

    instructions = env_cfg.get("instructions")
    instructions_file = env_cfg.get("instructions_file")
    if instructions and instructions_file:
        raise ValueError(
            "Specify either environment.instructions or environment.instructions_file, not both."
        )
    if instructions_file:
        resolved_instructions_path = _resolve_path(cfg_path, instructions_file)
        if resolved_instructions_path is None or not Path(resolved_instructions_path).exists():
            raise FileNotFoundError(
                f"Instructions file not found: {resolved_instructions_path}"
            )
        instructions = Path(resolved_instructions_path).read_text()
    if not instructions:
        raise ValueError(
            "Environment configuration must provide either inline 'instructions' or a valid 'instructions_file'."
        )

    setup_commands = env_cfg.get("setup_commands")
    if setup_commands is None:
        setup_commands_tuple: tuple[str, ...] = ()
    elif isinstance(setup_commands, str):
        setup_commands_tuple = (setup_commands.strip(),) if setup_commands.strip() else ()
    else:
        setup_commands_tuple = tuple(
            str(cmd).strip() for cmd in setup_commands if str(cmd).strip()
        )

    terminal_kwargs = env_cfg.get("terminal_kwargs") or {}
    if not isinstance(terminal_kwargs, dict):
        raise ValueError("environment.terminal_kwargs must be a mapping")

    dir_tree_depth = env_cfg.get("dir_tree_depth", 1)
    try:
        dir_tree_depth = int(dir_tree_depth)
    except (TypeError, ValueError) as exc:
        raise ValueError("environment.dir_tree_depth must be an integer") from exc

    init_git_raw = env_cfg.get("init_git", True)
    if isinstance(init_git_raw, str):
        init_git = init_git_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        init_git = bool(init_git_raw)

    agent_cfg_raw = config_data.get("agent")
    if not isinstance(agent_cfg_raw, dict):
        raise ValueError("Configuration must include an 'agent' mapping.")
    agent_cfg = dict(agent_cfg_raw)

    for numeric_key in ("max_steps", "max_rewrite_steps"):
        if numeric_key not in agent_cfg:
            raise KeyError(f"Agent configuration missing required key '{numeric_key}'.")
        agent_cfg[numeric_key] = int(agent_cfg[numeric_key])

    memory_size_raw = agent_cfg.get("memory_size", agent_cfg["max_steps"])
    agent_cfg["memory_size"] = int(memory_size_raw)

    system_prompt = agent_cfg.get("system_prompt")
    system_prompt_file = agent_cfg.get("system_prompt_file")
    if system_prompt and system_prompt_file:
        raise ValueError(
            "Agent configuration must not specify both 'system_prompt' and 'system_prompt_file'."
        )

    if system_prompt_file:
        resolved_prompt_path = _resolve_path(cfg_path, system_prompt_file)
        if resolved_prompt_path is None or not Path(resolved_prompt_path).exists():
            raise FileNotFoundError(
                f"Agent system prompt file not found: {resolved_prompt_path}"
            )
        agent_cfg["system_prompt"] = Path(resolved_prompt_path).read_text()
    elif system_prompt is not None:
        text_value = str(system_prompt)
        if text_value.strip():
            agent_cfg["system_prompt"] = text_value
        else:
            agent_cfg.pop("system_prompt", None)

    agent_cfg.pop("system_prompt_file", None)

    issue_gen_config_value = config_data.get("issue_gen_config")
    if not issue_gen_config_value:
        raise ValueError("Configuration must specify 'issue_gen_config'.")

    resolved_issue_gen = _resolve_path(cfg_path, issue_gen_config_value)
    if resolved_issue_gen is None or not Path(resolved_issue_gen).exists():
        raise FileNotFoundError(
            f"Issue generator config file not found: {resolved_issue_gen}"
        )
    issue_gen_config_resolved = Path(resolved_issue_gen)

    run_cfg = config_data.get("run")
    if not isinstance(run_cfg, dict):
        raise ValueError("Configuration must include a 'run' mapping.")

    images_path = run_cfg.get("images")
    if not images_path:
        raise ValueError("run.images must specify a file containing image names.")
    images_path_resolved = _resolve_path(cfg_path, images_path)
    if images_path_resolved is None or not Path(images_path_resolved).exists():
        raise FileNotFoundError(f"Images list file not found: {images_path_resolved}")

    output_file = run_cfg.get("output_file")
    if not output_file:
        raise ValueError("run.output_file must be provided.")
    output_file_resolved = Path(_resolve_path(cfg_path, output_file) or output_file).resolve()

    logdir_value = run_cfg.get("logdir")
    if not logdir_value:
        raise ValueError("run.logdir must be provided.")
    logdir_resolved = Path(_resolve_path(cfg_path, logdir_value) or logdir_value).resolve()

    run_id = str(run_cfg.get("run_id") or datetime.now().strftime("%Y%m%d-%H%M%S"))

    def _as_int(name: str, default: int) -> int:
        raw = run_cfg.get(name, default)
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:  # pragma: no cover - validation
            raise ValueError(f"run.{name} must be an integer") from exc

    seed_per_image = _as_int("seed_per_image", 1)
    max_workers = _as_int("max_workers", 1)
    max_tries = _as_int("max_tries", 1)

    shuffle_raw = run_cfg.get("shuffle", False)
    if isinstance(shuffle_raw, str):
        shuffle = shuffle_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        shuffle = bool(shuffle_raw)

    timeout_value = run_cfg.get("validation_timeout")
    validation_timeout: int | None
    if timeout_value in (None, "", False):
        validation_timeout = None
    else:
        try:
            validation_timeout = int(timeout_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("run.validation_timeout must be an integer") from exc

    if "max_fail_fraction" not in run_cfg:
        raise ValueError("run.max_fail_fraction must be provided in the configuration")
    max_fail_fraction_raw = run_cfg.get("max_fail_fraction")
    try:
        max_fail_fraction = float(max_fail_fraction_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("run.max_fail_fraction must be a floating point value") from exc
    if not 0 < max_fail_fraction <= 1:
        raise ValueError("run.max_fail_fraction must be between 0 and 1 (inclusive of 1)")

    session_config = DebugGymSessionConfig(
        llm_name=llm_name,
        tools=tools,
        env_terminal=terminal,
        env_workspace_dir=workspace_dir,
        env_instructions=instructions,
        env_setup_commands=setup_commands_tuple,
        env_terminal_kwargs=dict(terminal_kwargs),
        env_dir_tree_depth=dir_tree_depth,
        env_init_git=init_git,
        agent_config=agent_cfg,
        issue_gen_config=issue_gen_config_resolved,
    )

    runtime_config = BuggenRuntimeConfig(
        images_file=Path(images_path_resolved),
        output_file=output_file_resolved,
        logdir=logdir_resolved,
        run_id=run_id,
        seed_per_image=seed_per_image,
        max_workers=max_workers,
        max_tries=max_tries,
        shuffle=shuffle,
        validation_timeout=validation_timeout,
        max_fail_fraction=max_fail_fraction,
    )

    return session_config, runtime_config
