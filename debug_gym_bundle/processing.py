"""Per-job execution flow for Debug-Gym bug generation runs."""

from __future__ import annotations

import copy
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from textwrap import shorten
from typing import Any, Iterable

from debug_gym.agents.free_agent import FreeAgent
from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.terminals import select_terminal
from debug_gym.gym.terminals.terminal import Terminal
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.llms.base import LLM
from debug_gym.logger import DebugGymLogger

from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS
from swesmith.constants import LOG_DIR_RUN_VALIDATION, KEY_IMAGE_NAME

from .config import DebugGymSessionConfig
from .issue_generation import CustomIssueGen, _generate_issue_payload
from .utils import (
    _cleanup_previous_outputs,
    assess_validation_report,
    create_instance_id,
    derive_agent_seed,
    extract_repo_commit,
    LOG_REPORT,
    remove_added_test_files,
)
from .validation import ensure_validation_registry_support, run_validation

logger = logging.getLogger(__name__)

JobSpec = tuple[str, str]

def _build_terminal(
    image_name: str,
    workspace_dir: str,
    setup_commands: tuple[str, ...],
    terminal_setting: Terminal | str | dict[str, Any] | None,
    overrides: dict[str, Any],
    logger: DebugGymLogger,
) -> Terminal | None:
    if isinstance(terminal_setting, Terminal):
        return terminal_setting

    if terminal_setting is None:
        terminal_config: dict[str, Any] = {"type": "docker"}
    elif isinstance(terminal_setting, str):
        terminal_config = {"type": terminal_setting}
    else:
        terminal_config = dict(terminal_setting)

    terminal_config = {**terminal_config, **overrides}
    terminal_config.setdefault("type", "docker")
    terminal_config["type"] = str(terminal_config["type"]).lower()
    terminal_config.setdefault("base_image", image_name)
    terminal_config.setdefault("working_dir", workspace_dir)

    if setup_commands:
        terminal_config.setdefault("setup_commands", list(setup_commands))

    return select_terminal(terminal_config, logger=logger)


def _add_tools(
    env: FreeEnv,
    tool_specs: Iterable[str | dict[str, Any]],
    logger: DebugGymLogger,
) -> None:
    for spec in tool_specs:
        tool_kwargs: dict[str, Any] = {}
        if isinstance(spec, dict):
            if len(spec) != 1:
                raise ValueError("Tool configuration entries must contain exactly one tool name")
            name, options = next(iter(spec.items()))
            tool_name = str(name).strip()
            if not tool_name:
                raise ValueError("Tool name in mapping cannot be empty")
            tool_kwargs = dict(options or {})
        else:
            tool_name = str(spec).strip()

        if not tool_name:
            raise ValueError("Tool names cannot be empty")

        if tool_name == "submit" and "eval_on_submit" not in tool_kwargs:
            tool_kwargs = {**tool_kwargs, "eval_on_submit": False}

        try:
            env.add_tool(Toolbox.get_tool(tool_name, **tool_kwargs))
        except ValueError as exc:  # pragma: no cover - validation guard
            raise RuntimeError(
                f"Failed to load tool '{tool_name}': {exc}"
            ) from exc
        logger.debug("Added tool %s with options %s", tool_name, tool_kwargs)


def process_single_job(
    jspec: JobSpec,
    logdir: Path | str,
    model_name: str,
    run_id: str,
    issue_generator: CustomIssueGen,
    session_config: DebugGymSessionConfig,
    validation_timeout: int | None,
    max_fail_fraction: float,
) -> tuple[dict[str, Any] | None, bool, str | None]:
    """Generate, validate, and describe a single potential bug instance."""

    env: FreeEnv | None = None
    debug_logger: DebugGymLogger | None = None
    image_name, seed = jspec
    jid = jspec
    instance_id = create_instance_id(image_name=image_name, seed=seed)

    try:
        logger.info("Starting job for %s", jid)

        logdir = Path(logdir)
        image_output_dir = logdir / image_name / f"seed_{seed}"
        image_output_dir.mkdir(exist_ok=True, parents=True)
        _cleanup_previous_outputs(image_output_dir)

        repo_name, short_commit_sha = extract_repo_commit(image_name)

        debug_logger = DebugGymLogger(
            f"buggen:{shorten(instance_id, width=40)}",
            log_dir=str(image_output_dir / "debug_gym_logs"),
        )
        debug_logger.setLevel(logging.DEBUG)

        terminal = _build_terminal(
            image_name=image_name,
            workspace_dir=session_config.env_workspace_dir,
            setup_commands=session_config.env_setup_commands,
            terminal_setting=session_config.env_terminal,
            overrides=session_config.env_terminal_kwargs,
            logger=debug_logger,
        )

        env = FreeEnv(
            image=image_name,
            terminal=terminal,
            mount_path=None,
            setup_commands=list(session_config.env_setup_commands),
            instructions=session_config.env_instructions,
            init_git=session_config.env_init_git,
            workspace_dir=session_config.env_workspace_dir,
            logger=debug_logger,
            dir_tree_depth=session_config.env_dir_tree_depth,
        )

        _add_tools(env, session_config.tools, debug_logger)

        llm = LLM.instantiate(
            llm_name=model_name,
            logger=debug_logger,
        )
        if llm is None:
            raise RuntimeError(f"Failed to instantiate LLM '{model_name}'")

        agent_config = copy.deepcopy(session_config.agent_config)
        agent_config["output_path"] = str(image_output_dir / "debug_gym_runs")
        agent_config["random_seed"] = derive_agent_seed(seed)

        agent = FreeAgent(
            config=agent_config,
            env=env,
            llm=llm,
            logger=debug_logger,
        )

        resolved = agent.run(task_name=instance_id)
        debug_logger.info("Agent run completed. Resolved=%s", resolved)

        agent.save_trajectory(task_name=instance_id)
        agent.save_patch(task_name=instance_id)

        agent_output_dir = Path(agent_config["output_path"]) / agent._uuid
        trajectory_src = agent_output_dir / instance_id / "trajectory.json"
        patch_src = agent_output_dir / instance_id / "debug_gym.patch"

        if trajectory_src.exists():
            shutil.copy2(trajectory_src, image_output_dir / "trajectory.json")

        if patch_src.exists():
            shutil.copy2(patch_src, image_output_dir / "debug_gym.patch")
            patch_text = patch_src.read_text()
        else:
            patch_text = env.patch

        if not patch_text or patch_text.strip() == "":
            return None, False, "Debug-Gym agent produced empty patch"

        patch_text = remove_added_test_files(patch_text)

        logger.info(
            "Successfully generated patch for %s with seed %s. Validating generated bug.",
            image_name,
            seed,
        )
        report_path = LOG_DIR_RUN_VALIDATION / run_id / instance_id / LOG_REPORT

        registry_value = session_config.env_terminal_kwargs.get("registry")
        if (not registry_value) and isinstance(session_config.env_terminal, dict):
            registry_value = session_config.env_terminal.get("registry")
        if registry_value is not None:
            registry_value = str(registry_value).strip()
            if not registry_value:
                registry_value = None

        ensure_validation_registry_support()

        if not report_path.exists():
            instance_data = {
                "strategy": "debuggym",
                "instance_id": instance_id,
                "patch": patch_text,
                "image_name": image_name,
            }
            if registry_value is not None:
                instance_data["image_registry"] = registry_value

            try:
                run_kwargs = {
                    "instance": instance_data,
                    "run_id": run_id,
                    "run_min_pregold": True,
                }
                if validation_timeout is not None:
                    run_kwargs["timeout"] = validation_timeout
                run_validation(**run_kwargs)
            except Exception as exc:  # pragma: no cover - defensive guard
                message = f"Validation routine raised unexpected error: {exc}"
                logger.exception(message)
                return (None, False, f"non_retryable: {message}")

        if not report_path.exists():
            logger.info("Could not find validation run report for %s", jid)
            return None, False, "Could not find validation run report"

        logger.info("Found report after running validation check for %s", jid)
        try:
            with report_path.open("r", encoding="utf-8") as report_handle:
                report = json.load(report_handle)
        except json.JSONDecodeError as exc:
            message = (
                "Validation report is not valid JSON; treating as non-retryable: "
                f"{exc}"
            )
            logger.error(message)
            return (None, False, f"non_retryable: {message}")
        is_buggy, f2p, p2p, rejection_msg = assess_validation_report(
            report,
            max_fail_fraction=max_fail_fraction,
        )
        if not is_buggy:
            message = rejection_msg or "Validation rejected"
            logger.info("Rejected %s: %s", jid, message)
            return None, False, message

        instance_data = {
            "instance_id": instance_id,
            "repo": f"swesmith/{repo_name}.{short_commit_sha}",
            "patch": patch_text,
            FAIL_TO_PASS: f2p,
            PASS_TO_PASS: p2p,
            "created_at": datetime.now().isoformat(),
            "image_name": image_name,
            "agent_resolved": resolved,
            "agent_uuid": agent._uuid,
        }
        if registry_value is not None:
            instance_data["image_registry"] = registry_value

        logger.info("Successfully analysed validation report for %s", jid)
        logger.info("Generating problem description text for %s", jid)

        try:
            logger.info("Calling issue_generator.generate_issue for %s", jid)
            instance_data = _generate_issue_payload(issue_generator, instance_data)
            logger.info("Successfully generated issue for %s", jid)
        except Exception as err:  # pragma: no cover - defensive logging
            logger.exception("Error generating issue for %s: %s", jid, err)
            return (None, False, f"Error generating issue: {err}")

        return (instance_data, True, None)

    except Exception as err:  # pragma: no cover - defensive logging
        error_msg = f"Error processing {jid}: {err}"
        logger.exception(error_msg)
        return (None, False, error_msg)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # pragma: no cover - best effort close
                logger.warning("Failed to close environment for %s: %s", jid, exc)
        if debug_logger is not None:
            debug_logger.close()