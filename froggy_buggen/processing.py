"""Per-job execution flow for Froggy bug generation runs."""

from __future__ import annotations

import copy
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from textwrap import shorten
from typing import Any

from debug_gym.agents import FroggyAgent
from debug_gym.agents.utils import save_patch, save_trajectory
from debug_gym.experiment import add_tools
from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.terminals import select_terminal
from debug_gym.gym.terminals.terminal import Terminal
from debug_gym.llms.base import LLM
from debug_gym.logger import DebugGymLogger

from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS
from swesmith.constants import LOG_DIR_RUN_VALIDATION

from .config import FroggySessionConfig
from .issue_generation import CustomIssueGen
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
    terminal_setting: Terminal | str | dict[str, Any] | None,
    overrides: dict[str, Any],
    logger: DebugGymLogger,
) -> Terminal | None:
    """Build a terminal instance from configuration.

    Note: setup_commands should NOT be passed here since FreeEnv handles them
    in setup_terminal(). Passing them to both would cause duplicate execution.
    """
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

    return select_terminal(terminal_config, logger=logger)


def process_single_job(
    jspec: JobSpec,
    logdir: Path | str,
    model_name: str,
    run_id: str,
    issue_generator: CustomIssueGen,
    session_config: FroggySessionConfig,
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
            terminal_setting=session_config.env_terminal,
            overrides=session_config.env_terminal_kwargs,
            logger=debug_logger,
        )

        # setup_commands are passed only to FreeEnv which executes them in setup_terminal().
        # They should NOT be passed to the terminal to avoid duplicate execution.
        # Only pass setup_commands if explicitly configured; otherwise let FreeEnv use its default
        # (which installs git).
        env_kwargs: dict[str, Any] = {
            "image": image_name,
            "terminal": terminal,
            "workspace_dir": session_config.env_workspace_dir,
            "logger": debug_logger,
        }
        if session_config.env_setup_commands:
            env_kwargs["setup_commands"] = list(session_config.env_setup_commands)

        env = FreeEnv(**env_kwargs)

        # Use debug_gym's add_tools with config dict format
        add_tools(env, {"tools": list(session_config.tools)}, debug_logger)

        llm = LLM.instantiate(
            config={"name": model_name},
            logger=debug_logger,
        )
        if llm is None:
            raise RuntimeError(f"Failed to instantiate LLM '{model_name}'")

        agent_config = copy.deepcopy(session_config.agent_config)
        agent_config["random_seed"] = derive_agent_seed(seed)

        agent = FroggyAgent(agent_args=agent_config, logger=debug_logger)

        # Run the agent - resolved status is on env after run completes
        agent.run(env, llm)
        resolved = env.resolved
        debug_logger.info("Agent run completed. Resolved=%s", resolved)

        # Save trajectory and patch using utility functions
        problem_path = image_output_dir / "debug_gym_runs" / agent.args.uuid / instance_id
        problem_path.mkdir(parents=True, exist_ok=True)
        save_trajectory(agent, problem_path, debug_logger)
        save_patch(env, problem_path, debug_logger)

        trajectory_src = problem_path / "trajectory.json"
        patch_src = problem_path / "debug_gym.patch"

        if trajectory_src.exists():
            shutil.copy2(trajectory_src, image_output_dir / "trajectory.json")

        if patch_src.exists():
            shutil.copy2(patch_src, image_output_dir / "debug_gym.patch")
            patch_text = patch_src.read_text()
        else:
            patch_text = env.patch

        if not patch_text or patch_text.strip() == "":
            return None, False, "Agent produced empty patch"

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
            "agent_uuid": agent.args.uuid,
        }
        if registry_value is not None:
            instance_data["image_registry"] = registry_value

        logger.info("Successfully analysed validation report for %s", jid)
        logger.info("Generating problem description text for %s", jid)

        try:
            logger.info("Calling issue_generator.generate_issue for %s", jid)
            instance_data = issue_generator.generate_issue(instance_data)
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