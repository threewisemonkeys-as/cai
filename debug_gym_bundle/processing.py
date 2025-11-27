"""Per-job execution flow for Debug-Gym bug generation runs."""

from __future__ import annotations

import copy
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from textwrap import shorten
from typing import Any

from debug_gym.agents.free_agent import FreeAgent
from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.llms.base import LLM
from debug_gym.logger import DebugGymLogger

from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS

from .config import DebugGymSessionConfig
from .issue_generation import CustomIssueGen, _generate_issue_payload
from .utils import (
    _cleanup_previous_outputs,
    assess_validation_report,
    create_instance_id,
    derive_agent_seed,
    remove_added_test_files,
)

logger = logging.getLogger(__name__)

JobSpec = tuple[str, str]


def process_single_job(
    jspec: JobSpec,
    logdir: Path | str,
    model_name: str,
    run_id: str,
    issue_generator: CustomIssueGen,
    session_config: DebugGymSessionConfig,
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

        _, _, repo_name, short_commit_sha = image_name.split(".")

        debug_logger = DebugGymLogger(
            f"buggen:{shorten(instance_id, width=40)}",
            log_dir=str(image_output_dir / "debug_gym_logs"),
        )
        debug_logger.setLevel(logging.INFO)

        env = FreeEnv(
            image=image_name,
            terminal=session_config.env_terminal,
            mount_path=None,
            setup_commands=list(session_config.env_setup_commands),
            instructions=session_config.env_instructions,
            init_git=session_config.env_init_git,
            workspace_dir=session_config.env_workspace_dir,
            logger=debug_logger,
            terminal_kwargs=session_config.env_terminal_kwargs,
            dir_tree_depth=session_config.env_dir_tree_depth,
        )

        for tool_name in session_config.tools:
            try:
                env.add_tool(Toolbox.get_tool(tool_name))
            except ValueError as exc:
                raise RuntimeError(
                    f"Failed to load tool '{tool_name}': {exc}"
                ) from exc

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
        from swesmith.constants import LOG_DIR_RUN_VALIDATION, LOG_REPORT

        report_path = LOG_DIR_RUN_VALIDATION / run_id / instance_id / LOG_REPORT

        if not report_path.exists():
            instance_data = {
                "strategy": "debuggym",
                "instance_id": instance_id,
                "patch": patch_text,
                "image_name": image_name,
            }
            from swesmith.harness.valid import run_validation

            run_validation(
                instance=instance_data,
                run_id=run_id,
                run_min_pregold=True,
            )

        if not report_path.exists():
            logger.info("Could not find validation run report for %s", jid)
            return None, False, "Could not find validation run report"

        logger.info("Found report after running validation check for %s", jid)
        with report_path.open("r", encoding="utf-8") as report_handle:
            report = json.load(report_handle)
        is_buggy, f2p, p2p, rejection_msg = assess_validation_report(report)
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