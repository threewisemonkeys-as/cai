"""Debug-Gym powered bug generation pipeline.

This module orchestrates the end-to-end "buggen" flow using Debug-Gym.
It prepares execution environments, drives the FreeAgent, validates the
resulting patch with SWE-bench tooling, and produces human-readable issue
reports. The CLI entry point is ``regular``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import shutil
import tempfile
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import shorten
from typing import Any

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from unidiff import PatchSet

from debug_gym.agents.free_agent import FreeAgent
from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.llms.base import LLM
from debug_gym.logger import DebugGymLogger

from swebench.harness.constants import FAIL_TO_PASS, LOG_REPORT, PASS_TO_PASS
from swesmith.constants import KEY_TIMED_OUT, LOG_DIR_RUN_VALIDATION
from swesmith.harness.valid import run_validation
from swesmith.issue_gen.generate import IssueGen

load_dotenv()

CUR_DIR = Path(__file__).parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = CUR_DIR / "debug_gym_buggen.yaml"

ISSUE_GEN_CONFIG_FILE_PATH = CUR_DIR / Path("sans_patch_issue_gen.yaml")

JobSpec = tuple[str, str]


class CustomIssueGen(IssueGen):
    """Issue generator that skips cloning test repositories for Debug-Gym outputs."""

    def __init__(
        self,
        model: str,
        use_existing: bool,
        n_workers: int,
        experiment_id: Path | str,
    ):
        self.experiment_id = Path(experiment_id)
        self.model = model
        self.use_existing = use_existing
        self.n_workers = n_workers

        # The SWE-bench dataset is required for prompt templates and metadata.
        self.swebv = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

        self.config = yaml.safe_load(ISSUE_GEN_CONFIG_FILE_PATH.read_text())
        settings = self.config.get("settings", {})
        self.n_instructions = settings.get("n_instructions", 1)
        self.max_var_tokens = settings.get("max_var_tokens", 10_000)

        self._lock = threading.Lock()

    def get_test_functions(self, instance: dict) -> tuple[list[str], list[str]]:
        """Avoid cloning missing repositories when generating issues."""

        return [], []


@dataclass(frozen=True)
class DebugGymSessionConfig:
    """Runtime configuration for FreeEnv, FreeAgent, and supporting tools."""

    llm_name: str | None
    tools: tuple[str, ...]
    env_terminal: str
    env_workspace_dir: str
    env_instructions: str
    env_setup_commands: tuple[str, ...]
    env_terminal_kwargs: dict[str, Any]
    env_dir_tree_depth: int
    env_init_git: bool
    agent_config: dict[str, Any]


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


def remove_added_test_files(patch: str) -> str:
    """Strip any newly added test files from the generated patch."""

    return str(
        PatchSet(
            [
                str(file_patch)
                for file_patch in PatchSet(patch)
                if not (
                    file_patch.is_added_file
                    and file_patch.path.endswith(".py")
                    and "test_" in file_patch.path
                )
            ]
        )
    )


def create_instance_id(image_name: str, seed: str) -> str:
    """Create a unique SWE-bench style identifier for this run."""

    _, _, repo_name, short_commit_sha = image_name.split(".")
    return f"{repo_name}.{short_commit_sha}.debuggym.seed_{seed}"


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

    tools = config_data.get("tools")
    if not tools:
        raise ValueError("Configuration must specify a non-empty 'tools' list.")
    tools = tuple(str(tool).strip() for tool in tools if str(tool).strip())
    if not tools:
        raise ValueError("Configuration produced an empty tool list after stripping values.")

    env_cfg = config_data.get("environment")
    if not isinstance(env_cfg, dict):
        raise ValueError("Configuration must include an 'environment' mapping.")

    terminal_cfg = env_cfg.get("terminal")
    terminal = str(terminal_cfg).strip() if terminal_cfg is not None else "docker"
    if not terminal:
        terminal = "docker"

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
    )

    return session_config, runtime_config


def derive_agent_seed(seed: str) -> int:
    """Derive a stable integer seed for the agent from a UUID-like string."""

    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _cleanup_previous_outputs(output_dir: Path) -> None:
    """Remove artifacts from previous runs to avoid mixing state."""

    for stale_file in ("debug_gym.patch", "trajectory.json"):
        candidate = output_dir / stale_file
        if candidate.exists():
            candidate.unlink()
    for stale_dir in ("debug_gym_runs", "debug_gym_logs"):
        candidate_dir = output_dir / stale_dir
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)


def _load_existing_results(output_file: Path) -> list[dict[str, Any]]:
    """Read any previously saved results, guarding against decode failures."""

    if not output_file.exists():
        return []
    try:
        with output_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if not isinstance(data, list):
                logger.warning("Existing results file is not a list; starting fresh.")
                return []
            return data
    except json.JSONDecodeError:
        logger.warning("Existing results file is corrupt; starting fresh.")
        return []


def _persist_results(output_file: Path, results: list[dict[str, Any]]) -> None:
    """Write the accumulated results list to disk."""

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def _record_success(
    result: dict[str, Any],
    *,
    output_file: Path,
    results: list[dict[str, Any]],
    lock: threading.Lock,
) -> None:
    """Append a new successful instance in a thread-safe manner."""

    with lock:
        results.append(result)
        _persist_results(output_file, results)


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
        logger.info(f"Starting job for {jid}")

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
        debug_logger.info(f"Agent run completed. Resolved={resolved}")

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
            f"Successfully generated patch for {image_name} with seed {seed}. "
            "Validating generated bug."
        )
        report_path = LOG_DIR_RUN_VALIDATION / run_id / instance_id / LOG_REPORT

        if not report_path.exists():
            instance_data = {
                "strategy": "debuggym",
                "instance_id": instance_id,
                "patch": patch_text,
                "image_name": image_name,
            }
            run_validation(
                instance=instance_data,
                run_id=run_id,
                run_min_pregold=True,
            )

        if not report_path.exists():
            logger.info(f"Could not find validation run report for {jid}")
            return None, False, "Could not find validation run report"

        logger.info(f"Found report after running validation check for {jid}")
        with report_path.open("r", encoding="utf-8") as report_handle:
            report = json.load(report_handle)
        f2p, p2p = report[FAIL_TO_PASS], report[PASS_TO_PASS]
        if KEY_TIMED_OUT in report or len(f2p) == 0 or len(p2p) == 0:
            logger.info(f"Generated patch for {jid} not buggy.")
            return None, False, "Generated patch not buggy"

        if len(f2p) > 5:
            logger.info(
                f"Generated bug results in more than 5 failing tests for {jid}"
            )
            return None, False, "Too many failing tests"

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

        logger.info(f"Successfully analysed validation report for {jid}")
        logger.info(f"Generating problem description text for {jid}")

        try:
            with tempfile.NamedTemporaryFile(delete_on_close=False, mode="w+") as fp:
                logger.info(f"Calling issue_generator.generate_issue for {jid}")
                issue_generator.generate_issue(instance_data, 0, fp)
                fp.flush()
                fp.seek(0)
                instance_data = json.load(fp)
                logger.info(f"Successfully generated issue for {jid}")
        except Exception as err:  # pragma: no cover - defensive logging
            logger.error(f"Error generating issue for {jid}: {err}")
            logger.error(traceback.format_exc())
            return (None, False, f"Error generating issue: {err}")

        return (instance_data, True, None)

    except Exception as err:  # pragma: no cover - defensive logging
        traceback.print_exc()
        error_msg = f"Error processing {jid}: {err}"
        logger.error(error_msg)
        return (None, False, error_msg)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # pragma: no cover - best effort close
                logger.warning(f"Failed to close environment for {jid}: {exc}")
        if debug_logger is not None:
            debug_logger.close()


def regular(
    config: str | Path | None = None,
) -> dict[str, int]:
    """Entry point for the CLI: parameters now sourced entirely from YAML."""

    session_config, runtime_config = load_pipeline_config(config)

    model_name = (session_config.llm_name or "").strip()
    if not model_name:
        raise ValueError("Model name is required. Set 'llm' in the YAML config.")

    image_names = runtime_config.images_file.read_text().splitlines()
    jobs_specs: list[JobSpec] = [
        (image_name, str(uuid.uuid4())[:10])
        for image_name in image_names
        for _ in range(runtime_config.seed_per_image)
    ]

    if runtime_config.shuffle:
        random.shuffle(jobs_specs)

    output_file = runtime_config.output_file
    output_file.parent.mkdir(exist_ok=True, parents=True)

    existing_results = _load_existing_results(output_file)
    existing_instance_ids = {entry["instance_id"] for entry in existing_results}
    jobs_specs = [
        jspec
        for jspec in jobs_specs
        if create_instance_id(*jspec) not in existing_instance_ids
    ]

    logger.info(
        f"Processing {len(jobs_specs)} jobs with {runtime_config.max_workers} workers, "
        f"max {runtime_config.max_tries} tries per job using Debug-Gym pipeline."
    )

    logger.info("Initializing shared issue generator (loading SWE-bench dataset)...")
    shared_issue_generator = CustomIssueGen(
        model=model_name,
        use_existing=True,
        n_workers=1,
        experiment_id=runtime_config.run_id,
    )
    logger.info("Issue generator initialized and ready")

    successful_processes = 0
    failed_processes = 0
    retry_queue: list[tuple[JobSpec, int]] = []
    current_batch: list[tuple[JobSpec, int]] = [(jspec, 1) for jspec in jobs_specs]
    results_lock = threading.Lock()

    while current_batch:
        logger.info(f"Processing batch of {len(current_batch)} images...")

        with ThreadPoolExecutor(max_workers=runtime_config.max_workers) as executor:
            future_to_jspec = {
                executor.submit(
                    process_single_job,
                    jspec,
                    runtime_config.logdir,
                    model_name,
                    runtime_config.run_id,
                    shared_issue_generator,
                    session_config,
                ): (jspec, attempt)
                for jspec, attempt in current_batch
            }

            for future in as_completed(future_to_jspec):
                jspec, attempt = future_to_jspec[future]
                result, success, error_msg = future.result()

                if success and result is not None:
                    successful_processes += 1
                    logger.info(
                        f"✓ Completed {jspec} on attempt {attempt} "
                        f"({successful_processes}/{len(jobs_specs)})"
                    )
                    _record_success(
                        result,
                        output_file=output_file,
                        results=existing_results,
                        lock=results_lock,
                    )
                else:
                    if attempt < runtime_config.max_tries:
                        retry_queue.append((jspec, attempt + 1))
                        logger.warning(
                            f"⚠ Failed {jspec} on attempt {attempt}/{runtime_config.max_tries}: "
                            f"{error_msg}. Will retry."
                        )
                    else:
                        failed_processes += 1
                        logger.error(
                            f"✗ Failed {jspec} after {runtime_config.max_tries} attempts: {error_msg}. "
                            "Giving up."
                        )

        current_batch = retry_queue.copy()
        retry_queue.clear()

    total_processed = successful_processes + failed_processes
    logger.info(
        f"Processing complete! Success: {successful_processes}, "
        f"Failed: {failed_processes}, Total attempted: {total_processed}"
    )

    return {
        "successful": successful_processes,
        "failed": failed_processes,
        "total": len(jobs_specs),
    }


if __name__ == "__main__":
    import fire

    fire.Fire(regular)
