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
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from textwrap import shorten
from typing import Any

import yaml
from datasets import load_dataset
from unidiff import PatchSet

from debug_gym.agents.free_agent import FreeAgent
from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.llms.base import LLM
from debug_gym.logger import DebugGymLogger

from swebench.harness.constants import (
    FAIL_TO_PASS,
    PASS_TO_PASS,
    LOG_REPORT,
)
from swesmith.constants import LOG_DIR_RUN_VALIDATION, KEY_TIMED_OUT
from swesmith.harness.valid import run_validation
from swesmith.issue_gen.generate import IssueGen

from dotenv import load_dotenv

load_dotenv()

CUR_DIR = Path(__file__).parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BACKEND = "kubernetes"

DEFAULT_CONFIG_PATH = CUR_DIR / "debug_gym_buggen.yaml"
DEFAULT_INSTRUCTIONS_PATH = CUR_DIR / "free_env_buggen_instructions.md"
DEFAULT_INSTRUCTIONS_FALLBACK = (
    "You are inside an isolated container with the project at /testbed. "
    "Use the available tools to explore, edit, and run commands. "
    "Aim to craft a realistic regression without altering existing tests. "
    "Summarize your work with the submit tool once complete."
)

DEFAULT_TOOLS = ("listdir", "view", "grep", "rewrite", "bash", "submit")
DEFAULT_WORKSPACE_DIR = "/testbed"
DEFAULT_AGENT_MAX_STEPS = 80
DEFAULT_AGENT_MAX_REWRITE_STEPS = 40
DEFAULT_AGENT_MEMORY = 120
DEFAULT_DIR_TREE_DEPTH = 2
DEFAULT_INIT_GIT = True

ISSUE_GEN_CONFIG_FILE_PATH = CUR_DIR / Path("sans_patch_issue_gen.yaml")


class CustomIssueGen(IssueGen):
    def __init__(
        self,
        model: str,
        use_existing: bool,
        n_workers: int,
        experiment_id: Path,
    ):
        self.experiment_id = experiment_id
        self.model = model
        self.use_existing = use_existing
        self.n_workers = n_workers

        self.swebv = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

        self.config = yaml.safe_load(ISSUE_GEN_CONFIG_FILE_PATH.read_text())
        settings = self.config.get("settings", {})
        self.n_instructions = settings.get("n_instructions", 1)
        self.max_var_tokens = settings.get("max_var_tokens", 10_000)

        self._lock = threading.Lock()

    def get_test_functions(self, instance: dict) -> tuple[list[str], list[str]]:
        """
        Override to avoid cloning repos that don't exist on GitHub.
        For Debug-Gym generated bugs, we don't have access to test source code.

        Returns:
            Empty list of test functions, empty list of repos to remove
        """
        return [], []


def remove_added_test_files(patch: str) -> str:
    return str(
        PatchSet(
            [
                str(f)
                for f in PatchSet(patch)
                if not (f.is_added_file and f.path.endswith(".py") and "test_" in f.path)
            ]
        )
    )


def create_instance_id(image_name: str, seed: str) -> str:
    _, _, repo_name, short_commit_sha = image_name.split(".")
    return f"{repo_name}.{short_commit_sha}.debuggym.seed_{seed}"


@dataclass(frozen=True)
class DebugGymSessionConfig:
    llm_config_file: str | None
    tools: tuple[str, ...]
    env_terminal: str
    env_workspace_dir: str
    env_instructions: str
    env_setup_commands: tuple[str, ...]
    env_terminal_kwargs: dict[str, Any]
    env_dir_tree_depth: int
    env_init_git: bool
    agent_config: dict[str, Any]


def _resolve_path(base: Path, maybe_path: str | None) -> str | None:
    if maybe_path is None:
        return None
    path = Path(maybe_path)
    if not path.is_absolute():
        path = (base.parent / path).resolve()
    return str(path)


def load_session_config(config_path: str | Path | None) -> DebugGymSessionConfig:
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Debug-Gym buggen config not found: {cfg_path}")

    config_data = yaml.safe_load(cfg_path.read_text()) or {}

    llm_config_file = config_data.get("llm_config_file")
    llm_config_file = _resolve_path(cfg_path, llm_config_file)

    tools = config_data.get("tools") or list(DEFAULT_TOOLS)
    tools = tuple(str(tool).strip() for tool in tools if str(tool).strip())
    if not tools:
        tools = DEFAULT_TOOLS

    env_cfg = config_data.get("environment") or {}
    terminal = str(env_cfg.get("terminal", BACKEND))
    workspace_dir = str(env_cfg.get("workspace_dir", DEFAULT_WORKSPACE_DIR))

    instructions = env_cfg.get("instructions")
    instructions_file = env_cfg.get("instructions_file")
    if instructions and instructions_file:
        raise ValueError(
            "Specify either environment.instructions or environment.instructions_file, not both."
        )
    if instructions_file:
        resolved_instructions_path = _resolve_path(cfg_path, instructions_file)
        if resolved_instructions_path is None or not Path(
            resolved_instructions_path
        ).exists():
            raise FileNotFoundError(
                f"Instructions file not found: {resolved_instructions_path}"
            )
        instructions = Path(resolved_instructions_path).read_text()
    if not instructions:
        if DEFAULT_INSTRUCTIONS_PATH.exists():
            instructions = DEFAULT_INSTRUCTIONS_PATH.read_text()
        else:
            instructions = DEFAULT_INSTRUCTIONS_FALLBACK

    setup_commands = env_cfg.get("setup_commands") or []
    if isinstance(setup_commands, str):
        setup_commands = [setup_commands]
    setup_commands = tuple(
        str(cmd).strip() for cmd in setup_commands if str(cmd).strip()
    )

    terminal_kwargs = env_cfg.get("terminal_kwargs") or {}
    if not isinstance(terminal_kwargs, dict):
        raise ValueError("environment.terminal_kwargs must be a mapping")

    dir_tree_depth = int(env_cfg.get("dir_tree_depth", DEFAULT_DIR_TREE_DEPTH))
    init_git = bool(env_cfg.get("init_git", DEFAULT_INIT_GIT))

    agent_cfg = dict(config_data.get("agent") or {})
    agent_cfg["max_steps"] = int(
        agent_cfg.get("max_steps", DEFAULT_AGENT_MAX_STEPS)
    )
    agent_cfg["max_rewrite_steps"] = int(
        agent_cfg.get("max_rewrite_steps", DEFAULT_AGENT_MAX_REWRITE_STEPS)
    )
    agent_cfg["memory_size"] = int(
        agent_cfg.get("memory_size", max(agent_cfg["max_steps"], DEFAULT_AGENT_MEMORY))
    )

    return DebugGymSessionConfig(
        llm_config_file=llm_config_file,
        tools=tools,
        env_terminal=terminal,
        env_workspace_dir=workspace_dir,
        env_instructions=instructions,
        env_setup_commands=setup_commands,
        env_terminal_kwargs=dict(terminal_kwargs),
        env_dir_tree_depth=dir_tree_depth,
        env_init_git=init_git,
        agent_config=agent_cfg,
    )


def derive_agent_seed(seed: str) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def process_single_job(
    jspec,
    logdir: Path | str,
    model_name: str,
    run_id: str,
    issue_generator: CustomIssueGen,
    session_config: DebugGymSessionConfig,
) -> tuple[dict | None, bool, str | None]:
    """
    Process a single Docker image with Debug-Gym.

    Args:
        jspec: Job specification tuple (image_name, seed)
        logdir: Directory for logs
        model_name: Model name for the agent
        run_id: Run identifier
        issue_generator: Shared IssueGen instance (reused across all workers)
        session_config: Loaded configuration for environment and agent

    Returns:
        Tuple of (instance_data, success, error_message)
    """
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

        for stale_file in ("debug_gym.patch", "trajectory.json"):
            candidate = image_output_dir / stale_file
            if candidate.exists():
                candidate.unlink()
        for stale_dir in ("debug_gym_runs", "debug_gym_logs"):
            candidate_dir = image_output_dir / stale_dir
            if candidate_dir.exists():
                shutil.rmtree(candidate_dir)

        _, _, repo_name, short_commit_sha = image_name.split('.')

        debug_logger = DebugGymLogger(
            f"buggen:{shorten(instance_id, width=40)}",
            log_dir=str(image_output_dir / "debug_gym_logs"),
        )
        debug_logger.setLevel(logging.INFO)

        env = FreeEnv(
            image=image_name,
            terminal=session_config.env_terminal or BACKEND,
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
            llm_config_file_path=session_config.llm_config_file,
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
        with report_path.open("r") as report_handle:
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
        except Exception as e:
            logger.error(f"Error generating issue for {jid}: {str(e)}")
            logger.error(traceback.format_exc())
            return (None, False, f"Error generating issue: {str(e)}")

        return (instance_data, True, None)

    except Exception as e:
        traceback.print_exc()
        error_msg = f"Error processing {jid}: {str(e)}"
        logger.error(error_msg)
        return (None, False, error_msg)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                logger.warning(f"Failed to close environment for {jid}: {exc}")
        if debug_logger is not None:
            debug_logger.close()


def regular(
    images: str | Path,
    model_name: str,
    output_file: str | Path,
    run_id: str,
    logdir: str | Path,
    seed_per_image: int = 1,
    max_workers: int = 1,
    max_tries: int = 1,
    shuffle: bool = False,
    config: str | Path | None = None,
    llm_config_file: str | Path | None = None,
):
    """
    Process Docker images in parallel with retry logic using Debug-Gym.

    Args:
        images: Path to text file containing newline separated list of images
        model_name: Model name for the agent
        output_file: File to store outputs
        run_id: Name for run
        seed_per_image: Number of different seeds per image attempt
        max_workers: Maximum number of parallel workers
        max_tries: Maximum number of retry attempts per image
        shuffle: Whether to shuffle the job list
        config: Optional YAML config with Debug-Gym settings
        llm_config_file: Optional override for the LLM configuration file path
    """

    session_config = load_session_config(config)
    if llm_config_file:
        session_config = replace(
            session_config,
            llm_config_file=str(Path(llm_config_file).resolve()),
        )

    image_names = Path(images).read_text().splitlines()
    jobs_specs = [
        (i, str(uuid.uuid4())[:10])
        for i in image_names
        for _ in range(seed_per_image)
    ]

    if shuffle:
        random.shuffle(jobs_specs)

    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    if output_file.exists():
        with output_file.open("r") as handle:
            pre_existing_data = json.load(handle)
        existing_instance_ids = {i['instance_id'] for i in pre_existing_data}
        jobs_specs = [
            (i, s)
            for (i, s) in jobs_specs
            if create_instance_id(i, s) not in existing_instance_ids
        ]

    num_jobs = len(jobs_specs)
    jobs_specs = jobs_specs[:num_jobs]

    logger.info(
        f"Processing {len(jobs_specs)} jobs with {max_workers} workers, "
        f"max {max_tries} tries per job using Debug-Gym pipeline."
    )

    logger.info("Initializing shared issue generator (loading SWE-bench dataset)...")
    shared_issue_generator = CustomIssueGen(
        model=model_name,
        use_existing=True,
        n_workers=1,
        experiment_id=run_id,
    )
    logger.info("Issue generator initialized and ready")

    successful_processes = 0
    failed_processes = 0
    retry_queue: list[tuple[tuple[str, str], int]] = []
    current_batch = [(jspec, 1) for jspec in jobs_specs]

    while current_batch:
        logger.info(f"Processing batch of {len(current_batch)} images...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_jspec = {
                executor.submit(
                    process_single_job,
                    jspec,
                    logdir,
                    model_name,
                    run_id,
                    shared_issue_generator,
                    session_config,
                ): (jspec, attempt)
                for jspec, attempt in current_batch
            }

            for future in as_completed(future_to_jspec):
                jspec, attempt = future_to_jspec[future]
                result, success, error_msg = future.result()

                if success:
                    successful_processes += 1
                    logger.info(
                        f"✓ Completed {jspec} on attempt {attempt} ({successful_processes}/{len(jobs_specs)})"
                    )
                    if output_file.exists():
                        with output_file.open("r") as handle:
                            pre_existing_data = json.load(handle)
                    else:
                        pre_existing_data = []
                    pre_existing_data.append(result)
                    with output_file.open("w") as handle:
                        json.dump(pre_existing_data, handle, indent=2)
                else:
                    if attempt < max_tries:
                        retry_queue.append((jspec, attempt + 1))
                        logger.warning(
                            f"⚠ Failed {jspec} on attempt {attempt}/{max_tries}: {error_msg}. Will retry."
                        )
                    else:
                        failed_processes += 1
                        logger.error(
                            f"✗ Failed {jspec} after {max_tries} attempts: {error_msg}. Giving up."
                        )

        current_batch = retry_queue.copy()
        retry_queue.clear()

    total_processed = successful_processes + failed_processes
    logger.info(
        f"Processing complete! Success: {successful_processes}, "
        f"Failed: {failed_processes}, Total attempted: {total_processed}"
    )

    return {
        'successful': successful_processes,
        'failed': failed_processes,
        'total': len(jobs_specs),
    }


if __name__ == "__main__":
    import fire

    fire.Fire(regular)
