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
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from textwrap import shorten
from typing import Any

import jinja2
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from unidiff import PatchSet

from debug_gym.agents.free_agent import FreeAgent
from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.llms.base import LLM
from debug_gym.logger import DebugGymLogger

from swebench.harness.constants import (
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    PASS_TO_PASS,
)
from swesmith.constants import (
    KEY_TIMED_OUT,
    LOG_DIR_ISSUE_GEN,
    LOG_DIR_RUN_VALIDATION,
    LOG_TEST_OUTPUT_PRE_GOLD,
    TEST_OUTPUT_END,
    TEST_OUTPUT_START,
)
from swesmith.harness.valid import run_validation

load_dotenv()

CUR_DIR = Path(__file__).parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = CUR_DIR / "debug_gym_buggen.yaml"
# Bound the number of failing tests we accept for a synthesized bug.
MAX_FAILING_TESTS = 10

JobSpec = tuple[str, str]


class CustomIssueGen:
    """Minimal issue generator that relies on Debug-Gym's LLM stack."""

    def __init__(
        self,
        model: str,
        use_existing: bool,
        n_workers: int,
        experiment_id: Path | str,
        config_path: Path | str,
    ):
        self.experiment_id = Path(experiment_id)
        self.model = model
        self.use_existing = use_existing
        self.n_workers = n_workers
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Issue generation config not found: {self.config_path}"
            )

        raw_config = yaml.safe_load(self.config_path.read_text()) or {}
        self.config = raw_config

        system_prompt = self.config.get("system")
        instance_prompt = self.config.get("instance")
        if not system_prompt or not instance_prompt:
            raise ValueError(
                "Issue generation config must provide both 'system' and 'instance' prompts."
            )

        settings = self.config.get("settings", {})
        self.n_instructions = int(settings.get("n_instructions", 1))
        self.max_var_tokens = int(settings.get("max_var_tokens", 10_000))

        # The SWE-bench dataset is required for prompt demonstrations.
        self.swebv = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

        self._lock = threading.Lock()
        self._llm_lock = threading.Lock()
        self._llm: LLM | None = None

        log_dir = LOG_DIR_ISSUE_GEN / self.experiment_id / "_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = DebugGymLogger(
            f"issue-gen:{shorten(str(self.experiment_id), width=30)}",
            log_dir=str(log_dir),
        )
        self.logger.setLevel(logging.INFO)

        self._jinja_env = jinja2.Environment()
        self._jinja_env.filters["shuffle"] = lambda seq: random.sample(
            list(seq), k=len(seq)
        )

    def _ensure_llm(self) -> LLM:
        with self._llm_lock:
            if self._llm is None:
                self._llm = LLM.instantiate(
                    llm_name=self.model,
                    logger=self.logger,
                )
            return self._llm

    def _maybe_shorten(self, text_str: str) -> str:
        if not text_str or self.max_var_tokens <= 0:
            return text_str

        approx_token_count = len(text_str) // 4
        if approx_token_count <= self.max_var_tokens:
            return text_str

        approx_chars = max(self.max_var_tokens * 4, 1)
        head = text_str[: approx_chars // 2]
        tail = text_str[-approx_chars // 2 :]
        return f"{head}\n\n(...)\n\n{tail}"

    def _format_prompt(self, template: str | None, context: dict[str, Any]) -> str:
        if not template:
            return ""
        compiled = self._jinja_env.from_string(template)
        return compiled.render(**context, **(self.config.get("parameters", {})))

    def _get_demo_issues(self) -> list[str]:
        problem_statements = [
            self._maybe_shorten(instance["problem_statement"])
            for instance in self.swebv
            if instance.get("problem_statement")
        ]
        random.shuffle(problem_statements)
        return problem_statements

    def get_test_functions(self, instance: dict[str, Any]) -> tuple[list[str], list[str]]:
        return [], []

    def get_test_output(self, instance: dict[str, Any]) -> str:
        repo_key = (instance.get("repo") or "").split("/")[-1]
        instance_id = instance.get(KEY_INSTANCE_ID) or instance.get("instance_id")
        if instance_id is None:
            raise KeyError("instance does not contain KEY_INSTANCE_ID")

        candidate_dirs = [LOG_DIR_RUN_VALIDATION / self.experiment_id / instance_id]

        if repo_key:
            candidate_dirs.append(LOG_DIR_RUN_VALIDATION / repo_key / instance_id)

        for folder in candidate_dirs:
            for filename in (LOG_TEST_OUTPUT, LOG_TEST_OUTPUT_PRE_GOLD):
                output_path = folder / filename
                if output_path.exists():
                    test_output = output_path.read_text()
                    start_idx = test_output.find(TEST_OUTPUT_START)
                    end_idx = test_output.find(TEST_OUTPUT_END)
                    if start_idx == -1 or end_idx == -1:
                        return self._maybe_shorten(test_output)
                    start_idx += len(TEST_OUTPUT_START)
                    return self._maybe_shorten(test_output[start_idx:end_idx])

        raise FileNotFoundError(
            f"Could not locate validation test output for {instance_id}"
        )

    def _build_messages(self, instance: dict[str, Any]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.config["system"]},
        ]

        demonstration_template = self.config.get("demonstration")
        if demonstration_template:
            messages.append(
                {
                    "role": "user",
                    "content": self._format_prompt(
                        demonstration_template,
                        {"demo_problem_statements": self._get_demo_issues()},
                    ),
                }
            )

        test_funcs, _ = self.get_test_functions(instance)
        instance_payload = instance | {
            "test_output": self.get_test_output(instance),
            "test_funcs": test_funcs,
        }

        messages.append(
            {
                "role": "user",
                "content": self._format_prompt(
                    self.config["instance"],
                    instance_payload,
                ),
            }
        )

        return messages

    def generate_issue(self, instance: dict[str, Any]) -> dict[str, Any]:
        instance_id = instance.get(KEY_INSTANCE_ID) or instance.get("instance_id")
        if not instance_id:
            raise KeyError("Instance data must contain KEY_INSTANCE_ID")

        repo = (instance.get("repo") or "fallback/repo").split("/")[-1]
        inst_dir = LOG_DIR_ISSUE_GEN / self.experiment_id / repo / instance_id
        inst_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = inst_dir / "metadata.json"
        if self.use_existing and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        def _append_progress_log(
            log_path: Path,
            lock: threading.Lock,
            entry: dict[str, Any],
        ) -> None:
            """Write a single JSON line describing pipeline progress."""

            log_record = dict(entry)
            log_record.setdefault("timestamp", datetime.now().isoformat())
            with lock:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(log_record) + "\n")


        def assess_validation_report(
            report: dict[str, Any],
        ) -> tuple[bool, list[str], list[str], str | None]:
            """Check whether a validation report represents an acceptable bug."""

            f2p: list[str] = report.get(FAIL_TO_PASS, []) or []
            p2p: list[str] = report.get(PASS_TO_PASS, []) or []

            if KEY_TIMED_OUT in report:
                return False, f2p, p2p, "Validation timed out"
            if not f2p:
                return False, f2p, p2p, "No tests regressed (FAIL_TO_PASS empty)"
            if not p2p:
                return False, f2p, p2p, "All tests failed (PASS_TO_PASS empty)"
            if len(f2p) > MAX_FAILING_TESTS:
                return (
                    False,
                    f2p,
                    p2p,
                    f"Too many failing tests ({len(f2p)} > {MAX_FAILING_TESTS})",
                )

            return True, f2p, p2p, None


        def _base_progress_entry(
            *,
            run_id: str,
            instance_id: str,
            mode: str,
            image_name: str | None = None,
            seed: str | None = None,
            attempt: int | None = None,
        ) -> dict[str, Any]:
            """Create a baseline progress record for a given instance."""

            repo_name, commit_sha, parsed_seed = parse_instance_id(instance_id)
            entry: dict[str, Any] = {
                "run_id": run_id,
                "mode": mode,
                "instance_id": instance_id,
                "repo": repo_name,
                "commit": commit_sha,
                "seed": seed if seed is not None else parsed_seed,
            }
            if image_name is not None:
                entry["image_name"] = image_name
            if attempt is not None:
                entry["attempt"] = attempt
            return entry


            for key, value in metadata.get("responses", {}).items():
                instance[key] = value
            return dict(instance)
        ) -> tuple[dict[str, Any] | None, str | None]:
            """Convert a validation report directory into ``instance_data`` payload."""

        with (inst_dir / "messages.json").open("w", encoding="utf-8") as handle:
                return None, "Instance directory missing"

        llm = self._ensure_llm()
        if llm is None:
            raise RuntimeError(f"Failed to instantiate LLM '{self.model}' for issue generation")

        responses: dict[str, str] = {}
        token_stats: list[dict[str, int]] = []
                return None, "Report missing"
        for idx in range(self.n_instructions):
            llm_response = llm(
                messages=copy.deepcopy(messages),
                tools=[],
            is_buggy, f2p, p2p, rejection_msg = assess_validation_report(report)
            if not is_buggy:
                logger.info("Skipping %s: %s", instance_id, rejection_msg)
                return None, rejection_msg or "Rejected by filters"
                    {
                        "prompt": llm_response.token_usage.prompt or 0,
                        "response": llm_response.token_usage.response or 0,
                    }
                )

        metadata = {
            "responses": responses,
            "token_usage": token_stats,
            "model": self.model,
        }

        with self._lock:
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            with (inst_dir / "issue.json").open("w", encoding="utf-8") as handle:
                json.dump(instance, handle, indent=2)

        return dict(instance)


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


def parse_instance_id(instance_id: str) -> tuple[str, str, str]:
    """Return ``(repo_name, commit_sha, seed)`` extracted from the identifier."""

    try:
        repo_commit, _, seed_token = instance_id.rpartition(".debuggym.")
        repo_name, _, commit_sha = repo_commit.rpartition(".")
        if not seed_token.startswith("seed_"):
            raise ValueError
        seed = seed_token.removeprefix("seed_")
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unrecognized instance id format: {instance_id}") from exc

    if not repo_name or not commit_sha or not seed:
        raise ValueError(f"Unrecognized instance id format: {instance_id}")

    return repo_name, commit_sha, seed


def locate_image_folder(logdir: Path, repo_name: str, commit_sha: str) -> Path | None:
    """Return the Debug-Gym output folder that matches the repo/commit."""

    if not logdir.exists():
        return None

    suffix = f"{repo_name}.{commit_sha}"
    for candidate in logdir.iterdir():
        if candidate.is_dir() and candidate.name.endswith(suffix):
            return candidate
    return None


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
    )

    return session_config, runtime_config


def _extract_agent_metadata(
    logdir: Path,
    repo_name: str,
    commit_sha: str,
    seed: str,
) -> tuple[str | None, str | None, Path | None]:
    """Locate Debug-Gym outputs (patch + agent UUID) for an existing run."""

    image_folder = locate_image_folder(logdir, repo_name, commit_sha)
    if image_folder is None:
        return None, None, None

    seed_folder = image_folder / f"seed_{seed}"
    if not seed_folder.exists():
        return str(image_folder.name), None, None

    patch_candidates = [
        seed_folder / "debug_gym.patch",
        seed_folder / "patch.diff",
    ]
    patch_path = next((cand for cand in patch_candidates if cand.exists()), None)

    agent_uuid = None
    runs_dir = seed_folder / "debug_gym_runs"
    if runs_dir.exists():
        for candidate in runs_dir.iterdir():
            if candidate.is_dir():
                agent_uuid = candidate.name
                break

    return str(image_folder.name), agent_uuid, patch_path


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


def assess_validation_report(
    report: dict[str, Any],
) -> tuple[bool, list[str], list[str], str | None]:
    """Check whether a validation report represents an acceptable bug."""

    f2p: list[str] = report.get(FAIL_TO_PASS, []) or []
    p2p: list[str] = report.get(PASS_TO_PASS, []) or []

    if KEY_TIMED_OUT in report:
        return False, f2p, p2p, "Validation timed out"
    if not f2p:
        return False, f2p, p2p, "No tests regressed (FAIL_TO_PASS empty)"
    if not p2p:
        return False, f2p, p2p, "All tests failed (PASS_TO_PASS empty)"
    if len(f2p) > MAX_FAILING_TESTS:
        return (
            False,
            f2p,
            p2p,
            f"Too many failing tests ({len(f2p)} > {MAX_FAILING_TESTS})",
        )

    return True, f2p, p2p, None


def _build_instance_from_logs(
    instance_dir: Path,
    runtime_config: BuggenRuntimeConfig,
) -> dict[str, Any] | None:
    """Convert a validation report directory into ``instance_data`` payload."""

    if not instance_dir.is_dir():
        return None

    instance_id = instance_dir.name
    repo_name, commit_sha, seed = parse_instance_id(instance_id)

    report_path = instance_dir / LOG_REPORT
    if not report_path.exists():
        logger.info("Skipping %s, report.json missing", instance_id)
        return None

    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    is_buggy, f2p, p2p, rejection_msg = assess_validation_report(report)
    if not is_buggy:
        logger.info("Skipping %s: %s", instance_id, rejection_msg)
        return None

    image_folder_name, agent_uuid, patch_path = _extract_agent_metadata(
        runtime_config.logdir,
        repo_name,
        commit_sha,
        seed,
    )

    patch_candidates = [instance_dir / "patch.diff"]
    if patch_path is not None:
        patch_candidates.insert(0, patch_path)

    patch_text = next((cand.read_text() for cand in patch_candidates if cand.exists()), None)
    if not patch_text:
        raise FileNotFoundError(f"Could not locate patch artifacts for {instance_id}")

    patch_text = remove_added_test_files(patch_text)

    image_name = image_folder_name or f"swesmith.x86_64.{repo_name}.{commit_sha}"

    instance_data = {
        "instance_id": instance_id,
        "repo": f"swesmith/{repo_name}.{commit_sha}",
        "patch": patch_text,
        FAIL_TO_PASS: f2p,
        PASS_TO_PASS: p2p,
        "created_at": datetime.now().isoformat(),
        "image_name": image_name,
        "agent_resolved": None,
    }
    if agent_uuid:
        instance_data["agent_uuid"] = agent_uuid

    return instance_data, None


def _generate_issue_payload(
    issue_generator: CustomIssueGen,
    instance_data: dict[str, Any],
) -> dict[str, Any]:
    """Run the issue generator and return its JSON payload."""

    return issue_generator.generate_issue(instance_data)


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

        logger.info(f"Successfully analysed validation report for {jid}")
        logger.info(f"Generating problem description text for {jid}")

        try:
            logger.info(f"Calling issue_generator.generate_issue for {jid}")
            instance_data = _generate_issue_payload(issue_generator, instance_data)
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


def generate_issues_from_existing(
    *,
    session_config: DebugGymSessionConfig,
    runtime_config: BuggenRuntimeConfig,
    model_name: str,
    progress_log_path: Path,
    progress_lock: threading.Lock,
) -> dict[str, int]:
    """Produce issue texts for already-validated patches.

    This bypasses the FreeAgent run and consumes data from
    ``logs/run_validation/<run_id>``. Results are appended to the standard
    output JSON file so callers can mix live bug generation with this
    post-processing mode.
    """

    validations_root = LOG_DIR_RUN_VALIDATION / runtime_config.run_id
    if not validations_root.exists():
        raise FileNotFoundError(
            f"Validation logs for run {runtime_config.run_id} not found at {validations_root}"
        )

    existing_results = _load_existing_results(runtime_config.output_file)
    existing_instance_ids = {entry["instance_id"] for entry in existing_results}

    instance_dirs = [path for path in validations_root.iterdir() if path.is_dir()]
    logger.info(
        "Found %d existing results; %d instances logged in validation directory.",
        len(existing_results),
        len(instance_dirs),
    )

    issue_generator = CustomIssueGen(
        model=model_name,
        use_existing=True,
        n_workers=1,
        experiment_id=runtime_config.run_id,
        config_path=session_config.issue_gen_config,
    )

    processed = skipped = failures = 0
    results_lock = threading.Lock()

    for instance_dir in sorted(instance_dirs):
        instance_id = instance_dir.name
        if instance_id in existing_instance_ids:
            logger.info("Skipping %s, already present in results output", instance_id)
            _append_progress_log(
                progress_log_path,
                progress_lock,
                {
                    **_base_progress_entry(
                        run_id=runtime_config.run_id,
                        instance_id=instance_id,
                        mode="issue_only",
                    ),
                    "status": "skipped_existing",
                    "reason": "Already present in results output",
                },
            )
            skipped += 1
            continue

        try:
            instance_data, rejection_msg = _build_instance_from_logs(
                instance_dir, runtime_config
            )
            if instance_data is None:
                _append_progress_log(
                    progress_log_path,
                    progress_lock,
                    {
                        **_base_progress_entry(
                            run_id=runtime_config.run_id,
                            instance_id=instance_id,
                            mode="issue_only",
                        ),
                        "status": "skipped_filtered",
                        "reason": rejection_msg,
                    },
                )
                skipped += 1
                continue
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to prepare %s from existing artifacts: %s", instance_id, exc)
            _append_progress_log(
                progress_log_path,
                progress_lock,
                {
                    **_base_progress_entry(
                        run_id=runtime_config.run_id,
                        instance_id=instance_id,
                        mode="issue_only",
                    ),
                    "status": "error",
                    "reason": str(exc),
                },
            )
            failures += 1
            continue

        try:
            logger.info("Generating issue for existing instance %s", instance_id)
            instance_data = _generate_issue_payload(issue_generator, instance_data)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Issue generation failed for %s: %s", instance_id, exc)
            _append_progress_log(
                progress_log_path,
                progress_lock,
                {
                    **_base_progress_entry(
                        run_id=runtime_config.run_id,
                        instance_id=instance_id,
                        mode="issue_only",
                    ),
                    "status": "error",
                    "reason": str(exc),
                },
            )
            failures += 1
            continue

        _record_success(
            instance_data,
            output_file=runtime_config.output_file,
            results=existing_results,
            lock=results_lock,
        )
        _append_progress_log(
            progress_log_path,
            progress_lock,
            {
                **_base_progress_entry(
                    run_id=runtime_config.run_id,
                    instance_id=instance_data["instance_id"],
                    mode="issue_only",
                    image_name=instance_data.get("image_name"),
                ),
                "status": "issue_generated",
            },
        )
        processed += 1
        logger.info("✓ Generated issue for %s (%d total)", instance_id, processed)

    logger.info(
        "Issue-only pass complete. Generated: %d, skipped: %d, failures: %d",
        processed,
        skipped,
        failures,
    )

    return {"generated": processed, "skipped": skipped, "failed": failures}


def regular(
    config: str | Path | None = None,
    issue_only: bool = False,
    run_id: str | None = None,
) -> dict[str, int]:
    """Primary CLI entry point.

    Args:
        config: Optional path to the YAML configuration file.
        issue_only: When ``True`` skip FreeAgent runs and only generate issues
            for existing validation artifacts.
        run_id: Override the ``run.run_id`` from the YAML; useful when
            reprocessing an earlier validation run.
    """

    session_config, runtime_config = load_pipeline_config(config)

    if run_id:
        runtime_config = replace(runtime_config, run_id=run_id)

    model_name = (session_config.llm_name or "").strip()
    if not model_name:
        raise ValueError("Model name is required. Set 'llm' in the YAML config.")

    progress_log_path = runtime_config.output_file.with_suffix(".progress.jsonl")
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_lock = threading.Lock()

    if issue_only:
        return generate_issues_from_existing(
            session_config=session_config,
            runtime_config=runtime_config,
            model_name=model_name,
            progress_log_path=progress_log_path,
            progress_lock=progress_lock,
        )

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

    filtered_jobs: list[JobSpec] = []
    for jspec in jobs_specs:
        instance_id = create_instance_id(*jspec)
        if instance_id in existing_instance_ids:
            _append_progress_log(
                progress_log_path,
                progress_lock,
                {
                    **_base_progress_entry(
                        run_id=runtime_config.run_id,
                        instance_id=instance_id,
                        mode="buggen",
                        image_name=jspec[0],
                        seed=jspec[1],
                    ),
                    "status": "skipped_existing",
                    "reason": "Already present in results output",
                },
            )
            continue
        filtered_jobs.append(jspec)
    jobs_specs = filtered_jobs

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
        config_path=session_config.issue_gen_config,
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
                    _append_progress_log(
                        progress_log_path,
                        progress_lock,
                        {
                            **_base_progress_entry(
                                run_id=runtime_config.run_id,
                                instance_id=result["instance_id"],
                                mode="buggen",
                                image_name=jspec[0],
                                seed=jspec[1],
                                attempt=attempt,
                            ),
                            "status": "issue_generated",
                        },
                    )
                else:
                    entry = {
                        **_base_progress_entry(
                            run_id=runtime_config.run_id,
                            instance_id=create_instance_id(*jspec),
                            mode="buggen",
                            image_name=jspec[0],
                            seed=jspec[1],
                            attempt=attempt,
                        ),
                        "reason": error_msg,
                    }
                    if attempt < runtime_config.max_tries:
                        retry_queue.append((jspec, attempt + 1))
                        logger.warning(
                            f"⚠ Failed {jspec} on attempt {attempt}/{runtime_config.max_tries}: "
                            f"{error_msg}. Will retry."
                        )
                        entry["status"] = "retry_scheduled"
                    else:
                        failed_processes += 1
                        logger.error(
                            f"✗ Failed {jspec} after {runtime_config.max_tries} attempts: {error_msg}. "
                            "Giving up."
                        )
                        entry["status"] = "failed"
                    _append_progress_log(progress_log_path, progress_lock, entry)

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
