"""Issue generation helpers built on top of Debug-Gym's LLM stack."""

from __future__ import annotations

import copy
import json
import logging
import random
import threading
from pathlib import Path
from textwrap import shorten
from typing import Any

import jinja2
import yaml
from datasets import load_dataset

from debug_gym.llms.base import LLM
from debug_gym.logger import DebugGymLogger
from swesmith.constants import LOG_DIR_ISSUE_GEN

logger = logging.getLogger(__name__)


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
        """Instantiate (or reuse) the configured LLM under a lock."""

        with self._llm_lock:
            if self._llm is None:
                self._llm = LLM.instantiate(
                    llm_name=self.model,
                    logger=self.logger,
                )
            return self._llm

    def _maybe_shorten(self, text_str: str) -> str:
        """Truncate large text blobs so prompts stay within token limits."""

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
        """Render a Jinja template with the provided context and defaults."""

        if not template:
            return ""
        compiled = self._jinja_env.from_string(template)
        return compiled.render(**context, **(self.config.get("parameters", {})))

    def _get_demo_issues(self) -> list[str]:
        """Return a shuffled collection of prompt demonstrations."""

        problem_statements = [
            self._maybe_shorten(instance["problem_statement"])
            for instance in self.swebv
            if instance.get("problem_statement")
        ]
        random.shuffle(problem_statements)
        return problem_statements

    def get_test_functions(self, instance: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Placeholder hook for future test extraction logic."""

        return [], []

    def get_test_output(self, instance: dict[str, Any]) -> str:
        """Load the captured test output for the provided instance."""

        from swesmith.constants import LOG_DIR_RUN_VALIDATION
        from swebench.harness import constants as swe_constants
        import swesmith.constants as smith_constants

        KEY_INSTANCE_ID = swe_constants.KEY_INSTANCE_ID
        LOG_TEST_OUTPUT = swe_constants.LOG_TEST_OUTPUT

        TEST_OUTPUT_START = getattr(swe_constants, "TEST_OUTPUT_START", None)
        TEST_OUTPUT_END = getattr(swe_constants, "TEST_OUTPUT_END", None)

        candidate_filenames = [LOG_TEST_OUTPUT]
        _pre_gold = getattr(swe_constants, "LOG_TEST_OUTPUT_PRE_GOLD", None)
        if _pre_gold is None:
            _pre_gold = getattr(smith_constants, "LOG_TEST_OUTPUT_PRE_GOLD", None)

        if TEST_OUTPUT_START is None or TEST_OUTPUT_END is None:
            TEST_OUTPUT_START = TEST_OUTPUT_START or getattr(
                smith_constants, "TEST_OUTPUT_START", None
            )
            TEST_OUTPUT_END = TEST_OUTPUT_END or getattr(
                smith_constants, "TEST_OUTPUT_END", None
            )

        if _pre_gold:
            candidate_filenames.append(_pre_gold)

        repo_key = (instance.get("repo") or "").split("/")[-1]
        instance_id = instance.get(KEY_INSTANCE_ID) or instance.get("instance_id")
        if instance_id is None:
            raise KeyError("instance does not contain KEY_INSTANCE_ID")

        candidate_dirs = [LOG_DIR_RUN_VALIDATION / self.experiment_id / instance_id]

        if repo_key:
            candidate_dirs.append(LOG_DIR_RUN_VALIDATION / repo_key / instance_id)

        for folder in candidate_dirs:
            for filename in candidate_filenames:
                output_path = folder / filename
                if output_path.exists():
                    test_output = output_path.read_text()
                    if TEST_OUTPUT_START and TEST_OUTPUT_END:
                        start_idx = test_output.find(TEST_OUTPUT_START)
                        end_idx = test_output.find(TEST_OUTPUT_END)
                        if start_idx != -1 and end_idx != -1:
                            start_idx += len(TEST_OUTPUT_START)
                            snippet = test_output[start_idx:end_idx]
                            return self._maybe_shorten(snippet)
                    return self._maybe_shorten(test_output)

        raise FileNotFoundError(
            f"Could not locate validation test output for {instance_id}"
        )

    def _build_messages(self, instance: dict[str, Any]) -> list[dict[str, str]]:
        """Construct the conversational payload delivered to the LLM."""

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
        """Produce an issue description and persist the associated metadata."""

        from swebench.harness.constants import KEY_INSTANCE_ID

        instance_id = instance.get(KEY_INSTANCE_ID) or instance.get("instance_id")
        if not instance_id:
            raise KeyError("Instance data must contain KEY_INSTANCE_ID")

        repo = (instance.get("repo") or "fallback/repo").split("/")[-1]
        inst_dir = LOG_DIR_ISSUE_GEN / self.experiment_id / repo / instance_id
        inst_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = inst_dir / "metadata.json"
        if self.use_existing and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Corrupt metadata for %s (%s); regenerating", instance_id, exc
                )
            else:
                for key, value in metadata.get("responses", {}).items():
                    instance[key] = value
                return dict(instance)

        messages = self._build_messages(instance)

        with (inst_dir / "messages.json").open("w", encoding="utf-8") as handle:
            json.dump(messages, handle, indent=2)

        llm = self._ensure_llm()
        if llm is None:
            raise RuntimeError(
                f"Failed to instantiate LLM '{self.model}' for issue generation"
            )

        responses: dict[str, str] = {}
        token_stats: list[dict[str, int]] = []

        for idx in range(self.n_instructions):
            llm_response = llm(
                messages=copy.deepcopy(messages),
                tools=[],
            )
            issue_text = llm_response.response or ""
            key = "problem_statement" if self.n_instructions == 1 else f"ps_basic_{idx}"
            instance[key] = issue_text
            responses[key] = issue_text

            if llm_response.token_usage:
                token_stats.append(
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


def _generate_issue_payload(
    issue_generator: CustomIssueGen,
    instance_data: dict[str, Any],
) -> dict[str, Any]:
    """Run the issue generator and return its JSON payload."""

    return issue_generator.generate_issue(instance_data)
