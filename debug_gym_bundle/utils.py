"""Shared helpers for Debug-Gym bug generation pipeline components."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from unidiff import PatchSet

from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS, LOG_REPORT as SWE_LOG_REPORT
from swesmith.constants import KEY_TIMED_OUT

try:  # Prefer SWESmith's constant when available for backward compatibility
    from swesmith.constants import LOG_REPORT as SWESMITH_LOG_REPORT  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - SWESmith versions without LOG_REPORT
    SWESMITH_LOG_REPORT = None

LOG_REPORT = SWESMITH_LOG_REPORT or SWE_LOG_REPORT

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .config import BuggenRuntimeConfig


def _normalize_image_identifier(image_name: str) -> str:
    """Strip registry prefixes and tags to recover the SWE-Smith image identifier."""

    # Drop leading registry segment (e.g., ``docker.io/`` or ``jyangballin/``)
    image_id = image_name.rsplit("/", 1)[-1]
    # Remove docker tag suffix (e.g., ``:latest``)
    if ":" in image_id:
        image_id = image_id.rsplit(":", 1)[0]
    return image_id


def extract_repo_commit(image_name: str) -> tuple[str, str]:
    """Return ``(repo_name, commit_sha)`` derived from a Debug-Gym image name."""

    normalized = _normalize_image_identifier(image_name)
    parts = normalized.split(".")
    if len(parts) < 4:
        raise ValueError(
            "Image name must contain at least four period-delimited segments"
        )
    repo_name = parts[-2]
    commit_sha = parts[-1]
    if not repo_name or not commit_sha:
        raise ValueError("Image name is missing repository or commit information")
    return repo_name, commit_sha


def remove_added_test_files(patch: str) -> str:
    """Strip any newly added test files from the generated patch."""

    try:
        parsed_patch = PatchSet(patch)
        try:
            file_patches = list(parsed_patch)
        except Exception as exc:  # pragma: no cover - depends on diff shape
            logger.warning(
                "Failed to iterate patch entries while filtering tests: %s", exc
            )
            return patch

        filtered_entries: list[str] = []
        for file_patch in file_patches:
            try:
                if (
                    file_patch.is_added_file
                    and file_patch.path.endswith(".py")
                    and "test_" in file_patch.path
                ):
                    continue
                filtered_entries.append(str(file_patch))
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    "Encountered error while processing patch entry; using original patch: %s",
                    exc,
                )
                return patch

        if not filtered_entries:
            logger.debug(
                "No non-test diff hunks remained after filtering; retaining original patch"
            )
            return patch

        rebuilt = "".join(filtered_entries)
        if not rebuilt.strip():
            logger.warning(
                "Filtered patch rebuild produced empty output; returning original patch"
            )
            return patch

        try:
            PatchSet(rebuilt)
        except Exception as exc:  # pragma: no cover - diff-dependent
            logger.warning(
                "Filtered patch could not be re-parsed (%s); returning original patch",
                exc,
            )
            return patch

        return rebuilt

    except Exception as exc:  # pragma: no cover - diff-dependent
        logger.warning(
            "Unexpected error while filtering patch; returning original patch: %s",
            exc,
        )
        return patch


def create_instance_id(image_name: str, seed: str) -> str:
    """Create a unique SWE-bench style identifier for this run."""

    repo_name, short_commit_sha = extract_repo_commit(image_name)
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


def _base_progress_entry(
    *,
    run_id: str,
    instance_id: str,
    llm_name: str,
    image_name: str | None = None,
    seed: str | None = None,
    attempt: int | None = None,
) -> dict[str, Any]:
    """Create a baseline progress record for a given instance."""

    repo_name, commit_sha, parsed_seed = parse_instance_id(instance_id)
    entry: dict[str, Any] = {
        "run_id": run_id,
        "instance_id": instance_id,
        "repo": repo_name,
        "commit": commit_sha,
        "seed": seed if seed is not None else parsed_seed,
        "llm": llm_name,
    }
    if image_name is not None:
        entry["image_name"] = image_name
    if attempt is not None:
        entry["attempt"] = attempt
    return entry


def locate_image_folder(logdir: Path, repo_name: str, commit_sha: str) -> Path | None:
    """Return the Debug-Gym output folder that matches the repo/commit."""

    if not logdir.exists():
        return None

    suffix = f"{repo_name}.{commit_sha}"
    for candidate in logdir.iterdir():
        if candidate.is_dir() and candidate.name.endswith(suffix):
            return candidate
    return None


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
    *,
    max_fail_fraction: float,
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
    total_checked = len(f2p) + len(p2p)
    if total_checked > 0:
        failing_fraction = len(f2p) / total_checked
        if failing_fraction > max_fail_fraction:
            percent_failed = failing_fraction * 100
            percent_limit = max_fail_fraction * 100
            return (
                False,
                f2p,
                p2p,
                f"Too many failing tests ({len(f2p)} of {total_checked} = {percent_failed:.1f}% > {percent_limit:.1f}%)",
            )

    return True, f2p, p2p, None


def _build_instance_from_logs(
    instance_dir: Path,
    runtime_config: "BuggenRuntimeConfig",
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert a validation report directory into ``instance_data`` payload."""

    if not instance_dir.is_dir():
        return None, "Instance directory missing"

    instance_id = instance_dir.name
    repo_name, commit_sha, seed = parse_instance_id(instance_id)

    report_path = instance_dir / LOG_REPORT
    if not report_path.exists():
        logger.info("Skipping %s, report.json missing", instance_id)
        return None, "Report missing"

    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    is_buggy, f2p, p2p, rejection_msg = assess_validation_report(
        report,
        max_fail_fraction=runtime_config.max_fail_fraction,
    )
    if not is_buggy:
        logger.info("Skipping %s: %s", instance_id, rejection_msg)
        return None, rejection_msg or "Rejected by filters"

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
