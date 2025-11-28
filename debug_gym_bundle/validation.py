"""Validation helpers with registry-aware SWE-smith integration."""

from __future__ import annotations

import json
import logging
import shutil
import traceback
from pathlib import Path
from typing import Any

from swebench.harness.constants import (
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    KEY_INSTANCE_ID,
    LOG_INSTANCE,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    TESTS_TIMEOUT,
    UTF8,
)
from swebench.harness.docker_build import close_logger, setup_logger
from swebench.harness.docker_utils import (
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
)

from swesmith.constants import (
    ENV_NAME,
    KEY_IMAGE_NAME,
    KEY_PATCH,
    KEY_TIMED_OUT,
    LOG_DIR_RUN_VALIDATION,
    LOG_TEST_OUTPUT_PRE_GOLD,
    REF_SUFFIX,
    TEST_OUTPUT_END,
    TEST_OUTPUT_START,
    TIMEOUT,
)

try:
    from docker.utils import parse_repository_tag
except ImportError:  # pragma: no cover - defensive fallback
    def parse_repository_tag(ref: str) -> tuple[str, str | None]:
        if "@" in ref:  # digest form
            return ref, None
        if ":" in ref and "/" not in ref.rsplit(":", 1)[1]:
            repo, tag = ref.rsplit(":", 1)
            return repo, tag
        return ref, None

try:  # SWE-smith is installed alongside Debug-Gym in bug-gen pipeline
    from swesmith.harness import utils as _swesmith_utils
    from swesmith.harness import valid as _swesmith_valid
    from swesmith.harness.grading import get_valid_report
except ImportError:  # pragma: no cover - defensive guard for lean installs
    _swesmith_utils = None  # type: ignore[assignment]
    _swesmith_valid = None  # type: ignore[assignment]
    get_valid_report = None  # type: ignore[assignment]

__all__ = [
    "REGISTRY_CACHE",
    "ensure_validation_registry_support",
    "run_validation",
]

logger = logging.getLogger(__name__)

REGISTRY_CACHE: dict[str, str] = {}
_REGISTRY_PATCH_APPLIED = False


def _split_repo_and_tag(image_ref: str) -> tuple[str, str | None]:
    repo, tag = parse_repository_tag(image_ref)
    if tag == "":  # pragma: no cover - older docker returns empty string
        tag = None
    return repo, tag


def _canonical_image_key(image_ref: str) -> str:
    repo, _ = _split_repo_and_tag(image_ref)
    _, _, base = repo.rpartition("/")
    if not base:
        base = repo
    return base.replace("_1776_", "__")


def _repo_variants(image_ref: str) -> tuple[list[str], str | None]:
    repo, tag = _split_repo_and_tag(image_ref)
    prefix, _, base = repo.rpartition("/")
    if not base:
        base = repo
        prefix = ""

    base_variants = {base}
    if "_1776_" in base:
        base_variants.add(base.replace("_1776_", "__"))
    if "__" in base:
        base_variants.add(base.replace("__", "_1776_"))

    variants: list[str] = []
    for candidate in base_variants:
        if prefix:
            variants.append(f"{prefix}/{candidate}")
        else:
            variants.append(candidate)
    return variants, tag


def _infer_registry_host(image_ref: str) -> str | None:
    repo, _ = _split_repo_and_tag(image_ref)
    first_segment = repo.split("/", 1)[0]
    if "." in first_segment or ":" in first_segment:
        return first_segment
    return None


def _candidate_image_refs(
    image_ref: str,
    registry_value: str | None,
) -> list[str]:
    repo_variants, tag = _repo_variants(image_ref)
    tag_suffix = f":{tag}" if tag else ""

    candidates: list[str] = []
    seen: set[str] = set()

    normalized_registry = registry_value.rstrip("/") if registry_value else None

    if normalized_registry:
        for variant in repo_variants:
            if variant.startswith(f"{normalized_registry}/"):  # already absolute
                candidate = f"{variant}{tag_suffix}"
            else:
                candidate = f"{normalized_registry}/{variant}{tag_suffix}"
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

    for variant in repo_variants:
        candidate = f"{variant}{tag_suffix}"
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    return candidates


def ensure_validation_registry_support() -> None:
    """Monkeypatch SWE-smith validation to honour optional image registries."""

    global _REGISTRY_PATCH_APPLIED
    if _REGISTRY_PATCH_APPLIED or _swesmith_utils is None or _swesmith_valid is None:
        return

    import docker

    def _run_patch_with_registry(
        instance: dict,
        run_id: str,
        log_dir: Path,
        patch: str | None = None,
        commit: str | None = None,
        is_gold: bool = False,
        timeout: int = _swesmith_utils.TIMEOUT,
    ):
        container = None
        client = docker.from_env()
        instance_id = instance[KEY_INSTANCE_ID]
        image_name = instance[KEY_IMAGE_NAME]
        raw_image = instance.get("raw_image_name", image_name)
        registry_value = instance.get("image_registry")
        if registry_value:
            REGISTRY_CACHE[image_name] = str(registry_value).strip()
        registry_prefix = REGISTRY_CACHE.get(image_name)
        if registry_prefix:
            registry_prefix = str(registry_prefix).strip()
        else:
            registry_prefix = None
        container_logger = None

        try:
            container_type = None
            if log_dir == RUN_EVALUATION_LOG_DIR:
                container_type = "eval"
            elif log_dir == LOG_DIR_RUN_VALIDATION:
                container_type = "val"

            run_log_dir = log_dir / run_id / instance_id
            run_log_dir.mkdir(parents=True, exist_ok=True)
            container_name = f"swesmith.{container_type}.{run_id}.{instance_id}"
            log_file = run_log_dir / LOG_INSTANCE
            container_logger = setup_logger(container_name, log_file)

            resolved_image: str | None = None
            last_error: Exception | None = None
            candidate_refs = _candidate_image_refs(raw_image, registry_prefix)

            for candidate in candidate_refs:
                try:
                    container = client.containers.create(
                        image=candidate,
                        name=container_name,
                        user=DOCKER_USER,
                        detach=True,
                        command="tail -f /dev/null",
                        platform="linux/x86_64",
                        mem_limit="10g",
                    )
                    resolved_image = candidate
                    container_logger.info(
                        "Created container for %s using image '%s'", instance_id, candidate
                    )
                    break
                except docker.errors.ImageNotFound as error:
                    last_error = error
                    container_logger.info(
                        "Image '%s' not found; trying next candidate (if any)", candidate
                    )
                except Exception as error:  # pragma: no cover - unexpected docker error
                    last_error = error
                    container_logger.info(
                        "Failed to create container for %s with image '%s': %s",
                        instance_id,
                        candidate,
                        error,
                    )

            if resolved_image is None:
                raise last_error or RuntimeError(
                    f"Unable to resolve container image for {instance_id}"
                )

            container.start()

            if commit is not None:
                container_logger.info("Checking out commit %s", commit)
                container.exec_run(
                    "git fetch",
                    workdir=DOCKER_WORKDIR,
                    user=DOCKER_USER,
                )
                val = container.exec_run(
                    f"git checkout {commit}",
                    workdir=DOCKER_WORKDIR,
                    user=DOCKER_USER,
                )
                if val.exit_code != 0:
                    container_logger.info(
                        "CHECKOUT FAILED: %s", val.output.decode(UTF8)
                    )
                    return container_logger, False

            if patch is not None:
                patch_file = run_log_dir / "patch.diff"
                patch_file.write_text(patch)
                container_logger.info(
                    "Patch written to %s, now applying to container...",
                    patch_file,
                )
                copy_to_container(container, patch_file, Path(DOCKER_PATCH))
                _swesmith_utils._apply_patch(  # type: ignore[attr-defined]
                    instance_id,
                    container,
                    container_logger,
                    is_gold,
                )

            eval_file = run_log_dir / "eval.sh"
            test_command, _ = _swesmith_utils.get_test_command(instance)
            eval_file.write_text(
                "\n".join(
                    [
                        "#!/bin/bash",
                        "set -uxo pipefail",
                        "source /opt/miniconda3/bin/activate",
                        f"conda activate {ENV_NAME}",
                        f"cd {DOCKER_WORKDIR}",
                        f": '{TEST_OUTPUT_START}'",
                        test_command,
                        f": '{TEST_OUTPUT_END}'",
                    ]
                )
                + "\n"
            )
            copy_to_container(container, eval_file, Path("/eval.sh"))

            test_output, timed_out, total_runtime = exec_run_with_timeout(
                container,
                "/bin/bash /eval.sh",
                timeout=timeout,
            )
            test_output_path = run_log_dir / LOG_TEST_OUTPUT
            container_logger.info("Test Runtime: %.2f seconds", total_runtime)
            with open(test_output_path, "w", encoding="utf-8") as handle:
                handle.write(test_output)
                if timed_out:
                    timeout_error = f"{TESTS_TIMEOUT}: {timeout} seconds exceeded"
                    handle.write(f"\n\n{timeout_error}")

            container_logger.info(
                "Test output for %s written to %s", instance_id, test_output_path
            )
            cleanup_container(client, container, container_logger)
            return container_logger, timed_out
        except Exception as exc:  # pragma: no cover - defensive parity with upstream
            error_msg = (
                f"Error validating {instance_id}: {exc}\n{traceback.format_exc()}"
            )
            if container_logger is not None:
                container_logger.info(error_msg)
                print(
                    f"Error validating {instance_id}: {exc}."
                    f" See {container_logger.log_file} for details."
                )
            else:  # pragma: no cover - defensive guard for early failures
                print(error_msg)
            if container_logger is not None:
                cleanup_container(client, container, container_logger)
            else:  # pragma: no cover - fallback when logger setup fails
                if container is not None:
                    try:
                        container.stop(timeout=1)
                    except Exception:
                        pass
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass
                client.close()
            return container_logger, False

    _swesmith_utils.run_patch_in_container = _run_patch_with_registry  # type: ignore[attr-defined]
    _swesmith_valid.run_patch_in_container = _run_patch_with_registry  # type: ignore[attr-defined]
    _REGISTRY_PATCH_APPLIED = True


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def run_validation(
    instance: dict[str, Any],
    run_id: str,
    timeout: int = TIMEOUT,
    run_min_pregold: bool = False,
) -> None:
    """Run SWE-smith validation with local registry awareness."""

    if _swesmith_utils is None or get_valid_report is None:
        raise RuntimeError(
            "SWE-smith validation utilities unavailable; cannot run validation"
        )

    ensure_validation_registry_support()

    instance_payload = dict(instance)
    instance_id = instance_payload[KEY_INSTANCE_ID]
    raw_image = instance_payload.get("raw_image_name", instance_payload[KEY_IMAGE_NAME])
    instance_payload["raw_image_name"] = raw_image

    canonical_name = _canonical_image_key(raw_image)
    instance_payload[KEY_IMAGE_NAME] = canonical_name

    registry_value = instance_payload.get("image_registry")
    if not registry_value:
        registry_value = _infer_registry_host(raw_image)
    if registry_value:
        registry_value = str(registry_value).strip()
        instance_payload["image_registry"] = registry_value
        REGISTRY_CACHE[canonical_name] = registry_value
    else:
        instance_payload.pop("image_registry", None)

    valid_folder = LOG_DIR_RUN_VALIDATION / run_id
    instance_dir = valid_folder / instance_id
    if instance_dir.exists():
        shutil.rmtree(instance_dir)
    valid_folder.mkdir(parents=True, exist_ok=True)
    report_path = instance_dir / LOG_REPORT

    patch_text = instance_payload.get(KEY_PATCH)
    if patch_text is None:
        patch_text = instance_payload.get("patch")
    if patch_text is None:
        raise ValueError(f"Instance '{instance_id}' missing patch text for validation")
    patch_text = str(patch_text)
    instance_payload[KEY_PATCH] = patch_text

    runner = _swesmith_utils.run_patch_in_container

    val_postgold_path = valid_folder / f"{canonical_name}{REF_SUFFIX}" / LOG_TEST_OUTPUT

    if run_min_pregold:
        ref_instance_id = f"{instance_id}{REF_SUFFIX}"
        ref_instance = dict(instance_payload)
        ref_instance[KEY_INSTANCE_ID] = ref_instance_id
        ref_instance.pop(KEY_PATCH, None)
        ref_result = runner(
            ref_instance,
            run_id,
            LOG_DIR_RUN_VALIDATION,
            timeout=timeout,
        )
        if ref_result is None:
            logger.error("Pre-gold container run failed for %s", ref_instance_id)
            _write_report(
                report_path,
                {
                    "error": "Pre-gold container run failed",
                    "stage": "pregold",
                    "instance": ref_instance_id,
                },
            )
            return
        pre_logger, pre_timed_out = ref_result
        try:
            if pre_timed_out:
                logger.info("Pre-gold run timed out for %s", ref_instance_id)
                _write_report(
                    report_path,
                    {KEY_TIMED_OUT: True, "timeout": timeout, "stage": "pregold"},
                )
                return

            ref_dir = valid_folder / ref_instance_id
            ref_output = ref_dir / LOG_TEST_OUTPUT
            if not ref_output.exists():
                logger.error(
                    "Reference test output missing at %s for %s", ref_output, ref_instance_id
                )
                _write_report(
                    report_path,
                    {
                        "error": "Reference test output not found",
                        "stage": "pregold",
                        "missing_path": str(ref_output),
                        "instance": ref_instance_id,
                    },
                )
                return

            val_postgold_path = instance_dir / LOG_TEST_OUTPUT_PRE_GOLD
            val_postgold_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ref_output, val_postgold_path)
        finally:
            if pre_logger is not None:
                close_logger(pre_logger)

    patched_result = runner(
        instance_payload,
        run_id,
        LOG_DIR_RUN_VALIDATION,
        patch=patch_text,
        timeout=timeout,
    )
    if patched_result is None:
        logger.error("Patched container run failed for %s", instance_id)
        _write_report(
            report_path,
            {
                "error": "Patched container run failed",
                "instance": instance_id,
                "stage": "postgold",
            },
        )
        return

    patched_logger, timed_out = patched_result
    try:
        if timed_out:
            logger.info("Patched run timed out for %s", instance_id)
            _write_report(
                report_path,
                {KEY_TIMED_OUT: True, "timeout": timeout, "stage": "postgold"},
            )
            return

        val_pregold_path = instance_dir / LOG_TEST_OUTPUT
        if not val_pregold_path.exists():
            logger.error(
                "Patched test output missing at %s for %s", val_pregold_path, instance_id
            )
            _write_report(
                report_path,
                {
                    "error": "Patched test output not found",
                    "stage": "postgold",
                    "missing_path": str(val_pregold_path),
                    "instance": instance_id,
                },
            )
            return

        report = get_valid_report(  # type: ignore[misc]
            val_pregold_path=val_pregold_path,
            val_postgold_path=val_postgold_path,
            instance=instance_payload,
        )
        _write_report(report_path, report)
    finally:
        if patched_logger is not None:
            close_logger(patched_logger)