"""High-level orchestration logic for Debug-Gym bug generation runs."""

from __future__ import annotations

import logging
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import BuggenRuntimeConfig, DebugGymSessionConfig, load_pipeline_config
from .issue_generation import CustomIssueGen
from .processing import JobSpec, process_single_job
from .utils import (
    _append_progress_log,
    _base_progress_entry,
    _load_existing_results,
    _record_success,
    create_instance_id,
)

logger = logging.getLogger(__name__)

def regular(
    config: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, int]:
    """Run the full Debug-Gym bug generation pipeline.

    Args:
        config: Optional override path for the YAML configuration used to set
            up the environment, agent, and runtime parameters. When omitted the
            default bundled configuration is used.
        run_id: Optional identifier to override the ID specified in the YAML
            configuration. Useful for resuming or grouping multiple runs.

    Returns:
        A dictionary summarising the number of successful and failed jobs along
        with the total jobs attempted.
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

    image_names = runtime_config.images_file.read_text().splitlines()
    jobs_specs: list[JobSpec] = [
        (image_name, str(uuid.uuid4())[:20])
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
                        llm_name=model_name,
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

    progress_bar = None
    processing_logger = logging.getLogger("debug_gym_bundle.processing")
    original_processing_level = processing_logger.level
    processing_level_overridden = False

    def _log(level: int, msg: str, *args: Any) -> None:
        if progress_bar is not None:
            try:
                rendered = msg % args if args else msg
            except TypeError:
                rendered = msg.format(*args)
            progress_bar.write(f"[{logging.getLevelName(level)}] {rendered}")
        else:
            logger.log(level, msg, *args)

    def info(msg: str, *args: Any) -> None:
        _log(logging.INFO, msg, *args)

    def warning(msg: str, *args: Any) -> None:
        _log(logging.WARNING, msg, *args)

    def error(msg: str, *args: Any) -> None:
        _log(logging.ERROR, msg, *args)

    if runtime_config.show_progress and jobs_specs:
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover - fallback when tqdm missing
            warning(
                "Progress display requested but tqdm is not installed; continuing without progress bar.",
            )
        else:
            progress_bar = tqdm(
                total=len(jobs_specs),
                desc="Buggen Jobs",
                unit="job",
                dynamic_ncols=True,
                leave=True,
            )
            processing_logger.setLevel(logging.WARNING)
            processing_level_overridden = True

    try:
        info(
            "Processing %d jobs with %d workers, max %d tries per job using Debug-Gym pipeline.",
            len(jobs_specs),
            runtime_config.max_workers,
            runtime_config.max_tries,
        )

        info("Initializing shared issue generator (loading SWE-bench dataset)...")
        shared_issue_generator = CustomIssueGen(
            model=model_name,
            use_existing=True,
            n_workers=1,
            experiment_id=runtime_config.run_id,
            config_path=session_config.issue_gen_config,
        )
        # Reuse a single issue generator so the SWE-bench dataset and LLM stay warm.
        info("Issue generator initialized and ready")

        successful_processes = 0
        failed_processes = 0
        retry_queue: list[tuple[JobSpec, int]] = []
        current_batch: list[tuple[JobSpec, int]] = [(jspec, 1) for jspec in jobs_specs]
        results_lock = threading.Lock()

        while current_batch:
            info("Processing batch of %d images...", len(current_batch))

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
                        runtime_config.validation_timeout,
                        runtime_config.max_fail_fraction,
                    ): (jspec, attempt)
                    for jspec, attempt in current_batch
                }

                for future in as_completed(future_to_jspec):
                    jspec, attempt = future_to_jspec[future]
                    result, success, error_msg = future.result()

                    retryable = True
                    reason = error_msg
                    if error_msg and error_msg.startswith("non_retryable:"):
                        retryable = False
                        reason = error_msg.removeprefix("non_retryable:").strip()

                    if success and result is not None:
                        successful_processes += 1
                        info(
                            "✓ Completed %s on attempt %d (%d/%d)",
                            jspec,
                            attempt,
                            successful_processes,
                            len(jobs_specs),
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
                                    llm_name=model_name,
                                    image_name=jspec[0],
                                    seed=jspec[1],
                                    attempt=attempt,
                                ),
                                "status": "issue_generated",
                            },
                        )
                        if progress_bar is not None:
                            progress_bar.update(1)
                            progress_bar.set_postfix(
                                success=successful_processes,
                                failed=failed_processes,
                                retries=len(retry_queue),
                            )
                    else:
                        entry = {
                            **_base_progress_entry(
                                run_id=runtime_config.run_id,
                                instance_id=create_instance_id(*jspec),
                                llm_name=model_name,
                                image_name=jspec[0],
                                seed=jspec[1],
                                attempt=attempt,
                            ),
                            "reason": reason,
                        }
                        if retryable and attempt < runtime_config.max_tries:
                            retry_queue.append((jspec, attempt + 1))
                            warning(
                                "⚠ Failed %s on attempt %d/%d: %s. Will retry.",
                                jspec,
                                attempt,
                                runtime_config.max_tries,
                                reason,
                            )
                            entry["status"] = "retry_scheduled"
                            if progress_bar is not None:
                                progress_bar.set_postfix(
                                    success=successful_processes,
                                    failed=failed_processes,
                                    retries=len(retry_queue),
                                )
                        else:
                            failed_processes += 1
                            error(
                                "✗ Failed %s after %d attempts: %s. Giving up.",
                                jspec,
                                runtime_config.max_tries,
                                reason,
                            )
                            entry["status"] = "failed"
                            if progress_bar is not None:
                                progress_bar.update(1)
                                progress_bar.set_postfix(
                                    success=successful_processes,
                                    failed=failed_processes,
                                    retries=len(retry_queue),
                                )
                        _append_progress_log(progress_log_path, progress_lock, entry)

            current_batch = retry_queue.copy()
            retry_queue.clear()

        total_processed = successful_processes + failed_processes

        info(
            "Processing complete! Success: %d, Failed: %d, Total attempted: %d",
            successful_processes,
            failed_processes,
            total_processed,
        )

        return {
            "successful": successful_processes,
            "failed": failed_processes,
            "total": len(jobs_specs),
        }
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if processing_level_overridden:
            processing_logger.setLevel(original_processing_level)
