#!/usr/bin/env python3
"""Generate issue descriptions for existing Debug-Gym validation logs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:  # Running as a script; ensure package root is importable
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from debug_gym_bundle.config import load_pipeline_config
from debug_gym_bundle.issue_generation import CustomIssueGen
from debug_gym_bundle.utils import (
    _build_instance_from_logs,
    _load_existing_results,
    _persist_results,
)
from swebench.harness.constants import KEY_INSTANCE_ID
from swesmith.constants import LOG_DIR_RUN_VALIDATION


def _list_instance_dirs(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        logging.warning("Run directory missing: %s", run_dir)
        return []
    return [child for child in sorted(run_dir.iterdir()) if child.is_dir()]


def _normalise_instance_id(entry: dict[str, object]) -> str | None:
    value = entry.get(KEY_INSTANCE_ID) or entry.get("instance_id")
    if isinstance(value, str):
        value = value.strip()
    return value or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Override buggen YAML config")
    parser.add_argument(
        "--run-id",
        type=str,
        help="Limit processing to a single run directory under LOG_DIR_RUN_VALIDATION",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached issue metadata and regenerate everything",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most N instances",
    )
    parser.add_argument(
        "--instance",
        dest="instances",
        action="append",
        help="Only process specific instance ids (repeatable)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    session_config, runtime_config = load_pipeline_config(args.config)
    model_name = (session_config.llm_name or "").strip()
    if not model_name:
        raise ValueError("Pipeline config is missing an LLM name")

    output_path = runtime_config.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_results = _load_existing_results(output_path)
    index = {
        inst_id: idx
        for idx, inst_id in enumerate(
            [_normalise_instance_id(entry) for entry in existing_results]
        )
        if inst_id
    }

    validation_root = LOG_DIR_RUN_VALIDATION
    run_dirs = [validation_root / args.run_id] if args.run_id else [
        path for path in sorted(validation_root.iterdir()) if path.is_dir()
    ]
    if not run_dirs:
        logging.warning("No validation logs found under %s", validation_root)
        return

    selected_ids = set(args.instances or [])

    processed = generated = skipped = 0
    for run_dir in run_dirs:
        issue_gen = CustomIssueGen(
            model=model_name,
            use_existing=not args.refresh,
            n_workers=1,
            experiment_id=run_dir.name,
            config_path=session_config.issue_gen_config,
        )

        for instance_dir in _list_instance_dirs(run_dir):
            if args.limit is not None and processed >= args.limit:
                break

            instance_id = instance_dir.name
            if selected_ids and instance_id not in selected_ids:
                continue

            processed += 1

            instance_payload, rejection = _build_instance_from_logs(
                instance_dir,
                runtime_config,
            )
            if instance_payload is None:
                skipped += 1
                if rejection:
                    logging.info("Skipping %s: %s", instance_id, rejection)
                continue

            if instance_id in index and not args.refresh:
                skipped += 1
                logging.info(
                    "Skipping %s: already present in %s",
                    instance_id,
                    output_path,
                )
                continue

            updated_payload = issue_gen.generate_issue(instance_payload)
            if instance_id in index:
                existing_results[index[instance_id]] = updated_payload
            else:
                index[instance_id] = len(existing_results)
                existing_results.append(updated_payload)
            generated += 1

    if generated:
        _persist_results(output_path, existing_results)
        logging.info("Persisted %d instances to %s", len(existing_results), output_path)

    logging.info(
        "Completed run: processed=%d generated=%d skipped=%d",
        processed,
        generated,
        skipped,
    )


if __name__ == "__main__":
    main()
