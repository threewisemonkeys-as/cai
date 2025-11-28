# Debug Gym Bug Generation Bundle

This folder contains a self-contained copy of the Debug-Gym bug generation
pipeline refactored into modular components. You can copy
`debug_gym_bundle/` elsewhere (for example into `cai/`) and run
`debug_gym_buggen.py` directly.

## Contents

- `debug_gym_buggen.py`: main entry point (exposes `regular`) for orchestrating bug generation.
- `__init__.py`: re-exports `regular` for convenient imports.
- `config.py`: dataclasses and YAML loader for pipeline configuration.
- `issue_generation.py`: Debug-Gym powered issue synthesis utilities.
- `processing.py`: execution of a single Debug-Gym run and validation pass.
- `pipeline.py`: orchestration layer that coordinates multi-image runs.
- `utils.py`: shared helpers for logging, persistence, and patch handling.
- `generate_issues_from_logs.py`: replays issue generation against the existing
   `logs/run_validation` tree so past Debug-Gym runs receive problem statements
   without re-running the full pipeline. Invoke it with the same YAML config used
   for `debug_gym_buggen.py` to reuse prompts and output locations.
- `debug_gym_buggen.yaml`: default configuration. All runtime options live here.
- `free_env_buggen_instructions.md`: instructions read by the Debug-Gym FreeEnv.
- `free_agent_system_prompt.md`: optional override for the FreeAgent system prompt.
- `sans_patch_issue_gen.yaml`: SWE-smith issue generation prompt configuration.
- `swesmith/image_names.txt`: list of SWE-smith images to process.
- `results/`: destination for generated logs and JSON outputs (initially empty).

## How to Run

1. Install the same Python environment specified for the parent project
   (see the original repository requirements).
2. Ensure required third-party packages are installed: `fire` for the CLI,
   plus Debug-Gym, SWE-bench, SWE-smith, Jinja2, Datasets, and Unidiff.
3. From this directory, launch the pipeline:

   ```bash
   python debug_gym_buggen.py
   ```

   or point to a different configuration file and/or override the run ID:

   ```bash
   python debug_gym_buggen.py --config /path/to/override.yaml --run_id custom-run
   ```

4. Results accumulate in `results/debug_gym_buggen_results.json`. Log files
   live in `results/debug_gym_runs/` and per-run progress entries (including the
   LLM name used) are appended to `results/debug_gym_buggen_results.progress.jsonl`.

### Updating Configuration

- Change the LLM by editing `llm` inside `debug_gym_buggen.yaml`.
- Point to an alternate issue-generation template via `issue_gen_config`.
- Adjust images, output locations, or worker counts in the `run` section.
- Set `validation_timeout` (seconds) inside the `run` section to override the default
   SWE-smith validation timeout (leave unset to use the library default).
- Specify `max_fail_fraction` (0–1] in the `run` section to control how large a
  regression we accept. The pipeline rejects validation reports where more than that
  fraction of evaluated tests flip to failing; this field is required.
- Provide a different instruction file by updating `environment.instructions_file`.
- Override the agent system prompt by editing `agent.system_prompt_file` or supplying
   an inline `agent.system_prompt` value.
- Any additional keys under `environment` (for example `type`, `registry`,
  Kubernetes `namespace`/`pod_spec_kwargs`, or Docker credentials) are forwarded to
  the underlying `select_terminal` call. When `registry` is provided, the same value
  is reused during SWE-smith validation so images are pulled from your private
  registry instead of Docker Hub.

> Note: The script requires the `fire` package for its CLI interface, plus all
> Debug-Gym, SWE-bench, SWE-smith, and supporting dependencies used in the
> original repo.
