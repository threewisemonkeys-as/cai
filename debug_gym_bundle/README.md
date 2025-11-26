# Debug Gym Bug Generation Bundle

This folder contains a self-contained copy of the Debug-Gym bug generation
pipeline refactored during this session. You can copy `debug_gym_bundle/`
elsewhere (for example into `cai/`) and run `debug_gym_buggen.py` directly.

## Contents

- `debug_gym_buggen.py`: main entry point (`regular`) for orchestrating bug generation.
- `debug_gym_buggen.yaml`: default configuration. All runtime options live here.
- `free_env_buggen_instructions.md`: instructions read by the Debug-Gym FreeEnv.
- `sans_patch_issue_gen.yaml`: SWE-smith issue generation prompt configuration.
- `swesmith/image_names.txt`: list of SWE-smith images to process.
- `results/`: destination for generated logs and JSON outputs (initially empty).

## How to Run

1. Install the same Python environment specified for the parent project
   (see the original repository requirements).
2. From this directory, launch the pipeline:

   ```bash
   python debug_gym_buggen.py
   ```

   or point to a different configuration file:

   ```bash
   python debug_gym_buggen.py --config /path/to/override.yaml
   ```

3. Results accumulate in `results/debug_gym_buggen_results.json`.  Log files
   live in `results/debug_gym_runs/`.

### Updating Configuration

- Change the LLM by editing `llm` inside `debug_gym_buggen.yaml`.
- Adjust images, output locations, or worker counts in the `run` section.
- Provide a different instruction file by updating `environment.instructions_file`.

> Note: The script requires the `fire` package for its CLI interface, plus all
> Debug-Gym, SWE-bench, and SWE-smith dependencies used in the original repo.
