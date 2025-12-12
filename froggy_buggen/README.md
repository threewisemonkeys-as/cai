# Froggy Buggen

Bug generation pipeline using `FreeEnv` and `FroggyAgent` from debug_gym,
validated via SWE-smith.

## Contents

```
froggy_buggen/
├── __init__.py              # Package exports (re-exports `regular`)
├── __main__.py              # CLI entry point
├── config.py                # Dataclasses and YAML loader for pipeline configuration
├── pipeline.py              # Orchestration layer that coordinates multi-image runs
├── processing.py            # Execution of a single bug generation run and validation pass
├── issue_generation.py      # LLM-powered issue synthesis utilities
├── validation.py            # SWE-smith validation with registry support
├── utils.py                 # Shared helpers for logging, persistence, and patch handling
├── generate_issues_from_logs.py  # Regenerate issues from existing validation logs
├── froggy_buggen.yaml       # Default configuration
├── prompts/                 # Prompt templates
│   ├── system_prompt.md     # FroggyAgent system prompt
│   ├── instance_prompt.md   # Task-specific instructions
│   └── issue_gen.yaml       # Issue generation prompt configuration
├── swesmith/image_names.txt # List of SWE-smith images to process
└── results/                 # Generated logs and JSON outputs (initially empty)
```

## How to Run

1. Install the required dependencies:
   - `debug_gym` (error_handling branch or later)
   - `swebench`, `swesmith`
   - `fire`, `jinja2`, `datasets`, `unidiff`, `pyyaml`

2. Run the pipeline as a module:

   ```bash
   python -m froggy_buggen
   ```

   Or with a custom configuration file and/or run ID:

   ```bash
   python -m froggy_buggen --config /path/to/override.yaml --run_id custom-run
   ```

3. To regenerate issue descriptions from existing validation logs:

   ```bash
   python -m froggy_buggen.generate_issues_from_logs --config /path/to/config.yaml
   ```

4. Results accumulate in `results/froggy_buggen_results.json`. Log files
   live in `results/froggy_runs/` and per-run progress entries are appended
   to `results/froggy_buggen_results.progress.jsonl`.

## Configuration

Edit `froggy_buggen.yaml` to customize the pipeline:

### LLM & Prompts
- `llm`: Model name for the agent and issue generation
- `agent.system_prompt_file`: Path to FroggyAgent system prompt (or inline via `system_prompt`)
- `agent.instance_prompt_file`: Path to task instructions (or inline via `instance_prompt`)
- `agent.max_steps`: Maximum agent steps per run
- `issue_gen_config`: Path to issue generation prompt template

### Environment
- `environment.type`: Terminal type (`docker` or `kubernetes`)
- `environment.workspace_dir`: Working directory inside container (default: `/testbed`)
- `environment.registry`: Private registry for pulling images
- Additional keys (`namespace`, `pod_spec_kwargs`, etc.) are forwarded to `select_terminal`

### Runtime
- `run.images`: File containing image names to process
- `run.seed_per_image`: Number of seeds per image
- `run.max_workers`: Parallel job count
- `run.max_tries`: Retry attempts per job
- `run.validation_timeout`: Timeout in seconds for SWE-smith validation
- `run.max_fail_fraction`: Maximum fraction of failing tests to accept (0-1, required)
- `run.output_file`: Path for results JSON
- `run.logdir`: Directory for logs
