import shutil
import random
import uuid
from textwrap import shorten
from datetime import datetime
from pathlib import Path
import tempfile
import logging
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import yaml
import threading
import traceback
import jinja2

from unidiff import PatchSet

from datasets import load_dataset


from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent

from swebench.harness.constants import (
    FAIL_TO_PASS,
    PASS_TO_PASS,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    KEY_INSTANCE_ID,
)
from swesmith.constants import (
    LOG_DIR_RUN_VALIDATION,
    KEY_TIMED_OUT,
    TEST_OUTPUT_START,
    TEST_OUTPUT_END,
)
from swesmith.harness.valid import run_validation

from athils import JsonLinesFile

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from dotenv import load_dotenv
load_dotenv()

CUR_DIR = Path(__file__).parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
AZURE_AD_TOKEN_PROVIDER = get_bearer_token_provider(DefaultAzureCredential(), os.getenv("AZURE_API_SCOPE", None))

# BACKEND = "docker"
BACKEND = "kubernetes"



def remove_added_test_files(patch: str) -> str:
    return str(PatchSet([
        str(f) for f in PatchSet(patch)
        if not (f.is_added_file and f.path.endswith(".py") and "test_" in f.path)
    ]))
    

def create_instance_id(image_name: str, seed: str) -> str:
    _, arch, repo_name, short_commit_sha = image_name.split('.')
    return f"{repo_name}.{short_commit_sha}.r2egym_featadd.seed_{seed}"



R2E_GYM_CONFIG = CUR_DIR / Path("r2egym_featadd.yaml")
R2E_GYM_PS_CONFIG = CUR_DIR / Path("r2egym_ps_generation.yaml")


def get_demo_problem_statements(swebv_dataset, n_demos: int = 3) -> list[str]:
    """
    Get a random sample of demonstration problem statements from SWE-bench Verified dataset.
    Similar to IssueGen.get_demo_issues()

    Args:
        swebv_dataset: The SWE-bench Verified dataset
        n_demos: Number of demonstration examples to return

    Returns:
        List of n_demos random problem statements, truncated to 2000 chars each
    """
    problem_statements = [
        instance["problem_statement"][:2000]  # Truncate to 2000 chars
        for instance in swebv_dataset
    ]
    # Return a random sample (different each time this is called)
    return random.sample(problem_statements, min(n_demos, len(problem_statements)))


def get_test_output(instance_id: str, run_id: str, max_chars: int = 20000) -> str:
    """
    Get test output from the validation run.
    Similar to IssueGen.get_test_output()

    Args:
        instance_id: The instance ID
        run_id: The run/experiment ID
        max_chars: Maximum number of characters to return

    Returns:
        Test output string, truncated and extracted between START and END markers
    """
    test_output_path = (
        LOG_DIR_RUN_VALIDATION
        / run_id
        / instance_id
        / LOG_TEST_OUTPUT
    )

    if not test_output_path.exists():
        logger.warning(f"Test output file not found: {test_output_path}")
        return "Test output not available."

    test_output = test_output_path.read_text()

    # Extract content between TEST_OUTPUT_START and TEST_OUTPUT_END markers
    start_idx = test_output.find(TEST_OUTPUT_START)
    end_idx = test_output.find(TEST_OUTPUT_END)

    if start_idx == -1 or end_idx == -1:
        logger.warning("Could not find test output markers, using full output")
        extracted = test_output
    else:
        extracted = test_output[start_idx + len(TEST_OUTPUT_START):end_idx]

    # Truncate to max_chars (simple character-based truncation)
    if len(extracted) > max_chars:
        half = max_chars // 2
        extracted = extracted[:half] + "\n\n(...)\n\n" + extracted[-half:]

    return extracted.strip()


def process_single_job(
    jspec,
    logdir: Path | str,
    model_name: str,
    run_id: str,
    demo_problem_statements: list[str],
) -> tuple[dict | None, bool, str | None]:
    """
    Process a single Docker image.

    Args:
        jspec: Job specification tuple (image_name, seed)
        logdir: Directory for logs
        model_name: Model name for the agent
        run_id: Run identifier
        demo_problem_statements: List of demo problem statements for this specific job

    Returns:
        Tuple of (jspec, success, error_message)
    """
    env = None
    image_name, seed = jspec
    jid = jspec
    instance_id = create_instance_id(image_name=image_name, seed=seed)

    try:
        logger.info(f"Starting job for {jid}")
        
        logdir = Path(logdir)
        image_output_dir = logdir / image_name / f"seed_{seed}"
        image_output_dir.mkdir(exist_ok=True, parents=True)

        pre_existing_patches = list(image_output_dir.rglob("*.patch"))
        if len(pre_existing_patches) > 0:
            for patch_file in pre_existing_patches:
                shutil.rmtree(patch_file.parent)

        _, arch, repo_name, short_commit_sha = image_name.split('.')
        repo = repo_name.replace('__', '/')

        ds = {
            "image_name": image_name,
            "repo": repo,
            "problem_statement": "",
            "base_commit": "main",
            "patch": ""  # Empty patch for bug generation mode
            # FAIL_TO_PASS: [],
            # PASS_TO_PASS: [],

        }
        env_args = EnvArgs(ds=ds)
        env = RepoEnv(env_args, logger=logger, backend=BACKEND, step_timeout=180)
        agent_args = AgentArgs.from_yaml(R2E_GYM_CONFIG)
        agent_args.llm_name = model_name
        agent = Agent(
            name="EditAgent",
            args=agent_args,
            logger=logger,
            litellm_completion_kwargs={"azure_ad_token_provider": AZURE_AD_TOKEN_PROVIDER}
        )

        trajectory = agent.run(
            env,
            max_steps=100,
            temperature=1.0,
            max_steps_absolute=100,
            use_fn_calling=True,
            scaffold="r2egym",
            max_token_limit=128000,
        )
        env.close()
        if trajectory.exit_reason != "agent":
            env.close()
            return None, False, f"R2E-Gym Agent Error: {trajectory.exit_reason}"


        if trajectory.output_patch is None or trajectory.output_patch.strip() == "":
            return None, False, "R2E-Gym agent empty patch"

        logger.info(f"Completed processing for image: {image_name}")

        patch_text = trajectory.output_patch
        patch_text = remove_added_test_files(patch_text)
        
        # Save trajectory to disk
        trajectory_output_path = image_output_dir / "trajectory.json"
        trajectory_output_path.write_text(trajectory.model_dump_json(indent=2))
        logger.info(f"Saved trajectory to {trajectory_output_path}")
            
        logger.info(f"Successfully generated patch for {image_name} with seed {seed}. Checking if it fails tests")
        report_path = LOG_DIR_RUN_VALIDATION / run_id / instance_id / LOG_REPORT

        if not report_path.exists():
            instance_data = {
                "strategy": "r2egym_featadd",
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

        logger.info(f"Found report after running validatoin check for {jid} ")
        report = json.load(open(report_path, "r"))
        f2p, p2p = report[FAIL_TO_PASS], report[PASS_TO_PASS]

        if len(f2p) > 5:
            logger.info(f"Generated bug results in more than 5 failing tests for {jid}")
            return None, False, "Too many failing tests"


        if KEY_TIMED_OUT in report or len(f2p) == 0 or len(p2p) == 0:
            logger.info(f"Generated patch for {jid} not buggy.")
            return None, False, "Generated patch not buggy"

        instance_data = {
            "instance_id": instance_id,
            "repo": f"swesmith/{repo_name}.{short_commit_sha}",
            "patch": patch_text,
            FAIL_TO_PASS: f2p,
            PASS_TO_PASS: p2p,
            "created_at": datetime.now().isoformat(),
            "image_name": image_name,
        }

        logger.info(f"Successfully analysed validation report for {jid}")
        logger.info(f"Generating problem description text for {jid}")

        try:
            # Get test output from validation run
            test_output = get_test_output(instance_id, run_id, max_chars=20000)
            logger.info(f"Retrieved test output ({len(test_output)} chars) for {jid}")

            ps_env = RepoEnv(env_args, logger=logger, backend=BACKEND, step_timeout=180)

            # Apply the patch to the environment so the agent can see the changes
            logger.info(f"Applying buggy patch to environment for problem statement generation")
            ps_env.runtime.apply_patch(patch_text)

            # Load the problem statement generation agent config
            ps_agent_args = AgentArgs.from_yaml(R2E_GYM_PS_CONFIG)
            ps_agent_args.llm_name = model_name

            # Render instance_prompt with demonstrations and test output using Jinja2
            if demo_problem_statements:
                env_jinja = jinja2.Environment()
                template = env_jinja.from_string(ps_agent_args.instance_prompt)
                rendered_instance_prompt = template.render(
                    demo_problem_statements=demo_problem_statements,
                    test_output=test_output,
                )
                ps_agent_args.instance_prompt = rendered_instance_prompt
                logger.info(f"Rendered instance prompt with {len(demo_problem_statements)} demonstrations and test output")

            ps_agent = Agent(
                name="ProblemStatementAgent",
                args=ps_agent_args,
                logger=logger,
                litellm_completion_kwargs={"azure_ad_token_provider": AZURE_AD_TOKEN_PROVIDER}
            )

            # Run the agent to generate problem statement
            ps_trajectory = ps_agent.run(
                ps_env,
                max_steps=50,
                temperature=0.7,
                max_steps_absolute=50,
                use_fn_calling=True,
                scaffold="r2egym",
                max_token_limit=128000,
            )

            # Save the PS generation trajectory to disk
            ps_trajectory_output_path = image_output_dir / "ps_generation_trajectory.json"
            ps_trajectory_output_path.write_text(ps_trajectory.model_dump_json(indent=2))
            logger.info(f"Saved PS generation trajectory to {ps_trajectory_output_path}")

            # Extract the problem statement from /testbed/problem_statement.txt
            problem_statement_output, error_code = ps_env.runtime.run("cat /testbed/problem_statement.txt")

            ps_env.close()

            if error_code != "0" or not problem_statement_output.strip():
                logger.warning(f"Could not generate problem statement for {jid}.")
                return None, False, "Could not generate problem statement"
            else:
                problem_statement = problem_statement_output.strip()
                logger.info(f"Successfully extracted problem statement for {jid}")

            # Add problem statement to instance data
            instance_data["problem_statement"] = problem_statement

        except Exception as e:
            logger.error(f"Error generating problem statement for {jid}: {str(e)}")
            logger.error(traceback.format_exc())
            return None, False, "Error generating problem statement"
            
        return (instance_data, True, None)
            
    except Exception as e:
        traceback.print_exc()
        error_msg = f"Error processing {jid}: {str(e)}"
        logger.error(error_msg)
        if env is not None:
            env.close()
        return (None, False, error_msg)


def regular(
    images: str | Path,
    model_name: str,
    output_file: str | Path,
    run_id: str,
    logdir: str | Path,
    seed_per_image: int = 1,
    max_workers: int = 1,
    max_tries: int = 1,
    num_jobs: int | None = None,
    shuffle: bool = False,
):
    """
    Process Docker images in parallel with retry logic.

    Args:
        images: Path to text file containing newline seperated list of images
        model_name: Model name for the agent
        output_file: File to store outputs
        run_id: Name for run
        seed_per_image: Number of different seeds per image attempt
        api_key: API key for the model
        num_images: Limit number of images to process
        max_workers: Maximum number of parallel workers
        max_tries: Maximum number of retry attempts per image
        num_jobs: Number of jobs to attempt. If None then attempt all
    """


    image_names = Path(images).read_text().splitlines()
    jobs_specs = [(i, str(uuid.uuid4())[:10]) for i in image_names for _ in range(seed_per_image)]

    if shuffle:
        random.shuffle(jobs_specs)

    # Create output directory
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    if output_file.exists():
        pre_existing_data = json.load(open(output_file, "r"))
        existing_instance_ids = set([i['instance_id'] for i in pre_existing_data])
        jobs_specs = [(i, s) for (i, s) in jobs_specs if create_instance_id(i, s) not in existing_instance_ids]

    num_jobs = len(jobs_specs) if num_jobs is None else num_jobs
    jobs_specs = jobs_specs[:num_jobs]

    logger.info(f"Processing {len(jobs_specs)} jobs with {max_workers} workers, max {max_tries} tries per job")

    # Create attempt log file to track all attempts (image, seed) with their outcomes
    attempt_log_file = output_file.parent / f"{output_file.stem}_attempts.jsonl"
    attempt_log_lock = threading.Lock()

    def log_attempt(image_name: str, seed: str, attempt: int, status: str, error_reason: str | None = None, instance_id: str | None = None):
        """
        Log an attempt to the attempt log file.

        Args:
            image_name: Name of the Docker image
            seed: Random seed used
            attempt: Attempt number (1-indexed)
            status: Status of the attempt ("success" or "failed")
            error_reason: Reason for failure if status is "failed"
            instance_id: Instance ID if available
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "image_name": image_name,
            "seed": seed,
            "instance_id": instance_id or create_instance_id(image_name, seed),
            "attempt": attempt,
            "status": status,
            "error_reason": error_reason,
        }

        with attempt_log_lock:
            with open(attempt_log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

    logger.info(f"Attempt log will be written to: {attempt_log_file}")

    # Load SWE-bench Verified for demonstration problem statements (loaded once, used by all workers)
    logger.info("Loading SWE-bench Verified dataset for demonstration problem statements...")
    swebv_dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    logger.info(f"Loaded SWE-bench Verified dataset with {len(swebv_dataset)} instances")

    # Track results and retry queue
    successful_processes = 0
    failed_processes = 0
    retry_queue = []  # List of (image_name, attempt_count) tuples

    # Initialize all images with attempt count 1
    current_batch = [(jspec, 1) for jspec in jobs_specs]

    while current_batch:
        logger.info(f"Processing batch of {len(current_batch)} images...")

        # Process current batch in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks for current batch, each with a fresh random sample of demos
            future_to_jspec = {
                executor.submit(
                    process_single_job,
                    jspec,
                    logdir,
                    model_name,
                    run_id,
                    get_demo_problem_statements(swebv_dataset, n_demos=3)  # Fresh random sample for each job
                ): (jspec, attempt)
                for jspec, attempt in current_batch
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_jspec):
                jspec, attempt = future_to_jspec[future]
                image_name, seed = jspec
                result, success, error_msg = future.result()

                if success:
                    successful_processes += 1
                    instance_id = result.get('instance_id') if result else None
                    log_attempt(image_name, seed, attempt, "success", error_reason=None, instance_id=instance_id)
                    logger.info(f"✓ Completed {jspec} on attempt {attempt} ({successful_processes}/{len(jobs_specs)})")
                    if output_file.exists():
                        pre_existing_data = json.load(open(output_file, "r"))
                    else:
                        pre_existing_data = []
                    pre_existing_data.append(result)
                    json.dump(pre_existing_data, open(output_file, "w"), indent=2)
                else:
                    log_attempt(image_name, seed, attempt, "failed", error_reason=error_msg)
                    if attempt < max_tries:
                        # Add to retry queue for next batch
                        retry_queue.append((jspec, attempt + 1))
                        logger.warning(f"⚠ Failed {jspec} on attempt {attempt}/{max_tries}: {error_msg}. Will retry.")
                    else:
                        # Max tries reached, mark as permanently failed
                        failed_processes += 1
                        logger.error(f"✗ Failed {jspec} after {max_tries} attempts: {error_msg}. Giving up.")
        
        # Prepare next batch from retry queue
        current_batch = retry_queue.copy()
        retry_queue.clear()

    # Summary
    total_processed = successful_processes + failed_processes
    logger.info(f"Processing complete! Success: {successful_processes}, Failed: {failed_processes}, Total: {total_processed}")

    # Generate summary statistics from attempt log
    if attempt_log_file.exists():
        logger.info(f"\n{'='*60}")
        logger.info("ATTEMPT LOG SUMMARY")
        logger.info(f"{'='*60}")

        attempt_data = []
        with open(attempt_log_file, "r") as f:
            for line in f:
                attempt_data.append(json.loads(line))

        # Count failures by error reason
        error_counts = {}
        total_attempts = len(attempt_data)
        success_count = sum(1 for a in attempt_data if a["status"] == "success")
        failed_count = sum(1 for a in attempt_data if a["status"] == "failed")

        for entry in attempt_data:
            if entry["status"] == "failed" and entry["error_reason"]:
                error_reason = entry["error_reason"]
                error_counts[error_reason] = error_counts.get(error_reason, 0) + 1

        logger.info(f"Total attempts logged: {total_attempts}")
        logger.info(f"Successful attempts: {success_count} ({success_count/total_attempts*100:.1f}%)")
        logger.info(f"Failed attempts: {failed_count} ({failed_count/total_attempts*100:.1f}%)")
        logger.info(f"\nFailure breakdown by reason:")
        for reason, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {reason}: {count} ({count/failed_count*100:.1f}% of failures)")
        logger.info(f"{'='*60}\n")

    return {
        'successful': successful_processes,
        'failed': failed_processes,
        'total': len(jobs_specs),
        'attempt_log_file': str(attempt_log_file)
    }



if __name__ == '__main__':
    import fire
    fire.Fire(regular)
