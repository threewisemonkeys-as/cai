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

from unidiff import PatchSet

from datasets import load_dataset


from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent

from swebench.harness.constants import (
    FAIL_TO_PASS,
    PASS_TO_PASS,
    LOG_REPORT,
)
from swesmith.constants import LOG_DIR_RUN_VALIDATION, KEY_TIMED_OUT
from swesmith.harness.valid import run_validation
from swesmith.issue_gen.generate import IssueGen

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


ISSUE_GEN_CONFIG_FILE_PATH = CUR_DIR / Path("sans_patch_issue_gen.yaml")


class CustomIssueGen(IssueGen):
    def __init__(
        self,
        model: str,
        use_existing: bool,
        n_workers: int,
        experiment_id: Path,
    ):
        self.experiment_id = experiment_id
        self.model = model
        self.use_existing = use_existing
        self.n_workers = n_workers

        self.swebv = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

        self.config = yaml.safe_load(ISSUE_GEN_CONFIG_FILE_PATH.read_text())
        settings = self.config.get("settings", {})
        self.n_instructions = settings.get("n_instructions", 1)
        self.max_var_tokens = settings.get("max_var_tokens", 10_000)

        self._lock = threading.Lock()
    
    def get_test_functions(self, instance: dict) -> tuple[list[str], list[str]]:
        """
        Override to avoid cloning repos that don't exist on GitHub.
        For R2E-Gym generated bugs, we don't have access to test source code.
        
        Returns:
            Empty list of test functions, empty list of repos to remove
        """
        return [], []


def remove_added_test_files(patch: str) -> str:
    return str(PatchSet([
        str(f) for f in PatchSet(patch)
        if not (f.is_added_file and f.path.endswith(".py") and "test_" in f.path)
    ]))
    

def create_instance_id(image_name: str, seed: str) -> str:
    _, arch, repo_name, short_commit_sha = image_name.split('.')
    return f"{repo_name}.{short_commit_sha}.r2egym_featadd.seed_{seed}"



R2E_GYM_CONFIG = CUR_DIR / Path("r2egym_featadd.yaml")

def process_single_job(
    jspec, 
    logdir: Path | str, 
    model_name: str, 
    run_id: str,
    issue_generator: CustomIssueGen,
) -> tuple[dict | None, bool, str | None]:
    """
    Process a single Docker image.
    
    Args:
        jspec: Job specification tuple (image_name, seed)
        logdir: Directory for logs
        model_name: Model name for the agent
        run_id: Run identifier
        issue_generator: Shared IssueGen instance (reused across all workers)
    
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
            # litellm_completion_kwargs={"azure_ad_token_provider": AZURE_AD_TOKEN_PROVIDER}
        )

        trajectory = agent.run(
            env,
            max_steps=100,
            temperature=1.0,
            max_steps_absolute=100,
            use_fn_calling=False,
            scaffold="r2egym",
            max_token_limit=128000,
        )
        env.close()
        if trajectory.exit_reason != "agent":
            env.close()
            return None, False, "R2E-Gym Agent Error"


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
                timeout=600,  # 10 minutes timeout for validation
            )

        if not report_path.exists():
            logger.info(f"Could not find validation run report for {jid}")
            return None, False, "Could not find validation run report"

        logger.info(f"Found report after running validatoin check for {jid} ")
        report = json.load(open(report_path, "r"))
        f2p, p2p = report[FAIL_TO_PASS], report[PASS_TO_PASS]
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
            with tempfile.NamedTemporaryFile(delete_on_close=False, mode="w+") as fp:
                logger.info(f"Calling issue_generator.generate_issue for {jid}")
                issue_generator.generate_issue(instance_data, 0, fp)
                fp.flush()
                fp.seek(0)
                instance_data = json.load(fp)
                logger.info(f"Successfully generated issue for {jid}")
        except Exception as e:
            logger.error(f"Error generating issue for {jid}: {str(e)}")
            logger.error(traceback.format_exc())
            return (None, False, f"Error generating issue: {str(e)}")

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
    # num_jobs: int | None = None,
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

    num_jobs = len(jobs_specs)
    jobs_specs = jobs_specs[:num_jobs]
    
    logger.info(f"Processing {len(jobs_specs)} jobs with {max_workers} workers, max {max_tries} tries per job")
    
    # Create shared issue generator instance (loaded once, used by all workers)
    logger.info("Initializing shared issue generator (loading SWE-bench dataset)...")
    shared_issue_generator = CustomIssueGen(
        model=model_name,
        use_existing=True,
        n_workers=1,
        experiment_id=run_id,
    )
    logger.info("Issue generator initialized and ready")
    
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
            # Submit all tasks for current batch
            future_to_jspec = {
                executor.submit(process_single_job, jspec, logdir, model_name, run_id, shared_issue_generator): (jspec, attempt)
                for jspec, attempt in current_batch
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_jspec):
                jspec, attempt = future_to_jspec[future]
                result, success, error_msg = future.result()
                
                if success:
                    successful_processes += 1
                    logger.info(f"✓ Completed {jspec} on attempt {attempt} ({successful_processes}/{len(jobs_specs)})")
                    if output_file.exists():
                        pre_existing_data = json.load(open(output_file, "r"))
                    else:
                        pre_existing_data = []
                    pre_existing_data.append(result)
                    json.dump(pre_existing_data, open(output_file, "w"), indent=2)
                else:
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
    
    return {
        'successful': successful_processes,
        'failed': failed_processes,
        'total': len(jobs_specs)
    }



if __name__ == '__main__':
    import fire
    fire.Fire(regular)