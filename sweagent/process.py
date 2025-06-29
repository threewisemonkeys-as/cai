import shutil
from pathlib import Path
import tempfile
import logging
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import traceback


from sweagent.run.common import BasicCLI
from sweagent.run.run_single import RunSingle, RunSingleConfig


from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from dotenv import load_dotenv
load_dotenv()

CUR_DIR = Path(__file__).parent
BUGGEN_CONFIG_FILE = CUR_DIR / "buggen.yaml"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COST_LIM = 20.0
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
AZURE_AD_TOKEN_PROVIDER = get_bearer_token_provider(DefaultAzureCredential(), os.getenv("AZURE_API_SCOPE", None))

def get_full_commit_id(repo: str, short_commit_id: str) -> str | None:
    """Fetch the full commit SHA from GitHub API."""
    response = requests.get(
        url=f"https://api.github.com/repos/{repo}/commits/{short_commit_id}",
        headers={"Authorization": GITHUB_TOKEN}
    )
    
    if response.status_code == 200:
        commit_data = response.json()
        return commit_data['sha']
    else:
        return None

# PROBLEM_STATEMENT = "" # TODO


def create_instance_id(image_name: str, seed: int) -> str:
    _, arch, repo_name, short_commit_sha = image_name.split('.')
    return f"{repo_name}.{short_commit_sha}.sweagent_buggen.seed_{seed}"

def process_single_job(
    jspec, 
    sweagent_logdir: Path | str, 
    model_name: str, 
    run_id: str,
    api_key: str | None = None,
) -> tuple[dict | None, bool, str | None]:
    """
    Process a single Docker image.
    
    Returns:
        Tuple of (jspec, success, error_message)
    """
    image_name, seed = jspec
    instance_id = create_instance_id(image_name=image_name, seed=seed)

    try:
        logger.info(f"Starting job for image {image_name} with seed {seed}")
        
        # Create output directory for this image
        sweagent_logdir = Path(sweagent_logdir)
        image_output_dir = sweagent_logdir / image_name / f"seed_{seed}"
        image_output_dir.mkdir(exist_ok=True, parents=True)

        pre_existing_patches = list(image_output_dir.rglob("*.patch"))
        if len(pre_existing_patches) > 0:
            for patch_file in pre_existing_patches:
                shutil.rmtree(patch_file.parent)

        _, arch, repo_name, short_commit_sha = image_name.split('.')
        
        repo = repo_name.replace('__', '/')
        full_commit_sha = get_full_commit_id(repo, short_commit_sha)
        if full_commit_sha is None:
            error_msg = f"Could not get full commit sha for {image_name}"
            logger.warning(error_msg)
            return (image_name, False, error_msg)
            
        
        # Build sweagent arguments
        url = f"https://github.com/{repo}"
        config_file = BUGGEN_CONFIG_FILE
        args = ["run"]
        
        if config_file is not None: 
            args.extend([f"--config", str(config_file)])

            
        args += [
            f"--agent.model.name={model_name}",
            f"--agent.model.per_instance_cost_limit={COST_LIM}",
            f"--env.repo.github_url={url}",
            f"--env.repo.base_commit={full_commit_sha}",
            f"--env.deployment.image={image_name}",
            f"--output_dir={str(image_output_dir)}",
        ]
        
        if api_key is not None:
            args.append(f"--agent.model.api_key={api_key}")
            

        # Execute sweagent
        config = BasicCLI(RunSingleConfig).get_config(args[1:])
        # config.problem_statement = TextProblemStatement(text=PROBLEM_STATEMENT)
        run_single = RunSingle.from_config(config)
        run_single.agent.model.config.completion_kwargs["azure_ad_token_provider"] = AZURE_AD_TOKEN_PROVIDER
        run_single.run()

        logger.info(f"Completed processing for image: {image_name}")

        traj_files = list(image_output_dir.rglob("*.traj"))
        if len(traj_files) == 0:
            logger.info(f"Could not find trajectory file for image {image_name} with {seed}")
            return (None, False, None)

        traj_data = json.load(open(traj_files[0], "r"))
        if not ("info" in traj_data and "exit_status" in traj_data["info"]):
            logger.info(f"Malformed trajectory file for image {image_name} with seed {seed}")
            return (None, False, None)

        if traj_data["info"]["exit_status"] == "submitted (exit_cost)":
            logger.info(f"Processing completed early due to cost limit being reached for image {image_name} with seed {seed}")
            return (None, False, None)
        
        # try to extract result from last action
        




        return (instance_data, True, None)
        
    
        
    except Exception as e:
        traceback.print_exc()
        error_msg = f"Error processing {image_name}: {str(e)}"
        logger.error(error_msg)
        return (None, False, error_msg)





def main(
    images: str | Path,
    model_name: str,
    output_file: str | Path,
    run_id: str,
    sweagent_logdir: str | Path,
    seed_per_image: int = 1,
    api_key: str | None = None,
    max_workers: int = 1,
    max_tries: int = 1,
    num_jobs: int | None = None,
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
    jobs_specs = [(i, s) for i in image_names for s in range(seed_per_image)]
    
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
                executor.submit(process_single_job, jspec, sweagent_logdir, model_name, run_id, api_key): (jspec, attempt)
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
    fire.Fire(main)