from pathlib import Path
import logging
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
import json

from sweagent.run.common import BasicCLI
from sweagent.run.run_single import RunSingle, RunSingleConfig

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from dotenv import load_dotenv
load_dotenv()

CUR_DIR = Path(__file__).parent
BUGGEN_CONFIG_FILE = CUR_DIR / "buggen.yaml"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOCKER_ORG = "jyangballin"
TAG = "latest"
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


def process_single_image(
    image_name: str, 
    output_dir: Path, 
    model_name: str, 
    api_key: Optional[str]
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single Docker image.

    Returns:
        Tuple of (image_name, success, error_message)
    """
    try:
        logger.info(f"Starting processing for image: {image_name}")
        
        image_output_dir = output_dir / image_name
        image_output_dir.mkdir(exist_ok=True, parents=True)

        # Step 1: Check for a pre-existing SUCCESSFUL patch to avoid re-work.
        pre_existing_patches = list(image_output_dir.rglob("*.patch"))
        if len(pre_existing_patches) > 0:
            for patch_file in pre_existing_patches:
                traj_files = list(patch_file.parent.glob("*.traj"))
                if traj_files:
                    traj_data = json.load(open(traj_files[0], "r"))
                    # Check for a valid trajectory file indicating success
                    if "info" in traj_data and "exit_status" in traj_data["info"] and traj_data["info"]["exit_status"] != "submitted (exit_cost)":
                        logger.info(f"Image {image_name} was already successfully processed. Skipping.")
                        return (image_name, True, None)
            
            logger.info(f"Found patches from previous failed attempts for {image_name}. Starting new attempt.")
        
        # Step 2: If we're here, no successful run exists. Execute a new attempt.
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
        run_single = RunSingle.from_config(config)
        run_single.agent.model.config.completion_kwargs["azure_ad_token_provider"] = AZURE_AD_TOKEN_PROVIDER
        run_single.run()

        logger.info(f"Agent finished running for image: {image_name}. Verifying results.")

        # Step 3: Verify the results of the new attempt.
        current_patches = list(image_output_dir.rglob("*.patch"))
        if not current_patches:
            logger.warning(f"Run for {image_name} did not produce a patch.")
            return (image_name, False, "Run did not produce a patch.")

        for patch_file in current_patches:
            traj_files = list(patch_file.parent.glob("*.traj"))
            if len(traj_files) > 0:
                traj_data = json.load(open(traj_files[0], "r"))
                if "info" in traj_data and "exit_status" in traj_data["info"]:
                    if traj_data["info"]["exit_status"] == "submitted (exit_cost)":
                        logger.info(f"Processing for {image_name} stopped due to cost limit.")
                        continue  # This patch is a failure, check if other patches were generated
                    else:
                        logger.info(f"Successfully generated patch for {image_name}.")
                        return (image_name, True, None)  # Found a successful patch
            else:
                logger.warning(f"Found patch for {image_name} but its trajectory file is malformed.")
                continue # Malformed, check other patches

        # If we finish the loop, no successful patch was found in the new attempt.
        return (image_name, False, "Attempt failed to produce a valid patch.")
    
    except Exception as e:
        error_msg = f"Error processing {image_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return (image_name, False, error_msg)


def main(
    image_names_path: str | Path,
    model_name: str,
    output_dir: str | Path,
    api_key: str | None,
    num_images: int | None = None,
    max_workers: int = 4,
    max_tries: int = 5,
):
    """
    Process Docker images in parallel with retry logic.
    
    Args:
        image_names_path: Path to file containing image names
        model_name: Model name for the agent
        output_dir: Directory to store outputs
        api_key: API key for the model
        num_images: Limit number of images to process
        max_workers: Maximum number of parallel workers
        max_tries: Maximum number of retry attempts per image
    """
    # Load and filter image names
    images_names = Path(image_names_path).read_text().splitlines()
    if num_images is not None:
        images_names = images_names[:num_images]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Processing {len(images_names)} images with {max_workers} workers, max {max_tries} tries per image")
    
    # Track results and retry queue
    successful_processes = 0
    failed_processes = 0
    retry_queue = []  # List of (image_name, attempt_count) tuples
    
    # Initialize all images with attempt count 1
    current_batch = [(image_name, 1) for image_name in images_names]
    
    while current_batch:
        logger.info(f"Processing batch of {len(current_batch)} images...")
        
        # Process current batch in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks for current batch
            future_to_image_info = {
                executor.submit(process_single_image, image_name, output_dir, model_name, api_key): (image_name, attempt)
                for image_name, attempt in current_batch
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_image_info):
                image_name, attempt = future_to_image_info[future]
                _, success, error_msg = future.result()
                
                if success:
                    successful_processes += 1
                    logger.info(f"✓ Completed {image_name} on attempt {attempt} ({successful_processes}/{len(images_names)})")
                else:
                    if attempt < max_tries:
                        # Add to retry queue for next batch
                        retry_queue.append((image_name, attempt + 1))
                        logger.warning(f"⚠ Failed {image_name} on attempt {attempt}/{max_tries}: {error_msg}. Will retry.")
                    else:
                        # Max tries reached, mark as permanently failed
                        failed_processes += 1
                        logger.error(f"✗ Failed {image_name} after {max_tries} attempts: {error_msg}. Giving up.")
        
        # Prepare next batch from retry queue
        current_batch = retry_queue.copy()
        retry_queue.clear()
    
    # Summary
    total_processed = successful_processes + failed_processes
    logger.info(f"Processing complete! Success: {successful_processes}, Failed: {failed_processes}, Total: {total_processed}")
    
    return {
        'successful': successful_processes,
        'failed': failed_processes,
        'total': len(images_names)
    }

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Process Docker images in parallel using SweAgent")
    parser.add_argument("-i", "--image_names", type=str, required=True,
                        help="Path to file containing image names")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("-m", "--model_name", type=str, default="azure/o3_2025-04-16",
                        help="Model name for the agent")
    parser.add_argument("-k", "--api_key", type=str, default=None,
                        help="API key for the model")
    parser.add_argument("-n", "--num_images", type=int, default=None,
                        help="Number of images to process (limits the input)")
    parser.add_argument("-w", "--max_workers", type=int, default=4,
                        help="Maximum number of parallel workers")
    
    args = parser.parse_args()
    
    main(
        image_names_path=args.image_names,
        output_dir=args.output_dir,
        model_name=args.model_name,
        api_key=args.api_key,
        num_images=args.num_images,
        max_workers=args.max_workers,
    )