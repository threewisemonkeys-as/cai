from pathlib import Path
import logging
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
import tempfile
from sweagent.run.run import main as sweagent_main

CUR_DIR = Path(__file__).parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOCKER_ORG = "jyangballin"
TAG = "latest"
COST_LIM = 5.0
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

MODE_TO_CONFIG_MAP = {
    "bugfix": None,
    "buggen": CUR_DIR / "buggen.yaml",
}

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
    mode: str,
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
        
        # Create output directory for this image
        image_output_dir = output_dir / image_name
        image_output_dir.mkdir(exist_ok=True, parents=True)

        pre_existing_patches = list(image_output_dir.rglob("*.patch"))
        if len(pre_existing_patches) > 0:
            logger.info(f"Pre-existing patch found for image {image_name}, skipping ....")
            return (image_name, True, None)
        
        _, arch, repo_name, short_commit_sha = image_name.split('.')
        
        repo = repo_name.replace('__', '/')
        full_commit_sha = get_full_commit_id(repo, short_commit_sha)
        if full_commit_sha is None:
            error_msg = f"Could not get full commit sha for {image_name}"
            logger.warning(error_msg)
            return (image_name, False, error_msg)
            
        
        # Build sweagent arguments
        url = f"https://github.com/{repo}"
        config_file = MODE_TO_CONFIG_MAP[mode]
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

        fp = None
        if mode in ["buggix"]:
            fp = tempfile.NamedTemporaryFile(delete_on_close=False, mode="w") 
            fp.write()
            fp.close()
            args.append(f"--problem_statement.path={str(fp.name)}")
            
        if api_key is not None:
            args.append(f"--agent.model.api_key={api_key}")
            
        # Execute sweagent
        sweagent_main(args)
        
        if fp is not None:
            fp.delete()

        logger.info(f"Successfully completed processing for image: {image_name}")
        return (image_name, True, None)
        
    except Exception as e:
        error_msg = f"Error processing {image_name}: {str(e)}"
        logger.error(error_msg)
        return (image_name, False, error_msg)


def main(
    mode: str,
    image_names_path: str | Path,
    model_name: str,
    output_dir: str | Path,
    api_key: str | None,
    num_images: int | None = None,
    max_workers: int = 4,
):
    """
    Process Docker images in parallel.
    
    Args:
        image_names_path: Path to file containing image names
        model_name: Model name for the agent
        output_dir: Directory to store outputs
        api_key: API key for the model
        num_images: Limit number of images to process
        max_workers: Maximum number of parallel workers
    """
    # Load and filter image names
    images_names = Path(image_names_path).read_text().splitlines()
    if num_images is not None:
        images_names = images_names[:num_images]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Processing {len(images_names)} images with {max_workers} workers")
    
    # Track results
    successful_processes = 0
    failed_processes = 0
    
    # Process images in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(process_single_image, mode, image_name, output_dir, model_name, api_key): image_name
            for image_name in images_names
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_image):
            image_name, success, error_msg = future.result()
            
            if success:
                successful_processes += 1
                logger.info(f"✓ Completed {image_name} ({successful_processes}/{len(images_names)})")
            else:
                failed_processes += 1
                logger.error(f"✗ Failed {image_name}: {error_msg}")
    
    # Summary
    logger.info(f"Processing complete! Success: {successful_processes}, Failed: {failed_processes}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Process Docker images in parallel using SweAgent")
    parser.add_argument("--mode", type=str, required=True, choices=["bugfix", "buggen"])
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
        mode=args.mode,
        image_names_path=args.image_names,
        output_dir=args.output_dir,
        model_name=args.model_name,
        api_key=args.api_key,
        num_images=args.num_images,
        max_workers=args.max_workers,
    )
