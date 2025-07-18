from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from sweagent.environment.swe_env import SWEEnv
from sweagent.environment.repo import GithubRepoConfig
from swerex.deployment.docker import DockerDeployment

from athils import JsonLinesFile




logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def process_image(
    image_name: str, 
) -> tuple[str, str]:
    logger.info(f"Starting processing for image: {image_name}")
    env = None

    try:
        repo_url = f"https://github.com/swesmith/{image_name.split('swesmith.x86_64.')[1].split(':')[0]}"
        repo_config = GithubRepoConfig(github_url=repo_url)
        env = SWEEnv(
            deployment=DockerDeployment(image=image_name, startup_timeout=600),
            repo=repo_config,
            post_startup_commands=[],
        )
        env.start()

        # get total size of all python files in the repository
        repo_size = env.communicate("find . -name '*.py' -exec du -b {} + | awk '{sum += $1} END {print sum}'")

        ret = (repo_size, True, None)
    
    except Exception as e:
        ret = ([], False, e)
    
    finally:
        if env is not None:
            try:
                env.stop()
            except Exception as e:
                logger.error(f"Error stopping environment for {image_name}: {e}")

    return ret



    

def main(
    images: str | Path,
    output_path: str | Path,
    num_images: int | None = None,
    max_workers: int = 1,
):

    # Load and filter image names
    images_names = Path(images).read_text().splitlines()
    if num_images is not None:
        images_names = images_names[:num_images]
    
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.parent.mkdir(exist_ok=True, parents=True)
        output_path.touch()
     
    logger.info(f"Processing {len(images_names)} images with {max_workers} workers")
    
    # Track results and retry queue
    successful_processes = 0
    failed_processes = 0
    
    # Initialize all images with attempt count 1
    current_batch = [(f"{image_name}", 1) for image_name in images_names]
    
    logger.info(f"Processing batch of {len(current_batch)} images...")
    
    # Process current batch in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks for current batch
        future_to_image_info = {
            executor.submit(process_image, image_name): (image_name, attempt)
            for image_name, attempt in current_batch
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_image_info):
            image_name, attempt = future_to_image_info[future]
            result, success, error_msg = future.result()
            
            if success:
                JsonLinesFile.add_to(output_path, (image_name, result))
                successful_processes += 1
                logger.info(f"✓ Completed {image_name} on attempt {attempt} ({successful_processes}/{len(images_names)})")
            else:
                failed_processes += 1
                logger.error(f"✗ Failed {image_name}: {error_msg}")

    # Summary
    total_processed = successful_processes + failed_processes
    logger.info(f"Processing complete! Success: {successful_processes}, Failed: {failed_processes}, Total: {total_processed}")
    
    return {
        'successful': successful_processes,
        'failed': failed_processes,
        'total': len(images_names)
    }

if __name__ == '__main__':
    import fire
    fire.Fire(main)