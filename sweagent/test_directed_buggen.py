from pathlib import Path
import logging
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
import json
from textwrap import shorten

from sweagent.run.common import BasicCLI
from sweagent.run.run_single import RunSingle, RunSingleConfig
from sweagent.agent.problem_statement import TextProblemStatement

from athils import JsonLinesFile

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from dotenv import load_dotenv
load_dotenv()

CUR_DIR = Path(__file__).parent
BUGGEN_CONFIG_FILE = CUR_DIR / "test_directed_buggen.yaml"
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



def fmt_args(kwargs: dict, max_len: int = 100) -> str:
    """Render kwargs dict compactly (hides `self`)."""
    cleaned = {k: v for k, v in kwargs.items() if k != "self"}
    if not cleaned:
        return ""
    return "  args={" + shorten(repr(cleaned)[1:-1], max_len) + "}"


def fmt_event(ev: dict) -> str:
    """Return one formatted line for a trace event."""
    depth = ev.get("depth", 0)
    indent = " " * (depth * 2)

    call = ev["name"]
    ctype = ev["call_type"]
    loc = ev["location"]
    parent = ev.get("parent_call")

    line = f"{indent} {call} @ {loc}"
    line += fmt_args(ev.get("arguments", {}))
    if parent and parent != call:
        line += f"  ctx: {shorten(parent, 40)}"
    return line



INCLUDE_TYPES = ["function_call"]

def make_trace_desc(trace) -> str:
    events = trace['trace_data']

    if INCLUDE_TYPES is not None:
        events = [
            ev for ev in events
            if ev.get("call_type") in INCLUDE_TYPES
        ]

    trace_str = "\n".join(fmt_event(ev) for ev in events)
    return trace_str


def process_single_job(
    jspec, 
    output_dir: Path, 
    model_name: str, 
    api_key: Optional[str]
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single Docker image.
    
    Returns:
        Tuple of (jspec, success, error_message)
    """
    image_name, trace_data = jspec
    test, trace = trace_data


    trace_desc = make_trace_desc(trace)
    test_trace_desc = f"Runtime execution trace with for the test: {test} -\nFormat: <function name> @ <location>  args={{<function arguments>}}  ctx: <calling context>\n{trace_desc}"

    try:
        logger.info(f"Starting processing for image {image_name} with trace of test {test}")
        
        # Create output directory for this image
        image_output_dir = output_dir / image_name
        image_output_dir.mkdir(exist_ok=True, parents=True)

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

        _image_name, _tag = image_name.split(":")
        _, arch, repo_name, short_commit_sha = _image_name.split('.')
        
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

        docker_image_name = image_name.replace('__', '_1776_')
            
        args += [
            f"--agent.model.name={model_name}",
            f"--agent.model.per_instance_cost_limit={COST_LIM}",
            f"--env.repo.github_url={url}",
            f"--env.repo.base_commit={full_commit_sha}",
            f"--env.deployment.image={docker_image_name}",
            f"--output_dir={str(image_output_dir)}",
        ]
        
        if api_key is not None:
            args.append(f"--agent.model.api_key={api_key}")
            

        # Execute sweagent
        config = BasicCLI(RunSingleConfig).get_config(args[1:])
        config.problem_statement = TextProblemStatement(text=test_trace_desc)
        run_single = RunSingle.from_config(config)
        run_single.agent.model.config.completion_kwargs["azure_ad_token_provider"] = AZURE_AD_TOKEN_PROVIDER
        run_single.run()

        logger.info(f"Completed processing for image: {image_name}")

        current_patches = list(image_output_dir.rglob("*.patch"))
        for patch_file in current_patches:
            traj_files = list(patch_file.parent.glob("*.traj"))
            if len(traj_files) > 0:
                traj_data = json.load(open(traj_files[0], "r"))
                if "info" in traj_data and "exit_status" in traj_data["info"]:
                    if traj_data["info"]["exit_status"] == "submitted (exit_cost)":
                        logger.info(f"Processing completed early due to cost limit being reached for image: {image_name}")
                        continue
                    else:
                        return (image_name, True, None)
            
            logger.info(f"Malformed trajectory data for image: {image_name}")
            continue
            
        return (image_name, False, None)
        
    except Exception as e:
        error_msg = f"Error processing {image_name}: {str(e)}"
        logger.error(error_msg)
        return (image_name, False, error_msg)


def main(
    traces: str | Path,
    model_name: str,
    output_dir: str | Path,
    api_key: str | None = None,
    max_workers: int = 4,
    max_tries: int = 5,
):
    """
    Process Docker images in parallel with retry logic.
    
    Args:
        traces: Path to file containing traces
        model_name: Model name for the agent
        output_dir: Directory to store outputs
        api_key: API key for the model
        num_images: Limit number of images to process
        max_workers: Maximum number of parallel workers
        max_tries: Maximum number of retry attempts per image
    """

    traces_data = JsonLinesFile.read_from(traces)
    jobs_specs = []
    for image_name, image_traces in traces_data:
        for trace in image_traces:
            jobs_specs.append((image_name, trace))

    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
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
                executor.submit(process_single_job, jspec, output_dir, model_name, api_key): (jspec, attempt)
                for jspec, attempt in current_batch
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_jspec):
                jspec, attempt = future_to_jspec[future]
                _, success, error_msg = future.result()
                
                if success:
                    successful_processes += 1
                    logger.info(f"✓ Completed {jspec[0]} on attempt {attempt} ({successful_processes}/{len(jobs_specs)})")
                else:
                    if attempt < max_tries:
                        # Add to retry queue for next batch
                        retry_queue.append((jspec, attempt + 1))
                        logger.warning(f"⚠ Failed {jspec[0]} on attempt {attempt}/{max_tries}: {error_msg}. Will retry.")
                    else:
                        # Max tries reached, mark as permanently failed
                        failed_processes += 1
                        logger.error(f"✗ Failed {jspec[0]} after {max_tries} attempts: {error_msg}. Giving up.")
        
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