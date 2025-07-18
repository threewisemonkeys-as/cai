from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random

from datasets import load_dataset

from sweagent.environment.swe_env import SWEEnv
from sweagent.environment.repo import GithubRepoConfig
from swerex.deployment.docker import DockerDeployment

from athils import JsonLinesFile


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# TRACER_INSTALL_CMD = "pip install git+https://github.com/pclucas14/execution-tracing.git"
TRACER_INSTALL_CMD = "pip install git+https://github.com/threewisemonkeys-as/execution-tracing.git@atharv/no_args_logging"


def get_test_traces_for_bug(
    bug: dict,
) -> tuple[str, str]:
    instance_id = bug['instance_id']
    logger.info(f"Starting processing for instance: {instance_id}")

    try:

        image_name = f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}"
        repo_url = f"https://github.com/{bug['repo']}"
        repo_config = GithubRepoConfig(github_url=repo_url, base_commit=bug['base_commit'])
        env = SWEEnv(
            deployment=DockerDeployment(
                image=image_name,
                startup_timeout=600
            ),
            repo=repo_config,
            post_startup_commands=[TRACER_INSTALL_CMD],
        )
        env.start()

        tests = bug['FAIL_TO_PASS']
        if isinstance(tests, str):
            tests = eval(tests)

        cwd = env.communicate("pwd").strip()

        results = []
        for test in tests:
            try:
                trace_output = env.communicate(f"cd /testbed && git checkout {bug['base_commit']} && trace_pytest --no-arg-values --no-external-calls '{test}'", timeout=300)
            except Exception as e:
                logger.warning(f"Ran into error while tracing test: {test} -\n{e}")
                continue

            try:
                trace = json.loads(env.read_file(f"/testbed/pytest_trace_output.json"))
            except Exception as e:
                logger.warning(f"Ran into error while reading trace for test: {test} -\n{e}")
                continue
            breakpoint()

            if "trace_data" not in trace:
                continue

            results.append((test, trace))

        env.close()
        return (results, True, None)
    
    except Exception as e:
        return ([], False, e)



    

def main(
    dataset: str,
    output_path: str | Path,
    max_workers: int = 1,
    split: str = "test",
):

    data = load_dataset(dataset)[split]
    data = [d for d in data if d['instance_id'] in ['astropy__astropy-13453']]
    
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.parent.mkdir(exist_ok=True, parents=True)
        output_path.touch()
     
    logger.info(f"Processing {len(data)} bugs with {max_workers} workers")
    
    # Track results and retry queue
    successful_processes = 0
    failed_processes = 0
    

    current_batch = [(i, 1) for i in data]
    logger.info(f"Processing batch of {len(current_batch)} bugs...")
    
    # Process current batch in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks for current batch
        future_to_job_info = {
            executor.submit(get_test_traces_for_bug, instance): (instance, attempt)
            for instance, attempt in current_batch
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_job_info):
            instance, attempt = future_to_job_info[future]
            test_traces, success, error_msg = future.result()
            
            if success:
                JsonLinesFile.add_to(output_path, (instance, test_traces))
                successful_processes += 1
                logger.info(f"✓ Completed {instance['instance_id']} on attempt {attempt} ({successful_processes}/{len(data)})")
            else:
                failed_processes += 1
                logger.error(f"✗ Failed {instance['instance_id']}: {error_msg}")

    # Summary
    total_processed = successful_processes + failed_processes
    logger.info(f"Processing complete! Success: {successful_processes}, Failed: {failed_processes}, Total: {total_processed}")
    
    return {
        'successful': successful_processes,
        'failed': failed_processes,
        'total': len(data)
    }

if __name__ == '__main__':
    import fire
    fire.Fire(main)