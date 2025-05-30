from pathlib import Path
import logging
import requests
import os

from sweagent.run.run import main as sweagent_main

CUR_DIR = Path(__file__).parent
BUGGEN_CONFIG_FILE = CUR_DIR / "buggen.yaml"


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOCKER_ORG = "jyangballin"
TAG = "latest"
COST_LIM = 5.0
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def get_full_commit_id(repo: str, short_commit_id: str) -> str | None:
    response = requests.get(
        url=f"https://api.github.com/repos/{repo}/commits/{short_commit_id}",
        headers={"Authorization": GITHUB_TOKEN}
    )
    
    if response.status_code == 200:
        commit_data = response.json()
        return commit_data['sha']
    else:
        # raise Exception(f"Failed to fetch commit: {response.status_code}")
        return None
    

def main(
    image_names_path: str | Path,
    model_name: str,
    output_dir: str | Path,
    api_key: str | None,
    num_images: int | None = None,
):
    
    images_names = Path(image_names_path).read_text().splitlines()
    if num_images is not None:
        images_names = images_names[:num_images]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False, parents=True)

    for image_name in images_names:
        _, arch, repo_name, short_commit_sha = image_name.split('.')
        repo = repo_name.replace('__', '/')
        full_commit_sha = get_full_commit_id(repo, short_commit_sha)
     
        if full_commit_sha is None:
            logger.warning(f"Could not get full commit sha for {image_name}")
            continue

        image_output_dir = output_dir / image_name

        url = f"https://github.com/{repo}"

        config_file = BUGGEN_CONFIG_FILE

        args = ["run"]

        if config_file is not None:
            args.extend([f"--config",  str(config_file)])

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

        sweagent_main(args)



if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-i", "--image_names", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, default="azure/o3_2025-04-16")
    parser.add_argument("-k", "--api_key", type=str, default=None)
    parser.add_argument("-n", "--num_images", help="Number of images to run for", type=int, default=None)
    args = parser.parse_args()

    main(
        image_names_path=args.image_names,
        output_dir=args.output_dir,
        model_name=args.model_name,
        api_key=args.api_key,
        num_images=args.num_images,
    )
