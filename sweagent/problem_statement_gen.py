from pathlib import Path
import tempfile
import logging
import os
import json
import yaml
import threading


from datasets import load_dataset


from swesmith.issue_gen.generate import IssueGen


from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from dotenv import load_dotenv
load_dotenv()

CUR_DIR = Path(__file__).parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
AZURE_AD_TOKEN_PROVIDER = get_bearer_token_provider(DefaultAzureCredential(), os.getenv("AZURE_API_SCOPE", None))



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



def problem_statement_gen(
    instances: list[dict],
    model_name: str,
    run_id: str,
) -> list[dict]:


    issue_generator = CustomIssueGen(
        model=model_name,
        use_existing=True,
        n_workers=1,
        experiment_id=run_id,
    )
    new_instances = []
    for instance in instances:
        del instance["problem_statement"]
        with tempfile.NamedTemporaryFile(delete_on_close=False, mode="w+") as fp:
            issue_generator.generate_issue(instance, 0, fp)
            fp.flush()
            fp.seek(0)
            new_instance = json.load(fp)

        new_instance |= instance
        new_instances.append(new_instance)
    return new_instances



def main(
    input: Path | str,
    output: Path | str,
    model: str,
    run_id: str,
):
    if Path(output).exists():
        raise RuntimeError(f"File aready exists at: {output}")
    
    instances_data = json.load(open(input, "r"))
    results = problem_statement_gen(
        instances_data,
        model_name=model,
        run_id=run_id,
    )
    json.dump(results, open(output, "w"), indent=2)
    

if __name__ == '__main__':
    import fire
    fire.Fire(main)