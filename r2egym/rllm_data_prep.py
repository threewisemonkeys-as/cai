from pathlib import Path
import json

from rllm.data.dataset import DatasetRegistry


def prepare(
    data: str | Path,
    name: str,
    split: str = "train",
) -> None:

    dataset = json.load(open(data, 'r'))
    DatasetRegistry.register_dataset(name, dataset, split)


if __name__ == '__main__':
    import fire
    fire.Fire(prepare)
