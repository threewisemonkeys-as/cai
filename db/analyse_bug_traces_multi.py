from typing import Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging

from utils import snip_trace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

avg = lambda d: sum(d) / len(d) if len(d) > 0 else None
smin = lambda d: min(d) if len(d) > 0 else None

def process_single_trace(
    trace: tuple[str, dict],
) -> dict[str, Any]:
    logger.debug("Processing single trace for %s", trace[0])
    
    test_name, trace_dict = trace
    trace_data = trace_dict['trace_data']
    trace_len = len(trace_data)

    compute_unique_calls = lambda es: len(set([(e['location'], e['name']) for e in es]))
    compute_max_depth = lambda es: max(e['depth'] for e in es)

    unique_calls = compute_unique_calls(trace_data)
    max_depth = compute_max_depth(trace_data)
    
    try:
        snipped_trace = snip_trace(trace_data, test_name)
        trace_snipped = True
        snipped_unique_calls = compute_unique_calls(snipped_trace)
        snipped_max_depth = compute_max_depth(snipped_trace)
    except Exception as e:
        logger.warning("Snip trace failed for %s: %s", test_name, e)
        trace_snipped = False
        snipped_unique_calls = None
        snipped_max_depth = None

    return {
        "length": trace_len,
        "unique_calls": unique_calls,
        "max_depth": max_depth,
        "trace_snipped": trace_snipped,
        "snipped_unique_calls": snipped_unique_calls,
        "snipped_max_depth": snipped_max_depth,
    }


def process_single_instance_traces(
    instance_data: tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]
) -> dict[str, Any]:
    instance, test_traces = instance_data
    logger.debug("Processing instance %s with %d traces", instance.get("id", "<unknown>"), len(test_traces))
    results = [process_single_trace(t) for t in test_traces]

    def _vals(field: str):
        return [r[field] for r in results if r[field] is not None]

    summary = {
        'instance_id': instance['instance_id'],
        "length": smin(_vals("length")),
        "unique_calls": smin(_vals("unique_calls")),
        "max_depth": smin(_vals("max_depth")),
        "trace_snipped": avg(_vals("trace_snipped")),
        "snipped_unique_calls": smin(_vals("snipped_unique_calls")),
        "snipped_max_depth": smin(_vals("snipped_max_depth")),
    }
    logger.debug("Finished processing instance summary: %s", summary.get("id", "<unknown>"))
    return summary

def _read_nth_line(fp: Path, n: int) -> str:
    """Return *line n* (0-based) from *fp*.

    Each worker process re-opens the file - this keeps the implementation
    simple and avoids serialising large strings through shared memory.
    """
    with fp.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == n:
                return line
    raise IndexError(f"File {fp} has no line {n}")


def process_line(file_path: str | Path, line_num: int) -> dict[str, Any]:
    raw = _read_nth_line(Path(file_path), line_num).strip()
    if not raw:
        return {}

    data = json.loads(raw)
    return process_single_instance_traces(data)



def analyse(
    traces: str | Path,
    output_path: str | Path,
    max_workers: int = 1,
):
    """Analyse *traces* JSONL file and write summaries to *output_path*.

    Parameters
    ----------
    traces
        Path to the input JSONL file.
    output_path
        Where to write the summary JSONL; directories are created as needed.
    max_workers
        Concurrency for ``ProcessPoolExecutor``.  Defaults to 1 (serial).
    """

    traces = Path(traces)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count lines quickly (stream once) – this lets us partition work.
    with traces.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Submit each line as a separate task.
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for line_no in range(total_lines):
            fut = executor.submit(process_line, traces, line_no)
            futures[fut] = line_no

        with output_path.open("w", encoding="utf‑8") as out:
            for fut in as_completed(futures):
                result = fut.result()
                if result:  # skip blank lines
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import fire

    fire.Fire(analyse)
