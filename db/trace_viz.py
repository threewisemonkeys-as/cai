import json
import pathlib
from textwrap import shorten

from utils import snip_trace

INDENT = 2  # spaces per depth‑level

###############################################################################
# Utility helpers
###############################################################################

def load_trace(src: str | pathlib.Path) -> list[dict]:
    """Load JSON‑encoded list either from a file path or a raw string."""
    if isinstance(src, pathlib.Path) or pathlib.Path(src).is_file():
        return json.loads(pathlib.Path(src).read_text())
    return json.loads(src)


def fmt_args(kwargs: dict, max_len: int = 60) -> str:
    """Render kwargs dict compactly (hides `self`)."""
    cleaned = {k: v for k, v in kwargs.items() if k != "self"}
    if not cleaned:
        return ""
    return "  args=" + shorten(repr(cleaned), max_len)


def fmt_event(ev: dict) -> str:
    """Return one formatted line for a trace event."""
    depth = ev.get("depth", 0)
    indent = " " * (depth * INDENT)

    call = ev["name"]
    ctype = ev["call_type"]
    loc = ev["location"]
    parent = ev.get("parent_call")

    line = f"{indent} {call}  ({ctype})  @ {loc}"
    if parent and parent != call:
        line += f" ← {shorten(parent, 40)}"
    line += fmt_args(ev.get("arguments", {}))
    return line




###############################################################################
# New public rendering function
###############################################################################

def render_trace(events: list[dict], entrypoint: str | None = None, include_types: list[str] = list()) -> str:
    """Return formatted text for the given *events*.

    Parameters
    ----------
    events : list[dict]
        The raw trace events (list of dictionaries).
    entrypoint: str | None
        If provided, the name of the of entrypoint from which to render the trace.
        Should be provided in format file_path::function_name
    include_types : list[str]
        If provided, only events whose ``call_type`` is in this iterable
        are rendered.  Comparison is exact (case-sensitive). 
    """
    if entrypoint is not None:
        events = snip_trace(events, entrypoint)

    if include_types != []:
        include_set = set(include_types)
        events = [ev for ev in events if ev.get("call_type") in include_set]
    return "\n".join(fmt_event(ev) for ev in events)


def main(
    events_path: pathlib.Path | str,
    entrypoint: str | None = None,
    include_types: list[str] = list(),
):
    events = load_trace(events_path)
    text = render_trace(events, entrypoint=entrypoint, include_types=include_types)
    print(text)


if __name__ == "__main__":
    import fire
    fire.Fire(main)