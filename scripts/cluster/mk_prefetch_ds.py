#!/usr/bin/env python3
"""
mk_prefetch_ds.py - Generate a Kubernetes DaemonSet manifest that pre-pulls
container images listed in a text file.
Lines starting with "#" in the image list are ignored.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import sys
from typing import List

import yaml

_DNS_LABEL_RE = re.compile(r"[^a-z0-9-]")


def _dns1123(part: str) -> str:
    """Return *part* sanitised into a DNS‑1123 label fragment."""
    return _DNS_LABEL_RE.sub("-", part.lower()).strip("-")


def _make_container_name(image: str, index: int) -> str:
    """Generate a unique, DNS‑1123‑compliant initContainer name."""
    repo, _, tag = image.partition(":")

    repo_slug = _dns1123(repo.split("/")[-1]) or "img"
    tag_slug = _dns1123(tag) or "latest"

    base = f"{repo_slug}-{tag_slug}"
    if len(base) > 55:  # leave room for suffix
        base = base[:55]

    name = f"pull-{base}-{index}"

    if len(name) > 63:  # final guard – fall back to hash of full ref
        digest = hashlib.sha1(image.encode()).hexdigest()[:8]
        name = f"pull-{repo_slug[:20]}-{digest}"

    return name


def _build_daemonset(images: List[str]):
    """Return the DaemonSet manifest as a Python dict."""
    ds = {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {"name": "image-prepuller"},
        "spec": {
            "selector": {"matchLabels": {"app": "image-prepuller"}},
            "template": {
                "metadata": {"labels": {"app": "image-prepuller"}},
                "spec": {
                    "initContainers": [],
                    "containers": [
                        {
                            "name": "pause",
                            "image": "gcr.io/google_containers/pause:3.2",
                            "resources": {
                                "requests": {"cpu": "1m", "memory": "8Mi"},
                                "limits": {"cpu": "1m", "memory": "8Mi"},
                            },
                        }
                    ],
                    "restartPolicy": "Always",
                },
            },
        },
    }

    ic = ds["spec"]["template"]["spec"]["initContainers"]
    for idx, img in enumerate(images, 1):
        ic.append(
            {
                "name": _make_container_name(img, idx),
                "image": img,
                "command": ["sh", "-c", "true"],
            }
        )
    return ds


def generate(imagelist: str, outfile: str = "-") -> None:
    """Generate the DaemonSet YAML.

    Args:
        imagelist: Path to a text file containing one image[:tag] per line.
        outfile  : Destination file. Use "-", "", or "stdout" for STDOUT (default).
    """
    lines = pathlib.Path(imagelist).read_text().splitlines()
    images = [l.strip() for l in lines if l.strip() and not l.lstrip().startswith("#")]

    if not images:
        sys.exit("[mk_prefetch_ds] No images found in the provided list.")

    manifest = _build_daemonset(images)

    yaml.SafeDumper.add_representer(  # render None as empty scalar (nicer YAML)
        type(None),
        lambda dumper, _: dumper.represent_scalar("tag:yaml.org,2002:null", ""),
    )

    rendered = yaml.safe_dump(
        manifest, sort_keys=False, default_flow_style=False, width=1000
    )

    if outfile in {"-", "", "stdout"}:  # STDOUT
        print(rendered)
    else:
        pathlib.Path(outfile).write_text(rendered)
        print(f"[mk_prefetch_ds] Wrote DaemonSet to {outfile}", file=sys.stderr)



if __name__ == "__main__":
    import fire
    fire.Fire(generate)
