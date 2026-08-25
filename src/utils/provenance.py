"""
utils/provenance.py
===================
Which code produced a stored result.

The experiment services run the source baked into the image and the development
services mount the working tree over it, so the same command name can execute
two different programs and a results file on its own does not say which.  This
module digests the source at build time and again at run time.  Equal digests
mean the run executed the image's own code; different ones mean something was
mounted over it.  A run that writes its provenance beside its results can be
traced to code without inference.

Build time, inside the image:
    python -m src.utils.provenance --write --git-rev "$GIT_REV"

Before a run whose output is committed, with the tree mounted:
    docker compose run --rm --entrypoint python shell -m src.utils.provenance --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

# The directories whose contents decide what a run computes.  Tests and the
# manuscript do not, so a change to either leaves a stored result valid.
SOURCE_ROOTS = ("src", "experiments")

# Outside the working directory on purpose: the development services mount the
# tree over /workspace, which would hide a stamp written inside it and leave
# the staleness check unable to see the image it is meant to compare against.
BUILD_INFO_PATH = "/opt/n2dbn/BUILD_INFO.json"

# Packages whose version can move a number.  Recorded per run, since the image
# resolves them at build time and an unpinned bound can drift between builds.
TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "networkx",
    "networkx-temporal",
    "matplotlib",
    "pyAgrum",
    "pgmpy",
    "teneto",
)


def _python_files(root: str, roots: Sequence[str]) -> Iterator[str]:
    """Every source file under the given roots, as paths relative to root."""
    for name in roots:
        base = Path(root) / name
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield str(path.relative_to(root))


def source_digest(root: str = ".", roots: Sequence[str] = SOURCE_ROOTS) -> str:
    """A digest of the source that decides what a run computes."""
    digest = hashlib.sha256()
    for rel in sorted(_python_files(root, roots)):
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update((Path(root) / rel).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _versions() -> Dict[str, str]:
    """Installed versions of the packages a result depends on."""
    from importlib.metadata import PackageNotFoundError, version

    out: Dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "absent"
    return out


def write_build_info(
    path: str = BUILD_INFO_PATH,
    root: str = ".",
    git_rev: str = "unknown",
) -> Dict[str, str]:
    """Record what the image was built from, at build time."""
    info = {
        "source_digest": source_digest(root),
        "git_rev": git_rev,
        "built_at": _now(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return info


def read_build_info(path: str = BUILD_INFO_PATH) -> Dict[str, str]:
    """What the image was built from, or an empty record if it was not stamped."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def provenance(
    root: str = ".",
    build_info_path: str = BUILD_INFO_PATH,
) -> Dict[str, object]:
    """What produced this run: the image's source, the live source, and both."""
    baked = read_build_info(build_info_path)
    live = source_digest(root)
    return {
        "image_source_digest": baked.get("source_digest", "unknown"),
        "live_source_digest": live,
        "runs_image_source": baked.get("source_digest") == live,
        "git_rev": baked.get("git_rev", "unknown"),
        "image_built_at": baked.get("built_at", "unknown"),
        "run_at": _now(),
        "python": platform.python_version(),
        "packages": _versions(),
    }


def write_provenance(
    output_dir: str,
    root: str = ".",
    build_info_path: str = BUILD_INFO_PATH,
) -> Dict[str, object]:
    """Write the provenance of this run beside its results."""
    record = provenance(root, build_info_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / "provenance.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def check(root: str = ".", build_info_path: str = BUILD_INFO_PATH) -> int:
    """Report whether the image was built from the source visible here."""
    record = provenance(root, build_info_path)
    if record["image_source_digest"] == "unknown":
        print("image carries no build stamp, so it predates provenance recording")
        return 2
    if record["runs_image_source"]:
        print(f"image is current with this source ({record['git_rev']})")
        return 0
    print(
        "image is STALE against this source.\n"
        f"  image built {record['image_built_at']} from {record['git_rev']}\n"
        f"  image digest {record['image_source_digest'][:16]}\n"
        f"  live  digest {record['live_source_digest'][:16]}\n"
        "Rebuild before a run whose output is committed:\n"
        "  docker compose build tests"
    )
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="stamp the image")
    parser.add_argument("--check", action="store_true", help="is the image stale")
    parser.add_argument("--git-rev", default="unknown")
    parser.add_argument("--root", default=".")
    parser.add_argument("--build-info", default=BUILD_INFO_PATH)
    args = parser.parse_args(argv)

    if args.write:
        info = write_build_info(args.build_info, args.root, args.git_rev)
        print(f"stamped {args.build_info}: {info['source_digest'][:16]}")
        return 0
    if args.check:
        return check(args.root, args.build_info)
    print(json.dumps(provenance(args.root, args.build_info), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
