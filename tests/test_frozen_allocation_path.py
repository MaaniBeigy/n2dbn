"""
tests/test_frozen_allocation_path.py
====================================
The allocation formulation is frozen, and this is what enforces it.

The modules listed in `FROZEN` produced every stored result and keep doing so.
They are not developed further and nothing new is built on them, so the set of
modules under `src/` that import them is closed at the value it had when the
freeze was taken.  A rename alone would not enforce that; this test does.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

# The modules of the allocation formulation: the top-down generator, the
# families and grids that configure it, the Dirichlet-Multinomial split it
# samples from, and the two screens that tune it.
FROZEN: Set[str] = {
    "src/model/allocation.py",
    "src/synthetic/calibration.py",
    "src/synthetic/families.py",
    "src/synthetic/generator.py",
    "src/synthetic/latent_screen.py",
    "src/synthetic/scenarios.py",
}

# Every module under `src/` that imported a frozen module when the freeze was
# taken.  Adding an entry is a new dependency on a path that is not developed
# further, so the list is closed rather than a starting point.  Frozen modules
# import each other freely and are not listed.
PERMITTED_IMPORTERS: Set[str] = {
    "src/data/event_log.py",
    "src/evaluation/protocol.py",
    "src/utils/temporal_viz.py",
}


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _source_modules() -> Iterator[Path]:
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def _package_of(path: Path) -> List[str]:
    """The dotted package parts a relative import inside `path` resolves against."""
    parts = list(path.relative_to(REPO).parts)
    return parts[:-1] if path.name != "__init__.py" else parts


def _resolve(node: ast.AST, path: Path) -> Optional[str]:
    """The repo-relative module file an import statement names, when it is one."""
    if isinstance(node, ast.ImportFrom):
        if node.level:
            base = _package_of(path)
            # `from ..x import y` at level n drops n-1 packages, not n, because
            # level 1 is the module's own package rather than its parent.
            keep = len(base) - (node.level - 1)
            if keep < 0:
                return None
            target = base[:keep] + (node.module.split(".") if node.module else [])
        else:
            target = node.module.split(".") if node.module else []
    elif isinstance(node, ast.Import):
        return None
    else:
        return None
    if not target:
        return None
    return "/".join(target) + ".py"


def _imports_of(path: Path) -> Iterator[Tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            resolved = _resolve(node, path)
            if resolved is not None:
                yield resolved, node.lineno


def test_the_frozen_modules_all_exist() -> None:
    missing = sorted(name for name in FROZEN if not (REPO / name).is_file())
    assert missing == [], f"frozen modules named but absent: {missing}"


def test_the_permitted_importers_all_exist() -> None:
    missing = sorted(
        name for name in PERMITTED_IMPORTERS if not (REPO / name).is_file()
    )
    assert missing == [], f"permitted importers named but absent: {missing}"


def test_no_new_source_module_imports_the_allocation_formulation() -> None:
    offenders: Dict[str, List[str]] = {}
    for path in _source_modules():
        rel = _rel(path)
        if rel in FROZEN or rel in PERMITTED_IMPORTERS:
            continue
        for target, lineno in _imports_of(path):
            if target in FROZEN:
                offenders.setdefault(rel, []).append(f"{target} at line {lineno}")
    assert offenders == {}, (
        "the allocation formulation is frozen and these modules import it: "
        f"{offenders}"
    )


def test_every_permitted_importer_still_imports_something_frozen() -> None:
    """A permitted importer that stopped importing leaves the list overstated."""
    stale = []
    for rel in sorted(PERMITTED_IMPORTERS):
        targets = {target for target, _ in _imports_of(REPO / rel)}
        if not targets & FROZEN:
            stale.append(rel)
    assert stale == [], f"permitted but no longer importing a frozen module: {stale}"
