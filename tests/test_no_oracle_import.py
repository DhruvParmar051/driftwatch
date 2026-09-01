"""GATE - C3: detect/ must never reach oracle/. Must pass from day one.

The detectors are label-free BY DEFINITION. If any module under detect/ can
reach the `oracle` package - directly or through a chain of first-party imports
- then the monitor has had access to ground truth it does not have in
deployment, and every detection number in the study is invalid.

The check is transitive on purpose. A direct `import oracle` in detect/ is easy
to spot in review; `detect -> signals -> some_helper -> oracle` is not, and it
does exactly the same damage.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.conftest import PACKAGE_ROOT, python_files

pytestmark = pytest.mark.gate

FORBIDDEN_PACKAGE = "oracle"


def first_party_names() -> set[str]:
    """Top-level module and package names that belong to this project.

    Derived from the source tree rather than hardcoded, so adding a package
    does not silently fall outside the check.
    """
    names = set()
    for child in PACKAGE_ROOT.iterdir():
        if child.name.startswith((".", "_")) or child.name == "config":
            continue
        if child.is_dir() and (child / "__init__.py").exists():
            names.add(child.name)
        elif child.suffix == ".py":
            names.add(child.stem)
    return names


FIRST_PARTY = first_party_names()


def module_name(path: Path) -> str:
    rel = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def first_party_imports(path: Path) -> set[str]:
    """Every first-party module this file imports."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()

    def record(dotted: str) -> None:
        if dotted.split(".")[0] in FIRST_PARTY:
            found.add(dotted)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                record(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:  # relative import
                continue
            record(node.module)
            for alias in node.names:
                record(f"{node.module}.{alias.name}")
    return found


def build_import_graph() -> dict[str, set[str]]:
    return {module_name(p): first_party_imports(p) for p in python_files(PACKAGE_ROOT)}


def reaches_oracle(start: str, graph: dict[str, set[str]]) -> list[str] | None:
    """Return an import path from `start` to the oracle package, or None."""
    seen: set[str] = set()
    stack: list[tuple[str, list[str]]] = [(start, [start])]
    while stack:
        node, trail = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for target in graph.get(node, set()):
            if target == FORBIDDEN_PACKAGE or target.startswith(FORBIDDEN_PACKAGE + "."):
                return trail + [target]
            # An import may name a symbol rather than a module; resolve to the
            # nearest known module so the walk keeps going.
            for candidate in (target, target.rsplit(".", 1)[0]):
                if candidate in graph and candidate not in seen:
                    stack.append((candidate, trail + [candidate]))
    return None


def test_first_party_names_were_discovered():
    """Guard against a check that passes because it found nothing to look at."""
    assert {"detect", "oracle", "signals", "provider", "corpus"} <= FIRST_PARTY


def test_detect_package_exists():
    assert (PACKAGE_ROOT / "detect").is_dir()


def test_oracle_package_exists():
    assert (PACKAGE_ROOT / "oracle").is_dir()


def test_detect_never_reaches_oracle():
    graph = build_import_graph()
    detect_modules = [m for m in graph if m == "detect" or m.startswith("detect.")]
    assert detect_modules, "no detect/ modules found - the check would pass vacuously"

    violations = {}
    for module in detect_modules:
        trail = reaches_oracle(module, graph)
        if trail is not None:
            violations[module] = " -> ".join(trail)

    assert not violations, (
        "C3 VIOLATION - detect/ can reach oracle/:\n"
        + "\n".join(f"  {m}: {t}" for m, t in violations.items())
        + "\n\nThe detectors are label-free by definition. If a shared helper is "
        "genuinely needed, put it in a NEUTRAL module both may import."
    )


def test_signals_never_reach_oracle():
    """Signals are label-free too; S3 in particular is tempted by the oracle's normaliser."""
    graph = build_import_graph()
    for module in [m for m in graph if m == "signals" or m.startswith("signals.")]:
        trail = reaches_oracle(module, graph)
        assert trail is None, f"C3 VIOLATION in signals: {' -> '.join(trail)}"


def test_the_check_can_actually_fail():
    """Guard against a checker that passes because it never looks at anything."""
    graph = {"detect.fake": {"oracle.score"}}
    assert reaches_oracle("detect.fake", graph) is not None

    graph = {"detect.fake": {"helper"}, "helper": {"oracle.score"}}
    assert reaches_oracle("detect.fake", graph) is not None, "transitive path missed"
