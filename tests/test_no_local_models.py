"""GATE - C7: no local model inference, anywhere. Must pass from day one.

Hardware independence is a project requirement. Embeddings come from the
`embedding` role served by the batch endpoint - which this platform does
provide - never from a local model.

This gate matters more than it looks: developer machines here already carry a
HuggingFace cache with bge and all-MiniLM-L6-v2 sitting in it. Importing one is
a single line, it would work immediately, and nothing else in the project would
notice. This test is what notices.
"""

from __future__ import annotations

import ast

import pytest

from tests.conftest import PACKAGE_ROOT, python_files

pytestmark = pytest.mark.gate

FORBIDDEN_MODULES = {
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "sentence_transformers",
    "accelerate",
    "onnxruntime",
    "llama_cpp",
    "vllm",
    "ctransformers",
}


def imported_top_level_modules(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module.split(".")[0])
    return found


def test_no_forbidden_imports_in_package():
    violations = {}
    for path in python_files(PACKAGE_ROOT):
        bad = imported_top_level_modules(path) & FORBIDDEN_MODULES
        if bad:
            violations[str(path.relative_to(PACKAGE_ROOT.parent))] = sorted(bad)

    assert not violations, (
        "C7 VIOLATION - local model inference:\n"
        + "\n".join(f"  {f}: {m}" for f, m in violations.items())
        + "\n\nEmbeddings come from the `embedding` role on the batch endpoint "
        "(POST /v1/embeddings, same JSONL pipeline as chat). The local HF cache "
        "is not a fallback."
    )


def test_forbidden_packages_not_declared_as_dependencies():
    from tests.conftest import PROJECT_ROOT

    text = (PROJECT_ROOT / "pyproject.toml").read_text()
    in_deps = False
    declared = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("dependencies", "dev =")) or stripped == "[dependency-groups]":
            in_deps = True
        elif stripped.startswith("[") and stripped != "[dependency-groups]":
            in_deps = False
        if in_deps:
            for mod in FORBIDDEN_MODULES:
                if f'"{mod}' in stripped or f'"{mod.replace("_", "-")}' in stripped:
                    declared.append(stripped)
    assert not declared, f"C7 VIOLATION - forbidden dependency declared: {declared}"


def test_datasets_is_allowed_but_only_in_corpus():
    """`datasets` builds the corpus; it must never be a route to pulling a model."""
    offenders = []
    for path in python_files(PACKAGE_ROOT):
        if "datasets" in imported_top_level_modules(path) and path.parent.name != "corpus":
            offenders.append(str(path.relative_to(PACKAGE_ROOT.parent)))
    assert not offenders, f"`datasets` imported outside corpus/: {offenders}"
