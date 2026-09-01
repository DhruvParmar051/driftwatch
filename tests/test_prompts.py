"""GATE - prompt configuration. Must pass from day one.

Prompts are experimental substance, not incidental strings. The thresholds are
calibrated against model behaviour ON THESE PROMPTS, so a template that drifts
after the freeze silently redefines the experiment. These tests keep prompts in
one versioned file and out of the source tree.
"""

from __future__ import annotations

import re
import string

import pytest

from settings import SLICES, load_prompts, slice_template, system_prompt_for_condition
from tests.conftest import PACKAGE_ROOT, python_files

pytestmark = pytest.mark.gate


def test_prompts_file_is_versioned():
    """The version is recorded in thresholds.json and checked at verify()."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", load_prompts()["version"])


def test_every_slice_has_a_template():
    templates = load_prompts()["slice_templates"]
    assert set(templates) == set(SLICES)
    for name, text in templates.items():
        assert text.strip(), f"{name} template is empty"


def test_templates_only_use_known_placeholders():
    """A typo'd placeholder would raise at corpus-build time, a day in."""
    allowed = {"question", "context", "label_space"}
    for name in SLICES:
        fields = {f for _, f, _, _ in string.Formatter().parse(slice_template(name)) if f}
        unknown = fields - allowed
        assert not unknown, f"{name} template uses unknown placeholder(s): {unknown}"


def test_slices_needing_context_have_it():
    for name in ("squad2", "xsum"):
        assert "{context}" in slice_template(name), f"{name} must include the passage"


def test_banking77_template_injects_the_label_space():
    """S6 maps completions to labels; the model must be shown the label set."""
    assert "{label_space}" in slice_template("banking77")


def test_jsonx_template_states_the_schema_in_prose():
    """jsonx rows carry no response_format, so the schema must be in the prompt.

    If it were passed as a strict json_schema instead, the platform would
    validate server-side and fail malformed rows - and S5 would have nothing
    left to measure.
    """
    template = slice_template("jsonx")
    for field in ("title", "category", "priority", "tags", "published"):
        assert field in template, f"jsonx template does not mention {field!r}"


def test_jsonx_template_matches_the_schema_it_is_validated_against():
    """Prompt and validator must describe the same object, or S5 measures a
    mismatch between two configs rather than model degradation."""
    from settings import load_jsonx_schema

    template = slice_template("jsonx")
    for field in load_jsonx_schema()["required"]:
        assert field in template, f"schema requires {field!r} but the prompt never asks for it"


def test_f3_is_the_only_condition_that_changes_the_system_prompt():
    """F3 IS the corrupted prompt; F1/F2/F5 change the model, F4 the query mix."""
    healthy = system_prompt_for_condition("healthy")
    assert system_prompt_for_condition("F3") != healthy
    for condition in ("F1", "F2", "F4", "F5"):
        assert system_prompt_for_condition(condition) == healthy, condition


def test_f3_prompt_is_a_plausible_edit_not_obvious_sabotage():
    """F3 is the SUBTLE rung of the ladder.

    It must read like something a careless operator would actually ship. An
    obviously broken prompt would be trivially detectable and would say nothing
    about the subtle end of the detection curve, which is the interesting part.
    """
    healthy = system_prompt_for_condition("healthy")
    corrupted = system_prompt_for_condition("F3")
    assert corrupted.startswith(healthy.strip()[:40]), (
        "F3 should be an EDIT of the healthy prompt, not a different prompt"
    )
    ratio = len(corrupted) / len(healthy)
    assert 1.0 < ratio < 3.0, f"F3 prompt is {ratio:.1f}x the healthy one - not a subtle edit"


def test_no_prompt_text_is_inlined_in_source():
    """Prompts live in config/prompts.yaml and nowhere else.

    An inline prompt bypasses the version that verify() checks, so it can drift
    after a freeze without anything noticing.
    """
    fragments = ["You are a helpful assistant", "Answer the user's question",
                 "Summarise the article", "Solve this problem"]
    offenders = []
    for path in python_files(PACKAGE_ROOT):
        text = path.read_text()
        for fragment in fragments:
            if fragment in text:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {fragment!r}")
    assert not offenders, "prompt text inlined in source:\n" + "\n".join(offenders)


def test_unknown_slice_raises():
    with pytest.raises(KeyError):
        slice_template("not_a_slice")
