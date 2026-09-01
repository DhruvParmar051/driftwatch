"""GATE - structural invariants. Must pass from day one.

Checks the scaffold itself holds together: the registry agrees with the settings
vocabulary, every signal satisfies the protocol, the CLI builds, model ids live
only in config, and the schemas enforce their invariants.
"""

from __future__ import annotations

import pytest

from tests.conftest import PACKAGE_ROOT, python_files

pytestmark = pytest.mark.gate


def test_registry_matches_signal_names():
    from settings import SIGNAL_NAMES
    from signals.registry import SIGNAL_CLASSES

    assert tuple(SIGNAL_CLASSES) == SIGNAL_NAMES


def test_every_signal_class_declares_its_registered_name():
    """A mismatch would put a score under the wrong threshold key at detection time."""
    from signals.registry import SIGNAL_CLASSES

    for registered_name, cls in SIGNAL_CLASSES.items():
        assert cls.name == registered_name, f"{cls.__name__}.name is {cls.name!r}, registered as {registered_name!r}"


def test_every_signal_declares_fit_and_score():
    from signals.registry import SIGNAL_CLASSES

    for cls in SIGNAL_CLASSES.values():
        assert callable(getattr(cls, "fit", None)), f"{cls.__name__} has no fit()"
        assert callable(getattr(cls, "score", None)), f"{cls.__name__} has no score()"


def test_cli_app_builds_and_lists_every_stage():
    """`driftwatch --help` must work before anything is implemented."""
    from typer.testing import CliRunner

    from cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for command in ("corpus", "synth", "probe", "expand", "submit", "collect",
                    "calibrate", "freeze", "verify", "oracle", "evaluate", "report"):
        assert command in result.output, f"CLI is missing `{command}`"


def test_model_ids_appear_only_in_config():
    """No model id, backend URL or provider name in the source tree."""
    from settings import load_model_roles

    ids = {entry["id"] for entry in load_model_roles().values()}
    assert ids, "no roles configured"

    offenders = []
    for path in python_files(PACKAGE_ROOT):
        text = path.read_text()
        for model_id in ids:
            # Docstrings may DISCUSS a model; code may not depend on one.
            for lineno, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if model_id in stripped and not stripped.startswith(("#", "*", '"""', "'''")):
                    offenders.append(f"{path.name}:{lineno}: {stripped[:70]}")
    assert not offenders, "model ids must live only in config/models.yaml:\n" + "\n".join(offenders)


def test_settings_has_no_default_backend_url():
    """A default base_url in the source tree is exactly what this project forbids."""
    from settings import Settings

    assert Settings.model_fields["base_url"].is_required()
    assert Settings.model_fields["api_key"].is_required()


def test_config_files_load():
    from settings import load_jsonx_schema, load_model_roles, load_refusal_patterns

    roles = load_model_roles()
    for required in ("healthy", "sibling", "weak", "embedding"):
        assert required in roles, f"role {required!r} missing from config/models.yaml"

    patterns = load_refusal_patterns()
    assert patterns["version"], "refusal patterns must be versioned - it is recorded at freeze time"
    assert patterns["patterns"]

    schema = load_jsonx_schema()
    assert schema["type"] == "object" and schema["required"]


def test_refusal_patterns_compile():
    import re

    from settings import load_refusal_patterns

    for pattern in load_refusal_patterns()["patterns"]:
        re.compile(pattern, re.IGNORECASE)


def test_condition_roles_cover_every_condition():
    from settings import CONDITION_ROLES, load_model_roles

    roles = load_model_roles()
    for condition, role in CONDITION_ROLES.items():
        assert role in roles, f"condition {condition} maps to unknown role {role}"


def test_f3_and_f4_run_on_the_healthy_model():
    """F3 corrupts the system prompt and F4 shifts queries - neither changes weights."""
    from settings import CONDITION_ROLES

    assert CONDITION_ROLES["F3"] == "healthy"
    assert CONDITION_ROLES["F4"] == "healthy"


def test_query_rejects_missing_ground_truth_on_verifiable_slice():
    from pydantic import ValidationError

    from corpus.schemas import Query

    with pytest.raises(ValidationError):
        Query(query_id="gsm8k_1", slice="gsm8k", prompt="p", ground_truth=None, split="calib")


def test_query_rejects_label_space_outside_banking77():
    from pydantic import ValidationError

    from corpus.schemas import Query

    with pytest.raises(ValidationError):
        Query(query_id="xsum_1", slice="xsum", prompt="p", label_space=["a"], split="calib")


def test_thresholds_rejects_signal_count_mismatch():
    """The Sidak level is computed for a specific m; a mismatch invalidates it."""
    from datetime import datetime, timezone

    from pydantic import ValidationError

    from corpus.schemas import Thresholds

    with pytest.raises(ValidationError):
        Thresholds(
            frozen_at=datetime.now(timezone.utc), git_commit="abc1234",
            calib_rows_hash="sha256:x", k=4, window_size=200, window_stride=50,
            family_wise_budget=0.001, per_signal_alpha=0.000143, n_signals=7,
            bootstrap_windows=20000, calibration_seed=1, refusal_patterns_version="1.0.0",
            prompts_version="1.0.0",
            thresholds={"length_ks": 0.1},  # only 1, but n_signals says 7
        )


def test_completion_allows_empty_text_but_records_parse_failure():
    """C6 - an unparseable completion is data, not something to drop."""
    from datetime import datetime, timezone

    from corpus.schemas import Completion

    c = Completion(
        query_id="q1", sample_idx=0, seed=1, condition="F1", model_id="m", digest="d",
        digest_source="backend", text="", finish_reason="length", parse_ok=False,
        submitted_at=datetime.now(timezone.utc),
    )
    assert c.parse_ok is False and c.text == ""
