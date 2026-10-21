"""SPEC - remote generation through a router, offline (provider.remote).

Pins what a router must never be allowed to do silently: answer from another
host, think on some rows, spend past the cap, or turn a refused request into
a recorded "model failure".
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from provider.remote import RemoteClient

pytestmark = pytest.mark.spec

DEP = {
    "model": "vendor/model-0731",
    "provider": "host/fp8",
    "provider_name": "Host",
    "quantization": "fp8",
}


def _row(sample_query, model: str = DEP["model"]) -> dict:
    from provider.expand import build_chat_row

    return build_chat_row(sample_query, 0, model, "healthy", "api001", "sys", 1, 0.7, 64)


def _answer(provider: str = "Host", cost: float = 0.001, reasoning: int = 0) -> dict:
    return {
        "id": "gen-1",
        "provider": provider,
        "model": DEP["model"],
        "choices": [{"message": {"role": "assistant", "content": "42"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cost": cost,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


def _client(handler, *, budget: float = 1.0, spent: float = 0.0) -> RemoteClient:
    from provider.remote import RemoteClient

    return RemoteClient(
        "https://router.test/api/v1",
        "test-key",
        DEP,
        budget_usd=budget,
        already_spent_usd=spent,
        transport=httpx.MockTransport(handler),
    )


def test_every_configured_deployment_is_fully_pinned():
    from settings import load_model_roles, load_remote_config

    cfg = load_remote_config()
    assert cfg["budget_usd"] > 0
    local_ids = {e["id"] for e in load_model_roles().values()}
    for name, dep in cfg["deployments"].items():
        required = {"model", "provider", "provider_name", "quantization"}
        assert required <= set(dep) <= required | {"thinking_switch"}, name
        assert "/" in dep["provider"] and dep["model"] not in local_ids, name


def test_every_request_is_pinned_to_one_host_without_thinking(sample_query):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_answer())

    client = _client(handler)
    row = _row(sample_query)
    line = client.post_row(row)
    (body,) = seen
    assert body["provider"] == {
        "order": ["host/fp8"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert body["reasoning"] == {"enabled": False} and "reasoning_effort" not in body
    assert body["usage"] == {"include": True}
    assert body["seed"] == row["body"]["seed"] and body["messages"] == row["body"]["messages"]
    assert line["custom_id"] == row["custom_id"] and line["error"] is None
    assert client.spent == pytest.approx(0.001)


def test_a_model_that_cannot_think_gets_no_thinking_switch(sample_query):
    from provider.remote import RemoteClient

    plain = RemoteClient(
        "https://router.test/api/v1", "k", {**DEP, "thinking_switch": False}, budget_usd=1
    )
    body = plain.pinned_body(_row(sample_query)["body"])
    assert "reasoning" not in body and "reasoning_effort" not in body
    assert (
        body["provider"]["order"] == ["host/fp8"] and body["provider"]["allow_fallbacks"] is False
    )


def test_a_row_for_another_model_is_refused(sample_query):
    from provider.remote import RemoteError

    with pytest.raises(RemoteError, match="pins"):
        _client(lambda r: httpx.Response(200, json=_answer())).post_row(
            _row(sample_query, "vendor/other")
        )


@pytest.mark.parametrize(
    ("answer", "match"),
    [(_answer(provider="Fallback"), "answered by"), (_answer(reasoning=37), "reasoning tokens")],
)
def test_an_answer_from_another_host_or_with_thinking_stops_the_round(sample_query, answer, match):
    from provider.remote import RemoteError

    client = _client(lambda r: httpx.Response(200, json=answer))
    with pytest.raises(RemoteError, match=match):
        client.post_row(_row(sample_query))
    assert client.spent == pytest.approx(0.001)  # billed even so, and counted


def test_rate_limits_are_retried_not_recorded(sample_query, monkeypatch):
    import provider.remote as remote

    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    calls = iter(
        [httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(200, json=_answer())]
    )
    line = _client(lambda r: next(calls)).post_row(_row(sample_query))
    assert line["response"]["choices"][0]["message"]["content"] == "42"


@pytest.mark.parametrize(
    "failed",
    [{"error": {"code": 502, "message": "upstream error"}}, {"id": "gen-1", "usage": {}}],
)
def test_a_200_that_carries_an_upstream_failure_is_retried_not_refused(
    sample_query, monkeypatch, failed
):
    import provider.remote as remote

    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    calls = iter([httpx.Response(200, json=failed), httpx.Response(200, json=_answer())])
    line = _client(lambda r: next(calls)).post_row(_row(sample_query))
    assert line["response"]["provider"] == "Host"


def test_retries_back_off_patiently_then_stop_the_round(sample_query, monkeypatch):
    import provider.remote as remote

    waits: list[float] = []
    monkeypatch.setattr(remote.time, "sleep", waits.append)
    with pytest.raises(remote.RemoteError, match="429"):
        _client(lambda r: httpx.Response(429, json={"error": "busy"})).post_row(_row(sample_query))
    assert len(waits) == remote._RETRIES - 1
    assert waits == sorted(waits[:4]) + waits[4:]  # doubling at first
    assert waits[0] >= remote._BACKOFF_S
    assert max(waits) <= remote._BACKOFF_MAX_S * 1.25
    assert sum(waits) >= 600  # about ten minutes before a row gives up


def test_a_refused_request_is_never_recorded_as_the_models_answer(sample_query, tmp_path):
    from provider.local import run_rows
    from provider.remote import RemoteError

    client = _client(lambda r: httpx.Response(402, json={"error": {"message": "no credit"}}))
    checkpoint = tmp_path / "api001" / "results.jsonl"
    with pytest.raises(RemoteError, match="402"):
        run_rows(client, [_row(sample_query)], checkpoint, concurrency=1)
    assert not checkpoint.exists() or checkpoint.read_text() == ""


def test_nothing_is_sent_once_the_cap_is_reached(sample_query):
    from provider.remote import BudgetExceededError

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a request was sent past the cap")

    with pytest.raises(BudgetExceededError, match="cap"):
        _client(handler, budget=5.0, spent=5.0).post_row(_row(sample_query))


def test_with_the_users_own_host_key_both_charges_count():
    from provider.remote import answer_cost

    upstream = {"upstream_inference_cost": 0.03}
    assert answer_cost({"cost": 0.03, "is_byok": False, "cost_details": upstream}) == 0.03
    assert answer_cost({"cost": 0.001, "is_byok": True, "cost_details": upstream}) == pytest.approx(
        0.031
    )
    assert answer_cost({}) == 0.0


def test_spend_is_read_back_from_every_answer_on_disk(sample_query, tmp_path):
    from provider.local import run_rows
    from provider.remote import spent_usd

    client = _client(lambda r: httpx.Response(200, json=_answer(cost=0.25)))
    run_rows(
        client,
        [_row(sample_query)],
        tmp_path / "api" / "rounds" / "a001" / "results.jsonl",
        concurrency=1,
    )
    local = tmp_path / "rounds" / "r001" / "results.jsonl"
    local.parent.mkdir(parents=True)
    local.write_text(
        json.dumps({"custom_id": "x", "response": {"usage": {}}, "error": None}) + "\n"
    )
    assert spent_usd(tmp_path) == pytest.approx(0.25)


def test_the_round_is_pinned_to_the_endpoint_or_refused():
    from provider.remote import RemoteError

    listed = {"data": {"endpoints": [{"tag": "host/fp8"}, {"tag": "other/fp4"}]}}
    pin = _client(lambda r: httpx.Response(200, json=listed)).resolve("dep")
    assert pin.digest == "vendor/model-0731@host/fp8" and pin.digest_source == "pinned"
    gone = {"data": {"endpoints": [{"tag": "other/fp4"}]}}
    with pytest.raises(RemoteError, match="not served"):
        _client(lambda r: httpx.Response(200, json=gone)).resolve("dep")


def test_the_remote_key_never_appears_in_the_settings_repr(monkeypatch):
    from settings import get_settings

    monkeypatch.setenv("DRIFTWATCH_REMOTE_API_KEY", "sk-secret-value")
    get_settings.cache_clear()
    try:
        assert "sk-secret-value" not in repr(get_settings())
    finally:
        get_settings.cache_clear()


def _rows(sample_query, n: int) -> list[dict]:
    from provider.expand import build_chat_row

    qs = [sample_query.model_copy(update={"query_id": f"gsm8k_{i:05d}"}) for i in range(n)]
    return [build_chat_row(q, 0, DEP["model"], "U1", "a003", "sys", 1, 0.7, 64) for q in qs]


def test_a_failed_round_resumes_exactly_where_it_stopped(sample_query, tmp_path):
    from provider.collect import read_completions
    from provider.local import generate_round
    from provider.remote import RemoteError

    rows = _rows(sample_query, 6)
    round_dir, out = tmp_path / "rounds" / "a003", tmp_path / "completions.parquet"
    sent = []

    def flaky(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["seed"])
        if len(sent) > 3:
            return httpx.Response(402, json={"error": {"message": "out of credit"}})
        return httpx.Response(200, json=_answer())

    kwargs = {
        "round_id": "a003",
        "condition": "U1",
        "model_id": DEP["model"],
        "digest": "vendor/model-0731@host/fp8",
        "digest_source": "pinned",
        "concurrency": 1,
    }
    with pytest.raises(RemoteError, match="402"):
        generate_round(_client(flaky), rows, round_dir, out, **kwargs)
    first = (round_dir / "results.jsonl").read_text().splitlines()
    assert len(first) == 3 and not out.exists()  # nothing half-appended

    resent = []

    def healthy(request: httpx.Request) -> httpx.Response:
        resent.append(json.loads(request.content)["seed"])
        return httpx.Response(200, json=_answer())

    generate_round(_client(healthy), rows, round_dir, out, **kwargs)
    done = [
        json.loads(line)["custom_id"]
        for line in (round_dir / "results.jsonl").read_text().splitlines()
    ]
    assert len(resent) == 3 and not set(resent) & set(sent[:3])  # only the missing rows
    assert sorted(done) == sorted(r["custom_id"] for r in rows)  # each once, none twice
    assert len(read_completions(out)) == 6

    generate_round(_client(healthy), rows, round_dir, out, **kwargs)
    assert len(resent) == 3 and len(read_completions(out)) == 6  # a rerun sends and appends nothing


def test_the_api_study_is_fully_pinned_and_varies_one_thing_per_fault():
    from typing import get_args

    from settings import Condition, load_api_study, load_remote_config

    study = load_api_study()
    deps = load_remote_config()["deployments"]
    rounds = study["rounds"]
    assert set(rounds) == {"healthy", "F7", "U1", "Q1"}
    assert set(rounds) <= set(get_args(Condition))
    assert len({r["round_id"] for r in rounds.values()}) == 4
    healthy, update, quant = (deps[rounds[c]["deployment"]] for c in ("healthy", "U1", "Q1"))
    assert rounds["F7"]["deployment"] == rounds["healthy"]["deployment"]
    assert rounds["F7"]["seeds"] == "f7" and rounds["healthy"]["seeds"] == "healthy"
    # U1 changes only the model version; Q1 only the host and its precision.
    assert update["model"] != healthy["model"]
    assert (update["provider"], update["quantization"]) == (
        healthy["provider"],
        healthy["quantization"],
    )
    assert quant["model"] == healthy["model"] and quant["quantization"] != healthy["quantization"]
    assert rounds["healthy"]["questions"] == "all"
    assert all(rounds[c]["questions"] == "fault" for c in ("F7", "U1", "Q1"))


def test_fault_rounds_wait_for_the_frozen_study_thresholds(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import settings
    from cli import app

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    result = CliRunner().invoke(app, ["remote-generate", "--condition", "U1"])
    assert result.exit_code != 0
    assert "frozen study thresholds" in result.output
