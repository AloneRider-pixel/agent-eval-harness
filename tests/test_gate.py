"""Release gate tests: baselines, budgets, compare logic, and the required
proof that a deliberately regressed agent FAILS the gate."""
import json

from app.gate import (baseline_path, check, compare, main, record)

CFG = "gate.config.json"


def test_compare_passes_when_metrics_hold():
    baseline = {"summary": {"success_rate": 0.6667, "mean_score": 0.6667,
                            "avg_latency_s": 0.001, "total_cost_usd": 0.0}}
    current = dict(baseline["summary"])
    passed, reasons = compare(current, baseline,
                              {"success_rate": 0.0, "mean_score": 0.05},
                              {"max_avg_latency_s": 5.0})
    assert passed and reasons == []


def test_compare_detects_regression_beyond_tolerance():
    baseline = {"summary": {"success_rate": 0.9, "mean_score": 0.9}}
    current = {"success_rate": 0.7, "mean_score": 0.87}
    passed, reasons = compare(current, baseline,
                              {"success_rate": 0.0, "mean_score": 0.05}, {})
    assert not passed
    assert any("success_rate" in r for r in reasons)
    assert not any("mean_score" in r for r in reasons)  # within tolerance


def test_compare_budgets_both_directions():
    current = {"avg_latency_s": 9.0, "success_rate": 0.4}
    passed, reasons = compare(current, {"summary": {}}, {},
                              {"max_avg_latency_s": 5.0, "min_success_rate": 0.6})
    assert not passed and len(reasons) == 2
    assert any(">" in r for r in reasons) and any("<" in r for r in reasons)


def test_compare_ignores_unlisted_metrics_and_missing_values():
    passed, reasons = compare({"success_rate": 0.1}, {"summary": {}},
                              {"success_rate": 0.0}, {})
    assert passed  # no baseline value recorded -> nothing to compare against


def test_record_writes_baseline_with_disclosure_metadata(tmp_path):
    b = record("scripted", None, None, root=tmp_path)
    path = baseline_path("scripted", None, tmp_path)
    assert path.exists()
    saved = json.loads(path.read_text())
    for key in ("agent", "dataset", "sample_count", "scorer", "recorded_at",
                "commit", "summary", "label"):
        assert key in saved
    assert saved["summary"]["success_rate"] == round(2 / 3, 4)
    assert saved["label"] == "harness validation (fixture agent)"
    assert saved["sample_count"] == 3


def test_gate_passes_unchanged_agent_and_fails_regressed_agent(tmp_path):
    # record a baseline for the agent, then simulate the same agent regressing
    record("scripted", None, None, root=tmp_path)
    code, _ = check("scripted", None, None, CFG, root=tmp_path)
    assert code == 0

    regressed = baseline_path("always_fail", None, tmp_path)
    regressed.parent.mkdir(parents=True, exist_ok=True)
    regressed.write_text(baseline_path("scripted", None, tmp_path).read_text())
    code, reasons = check("always_fail", None, None, CFG, root=tmp_path)
    assert code == 1, "a deliberately regressed agent must fail the gate"
    assert any("regression: success_rate" in r for r in reasons)
    assert any("budget breach" in r for r in reasons)


def test_check_without_baseline_is_an_error_not_a_pass(tmp_path):
    code, reasons = check("scripted", None, None, None, root=tmp_path)
    assert code == 2 and any("no baseline" in r for r in reasons)


def test_cli_record_then_check(tmp_path, monkeypatch, capsys):
    import app.gate as gate
    monkeypatch.setattr(gate, "BASELINES_DIR", tmp_path)
    assert main(["record", "--agent", "scripted"]) == 0
    assert "harness validation" in capsys.readouterr().out
    assert main(["check", "--agent", "scripted", "--config", CFG]) == 0
    assert "GATE PASS" in capsys.readouterr().out
