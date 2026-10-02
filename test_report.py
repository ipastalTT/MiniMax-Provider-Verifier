"""Run the MiniMax verifiers against a provider endpoint and write a Markdown report.

Two harnesses are covered, all suites running in parallel:

* pytest format checks (m3_format_check/): each suite runs in its own pytest
  process with its own pytest-xdist worker count and writes a JUnit XML report.
* verify.py (tool-call behaviour on sample.jsonl): one pass over the sample set;
  the aggregate metrics are graded against the reference thresholds in README.md.

The results are aggregated into a Markdown report with per-suite pass rates,
failures grouped by cause, and failures attributed to API features. Features
declared with --unsupported are reported as waived.

Usage:
    uv run ... python test_report.py \
        --base-url https://api.example.com/v1 \
        --model MiniMaxAI/MiniMax-M3 \
        --provider Tenstorrent \
        --workers 16 --unsupported video

    # Rebuild the report from an existing run without calling the API again
    python test_report.py --render-only reports/<run-dir>

    # Grade existing verify.py output instead of running verify.py
    python test_report.py ... --suites verify --verify-results output/x/results.jsonl

The API key is read from MINIMAX_API_KEY (or OPENAI_API_KEY) and is never written
to the report directory. Exit code: 0 if acceptance passed, 1 if it failed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
PYTEST_DIR = REPO_ROOT / "m3_format_check"
DEFAULT_SAMPLE = REPO_ROOT / "sample.jsonl"
DUMMY_API_KEY = "no-auth-dummy-key"

# suite -> (pytest file or None for verify.py, description)
SUITES = {
    "text": ("m3_text_tests.py", "Text chat: basics, SSE, thinking, sampling, max_tokens, message format, "
                                 "usage, role=root, tool calling, error codes"),
    "image": ("m3_image_tests.py", "Image input: base64/URL, multi-image, resolution tiers, size/count limits"),
    "video": ("m3_video_tests.py", "Video input: formats, fps/detail, resolution tiers, size limits"),
    "stream": ("m3_stream_tests.py", "SSE protocol and stream packet-length distribution"),
    "reasoning_effort": ("m3_a_reasoning_effort_tests.py", "reasoning_effort thinking-depth control (M3-a / M3.1)"),
    "verify": (None, "verify.py on sample.jsonl: tool-call trigger / schema metrics vs. reference thresholds"),
}
PYTEST_SUITES = [s for s, (f, _) in SUITES.items() if f]

# Suites that only apply to some models: suite -> pattern the model name must match.
# m3_a_reasoning_effort_tests.py belongs to the M3-a (M3.1) case family: it asserts that
# thinking is forced on (reasoning_effort=none and thinking.disabled still think) and that
# usage reports reasoning_tokens, which is not the MiniMax-M3 contract (m3_text_tests.py
# treats thinking as optional/adaptive).
M3A_PATTERN = r"m3[-_.]?a(?![a-z0-9])|m3\.1(?![0-9])"
MODEL_SCOPED_SUITES = {"reasoning_effort": M3A_PATTERN}

# Official-deployment results used as the ToolCalls-Trigger-Similarity gold standard
# (same folders scripts/calculate_batch_metrics.py compares against). First match wins.
BASELINES = [
    (M3A_PATTERN, "output-dir/MiniMax-M3-a"),
    (r"(?<![a-z0-9])m3(?![a-z0-9])", "output-dir/MiniMax-M3"),
    (r"m2\.7", "output-dir/MiniMax-M2.7"),
    (r"m2\.5", "output-dir/MiniMax-M2.5"),
]

FEATURES = {
    "chat": "Basic chat, multi-turn & message format",
    "streaming": "SSE streaming",
    "thinking": "Thinking / reasoning (reasoning_split, reasoning_effort)",
    "sampling_params": "Sampling / decoding parameters",
    "max_tokens": "max_tokens limits",
    "structured_output": "Structured output (response_format)",
    "usage": "Usage accounting",
    "role_root": "role=root messages",
    "semantics": "Text semantics & instruction following",
    "tool_calling": "Tool calling",
    "long_context": "Long context / parameter stress",
    "finish_reason": "finish_reason semantics",
    "error_handling": "Error codes for invalid requests",
    "auth": "Authentication (401 for missing / invalid key)",
    "model_compat": "Model-name compatibility",
    "vision": "Vision / image input",
    "video": "Video input",
    "availability": "Request success / availability",
    "other": "Other",
}

# (feature, suite pattern, test-id pattern, message pattern); first match wins.
# The test id is "<module>.<Class>::<test name>".
FEATURE_RULES: list[tuple[str, str, str, str]] = [
    ("vision", r"^image$", "", ""),
    ("video", r"^video$", "", ""),
    ("streaming", r"^stream$", "", ""),
    ("thinking", r"^reasoning_effort$", "", ""),
    ("auth", "", r"TestErrorCodes::test_20_0[57]_", ""),
    ("error_handling", "", r"TestErrorCodes", ""),
    ("structured_output", "", r"TestResponseFormat", ""),
    ("streaming", "", r"TestSSEStream", ""),
    ("thinking", "", r"TestThinking|TestReasoningSplit", ""),
    ("sampling_params", "", r"TestSampling", ""),
    ("max_tokens", "", r"TestMaxTokens", ""),
    ("usage", "", r"TestUsageField", ""),
    ("role_root", "", r"TestRoleRoot", ""),
    ("semantics", "", r"TestTextSemantic", ""),
    ("tool_calling", "", r"TestToolCall", ""),
    ("long_context", "", r"TestParamStress", ""),
    ("finish_reason", "", r"TestFinishReason", ""),
    ("model_compat", "", r"TestModelCompat", ""),
    ("chat", "", r"TestBasicText|TestMultiturn|TestMessageFormat", ""),
]

# Known failure causes, matched in order against the failure message.
# Anything unmatched is grouped by its normalised first line.
KNOWN_CAUSES: list[tuple[str, str]] = [
    (r"unknown variant `root`", "role=root rejected (HTTP 400: unknown variant `root`)"),
    (r"Expected 401, got 200", "Unauthenticated request accepted (expected HTTP 401)"),
    (r"Expected 4\d\d, got 200|should be rejected", "Invalid request accepted (expected HTTP 4xx)"),
    (r"missing trace_id", "Error response carries no trace id"),
    (r"packet quality FAIL.*no_non_empty_target_fragments",
     "Stream has no non-empty target fragments (content / arguments not streamed incrementally)"),
    (r"packet quality FAIL", "Stream packet-length distribution violates the quality rule"),
    (r"Timeout \(>[\d.]+s\) from pytest-timeout", "Test timed out (pytest-timeout)"),
    (r"ReadTimeout|read operation timed out", "HTTP read timeout"),
    (r"Server disconnected without sending a response|RemoteProtocolError|ConnectError|Connection refused",
     "Connection dropped / refused by the server"),
    (r"HTTP[= ]?500|got 500|500 == 200|Internal server error", "HTTP 500 Internal server error"),
    (r"(assert|got|HTTP[= ]?)\s*4\d\d( == 200|\b)", "Valid request rejected (HTTP 4xx, expected 200)"),
    (r"expected at least one tool_call, got none|expected tool_calls including .*got none",
     "No tool call returned"),
    (r"expected no tool_call, but got", "Tool call made although none was expected"),
    (r"expected tool name", "Tool call made with a wrong function name"),
    (r"arguments (is|are) not valid JSON|arguments not valid JSON", "Tool-call arguments are not valid JSON"),
    (r"type mismatch|must be numeric|not in enum|below minimum|above maximum|below min|above max"
     r"|required but missing|expected arg .* missing", "Tool-call arguments violate the JSON schema"),
    (r"value mismatch", "Tool-call argument value differs from the expected one"),
    (r"missing tool calls", "Expected parallel tool calls missing"),
    (r"expected thinking present", "Thinking expected but no reasoning signal returned"),
    (r"expected no thinking", "Thinking returned although it should be absent"),
    (r"reasoning_tokens > 0", "usage.completion_tokens_details.reasoning_tokens missing or 0"),
    (r"usage must appear only in the final stream chunk|no usage chunk in stream|final-chunk usage",
     "Stream usage chunk missing or misplaced"),
    (r"missing valid finish_reason", "Stream ends without a valid finish_reason"),
    (r"Missing 'choices'|KeyError: 'usage'|assert 'usage' in", "Response is missing `choices` / `usage`"),
    (r"stream has no (data )?chunks", "Stream returned no chunks"),
    (r"worker '\w+' crashed", "pytest-xdist worker crashed"),
]

STATUS_ICON = {"PASS": "✅", "FAIL": "❌", "NA": "🟨", "NOT GRADED": "⚪"}

# verify.py metric -> (label, comparison, threshold, threshold text, feature).
# Thresholds from README.md "Reference Thresholds". ToolCalls-Match-Rate is documented as
# "≈98% with a fluctuation of approximately ±1%", so it is graded as >= 97%.
VERIFY_METRICS = {
    "query_success_rate": ("Query-Success-Rate", ">=", 1.0, "100%", "availability"),
    "tool_calls_match_rate": ("ToolCalls-Match-Rate", ">=", 0.97, "≈98% (graded ≥97%)", "tool_calling"),
    "tool_calls_trigger_similarity": ("ToolCalls-Trigger-Similarity", ">=", 0.98, "≥98%", "tool_calling"),
    "tool_calls_schema_accuracy": ("ToolCalls-Schema-Accuracy", ">=", 0.98, "≥98%", "tool_calling"),
    "error_only_reasoning_rate": ("Error-Only-Reasoning-Rate", "<=", 0.0, "0%", "thinking"),
    "language_following_success_rate": ("Language-Following-Success-Rate", ">=", 0.40, "≥40%", "sampling_params"),
    "scenario_check_pass_rate": ("Scenario-Check-Pass-Rate", ">=", 1.0, "100%", "tool_calling"),
}


@dataclass
class TestResult:
    suite: str
    classname: str
    name: str
    outcome: str  # passed | failed | error | skipped | xfailed
    message: str = ""
    feature: str = ""
    waived: bool = False
    details: str = ""  # full pytest failure text (traceback + assertion)

    @property
    def is_failure(self) -> bool:
        return self.outcome in ("failed", "error")


@dataclass
class SuiteResult:
    name: str
    graded: bool = True
    tests: list[TestResult] = field(default_factory=list)
    duration: float = 0.0
    collection_error: str = ""
    verify: dict | None = None  # verify.py metrics and per-case failures

    def count(self, outcome: str) -> int:
        return sum(t.outcome == outcome for t in self.tests)

    @property
    def executed(self) -> int:
        return self.count("passed") + self.count("failed") + self.count("error")

    @property
    def failures(self) -> list[TestResult]:
        return [t for t in self.tests if t.is_failure]

    @property
    def waived(self) -> int:
        return sum(t.waived for t in self.failures)

    @property
    def pass_rate(self) -> float | None:
        return ratio(self.count("passed"), self.executed)

    @property
    def pass_rate_excl_unsupported(self) -> float | None:
        return ratio(self.count("passed"), self.executed - self.waived)

    @property
    def status(self) -> str:
        if not self.graded:
            return "NOT GRADED"
        if self.collection_error or any(not t.waived for t in self.failures):
            return "FAIL"
        return "PASS" if self.executed else "NA"


def ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def pct(value: float | None, digits: int = 1) -> str:
    return "NA" if value is None else f"{value * 100:.{digits}f}%"


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def one_line(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text[:limit] + ("…" if len(text) > limit else "")


def classify(message: str) -> str:
    for pattern, cause in KNOWN_CAUSES:
        if re.search(pattern, message, re.S):
            return cause
    first = message.strip().splitlines()[0] if message.strip() else "No failure message"
    first = re.sub(r"(chatcmpl|call)-[\w-]+", "<id>", first)
    return first[:160] + ("…" if len(first) > 160 else "")


def attribute_feature(test: TestResult) -> str:
    if test.feature:  # verify.py checks carry their feature already
        return test.feature
    test_id = f"{test.classname}::{test.name}"
    for feature, suite_re, name_re, msg_re in FEATURE_RULES:
        if suite_re and not re.search(suite_re, test.suite):
            continue
        if name_re and not re.search(name_re, test_id):
            continue
        if msg_re and not re.search(msg_re, test.message):
            continue
        return feature
    return "other"


def model_matches(pattern: str, model: str) -> bool:
    return bool(re.search(pattern, model, re.IGNORECASE))


# ---------------------------------------------------------------------------
# pytest (JUnit XML)
# ---------------------------------------------------------------------------

def parse_junit(path: Path, name: str) -> SuiteResult:
    suite = SuiteResult(name=name)
    if not path.exists():
        suite.collection_error = "No JUnit report produced (pytest did not run)"
        return suite
    root = ET.parse(path).getroot()
    for ts in root.iter("testsuite"):
        suite.duration = max(suite.duration, float(ts.get("time", 0) or 0))
        for tc in ts.iter("testcase"):
            outcome, message, details = "passed", "", ""
            for tag in ("failure", "error", "skipped"):
                node = tc.find(tag)
                if node is not None:
                    outcome = {"failure": "failed"}.get(tag, tag)
                    details = node.text or ""
                    if tag == "skipped" and node.get("type") == "pytest.xfail":
                        outcome = "xfailed"
                    message = node.get("message") or (node.text or "")
                    if tag in ("failure", "error"):
                        message = failure_text(node.text or "") or message
                    break
            # Collection errors show up as an <error> testcase with an empty classname.
            if outcome == "error" and not tc.get("classname"):
                lines = message.strip().splitlines()
                suite.collection_error = lines[-1] if lines else "collection error"
                continue
            suite.tests.append(TestResult(name, tc.get("classname", ""), tc.get("name", "?"), outcome, message,
                                          details=details))
    return suite


def failure_text(traceback: str) -> str:
    """The pytest `E` lines of a traceback (the assertion / exception text), without source lines."""
    e_lines = [l[1:].strip() for l in traceback.splitlines() if l.startswith("E ")]
    return "\n".join(l for l in e_lines if l)


# ---------------------------------------------------------------------------
# verify.py (results.jsonl)
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_baseline(model: str, override: str | None) -> Path | None:
    if override:
        return None if override == "none" else Path(override)
    for pattern, folder in BASELINES:
        if model_matches(pattern, model):
            files = sorted(glob.glob(str(REPO_ROOT / folder / "**" / "*_results.jsonl"), recursive=True))
            return Path(files[0]) if files else None
    return None


def finish_reason(row: dict) -> str | None:
    try:
        return (row.get("response") or {}).get("choices", [{}])[0].get("finish_reason")
    except (AttributeError, IndexError, TypeError):
        return None


def diagnose_tool_calls(row: dict) -> tuple[str, str]:
    """Explain why verify.py marked a tool-call response schema-invalid: (cause, detail)."""
    from jsonschema import ValidationError, validate
    from validator.tool_calls import is_valid_array_command

    tools = {t.get("function", {}).get("name"): t.get("function", {}).get("parameters")
             for t in (row.get("request") or {}).get("tools") or []}
    message = ((row.get("response") or {}).get("choices") or [{}])[0].get("message") or {}
    calls = message.get("tool_calls") or []
    if not calls:
        return "finish_reason=tool_calls but no tool_calls returned", ""
    for call in calls:
        fn = call.get("function") or {}
        name, raw = fn.get("name"), fn.get("arguments")
        schema = tools.get(name)
        if not schema:
            return "Tool call to an undefined tool name", f"{name!r}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as e:
            return "Tool-call arguments are not valid JSON", f"{name}: {e}; raw={one_line(str(raw), 120)}"
        try:
            validate(instance=args, schema=schema)
        except ValidationError as e:
            path = "/".join(str(p) for p in e.absolute_path) or "<root>"
            return "Tool-call arguments violate the JSON schema", f"{name} at {path}: {one_line(e.message, 160)}"
        for pname, pschema in (schema.get("properties") or {}).items():
            if (pname == "command" and pschema.get("type") == "array"
                    and (pschema.get("items") or {}).get("type") == "string"
                    and isinstance(args, dict) and args.get(pname) is not None
                    and not is_valid_array_command(args[pname])):
                return ("Array `command` argument merged into one string",
                        f"{name}: {one_line(json.dumps(args[pname], ensure_ascii=False), 120)}")
    return "Tool-call validation failed", ""


def verify_case_failures(rows: list[dict]) -> list[dict]:
    """Per-case verify.py failures, one entry per (case, cause)."""
    from verify import ValidatorRunner

    out = []

    def add(row, cause, feature, detail=""):
        out.append({"case": row.get("data_index"), "expected_tool_call": row.get("expected_tool_call"),
                    "finish_reason": finish_reason(row), "cause": cause, "feature": feature, "detail": detail})

    for row in rows:
        fr, expected = finish_reason(row), row.get("expected_tool_call")
        if row.get("status") != "success":
            add(row, "Request failed after all retries", "availability",
                one_line(str((row.get("response") or {}).get("error", "")), 200))
            continue
        if expected is True and fr != "tool_calls":
            if fr == "stop":
                add(row, "Tool call expected but not made (finish_reason=stop)", "tool_calling")
            else:
                add(row, f"Tool call expected but finish_reason={fr}", "tool_calling")
        elif expected is False and fr == "tool_calls":
            add(row, "Tool call made but not expected", "tool_calling")
        elif expected is False and fr != "stop":
            add(row, f"Plain answer expected but finish_reason={fr}", "tool_calling")
        if expected is True and fr == "tool_calls" and not row.get("tool_calls_valid"):
            cause, detail = diagnose_tool_calls(row)
            add(row, cause, "tool_calling", detail)
        if ValidatorRunner._is_error_only_reasoning_response(row.get("response")):
            add(row, "Reasoning-only response (no content, no tool call)", "thinking")
        if row.get("language_following_checked") and not row.get("language_following_valid"):
            add(row, "Language not followed (Russian characters in the answer)", "sampling_params")
        if row.get("scenario_check_checked") and not row.get("scenario_check_valid"):
            detail = row.get("scenario_check_detail") or {}
            add(row, "Tool parameter key order not preserved", "tool_calling",
                f"expected={detail.get('expected')} actual={detail.get('actual')}")
    return out


def verify_metrics(rows: list[dict], summary: dict | None, baseline: Path | None) -> dict:
    """Recompute the README metrics (same formulas as scripts/calculate_batch_metrics.py)."""
    from verify import ValidatorRunner

    success = sum(r.get("status") == "success" for r in rows)
    attempts = (summary or {}).get("all_count") or len(rows)
    labelled = [r for r in rows if r.get("expected_tool_call") in (True, False)]
    # Like ToolCallsValidator.compute_summary, use the validator's finish reason: cases with a
    # check_type other than tool_calls stay in the denominator but never count as matched.
    tp = sum(r["expected_tool_call"] is True and r.get("tool_calls_finish_reason") == "tool_calls" for r in labelled)
    tn = sum(r["expected_tool_call"] is False and r.get("tool_calls_finish_reason") == "stop" for r in labelled)
    unscored = sum("tool_calls_finish_reason" not in r for r in labelled)
    schema_ok = sum(r["expected_tool_call"] is True and r.get("tool_calls_finish_reason") == "tool_calls"
                    and bool(r.get("tool_calls_valid")) for r in labelled)
    eor = sum(ValidatorRunner._is_error_only_reasoning_response(r.get("response")) for r in rows)
    lang = [r for r in rows if r.get("language_following_checked")]
    scen = [r for r in rows if r.get("scenario_check_checked")]

    m = {
        "query_success_rate": {
            "value": ratio(success, attempts),
            "detail": f"{success} successful queries / {attempts} requests sent (incl. retries)"},
        "tool_calls_match_rate": {
            "value": ratio(tp + tn, len(labelled)),
            "detail": f"{tp + tn}/{len(labelled)} labelled cases (TP={tp}, TN={tn}"
                      + (f"; {unscored} labelled case(s) without the tool_calls check count as unmatched"
                         if unscored else "") + ")"},
        "tool_calls_schema_accuracy": {
            "value": ratio(schema_ok, tp),
            "detail": f"{schema_ok}/{tp} expected tool calls pass schema validation"},
        "error_only_reasoning_rate": {
            "value": ratio(eor, len(rows)),
            "detail": f"{eor}/{len(rows)} responses contain reasoning only"},
        "language_following_success_rate": {
            "value": ratio(sum(bool(r.get("language_following_valid")) for r in lang), len(lang)),
            "detail": f"{sum(bool(r.get('language_following_valid')) for r in lang)}/{len(lang)} checked cases"},
        "scenario_check_pass_rate": {
            "value": ratio(sum(bool(r.get("scenario_check_valid")) for r in scen), len(scen)),
            "detail": f"{sum(bool(r.get('scenario_check_valid')) for r in scen)}/{len(scen)} checked cases"},
    }

    sim = {"value": None, "detail": "no official baseline for this model"}
    if baseline and baseline.exists():
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from calculate_toolcall_similarity import calculate_tool_call_f1

        ref = {r.get("data_index"): r for r in load_jsonl(baseline)}
        pairs = [(ref[r["data_index"]], r) for r in rows if r.get("data_index") in ref]
        if pairs:
            f1 = calculate_tool_call_f1([p[0] for p in pairs], [p[1] for p in pairs])
            sim = {"value": f1["f1"],
                   "detail": (f"F1 vs. `{baseline.relative_to(REPO_ROOT) if baseline.is_relative_to(REPO_ROOT) else baseline}` "
                              f"on {len(pairs)} cases (TP={f1['tp']}, FP={f1['fp']}, FN={f1['fn']}, TN={f1['tn']})")}
    elif baseline:
        sim["detail"] = f"baseline {baseline} not found"
    m["tool_calls_trigger_similarity"] = sim

    ordered = {}
    for key, (label, _op, _threshold, threshold_text, feature) in VERIFY_METRICS.items():
        entry = m[key]
        ordered[key] = {"label": label, "value": entry["value"], "threshold": threshold_text,
                        "status": metric_status(key, entry["value"]), "feature": feature, "detail": entry["detail"]}
    return ordered


def metric_status(key: str, value: float | None) -> str:
    _label, op, threshold, _text, _feature = VERIFY_METRICS[key]
    if value is None:
        return "NA"
    if op == ">=":
        return "PASS" if value >= threshold - 1e-9 else "FAIL"
    return "PASS" if value <= threshold + 1e-9 else "FAIL"


def mean_metrics(per_loop: list[dict]) -> dict:
    """Grade each metric on its mean over the runs (the README thresholds are pass@N means)."""
    out = {}
    for key, first in per_loop[0].items():
        values = [m[key]["value"] for m in per_loop if m[key]["value"] is not None]
        mean = sum(values) / len(values) if values else None
        runs = ", ".join(pct(m[key]["value"], 2) for m in per_loop)
        out[key] = {**first, "value": mean, "status": metric_status(key, mean),
                    "detail": f"mean of {len(values)}/{len(per_loop)} runs ({runs}); run 1: {first['detail']}"}
    return out


def verify_loop_files(out_dir: Path) -> list[tuple[Path, Path]]:
    """(results, summary) per verify.py run: verify_results_loopNN.jsonl for --verify-loops > 1,
    otherwise the single verify_results.jsonl."""
    loops = sorted(out_dir.glob("verify_results_loop*.jsonl"))
    if not loops:
        return [(out_dir / "verify_results.jsonl", out_dir / "verify_summary.json")]
    return [(r, r.with_name(r.name.replace("verify_results_", "verify_summary_")).with_suffix(".json"))
            for r in loops]


def parse_verify(out_dir: Path, meta: dict) -> SuiteResult:
    suite = SuiteResult(name="verify")
    run = meta.get("runs", {}).get("verify", {})
    suite.duration = run.get("wall_seconds", 0.0)
    baseline = Path(meta["verify_baseline"]) if meta.get("verify_baseline") else None
    if baseline and not baseline.is_absolute():
        baseline = REPO_ROOT / baseline
    loop_files = verify_loop_files(out_dir)
    per_loop, cases, n_cases = [], [], 0
    for i, (results, summary_path) in enumerate(loop_files, start=1):
        rows = load_jsonl(results) if results.exists() else []
        if not rows:
            continue  # a run that produced nothing; reported via the loop count below
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        per_loop.append(verify_metrics(rows, summary, baseline))
        cases += [{**c, "loop": i} for c in verify_case_failures(rows)]
        n_cases = n_cases or len(rows)
    if not per_loop:
        suite.collection_error = "No verify.py results produced (see verify.log)"
        return suite
    metrics = per_loop[0] if len(per_loop) == 1 and len(loop_files) == 1 else mean_metrics(per_loop)
    suite.verify = {"metrics": metrics, "cases": cases, "n_cases": n_cases,
                    "loops": len(loop_files), "loops_with_results": len(per_loop)}
    if len(loop_files) > 1:
        suite.verify["per_loop"] = [{k: m["value"] for k, m in pl.items()} for pl in per_loop]
    for key, m in metrics.items():
        outcome = {"PASS": "passed", "FAIL": "failed", "NA": "skipped"}[m["status"]]
        message = (f"{m['label']} = {pct(m['value'], 2)} (threshold {m['threshold']}); {m['detail']}"
                   if m["value"] is not None else f"{m['label']} not computable: {m['detail']}")
        suite.tests.append(TestResult("verify", "verify", m["label"], outcome, message, feature=m["feature"]))
    return suite


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def parse_workers(values: list[str], suites: list[str]) -> dict[str, int]:
    """Parse `--workers 16` and/or `--workers text=4 verify=8`."""
    default = 1
    overrides: dict[str, int] = {}
    for value in values:
        if "=" in value:
            suite, _, n = value.partition("=")
            if suite not in SUITES:
                raise SystemExit(f"--workers: unknown suite {suite!r}")
            overrides[suite] = int(n)
        else:
            default = int(value)
    return {s: overrides.get(s, default) for s in suites}


def secrets_from_env() -> list[str]:
    values = [os.environ.get("MINIMAX_API_KEY", ""), os.environ.get("OPENAI_API_KEY", "")]
    try:
        headers = json.loads(os.environ.get("M3_EXTRA_HEADERS") or "{}")
        if isinstance(headers, dict):
            values += [str(v) for v in headers.values()]
    except json.JSONDecodeError:
        pass
    return sorted({v for v in values if len(v) >= 6 and v != DUMMY_API_KEY}, key=len, reverse=True)


def redact_dir(out_dir: Path, secrets: list[str], paths: list[Path] | None = None) -> list[str]:
    """Replace every secret in every file under out_dir (or in `paths`); return the changed files."""
    changed = []
    if not secrets:
        return changed
    # Case-insensitive: some tests lower-case response text before printing it.
    pattern = re.compile(b"|".join(re.escape(s.encode()) for s in secrets), re.IGNORECASE)
    for path in paths if paths is not None else out_dir.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        new, n = pattern.subn(b"***REDACTED***", data)
        if n:
            path.write_bytes(new)
            changed.append(str(path.relative_to(out_dir)))
    return changed


def redacted_cmd(cmd: list[str], secrets: list[str]) -> str:
    text = shlex.join(cmd)
    for s in secrets:
        text = re.sub(re.escape(s), "***REDACTED***", text, flags=re.IGNORECASE)
    return text


def run_suites(args: argparse.Namespace, meta: dict, out_dir: Path, workers: dict[str, int],
               api_key: str, secrets: list[str]) -> dict[str, dict]:
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    m3_env = {
        **base_env,
        "M3_BASE_URL": meta["m3_base_url"],
        "M3_API_KEY": api_key,
        "M3_AUTH_TYPE": meta["auth_type"],
        "M3_MODEL": args.model,
    }
    procs = {}
    for suite in args.suites:
        if suite == "verify":
            if args.verify_results:
                continue
            sample = Path(meta["verify_sample"])
            if args.verify_limit:
                subset = out_dir / "verify_sample.jsonl"
                with open(sample, encoding="utf-8") as src, open(subset, "w", encoding="utf-8") as dst:
                    for i, line in enumerate(src):
                        if i >= args.verify_limit:
                            break
                        dst.write(line)
                sample = subset
            def verify_cmd(results: Path, summary: Path) -> list[str]:
                c = [sys.executable, "verify.py", str(sample), "--model", args.model, "--base-url", args.base_url,
                     "--concurrency", str(workers[suite]), "--output", str(results), "--summary", str(summary)]
                if os.environ.get("M3_EXTRA_HEADERS"):
                    c += ["--extra-headers", os.environ["M3_EXTRA_HEADERS"]]
                return c

            if args.verify_loops <= 1:
                cmd = verify_cmd(out_dir / "verify_results.jsonl", out_dir / "verify_summary.json")
            else:
                # Runs one after another (same load as a single run); the suite's exit code is the
                # last non-zero one, and every run that wrote results is graded.
                runs = [shlex.join(verify_cmd(out_dir / f"verify_results_loop{i:02d}.jsonl",
                                              out_dir / f"verify_summary_loop{i:02d}.json"))
                        for i in range(1, args.verify_loops + 1)]
                script = "rc=0\n" + "".join(
                    f'echo "=== verify.py run {i}/{len(runs)}"\n{r} || rc=$?\n' for i, r in enumerate(runs, start=1)
                ) + "exit $rc\n"
                cmd = ["bash", "-c", script]
            env, cwd = {**base_env, "OPENAI_API_KEY": api_key, "PYTHONUNBUFFERED": "1"}, REPO_ROOT
        else:
            xdist = ["-n", str(workers[suite])] if workers[suite] > 1 else []
            cmd = [sys.executable, "-m", "pytest", SUITES[suite][0], "-p", "no:cacheprovider", "-q", *xdist,
                   *([] if args.include_slow else ["-m", "not slow"]),
                   *shlex.split(args.pytest_args or ""),
                   f"--junitxml={out_dir / f'{suite}.xml'}"]
            env = {**m3_env,
                   "M3_RUN_LOG": str(log_dir / f"{suite}.jsonl"),
                   "M3_STREAM_STATS_LOG": str(log_dir / f"{suite}_stream_stats.jsonl")}
            cwd = PYTEST_DIR
        log = (out_dir / f"{suite}.log").open("w")
        shown = redacted_cmd(cmd[1:] if cmd[0] == sys.executable else cmd, secrets)
        print(f"[run] {suite} (workers={workers[suite]}): {shown}", flush=True)
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs[suite] = (shown, proc, log, time.monotonic())

    runs = {}
    for suite, (shown, proc, log, start) in procs.items():
        code = proc.wait()
        log.close()
        elapsed = round(time.monotonic() - start, 1)
        print(f"[done] {suite}: exit code {code} after {elapsed:.0f}s", flush=True)
        runs[suite] = {"exit_code": code, "workers": workers[suite], "wall_seconds": elapsed, "command": shown}
    return runs


def attach_existing_verify(path: Path, out_dir: Path) -> str:
    """Copy an existing verify.py results file (and its summary, if found) into the run directory."""
    if not path.exists():
        raise SystemExit(f"--verify-results: {path} does not exist")
    shutil.copyfile(path, out_dir / "verify_results.jsonl")
    name = path.name
    summary_name = "summary.json" if name == "results.jsonl" else name.replace("_results.jsonl", "_summary.json")
    summary = path.with_name(summary_name)
    if summary != path and summary.exists():
        shutil.copyfile(summary, out_dir / "verify_summary.json")
    return str(path)


def last_log_line(path: Path) -> str:
    if not path.exists():
        return ""
    lines = [l.strip() for l in path.read_text(errors="replace").splitlines() if l.strip()]
    return lines[-1][:300] if lines else ""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def overall_status(graded: list[SuiteResult]) -> str:
    if any(s.status == "FAIL" for s in graded):
        return "FAIL"
    return "PASS" if any(s.status == "PASS" for s in graded) else "NA"


def summarize(meta: dict, suites: list[SuiteResult]) -> dict:
    graded = [s for s in suites if s.graded]
    failures = [t for s in graded for t in s.failures]
    passed = sum(s.count("passed") for s in graded)
    executed = sum(s.executed for s in graded)
    waived = sum(t.waived for t in failures)
    by_feature: dict[str, list[TestResult]] = {}
    for t in failures:
        by_feature.setdefault(t.feature, []).append(t)
    verify = next((s.verify for s in suites if s.verify), None)
    return {
        "status": overall_status(graded),
        "model": meta["model"],
        "provider": meta["provider"],
        "report_id": meta["report_id"],
        "passed": passed,
        "executed": executed,
        "failed": len(failures),
        "waived": waived,
        "pass_rate": ratio(passed, executed),
        "pass_rate_excl_unsupported": ratio(passed, executed - waived),
        "unsupported_features": meta["unsupported"],
        "unsupported_share_of_failures": ratio(waived, len(failures)),
        "failures_by_feature": {
            f: {"failed": len(ts), "share": len(ts) / len(failures), "unsupported": f in meta["unsupported"]}
            for f, ts in sorted(by_feature.items(), key=lambda kv: -len(kv[1]))
        },
        "suites": {
            s.name: {
                "graded": s.graded,
                "status": s.status,
                "passed": s.count("passed"),
                "failed": s.count("failed"),
                "errors": s.count("error"),
                "skipped": s.count("skipped"),
                "xfailed": s.count("xfailed"),
                "waived": s.waived,
                "pass_rate": s.pass_rate,
                "pass_rate_excl_unsupported": s.pass_rate_excl_unsupported,
                "duration_seconds": round(s.duration, 1),
                "collection_error": s.collection_error or None,
            }
            for s in suites
        },
        "verify_metrics": (
            {k: {"value": m["value"], "threshold": m["threshold"], "status": m["status"]}
             for k, m in verify["metrics"].items()} if verify else None),
        "verify_case_failures": len(verify["cases"]) if verify else None,
    }


def render_failures(lines: list[str], s: SuiteResult, unsupported: list[str]) -> None:
    failing = s.failures
    groups: dict[tuple[str, str], list[TestResult]] = {}
    for t in failing:
        groups.setdefault((classify(t.message), t.feature), []).append(t)
    lines += [
        "",
        "#### Failure causes",
        "",
        "| Cause | Feature | Failed Tests | Share of Failures | Example Test |",
        "|:------|:--------|-------------:|------------------:|:-------------|",
    ]
    for (cause, feature), tests in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        tag = " (unsupported)" if feature in unsupported else ""
        lines.append(f"| {md_escape(cause)} | {FEATURES[feature]}{tag} | {len(tests)} "
                     f"| {len(tests) / len(failing) * 100:.0f}% | `{md_escape(tests[0].name)}` |")
    lines += [
        "",
        f"<details><summary>Failing tests ({len(failing)})</summary>",
        "",
        "| Test | Class | Feature | Cause | Message |",
        "|:-----|:------|:--------|:------|:--------|",
    ]
    for t in failing:
        tag = " (waived)" if t.waived else ""
        cls = t.classname.rpartition(".")[2]
        lines.append(f"| `{md_escape(t.name)}` | `{cls}` | {FEATURES[t.feature]}{tag} "
                     f"| {md_escape(classify(t.message))} | {md_escape(one_line(t.message))} |")
    lines += ["", "</details>"]


def render_not_run(lines: list[str], s: SuiteResult, outcome: str, label: str) -> None:
    tests = [t for t in s.tests if t.outcome == outcome]
    if not tests:
        return
    lines += ["", f"<details><summary>{label} ({len(tests)})</summary>", "",
              "| Test | Reason |", "|:-----|:-------|"]
    for t in tests:
        reason = re.sub(r"^(Skipped|reason): ", "", t.message.strip())
        lines.append(f"| `{md_escape(t.name)}` | {md_escape(one_line(reason, 200))} |")
    lines += ["", "</details>"]


def render_verify(lines: list[str], s: SuiteResult, unsupported: list[str]) -> None:
    v = s.verify
    lines += [
        f"Graded metrics: **{s.count('passed')}/{s.executed} passed**"
        + (f", {s.waived} waived" if s.waived else "")
        + (f", {s.count('skipped')} not computable" if s.count("skipped") else "")
        + f" ({v['n_cases']} cases in the sample set"
        + (f", mean of {v['loops_with_results']}/{v['loops']} runs" if v.get("loops", 1) > 1 else "") + ")",
        "",
        "| Metric | Value | Threshold | Status | Detail |",
        "|:-------|------:|:----------|:-------|:-------|",
    ]
    for m in v["metrics"].values():
        if m["status"] == "NA":
            icon = "🟨 N/A"
        elif m["status"] == "PASS":
            icon = "✅"
        elif m["feature"] in unsupported:
            icon = "❌ (waived)"
        else:
            icon = "❌"
        lines.append(f"| {m['label']} | {pct(m['value'], 2)} | {m['threshold']} | {icon} | {md_escape(m['detail'])} |")

    cases = v["cases"]
    lines += ["", "#### Per-case failures", ""]
    if not cases:
        lines.append("No per-case failures.")
        return
    lines += [
        "Informational: these cases drive the metrics above and are not counted separately in the totals.",
        "",
        "| Cause | Feature | Cases | Share of Case Failures | Example Case |",
        "|:------|:--------|------:|-----------------------:|:-------------|",
    ]
    groups: dict[tuple[str, str], list[dict]] = {}
    for c in cases:
        groups.setdefault((c["cause"], c["feature"]), []).append(c)
    for (cause, feature), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        tag = " (unsupported)" if feature in unsupported else ""
        lines.append(f"| {md_escape(cause)} | {FEATURES[feature]}{tag} | {len(items)} "
                     f"| {len(items) / len(cases) * 100:.0f}% | `#{items[0]['case']}` |")
    lines += [
        "",
        f"<details><summary>Failing cases ({len(cases)})</summary>",
        "",
        "| Case (sample.jsonl line) | Run | expected_tool_call | finish_reason | Cause | Detail |",
        "|:-------------------------|----:|:-------------------|:--------------|:------|:-------|",
    ]
    for c in sorted(cases, key=lambda c: (c["case"] or 0, c.get("loop", 1))):
        lines.append(f"| `#{c['case']}` | {c.get('loop', 1)} | {c['expected_tool_call']} | {c['finish_reason']} "
                     f"| {md_escape(c['cause'])} | {md_escape(c['detail'])} |")
    lines += ["", "</details>"]


def render(meta: dict, suites: list[SuiteResult], summary: dict) -> str:
    graded = [s for s in suites if s.graded]
    unsupported = meta["unsupported"]
    overall = summary["status"]
    n_fail, n_waived = summary["failed"], summary["waived"]

    title = f"{meta['model']} on {meta['provider']}"
    report_meta = {**meta, "tests_passed": summary["passed"], "tests_executed": summary["executed"],
                   "pass_rate": pct(summary["pass_rate"])}
    lines = [
        f"## MiniMax Provider Verifier Test Report: {title}",
        "",
        f"### Metadata: {title}",
        "",
        "```json",
        json.dumps(report_meta, indent=4, ensure_ascii=False),
        "```",
        "",
        "### Acceptance Criteria",
        "",
        f"- Acceptance status: {STATUS_ICON[overall]} `{overall}`",
        f"- Overall pass rate: `{pct(summary['pass_rate'])}` ({summary['passed']}/{summary['executed']} graded "
        "checks passed; each pytest case and each verify.py metric is one check)",
    ]
    if unsupported:
        names = ", ".join(f"`{f}`" for f in unsupported)
        lines += [
            f"- Pass rate excluding unsupported features: `{pct(summary['pass_rate_excl_unsupported'])}` "
            f"({summary['passed']}/{summary['executed'] - n_waived} passed)",
            f"- Unsupported features (waived): {names} — {n_waived}/{n_fail} failures "
            f"(`{pct(summary['unsupported_share_of_failures'])}` of all failures)",
        ]
    for feature, reason in meta.get("auto_waived", {}).items():
        lines.append(f"- `{feature}` waived automatically: {reason}")
    for s in suites:
        if s.collection_error:
            detail = f"did not run: {s.collection_error}"
        else:
            unit = "metrics" if s.name == "verify" else "passed"
            detail = f"{s.count('passed')}/{s.executed} {unit}{' passed' if s.name == 'verify' else ''}, {pct(s.pass_rate)}"
            if s.waived:
                detail += f", {s.waived} waived"
            if s.count("skipped"):
                detail += f", {s.count('skipped')} {'not computable' if s.name == 'verify' else 'skipped'}"
            if s.count("xfailed"):
                detail += f", {s.count('xfailed')} xfailed"
        if not s.graded:
            detail += f"; not graded for {meta['model']}"
        lines.append(f"- `{s.name}`: {STATUS_ICON[s.status]} `{s.status}` ({detail})")
    if overall == "PASS":
        lines.append("- All acceptance criteria passed.")
    else:
        lines.append("- Acceptance criteria not met: every graded test and every verify.py metric must pass "
                     "or be waived as unsupported.")

    lines += [
        "",
        "---",
        "",
        "### Summary",
        "",
        "| Suite | Description | Graded | Passed | Failed | Waived | Skipped | XFailed | Pass Rate "
        "| Pass Rate (excl. unsupported) | Duration (s) | Status |",
        "|:------|:------------|:-------|-------:|-------:|-------:|--------:|--------:|----------:"
        "|------------------------------:|-------------:|:-------|",
    ]
    for s in suites:
        lines.append(
            f"| `{s.name}` | {SUITES.get(s.name, ('', ''))[1]} | {'yes' if s.graded else 'no'} | {s.count('passed')} "
            f"| {s.count('failed') + s.count('error')} | {s.waived} | {s.count('skipped')} | {s.count('xfailed')} "
            f"| {pct(s.pass_rate)} | {pct(s.pass_rate_excl_unsupported)} | {s.duration:.1f} "
            f"| {STATUS_ICON[s.status]} {s.status} |"
        )
    lines.append(
        f"| **Total (graded)** | | | **{summary['passed']}** | **{n_fail}** | **{n_waived}** "
        f"| **{sum(s.count('skipped') for s in graded)}** | **{sum(s.count('xfailed') for s in graded)}** "
        f"| **{pct(summary['pass_rate'])}** | **{pct(summary['pass_rate_excl_unsupported'])}** "
        f"| | {STATUS_ICON[overall]} {overall} |"
    )

    lines += ["", "---", "", "### Failures by Feature", ""]
    if not n_fail:
        lines.append("No failures in graded suites.")
    else:
        lines += [
            f"Share of the {n_fail} failures in graded suites attributed to each API feature "
            "(verify.py contributes one entry per failed metric)."
            + (" Features marked unsupported are waived." if unsupported else ""),
            "",
            "| Feature | Failed Tests | Share of Failures | Suites | Status |",
            "|:--------|-------------:|------------------:|:-------|:-------|",
        ]
        failures = [t for s in graded for t in s.failures]
        for feature, info in summary["failures_by_feature"].items():
            suites_hit = sorted({t.suite for t in failures if t.feature == feature})
            status = "🟨 Unsupported (waived)" if info["unsupported"] else "❌ Failing"
            lines.append(f"| {FEATURES[feature]} | {info['failed']} | {info['share'] * 100:.1f}% "
                         f"| {', '.join(f'`{x}`' for x in suites_hit)} | {status} |")

    for s in suites:
        heading = (f"### Tool-Call Metrics — verify for {title}" if s.name == "verify"
                   else f"### Pytest Verifier — {s.name} for {title}")
        lines += ["", "---", "", heading, ""]
        if not s.graded:
            lines += [f"⚪ Not graded: this suite is reserved for models matching "
                      f"`{MODEL_SCOPED_SUITES.get(s.name, '')}` (MiniMax-M3-a / M3.1). "
                      "Results are informational only.", ""]
        if s.collection_error:
            lines += [f"❌ Suite did not run: `{md_escape(s.collection_error)}`"]
            continue
        if s.name == "verify":
            render_verify(lines, s, unsupported)
            continue
        lines.append(f"Pass rate: **{pct(s.pass_rate)}** ({s.count('passed')}/{s.executed} passed"
                     + (f", {s.waived} waived" if s.waived else "")
                     + (f", {s.count('skipped')} skipped" if s.count("skipped") else "")
                     + (f", {s.count('xfailed')} xfailed" if s.count("xfailed") else "") + ")")
        if not s.failures:
            lines += ["", "All executed tests passed."]
        else:
            render_failures(lines, s, unsupported)
        render_not_run(lines, s, "xfailed", "Expected failures / known bugs (xfail)")
        render_not_run(lines, s, "skipped", "Skipped tests")

    lines += [
        "",
        "---",
        "",
        f"Full failure messages and tracebacks for every test: `verifier_results_{meta['report_id']}.json`.",
        "",
        "Note: pass rates exclude skipped and xfailed tests (xfail marks a known, expected failure and does not "
        "fail acceptance; an unexpected pass of an xfail test counts as passed). Totals exclude suites that are "
        "not graded for this model. Failures attributed to a feature declared unsupported are waived: they are "
        "still listed, but do not fail acceptance. verify.py metrics are recomputed from `verify_results.jsonl` "
        "with the formulas of `scripts/calculate_batch_metrics.py` and graded against the README reference "
        "thresholds; ToolCalls-Trigger-Similarity compares against the first official-deployment run in "
        "`output-dir/<model>/` and is N/A when no baseline exists. The README thresholds are calibrated on the "
        "mean of 10 runs (pass@10); this report grades a single run.",
        "",
    ]
    return "\n".join(lines)


def build_json_report(meta: dict, suites: list[SuiteResult], summary: dict, markdown: str) -> dict:
    """Machine-readable report in the shape of tt-inference-server's report JSON
    (metadata / sections / acceptance_*), with every test's full failure message.

    Deliberately NOT named report_*.json: the exabox merger ingests those and
    this verifier is report-only.
    """
    sections = []
    categories = []
    blockers: dict[str, str] = {}
    for s in suites:
        results = []
        suite_blockers: dict[str, str] = {}
        suite_waived: dict[str, str] = {}
        for t in s.tests:
            entry = {"test": t.name, "classname": t.classname, "outcome": t.outcome}
            if t.outcome != "passed":
                entry["message"] = t.message
                if t.details and t.details != t.message:
                    entry["details"] = t.details
            if t.is_failure:
                cause = classify(t.message)
                entry.update(feature=t.feature, cause=cause, waived=t.waived)
                key = f"{s.name}:{t.name}"
                if t.waived:
                    suite_waived[key] = f"{cause} (unsupported: {t.feature})"
                elif not s.graded:
                    suite_waived[key] = f"{cause} (not graded)"
                else:
                    suite_blockers[key] = cause
            results.append(entry)
        if s.collection_error:
            suite_blockers[f"{s.name}:collection"] = s.collection_error
        if s.graded:
            blockers.update(suite_blockers)
        data = {
            "suite": s.name,
            "description": SUITES.get(s.name, ("", ""))[1],
            "graded": s.graded,
            "status": s.status,
            "passed": s.count("passed"),
            "failed": s.count("failed"),
            "errors": s.count("error"),
            "skipped": s.count("skipped"),
            "xfailed": s.count("xfailed"),
            "waived": s.waived,
            "pass_rate": s.pass_rate,
            "pass_rate_excl_unsupported": s.pass_rate_excl_unsupported,
            "duration_s": round(s.duration, 1),
            "collection_error": s.collection_error or None,
        }
        section = {
            "kind": "vendor_verifier",
            "title": f"Pytest Verifier — {s.name}" if s.name != "verify" else "Tool-Call Metrics — verify.py",
            "task_type": "api_contract",
            "id": s.name,
            "data": data,
            "results": results,
        }
        if s.verify:
            section["verify"] = s.verify
        sections.append(section)
        categories.append({
            "name": s.name,
            "status": s.status,
            "total": s.executed,
            "passed": s.count("passed"),
            "failed": s.count("failed") + s.count("error"),
            "na": s.count("skipped") if s.name == "verify" else 0,
            "skipped": s.count("skipped") + s.count("xfailed"),
            "blockers": suite_blockers if s.graded else {},
            "waived": {**suite_waived, **({} if s.graded else suite_blockers)},
        })

    acceptance_md = markdown.split("### Acceptance Criteria", 1)[-1].split("\n---\n", 1)[0]
    return {
        "metadata": {
            "model_name": meta["model"].split("/")[-1],
            "model_repo": meta["model"],
            "provider": meta["provider"],
            "generated_at": meta["generated_at"],
            "report_id": meta["report_id"],
            "workflow": "vendor_verifier",
            "verifier": "minimax-provider-verifier",
            "report_partial": any(s.collection_error for s in suites),
            "report_blocks": len(sections),
            "server_mode": "API",
            **{k: v for k, v in meta.items() if k not in ("model", "provider", "generated_at", "report_id")},
        },
        "sections": sections,
        "acceptance_criteria": summary["status"] == "PASS",
        "acceptance_blockers": blockers,
        "acceptance_criteria_metadata": {
            "enforcement_result": summary["status"],
            "pass_rate": summary["pass_rate"],
            "pass_rate_excl_unsupported": summary["pass_rate_excl_unsupported"],
            "unsupported_features": meta["unsupported"],
            "failures_by_feature": summary["failures_by_feature"],
            "categories": categories,
        },
        "acceptance_summary_markdown": "### Acceptance Criteria" + acceptance_md.rstrip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("MINIMAX_BASE_URL", ""),
                        help="OpenAI-compatible base URL including /v1 (M3_BASE_URL is derived by stripping /v1)")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", ""))
    parser.add_argument("--provider", default="vendor", help="Provider name shown in the report title")
    parser.add_argument("--suites", nargs="+", choices=list(SUITES), default=list(SUITES))
    parser.add_argument("--workers", nargs="+", default=["1"], metavar="N|SUITE=N",
                        help="pytest-xdist workers (verify: concurrency): one number for every suite, "
                             "and/or SUITE=N overrides")
    parser.add_argument("--unsupported", nargs="*", choices=list(FEATURES), default=None,
                        help="Features the deployment does not support; their failures are waived")
    parser.add_argument("--auth-type", choices=["auto", "bearer", "none"], default="auto",
                        help="bearer: send the API key; none: no Authorization header in the pytest suites "
                             "(auto: bearer if an API key is set, else none)")
    parser.add_argument("--include-slow", action="store_true", help="Also run pytest cases marked slow")
    parser.add_argument("--pytest-args", default="", help='Extra arguments for every pytest suite, e.g. --pytest-args="-k basic" (use the = form)')
    parser.add_argument("--grade-model-scoped", choices=["auto", "always", "never"], default="auto",
                        help="Grade model-scoped suites (reasoning_effort) only for matching models (auto), "
                             "always, or never")
    parser.add_argument("--verify-sample", default=str(DEFAULT_SAMPLE), help="verify.py test set")
    parser.add_argument("--verify-limit", type=int, default=0, help="Only run the first N verify.py cases")
    parser.add_argument("--verify-loops", type=int, default=1,
                        help="Run verify.py N times, one after another, and grade the mean of the metrics "
                             "(the README thresholds are pass@N means; default: 1)")
    parser.add_argument("--verify-results", metavar="RESULTS_JSONL",
                        help="Grade an existing verify.py results file instead of running verify.py")
    parser.add_argument("--verify-baseline", default=None,
                        help="Official results.jsonl for ToolCalls-Trigger-Similarity, or 'none' "
                             "(default: picked from output-dir/ by model name)")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "reports"))
    parser.add_argument("--render-only", metavar="RUN_DIR",
                        help="Rebuild the report from RUN_DIR without running the tests")
    args = parser.parse_args()

    secrets = secrets_from_env()
    if args.render_only:
        out_dir = Path(args.render_only)
        meta = json.loads((out_dir / "run_meta.json").read_text())
        if args.unsupported is not None:
            meta["unsupported"] = args.unsupported
    else:
        if not args.base_url or not args.model:
            parser.error("--base-url and --model are required (or set MINIMAX_BASE_URL / MODEL_NAME)")
        if args.verify_results and "verify" not in args.suites:
            args.suites.append("verify")
        api_key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        auth_type = args.auth_type if args.auth_type != "auto" else ("bearer" if api_key else "none")
        if auth_type == "bearer" and not api_key:
            parser.error("--auth-type bearer needs MINIMAX_API_KEY (or OPENAI_API_KEY) in the environment")
        api_key = api_key or DUMMY_API_KEY
        base_url = args.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            print(f"[warn] --base-url {base_url} does not end in /v1", file=sys.stderr)
        workers = parse_workers(args.workers, args.suites)
        now = datetime.now()
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{args.provider}_{args.model}")
        out_dir = Path(args.output_dir) / f"{slug}_{now:%Y-%m-%d_%H-%M-%S}"
        out_dir.mkdir(parents=True, exist_ok=True)
        baseline = find_baseline(args.model, args.verify_baseline)
        meta = {
            "model": args.model,
            "provider": args.provider,
            "base_url": base_url,
            "m3_base_url": re.sub(r"/v1$", "", base_url),
            "generated_at": f"{now:%Y-%m-%d %H:%M:%S}",
            "report_id": out_dir.name,
            "suites": args.suites,
            "workers": workers,
            "unsupported": args.unsupported or [],
            "auth_type": auth_type,
            "include_slow": args.include_slow,
            "pytest_args": args.pytest_args or None,
            "verify_sample": args.verify_sample,
            "verify_limit": args.verify_limit or None,
            "verify_loops": args.verify_loops,
            "verify_baseline": (str(baseline.relative_to(REPO_ROOT)) if baseline and baseline.is_relative_to(REPO_ROOT)
                                else (str(baseline) if baseline else None)),
            "run_command": redacted_cmd(["python", "test_report.py", *sys.argv[1:]], secrets),
            "verifier_commit": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                capture_output=True, text=True).stdout.strip() or None,
        }
        if args.verify_results:
            meta["verify_results_source"] = attach_existing_verify(Path(args.verify_results), out_dir)
        start = time.monotonic()
        meta["runs"] = run_suites(args, meta, out_dir, workers, api_key, secrets)
        meta["duration_seconds"] = round(time.monotonic() - start, 1)
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    meta.setdefault("unsupported", [])
    meta["auto_waived"] = {}
    if meta.get("auth_type") == "none":
        meta["auto_waived"]["auth"] = ("the endpoint was tested without authentication (M3_AUTH_TYPE=none), "
                                       "so the 401 checks cannot pass")
    meta["unsupported"] = sorted(set(meta["unsupported"]) | set(meta["auto_waived"]))
    meta["ungraded_suites"] = [
        s for s, pattern in MODEL_SCOPED_SUITES.items()
        if s in meta["suites"] and (
            args.grade_model_scoped == "never"
            or (args.grade_model_scoped == "auto" and not model_matches(pattern, meta["model"]))
        )
    ]

    redacted = redact_dir(out_dir, secrets)
    if redacted:
        print(f"[redact] removed secrets from {len(redacted)} file(s): {', '.join(redacted[:10])}")

    suites = []
    for name in meta["suites"]:
        if name == "verify":
            s = parse_verify(out_dir, meta)
            code = meta.get("runs", {}).get("verify", {}).get("exit_code")
            if code not in (None, 0) and not s.collection_error:
                s.collection_error = f"verify.py exited with code {code}: {last_log_line(out_dir / 'verify.log')}"
        else:
            s = parse_junit(out_dir / f"{name}.xml", name)
            code = meta.get("runs", {}).get(name, {}).get("exit_code")
            if code in (2, 3, 4) and not s.tests and not s.collection_error:
                s.collection_error = f"pytest exited with code {code}: {last_log_line(out_dir / f'{name}.log')}"
        s.graded = s.name not in meta["ungraded_suites"]
        for t in s.failures:
            t.feature = attribute_feature(t)
            t.waived = t.feature in meta["unsupported"]
        suites.append(s)

    report_meta = {k: v for k, v in meta.items() if k != "runs"}
    summary = summarize(report_meta, suites)
    report_path = out_dir / f"report_{meta['report_id']}.md"
    markdown = render(report_meta, suites, summary)
    report_path.write_text(markdown)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    json_path = out_dir / f"verifier_results_{meta['report_id']}.json"
    json_path.write_text(json.dumps(build_json_report(report_meta, suites, summary, markdown),
                                    indent=4, ensure_ascii=False) + "\n")
    redact_dir(out_dir, secrets, [report_path, out_dir / "summary.json", json_path])
    print(f"[report] {report_path}")
    print(f"[json] {json_path}")
    print(f"[summary] {out_dir / 'summary.json'}")
    print(f"[status] {summary['status']} pass_rate={pct(summary['pass_rate'])} "
          f"excl_unsupported={pct(summary['pass_rate_excl_unsupported'])}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
