#!/usr/bin/env bash
# Run the pytest format-check suites and verify.py in parallel against a provider
# endpoint and write a Markdown report (plus summary.json) via test_report.py.
#
# Safe to call from any directory (e.g. from another repo's CI):
#
#   MINIMAX_API_KEY=sk-... \
#   MINIMAX_BASE_URL=https://api.example.com/v1 \
#   MODEL_NAME=MiniMaxAI/MiniMax-M3 \
#   PROVIDER=Tenstorrent UNSUPPORTED="video" \
#   /path/to/MiniMax-Provider-Verifier/run_verifier.sh
#
# Settings (environment variables; unset ones fall back to <repo>/.env, then defaults):
#   MINIMAX_API_KEY         API key (falls back to OPENAI_API_KEY; never written to the report).
#                           If neither is set, the endpoint is tested without authentication
#                           (M3_AUTH_TYPE=none) and the 401 checks are waived.
#   MINIMAX_BASE_URL        OpenAI-compatible base URL, including /v1 (required)
#   MODEL_NAME              Model id as served by the provider (required)
#   PROVIDER                Provider name for the report title      (default: vendor)
#   UNSUPPORTED             Space-separated features to waive, e.g. "video vision"
#   SUITES                  Space-separated suites to run            (default: text stream verify;
#                           also available: image video reasoning_effort)
#   TEXT_WORKERS            pytest-xdist workers per suite           (default: 20 each)
#   IMAGE_WORKERS
#   VIDEO_WORKERS
#   STREAM_WORKERS
#   REASONING_EFFORT_WORKERS
#   VERIFY_CONCURRENCY      verify.py --concurrency                  (default: 20)
#   VERIFY_LIMIT            Only run the first N sample.jsonl cases  (default: all)
#   VERIFY_LOOPS            Run verify.py N times in a row and grade the mean (pass@N) (default: 1)
#   INCLUDE_SLOW            0 to skip the pytest cases marked slow   (default: 1, run them)
#   RUN_ORDER               sequential: verify.py first, then the pytest suites (default);
#                           parallel: everything at once
#   M3_EXTRA_HEADERS        Optional JSON object of extra request headers (both harnesses)
#   REPORT_DIR              Where run directories are created       (default: <repo>/reports)
#
# Dependencies are installed by uv into a cached environment from requirements.txt and
# m3_format_check/requirements.txt, using PyPI (override with UV_DEFAULT_INDEX). The project's
# pyproject.toml / uv.lock are not used: they pin a mirror that is slow or unreachable outside
# China, and do not include the pytest dependencies.
#
# Extra arguments are passed through to test_report.py.
# Exit code: 0 if acceptance passed, 1 if it failed, 2 on setup errors.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { echo "run_verifier: $*" >&2; exit 2; }

# Load <repo>/.env without overriding variables that are already set.
if [[ -f "$REPO_DIR/.env" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
        [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
        key="${BASH_REMATCH[2]}"
        value="${BASH_REMATCH[3]}"
        value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
        if [[ -z "${!key+x}" ]]; then export "$key=$value"; fi
    done < "$REPO_DIR/.env"
fi

: "${MINIMAX_BASE_URL:?MINIMAX_BASE_URL is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"
PROVIDER="${PROVIDER:-vendor}"
UNSUPPORTED="${UNSUPPORTED:-}"
SUITES="${SUITES:-text stream verify}"
TEXT_WORKERS="${TEXT_WORKERS:-20}"
IMAGE_WORKERS="${IMAGE_WORKERS:-20}"
VIDEO_WORKERS="${VIDEO_WORKERS:-20}"
STREAM_WORKERS="${STREAM_WORKERS:-20}"
REASONING_EFFORT_WORKERS="${REASONING_EFFORT_WORKERS:-20}"
VERIFY_CONCURRENCY="${VERIFY_CONCURRENCY:-20}"
VERIFY_LIMIT="${VERIFY_LIMIT:-0}"
VERIFY_LOOPS="${VERIFY_LOOPS:-1}"
INCLUDE_SLOW="${INCLUDE_SLOW:-1}"
RUN_ORDER="${RUN_ORDER:-sequential}"
REPORT_DIR="${REPORT_DIR:-$REPO_DIR/reports}"

if [[ -z "${MINIMAX_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
    export MINIMAX_API_KEY="$OPENAI_API_KEY"
fi
if [[ -n "${MINIMAX_API_KEY:-}" ]]; then
    export MINIMAX_API_KEY
    auth_type=bearer
else
    echo "run_verifier: no MINIMAX_API_KEY / OPENAI_API_KEY set; testing without authentication" >&2
    auth_type=none
fi

[[ "$MINIMAX_BASE_URL" == */v1 || "$MINIMAX_BASE_URL" == */v1/ ]] || \
    echo "run_verifier: warning: MINIMAX_BASE_URL ($MINIMAX_BASE_URL) does not end in /v1" >&2

command -v uv >/dev/null 2>&1 || die "uv is not installed (https://docs.astral.sh/uv/)"

# Use the verifier's own environment, not whatever venv the caller has active.
unset VIRTUAL_ENV
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.org/simple}"
export UV_PYTHON="${UV_PYTHON:-3.12}"

args=(
    --base-url "$MINIMAX_BASE_URL"
    --model "$MODEL_NAME"
    --provider "$PROVIDER"
    --auth-type "$auth_type"
    --output-dir "$REPORT_DIR"
    --verify-limit "$VERIFY_LIMIT"
    --verify-loops "$VERIFY_LOOPS"
    --order "$RUN_ORDER"
    --workers
        "text=$TEXT_WORKERS"
        "image=$IMAGE_WORKERS"
        "video=$VIDEO_WORKERS"
        "stream=$STREAM_WORKERS"
        "reasoning_effort=$REASONING_EFFORT_WORKERS"
        "verify=$VERIFY_CONCURRENCY"
)
# shellcheck disable=SC2206  # word splitting of the space-separated lists is intended
args+=(--suites $SUITES)
if [[ -n "$UNSUPPORTED" ]]; then
    # shellcheck disable=SC2206
    args+=(--unsupported $UNSUPPORTED)
fi
case "$(printf %s "$INCLUDE_SLOW" | tr "[:upper:]" "[:lower:]")" in
    1|true|yes|on) args+=(--include-slow) ;;
    *)             args+=(--no-include-slow) ;;
esac

cd "$REPO_DIR"
exec uv run --no-project --no-config \
    --with-requirements "$REPO_DIR/requirements.txt" \
    --with-requirements "$REPO_DIR/m3_format_check/requirements.txt" \
    python "$REPO_DIR/test_report.py" "${args[@]}" "$@"
