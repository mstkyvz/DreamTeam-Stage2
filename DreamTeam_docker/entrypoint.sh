#!/usr/bin/env bash
# Dispatch between the two modes the organizer invokes, and nothing else.
#
#   <image> INPUT.csv OUTPUT.csv          -> solution mode   (network disabled)
#   <image> serve-ngrok PUBLIC_TEST.csv   -> explainability  (network for the tunnel only)
#
# Official runs pass `--user <host uid>:<host gid>`, which is an account that does not exist in
# this image's /etc/passwd. Nothing here may assume a resolvable user, a home directory that
# exists, or a writable rootfs: HOME points into the tmpfs and is created here, because several
# libraries call os.path.expanduser() during import and fail on a missing directory.
set -euo pipefail

export HOME="${HOME:-/tmp/home}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME:-/tmp/cache}" "${DT_ARTIFACT_DIR:-/tmp/dt_artifacts}"

if [[ "${1:-}" == "serve-ngrok" ]]; then
  shift
  if [[ -z "${NGROK_AUTHTOKEN:-}" ]]; then
    printf 'NGROK_AUTHTOKEN is required for serve-ngrok mode\n' >&2
    exit 2
  fi

  public_test_csv="${1:?public test CSV path is required}"
  gui_port="${DT_GUI_PORT:-7860}"
  gui_pid=""
  ngrok_pid=""

  cleanup() {
    [[ -n "${ngrok_pid}" ]] && kill "${ngrok_pid}" >/dev/null 2>&1 || true
    [[ -n "${gui_pid}" ]] && kill "${gui_pid}" >/dev/null 2>&1 || true
    wait >/dev/null 2>&1 || true
  }
  trap cleanup EXIT INT TERM

  python3 /app/src/api/server.py "${public_test_csv}" --host 127.0.0.1 --port "${gui_port}" &
  gui_pid=$!

  # The champion engine needs a few minutes to load 30.63 GiB. /health fails closed until the
  # engine is actually up, so this poll cannot start the tunnel in front of a dead model.
  # 160 x 3 s = 8 min, inside the organizer's 10-minute explainability startup limit.
  for _ in $(seq 1 160); do
    kill -0 "${gui_pid}" >/dev/null 2>&1 || {
      printf 'explainability server stopped during startup\n' >&2
      exit 1
    }
    curl --silent --fail "http://127.0.0.1:${gui_port}/health" >/dev/null 2>&1 && break
    sleep 3
  done
  curl --silent --fail "http://127.0.0.1:${gui_port}/health" >/dev/null || {
    printf 'explainability server startup timed out\n' >&2
    exit 1
  }

  # The token reaches ngrok through the environment only: never echoed, never written to a file.
  ngrok http "http://127.0.0.1:${gui_port}" --log stdout --log-format logfmt &
  ngrok_pid=$!

  wait -n "${gui_pid}" "${ngrok_pid}"
  exit $?
fi

exec python3 /app/src/main.py "$@"
