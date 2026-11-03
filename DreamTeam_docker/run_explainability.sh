#!/usr/bin/env bash
set -Eeuo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-example_submission}"
PUBLIC_TEST_CSV_PATH="${1:-}"
readonly GUI_CONTAINER_NAME="$(printf '%s_gui' "${CONTAINER_NAME}" | tr -c '[:alnum:]_.-' '_')"

cleanup() {
  docker rm -f "${GUI_CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if [[ -z "${PUBLIC_TEST_CSV_PATH}" ]]; then
  printf 'usage: CONTAINER_NAME=image %s PUBLIC_TEST_CSV_PATH\n' "$0" >&2
  exit 2
fi
if [[ ! -f "${PUBLIC_TEST_CSV_PATH}" ]]; then
  printf 'public test CSV does not exist: %s\n' "${PUBLIC_TEST_CSV_PATH}" >&2
  exit 2
fi
if [[ -z "${NGROK_AUTHTOKEN:-}" ]]; then
  printf 'NGROK_AUTHTOKEN is required; explainability mode cannot run without ngrok authentication.\n' >&2
  cleanup
  exit 2
fi

readonly INPUT_PATH="$(realpath -- "${PUBLIC_TEST_CSV_PATH}")"
readonly HOST_UID="$(id -u)"
readonly HOST_GID="$(id -g)"

cleanup
printf '[gui] starting explainability UI and ngrok inside %s\n' "${CONTAINER_NAME}"
printf '[gui] ngrok will print the public URL below after model startup\n'

docker run --rm \
  --name "${GUI_CONTAINER_NAME}" \
  --gpus all \
  --network bridge \
  --read-only \
  --user "${HOST_UID}:${HOST_GID}" \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --memory 70g \
  --memory-swap 70g \
  --tmpfs "/tmp:rw,exec,nosuid,nodev,size=8g,uid=${HOST_UID},gid=${HOST_GID},mode=1777" \
  --mount "type=bind,src=${INPUT_PATH},dst=/input/public_test.csv,readonly" \
  --env NGROK_AUTHTOKEN="${NGROK_AUTHTOKEN}" \
  "${CONTAINER_NAME}" \
  serve-ngrok /input/public_test.csv
