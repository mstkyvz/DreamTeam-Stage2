#!/usr/bin/env bash
set -Eeuo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-example_submission}"
INPUT_CSV_PATH="${1:-}"
OUTPUT_CSV_PATH="${2:-}"

if [[ -z "${INPUT_CSV_PATH}" || -z "${OUTPUT_CSV_PATH}" ]]; then
  printf 'usage: CONTAINER_NAME=image %s INPUT_CSV_PATH OUTPUT_CSV_PATH\n' "$0" >&2
  exit 2
fi
if [[ ! -f "${INPUT_CSV_PATH}" ]]; then
  printf 'input CSV does not exist: %s\n' "${INPUT_CSV_PATH}" >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  printf 'docker is not installed or not on PATH\n' >&2
  exit 2
fi

readonly RESOLVED_INPUT_CSV_PATH="$(realpath -- "${INPUT_CSV_PATH}")"
readonly HOST_UID="$(id -u)"
readonly HOST_GID="$(id -g)"
OUTPUT_PARENT="$(dirname -- "${OUTPUT_CSV_PATH}")"
mkdir -p "${OUTPUT_PARENT}"
OUTPUT_PARENT="$(realpath -- "${OUTPUT_PARENT}")"
OUTPUT_NAME="$(basename -- "${OUTPUT_CSV_PATH}")"

if [[ "${OUTPUT_NAME}" == "." || "${OUTPUT_NAME}" == ".." || "${OUTPUT_NAME}" == */* ]]; then
  printf 'invalid output filename: %s\n' "${OUTPUT_NAME}" >&2
  exit 2
fi

printf '[docker-run] image: %s\n' "${CONTAINER_NAME}"
printf '[docker-run] input: %s\n' "${RESOLVED_INPUT_CSV_PATH}"
printf '[docker-run] output: %s/%s\n' "${OUTPUT_PARENT}" "${OUTPUT_NAME}"
printf '[docker-run] network: disabled\n'
printf '[docker-run] GPUs: all visible GPUs; exclusive access must be enforced by the host scheduler\n'
printf '[docker-run] user: host uid=%s gid=%s\n' "${HOST_UID}" "${HOST_GID}"

docker run --rm \
  --gpus all \
  --network none \
  --read-only \
  --user "${HOST_UID}:${HOST_GID}" \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --memory 70g \
  --memory-swap 70g \
  --tmpfs "/tmp:rw,exec,nosuid,nodev,size=8g,uid=${HOST_UID},gid=${HOST_GID},mode=1777" \
  --mount "type=bind,src=${RESOLVED_INPUT_CSV_PATH},dst=/input/input.csv,readonly" \
  --mount "type=bind,src=${OUTPUT_PARENT},dst=/output" \
  "${CONTAINER_NAME}" \
  /input/input.csv "/output/${OUTPUT_NAME}"

printf '[docker-run] wrote %s/%s\n' "${OUTPUT_PARENT}" "${OUTPUT_NAME}"
