#!/usr/bin/env bash
# Build the submission image from this directory and nothing else.
#
# DELIBERATELY ABSENT: global Docker cleanup. The organizer's reference build script runs
# `docker image prune --all`, `docker volume prune --all` and friends, but the rules state that
# "a team script that attempts host administration, global Docker cleanup, or access outside its
# build context is grounds for rejection EVEN IF IT APPEARS IN A REFERENCE IMPLEMENTATION."
# Builds run on a disposable isolated worker that the organizer controls; cleaning it is not ours
# to do, and doing it would destroy other teams' cached layers on a shared daemon.
#
# ALSO DELIBERATELY ABSENT: a build secret. The checkpoint is vendored into the build context by
# `fetch_models.sh` before `docker build` starts, so the build itself performs no authenticated
# download and there is no token to leak into a layer, an env var, or the image history.
set -Eeuo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-DreamTeam_solution}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_DIRS=("${SCRIPT_DIR}/models/champ_g31int8")

if ! command -v docker >/dev/null 2>&1; then
  printf 'docker is not installed or not on PATH\n' >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  printf 'cannot query the Docker daemon; verify that Docker is running\n' >&2
  exit 2
fi

# The archive may ship with the weights already vendored, or without them. Both are permitted;
# this script handles both so the judge only ever runs ONE command.
#   vendored  -> the build is a pure local COPY: offline, deterministic, no secret, fastest.
#   absent    -> fetch them here, once, before `docker build` starts. Deliberately NOT inside the
#                Dockerfile: a 32 GB download in a RUN layer is a build that can die at minute 55
#                of a 60-minute limit and take the submission with it, and it re-downloads on
#                every cache miss. Out here it is resumable and cached across rebuilds.
if [[ ! -f "${MODEL_DIRS[0]}/config.json" ]]; then
  printf '[build] weights are not vendored in this archive; fetching them first (~32 GB)\n'
  printf '[build] this needs HF_TOKEN (or HF_AUTH_KEY) and network access\n'
  "${SCRIPT_DIR}/fetch_models.sh"
fi

# Fail before a 40-minute build rather than after it: an image whose weights are missing builds
# fine and then fails at inference time on a network-disabled judging run.
for model_dir in "${MODEL_DIRS[@]}"; do
  if [[ ! -f "${model_dir}/config.json" ]]; then
    printf 'model weights are not in the build context: %s\n' "${model_dir}" >&2
    printf 'fetch_models.sh did not produce them; check HF_TOKEN and network access.\n' >&2
    exit 2
  fi
  shards="$(find "${model_dir}" -name '*.safetensors' | wc -l | tr -d ' ')"
  if [[ "${shards}" -eq 0 ]]; then
    printf 'no .safetensors shards under %s\n' "${model_dir}" >&2
    exit 2
  fi
done

printf '[build] context: %s\n' "${SCRIPT_DIR}"
printf '[build] image:   %s\n' "${CONTAINER_NAME}"
for model_dir in "${MODEL_DIRS[@]}"; do
  printf '[build] model:   %s (%s shards, %s)\n' "$(basename "${model_dir}")" \
         "$(find "${model_dir}" -name '*.safetensors' | wc -l | tr -d ' ')" \
         "$(du -sh "${model_dir}" | cut -f1)"
done

export DOCKER_BUILDKIT=1
docker build --progress=plain --tag "${CONTAINER_NAME}" "${SCRIPT_DIR}"

printf '[build] built %s\n' "${CONTAINER_NAME}"
docker image inspect "${CONTAINER_NAME}" \
  --format '[build] image id={{.Id}} size={{.Size}} bytes'

# Prove the checkpoint is in the image and readable as a non-root uid, which is how judging
# runs. A file that is present but unreadable to an arbitrary uid fails only at inference time.
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python3 "${CONTAINER_NAME}" -c "
import glob, os
for p in ('/models/champ_g31int8',):
    assert os.path.isfile(p + '/config.json'), 'missing ' + p
    open(p + '/config.json').close()          # readable as THIS uid, not just present
    print('[build] weights present:', p, len(glob.glob(p + '/*.safetensors')), 'shards')
"

# Prove the contract path runs end to end inside the image, with no GPU and no network.
printf '[build] running the in-image smoke test\n'
docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,exec,size=1g --entrypoint bash "${CONTAINER_NAME}" -c '
set -e
export HOME=/tmp/home DT_ENGINE=mock DT_ARTIFACT_DIR=/tmp/art
mkdir -p "$HOME"
printf "row_id,query,title,category,brand,gender,age_group,attributes\n1,a,b,c,d,e,f,g\n" > /tmp/in.csv
python3 /app/src/main.py /tmp/in.csv /tmp/out.csv
head -2 /tmp/out.csv'
