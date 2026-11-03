#!/usr/bin/env bash
# Vendor the checkpoint into the build context. Run ONCE, before build_docker.sh.
#
# This is separate from the build on purpose. Build-time internet access is permitted, but a
# build that downloads 32 GB is a build that can fail at minute 55 of a 60-minute limit and take
# the submission with it. Downloading first makes the build a local COPY: deterministic, offline,
# and re-runnable.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# One checkpoint. The cascade gate is deliberately NOT fetched: it is a wall-clock optimisation
# that costs accuracy (0.8903 at 15% routing, 0.8935 at 20%, against champion-only's 0.9131), and
# the organizer's 150-minute limit leaves nothing to buy with the time it saves.
readonly REPOS=(
  "champ_g31int8:lokeshe09/gemma-4-31B-it-INT8"
)

TOKEN="${HF_TOKEN:-${HF_AUTH_KEY:-}}"
if [[ -z "${TOKEN}" ]]; then
  for f in "${HF_HOME:-${HOME}/.cache/huggingface}/token" "${HOME}/.huggingface/token"; do
    [[ -s "${f}" ]] && { TOKEN="$(tr -d '\r\n' < "${f}")"; break; }
  done
fi

for entry in "${REPOS[@]}"; do
  name="${entry%%:*}"
  repo="${entry#*:}"
  dest="${SCRIPT_DIR}/models/${name}"

  if [[ -f "${dest}/config.json" ]]; then
    printf '[fetch] already present: %s\n' "${dest}"
    continue
  fi

  mkdir -p "${dest}"
  printf '[fetch] %s -> %s\n' "${repo}" "${dest}"
  # `hf` is not guaranteed to exist on a bare build worker, so go through the library directly and
  # install it only if it is missing. snapshot_download resumes, which matters for 31 GB over a
  # link that may drop.
  HF_TOKEN="${TOKEN}" python3 - "${repo}" "${dest}" <<'PY'
import sys, subprocess
try:
    from huggingface_hub import snapshot_download
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'huggingface_hub'])
    from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], max_workers=8)
PY

  # The runtime never downloads, so anything missing here is missing forever.
  for required in config.json tokenizer.json; do
    [[ -f "${dest}/${required}" ]] || { printf '[fetch] MISSING %s in %s\n' "${required}" "${dest}" >&2; exit 1; }
  done
  printf '[fetch] done: %s, %s, %s shards\n' "${name}" "$(du -sh "${dest}" | cut -f1)" \
         "$(find "${dest}" -name '*.safetensors' | wc -l | tr -d ' ')"
done
