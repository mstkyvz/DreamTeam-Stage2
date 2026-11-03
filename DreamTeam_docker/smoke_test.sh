#!/usr/bin/env bash
# End-to-end smoke test with no GPU and no weights: proves the contract path, not the model.
#
# Runs main.py exactly as the judge would -- two positional arguments -- with DT_ENGINE=mock, then
# checks the output against the submission contract. Catches broken imports, path assumptions,
# read-only-filesystem violations and schema drift in about a second.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

cat > "${WORK}/in.csv" <<'CSV'
row_id,query,title,category,brand,gender,age_group,attributes
1,yesil topuklu ayakkabi,"Yesil, ""suet"" topuklu ayakkabi",Ayakkabi,MarkaA,kadin,yetiskin,"renk: yesil, topuk: 8cm"
2,penti plaj cantasi,Bikini alti,Plaj Giyim,Penti,kadin,yetiskin,"renk: mavi"
3,bershka sort,"Geniş yaka top
ikinci satir",Ust Giyim,Bershka,kadin,yetiskin,"renk: beyaz"
007,beyaz sik elbise,Cicek desenli balon kol elbise,Elbise,MarkaB,kadin,yetiskin,"renk: beyaz, desen: cicek"
CSV

# The mock engine short-circuits `build_engine_factories`, so the contract run below never
# touches the real prompt construction. Exercise it explicitly -- that gap once let a missing
# vendored module ship with every test green.
printf '[smoke] building the real champion prompt from the vendored core\n'
PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/src/core/champ" python3 - <<'PY'
import os
from app.config import load_settings
from app.wiring import _shot_block
from core import prompts as PR
src = os.path.join(os.environ['PYTHONPATH'].split(':')[0])
sysmsg = PR.champion_sysmsg(_shot_block(load_settings(src)))
assert len(sysmsg) > 5000, 'champion system message is %d chars' % len(sysmsg)
print('[smoke] champion system message: %d chars' % len(sysmsg))
PY

printf '[smoke] running main.py through the organizer contract\n'
DT_ENGINE=mock DT_ARTIFACT_DIR="${WORK}/artifacts" \
  python3 "${SCRIPT_DIR}/src/main.py" "${WORK}/in.csv" "${WORK}/out.csv"

printf '[smoke] verifying the submission contract\n'
python3 - "${WORK}/in.csv" "${WORK}/out.csv" <<'PY'
import sys, pandas as pd
src, out = (pd.read_csv(p, dtype=str, keep_default_na=False) for p in sys.argv[1:3])
assert list(out.columns) == ['row_id', 'real_label'], out.columns.tolist()
assert len(out) == len(src), (len(out), len(src))
assert list(out['row_id']) == list(src['row_id']), 'row_id order or value changed'
assert '007' in set(out['row_id']), 'leading-zero row_id was renumbered'
assert set(out['real_label']) <= {'irrelevant', 'semi-relevant', 'relevant'}, set(out['real_label'])
print('[smoke] OK: %d rows, columns %s' % (len(out), list(out.columns)))
PY
