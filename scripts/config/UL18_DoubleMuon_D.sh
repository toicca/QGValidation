#!/usr/bin/env bash
OUT_ID="DoubleMuon_D"
FILES="${COFFEAHOME}/data/UL18/DoubleMuon_D.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel zmm \
    --files $FILES \
    --run data \
    --workers 20
