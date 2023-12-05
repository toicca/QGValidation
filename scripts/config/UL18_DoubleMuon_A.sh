#!/usr/bin/env bash
OUT_ID="DoubleMuon_A"
FILES="${COFFEAHOME}/data/UL18/DoubleMuon_A.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel zmm \
    --files $FILES \
    --run data \
    --workers 20
