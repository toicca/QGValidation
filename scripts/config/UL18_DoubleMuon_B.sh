#!/usr/bin/env bash
OUT_ID="DoubleMuon_B"
FILES="${COFFEAHOME}/data/UL18/DoubleMuon_B.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel zmm \
    --files $FILES \
    --run data \
    --workers 20