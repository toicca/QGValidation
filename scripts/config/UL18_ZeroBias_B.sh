#!/usr/bin/env bash
OUT_ID="ZeroBias_B"
FILES="${COFFEAHOME}/data/UL18/ZeroBias_B.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel dijet \
    --files $FILES \
    --run data \
    --workers 20
