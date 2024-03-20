#!/usr/bin/env bash
OUT_ID="ZeroBias_A"
FILES="${COFFEAHOME}/data/UL18/ZeroBias_A.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel dijet \
    --files $FILES \
    --run data \
    --workers 20
