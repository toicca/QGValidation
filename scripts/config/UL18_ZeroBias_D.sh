#!/usr/bin/env bash
OUT_ID="ZeroBias_D"
FILES="${COFFEAHOME}/data/UL18/ZeroBias_D.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel dijet \
    --files $FILES \
    --run data \
    --campaign UL18 \
    --out_dir UL18 \
    --workers 30 \
    --jetvetomaps
