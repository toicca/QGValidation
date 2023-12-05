#!/usr/bin/env bash
OUT_ID="QCD_HT1000to1500"
FILES="${COFFEAHOME}/data/UL18/QCD_HT1000to1500.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel dijet \
    --files $FILES \
    --run mc \
    --workers 20
