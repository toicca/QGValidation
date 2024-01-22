#!/usr/bin/env bash
OUT_ID="QCD_HT100to200"
FILES="${COFFEAHOME}/data/UL18/QCD_HT100to200.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel dijet \
    --files $FILES \
    --run mc \
    --workers 20
