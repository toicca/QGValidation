#!/usr/bin/env bash
OUT_ID="QCD_HT1500to2000"
FILES="${COFFEAHOME}/data/UL18/QCD_HT1500to2000.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel dijet \
    --files $FILES \
    --run mc \
    --workers 20
