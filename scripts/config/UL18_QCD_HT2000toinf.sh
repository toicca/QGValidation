#!/usr/bin/env bash
OUT_ID="QCD_HT2000toinf"
FILES="${COFFEAHOME}/data/UL18/QCD_HT2000toinf.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel dijet \
    --files $FILES \
    --run mc \
    --workers 20
