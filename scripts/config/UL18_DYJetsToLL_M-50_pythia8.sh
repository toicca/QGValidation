#!/usr/bin/env bash
OUT_ID="UL18_DYJetsToLL_M-50_pythia8"
FILES="${COFFEAHOME}/data/UL18/UL18_DYJetsToLL_M-50_pythia8.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel zmm \
    --files $FILES \
    --run mc \
    --workers 20
