#!/usr/bin/env bash
OUT_ID="UL18_DYJetsToLL_M-50_pythia8"
FILES="${COFFEAHOME}/data/UL18/DYJetsToLL_M-50_pythia8.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel zmm \
    --files $FILES \
    --campaign UL18 \
    --run mc \
    --workers 30 \
    --jetvetomaps

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel zmm \
    --files $FILES \
    --out_dir UL18 \
    --run mc \
    --workers 30 \
    --campaign UL18 \
    --jes_up \
    --jetvetomaps

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel zmm \
    --files $FILES \
    --out_dir UL18 \
    --run mc \
    --workers 30 \
    --campaign UL18 \
    --jes_down \
    --jetvetomaps

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel zmm \
    --files $FILES \
    --out_dir UL18 \
    --run mc \
    --workers 30 \
    --campaign UL18 \
    --jer_up \
    --jetvetomaps

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel zmm \
    --files $FILES \
    --out_dir UL18 \
    --run mc \
    --workers 30 \
    --campaign UL18 \
    --jer_down \
    --jetvetomaps

