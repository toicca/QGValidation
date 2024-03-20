#!/usr/bin/env bash
OUT_ID="QCD_HT700to1000"
FILES="${COFFEAHOME}/data/UL18/QCD_HT700to1000.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel dijet \
    --files $FILES \
    --out_dir UL18 \
    --run mc \
    --workers 30 \
    --campaign UL18 \
    --jetvetomaps

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel dijet \
    --files $FILES \
    --out_dir UL18 \
    --run mc \
    --workers 30 \
    --campaign UL18 \
    --jes_up \
    --jetvetomaps

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel dijet \
    --files $FILES \
    --out_dir UL18 \
    --run mc \
    --workers 30 \
    --campaign UL18 \
    --jes_down \
    --jetvetomaps

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel dijet \
    --files $FILES \
    --out_dir UL18 \
    --run mc \
    --workers 30 \
    --campaign UL18 \
    --jer_up \
    --jetvetomaps

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel dijet \
    --files $FILES \
    --out_dir UL18 \
    --run mc \
    --workers 30 \
    --campaign UL18 \
    --jer_down \
    --jetvetomaps
