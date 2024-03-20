#!/usr/bin/env bash
conda env create -f coffeavenv.yml
conda activate coffeavenv

export COFFEAHOME=$PWD
export COFFEADATA=$PWD/data
export PATH=$HOME/bin:$PATH
