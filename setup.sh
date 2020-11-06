#!/usr/bin/env bash
export COFFEAHOME=$PWD
export COFFEADATA=$PWD/data
source /cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh
#source /cvmfs/sft.cern.ch/lcg/views/LCG_97python3/x86_64-centos7-gcc8-opt/setup.sh # to have an updated roofit
export PATH=$HOME/bin:$PATH
source activate COFFEA
