#!/bin/sh
datadir=../../testdata/epi_coronals_s8.dat
export PYTHONPATH=../root
python ../scripts/fdf2img $datadir 
