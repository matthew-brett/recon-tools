#!/bin/sh
. ./test-config.sh
dataset=$testdata_dir/epi_coronals_s8.dat
python ../scripts/fdf2img $dataset
