#!/bin/sh
datadir=../../testdata/epi_coronals_s8_post.dat
export PYTHONPATH=../root
python ../scripts/fdf2img $datadir 
