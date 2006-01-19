#!/bin/sh
export PYTHONPATH=./root
recon_cmd=./scripts/recon
testdata_dir=../testdata
output_dir=./Images
#dataset=Vari_ss_epi
dataset=asems
#dataset=Ravi_ns22

mkdir -p $output_dir
python $recon_cmd --config=recon.ops $testdata_dir/$dataset.fid ./$output_dir/${dataset}_recon 
