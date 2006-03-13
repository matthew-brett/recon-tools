#!/bin/sh
. ./test-config.sh
#dataset=Vari_ss_epi
dataset=asems
#dataset=epi_1sh_lin_64x64
#dataset=Ravi_ns22

mkdir -p $output_dir
python $scripts_dir/recon recon.ops $testdata_dir/$dataset.fid $output_dir/${dataset}_recon 
