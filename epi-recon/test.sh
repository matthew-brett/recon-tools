#!/bin/sh
export PYTHONPATH=./root:../varian-tools
recon_cmd=./recon
testdata_dir=../testdata
output_dir=./Images
#dataset=Vari_ss_epi
dataset=Ravi_ns22

mkdir -p $output_dir
python $recon_cmd --config=recon.cfg --file-format=spm --phs-corr=nonlinear \
    $testdata_dir/$dataset.fid ./$output_dir/${dataset}_recon 
