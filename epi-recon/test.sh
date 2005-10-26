#!/bin/sh
export PYTHONPATH=./root
recon_cmd=./recon
testdata_dir=../testdata
#dataset=Vari_ss_epi
dataset=Ravi_ns22

python $recon_cmd --config=recon.cfg --file-format=spm --phs-corr=nonlinear \
    $testdata_dir/$dataset.fid/fid \
    $testdata_dir/$dataset.fid/procpar \
    ./Images/${dataset}_recon 
