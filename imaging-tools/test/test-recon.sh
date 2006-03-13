#!/bin/sh
. ./test-config.sh
#dataset=Vari_ss_epi
#dataset=asems
#dataset=Ravi_ns22
#dataset=Varian_ns22
#dataset=epi_1sh_lin_64x64
#dataset=Bal_phs_corr2/epidw_PN
#dataset=mp_flash3d
#dataset=gems_anat
dataset=SSFP/mp_flash3d_1volume_nsccn
#dataset=SSFP/mp_flash3d_10volumes_nsccn
#dataset=SSFP/mp_flash3d_64x64x32_raw

mkdir -p $output_dir
sed "s|filename=.*|filename=$output_dir\/$dataset\_recon|" recon.ops > recon.ops.out
mv recon.ops.out recon.ops
python $scripts_dir/recon recon.ops $testdata_dir/$dataset.fid  
