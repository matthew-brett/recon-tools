#!/bin/sh
. ./test-config.sh
#dataset=Vari_ss_epi
#dataset=Bal_phs_corr/asems
#dataset=Ravi_ns22
#dataset=Varian_ns22
#dataset=epi_1sh_lin_64x64
#dataset=Bal_phs_corr/epidw_PN
dataset=Bal_phs_corr/epidw_one_ref
#dataset=mp_flash3d
#dataset=gems_anat
#dataset=SSFP/mp_flash3d_1volume_nsccn
#dataset=SSFP/mp_flash3d_10volumes_nsccn
#dataset=SSFP/mp_flash3d_64x64x32_raw

mkdir -p $output_dir

if [ -f ./$1 ]; then
  python $scripts_dir/recon -p $1 -f 'nifti dual'\
       $testdata_dir/$dataset.fid \
       $output_dir/$dataset.recon
else
  python $scripts_dir/recon -f 'nifti dual'\
       $testdata_dir/$dataset.fid \
       $output_dir/$dataset.recon
fi
