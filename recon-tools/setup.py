#!/usr/bin/env python
import sys
sys.path = ["./root"]+sys.path
import recon
import recon.varian
import os, glob
from distutils.core import setup, Extension

# a little help to make setup go smoothly
# grab extra file names to include in varian.tablib package
psfiles = glob.glob(recon.varian.tablib+'/*')
psfiles = ['tablib/'+os.path.split(file)[1] for file in psfiles]
# with RPM builds, TODOs is already pruned from MANIFEST.in
if 'tablib/TODOs' in psfiles:
    psfiles.remove('tablib/TODOs')

#print psfiles
punwrap_src = glob.glob(os.path.join("src/punwrap","*.c"))

setup(

  name='python-recon-tools',
  version=recon.__version__,
  description=\
   'Tools for manipulating MRI data, specializing in k-space reconstruction.',
  author='Mike Trumpis',
  author_email='mtrumpis@gmail.com',
  url='http://cirl.berkeley.edu/view/BIC/ReconTools',

  long_description = """
  An all-in-one tool for reconstructing MR images from sampled k-space data.
  recon-tools can reconstruct EPI, GEMS, and MPFLASH3D data, perform a
  series of artifact reduction operations, and write the image data to
  Analyze or NIFTI file formats.

  Originally developed by Brian Hawthorne <brian.lee.hawthorne@gmail.com>

  Currently developed and maintained by Mike Trumpis <mtrumpis@gmail.com>
  """,


  package_dir = {'':'root'},

  packages=[
    '',
    'recon',
    'recon.conf',
    'recon.operations',
    'recon.punwrap',
    'recon.tools',
    'recon.varian'],

  scripts=[
    'scripts/dumpheader',
    'scripts/getparam',
    'scripts/fdf2img',
    'scripts/viewimage',
    'scripts/recon'],

  package_data = {'recon.conf':['*.ops'],
                  'recon.varian': psfiles},
  
  ext_modules=[Extension('recon.punwrap._punwrap',
               punwrap_src)],
    
)
