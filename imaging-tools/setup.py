#!/usr/bin/env python
import sys
sys.path = ["./root"]+sys.path
import imaging
import imaging.varian
import os, glob
from distutils.core import setup, Extension

# a little help to make setup go smoothly
# grab extra file names to include in varian.tablib package
psfiles = glob.glob(imaging.varian.tablib+'/*')
psfiles = ['tablib/'+os.path.split(file)[1] for file in psfiles]
# with RPM builds, TODOs is already pruned from MANIFEST.in
if 'tablib/TODOs' in psfiles:
    psfiles.remove('tablib/TODOs')

print psfiles
punwrap_src = glob.glob(os.path.join("src/punwrap","*.c"))

setup(

  name='python-imaging-tools',
  version=imaging.__version__,
  description=\
   'Tools for manipulating MRI data, specializing in k-space reconstruction.',
  author='Mike Trumpis',
  author_email='mtrumpis@gmail.com',
  url='http://cirl.berkeley.edu/view/BIC/ImagingTools',

  long_description = """
  An all-in-one tool for reconstructing MR images from sampled k-space data.
  imaging-tools can reconstruct EPI, GEMS, and MPFLASH3D data, perform a
  series of artifact reduction operations, and write the image data to
  Analyze or NIFTI file formats.

  Originally developed by Brian Hawthorne <brian.lee.hawthorne@gmail.com>

  Currently developed and maintained by Mike Trumpis <mtrumpis@gmail.com>
  """,


  package_dir = {'':'root'},

  packages=[
    '',
    'imaging',
    'imaging.conf',
    'imaging.operations',
    'imaging.punwrap',
    'imaging.tools',
    'imaging.varian'],

  scripts=[
    'scripts/dumpheader',
    'scripts/getparam',
    'scripts/fdf2img',
    'scripts/viewimage',
    'scripts/recon'],

  package_data = {'imaging.conf':['*.ops'],
                  'imaging.varian': psfiles},
  
  ext_modules=[Extension('imaging.punwrap._punwrap',
               punwrap_src)],
    
)
