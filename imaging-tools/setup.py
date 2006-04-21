#!/usr/bin/env python
import sys
sys.path = ["./root"]+sys.path
import imaging
import os, glob
from distutils.core import setup, Extension

setup(

  name='python-imaging-tools',
  version=imaging.__version__,
  description=\
   'Tools for manipulating MRI data, specializing in k-space reconstruction.',
  author='Brian Hawthorne',
  author_email='brian.lee.hawthorne@gmail.com',
  url='http://cirl.berkeley.edu/view/BIC/ImagingTools',

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

  ext_modules=[Extension('imaging.punwrap._punwrap',
               glob.glob(os.path.join('src/punwrap','*.c')))]
  
##   ext_modules=[Extension('imaging.punwrap._punwrap',
##                          ['unwrap_phase.c', 'congruen.c', 'dct.c', \
##                           'dxdygrad.c', 'grad.c', 'laplace.c', 'lpnorm.c', \
##                           'pcg.c', 'raster.c', 'residues.c', 'solncos.c', \
##                           'util.c'])]
  
)
