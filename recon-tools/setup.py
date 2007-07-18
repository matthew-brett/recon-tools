#!/usr/bin/env python
import sys
sys.path = ["./root"]+sys.path
import recon
import recon.scanners.varian
import os, glob
from distutils.core import setup, Extension

# a little help to make setup go smoothly
# grab extra file names to include in varian.tablib package
psfiles = glob.glob(recon.scanners.varian.tablib+'/*')
psfiles = ['tablib/'+os.path.split(file)[1] for file in psfiles]
# with RPM builds, TODOs is already pruned from MANIFEST.in
if 'tablib/TODOs' in psfiles:
    psfiles.remove('tablib/TODOs')
punwrap2_src = glob.glob(os.path.join("src/punwrap2D","*.c"))
punwrap3_src = glob.glob(os.path.join("src/punwrap3D", "*.c"))
fftmod_src = ['src/fftmod/cmplx_fft.c',]
try:
    import numpy
    numpy_include = numpy.get_include()
    from numpy.distutils.system_info import get_info
except ImportError:
    raise ImportError("numpy module is not installed, quitting the setup")    

fftw_info = get_info('fftw3')
if not fftw_info:
    print "you need to install FFTW3 in single and double precision mode"
    sys.exit(1)

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

  Currently developed and maintained by Mike Trumpis <mtrumpis@gmail.com>
  """,


  package_dir = {'':'root'},

  packages=[
    '',
    'recon',
    'recon.conf',
    'recon.operations',
    'recon.punwrap',
    'recon.fftmod',
    'recon.tools',
    'recon.scanners',
    'recon.scanners.varian',],

  scripts=[
    'scripts/dumpheader',
    'scripts/getparam',
    'scripts/fdf2img',
    'scripts/viewimage',
    'scripts/recon'],

  package_data = {'recon.conf':['*.ops'],
                  'recon.scanners.varian': psfiles},
  
  ext_modules=[Extension('recon.punwrap._punwrap2D',
                         punwrap2_src,
                         include_dirs=[numpy_include,]),
               Extension('recon.punwrap._punwrap3D',
                         punwrap3_src,
                         include_dirs=[numpy_include,]),
               Extension('recon.fftmod._fftmod',
                         fftmod_src,
                         include_dirs=[numpy_include,]+fftw_info['include_dirs'],
                         library_dirs=fftw_info['library_dirs'],
                         libraries=['fftw3', 'fftw3f']),
               ],
    
)
