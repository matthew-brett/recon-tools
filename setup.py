#!/usr/bin/env python
import sys
sys.path = ["./root"]+sys.path
import recon
import recon.scanners
import os, glob
from distutils.core import setup, Extension

### A little help to make setup go smoothly ....

# grab extra file names to include in tablib directory
psfiles = glob.glob(recon.scanners.tablib+'/*')
psfiles = ['tablib/'+os.path.split(fname)[1] for fname in psfiles]
# with RPM builds, TODOs is already pruned from MANIFEST.in
if 'tablib/TODOs' in psfiles:
    psfiles.remove('tablib/TODOs')

# grab the Extension modules source files
punwrap2_src = glob.glob("src/punwrap2D/*.c")
punwrap3_src = glob.glob("src/punwrap3D/*.c")

# get the Numpy include directory and get_info utility
try:
    import numpy
    numpy_include = numpy.get_include()
    from numpy.distutils.system_info import get_info
except ImportError:
    raise ImportError("""
    The numpy package is needed to compile the recon tools extensions but
    is not installed. Quitting the setup.
    """)

# find out if FFTW3 is installed
fftw_info = get_info('fftw3')
if sys.platform!='win32' and not fftw_info:
    raise Exception("""
    The FFTW3 libraries compiled in single and double precision are needed
    to compile the recon tools extensions, but are not installed. Quitting
    the setup.
    """)

# if on OS X, define DARWIN preprocessor toggle
macros = []
if sys.platform.find("darwin") > -1:
    macros.append( ('DARWIN', None) )

ext_modules=[Extension('recon.punwrap._punwrap2D',
                       punwrap2_src,
                       include_dirs=[numpy_include,],
                       libraries=['m'],
                       define_macros=macros),
             Extension('recon.punwrap._punwrap3D',
                       punwrap3_src,
                       include_dirs=[numpy_include,],
                       libraries=['m']),
               ]

# beef up the extensions list with weave generated code
ext_modules += recon.find_extensions()


### The setup script ....
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
    'recon.tests',
    'recon.conf',
    'recon.operations',
    'recon.operations.tests',
    'recon.punwrap',
    'recon.punwrap.tests',
    'recon.fftmod',
    'recon.fftmod.tests',
    'recon.pmri',
    'recon.tools',
    'recon.tools.tests',
    'recon.scanners',
    'recon.visualization',],

  scripts=[
    'scripts/recon',
    'scripts/recon_gui'],
  
  package_data = {'recon.conf':['*.ops'],
                  'recon.fftmod' : ['src/*.c'],
                  'recon.scanners': psfiles},
  
  ext_modules=ext_modules
)
