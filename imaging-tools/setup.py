#!/usr/bin/env python
import sys
sys.path = ["./root"]+sys.path
import imaging
import os, glob
from distutils.core import setup, Extension

# a little help to make setup go smoothly
psfiles = glob.glob('root/imaging/varian/tablib/*')
psfiles.remove('root/imaging/varian/tablib/TODOs')
psfiles = ['%s'%(file[20:]) for file in psfiles]

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

  package_data = {'imaging.conf':['*.ops'],
                  'imaging.varian': psfiles},
  
  ext_modules=[Extension('imaging.punwrap._punwrap',
               glob.glob(os.path.join('src/punwrap','*.c')))],
    
)
