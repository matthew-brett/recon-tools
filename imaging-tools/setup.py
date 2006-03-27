#!/usr/bin/env python
import sys
sys.path = ["./root"]+sys.path
import imaging
from distutils.core import setup

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
    'imaging.operations',
    'imaging.tools',
    'imaging.varian'],

  scripts=[
    'scripts/dumpheader',
    'scripts/getparam',
    'scripts/fdf2img',
    'scripts/viewimage',
    'scripts/recon']
)
