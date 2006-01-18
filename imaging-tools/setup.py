#!/usr/bin/env python
from distutils.core import setup

setup(
  name='python-imaging-tools',
  version='0.1',
  description='Tools for manipulating MRI data, specializing in k-space reconstruction.',
  author='Brian Hawthorne',
  author_email='brian.lee.hawthorne@gmail.com',
  url='http://cirl.berkeley.edu/view/BIC/ImagingTools',
  package_dir = {'':'root'},
  packages=['', 'imaging', 'imaging.operations', 'imaging.tools', 'imaging.varian'],
  scripts=['scripts/dumpheader', 'scripts/getparam', 'scripts/fdf2img', 'scripts/recon']
)
