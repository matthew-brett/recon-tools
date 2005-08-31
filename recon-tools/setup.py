#!/usr/bin/env python
from distutils.core import setup

setup(
  name='python-recon-tools',
  version='0.1',
  description='Tools for reconstructing K-space MRI scan data.',
  author='Brian Hawthorne',
  author_email='brian.lee.hawthorne@gmail.com',
  url='http://cirl.berkeley.edu/view/BIC/ReconEpi',
  package_dir = {'':'root'},
  packages=['', 'recon', 'recon.lib', 'recon.tools'],
  scripts=['recon-epi']
)
