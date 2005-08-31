#!/usr/bin/env python
from distutils.core import setup

setup(
  name='python-varian-tools',
  version='0.1',
  description='Tools for reading Varian scan output.',
  author='Brian Hawthorne',
  author_email='brian.lee.hawthorne@gmail.com',
  url='http://cirl.berkeley.edu/view/BIC/VarianTools',
  packages=['varian', 'varian.lib', 'varian.tools'],
  scripts=['varian-dumpheader', 'varian-getparam']
)
