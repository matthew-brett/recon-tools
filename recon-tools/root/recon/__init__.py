__version__ = "0.5"

from numpy.testing import *
from os import path

def test(level=1):
    mod_file = path.split(path.split(__file__)[0])[-1]
    NumpyTest(mod_file).testall(level=level)
