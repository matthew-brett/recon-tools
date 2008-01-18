import os
from numpy.testing import *
import numpy as N
import sys
set_package_path()
from recon import analyze
from recon.tests.test_nifti import dtypes_test
restore_path()

class test_ana(NumpyTestCase):

    def setUp(self):
        self.pwd = os.path.split(__file__)[0]
        fname = os.path.join(self.pwd, 'gems')
        self.image = analyze.readImage(fname)

    def test_datatypes(self):
        # make a random valued complex image to cast into various dtypes
        image = self.image._subimage(
            4232.*N.random.rand(*self.image.shape) +
            1j*2852.*N.random.rand(*self.image.shape))
        dtypes_test(image, analyze, 'analyze')

    def test_orientation(self):
        msg = "coronal expected, got %s"%self.image.orientation
        assert self.image.orientation == 'coronal', msg
        ana = analyze.readImage(os.path.join(self.pwd, 'MRI_forSEF'))
        msg = "radiological expected, got %s"%ana.orientation
        assert ana.orientation == 'radiological', msg
