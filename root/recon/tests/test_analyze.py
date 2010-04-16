import os
from numpy.testing import *
import numpy as np
import sys
from recon import analyze
from recon.tests.test_nifti import dtypes_test
from recon.operations.FlipSlices import FlipSlices

class test_ana(TestCase):

    def setUp(self):
        self.pwd = os.path.split(__file__)[0]
        fname = os.path.join(self.pwd, 'gems')
        self.image = analyze.readImage(fname)

    def test_datatypes(self):
        # make a random valued complex image to cast into various dtypes
        image = self.image._subimage(
            4232.*np.random.rand(*self.image.shape) +
            1j*2852.*np.random.rand(*self.image.shape))
        dtypes_test(image, analyze, 'analyze')

    def test_orientation(self):
        xform = self.image.orientation_xform.tomatrix()
        oname = analyze.canonical_orient(xform)
        msg = 'coronal expected, got %s'%oname
        assert oname == 'coronal', msg
        ana = analyze.readImage(os.path.join(self.pwd, 'MRI_forSEF'))
        xform = ana.orientation_xform.tomatrix()
        oname = analyze.canonical_orient(xform)
        msg = 'radiological expected, got %s'%oname
        assert oname == 'radiological', msg
        FlipSlices(flipud=True).run(ana)
        xform = ana.orientation_xform.tomatrix()
        oname = analyze.canonical_orient(xform)
        msg = 'transverse exepcted, got %s'%oname
        assert oname == 'transverse', msg
        FlipSlices(fliplr=True).run(ana)
        xform = ana.orientation_xform.tomatrix()
        oname = analyze.canonical_orient(xform)
        msg = 'unknown transform expected, got %s'%oname
        assert oname == '', msg
