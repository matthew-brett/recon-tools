from numpy.testing import *
import numpy as N
import os
set_package_path()
from recon.operations import GeometricUndistortionK as GU
from recon.imageio import readImage
restore_path()

class test_GU(NumpyTestCase):

    def setUp(self):
        pwd = os.path.split(__file__)[0]
        kernfile = os.path.join(pwd, 'fmap_based_kernel_long')
        self.ref_kernel = N.fromstring(open(kernfile).read(),
                                       N.complex128)
        self.ref_kernel.shape = (20,64,64,64)
        fmapfile = os.path.join(pwd, "test_fmap.nii")
        fmapIm = readImage(fmapfile)
        self.fmap,self.chi = (fmapIm[0].astype(N.float64),
                              fmapIm[1].astype(N.float64))

    def test_getkernel(self, level=2):
        (ns, M2, M1) = self.fmap.shape
        N2 = M2
        fmap = N.swapaxes(self.fmap, -1, -2)
        chi = N.swapaxes(self.chi, -1, -2)
        b = N.arange(M2)-M2/2
        n2 = N.arange(M2)-M2/2
        Tl = 0.000742
        
        K = GU.get_kernel(M2, Tl, b, n2, fmap, chi)
        assert_array_almost_equal(K, self.ref_kernel)
