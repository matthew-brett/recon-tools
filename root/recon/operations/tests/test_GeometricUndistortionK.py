from numpy.testing import *
import numpy as np
import os
import time
from recon.operations import GeometricUndistortionK as GU
from recon.imageio import readImage

def slow_inv(k, lm):
    Kh = k.transpose().conjugate()
    KhK =  (lm**2) * np.identity(k.shape[1], k.dtype) + np.dot(Kh,k)
    # for the system (KhK + (lm**2)I)x = Kh
    # x is the regularized, generalized inverse of K
    return np.linalg.solve(KhK,Kh)
    

class test_GU(TestCase):

    def setUp(self):
        pwd = os.path.split(__file__)[0]
        kernfile = os.path.join(pwd, 'fmap_based_kernel_long')
        self.ref_kernel = np.fromstring(open(kernfile, 'rb').read(),
                                       np.complex128)
        self.ref_kernel.shape = (20,64,64,64)
        fmapfile = os.path.join(pwd, "test_fmap.nii")
        fmapIm = readImage(fmapfile)
        self.fmap,self.chi = (fmapIm[0].astype(np.float64),
                              fmapIm[1].astype(np.float64))
    
    def test_getkernel(self):
        (ns, M2, M1) = self.fmap.shape
        N2 = M2
        fmap = np.swapaxes(self.fmap, -1, -2)
        chi = np.swapaxes(self.chi, -1, -2)
        b = np.arange(M2)-M2/2
        n2 = np.arange(M2)-M2/2
        Tl = 0.000742
        
        K = GU.get_kernel(M2, Tl, b, n2, fmap, chi)
        assert_array_almost_equal(K, self.ref_kernel)

    def test_reginv(self):
        # make sure the fancy code is equal to the more literal, but slow way
        k = self.ref_kernel[10,20].copy()
        lm = 8.
        x = slow_inv(k, lm)
        k = self.ref_kernel[10,20].copy()        
        xf = GU.regularized_inverse(k, lm)
        assert_array_almost_equal(x, xf)

    @dec.slow
    def test_invspeed(self):
        nsl, nfe = self.ref_kernel.shape[:2]
        lm = 8.0
        slcs = range(nsl)
        cols = range(nfe)
        t0_slow = time.time()
        for s in slcs:
            for fe in cols:
                ik = slow_inv(self.ref_kernel[s,fe], lm)
        tf_slow = time.time()
        
        t0_fast = time.time()
        for s in slcs:
            for fe in cols:
                ik = GU.regularized_inverse(self.ref_kernel[s,fe], lm)
        tf_fast = time.time()
        print "literal numpy solution for %d inverses: %2.4fs"%(nsl*nfe,
                                                                tf_slow-t0_slow)
        print "slick solution for %d inverses: %2.4fs"%(nsl*nfe,
                                                              tf_fast-t0_fast)
        print "speedup:", (tf_slow-t0_slow)/(tf_fast-t0_fast)
