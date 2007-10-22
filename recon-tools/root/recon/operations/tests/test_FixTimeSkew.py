from recon.operations import FixTimeSkew as FTS
from recon import util
from numpy.testing import *
import numpy as N

class test_interp(NumpyTestCase):

    def setUp(self):
        self.test_sig_flat = N.ones(128)
        self.test_sig_rand_r = N.random.rand(10,128,20)
        self.test_sig_rand_i = N.random.rand(10,128,20) + \
                               1.j*N.random.rand(10,128,20)
        
                                    

    def check_1pixshift_r(self, level=1):
        test_copy1 = self.test_sig_rand_r.copy()
        test_copy2 = test_copy1.copy()
        FTS.subsampInterp(test_copy1, 1.0, axis=-2)
        util.shift(test_copy2, 1, axis=-2)
        assert_array_almost_equal(test_copy2[:,1:-1,:], test_copy1[:,1:-1,:])

    def check_1pixshift_i(self, level=1):
        test_copy1 = self.test_sig_rand_i.copy()
        test_copy2 = test_copy1.copy()
        FTS.subsampInterp(test_copy1, 1.0, axis=-2)
        util.shift(test_copy2, 1, axis=-2)
        assert_array_almost_equal(test_copy2[:,1:-1,:], test_copy1[:,1:-1,:])
    

    def check_flatshift(self, level=1):
        flat_sig = self.test_sig_flat.copy()
        FTS.subsampInterp(flat_sig, .15)
        assert_array_almost_equal(flat_sig, self.test_sig_flat)
    
