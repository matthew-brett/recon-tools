from numpy.testing import *
import numpy as N
set_package_path()
from recon import util

restore_path()

# a simplified 1D linear regression
def ref_linear_regression(Y, X=None, sigma=None, mask=None):
    if X is None:
        X = N.arange(Y.shape[0])
    if sigma is None:
        sigma = N.ones(Y.shape[0])
    if mask is not None:
        X *= mask
        Y *= mask
    else:
        mask = N.ones(Y.shape[0])
    nzpts = mask.nonzero()[0]
    S = (1./sigma[nzpts]).sum()
    Sx = (X/sigma).sum()
    Sy = (Y/sigma).sum()
    Sxx = (N.power(X,2.0)/sigma).sum()
    Sxy = (X*Y/sigma).sum()
    delta = S*Sxx - N.power(Sx,2.0)
    b = (Sxx*Sy - Sx*Sxy)/delta
    m = (S*Sxy - Sx*Sy)/delta
    res = abs(Y - (m*X + b)).sum()/Y.shape[0]
    return (b, m, res)
    

class test_linReg(NumpyTestCase):
    def setUp(self):
        gshape = (16,12,18)

        # chunk 0 = z-idx; chunk 1 = y-idx; chunk 2 = x-idx
        grad_indices = N.indices(gshape)
        # construct a linear gradient with artificial noise, and
        # a mask based on that noise, and a fake set of variances
        self.abg = (2.6, 0.37, 1.29)
        self.noisy_grad = sum([a*x for a,x in zip(grad_indices,self.abg)])
        noise = N.random.normal(scale=1.5, size=gshape)
        self.gmask = N.zeros(gshape)
        N.putmask(self.gmask, noise <= 1.5, 1)
        assert self.gmask.sum(axis=-1).all(), "some rows masked"
        assert self.gmask.sum(axis=-2).all(), "some cols masked"
        assert self.gmask.sum(axis=-3).all(), "some zdir masked"
        self.noisy_grad += noise
        # fake a variance measure for each point, but enforced positive sign
        self.sigma = N.abs(N.random.normal(loc=.45, scale=.1, size=gshape))


    def doLinRegTests(self, mask=None, sigma=None, axis=-1):
        gshape = self.noisy_grad.shape
        coef_shape = list(gshape)
        coef_shape.pop(axis)
        b1d,m1d,res1d = (N.zeros(coef_shape),
                         N.zeros(coef_shape),
                         N.zeros(coef_shape))

        bnd,mnd,resnd = util.linReg(self.noisy_grad,mask=mask,sigma=sigma,
                                    axis=axis)
        # these return with a None dim if axis is not -1
        bnd,mnd,resnd = map(lambda x: N.squeeze(x), (bnd,mnd,resnd))

        for m in range(coef_shape[0]):
            for n in range(coef_shape[1]):
                sl = [m,n]
                sl.insert(len(gshape)+axis, slice(0,gshape[axis]))
                sig = sigma if sigma is None else sigma[sl]
                msk = mask if mask is None else mask[sl]
                (b1d[m,n],
                 m1d[m,n],
                 res1d[m,n]) = ref_linear_regression(self.noisy_grad[sl],
                                                     sigma=sig,mask=msk)
                 
        assert_array_almost_equal(b1d, bnd, decimal=12)
        assert_array_almost_equal(m1d, mnd, decimal=12)
        assert_array_almost_equal(res1d, resnd, decimal=12)

        
    def testLinReg(self, level=1):
        self.doLinRegTests()

    def testLinRegOffAx(self, level=1):
        self.doLinRegTests(axis=-2)
        self.doLinRegTests(axis=-3)

    def testLinRegMasked(self, level=1):
        self.doLinRegTests(mask=self.gmask)

    def testLinRegVar(self, level=1):
        self.doLinRegTests(sigma=self.sigma)

    def testLinRegCombo(self, level=1):
        self.doLinRegTests(mask=self.gmask, sigma=self.sigma, axis=-2)
    
    def testReallyWorks(self, level=1):
        # the solution given masking should be close to the original gradient
        (_,mx,_) = util.linReg(self.noisy_grad, mask=self.gmask, axis=-1)
        (_,my,_) = util.linReg(self.noisy_grad, mask=self.gmask, axis=-2)
        (_,mz,_) = util.linReg(self.noisy_grad, mask=self.gmask, axis=-3)
        assert abs(mx.mean() - self.abg[2]) < .099
        assert abs(my.mean() - self.abg[1]) < .099
        assert abs(mz.mean() - self.abg[0]) < .099


class test_utils(NumpyTestCase):

    def test_1dphase(self, level=1):
        x = N.arange(32.)
        phs = N.angle(N.exp(1.j*x))
        uw_phs = util.unwrap1D(phs)
        assert_array_almost_equal(x, uw_phs, decimal=14)
        wr_phs = util.normalize_angle(uw_phs)
        assert_array_almost_equal(wr_phs, phs, decimal=14)

    def test_polyfit(self, level=1):
        x = N.linspace(0, 2.5, num=100, endpoint=False)
        poly_coefs = N.asarray([-10.0, 2.3, 2.85, 1.4, -1.09])
        nzy_poly = poly_coefs[0] + poly_coefs[1]*x
        for n,a in enumerate(poly_coefs[2:]):
            nzy_poly += a*N.power(x, n+2)
        solv_coefs = util.polyfit(x, nzy_poly, len(poly_coefs)-1)[::-1]
        assert_array_almost_equal(poly_coefs, solv_coefs)
        # now add noise
        nzy_poly += N.random.normal(scale=0.8, size=100)
        solv_coefs = util.polyfit(x, nzy_poly, len(poly_coefs)-1)[::-1]
        solv_poly = solv_coefs[0] + solv_coefs[1]*x
        for n,a in enumerate(solv_coefs[2:]):
            solv_poly += a*N.power(x, n+2)
        # these coefficients aren't nearly close!
        # instead try comparing the stdev of the diff to the stdev of the noise
        stdev_d = (nzy_poly - solv_poly).std()
        assert stdev_d <= 0.8*1.5

    def test_quaternion(self, level=1):
        rot1 = N.array([ [ 1., 0., 0.],
                         [ 0.,-1., 0.],
                         [ 0., 0., 1.], ])
        q1 = util.Quaternion(M=rot1)
        rot2 = N.array([ [ 1., 0., 0.],
                         [ 0., 0., 1.],
                         [ 0.,-1., 0.], ])
        q2 = util.Quaternion(M=rot2)

        q3 = q2*q1
        rot3 = q3.tomatrix()
        print rot3
        assert_array_equal(rot3, N.dot(rot2, rot1))
        
        
if __name__=="__main__":
    NumpyTest().run()
