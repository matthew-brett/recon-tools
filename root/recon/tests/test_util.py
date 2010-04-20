from numpy.testing import *
import numpy as np
from recon import util

def test_find_ramp():
    p_ramp = .025
    L = 128
    sig_rms = 5.0
    def iid_nz(scale):
        return np.random.normal(loc=0, scale=scale, size=L)

    fov_win = np.zeros((L,), 'i')
    fov_win[L/2-40:L/2+40] = 1

    sig_phs = np.arange(L) * p_ramp + np.random.randn(1)
    sig_mag = iid_nz(sig_rms)

    sig = np.abs(sig_mag) * np.exp(1j*sig_phs)

    # attenutate the signal according to a simulated object pattern
    atten = np.zeros(L)
    segs = np.unique(np.random.randint(0, high=10, size=5))
    segs = map(lambda s: int(round(s*(L-1)/10.)), segs)
    segs.sort()
    segs = np.insert(segs, len(segs), L)
    segs = np.insert(segs, 0, 0)
    s_val = [.1, 1]
    vi = 1
    for i in xrange(len(segs)-1):
        atten[segs[i]:segs[i+1]] = s_val[vi]
        vi = vi ^ 1
    sig *= atten
    # now add a small amount of noise to real/imag channels
    sig.real += iid_nz(sig_rms/50.)
    sig.imag += iid_nz(sig_rms/50.)
    
    m = util.find_ramp(np.angle(sig))
    print m

    #return sig
    
    

# a simplified 1D linear regression
def ref_linear_regression(Y, X=None, sigma=None, mask=None):
    if X is None:
        X = np.arange(Y.shape[0])
    if sigma is None:
        sigma = np.ones(Y.shape[0])
    if mask is not None:
        X *= mask
        Y *= mask
    else:
        mask = np.ones(Y.shape[0])
    nzpts = mask.nonzero()[0]
    S = (1./sigma[nzpts]).sum()
    Sx = (X/sigma).sum()
    Sy = (Y/sigma).sum()
    ti = X[nzpts] - (Sx/S)
    Stt = (np.power(ti, 2.0)/sigma[nzpts]).sum()
    ti /= sigma[nzpts]
    m = 1/Stt * (ti*Y[nzpts]).sum()
    b = (Sy - Sx*m)/S

    err = Y - (m*X + b*mask)
    chisq = np.power(err, 2.0).sum()
    return (m, b, chisq)
    

class test_lin_regression(TestCase):
    def setUp(self):
        gshape = (16,12,18)

        # chunk 0 = z-idx; chunk 1 = y-idx; chunk 2 = x-idx
        grad_indices = np.indices(gshape)
        # construct a linear gradient with artificial noise, and
        # a mask based on that noise, and a fake set of variances
        self.abg = (2.6, 0.37, 1.29)
        self.noisy_grad = sum([a*x for a,x in zip(grad_indices,self.abg)])
        noise = np.random.normal(scale=1.5, size=gshape)
        self.gmask = np.zeros(gshape)
        np.putmask(self.gmask, noise <= 1.5, 1)
        assert self.gmask.sum(axis=-1).all(), "some rows masked"
        assert self.gmask.sum(axis=-2).all(), "some cols masked"
        assert self.gmask.sum(axis=-3).all(), "some zdir masked"
        self.noisy_grad += noise
        # fake a variance measure for each point, but enforced positive sign
        self.sigma = np.abs(np.random.normal(loc=.45, scale=.1, size=gshape))


    def doLinRegTests(self, mask=None, sigma=None, axis=-1, test_id='basic'):
        gshape = self.noisy_grad.shape
        coef_shape = list(gshape)
        coef_shape.pop(axis)
        m1d,b1d,chi1d = (np.zeros(coef_shape),
                         np.zeros(coef_shape),
                         np.zeros(coef_shape))

        mnd,bnd,chind = util.lin_regression(self.noisy_grad,
                                            mask=mask,
                                            sigma=sigma,
                                            axis=axis)
        # these return with a None dim if axis is not -1
        mnd,bnd,chind = map(lambda x: np.squeeze(x), (mnd,bnd,chind))

        for m in range(coef_shape[0]):
            for n in range(coef_shape[1]):
                sl = [m,n]
                sl.insert(len(gshape)+axis, slice(0,gshape[axis]))
                if sigma is None:
                    sig = sigma
                else:
                    sig = sigma[sl]
                if mask is None:
                    msk = mask
                else:
                    msk = mask[sl]
                (m1d[m,n],
                 b1d[m,n],
                 chi1d[m,n]) = ref_linear_regression(self.noisy_grad[sl],
                                                     sigma=sig,mask=msk)
        msg = 'failed with ' + test_id
        assert_array_almost_equal(b1d, bnd, decimal=12, err_msg=msg)
        assert_array_almost_equal(m1d, mnd, decimal=12, err_msg=msg)
        assert_array_almost_equal(chi1d, chind, decimal=12, err_msg=msg)

        
    def test_linreg(self, level=1):
        self.doLinRegTests()

    def testLinRegOffAx(self, level=1):
        self.doLinRegTests(axis=-2, test_id='off axis')
        self.doLinRegTests(axis=-3, test_id='off axis')

    def testLinRegMasked(self, level=1):
        self.doLinRegTests(mask=self.gmask, test_id='mask')

    def testLinRegVar(self, level=1):
        self.doLinRegTests(sigma=self.sigma, test_id='variance')

    def testLinRegCombo(self, level=1):
        self.doLinRegTests(mask=self.gmask, sigma=self.sigma, axis=-2,
                           test_id='all')
    
    def testReallyWorks(self, level=1):
        # the solution given masking should be close to the original gradient
        (mx,_,_) = util.lin_regression(self.noisy_grad,
                                       mask=self.gmask, axis=-1)
        (my,_,_) = util.lin_regression(self.noisy_grad,
                                       mask=self.gmask, axis=-2)
        (mz,_,_) = util.lin_regression(self.noisy_grad,
                                       mask=self.gmask, axis=-3)
        assert abs(mx.mean() - self.abg[2]) < .099
        assert abs(my.mean() - self.abg[1]) < .099
        assert abs(mz.mean() - self.abg[0]) < .099


class test_utils(TestCase):

    def test_1dphase(self, level=1):
        x = np.arange(32.)
        phs = np.angle(np.exp(1.j*x))
        uw_phs = util.unwrap1D(phs)
        assert_array_almost_equal(x, uw_phs, decimal=14)
        wr_phs = util.normalize_angle(uw_phs)
        assert_array_almost_equal(wr_phs, phs, decimal=14)

    def test_polyfit(self, level=1):
        x = np.linspace(0, 2.5, num=100, endpoint=False)
        poly_coefs = np.asarray([-10.0, 2.3, 2.85, 1.4, -1.09])
        nzy_poly = poly_coefs[0] + poly_coefs[1]*x
        for n,a in enumerate(poly_coefs[2:]):
            nzy_poly += a*np.power(x, n+2)
        solv_coefs = util.polyfit(x, nzy_poly, len(poly_coefs)-1)[::-1]
        assert_array_almost_equal(poly_coefs, solv_coefs)
        # now add noise
        nzy_poly += np.random.normal(scale=0.8, size=100)
        solv_coefs = util.polyfit(x, nzy_poly, len(poly_coefs)-1)[::-1]
        solv_poly = solv_coefs[0] + solv_coefs[1]*x
        for n,a in enumerate(solv_coefs[2:]):
            solv_poly += a*np.power(x, n+2)
        # these coefficients aren't nearly close!
        # instead try comparing the stdev of the diff to the stdev of the noise
        stdev_d = (nzy_poly - solv_poly).std()
        # is it less than 1.5 times the stdev of the noise?
        assert stdev_d <= 0.8*1.5

    def test_quaternion(self, level=1):
        rot1 = np.array([ [ 1., 0., 0.],
                         [ 0.,-1., 0.],
                         [ 0., 0., 1.], ])
        q1 = util.Quaternion(M=rot1)
        rot2 = np.array([ [ 1., 0., 0.],
                         [ 0., 0., 1.],
                         [ 0.,-1., 0.], ])
        q2 = util.Quaternion(M=rot2)

        q3 = q2*q1
        rot3 = q3.tomatrix()
        assert_array_almost_equal(rot3, np.dot(rot2, rot1))
        
        
if __name__=="__main__":
    NumpyTest().run()
