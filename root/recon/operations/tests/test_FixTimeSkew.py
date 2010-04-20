from recon.operations import FixTimeSkew as FTS
from recon import util, visual
from numpy.testing import *
import numpy as np
## import matplotlib.pyplot as pp
## import matplotlib.mlab as mlab
import scipy as sp
from scipy import integrate

def upsample(a, m):
    npts = a.shape[0]*m
    b = a.copy()
    # upsampe m times, stick the original ts in the middle of m samples
    a_up = np.zeros((npts,), a.dtype)
    a_up[m/2::m] = a
    util.fft1(a_up, inplace=True, shift=True)
    # LP filter to get sinc interpolation
    lpf = np.zeros((npts,))
    bw = a.shape[0]
    lpf[npts/2-bw/2 : npts/2+bw/2] = 1.
    a_up *= lpf
    util.ifft1(a_up, inplace=True, shift=True)
    a_up *= float(m)
    return a_up
    

## def generate_distrubtions(epi=None, N=100):
##     """Find the pdf(rms_err) and pdf(med_err)
##     """
##     pdf_rms_err_real_std = []
##     pdf_rms_err_imag_std = []
##     pdf_rms_err_real_tst = []
##     pdf_rms_err_imag_tst = []
##     pdf_med_err_real_std = []
##     pdf_med_err_imag_std = []
##     pdf_med_err_real_tst = []
##     pdf_med_err_imag_tst = []
##     pt_cache = {}
##     def random_slice(epi):
##         ns, nr, nc = epi.shape[-3:]
##         s = np.random.randint(ns)
##         r = np.random.randint(nr)
##         c = np.random.randint(nc)
##         if not pt_cache.has_key((s,r,c)):
##             pt_cache[ (s,r,c) ] = 1
##             return epi[:,s,r,c]
##         else:
##             return None

##     n = 0
##     reps = 0
##     while(n < N):
##         if epi is not None:
##             # can't downsample the epi due to high frequencies
##             a1 = random_slice(epi)
##             if a1 is None:
##                 reps += 1
##                 continue
##             a_up = upsample(a1, 5)
##         else:
##             a_up = FTS.gen_mixed_freqs(over_sample=5)
##             # grab center of every 5 points [ooxoo,ooxoo,ooxoo,...]
##             a1 = a_up[2::5].copy()
##         a2 = a1.copy()

##         t_up_ax = np.linspace(0, 1, a_up.shape[0], endpoint=False)
##         # now do subsample interpolation on the regularly sampled data, and
##         # make c=0.2 so that this should hit the upsampled grid one point back
##         FTS.subsampInterp(a1, 0.2)
##         FTS.subsampInterp_regular(a2, 0.2)
##         # grab the upsampled grid at an offset of -1 from the middle of 5
##         t_ax = t_up_ax[1::5]
##         # ========= REAL PART ==========
##         # test
##         err1 = (a_up[1::5].real[20:-20] - a1.real[20:-20])
##         rms1 = np.sqrt((err1**2).mean())
##         mederr1 = np.median(np.abs(err1))
##         pdf_rms_err_real_tst.append(rms1)
##         pdf_med_err_real_tst.append(mederr1)
##         # std
##         err2 = (a_up[1::5].real[20:-20] - a2.real[20:-20])
##         rms2 = np.sqrt((err2**2).mean())
##         mederr2 = np.median(np.abs(err2))
##         pdf_rms_err_real_std.append(rms2)
##         pdf_med_err_real_std.append(mederr2)
##         # ========= IMAG PART ==========
##         err1 = (a_up[1::5].imag[20:-20] - a1.imag[20:-20])
##         rms1 = np.sqrt((err1**2).mean())
##         mederr1 = np.median(np.abs(err1))
##         pdf_rms_err_imag_tst.append(rms1)
##         pdf_med_err_imag_tst.append(mederr1)
##         # std
##         err2 = (a_up[1::5].imag[20:-20] - a2.imag[20:-20])
##         rms2 = np.sqrt((err2**2).mean())
##         mederr2 = np.median(np.abs(err2))
##         pdf_rms_err_imag_std.append(rms2)
##         pdf_med_err_imag_std.append(mederr2)
##         n += 1

##     def myplot(e, h):
##         err_ax = (e[1:]+e[:-1])/2
##         igrl = sp.integrate.cumtrapz(h, x=err_ax)
##         pp.plot(err_ax, h)
##         np_lt_90 = (igrl<0.9).sum()
##         pts = (h<=h.mean()).nonzero()[0]
##         mn_pt = (h==(h[pts].max())).nonzero()[0][0]
##         xs, ys = mlab.poly_between(err_ax[:np_lt_90], 0, h[:np_lt_90])
##         pp.gca().fill(xs, ys, facecolor='red', alpha=0.5)
##         pp.plot([err_ax[np_lt_90-1]], [h[np_lt_90-1]], 'ro',
##                label='90pct at %1.3f'%err_ax[np_lt_90-1])
##         pp.plot([err_ax[mn_pt]], [h[mn_pt]], 'go',
##                label='mean at %1.3f'%err_ax[mn_pt])
        
##         pp.legend(loc=1)
    
##     pp.figure(figsize=(16,8))
##     emax = 0.
##     pp.subplot(221)
##     h, e = pp.histogram(pdf_rms_err_real_std, normed=True, new=True, bins=N/10)
##     emax = max(emax, e.max())
##     myplot(e, h)
##     pp.title('pdf(rms_err) real, std')
##     pp.subplot(222)
##     h, e = pp.histogram(pdf_rms_err_real_tst, normed=True, new=True, bins=N/10)
##     emax = max(emax, e.max())    
##     myplot(e, h)
##     pp.title('pdf(rms_err) real, tst')
##     pp.subplot(223)
##     h, e = pp.histogram(pdf_med_err_real_std, normed=True, new=True, bins=N/10)
##     emax = max(emax, e.max())
##     myplot(e, h)
##     pp.title('pdf(med_err) real, std')
##     pp.subplot(224)
##     h, e = pp.histogram(pdf_med_err_real_tst, normed=True, new=True, bins=N/10)
##     emax = max(emax, e.max())
##     myplot(e, h)
##     pp.title('pdf(med_err) real, tst')
##     for plot in range(221,225):
##         pp.subplot(plot)
##         pp.gca().set_xlim((0, emax))

##     # imag
##     pp.figure(figsize=(16,8))
##     emax = 0.
##     pp.subplot(221)
##     h, e = pp.histogram(pdf_rms_err_imag_std, normed=True, new=True, bins=N/10)
##     myplot(e, h)
##     emax = max(emax, e.max())    
##     pp.title('pdf(rms_err) imag, std')
##     pp.subplot(222)
##     h, e = pp.histogram(pdf_rms_err_imag_tst, normed=True, new=True, bins=N/10)
##     myplot(e, h)
##     emax = max(emax, e.max())    
##     pp.title('pdf(rms_err) imag, tst')
##     pp.subplot(223)
##     h, e = pp.histogram(pdf_med_err_imag_std, normed=True, new=True, bins=N/10)
##     myplot(e, h)
##     emax = max(emax, e.max())    
##     pp.title('pdf(med_err) imag, std')
##     pp.subplot(224)
##     h, e = pp.histogram(pdf_med_err_imag_tst, normed=True, new=True, bins=N/10)
##     myplot(e, h)
##     emax = max(emax, e.max())    
##     pp.title('pdf(med_err) imag, tst')
##     for plot in range(221,225):
##         pp.subplot(plot)
##         pp.gca().set_xlim((0, emax))

##     print "avoided", reps, "repetitions"
##     pp.show()


## def visual_test(a = None):
##     if a is not None:
##         a1 = a
##         a_up = upsample(a1, 5)
##         L = a_up.shape[0]
##     else:
##         a_up = FTS.gen_mixed_freqs(over_sample=5)
##         a1 = a_up[2::5].copy()
##     a2 = a1.copy()
##     t_up_ax = np.linspace(0, 1, a_up.shape[0], endpoint=False)
##     # now do subsample interpolation on the regularly sampled data, and
##     # make c=0.2 so that this should hit the upsampled grid one point back
##     FTS.subsampInterp_TD(a1, 0.2)
##     FTS.subsampInterp_regular(a2, 0.2)
##     # grab the upsampled grid at an offset of -1 from the middle of 5
##     t_ax = t_up_ax[1::5]
##     # ========= REAL PART ==========
##     pp.figure(figsize=(16,8))
##     pp.subplot(221)
##     pp.plot(t_up_ax, a_up.real)
##     pp.plot(t_ax, a1.real, 'r.')
##     pp.subplot(223)
##     err1 = (a_up[1::5].real[20:-20] - a1.real[20:-20])
##     rms1 = np.sqrt((err1**2).mean())
##     mederr1 = np.median(err1**2)
##     pp.plot(t_ax[20:-20], np.abs(err1))
##     pp.title('RAMP TRICKS: rms_err = %f, med_err = %f'%(rms1, mederr1))
##     pp.subplot(222)
##     pp.plot(t_up_ax, a_up.real)
##     pp.plot(t_ax, a2.real, 'r.')
##     pp.subplot(224)
##     err2 = (a_up[1::5].real[20:-20] - a2.real[20:-20])
##     rms2 = np.sqrt((err2**2).mean())
##     mederr2 = np.median(err2**2)
##     pp.plot(t_ax[20:-20], np.abs(err2))
##     pp.title('REGULAR: rms_err = %f, med_err = %f'%(rms2, mederr2))
    
##     # ========= IMAG PART ==========
##     pp.figure(figsize=(16,8))
##     pp.subplot(221)
##     pp.plot(t_up_ax, a_up.imag)
##     pp.plot(t_ax, a1.imag, 'r.')
##     pp.subplot(223)
##     err1 = (a_up[1::5].imag[20:-20] - a1.imag[20:-20])
##     rms1 = np.sqrt((err1**2).mean())
##     mederr1 = np.median(err1**2)
##     pp.plot(t_ax[20:-20], np.abs(err1))
##     pp.title('RAMP TRICKS: rms_err = %f, med_err = %f'%(rms1, mederr1))
##     pp.subplot(222)
##     pp.plot(t_up_ax, a_up.imag)
##     pp.plot(t_ax, a2.imag, 'r.')
##     pp.subplot(224)
##     err2 = (a_up[1::5].imag[20:-20] - a2.imag[20:-20])
##     rms2 = np.sqrt((err2**2).mean())
##     mederr2 = np.median(err2**2)
##     pp.plot(t_ax[20:-20], np.abs(err2))
##     pp.title('REGULAR: rms_err = %f, med_err = %f'%(rms2, mederr2))
##     pp.show()

## def test_deramp_and_circ(a = None):
##     if a is not None:
##         a1 = a
##     else:
##         a1 = FTS.gen_mixed_freqs()
##     To = a1.shape[0]
##     ramps = np.empty_like(a1)
##     rax_shape = [1]*len(a1.shape)
##     rax_shape[-1] = To
##     rax = np.arange(To)
##     rax.shape = rax_shape
##     (mre,b,r) = util.lin_regression(a1.real, axis=-1)
##     ramps.real[:] = rax*mre
##     if a1.dtype.type in np.sctypes['complex']:
##         (mim,b,r) = util.lin_regression(a1.imag, axis=-1)
##         ramps.imag[:] = (rax*mim)
##     np.subtract(a1, ramps, a1)
##     # find biases and subtract them
##     ts_mean = a1.mean(axis=-1)
##     if len(ts_mean.shape):
##         mean_shape = list(a1.shape)
##         mean_shape[-1] = 1
##         ts_mean.shape = tuple(mean_shape)
##     np.subtract(a1, ts_mean, a1)

##     a1_circ = FTS.circularize(a1)

##     np.add(a1, ramps, a1)
##     np.add(a1, ts_mean, a1)
##     t_ax = np.arange(-To+1, 2*To-1)
##     pp.plot(t_ax[To-1:2*To-1], a1.real)
##     pp.plot(t_ax, a1_circ.real, 'r--')
##     pp.title('real part')
##     pp.figure()
##     pp.plot(t_ax[To-1:2*To-1], a1.imag)
##     pp.plot(t_ax, a1_circ.imag, 'r--')
##     pp.title('imag part')
##     pp.show()
   

class test_interp(TestCase):

    def setUp(self):
        self.test_sig_flat = np.ones(128)
        self.test_sig_rand_r = np.random.rand(200,64,64)
        self.test_sig_rand_i = np.empty((200,64,64), np.complex64)
        self.test_sig_rand_i.real[:] = np.random.rand(200,64,64)
        self.test_sig_rand_i.imag[:] = np.random.rand(200,64,64)
                                    
    def check_1pixshift_r(self):
        test_copy1 = self.test_sig_rand_r.copy()
        test_copy2 = test_copy1.copy()
        FTS.subsampInterp(test_copy1, 1.0, axis=0)
        util.shift(test_copy2, 1, axis=0)
        assert_array_almost_equal(test_copy2[1:-1,:,:],
                                  test_copy1[1:-1,:,:])

    def check_1pixshift_i(self):
        test_copy1 = self.test_sig_rand_i.copy()
        test_copy2 = test_copy1.copy()
        FTS.subsampInterp(test_copy1, 1.0, axis=0)
        util.shift(test_copy2, 1, axis=0)
        assert_array_almost_equal(test_copy2[1:-1,:,:],
                                  test_copy1[1:-1,:,:], decimal=5)
    
    def check_flatshift(self):
        flat_sig = self.test_sig_flat.copy()
        FTS.subsampInterp(flat_sig, .15)
        assert_array_almost_equal(flat_sig, self.test_sig_flat)

    
    
##     def check_two_interps(self, level=1):
##         test_copy1 = self.test_sig_rand_i.copy()
##         test_copy2 = self.test_sig_rand_i.copy()
##         ref_copy = self.test_sig_rand_i.copy()
##         FTS.subsampInterp(test_copy1, .25, axis=0)
##         FTS.subsampInterp_wv(test_copy2, .25, axis=0)
##         assert_array_almost_equal(test_copy1[1:-1], test_copy2[1:-1],
##                                   decimal=1)
## ##         pp.semilogy(np.abs(np.abs(test_copy1).sum(axis=-1).sum(axis=-1) -
## ##                          np.abs(test_copy2).sum(axis=-1).sum(axis=-1)))
## ##         pp.show()

##     def check_time(self, level=1):
##         test_copy = self.test_sig_rand_i.copy()
##         t1 = measure('FTS.subsampInterp(test_copy, .25, axis=0)', 1)
##         print "normal version: %f sec"%t1
##         t2 = measure('FTS.subsampInterp_wv(test_copy, .25, axis=0)', 1)
##         print "inline version: %f sec"%t2
