from recon.simulation import ghosts as g, newghosts as ng

import numpy as N, pylab as P
from recon import imageio, util
from recon.fftmod import fft1, ifft1, ifft2, fft2
from recon.scanners import siemens
from recon.operations.InverseFFT import InverseFFT
from recon.operations import Operation
from recon.operations.ForwardFFT import ForwardFFT
from recon.operations.ReorderSlices import ReorderSlices
from recon.operations.RotPlane import RotPlane
from recon.operations.FlipSlices import FlipSlices
from recon.operations.ViewImage import ViewImage
from recon.operations.ComputeFieldMap import build_3Dmask


def load_agems(T0,Tr,Tf,nr,nf, ro_offset=0):
    agems = imageio.readImage(g.file_lookup['apr24_1']['agems'], N1=128)
    agems.T0 = T0
    agems.Ti = T0
    agems.T_ramp = Tr
    agems.T_flat = Tf
    agems.Nc = agems.shape[-1]/2 + 1
    #agems.n_ramp = nr
    #agems.n_flat = nf
    agems.ro_offset = ro_offset
    rotation_mat = N.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
    agems.orientation_xform = util.Quaternion(M=rotation_mat)
    agems.run_op(ReorderSlices())
    #agems.run_op(RotPlane(orient_target='radiological'))
    #agems.run_op(FlipSlices(fliplr=True, flipud=True))
    return agems

def perturb_image(image, kernel, inv=0):
    imshape = tuple([d for d in image.shape[:-1]])
    N1 = image.shape[-1]
    imshape += (kernel.shape[-1],)
    #sn = N.empty(image.shape, image[:].dtype)
    sn = N.empty(imshape, image[:].dtype)

    dslice = [slice(None)] * image.ndim

    for n in [0, 1]:
        # get even rows [0,2,...] for n=0, odd rows for n=1
        dslice[-2] = slice(n, None, 2)
        #dslice1[-3] = slice(n, None, 2)
        # multiply n1xn1p (inverse) kernel with data sliced as
        # (nvol*nsl*evn/odd-rows) x (n1xn1p) where n1 is a dummy dimension
        # then sum over n1p dimension to reduce corrected data to shape
        # (nvol*nsl*evn/odd-rows) x n1
        if inv:
            f = N.linalg.inv(kernel[n])
        else:
            f = kernel[n]
        sn[dslice] = N.dot(image[dslice], f.transpose())
    unghosted = image._subimage(sn)
    g.copy_timing(image, unghosted)
    return unghosted
    
    
def linear_fit_nd(y, x=None, mask=None):
    yshape = y.shape
    nd = len(yshape)
    nrow = N.product(yshape)
    if x:
        assert len(x) == nd, "wrong number of axes arrays"
        for d in range(nd):
            assert len(x[d]) == yshape[d], "dimension %d is the wrong size"%d
    y.shape = (nrow,)
    if mask is not None:
        mask.shape = (nrow,)
    A = N.empty((nrow, nd+1), N.float64)
    # 1st dim is just [000000000000111111111111222222222222]
    # 2nd dim is      [000011112222333344445555666677778888]
    # 3rd dim is      [123412341234123412341234123412341234]
    idx_blob = N.empty(yshape, N.float64)
    for d in range(nd):
        copy_slice = [None]*nd
        copy_slice[d] = slice(None)
        if x:
            idx_blob[:] = x[d][copy_slice]
        else:
            idx_blob[:] = N.arange(yshape[d])[copy_slice]
        A[:,d] = idx_blob.flatten()
    A[:,-1] = 1.

    nz = slice(None) if mask is None else mask.nonzero()[0]
    # do svd soln
    u,s,vt = N.linalg.svd(A[nz], full_matrices=False)
    V = N.dot(vt.transpose(), N.dot(N.diag(1/s), N.dot(u.transpose(), y[nz])))
    #V2 = N.linalg.lstsq(A[nz], y[nz])
    y.shape = yshape
    if mask is not None:
        mask.shape = yshape
    return V



## def eval_pts(epi, a1, allchan=True):
##     qual_array = N.empty(len(pts), N.float64)
##     for n,p in enumerate(pts):
##         corr_list = []
##         for c in range(epi.n_chan):
##             epi.use_membuffer(c)
##             corr_list.append(undistort_img(epi, p[0], p[1], p[2]))
##             InverseFFT().run(corr_list[-1])
##         qual_array[n] = eval_ghosts(corr_list)
##     # find best point
##     bpt = (qual_array == qual_array.min()).nonzero()[0][0]
##     return pts[bpt]

## def eval_ghosts(corr_list):
##     #ROWS = range(10) + range(59,64)
##     #ROWS = [0,1,2,3,60,61,62,63]
##     ROWS = range(0,15) + range(54,64)
##     COLS = range(35,43)
##     d = N.zeros(corr_list[0].shape, N.float64)
##     for img in corr_list:
##         d += N.power(img[:].real, 2.0) + N.power(img[:].imag, 2.0)
##     meas = N.sqrt(d[:,ROWS,:][:,COLS]).sum()
##     return meas
        


def drive_search(epi, pct_list=[0.5,1.0,1.5,2.0]):
    #ROWS = range(0,15) + range(54,64)
    #COLS = range(34,42) + range(84,93)
    ROWS = range(6) + [62,63]
    COLS = range(128)
    coefs = ng.solve_coefs_l1(epi, pct=30.)
    best_ghost = 1e11
    ghost_strengths = []
    xy_idx = N.indices(epi.shape[-2:], N.float64)
    msk = N.where((N.power(xy_idx[0]-33, 2) + N.power(xy_idx[1]-63,2)) < 28.5**2, 0, 1)
    for pct in pct_list:
        deghost = ng.undistort_img(epi, coefs[0]*(1+pct/100.))
        InverseFFT().run(deghost)
        #ghost_strength = N.abs(deghost[:,ROWS,:][:,:,COLS]).sum()
        ghost_strength = N.abs(deghost[:]*msk).sum()
        print ghost_strength
        ghost_strengths.append(ghost_strength)
        if ghost_strength < best_ghost:
            best_ghost = ghost_strength
            best_point = (pct, ghost_strength)
    
    print "original a1:", coefs[0]
    print "best a1: %f (at %f pct)"%((1+best_point[0]/100.)*coefs[0],
                                     best_point[0])
    return best_point[0], (pct_list, N.array(ghost_strengths))

def drive_percentile_search(epi, npts):
    ROWS = range(0,15) + range(54,64)
    COLS = range(34,42) + range(84,93)
    best_ghost = 1e11
    ghost_strengths = []
    pct_list = N.linspace(1,100,npts)
    xy_idx = N.indices(epi.shape[-2:], N.float64)
    x0 = 64
    y0 = 33
    msk = N.where((N.power(xy_idx[0]-y0, 2) + N.power(xy_idx[1]-x0,2)) < 28.5**2, 0, 1)
    for pct in N.linspace(1,100,npts):
        coefs = ng.solve_coefs(epi, pct=pct)
        deghost = ng.undistort_img(epi, coefs[0])
        InverseFFT().run(deghost)
        #ghost_strength = N.abs(deghost[:,ROWS,:][:,:,COLS]).sum()
        ghost_strength = N.abs(deghost[:]*msk).sum()
        print ghost_strength
        ghost_strengths.append(ghost_strength)
        if ghost_strength < best_ghost:
            best_ghost = ghost_strength
            best_point = (pct, ghost_strength)

    return best_point[0], (pct_list, N.array(ghost_strengths))

def drive_2d_search(epi, a0max, pct_list):
    best_ghost = 1e11
    ghost_strengths = []
    coefs = ng.solve_coefs(epi, pct=30.)
    xy_idx = N.indices(epi.shape[-2:], N.float64)
    BWpe = 37.
    x0 = 64
    y0 = 33
    msk = N.where((N.power(xy_idx[0]-y0, 2) + N.power(xy_idx[1]-x0,2)) < 29**2, 0, 1)
    npts = len(pct_list)
    a1_list = coefs[0] * (1 + pct_list/100.)
    a0_list = N.linspace(-a0max, a0max, len(pct_list))
    ghost_strength_surf = N.empty((npts, npts), N.float64)
    for n,a1 in enumerate(a1_list):
        for p,a0 in enumerate(a0_list):
            deghost = ng.undistort_img_n2xn1(epi, a1, a0)
            InverseFFT().run(deghost)
            y0 = 33 + a0/BWpe
            msk = N.where((N.power(xy_idx[0]-y0, 2) + \
                           N.power(xy_idx[1]-x0,2)) < 28.5**2, 0, 1)
            ghost_strength = N.abs(deghost[:]*msk).sum()
            print ghost_strength
            ghost_strength_surf[n,p] = ghost_strength
            #ghost_strengths.append(ghost_strength)
            if ghost_strength < best_ghost:
                best_ghost = ghost_strength
                best_point = ((a1,a0), ghost_strength)

    return best_point[0], ((a1_list, a0_list), ghost_strength_surf)
    
xy_idx = N.indices((64,128), N.float64)

def eval_func_a1a0(a, epi, fdict):
    a1 = a[0]
    a0 = a[1]
    if fdict.has_key((a1, a0)):
        return fdict[(a1,a0)]
    
    dg = ng.undistort_img_n2xn1(epi, a1, a0)
    InverseFFT().run(dg)
    y0 = 33 + a0/37.
    x0 = 64
    msk = N.where((N.power(xy_idx[0]-y0, 2) + \
                   N.power(xy_idx[1]-x0,2)) < 28.5**2, 0, 1)
    ghost_strength = N.abs(dg[:]*msk).sum()
    print ghost_strength
    fdict[(a1,a0)] = ghost_strength
    return ghost_strength, dg, msk

def eval_func_a1_diag(a, epi, fdict, diag):
    a1 = a
    max_a1 = fdict[-1]
    if a1 < 0.0 or a1 > max_a1:
        return 1e11
    if fdict.has_key(a1):
        return fdict[a1]
    dg = ng.undistort_img_diag(epi, a1, ndiag=diag)
    InverseFFT().run(dg)
    msk = fdict[N.pi]
    ghost_strength = sim_idx1(N.abs(dg[:]), msk)
    print "func(%f) = %f"%(sigma1(epi,a1),ghost_strength)
    fdict[a1] = ghost_strength
    # save best SI
    if ghost_strength < fdict[-2]:
        fdict[-2] = ghost_strength
    return ghost_strength

def eval_func_a1(a, epi, fdict, crit):
    a1 = a
    max_a1 = fdict[-1]
    if a1 < 0.0 or a1 > max_a1:
        return 1e11
    if fdict.has_key(a1):
        return fdict[a1]
    dg = ng.undistort_img(epi, a1, n1p_fact=2.)
    InverseFFT().run(dg)
    msk = fdict[N.pi]
    if crit=='mask':
        ghost_strength = N.abs(dg[:]*msk).sum()
    else:
        ghost_strength = sim_idx1(N.abs(dg[:]), msk)
    print "func(%f) = %f"%(sigma1(epi,a1),ghost_strength)
    fdict[a1] = ghost_strength
    # save best SI
    if ghost_strength < fdict[-2]:
        fdict[-2] = ghost_strength
    return ghost_strength

def eval_sig(sig, epi, a1, tn, fdict):
    if fdict.has_key(sig):
        return fdict[sig]
    if sig>=0: #and sig < 0.8*sigma1(epi, a1):
        dg = eddycur_corr(epi, sig, a1, tn)
        InverseFFT().run(dg)
        msk = fdict[N.pi]
        ghost_strength = N.abs(dg[:]*msk).sum()
    else:
        ghost_strength = 1e11
    print "f(%f) = %f"%(sig, ghost_strength)
    fdict[sig] = ghost_strength
    return ghost_strength

def sim_idx1(epi, ref):
    n = (epi[:]*ref[:]).sum()
    d = (N.power(epi[:], 2.0).sum() * N.power(ref[:], 2.0).sum())**0.5
    return n/d

def sim_idx2(epi, ref):
    n = N.abs(epi[:] - ref)
    d = epi[:] + ref
    npt = float(N.product(epi.shape))
    return 1 - (n/d).sum()/npt
    
        
def eval_func_a1sig(a, epi, fdict):
    a1 = a[0]
    s = a[1]
    if fdict.has_key( (a1, s) ):
        return fdict[ (a1, s) ]
    # CONSTRAINTS:
    # both time constants need to be non-negative,
    # sigma should be within a reasonable limit of the solved for sigma,
    # and sigma-e should be significantly less than sigma1
    s1 = sigma1(epi, a1)
    # apparently driving a1 up explodes the inverse kernel, don't let
    # that happen
    max_a1 = fdict[-1]
    if a1 >= 0 and a1 < max_a1 and s >= 0 and s < 0.75*s1:
        dg = ng.undistort_img_a1sig(epi, a1, s, n1p_fact=1.)
        InverseFFT().run(dg)
        msk = fdict[N.pi]
        #ghost_strength = N.abs(dg[:]*msk).sum()
        ghost_strength = sim_idx1(N.abs(dg[:]), msk)
        #ghost_strength = sim_idx2(N.abs(dg[:]), msk)
    else:
        ghost_strength = 1e11
    print "func(%f, %f) = %f"%(s1, s, ghost_strength)
    fdict[ (s1, s) ] = ghost_strength
    return ghost_strength

def eddycur_corr(epi, sig, a1, tn1):
    N1 = epi.shape[-1]
    sig1 = sigma1(epi, a1)
    Tr = float(epi.T_ramp)
    Tf = float(epi.T_ramp)
    T0 = float(epi.T0)
    Nc = epi.Nc
    As = (Tf + Tr - T0**2/Tr)/2.
    
    Lx = epi.shape[-1]*epi.isize
    # gmaG0 = 2PI * (Nc-1)/(Lx*As)
    gma = 2*N.pi*42575575. # larmor in rad/tesla/sec
    gmaG0 = 2*N.pi * (Nc-1)/(Lx*As) # this is radians per microsecond per mm
    gmaG0_n2 = gmaG0 * N.array([1.0, -1.0])

    tn = N.empty((2,N1), N.float64)
    tn[0] = tn1
    tn[1] = tn1[::-1]
    
    phi0 = N.empty((2,N1), N.float64)
    reg1 = (tn-sig1-sig) <= Tr
    reg2 = ((tn-sig1-sig) > Tr) & ((tn-sig1-sig) <= Tr+Tf)
    reg3 = (tn-sig1-sig) > Tr+Tf
    Ttot = 2*Tr + Tf
    fr1 = -gmaG0_n2[:,None] * (sig/Tr) * (tn - sig1 - sig)
    fr2 = -gmaG0_n2[:,None] * (sig/Tr) * Tr * N.ones_like(tn)
    fr3 = -gmaG0_n2[:,None] * (sig/Tr) * (Ttot - (tn - sig1 - sig))
    phi0[reg1] = fr1[reg1]
    phi0[reg2] = fr2[reg2]
    phi0[reg3] = fr3[reg3]
    

    k = N.exp(1j*phi0)
    sn = N.empty(epi.shape, epi[:].dtype)
    dslice = [slice(None)]*epi.ndim
    dslice[-2] = slice(0,epi.shape[-2],2)
    sn[dslice] = epi[dslice]*k[0]
    dslice[-2] = slice(1,epi.shape[-2],2)
    sn[dslice] = epi[dslice]*k[1]
    dg = epi._subimage(sn)
    return dg

def get_epi_mask(epi):
    dg_est = ng.full_phs_corr(epi, pct=50.)
    vmask = build_3Dmask(dg_est[:], 0.25)
    # now close the mask,and dilate it a few times
    msk = morph.binary_closing(vmask[10])
    msk = morph.binary_dilation(msk, iterations=4)
    msk = (1-msk)
    return msk, dg_est

from scipy import optimize
from scipy.ndimage import morphology as morph

def sigma1(epi, a1):
    Ti = float(epi.Ti)
    T0 = float(epi.T0)
    Tr = float(epi.T_ramp)
    Tf = float(epi.T_flat)
    Nc = epi.Nc
    As = (Tf + Tr - Ti**2/Tr)/2.
    
    Lx = epi.shape[-1]*epi.isize
    # gmaG0 = 2PI * (Nc-1)/(Lx*As)
    gma = 2*N.pi*42575575. # larmor in rad/tesla/sec
    gmaG0 = 2*N.pi * (Nc-1)/(Lx*As) # this is radians per microsecond per mm
    # sigma1 is a1 / (gmaG0*dx) (microsecs)
    sigma1 = a1 / (gmaG0*epi.isize)
    return sigma1
    

def run_a1sig_search(epi, norm=1, plotting=True):
    # these steps should get a pretty good 1st pass mask
    #msk, dg_est = get_epi_mask(epi)
    msk = N.empty(epi.shape, N.float32)
    solv_a1 = []
    best_a1sig = []
    best_corrs = []
    Q3,Q2,Q1 = epi.shape[-3:]
    for c in range(epi.n_chan):
        epi.load_chan(c)
        ifft2(epi[:], inplace=True)
        msk[:,:Q2/2,:] = N.abs(epi[:,Q2/2:,:])
        msk[:,Q2/2:,:] = N.abs(epi[:,:Q2/2,:])
        fft2(epi[:], inplace=True)
        if norm==1:
            cf = ng.solve_coefs_l1(epi, pct=50.)
        elif norm==2:
            cf = ng.solve_coefs_l2(epi, pct=50.)
        solv_a1.append(cf[0])
        fdict = {}
        fdict[N.pi] = msk
        fdict[-1] = 1.2*cf[0]
        #x0 = N.array([cf[0], 0.25 * sigma1(epi,cf[0])])
        x0 = N.array([cf[0], 0.])        
        r = optimize.fmin(eval_func_a1sig, x0, args=(epi, fdict),
                          ftol=1e-5, xtol=1e-5)
        # good practice is to restart at the best found point
##         r = optimize.fmin(eval_func_a1sig, r, args=(epi, fdict),
##                           ftol=1e-4, xtol=1e-6)
        best_a1sig.append(r)
        best_corrs.append(ng.undistort_img_a1sig(epi, r[0], r[1]))
        InverseFFT().run(best_corrs[-1])
    solv_a1 = N.array(solv_a1)
    best_a1sig = N.array(best_a1sig)
    if plotting:
        plot_search(sigma1(epi, solv_a1), sigma1(epi, best_a1sig[:,0]))
        P.plot(best_a1sig[:,1], 'g--')
        P.plot(best_a1sig[:,1], 'go')
        P.show()
    best_dg = epi._subimage(g.sumsqr(best_corrs,
                                     channel_gains=epi.channel_gains))
    return (solv_a1, best_a1sig, msk, best_dg, best_corrs)

def run_a1_search(epi, norm=1, plotting=True, crit='si'):
    # these steps should get a pretty good 1st pass mask
    Q3,Q2,Q1 = epi.shape[-3:]
    if crit=='si':
        msk = N.empty(epi.shape, N.float32)
    else:
        msk, dg_est = get_epi_mask(epi)
    solv_a1 = []
    best_a1 = []
    best_corrs = []
    for c in range(epi.n_chan):
        epi.load_chan(c)
        if crit=='si':
            InverseFFT().run(epi)
            msk[:,:Q2/2,:] = N.abs(epi[:,Q2/2:,:])
            msk[:,Q2/2:,:] = N.abs(epi[:,:Q2/2,:])
            ForwardFFT().run(epi)
        if norm==1:
            cf = ng.solve_coefs_l1(epi, pct=50.)
        elif norm==2:
            cf = ng.solve_coefs_l2(epi, pct=50.)
        solv_a1.append(cf[0])
        fdict = {}
        fdict[N.pi] = msk
        fdict[-1] = 1.2*cf[0]
        r = optimize.brent(eval_func_a1, args=(epi, fdict, crit),
                           brack=[.8*cf[0], cf[0] * 1.2])
        print "done chan %d, best a1: %f"%(c, r)
        best_a1.append(r)
        best_corrs.append(ng.undistort_img(epi, r))
        InverseFFT().run(best_corrs[-1])
    solv_a1 = N.array(solv_a1)
    best_a1 = N.array(best_a1)
    if plotting:
        plot_search(solv_a1, best_a1)
        P.show()
    best_dg = epi._subimage(g.sumsqr(best_corrs,
                                     channel_gains=epi.channel_gains))
    return (solv_a1, best_a1, msk, best_dg, best_corrs)

from recon import fftmod
def find_si(epi):
    Q3,Q2,Q1 = epi.shape[-3:]
    msk = N.empty(epi.shape, N.float32)
    fftmod.ifft2(epi[:], inplace=True, shift=True)
    msk[:,:Q2/2,:] = N.abs(epi[:,Q2/2:,:])
    msk[:,Q2/2:,:] = N.abs(epi[:,:Q2/2,:])
    fftmod.fft2(epi[:], inplace=True, shift=True)
    max_diag = Q2
    min_diag = 1
    si_by_diag = N.empty((max_diag-min_diag,))
    cf = ng.solve_coefs_l1(epi, pct=50.)
    for diag in range(min_diag,max_diag):
        fdict = {}
        fdict[N.pi] = msk
        fdict[-1] = 1.2*cf[0]
        fdict[-2] = 1e10
        r = optimize.brent(eval_func_a1_diag, args=(epi, fdict, diag),
                           brack=[.8*cf[0], cf[0]*1.2])
        si_by_diag[diag-min_diag] = fdict[-2]
    P.plot(N.arange(min_diag,max_diag), si_by_diag)
    P.show()
    return si_by_diag
        
def run_sig_search(epi, best_a1, msk=None):
    if msk is None:
        msk, _ = get_epi_mask(epi)
    best_sig = []
    tn = epi.t_n1()
    for c in range(epi.n_chan):
        epi.load_chan(c)
        fdict = {}
        fdict[N.pi] = msk
        dg1 = ng.undistort_img(epi, best_a1[c])
        g.copy_timing(epi, dg1)
        r = optimize.brent(eval_sig, args=(dg1, best_a1[c], tn, fdict),
                           brack=[0.0, 10*sigma1(epi, best_a1[c])])
        best_sig.append(r)
    best_sig = N.array(best_sig)
    return best_sig

def plot_search(solve, search):
    P.plot(solve-search)
    P.plot(solve, 'b--')
    P.plot(solve, 'bo')
    P.plot(search, 'r--')
    P.plot(search, 'ro')


def run_searches(epi_list, norm=1):
    best_a1 = N.zeros((0,))
    solv_a1 = N.zeros((0,))
    best_dg = []
    solv_dg = []
    for epi_file in epi_list:
        epi = imageio.readImage(g.file_lookup[epi_file]['epi'],
                                vrange=(0,0), N1=128)
        epi.run_op(ReorderSlices())
        (sa1, ba1, msk, sdg, bdg) = run_a1_search(epi, norm=norm,
                                                  plotting=False)
        best_a1 = N.concatenate((best_a1, ba1))
        solv_a1 = N.concatenate((solv_a1, sa1))
        best_dg.append(bdg)
        solv_dg.append(sdg)
    return (solv_a1, best_a1, solv_dg, best_dg)


"""
Definition:     optimize.fmin(func, x0, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=None)
Docstring:
    Minimize a function using the downhill simplex algorithm.

    *Parameters*:

      func : callable func(x,*args)
          The objective function to be minimized.
      x0 : ndarray
          Initial guess.
      args : tuple
          Extra arguments passed to func, i.e. ``f(x,*args)``.
      callback : callable
          Called after each iteration, as callback(xk), where xk is the
          current parameter vector.

    *Returns*: (xopt, {fopt, iter, funcalls, warnflag})

      xopt : ndarray
          Parameter that minimizes function.
      fopt : float
          Value of function at minimum: ``fopt = func(xopt)``.
      iter : int
          Number of iterations performed.
      funcalls : int
          Number of function calls made.
      warnflag : int
          1 : Maximum number of function evaluations made.
          2 : Maximum number of iterations reached.
      allvecs : list
          Solution at each iteration.

    *Other Parameters*:

      xtol : float
          Relative error in xopt acceptable for convergence.
      ftol : number
          Relative error in func(xopt) acceptable for convergence.
      maxiter : int
          Maximum number of iterations to perform.
      maxfun : number
          Maximum number of function evaluations to make.
      full_output : bool
          Set to True if fval and warnflag outputs are desired.
      disp : bool
          Set to True to print convergence messages.
      retall : bool
          Set to True to return list of solutions at each iteration.

    *Notes*

        Uses a Nelder-Mead simplex algorithm to find the minimum of
        function of one or more variables.

"""
