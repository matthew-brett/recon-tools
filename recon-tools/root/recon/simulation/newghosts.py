import numpy as N, pylab as P
from recon import imageio, util
from recon.fftmod import fft1, ifft1, ifft2
from recon.scanners import siemens
from recon.operations.InverseFFT import InverseFFT as InverseFFT_2
from recon.operations import Operation
from recon.operations.ForwardFFT import ForwardFFT
from recon.operations.ReorderSlices import ReorderSlices
from recon.operations.ViewImage import ViewImage
from recon.operations import UnbalPhaseCorrection_3d as UBPC
from recon.operations import UnbalPhaseCorrection_dev as UBPC_dev
from recon.operations.ComputeFieldMap import build_3Dmask


from recon.simulation import ghosts as g1


def plot_regions(image):
    (a1, a0) = solve_coefs_l1(image)


    Tr = float(image.T_ramp)
    Tf = float(image.T_flat)
    T0 = float(image.T0)
    Nc = image.Nc
    As = (Tf + Tr - T0**2/Tr)/2.
    
    nr = image.n_ramp
    nf = image.n_flat

    Lx = image.shape[-1]*image.isize
    # gmaG0 = 2PI * (Nc-1)/(Lx*As)
    gma = 2*N.pi*42575575. # larmor in rad/tesla/sec
    gmaG0 = 2*N.pi * (Nc-1)/(Lx*As) # this is radians per microsecond per mm
    #gmaG0_n2 = N.array([1.0, -1.0]) * gmaG0
    G0 = 1e9*gmaG0/gma
    # sigma is a1 / (gmaG0*dx) (microsecs)
    sigma = a1 / (gmaG0*image.isize)
    print sigma

    tn = image.t_n1()
    print T0, Tr, Tf
    print nr, nf
    print sigma

    reg1 = tn <= Tr
    reg2 = (tn > Tr) & (tn <= Tr+sigma)
    reg3 = (tn > Tr+sigma) & (tn <= Tr+Tf)
    reg4 = (tn > Tr+Tf) & (tn <= Tr+Tf+sigma)
    reg5 = tn > Tr+Tf+sigma

    print "pts in r1:", reg1.sum()
    print "pts in r2:", reg2.sum()
    print "pts in r3:", reg3.sum()
    print "pts in r4:", reg4.sum()
    print "pts in r5:", reg5.sum()
    
    Ttot = 2*Tr + Tf
    fr1 = gmaG0 * sigma*(tn - sigma/2)/Tr
    fr2 = gmaG0 * (sigma - N.power(sigma - tn + Tr, 2.0)/Tr)
    fr3 = gmaG0 * sigma*N.ones(tn.shape, tn.dtype)
    fr4 = gmaG0 * (sigma - N.power(tn + Tr - Ttot, 2.0)/Tr)
    fr5 = gmaG0 * -sigma*(tn - Ttot - sigma/2)/Tr
    
    t = N.linspace(T0, Ttot-T0, 128, endpoint=False)
    grd = N.zeros((128,), N.float64)
    treg1 = tn < Tr
    treg2 = (tn>=Tr) & (tn<=Tr+Tf)
    treg3 = tn > Tr+Tf
    grd[treg1] = sigma*gmaG0 * tn[treg1]/Tr
    grd[treg2] = sigma*gmaG0
    grd[treg3] = sigma*gmaG0 * (1 - (tn[treg3] - Tr - Tf)/Tr)
    P.plot(tn, grd, 'c.')
    

    P.plot(tn,fr1)
    P.plot(tn,fr2)
    P.plot(tn,fr3)
    P.plot(tn,fr4)
    P.plot(tn,fr5)
    P.plot(tn[reg1], fr1[reg1], 'ko')
    P.plot(tn[reg2], fr2[reg2], 'ko')
    P.plot(tn[reg3], fr3[reg3], 'ko')
    P.plot(tn[reg4], fr4[reg4], 'ko')
    P.plot(tn[reg5], fr5[reg5], 'ko')
    ax = P.gca()
    ax.set_ylim((0,1.5*sigma*gmaG0))
    P.show()


def phs_surf(image, pct, zero_ramps=True, window_ramps=False):
    # want to return two surfaces..
    # 1st is Gpos - Gneg, 2nd is Gneg - Gpos
    # for siemens, this is angle(S[u=1]S*[u=0]) then angle(S[u=2]S*[u=1])

    # also modify the masking to only work with q1 points within the FOV
    rdata = image.ref_data.copy()
    if len(rdata.shape) > 3:
        rdata = rdata[0]
    
    Q3, N2, Q1 = rdata.shape

    nr = image.n_ramp
    nf = image.n_flat

    if zero_ramps:
        rdata[...,:nr+1] = 0.
        rdata[...,nf+1:] = 0.
        if (nr > 1) and (nf < Q1-1):
            rdata = g1.rsmooth(rdata, nr+1, nf+1)
    if window_ramps:
        g1.wsmooth(rdata, nr+1, nf+1)
    inv_ref = ifft1(rdata, shift=True)
    # this is S[u+1]S*[u] 
    inv_ref = inv_ref[:,1:,:] * N.conjugate(inv_ref[:,:-1,:])

    ro_fov = image.fov_x
    im_fov = Q1*image.isize
    fov_pts = int(Q1 * ro_fov/im_fov)
    fov_idx = [slice(None)] * len(rdata.shape)
    fov_idx[-1] = slice(Q1/2 - fov_pts/2, Q1/2 + fov_pts/2)
    no_img_idx = [slice(None)]*len(rdata.shape)
    no_img_idx[-1] = range(Q1/2-fov_pts/2) + range(Q1/2 + fov_pts/2, Q1)

    irmask = build_3Dmask(N.abs(inv_ref[fov_idx]), 0.1)
    fract_pts = irmask.sum()/float(Q3*(N2-1)*fov_pts)

    phs_vol = util.unwrap_ref_volume(inv_ref)

    # I should put high variance random values into phs_vol[no_img_idx]
    # to ensure that those points are not among the top P pct best points
    shape = phs_vol[no_img_idx].shape
    phs_vol[no_img_idx] = N.random.normal(scale=1e5, size=shape)
    phs_mean, mask = UBPC_dev.mean_and_mask(phs_vol[:,0,:],
                                            phs_vol[:,1,:],
                                            pct*fract_pts, N.arange(Q3))
    phs_mean[no_img_idx] = 0.
    #mask[no_img_idx] = 0.
    return phs_mean, mask

from scipy.special import gammaincc as gammq

def svd_solve(phs, mask):
    # phs is shaped (Q3, 2, Q1) where
    # row 0 corresponds to (-1)^(u+1) = -1 (u=0) (Gpos - Gneg)
    # row 1 corresponds to (-1)^(u+1) = +1 (u=1) (Gneg - Gpos)
    # phs.flatten() is Q3 repeats of (+1/-1)*( phs-line-to-fit )
    Q3,_,Q1 = phs.shape
    phs.shape = (Q3*2*Q1)
    mask.shape = (Q3*2*Q1)
    A = N.zeros((Q3*2*Q1,2), N.float64)
    # makes 2*Q3 rows of [0,1,2,...Q1-1] for the a1 term
    # q1_idx = N.outer(N.ones(2*Q3), N.arange(Q1))
    q1_idx = N.outer(N.ones(2*Q3), N.arange(-Q1/2., Q1/2.))
    # this is all ones for the a0 term
    q0_idx = N.ones(Q3*2*Q1)
    # this changes sign for every other group of Q1 pts
    #usign = N.repeat(-util.checkerline(Q3*2), Q1)
    usign = N.repeat(util.checkerline(Q3*2), Q1)
    A[:,0] = 2.0 * usign * q1_idx.flatten()
    A[:,1] = 2.0 * usign * q0_idx

    nz = mask.nonzero()[0]
    A = A[nz]
    phs = phs[nz]

    [u,w,vt] = N.linalg.svd(A, full_matrices=0)
    v = vt.transpose()
    V = N.dot(v, N.dot(N.diag(1/w), N.dot(u.transpose(), phs)))
    cov = N.zeros((v.shape[0], v.shape[0]), N.float64)
    for i,col in enumerate(vt):
        cov[:] += N.outer(col, col)/w[i]**2
    chi_sq = N.power(N.dot(A,V)-phs, 2.0).sum()
    Q = gammq((len(phs)-2)/2., chi_sq/2.)
    return V, chi_sq, w, v, cov, Q


def l1norm_solve(phs, mask):
    # make a giant 1D list of all samples and locations
    # flip the sign of the "neg-pos" phase differences according to model
    Q1 = phs.shape[-1]
    xpos = N.zeros((0,))
    xneg = N.zeros((0,))
    ypos = N.zeros((0,))
    yneg = N.zeros((0,))
    x = N.zeros((0,))
    y = N.zeros((0,))
    for dsl, msl in zip(phs, mask):
        xpos = msl[0].nonzero()[0]
        xneg = msl[1].nonzero()[0]
        ypos = dsl[0,xpos]
        yneg = dsl[0,xneg]
        y = N.concatenate((y, ypos, yneg))
        x = N.concatenate((x, xpos-Q1/2., xneg-Q1/2.))
    (b,m,adiff) = medfit(x,y)
    return N.array([m/2., b/2.])

def solve_coefs_l2(image, pct=30., zero_ramps=True):
    res = svd_solve(*phs_surf(image, pct, zero_ramps=zero_ramps))
    return res[0]

def solve_coefs_l1(image, pct=30., zero_ramps=True):
    res = l1norm_solve(*phs_surf(image, pct, zero_ramps=zero_ramps))
    return res

def perturbation_kernel_analytical(image, a1, a0):
    Q3, Q2, Q1 = image.shape[-3:]
    N2, N1 = image.shape[-2:]

    T0 = float(image.T0)
    Tr = float(image.T_ramp)
    Tf = float(image.T_flat)
    As = (Tf + Tr - T0**2/Tr)/2.
    Nc = image.Nc
    
    nr = image.n_ramp
    nf = image.n_flat

    # this is (-1)^(n2)
    # actually this can be reduced to two cases: [1, -1]
    n2sign = N.array([1.0, -1.0])
    
    # fn is shaped (2, N1)
    evn = N.array([1, 0])
    odd = N.array([0, 1])
    fn = (evn[:,None] * N.arange(N1)) + \
         (odd[:,None] * N.arange(N1-1, -1, -1))

    #delF = 2*N.pi * 1./(Tf + 2*Tr - 2*T0)
    delF = 2*N.pi * 1./(Tf + Tr - T0**2/Tr)
    sigma = a1/delF
    print sigma

    treg1 = fn <= nr    
    treg2 = (fn > nr) & (fn <= nf)
    treg3 = fn > nf

    tn = N.empty(fn.shape, fn.dtype)
    tn[treg1] = N.power(T0**2 + 2*fn[treg1]*Tr*As/(Nc-1), 0.5)
    tn[treg2] = fn[treg2]*As/(Nc-1) + Tr/2 + T0**2/(2*Tr)
    tn[treg3] = (2*Tr+Tf) - N.power(2*Tr*(Tf + Tr - T0**2/(2*Tr) - \
                                          fn[treg3]*As/(Nc-1)), 0.5)

##     tn[treg1] = N.power(T0**2 + 2*fn[treg1]*Tr*As/N1, 0.5)
##     tn[treg2] = fn[treg2]*As/N1 + Tr/2 + T0**2/(2*Tr)
##     tn[treg3] = (2*Tr+Tf) - N.power(2*Tr*(As + T0**2/(2*Tr) - \
##                                           fn[treg3]*As/N1), 0.5)

    reg1 = tn <= Tr
    reg2 = (tn > Tr) & (tn <= Tr+sigma)
    reg3 = (tn > Tr+sigma) & (tn <= Tr+Tf)
    reg4 = (tn > Tr+Tf) & (tn <= Tr+Tf+sigma)
    reg5 = tn > Tr+Tf+sigma
    print "pts in r1:", reg1[0].sum()
    print "pts in r2:", reg2[0].sum()
    print "pts in r3:", reg3[0].sum()
    print "pts in r4:", reg4[0].sum()
    print "pts in r5:", reg5[0].sum()

    Ttot = Tf + 2*Tr
    fr1 = -n2sign[:,None] * (tn - a1/(2*delF))/Tr
    fr2 = -n2sign[:,None] * (1 - (delF/(a1*Tr))*N.power(a1/delF - tn + Tr, 2.0))
    fr3 = -n2sign[:,None] * N.ones(tn.shape, tn.dtype)
    fr4 = -n2sign[:,None] * (1 - (delF/(a1*Tr))*N.power(tn + Tr - Ttot, 2.0))
    fr5 = -n2sign[:,None] * -(tn - Ttot - a1/(2*delF))/Tr

    box = N.empty((2,N1), N.float64)
    box[reg1] = fr1[reg1]
    box[reg2] = fr2[reg2]
    box[reg3] = fr3[reg3]
    box[reg4] = fr4[reg4]
    box[reg5] = fr5[reg5]

    #a1_n = a1*fr3 * Q1 /(2*N.pi)
    #a0_n = (a0 + (Q1/2)*a1)*fr3
    #a0_n = a0*fr3
    a1_n = a1*box * Q1 /(2*N.pi)#* Q1 / (2*N.pi)
    a0_n = a0*box
    
    # the kernel function for fixed n2 is N1 rows of sinc(n1p-n1-a1[n2,n1])
    n1_ax = N.linspace(0., N1, N1, endpoint=False)
    #n1_ax = N.linspace(-N1/2., N1/2., N1, endpoint=False)
    # offsets is (N2,N1)--1st two dimensions of kernel
    offsets = a1_n + n1_ax
    # snc_idx is (N2,N1,N1P) where each pt in N1P dimension is n1p-offset(n2,n1)
    snc_idx = n1_ax[None,None,:] - offsets[:,:,None]
    # put in the extra phase from a0[n2,n1],
    # plus extra phase from r1->q1 discretization (r0 offset)
    extra_phs_n1 = N.exp(2.j*N.pi*n1_ax*(Q1/2 - 0.5)/Q1)
    extra_phs_n1p = N.exp(-2.j*N.pi*n1_ax*(Q1/2 - 0.5)/Q1)

    ep = extra_phs_n1[:,None] * extra_phs_n1p[None,:]

    #k = #N.exp(1.j*(a0_n + Q1/2*a1))[:,:,None] * N.sinc(snc_idx) #* ep[None,:,:]
    k = N.sinc(snc_idx) * N.exp(-1.j*(a0_n))[:,:,None] #* ep
    return k
    #return a1_n, a0_n


def phi_n(image, a1):
    # get gma*G0 from BW/px
    # get sigma from G0 and a1
    # define phi[n2,n1]

    Q3, Q2, Q1 = image.shape[-3:]
    N2, N1 = image.shape[-2:]

    Ti = float(image.Ti)
    T0 = float(image.T0)
    Tr = float(image.T_ramp)
    Tf = float(image.T_flat)
    Nc = image.Nc
    As = (Tf + Tr - Ti**2/Tr)/2.
    
    Lx = Q1*image.isize
    # gmaG0 = 2PI * (Nc-1)/(Lx*As)
    # gma = 2*N.pi*42575575. # larmor in rad/tesla/sec
    gmaG0 = 2*N.pi * (Nc-1)/(Lx*As) # this is radians per microsecond per mm
    gmaG0_n2 = N.array([1.0, -1.0]) * gmaG0

    # sigma is a1 / (gmaG0*dx) (microsecs)
    sigma = a1 / (gmaG0*image.isize)
    print sigma

    tn = N.empty((2,N1), N.float64)
    tn[0] = image.t_n1()
    tn[1] = image.t_n1()[::-1]
    
    reg1 = tn <= Tr
    reg2 = (tn > Tr) & (tn <= Tr+sigma)
    reg3 = (tn > Tr+sigma) & (tn <= Tr+Tf)
    reg4 = (tn > Tr+Tf) & (tn <= Tr+Tf+sigma)
    reg5 = tn > Tr+Tf+sigma
    print "pts in r1:", reg1.sum()
    print "pts in r2:", reg2.sum()
    print "pts in r3:", reg3.sum()
    print "pts in r4:", reg4.sum()
    print "pts in r5:", reg5.sum()
    
    Ttot = 2*Tr + Tf
    fr1 = gmaG0_n2[:,None] * sigma*(tn - sigma/2)/Tr
    fr2 = gmaG0_n2[:,None] * (sigma - N.power(sigma - tn + Tr, 2.0)/Tr)
    fr3 = gmaG0_n2[:,None] * sigma*N.ones(tn.shape, tn.dtype)
    fr4 = gmaG0_n2[:,None] * (sigma - N.power(tn + Tr - Ttot, 2.0)/Tr)
    fr5 = gmaG0_n2[:,None] * -sigma*(tn - Ttot - sigma/2)/Tr
    
    phi = N.empty((2,N1), N.float64)
    # phi's units are rad/mm 
    phi[reg1] = fr1[reg1]
    phi[reg2] = fr2[reg2]
    phi[reg3] = fr3[reg3]
    phi[reg4] = fr4[reg4]
    phi[reg5] = fr5[reg5]
    return phi
    
def perturbation_kernel_ana_a1(image, a1, n1p_fact=1.):
    N2,N1 = image.shape[-2:]
    Lx = N1*image.isize
    phi = phi_n(image, a1) / (2*N.pi)
    #r0 = -a0*image.isize/a1
    r0 = image.ro_offset
    #r0 = image.isize
    # make a big sinc function shaped (N2,N1,N1P)
    # each pt will be sinc[n1p-n1+phi[n2,n1]*Lx/2]
    n1_ax = N.linspace(0., N1, N1, endpoint=False)
    n1p_ax = N.linspace(0. - (n1p_fact-1)*N1/2.,
                        N1 + (n1p_fact-1)*N1/2., n1p_fact*N1,
                        endpoint=False)
    snc_idx = phi[:,:,None]*Lx - n1_ax[None,:,None] + n1p_ax[None,None,:]
    kreal = N.sinc(snc_idx)
    # handle the r0 term that falls out of the integral
    if abs(r0) > 0:
        ephase = 1j*r0*(phi[:,:,None] + \
                        2*N.pi*(n1p_ax[None,None,:]-n1_ax[None,:,None])/Lx)
        return kreal*N.exp(ephase)
    else:
        return kreal

def perturbation_kernel_ana_a1a0(image, a1, a0, n1p_fact=1.):
    N2,N1 = image.shape[-2:]
    Lx = N1*image.isize
    phi = phi_n(image, a1) / (2*N.pi)
    #r0 = -a0*image.isize/a1
    r0 = image.ro_offset
    #r0 = image.isize
    # make a big sinc function shaped (N2,N1,N1P)
    # each pt will be sinc[n1p-n1+phi[n2,n1]*Lx/2]
    n1_ax = N.linspace(0., N1, N1, endpoint=False)
    n1p_ax = N.linspace(0. - (n1p_fact-1)*N1/2.,
                        N1 + (n1p_fact-1)*N1/2., n1p_fact*N1,
                        endpoint=False)
    snc_idx = phi[:,:,None]*Lx - n1_ax[None,:,None] + n1p_ax[None,None,:]
    kreal = N.sinc(snc_idx)
    # handle the r0 term that falls out of the integral

    t_idx = N.indices(image.shape[-2:], N.float64)
    t_idx[1][::2] = image.t_n1()
    t_idx[1][1::2] = image.t_n1()[::-1]
    
    tn = t_idx[0]*(image.T_flat + 2*image.T_ramp) + t_idx[1]
    tn *= 1e-6
    kexp = N.exp(1j*2*N.pi*a0*tn)
    k = N.empty((N2,N1,n1p_fact*N1), N.complex128)
    k[0::2] = kexp[0::2,:,None] * kreal[0,:,:]
    k[1::2] = kexp[1::2,:,None] * kreal[1,:,:]
    return k
    

## def perturbation_kernel(phs):
##     _,N1,Q1 = phs.shape
##     #n1_ax = N.linspace(-N1/2., N1/2., N1, endpoint=False)
##     n1_ax = N.linspace(0., N1, N1, endpoint=False)
##     chk = util.checkerboard(N1,Q1)
##     n1_by_q1 = -2.j*N.pi*n1_ax[:,None]*n1_ax[None,:]/N1

##     #util.shift(phs, -Q1/2)
##     zarg = 1.j*phs + n1_by_q1[None,:,:]

##     extra_phs_n1 = N.exp(2.j*N.pi*n1_ax*(N1/2 - 0.5)/N1)
##     extra_phs_n1p = N.exp(-2.j*N.pi*n1_ax*(N1/2 - 0.5)/N1)

##     ep = extra_phs_n1[:,None] * extra_phs_n1p[None,:]
    
##     return ifft1(N.exp(zarg), shift=False) *ep

   

def undistort_img(image, a1, a0, n1p_fact=1.):
    #phs = perturbation_phs(image, a1, a0, newQ1=128)
    #f = perturbation_kernel_analytical(image, a1, a0)
    f = perturbation_kernel_ana_a1(image, a1, n1p_fact=n1p_fact)
    imshape = tuple([d for d in image.shape[:-1]])
    N1 = image.shape[-1]
    imshape += (n1p_fact*N1,)
    #sn = N.empty(image.shape, image[:].dtype)
    sn = N.empty(imshape, image[:].dtype)

    dslice = [slice(None)] * image.ndim

    for n in [0, 1]:
        # get even rows [0,2,...] for n=0, odd rows for n=1
        dslice[-2] = slice(n, None, 2)
        # multiply n1xn1p inverse kernel with data sliced as
        # (nvol*nsl*evn/odd-rows) x (n1xn1p) where n1 is a dummy dimension
        # then sum over n1p dimension to reduce corrected data to shape
        # (nvol*nsl*evn/odd-rows) x n1
        if n1p_fact != 1:
            finv = N.linalg.pinv(f[n])
        else:
            finv = N.linalg.inv(f[n])
        sn[dslice] = N.dot(image[dslice], finv.transpose())

    midpt = imshape[-1]/2
    unghosted = image._subimage(sn[...,midpt-N1/2:midpt+N1/2])
    g1.copy_timing(image, unghosted)
    return unghosted

def undistort_img_n2xn1(image, a1, a0, n1p_fact=1.):
    #phs = perturbation_phs(image, a1, a0, newQ1=128)
    #f = perturbation_kernel_analytical(image, a1, a0)
    f = perturbation_kernel_ana_a1a0(image, a1, a0, n1p_fact=n1p_fact)
    imshape = tuple([d for d in image.shape[:-1]])
    N2,N1 = image.shape[-2:]
    imshape += (n1p_fact*N1,)
    #sn = N.empty(image.shape, image[:].dtype)
    sn = N.empty(imshape, image[:].dtype)

    dslice = [slice(None)] * image.ndim

    for n in range(N2):
        # get even rows [0,2,...] for n=0, odd rows for n=1
        dslice[-2] = n
        # multiply n1xn1p inverse kernel with data sliced as
        # (nvol*nsl*evn/odd-rows) x (n1xn1p) where n1 is a dummy dimension
        # then sum over n1p dimension to reduce corrected data to shape
        # (nvol*nsl*evn/odd-rows) x n1
        if n1p_fact != 1:
            finv = N.linalg.pinv(f[n])
        else:
            finv = N.linalg.inv(f[n])
        sn[dslice] = N.dot(image[dslice], finv.transpose())

    midpt = imshape[-1]/2
    unghosted = image._subimage(sn[...,midpt-N1/2:midpt+N1/2])
    g1.copy_timing(image, unghosted)
    return unghosted


def planar_phs_subtract(image, a1, a0):
    Q3,Q2,Q1 = image.shape[-3:]
    q1_ax = N.linspace(-Q1/2., Q1/2., Q1, endpoint=False)
    #soln_line = a0 + a1*q1_ax
    soln_line = a1*q1_ax
    n2sign = util.checkerline(Q2)
    soln_pln = soln_line[None,:] * n2sign[:,None]
    phs = N.exp(-1.j*soln_pln)
    pfac_inv = N.exp(-2.j*N.pi*q1_ax*(Q1/2 - 0.5)/Q1)
    pfac_for = N.exp(2.j*N.pi*q1_ax*(Q1/2 - 0.5)/Q1)
##     scorr = pfac_for*fft1( ifft1(pfac_inv*image[:], shift=False)*phs ,
##                            shift=False )
    scorr = fft1(ifft1(image[:])*phs)
    deghost = image._subimage(scorr)
    g1.copy_timing(image, deghost)
    return deghost

def full_phs_corr(image, pct=30., allchan=True):
    if hasattr(image, 'n_chan') and allchan:
        corr_list = []
        for c in range(image.n_chan):
            image.use_membuffer(c)
            cf = solve_coefs_l1(image, pct=pct, zero_ramps=False)
            corr_list.append(undistort_img(image, cf[0], cf[1], n1p_fact=2.))
            #corr_list.append(undistort_img(image, cf[0], 0.))
            InverseFFT_2().run(corr_list[-1])
            print cf
        unghosted = image._subimage(g1.sumsqr(corr_list,
                                              channel_gains=image.channel_gains))
##         unghosted = image._subimage(g1.sumsqr(corr_list))
        g1.copy_timing(image, unghosted)
        return unghosted
    else:
        a1,a0 = solve_coefs_l1(image, pct=pct)
        unghosted = undistort_img(image, a1, a0)
        InverseFFT_2().run(unghosted)
        print a1, a0
        return unghosted

def unbal_phs_corr(image, pct=30., allchan=True):
    if hasattr(image, 'n_chan') and allchan:
        corr_list = []
        for c in range(image.n_chan):
            image.use_membuffer(c)
            cf = solve_coefs_l1(image, pct=pct, zero_ramps=False)
            corr_list.append(planar_phs_subtract(image, *cf))
            InverseFFT_2().run(corr_list[-1])
            print cf
        unghosted = image._subimage(g1.sumsqr(corr_list,
                                              channel_gains=image.channel_gains))
        return unghosted
    else:
        a1,a0 = solve_coefs_l1(image, pct=pct, zero_ramps=False)
        unghosted = planar_phs_subtract(image, a1, a0)
        InverseFFT_2().run(unghosted)
        print a1, a0
        return unghosted
            


def L1cost(x,y,m):
    b = N.median(y-m*x)
    sgn = N.sign(y - (m*x + b))
    return (x*sgn).sum(), b

def medfit(x,y):
    npt = y.shape[0]
    if npt < 2:
        print "impossible to solve"
        return (0.,0.,1e30)
    (b1,mm,chisq) = util.lin_regression(y, X=x)
    sigb = N.sqrt(chisq / (npt * N.power(x,2).sum() - x.sum()**2))
    m1 = mm
    print "stdev:", sigb
    f1,b = L1cost(x,y,m1)
    m2 = m1 + N.sign(f1)*(3 * sigb)
    f2,_ = L1cost(x,y,m2)
    print "initial bracket vals:", (f1,f2)
    while(f1*f2 > 0):
        mm = 2*m2 - m1
        m1 = m2
        f1 = f2
        m2 = mm
        f2,_ = L1cost(x,y,m2)
    sigb *= 0.01
    fstore = 0.
    reps = 0
    while abs(m2-m1) > sigb:
        mm = 0.5*(m1+m2)
        if mm==m1 or mm==m2:
            break
        f,b = L1cost(x,y,mm)
        if f==fstore:
            reps += 1
##             if reps > 2:
##                 print "breaking due to repeat f(b) hits"
##                 break
        else:
            fstore = f
            reps = 0
        if f*f1 > 0:
            f1 = f
            m1 = mm
        else:
            f2 = f
            m2 = mm
    m = mm
    absdev = N.abs(y - (m*x + b)).sum()
    print "reps:", reps
    return b,m,absdev


def plot_coef_goodness(image, target_a1):
    pcts = N.linspace(1,100,100)
    f1 = P.figure()
    ax1 = f1.add_subplot(111)
    f2 = P.figure()
    ax2 = f2.add_subplot(111)
    for n in range(image.n_chan):
        image.load_chan(n)
        a1 = target_a1[n]
        err_l2 = []
        err_l1 = []
        for p in pcts:
            cf_l1 = solve_coefs_l1(image, pct=p) #, zero_ramps=False)
            cf_l2 = solve_coefs_l2(image, pct=p) #, zero_ramps=False)
            err_l1.append( a1-cf_l1[0] )
            err_l2.append( a1-cf_l2[0] )
        err_l1 = N.array(err_l1)/(.01 * a1)
        err_l2 = N.array(err_l2)/(.01 * a1)
        ax1.plot(pcts, err_l1, unique_line_style(n), label='chan %d'%n)
        ax2.plot(pcts, err_l2, unique_line_style(n), label='chan %d'%n)
        bst_l1 = N.sort(N.abs(err_l1))
        bst_l2 = N.sort(N.abs(err_l2))
        pts_l1 = N.array([ (N.abs(err_l1)==b).nonzero()[0][0]
                           for b in bst_l1[:10] ])
        pts_l2 = N.array([ (N.abs(err_l2)==b).nonzero()[0][0]
                           for b in bst_l2[:10] ])
        ax1.plot(pcts[pts_l1],err_l1[pts_l1], 'ro', alpha=0.5, label='__nolabel__')
        ax2.plot(pcts[pts_l2],err_l2[pts_l2], 'ro', alpha=0.5, label='__nolabel__')
    ax1.set_ylim(-10,10)
    ax2.set_ylim(-10,10)
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    ax1.set_title('l1 norm')
    ax2.set_title('l2 norm')
    P.show()



line_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
line_styles = ['-', '--', '-.', ':']
def unique_line_style(n):
    num_colors = len(line_colors)
    num_styles = len(line_styles)
    max_types = num_colors * num_styles

    cnum = (n%max_types) % num_colors
    snum = (n%max_types) / num_colors
    return line_colors[cnum]+line_styles[snum]

class InverseFFT(Operation):

    def run(self, image):
        N2,N1 = image.shape[-2:]
        phsn2 = N.exp(-2.j*N.pi*N.arange(N2)*(N2/2 - 0.5)/N2)
        phsn1 = N.exp(-2.j*N.pi*N.arange(N1)*(N1/2 - 0.5)/N1)
        pfac = phsn2[:,None]*phsn1[None,:]
        #pfac = N.ones(N2)[:,None]*phsn1[None,:]
        for vol in image:
            vol[:] = ifft2(pfac*vol[:], shift=False)

class ZeroRamps(Operation):

    def run(self, image):
        nr = image.n_ramp
        nf = image.n_flat
        evn_sl = [slice(None)]*(image.ndim-1)
        odd_sl = [slice(None)]*(image.ndim-1)
        evn_sl[-1] = slice(0,None,2)
        odd_sl[-1] = slice(1,None,2)
        image.data[tuple(evn_sl) + (slice(0,nr+1),)] = 0.
        image.data[tuple(evn_sl) + (slice(nf+1,None),)] = 0.
        image.data[tuple(odd_sl) + (slice(0,nr),)] = 0.
        image.data[tuple(odd_sl) + (slice(nf,None),)] = 0.
