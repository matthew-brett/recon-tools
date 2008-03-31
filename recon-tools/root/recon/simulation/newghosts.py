import numpy as N, pylab as P
from recon import imageio, util
from recon.fftmod import fft1, ifft1
from recon.scanners import siemens
from recon.operations.InverseFFT import InverseFFT
from recon.operations.ForwardFFT import ForwardFFT
from recon.operations.ReorderSlices import ReorderSlices
from recon.operations.ViewImage import ViewImage
from recon.operations import UnbalPhaseCorrection_3d as UBPC
from recon.operations import UnbalPhaseCorrection_dev as UBPC_dev
from recon.operations.ComputeFieldMap import build_3Dmask


from recon.simulation import ghosts as g1


def plot_regions(image):
    (a1, a0) = solve_coefs(image)


    Tr = float(image.T_ramp)
    Tf = float(image.T_flat)
    T0 = float(image.T0)
    N1 = image.shape[-1] + 1
    As = Tf + Tr - T0**2/Tr

    nr = image.n_ramp
    nf = image.n_flat

    delF = 2*N.pi * 1./(Tf + 2*Tr - 2*T0)
    sigma = a1/delF
    fn = N.arange(N1-1, dtype=N.float64)

    treg1 = fn <= nr    
    treg2 = (fn > nr) & (fn <= nf)
    treg3 = fn > nf

    tn = N.empty(fn.shape, fn.dtype)
    tn[treg1] = N.power(T0**2 + 2*fn[treg1]*Tr*As/N1, 0.5)
    tn[treg2] = fn[treg2]*As/N1 + Tr/2 + T0**2/(2*Tr)
    tn[treg3] = (2*Tr+Tf) - N.power(2*Tr*(As + T0**2/(2*Tr) - \
                                          fn[treg3]*As/N1), 0.5)

    print T0, Tr, Tf
    print nr, nf
    print sigma

    reg1 = tn <= Tr
    reg2 = (tn > Tr) & (tn <= Tr+sigma)
    reg3 = (tn > Tr+sigma) & (tn <= Tr+Tf)
    reg4 = (tn > Tr+Tf) & (tn <= Tr+Tf+sigma)
    reg5 = tn > Tr+Tf+sigma
    Ttot = 2*Tr + Tf
    fr1 = (tn - a1/(2*delF))/Tr
    fr2 = 1 - (delF/(a1*Tr))*N.power(a1/delF - tn + Tr, 2.0)
    fr3 = N.ones(tn.shape)
    fr4 = 1 - (delF/(a1*Tr))*N.power(tn + Tr - Ttot, 2.0)
    fr5 = -(tn-Ttot-a1/(2*delF))/Tr

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
    ax.set_ylim((0,1.5))
    P.show()


def phs_surf(image, pct):
    rdata = image.ref_data.copy()
    if len(rdata.shape) > 3:
        rdata = rdata[0]
    
    Q3, N2, Q1 = rdata.shape
    Tr = float(image.T_ramp)
    T0 = float(image.T0)
    Tf = float(image.T_flat)
    As = Tf + Tr - T0**2/Tr

    nr = int((Q1-1)*(Tr**2 - T0**2)/(2*Tr*As))
    nf = int((Q1-1)*(2*Tf*Tr + Tr**2 - T0**2)/(2*Tr*As))
##     nr = int(Q1*(Tr**2 - T0**2)/(2*Tr*As))
##     nf = int(Q1*(2*Tf*Tr + Tr**2 - T0**2)/(2*Tr*As))

    rdata[...,:nr+1] = 0.
    rdata[...,nf+1:] = 0.
    if (nr > 1) and (nf < Q1-1):
        rdata = g1.rsmooth(rdata, nr, nf)
    inv_ref = ifft1(rdata)
    # this is S[u+1]S*[u] (but ref raster is backwards from epi, so change sign)
    #inv_ref = inv_ref[:,1:,:] * N.conjugate(inv_ref[:,:-1,:])
    inv_ref = inv_ref[:,:-1,:] * N.conjugate(inv_ref[:,1:,:])

    irmask = build_3Dmask(N.abs(inv_ref), 0.1)
    fract_pts = irmask.sum()/float(Q3*(N2-1)*Q1)

    phs_vol = util.unwrap_ref_volume(inv_ref)
    phs_mean, mask = UBPC_dev.mean_and_mask(phs_vol[:,0,:],
                                            phs_vol[:,1,:],
                                            pct*fract_pts, N.arange(Q3))
    
    return phs_mean, mask

def svd_solve(phs, mask):
    # phs is shaped (Q3, 2, Q1) where
    # row 0 corresponds to (-1)^(u+1) = -1 (u=0)
    # row 1 corresponds to (-1)^(u+1) = +1 (u=1)
    # phs.flatten() is Q3 repeats of (+1/-1)*( phs-line-to-fit )
    Q3,_,Q1 = phs.shape
    phs.shape = (Q3*2*Q1)
    mask.shape = (Q3*2*Q1)
    A = N.zeros((Q3*2*Q1,2), N.float64)
    # makes 2*Q3 rows of [0,1,2,...Q1-1] for the a1 term
    q1_idx = N.outer(N.ones(2*Q3), N.arange(Q1))
    # this is all ones for the a0 term
    q0_idx = N.ones(Q3*2*Q1)
    # this changes sign for every other group of Q1 pts
    usign = N.repeat(-util.checkerline(Q3*2), Q1)
    A[:,0] = 2.0 * usign * q1_idx.flatten()
    A[:,1] = 2.0 * usign * q0_idx

    nz = mask.nonzero()[0]
    A = A[nz]
    phs = phs[nz]

    [u,s,vt] = N.linalg.svd(A, full_matrices=0)
    V = N.dot(vt.transpose(), N.dot(N.diag(1/s), N.dot(u.transpose(), phs)))
    return V


def solve_coefs(image, pct=30.):
    return svd_solve(*phs_surf(image, pct))

def perturbation_phs(image, a1, a0):
    
    Q3, Q2, Q1 = image.shape[-3:]
    N2, N1 = image.shape[-2:]

    T0 = float(image.T0)
    Tr = float(image.T_ramp)
    Tf = float(image.T_flat)
    As = Tf + Tr - T0**2/Tr

    nr = image.n_ramp
    nf = image.n_flat

    # this is (-1)^(n2)
    # actually this can be reduced to two cases: [1, -1]
    n2sign = N.array([1.0, -1.0])
    
    # fn is shaped (2, N1)
    fn = N.outer(0.5*(1-n2sign)*(N1-1), N.ones(N1)) + \
         N.outer(n2sign, N.arange(N1))
    
    # soln_plane is shaped (Q3, Q1)
    soln_line = a1*N.arange(Q1) + a0    

    delF = 2*N.pi * 1./(Tf + 2*Tr - 2*T0)
    sigma = a1/delF
    print sigma

    treg1 = fn <= nr    
    treg2 = (fn > nr) & (fn <= nf)
    treg3 = fn > nf

    tn = N.empty(fn.shape, fn.dtype)
    tn[treg1] = N.power(T0**2 + 2*fn[treg1]*Tr*As/N1, 0.5)
    tn[treg2] = fn[treg2]*As/(N1-1) + Tr/2 + T0**2/(2*Tr)
    tn[treg3] = (2*Tr+Tf) - N.power(2*Tr*(As + T0**2/(2*Tr) - \
                                          fn[treg3]*As/N1), 0.5)

##     tn[treg1] = N.power(T0**2 + 2*fn[treg1]*Tr*As/N1, 0.5)
##     tn[treg2] = fn[treg2]*As/N1 + Tr/2 + T0**2/(2*Tr)
##     tn[treg3] = (2*Tr+Tf) - N.power(2*Tr*(As + T0**2/(2*Tr) - \
##                                           fn[treg3]*As/N1), 0.5)


    reg1 = tn <= Tr
    reg2 = (tn > Tr) & (tn <= Tr+sigma)
    reg3 = (tn > Tr+sigma) & (tn <= Tr+Tf)
    reg4 = (tn > Tr+Tf) & (tn <= Tr+Tf+sigma)
    reg5 = tn > Tr+Tf+sigma
    
    # phs will be shaped (2, N1, Q1)
    # so it will be soln_line (None, None, Q1) *
    # some function shaped (2, N1, None)
    Ttot = 2*Tr + Tf
    phs = N.empty((2,N1,Q1), N.float64)
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
    phs = -soln_line[None,None,:] * box[:,:,None]
    return phs


def perturbation_kernel(phs):
    _,N1,Q1 = phs.shape
    #n1_ax = N.linspace(-N1/2., N1/2., N1, endpoint=False)
    n1_ax = N.linspace(0., N1, N1, endpoint=False)
    chk = util.checkerboard(N1,N1)
    n1_by_q1 = -2.j*N.pi*n1_ax[:,None]*n1_ax[None,:]/N1

    #util.shift(phs, -Q1/2)
    zarg = 1.j*phs + n1_by_q1[None,:,:]

    return ifft1(N.exp(zarg), shift=False)*chk

def undistort_img(image, a1, a0):
    phs = perturbation_phs(image, a1, a0)
    f = perturbation_kernel(phs)
    sn = N.empty(image.shape, image[:].dtype)

    dslice = [slice(None)] * image.ndim
    dslice1 = [slice(None)] * (image.ndim+1)
    # this is a place-holder for the n1 dimension
    dslice1[-2] = None

    for n in [0, 1]:
        # get even rows [0,2,...] for n=0, odd rows for n=1
        dslice[-2] = slice(n, None, 2)
        dslice1[-3] = slice(n, None, 2)
        # multiply n1xn1p inverse kernel with data sliced as
        # (nvol*nsl*evn/odd-rows) x (n1xn1p) where n1 is a dummy dimension
        # then sum over n1p dimension to reduce corrected data to shape
        # (nvol*nsl*evn/odd-rows) x n1
        sn[dslice] = (N.linalg.inv(f[n]) * image[dslice1]).sum(axis=-1)
    unghosted = image._subimage(sn)
    g1.copy_timing(image, unghosted)
    return unghosted

def full_phs_corr(image, pct=30., allchan=True):
    if hasattr(image, 'n_chan') and allchan:
        corr_list = []
        for c in range(image.n_chan):
            image.use_membuffer(c)
            cf = solve_coefs(image, pct=pct)
            corr_list.append(undistort_img(image, *cf))
            InverseFFT().run(corr_list[-1])
            print cf
        unghosted = image._subimage(g1.sumsqr(corr_list))
        g1.copy_timing(image, unghosted)
        return unghosted
    else:
        a1,a0 = solve_coefs(image, pct=pct)
        unghosted = undistort_img(image, a1, a0)
        InverseFFT().run(unghosted)
        print a1, a0
        return unghosted

