import numpy as N, pylab as P
from recon import imageio, util
from recon.fftmod import fft1, ifft1
from recon.scanners import siemens
from recon.operations.InverseFFT import InverseFFT
from recon.operations.ForwardFFT import ForwardFFT
from recon.operations.ReorderSlices import ReorderSlices
from recon.operations.ViewImage import ViewImage
from recon.operations import UnbalPhaseCorrection as UBPC
from recon.operations.ComputeFieldMap import build_3Dmask

def plot_regions(Tr, T0, Tf, N1):
    (Tr, Tf, T0) = map(float, (Tr, Tf, T0))
    delT = (Tr + Tf - (T0**2)/Tr)/(N1-1)
    (Tr, T0, Tf) = map(float, (Tr, T0, Tf))
    n1 = int(N.floor( (Tr/2. - T0**2/(2.*Tr))/delT ))
    n2 = int(N.floor( (Tr/2. + Tf - T0**2/(2.*Tr))/delT ))
    r1 = N.arange(0, n1+1)
    r2 = N.arange(n1+1, n2+1)
    r3 = N.arange(n2+1, N1)
    fn = N.arange(N1)
    fr1 = N.power( (T0/Tr)**2 + 2*fn*delT/Tr, 0.5)
    fr2 = N.ones(N1)
    fr3 = N.power( 2 + 2*Tf/Tr - (T0/Tr)**2 - 2*fn*delT/Tr, 0.5)
    P.plot(fr1)
    P.plot(fr2)
    P.plot(fr3)
    P.plot(r1, fr1[r1], 'ro')
    P.plot(r2, fr2[r2], 'ro')
    P.plot(r3, fr3[r3], 'ro')
    P.show()

## def simphs(image, cf_evn, cf_odd):

##     Q3, Q2, Q1 = image.shape[-3:]
##     N2, N1 = image.shape[-2:]

##     T0 = float(image.T0)
##     Tr = float(image.T_ramp)
##     Tf = float(image.T_flat)
##     delT = (Tr + Tf - (T0**2)/Tr)/(N1 - 1.0)
##     # this is (-1)^(n2)
##     n2sign = util.checkerline(N2)
    
##     # fn is shaped (N2, N1)
##     fn = N.outer(0.5*(1-n2sign)*(N1-1), N.ones(N1)) + \
##          N.outer(n2sign, N.arange(N1))
    
##     # soln_plane is shaped (Q3, Q1)
##     Qplane = N.indices((Q3, Q1), dtype=N.float64)
##     soln_evn = cf_evn[1]*Qplane[0] + cf_evn[0]*(Qplane[1]-Q1/2) + cf_evn[2]
##     soln_odd = cf_odd[1]*Qplane[0] + cf_odd[0]*(Qplane[1]-Q1/2) + cf_odd[2]
##     #soln_plane = a3*Qplane[0] + a1*(Qplane[1]-Q1/2) + a0
    
##     nr = int( (Tr/2. - T0**2/(2.*Tr))/delT )
##     nf = int( (Tr/2. + Tf - T0**2/(2.*Tr))/delT )
##     print nr, nf

##     reg1 = fn <= nr
##     reg2 = (fn > nr) & (fn <= nf)
##     reg3 = fn > nf
    
##     # phs will be shaped (Q3, N2, N1, Q1)
##     # so it will be soln_plane (Q3, None, None, Q1) *
##     # some function shaped (None, N2, N1, None)
##     phs = N.empty((Q3,N2,N1,Q1), N.float64)
##     fr1 = N.power(( (T0/Tr)**2 + 2*fn*delT/Tr ), 0.5) * -n2sign[:,None]
##     fr2 = N.ones((N2,N1)) * -n2sign[:,None]
##     sqrt_term = 2 + 2*Tf/Tr - (T0/Tr)**2 - 2*fn*delT/Tr
##     sqrt_term = N.where(sqrt_term < 0, 0, sqrt_term)
##     fr3 = N.power(sqrt_term, 0.5) * -n2sign[:,None]
##     box = N.empty((N2,N1), N.float64)
##     box[reg1] = fr1[reg1]
##     box[reg2] = fr2[reg2]
##     box[reg3] = fr3[reg3]
##     phs = N.empty((Q3, N2, N1, Q1), N.float64)
##     phs[:,0::2,:,:] = soln_evn[:,None,None,:] * box[None,0::2,:,None]
##     phs[:,1::2,:,:] = -soln_odd[:,None,None,:] * box[None,1::2,:,None]
##     return phs
def simphs(image, a1, a3, a0):

    Q3, Q2, Q1 = image.shape[-3:]
    N2, N1 = image.shape[-2:]

    T0 = float(image.T0)
    Tr = float(image.T_ramp)
    Tf = float(image.T_flat)
    delT = (Tr + Tf - (T0**2)/Tr)/(N1 - 1.0)
    # this is (-1)^(n2)
    n2sign = util.checkerline(N2)
    
    # fn is shaped (N2, N1)
    fn = N.outer(0.5*(1-n2sign)*(N1-1), N.ones(N1)) + \
         N.outer(n2sign, N.arange(N1))
    
    # soln_plane is shaped (Q3, Q1)
    Qplane = N.indices((Q3, Q1), dtype=N.float64)
    soln_plane = a3*Qplane[0] + a1*(Qplane[1]-Q1/2) + a0
    
    nr = int( (Tr/2. - T0**2/(2.*Tr))/delT )
    nf = int( (Tr/2. + Tf - T0**2/(2.*Tr))/delT )
    print nr, nf

    reg1 = fn <= nr
    reg2 = (fn > nr) & (fn <= nf)
    reg3 = fn > nf
    
    # phs will be shaped (Q3, N2, N1, Q1)
    # so it will be soln_plane (Q3, None, None, Q1) *
    # some function shaped (None, N2, N1, None)
    phs = N.empty((Q3,N2,N1,Q1), N.float64)
    fr1 = N.power(( (T0/Tr)**2 + 2*fn*delT/Tr ), 0.5) * -n2sign[:,None]
    fr2 = N.ones((N2,N1)) * -n2sign[:,None]
    sqrt_term = 2 + 2*Tf/Tr - (T0/Tr)**2 - 2*fn*delT/Tr
    sqrt_term = N.where(sqrt_term < 0, 0, sqrt_term)
    fr3 = N.power(sqrt_term, 0.5) * -n2sign[:,None]
    box = N.empty((N2,N1), N.float64)
    box[reg1] = fr1[reg1]
    box[reg2] = fr2[reg2]
    box[reg3] = fr3[reg3]
    phs = soln_plane[:,None,None,:] * box[None,:,:,None]
    return phs

def kern_f2(phs, q3):
    # phs will be shaped (Q3, N2, N1, Q1)
    # F will be shaped (N2, N1, Np2, Np1)
    # will be a product of:
    # exp(i*fa[N2,   None, Np2,  None, Q2, None]) --> this reduces to I(64x64)
    # exp(i*fb[None, N1,   None, Np1,  None, Q1])
    # exp(i*phs[@q3, N2,   N1,   None, None, None, Q1])
    # then summed over q1 and q2 axes
    Q3, N2, N1, Q1 = phs.shape

    n1_ax = N.linspace(-N1/2., N1/2., N1, endpoint=False)
    # fb is (N1, Q1) 
    fb = -2.j*N.pi*n1_ax[:,None]*n1_ax[None,:]/N1
    # fz is (N2, N1, Q1) --> (N2, N1, NP1) upon IFFT
    zarg = 1.j*phs[q3,:,:,:]+fb[None,:,:]
    fz = ifft1(N.exp(zarg), axis=-1) * N1
    return fz

def kern_full(phs):
    Q3,N2,N1,Q1 = phs.shape
    n1_ax = N.linspace(-N1/2., N1/2., N1, endpoint=False)
    # fb is (N1, Q1)
    fb = -2.j*N.pi*(n1_ax[:,None])*(n1_ax[None,:])/N1
    # fz is (Q3, N2, N1, Q1) --> (Q3, N2, N1, NP1) upon IFFT
    zarg = 1.j*phs + fb[None,None,:,:]
    fz = ifft1(N.exp(zarg), axis=-1)
    return fz

def deghost(image, a1, a3, a0):
    (Q3, N2, Q1) = image.shape[-3:]
    Qplane = N.indices((Q3,Q1), dtype=N.float64)
    n2sign = util.checkerline(N2)
    soln_plane = a3*Qplane[0] + a1*(Qplane[1] - Q1/2) + a0
    soln_plane = soln_plane[:,None,:] * n2sign[None,:,None]
    soln_phs = N.exp(1.j*soln_plane)
    scorr = util.apply_phase_correction(image[:], soln_phs)
    deghost = image._subimage(scorr)
    copy_timing(image, deghost)
    return deghost


def cook_img(image, N1):
    #nx = image.shape[-1]
    image.T0 = 42.
    image.T_ramp = 120.
    image.T_flat = 240.
    #image.delT = 2*3.1
    #image.delT = (image.T_ramp + image.T_flat - (image.T0**2)/image.T_ramp)/(N1 - 1.0)
    #InverseFFT().run(image)
    #image.setData(image[...,nx/2-N1/2:nx/2+N1/2])
    #ForwardFFT().run(image)

def copy_timing(src, dest):
    dest.T_ramp = src.T_ramp
    dest.T0 = src.T0
    dest.Ti = src.Ti
    dest.Nc = src.Nc
    dest.T_flat = src.T_flat
    dest.n_ramp = src.n_ramp
    dest.n_flat = src.n_flat

def distort_img(image, a1, a3, a0):
    # phs is (Q3, N2, N1, Q1)
    (Q3, N2, Q1) = image.shape[-3:]
    phs = simphs(image, a1, a3, a0)
    #phs = simphs(image, cf_evn, cf_odd)
    #s_hat = N.empty(image.shape, image[:].dtype)
    rdata = N.empty((Q3, 3, Q1), N.complex64)
    # repeat PE line 0 three times for "ref data"
    mid_row = [slice(None)]*image.ndim
    mid_row[-2] = N2/2
    if len(mid_row) > 3:
        mid_row[0] = 0
    rdata[:] = (image.data[mid_row])[:,None,:]
    f = kern_full(phs)
    dslicer = [slice(None)] * (image.ndim+1)
    dslicer[-2] = None
    s_hat = (f * image.data[dslicer]).sum(axis=-1)
    rdata = (f[:,31:34,:,:] * rdata[:,:,None,:]).sum(axis=-1)
    ghosted = image._subimage(s_hat)
    ghosted.ref_data = rdata
    copy_timing(image, ghosted)
    return ghosted

def undistort_img(image, a1, a3, a0):
    (Q3, N2, Q1) = image.shape[-3:]
    phs = simphs(image, a1, a3, a0)
    #phs = simphs(image, cf_evn, cf_odd)    
    f = kern_full(phs)
    sn = N.empty(image.shape, image[:].dtype)
    snslice = [slice(None)] * image.ndim
    dslice1 = [slice(None)] * (image.ndim+1)
    dslice1[-2] = None
    for q3 in range(Q3):
        print "correcting slice", q3
        snslice[-3] = q3
        dslice1[-4] = q3
        for n2 in range(N2):
            snslice[-2] = n2
            dslice1[-3] = n2
            sn[snslice] = (N.linalg.inv(f[q3,n2]) * \
                           image[dslice1]).sum(axis=-1)
    unghosted = image._subimage(sn)
    copy_timing(image, unghosted)
    return unghosted

def solve_params(image, pct=75.0, cutn1=True, smooth=False, window=False):
    Q3, N2, Q1 = image.shape[-3:]
    Tr = float(image.T_ramp)
    T0 = float(image.T0)
    Tf = float(image.T_flat)
    delT = (Tr + Tf - (T0**2)/Tr)/(Q1 - 1.0)    
    n1 = int(N.floor( (Tr/2. - T0**2/(2.*Tr))/delT )) + 1
    n2 = int(N.floor( (Tr/2. + Tf - T0**2/(2.*Tr))/delT )) + 1
    if len(image.ref_data.shape) > 3:
        rdata = image.ref_data[0].copy()
    else:
        rdata = image.ref_data.copy()
    irdata = N.abs(ifft1(rdata))
    irmask = build_3Dmask(irdata, 0.1)
    (Q3,_,Q1) = rdata.shape
    # this is the ratio of "available" data to the size of the plane
    pct_scale = irmask[:,0,:].sum() / ((Q3-2)*(Q1-2))
    #pct_scale = 1.0
    if cutn1:
        rdata[:,:,:n1] = 0.
        rdata[:,:,n2:] = 0.
    if cutn1 and smooth:
        rdata = rsmooth(rdata, n1, n2)
    if not cutn1 and not smooth and window:
        wsmooth(rdata, n1, n2)
    inv_ref = ifft1(rdata)
    inv_ref = inv_ref[:,:-1,:] * N.conjugate(inv_ref[:,1:,:])
    phs_vol = util.unwrap_ref_volume(inv_ref)
    # reverse evn/odd segments here to make sign consistent with old SVD soln
    #phs_vol = N.take(phs_vol, [1, 0], axis=-2)
    p_evn = phs_vol[:,0,:].copy()
    #p_evn.shape = (Q3, 1, Q1)
    p_odd = phs_vol[:,1,:].copy()
    #p_odd.shape = (Q3, 1, Q1)
    #q1_mask = N.empty((Q3,2,Q1))
    phs_mean, q1_mask = UBPC.mean_and_mask(p_evn, p_odd,
                                               pct*pct_scale, N.arange(Q3))

    a1,a3,a0 = UBPC.solve_phase_3d(phs_mean, q1_mask)
    return (a1,a3,a0)

def solve_params_epi(epi, svdsize=3, pct=25., cutn1=True, smooth=False):
    refShape = epi.ref_data.shape[-3:]
    volShape = epi.shape[-3:]
    #alpha,beta,_,ralpha= epi.epi_trajectory()
    # assume xleave == 1
    n_conj_rows = refShape[-2] - 1
    if len(epi.ref_data.shape) > 3:
        rdata = epi.ref_data[0].copy()
    else:
        rdata = epi.ref_data.copy()
    if cutn1:
        print "cutting"
        N1 = epi.shape[-1]
        Tr = float(epi.T_ramp)
        T0 = float(epi.T0)
        Tf = float(epi.T_flat)
        As = (Tr + Tf - (T0**2)/Tr)
        nr = int( (N1-1)*(Tr**2 - T0**2)/(2*Tr*As) )
        nf = int( (N1-1)*(2*Tf*Tr + Tr**2 - T0**2)/(2*Tr*As) )
        rdata[...,:nr+1] = 0
        rdata[...,nf+1:] = 0
        if smooth and (nr > 1) and (nf < N1-1):
            print "smoothing"
            rsmooth(rdata, nr, nf)
        
    inv_ref = ifft1(rdata)
    (Q3,_,Q1) = epi.shape[-3:]
    #pct_scale = 1.0
    inv_ref = inv_ref[:,:-1,:] * N.conjugate(inv_ref[:,1:,:])
    irmask = build_3Dmask(N.abs(inv_ref), 0.1)
    pct_scale = irmask.sum() / N.product(irmask.shape)
    #pos_order = (ralpha[:n_conj_rows] > 0).nonzero()[0]
    #neg_order = (ralpha[:n_conj_rows] < 0).nonzero()[0]
    pos_order = 0
    neg_order = 1
    phs_vol = util.unwrap_ref_volume(inv_ref)
    phs_mean, q1_mask = UBPC.mean_and_mask(phs_vol[:,pos_order,:],
                                               phs_vol[:,neg_order,:],
                                               pct*pct_scale,
                                               N.arange(refShape[-3]))
    if svdsize == 3:
        coefs = UBPC.solve_phase_3d(phs_mean, q1_mask)
    else:
        coefs = UBPC.solve_phase_6d(phs_mean, q1_mask)
    return coefs

def full_phs_corr(image, **kwargs):
    # if image.n_chan exists, returns a sum-of-squares image
    if hasattr(image, 'n_chan'):
        corr_list = []
        for c in range(image.n_chan):
            image.use_membuffer(c)
            cf = solve_params(image, **kwargs)
            corr_list.append(undistort_img(image, *cf))
            InverseFFT().run(corr_list[-1])
            print cf
        unghosted = image._subimage(sumsqr(corr_list))
        copy_timing(image, unghosted)
        return unghosted
    else:
        a1,a3,a0 = solve_params(image, **kwargs)
        unghosted = undistort_img(image, a1, a3, a0)
        InverseFFT().run(unghosted)
        print a1,a3,a0
        return unghosted

def unbal_phs_corr(image, **kwargs):
    # if image.n_chan exists, returns a sum-of-squares image
    if hasattr(image, 'n_chan'):
        corr_list = []
        for c in range(image.n_chan):
            image.use_membuffer(c)
            cf = solve_params(image, **kwargs)
            corr_list.append(deghost(image, *cf))
            InverseFFT().run(corr_list[-1])
            print cf
        unghosted = image._subimage(sumsqr(corr_list))
        copy_timing(image, unghosted)
        return unghosted
    else:
        a1,a3,a0 = solve_params(image, **kwargs)
        unghosted = deghost(image, a1, a3, a0)
        InverseFFT().run(unghosted)
        print a1,a3,a0
    return unghosted

def graph_pct_error(image, A1, A3, A0, **kwargs):
    pctiles = N.linspace(5,100,50,endpoint=False)
    a1_err = []
    a3_err = []
    a0_err = []
    for p in pctiles:
        (sa1, sa3, sa0) = solve_params(image, pct=p, **kwargs)
        a1_err.append( abs(100.*(A1+sa1)/A1) )
        a3_err.append( abs(100.*(A3+sa3)/A3) )
        a0_err.append( abs(100.*(A0+sa0)/A0) )
    a1_err = N.array(a1_err)
    a3_err = N.array(a3_err)
    a0_err = N.array(a0_err)
    P.plot(pctiles, a1_err, 'b', label='a1 err')
    P.plot(pctiles, a3_err, 'g', label='a3_err')
    P.plot(pctiles, a0_err, 'r', label='a0_err')
    P.plot(pctiles, N.zeros(50), 'k--', label='_nolegend_')
    ax = P.gca()
    ax.set_ylim((-1.0, 5.0))
    P.legend(loc=2)
    #P.show()

def graph_err(image, A1, A3, A0):
    (na3, na0) = map(lambda x: x * 5*N.random.randn(1), (A3, A0))
    max_na1 = 2*N.pi/image.idim
    na1 = 2*max_na1*(N.random.random_sample(1) - 0.5)
    ghosted = distort_img(image, na1, na3, na0)
    graph_pct_error(ghosted, na1, na3, na0, cutn1=True)
    ax = P.gca()
    ax.set_title("cut ksp, a1: %f, a3: %f, a0: %f"%(na1,na3,na0))
    P.figure()
    graph_pct_error(ghosted, na1, na3, na0, cutn1=True, smooth=True)
    ax = P.gca()
    ax.set_title("cut smth ksp, a1: %f, a3: %f, a0: %f"%(na1,na3,na0))
    P.figure()
    graph_pct_error(ghosted, na1, na3, na0, cutn1=False, window=True)
    ax = P.gca()
    ax.set_title("windowed ksp, a1: %f, a3: %f, a0: %f"%(na1,na3,na0))
    P.show()

def fsmooth(f, n1, n2):
    #avgr = N.exp(-N.power(N.arange(-1,2),2.0)/(2*.6**2))
    #avgr /= avgr.sum()
    avgr = N.sinc(-N.linspace(-1.5,1.5,3))
    avgr /= N.abs(avgr).sum()
    fs = f.copy()
    for x in range(n1-2,n1):
        fs[...,x] = (avgr*f[...,x:x+3]).sum(axis=-1)
    for x in range(n2,n2+2):
        fs[...,x] = (avgr*f[...,x-3:x]).sum(axis=-1)
    return fs

def rsmooth(f, n1, n2):
    s_size = 3
    avgr = N.ones(s_size, N.float64)/s_size
    fs = f.copy()
    for x in range(n1-s_size/2,n1):
        fs[...,x] = (avgr*f[...,x-s_size/2:x+s_size/2+1]).sum(axis=-1)
    for x in range(n2,n2+s_size/2):
        fs[...,x] = (avgr*f[...,x-s_size/2:x+s_size/2+1]).sum(axis=-1)
    return fs

def wsmooth(f, n1, n2):
    np = f.shape[-1]
    win1 = N.hanning(2*n1+1)
    win2 = N.hanning(2*(np-n2)+1)
    f[...,:n1] *= win1[:n1]
    f[...,n2:] *= win2[(np-n2)+1:]


def graph_epi_solns(epi, **kwargs):
    pcts = N.linspace(1,90,100)
    coef_list = []
    for p in pcts:
        kwargs['pct'] = p
        coef_list.append(list(solve_params(epi, **kwargs)))
    coef_list = N.array(coef_list).transpose()
    for n,cgraph in enumerate(coef_list):
        mn = cgraph.min()
        mx = cgraph.max()
        # relocate graphs so they range between [0,1]
        #P.plot(pcts, (cgraph-mn)/(mx-mn), label="a%d"%(n+1))
        #P.plot(pcts, cgraph, label="a%d"%(n+1))
        hist, bins = N.histogram(cgraph, bins=10)
        bigbin = (hist==hist.max()).nonzero()[0][0]
        binwid = (cgraph.max()-cgraph.min())/10.
        common = bins[0] + binwid*(2*bigbin+1)/2.
        w = (cgraph >= bins[0]+binwid*bigbin) & \
            (cgraph < bins[0]+binwid*(bigbin+1))
        print "a%d common val %f in neighborhood of:"%(n+1, common), pcts[w]
        pct_ex = N.abs(100.*(cgraph - common)/common)
        P.plot(pcts, pct_ex, label="a%d"%(n+1))
        P.plot(pcts[w], pct_ex[w], 'k.', alpha=0.5, label='_nolagend_')
        #print "min/max excursion from mean a%d:"%(n+1), pct_excursion.min(), pct_excursion.max()
    P.legend()
    P.show()


def svd3d_notoggle(phs, ptmask):
    Q3,Q1 = phs.shape
    A1, A3, A0 = (0,1,2)
    # build the full matrix first, collapse the zero-rows afterwards
    nrows = N.product(phs.shape)
    A = N.zeros((nrows, 3), N.float64)
    P = phs.copy()
    P.shape = (nrows,)
    ptmask = N.reshape(ptmask, (nrows,))
    q1_line = N.linspace(-Q1/2., Q1/2., Q1, endpoint=False)
    q3_line = N.linspace(0., Q3, Q3, endpoint=False)
    #n2sign = N.outer(checkerline(N2*Q3), N.ones(Q1))
    #n2sign = N.repeat(checkerline(N2*Q3), Q1)
    A[:,A1] = (2. * N.ones((Q3, Q1))*q1_line).flatten()
    A[:,A3] = 2. * N.repeat(q3_line, Q1)
    A[:,A0] = 2.
    
    nz1 = ptmask.nonzero()[0]
    A = A[nz1]
    P = P[nz1]
    
    [u,s,vt] = N.linalg.svd(A, full_matrices=0)
    V = N.dot(vt.transpose(), N.dot(N.diag(1/s), N.dot(u.transpose(), P)))
    return V

def write_list(img_list, fname):
    if img_list[0].ndim > 3:
        nchan_img = img_list[0].concatenate(img_list[1], newdim=False)
    else:
        nchan_img = img_list[0].concatenate(img_list[1], newdim=True)
    for img in img_list[2:]:
        nchan_img = nchan_img.concatenate(img)
    nchan_img.writeImage(fname)
    

def sumsqr(img_list, channel_gains=None):
    d = N.zeros(img_list[0].shape, N.float64)
    if channel_gains is None:
        channel_gains = N.ones(len(img_list))
    for n,img in enumerate(img_list):
        d += N.power(channel_gains[n]*img[:].real, 2.0) + \
             N.power(channel_gains[n]*img[:].imag, 2.0)
    return N.sqrt(d)

def write_sumsqr(img_list, fname):
    sumsqr_img = img_list[0]._subimage(sumsqr(img_list))
    sumsqr_img.writeImage(fname)


def pellipsoid(a1, a3, a0, np, pct):
    ca1 = pct*a1
    ca3 = pct*a3
    ca0 = pct*a0
    theta = N.random.rand(np)*N.pi
    theta.sort()
    phi = N.random.rand(np)*2*N.pi
    phi.sort()
    surf_pts = []
    for p in phi:
        for t in theta:
            surf_pts.append( ( a1 + ca1*N.sin(t)*N.cos(p),
                               a3 + ca3*N.sin(t)*N.sin(p),
                               a0 + ca0*N.cos(t) ) )
    return surf_pts

def eval_pts(epi, pts):
    qual_array = N.empty(len(pts), N.float64)
    for n,p in enumerate(pts):
        corr_list = []
        for c in range(epi.n_chan):
            epi.use_membuffer(c)
            corr_list.append(undistort_img(epi, p[0], p[1], p[2]))
            InverseFFT().run(corr_list[-1])
        qual_array[n] = eval_ghosts(corr_list)
    # find best point
    bpt = (qual_array == qual_array.min()).nonzero()[0][0]
    return pts[bpt]

def eval_ghosts(corr_list):
    #ROWS = range(10) + range(59,64)
    ROWS = [0,1,2,3,60,61,62,63]
    d = N.zeros(corr_list[0].shape, N.float64)
    for img in corr_list:
        d += N.power(img[:].real, 2.0) + N.power(img[:].imag, 2.0)
    meas = N.sqrt(d[:,ROWS,:]).sum()
    return meas
        


def drive_search(base_fname, np, pct_list=[0.5,1.0,1.5,2.0]):
    epi = imageio.readImage(file_lookup[base_fname]['epi'], vrange=(0,0))
    epi.run_op(ReorderSlices())
    coefs = solve_params(epi, smooth=True)
    best_points = []
    for pct in pct_list:
        pts = pellipsoid(coefs[0], coefs[1], coefs[2], np, pct/100.)
        best_points.append(eval_pts(epi, pts))
    print "original coefficients:", coefs
    for n in range(len(pct_list)):
        print "best pnt for %2.1f pct:"%pct_list[n], best_points[n]
    return coefs, best_points

def do_sim_and_real_recon(base_fname, pct):
    agems_fname = file_lookup[base_fname]['agems']
    epi_fname = file_lookup[base_fname]['epi']
    epi = imageio.readImage(epi_fname, vrange=(0,0))
    agems = imageio.readImage(agems_fname)
    epi.run_op(ReorderSlices())
    agems.run_op(ReorderSlices())
    copy_timing(epi, agems)
    coef_list = []
    ghosted_list = []
    if base_fname in rot_list:
        for c in range(agems.n_chan):
            agems.load_chan(c)
            agems[:] = N.swapaxes(agems[:], -1, -2)
    if base_fname in revx_list:
        for c in range(agems.n_chan):
            agems.load_chan(c)
            agems[:] = agems[...,::-1]
    

    for c in range(epi.n_chan):
        epi.use_membuffer(c)
        coef_list.append(solve_params(epi, pct=pct, smooth=True))
    for c in range(agems.n_chan):
        agems.use_membuffer(c)
        ghosted_list.append(distort_img(agems, *coef_list[c]))
    for img in ghosted_list: InverseFFT().run(img)
    write_list(ghosted_list, base_fname+"_sim.ghosted")
    for img in ghosted_list: ForwardFFT().run(img)
    sim_unbal_list = []
    sim_undist_list = []
    epi_unbal_list = []
    epi_undist_list = []
    for img in ghosted_list:
        sim_unbal_list.append(unbal_phs_corr(img, pct=pct, cutn1=False))
        sim_undist_list.append(full_phs_corr(img, pct=pct, smooth=True))
        #sim_undist_list.append(full_phs_corr(img, pct=pct, cutn1=False))

    write_list(sim_unbal_list, base_fname+"_sim.unbal")
    write_list(sim_undist_list, base_fname+"_sim.undist")
    write_sumsqr(sim_unbal_list, base_fname+"_sim.unbal.sumsqr")
    write_sumsqr(sim_undist_list, base_fname+"_sim.undist.sumsqr")
    #for c in range(epi.n_chan):
    for c in range(6):
        epi.use_membuffer(c)
        epi_unbal_list.append(unbal_phs_corr(epi, pct=pct, cutn1=False))
        epi_undist_list.append(full_phs_corr(epi, pct=pct, smooth=True))
        #epi_undist_list.append(full_phs_corr(epi, pct=pct, cutn1=False))
    write_list(epi_unbal_list, base_fname+".unbal")
    write_list(epi_undist_list, base_fname+".undist")
    write_sumsqr(epi_unbal_list, base_fname+".unbal.sumsqr")
    write_sumsqr(epi_undist_list, base_fname+".undist.sumsqr")
    agems.run_op(InverseFFT())
    agems.combine_channels()
    agems.subImage(0).writeImage(base_fname+".ref")
    return ghosted_list

def snr_meas(base_fname, threshfact):
    epifile = file_lookup[base_fname]['epi']
    d = siemens.parse_siemens_hdr(epifile)
    print "T_ramp: %d, T_flat: %d, T0: %d"%(d['T_ramp'], d['T_flat'], d['T0'])
    ref = imageio.readImage(base_fname+".ref.hdr")
    undist = imageio.readImage(base_fname+".undist.hdr")
    unbal = imageio.readImage(base_fname+".unbal.hdr")
    undist_ssq = imageio.readImage(base_fname+".undist.sumsqr.hdr")
    unbal_ssq = imageio.readImage(base_fname+".unbal.sumsqr.hdr")
    msk = build_3Dmask(ref[:], threshfact)
    
    sig_by_chan_undist = (undist[:] * msk[None,:,:,:]).sum(axis=-1).sum(axis=-1).sum(axis=-1)
    nz_by_chan_undist = (undist[:] * (1-msk[None,:,:,:])).sum(axis=-1).sum(axis=-1).sum(axis=-1)

    snr_by_chan_undist = sig_by_chan_undist/nz_by_chan_undist
    
    combined_snr_undist = (undist_ssq[:] * msk).sum() / (undist_ssq[:]*(1-msk)).sum()
    
    sig_by_chan_unbal = (unbal[:] * msk[None,:,:,:]).sum(axis=-1).sum(axis=-1).sum(axis=-1)
    nz_by_chan_unbal = (unbal[:] * (1-msk[None,:,:,:])).sum(axis=-1).sum(axis=-1).sum(axis=-1)

    snr_by_chan_unbal = sig_by_chan_unbal/nz_by_chan_unbal
    
    combined_snr_unbal = (unbal_ssq[:] * msk).sum() / (unbal_ssq[:]*(1-msk)).sum()
        
    ViewImage(title="planar phs corr").run(ref._subimage((1-msk)*unbal_ssq[:]))
    ViewImage(title="planar phs corr").run(ref._subimage(msk*unbal_ssq[:]))
    ViewImage(title="kernel phs corr").run(ref._subimage((1-msk)*undist_ssq[:]))
    ViewImage(title="kernel phs corr").run(ref._subimage(msk*undist_ssq[:]))
    P.plot(snr_by_chan_unbal, label="planar phs corr")
    P.plot(snr_by_chan_undist, label="kernel phs corr")
    P.legend()
    P.show()
    return snr_by_chan_unbal, snr_by_chan_undist, combined_snr_unbal, combined_snr_undist

def snr_meas_sim(base_fname, threshfact):
    epifile = file_lookup[base_fname]['epi']
    d = siemens.parse_siemens_hdr(epifile)
    print "T_ramp: %d, T_flat: %d, T0: %d"%(d['T_ramp'], d['T_flat'], d['T0'])
    ref = imageio.readImage(base_fname+".ref.hdr")
    undist = imageio.readImage(base_fname+"_sim.undist.hdr")
    undist.setData(undist[::2])
    unbal = imageio.readImage(base_fname+"_sim.unbal.hdr")
    unbal.setData(unbal[::2])
    undist_ssq = imageio.readImage(base_fname+"_sim.undist.sumsqr.hdr")
    undist_ssq.setData(undist_ssq[0])
    unbal_ssq = imageio.readImage(base_fname+"_sim.unbal.sumsqr.hdr")
    unbal_ssq.setData(unbal_ssq[0])
    msk = build_3Dmask(ref[:], threshfact)
    
    sig_by_chan_undist = (undist[:] * msk[None,:,:,:]).sum(axis=-1).sum(axis=-1).sum(axis=-1)
    nz_by_chan_undist = (undist[:] * (1-msk[None,:,:,:])).sum(axis=-1).sum(axis=-1).sum(axis=-1)

    snr_by_chan_undist = sig_by_chan_undist/nz_by_chan_undist
    
    combined_snr_undist = (undist_ssq[:] * msk).sum() / (undist_ssq[:]*(1-msk)).sum()
    
    sig_by_chan_unbal = (unbal[:] * msk[None,:,:,:]).sum(axis=-1).sum(axis=-1).sum(axis=-1)
    nz_by_chan_unbal = (unbal[:] * (1-msk[None,:,:,:])).sum(axis=-1).sum(axis=-1).sum(axis=-1)

    snr_by_chan_unbal = sig_by_chan_unbal/nz_by_chan_unbal
    
    combined_snr_unbal = (unbal_ssq[:] * msk).sum() / (unbal_ssq[:]*(1-msk)).sum()
        
    ViewImage(title="planar phs corr").run(ref._subimage((1-msk)*unbal_ssq[:]))
    ViewImage(title="planar phs corr").run(ref._subimage(msk*unbal_ssq[:]))
    ViewImage(title="kernel phs corr").run(ref._subimage((1-msk)*undist_ssq[:]))
    ViewImage(title="kernel phs corr").run(ref._subimage(msk*undist_ssq[:]))
    P.plot(snr_by_chan_unbal, label="planar phs corr")
    P.plot(snr_by_chan_undist, label="kernel phs corr")
    P.legend()
    P.show()
    
    return snr_by_chan_unbal, snr_by_chan_undist, combined_snr_unbal, combined_snr_undist

rot_list = [ 'feb13_1', 'feb13_2', 'feb25_1', 'feb25_2']
revx_list = [ 'feb13_1', 'feb13_2' ]
#data_dir = '/home/mike/bic_sandbox/trunk/testdata/siemens/'
data_dir = '/Users/miket/sandbox/trunk/testdata/siemens/'
file_lookup = {
    'feb28_4':
    { 'agems': data_dir+'feb28_2/RAW/meas_MID126_gre_field_mapping_FID2608.dat',
      'epi': data_dir+'feb28_2/RAW/meas_MID129_ep2d_pace_dynt_moco_FID2611.dat'
      },
    'feb28_3':
    { 'agems': data_dir+'feb28_2/RAW/meas_MID126_gre_field_mapping_FID2608.dat',
      'epi': data_dir+'feb28_2/RAW/meas_MID128_ep2d_pace_dynt_moco_FID2610.dat',
      },
    'feb28_2':
    { 'agems': data_dir+'feb28/Raw_phase_rot/fmap/meas_MID46_gre_field_mapping_FID2528.dat',
      'epi': data_dir+'feb28/Raw_phase_rot/Data/meas_MID49_ep2d_pace_dynt_moco_FID2531.dat'
      },
    'feb28_1':
    { 'agems': data_dir+'feb28/Raw_read_rot/fmap/meas_MID51_gre_field_mapping_FID2533.dat',
      'epi': data_dir+'feb28/Raw_read_rot/Data/meas_MID53_ep2d_pace_dynt_moco_FID2535.dat'
      },
    'feb25_2':
    { 'agems': '/media/puma/Users/miket/sandbox/trunk/testdata/siemens/feb25/RAW_brain/meas_MID46_gre_field_mapping_FID2420.dat',
      'epi': '/media/puma/Users/miket/sandbox/trunk/testdata/siemens/feb25/RAW_brain/meas_MID50_DanEPI_64x64_FID2424.dat'
      },
    'feb25_1':
    { 'agems': '/media/puma/Users/miket/sandbox/trunk/testdata/siemens/feb25/RAW_phantom/meas_MID34_gre_field_mapping_FID2408.dat',
      'epi': '/media/puma/Users/miket/sandbox/trunk/testdata/siemens/feb25/RAW_phantom/meas_MID36_DanEPI_64x64_FID2410.dat'
      },
    'feb13_2':
    { 'agems': data_dir+'feb13/meas_MID79_gre_field_mapping_FID2182.dat',
      'epi': data_dir+'feb13/meas_MID81_ep2d_pace_dynt_moco_FID2184.dat'
      },
    'feb13_1':
    { 'agems': data_dir+'feb13/meas_MID56_gre_field_mapping_FID1973.dat',
      'epi': data_dir+'feb13/meas_MID58_ep2d_pace_dynt_moco_FID1975.dat'
      },
    'mar5_1':
    { 'agems': None,
      'epi': data_dir+'mar5/meas_MID17_ep2d_pace_dynt_moco_FID2745.dat'
      },
    'apr16_1':
    { 'agems': data_dir+'MoreData_4_16_08/Raw/meas_MID92_gre_field_mapping_FID3407.dat',
      'epi': data_dir+'MoreData_4_16_08/Raw/meas_MID94_DanEPI_64x64_FID3409.dat'
      },
    'apr16_2':
    { 'agems': data_dir+'Data_4_16_08/Raw/meas_MID79_gre_field_mapping_FID3394.dat',
      'epi': data_dir+'Data_4_16_08/Raw/meas_MID81_DanEPI_64x64_FID3396.dat'
      },
    'apr24_1':
    { 'agems': data_dir+'Data_4_24_08/longramp/Raw/meas_MID92_gre_field_mapping_FID3726.dat',
      'epi': data_dir+'Data_4_24_08/longramp/Raw/meas_MID95_DanEPI_64x64_FID3729.dat'
      },
    'apr28_1':
    { 'agems': data_dir+'Data_4_28_08/Raw/meas_MID109_gre_field_mapping_FID4118.dat',
      'epi': data_dir+'Data_4_28_08/Raw/meas_MID112_DanEPI_64x64_FID4121.dat'
      },
    'may22_1':
    { 'agems': data_dir+'data_5_22_08/RAW/Fmap/meas_MID33_gre_field_mapping_FID5624.dat',
      'epi': data_dir+'data_5_22_08/RAW/EPI/meas_MID35_DanEPI_64x64_FID5626.dat'
      }

}
    
