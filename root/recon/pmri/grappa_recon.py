import numpy as np
from recon.operations import Operation, Parameter, \
     ChannelIndependentOperation, ChannelAwareOperation
from recon.pmri.grappa_sim import lstsq
from recon import util
from scipy.weave import inline
from scipy.weave.converters import blitz
from scipy.interpolate import interp1d

# For a 64x64 grid accelerated EPI scan, with R=2...
# ACS block is acquired densely for lines {19,20,21,22,...,42} (n=24)
# for volume data, N2 is undersampled at {1,3,5,7,...,61} (n=31)

def grappa_sampling(n_pe, a, n_acs):
    n_samp_lines = (n_pe-1)/a
    samp_lines = np.arange(1, a * n_samp_lines, a)
    k0 = 1 + a * (n_samp_lines/2)
    acs_lines = np.arange(k0-n_acs/2, k0+n_acs/2)
    return samp_lines, acs_lines

def window_params(n_acs, n_ch, n1, n_blks, accel, n1_window, fixed=True):
    if not n1_window or n1_window < 0 or n1_window=='max':
        return n1, 1
    elif n1_window=='min':
        # this is the number of solutions for a given harmonic
        # obtained by combining rows in a single column (worse case)
        n_soln = n_acs - (n_blks-1)*accel
        # want to find the min number of columns L such that
        # L*n_soln >= n_ch*n_blks ( the number of unknowns )
        win_sz = n_ch*n_blks / n_soln
        while win_sz * n_soln < n_ch*n_blks:
            win_sz += 1
    else:
        assert type(n1_window)==type(1), "n1_window argument not understood: "+str(n1_window)
        win_sz = min(n1, n1_window)
    if fixed:
        # for fixed windows, also make sure win_sz divides n1
        while (float(n1)/win_sz - n1/win_sz):
            win_sz += 1
        n_win = n1/win_sz
    else:
        n_win = n1 - win_sz + 1
    
    return win_sz, n_win

def grappa_coefs(acs_blk, accel, nblks, sliding=0, n1_window=None,
                 fixed_window=True, loud=False):
    """Fit system lines to data lines, finding fit coefficient tensor
    N(m,[n1],s,b,j,l)

    acs_blk should have dimensions (n_ch, n_acs, n1)
    
    """
    
    n_ch, n_acs, n1 = acs_blk.shape
##     if nblks < 2:
##         sliding = 0
    if sliding:
        max_blocks = n_acs/(2*accel)
        # number of sliding blocks
        nsb = 2*nblks-1
        # want 1st slide to equal -max(middle_blocks)
        # should be -np.arange(-((nblks+1)/2)+1, -((nblks+1)/2)+1+nblks)
        #slides = -np.arange(-(nblks/2)+1, -(nblks/2)+1+nblks)
        slides = -np.arange(-((nblks+1)/2)+1, -((nblks+1)/2)+1+nblks)
    else:
        max_blocks = n_acs/accel
        nsb = nblks
        slides = [0] #if nblks>1 else [-1]
        # if slides def changed, slides always == [0]

    assert nblks <= max_blocks, 'too many blocks for N_acs'
    #n_fits = n_acs/accel - (nsb - 1)
    #n_fits = n_acs/accel - (nblks - 1)

    

##     # HARDWIRE THIS FOR NOW
##     win_sz = n1
##     n_win = 1
    win_sz, n_win = window_params(n_acs, n_ch, n1, nblks, accel, n1_window,
                                  fixed=fixed_window)
    print win_sz, n_win
    
    # the fit coefficient tensor
    N = np.empty((accel-1,n_win,len(slides),n_ch,nblks,n_ch), acs_blk.dtype)
    # the fit L2-squared error at each channel
    err = np.zeros((accel-1, n_win, len(slides), n_ch), 'd')
    
    sys_slice = [slice(None)]*3
    dat_slice = [slice(None)]*3
    for m in range(accel-1):
        if nblks > 1 and not sliding:
            n_fits = n_acs - (nblks-1)*accel #- (m+1) + 1
        else:
            n_fits = n_acs - (nblks-1)*accel - (m+1)
        print "harmonic:", (m+1),"n_fits:",n_fits

        sys_rows = np.empty((n_fits, nblks), 'i')
        for r in range(n_fits):
            sys_rows[r] = np.arange(r, r + accel*nblks, accel)
        if loud: print sys_rows

        for w in xrange(n_win):
            

            sys = np.empty((n_fits*win_sz, n_ch*nblks), acs_blk.dtype)
            # so if sys_rows is PxQ, the system matrix will be NxM where...
            # N = win_sz*P, M = N_ch*Q
            if fixed_window:
                sys_slice[-1] = slice(w*win_sz, (w+1)*win_sz)
                dat_slice[-1] = slice(w*win_sz, (w+1)*win_sz)
            else:
                sys_slice[-1] = slice(w, w+win_sz)
                dat_slice[-1] = slice(w, w+win_sz)
            for n,r in enumerate(sys_rows):
                sys_slice[-2] = r
                sprt = acs_blk[sys_slice].reshape(n_ch*nblks, win_sz)
                sys[n*win_sz:(n+1)*win_sz] = sprt.transpose().copy()
            # sys is finally (n_fits*win_sz, n_ch*nblk)
            # conjugating sys splits the hermitian transpose into two steps
            #[u,svals,vt] = np.linalg.svd(sys.conjugate(), full_matrices=0)
            #Apinv = np.dot(vt.transpose(), ((1/svals)[:,None]*u.transpose()))
            Apinv = np.linalg.pinv(sys) #, rcond=0.01)
            #print "matrix condition #:",svals[0]/svals[-1]
            assert Apinv.shape == (nblks*n_ch, n_fits*win_sz), 'unexpected matrix shape'
            for n,s in enumerate(slides):
                # m+1 is the harmonic to which to fit (m itself will index)
                # s is which sliding block we're on
                blist = range(-((nblks+1)/2)+1+s, -((nblks+1)/2)+1+s+nblks)
                # find which column b=0, and set the data rows to be +(m+1)
                col = blist.index(0)
                dat_rows = sys_rows[:,col]+m+1
                if loud: print dat_rows
                dat_slice[-2] = dat_rows
                dat = acs_blk[dat_slice].reshape(n_ch, n_fits*win_sz).transpose().copy()
                soln = np.dot(Apinv, dat)
                N[m,w,n] = soln.reshape(n_ch, nblks, n_ch)
                e = np.dot(sys, soln) - dat
                err[m,w,n] = np.array([np.linalg.norm(v, 2)**2 for v in e.T])

    return N, err




def grappa_synthesize(Ssub, N, n2_sampling, weights=None,
                      fixed_window=True, loud=False):
    accel = n2_sampling[1] - n2_sampling[0]
    # first sampled line
    r0 = n2_sampling[0]
    # last sampled line
    rf = n2_sampling[-1]
    # make sure that we don't try to synthesize outside of the array
    # if the last sampled line is the last array line
    if rf == Ssub.shape[-2]-1:
        rf -= accel

    # why this?? n_pe = rf-r0+accel
    n_pe = Ssub.shape[-2]
    n1 = Ssub.shape[-1]
    # I guess Ssub.shape will always be (nc, nv, n2, n1) even if nv=1..
    # so this is not necessary
    nv = Ssub.shape[1] if len(Ssub.shape) > 3 else 1
    n_harm, n_win, n_slides, n_ch, nblks = N.shape[:-1]
    if fixed_window:
        win_sz = n1/n_win
        win_idx = np.repeat(np.arange(n_win), win_sz)
        win_idx_generates = np.arange(n1).reshape((n_win, win_sz))
    else:
        win_sz = n1 - n_win + 1
        win_idx = [0,]*(win_sz/2) + range(0,n_win-1) + [n_win-1]*(win_sz/2+1)
        if n_win == 1:
            win_idx_generates = [range(0,n1)]
        else:
            win_idx_generates = [range(0,win_sz/2+1)] + \
                                [[x] for x in range(win_sz/2+1,
                                                    win_sz/2+n_win-1)] + \
                                [range(win_sz/2+n_win-1, n1)]
            
                                                                      

    if weights is None:
        weights = np.ones((n_harm, n_slides, n_win, n_ch), 'd')/n_slides

    # if sliding, this is like a convolution of blocks and rows
    n_fits = n_pe/accel + nblks - 1 #- (nblks - 1)
    if n_slides > 1:
        #slides = -np.arange(-(nblks/2)+1, -(nblks/2)+1+nblks)
        slides = -np.arange(-((nblks+1)/2)+1, -((nblks+1)/2)+1+nblks)
    else:
        slides = [0] if nblks>1 else [-1]

    sys_slice = [slice(None)]*len(Ssub.shape)
    synth_slice = [slice(None)]*len(Ssub.shape)
    Ssyn = Ssub.copy()
    for m in range(n_harm):
        sys_rows = np.empty((n_fits, nblks), 'i')
        for i,r in enumerate(range(-nblks+1, -nblks+1+n_fits)):
            sys_rows[i] = np.arange(accel*r+r0, accel*(r+nblks)+r0, accel)
            np.putmask(sys_rows[i], sys_rows[i]<0, r0+m+1)
            np.putmask(sys_rows[i], sys_rows[i]>=n_pe, r0+m+1)
        #print sys_rows
        printed = [False]*len(slides)
        #for recon_col in xrange(n1):
        for w, recon_window in enumerate(win_idx_generates):
            col_slice = slice(recon_window[0], recon_window[-1]+1)
            # redefine this here, because for floating window, it can vary
            win_sz = recon_window[-1] - recon_window[0] + 1
            #w = win_idx[recon_col]
            sys_slice[-1] = col_slice
            synth_slice[-1] = col_slice
            # the rows of sys represent # of synth lines to create
            # (with win_sz pts each), across nv volumes..
            # the total # rows is nv*n_fits*win_sz, but the rows are segmented
            # by fit.. so expanded shape would be (n_fits, nv, win_sz)
            sys = np.empty((nv*n_fits*win_sz, n_ch*nblks), Ssub.dtype)
            for n,r in enumerate(sys_rows):
                sys_slice[-2] = r
                # this slice will be (nc, nv, nblks, win_sz),
                # transpose it to (nv, win_sz, nc, nblks)
                sprt = Ssub[sys_slice].transpose(1, 3, 0, 2).copy()
                sprt = sprt.reshape(nv*win_sz, n_ch*nblks)
                #sprt = sprt.transpose(1, 0)
                #sprt = sprt.reshape(n_ch*nblks, nv)
                # this SHOULD be the only expensive step!
                sys[n*nv*win_sz:(n+1)*nv*win_sz] = sprt
            
            for n,s in enumerate(slides):
                blist = range(-((nblks+1)/2)+1+s, -((nblks+1)/2)+1+s+nblks)
                col = blist.index(0)
                synth_rows = (sys_rows[:,col]+m+1).tolist()
                sr0 = synth_rows.index(r0+m+1)
                srf = synth_rows.index(rf+m+1)
                synth_rows = np.array(synth_rows[sr0:srf+1])
                #print "slide", s, "synthesizing rows:", synth_rows
                if loud and not printed[n]:
                    print 'harmonic:',m,'slide:',s
                    for row, syn in zip(sys_rows[sr0:srf+1], synth_rows):
                        print row, '--->', syn
                    printed[n] = True
                synth_slice[-2] = synth_rows
                s = np.dot(sys[sr0*nv*win_sz:(srf+1)*nv*win_sz],
                           N[m,w,n].reshape(n_ch*nblks, n_ch)).transpose()
                # s is shaped (n_ch, n_fits*nv*win_sz)
                s.shape = (n_ch, len(synth_rows), nv, win_sz)
                s = s.transpose(0, 2, 1, 3)
                Ssyn[synth_slice] += s * weights[m,w,n][:,None,None,None]
    
    return Ssyn

## The following calibration/synthesis variations are organized largely
## along the categories outlined in:
## Comparison of Reconstruction Accuracy and Efficiency 
## Among Autocalibrating Data-Driven Parallel Imaging 
## Methods 
## Anja C.S. Brau et al

def basic_grappa_1D(image, Nblk, sl=-1, regularize=False):
    R = int(image.accel)
    Nc = image.n_chan
    Nacs = int(image.n_acs)
    Nx = image.N1
    Nv = image.n_vol
    Nf = Nacs - (Nblk-1)*R
    Npe = image.n_pe
    acs = image.cacs_data
    acs_shape = acs.shape
    acs.shape = tuple([d for d in acs.shape if d > 1])
    cdata = image.cdata
    #W = np.empty((Nblk*Nc, (R-1)*Nc), 'F')

    # if ky neighborhood is not symmetric, bias it towards earlier acquisition
    blk_offsets = np.arange(-((Nblk-1)/2), -((Nblk-1)/2)+Nblk)*R - 1

    # fitting stage
    Ssrc = np.empty((Nf*Nx, Nblk*Nc), 'F')
    Stgt = np.empty((Nf*Nx, (R-1)*Nc), 'F')
    tgt_row0 = R * ((Nblk-1)/2) + 1
    tgt_rows = np.arange(tgt_row0, tgt_row0 + Nf)

    # synthesis stage
    # true # of PE lines is last-sampled - first-sampled + 1
    pe_sampling = image.pe_sampling[:]
    pe_sampling.sort()
    acq_row0 = pe_sampling[0] # == 1
    acq_rowf = pe_sampling[-1]+1 # == last-sampled + 1
    syn_rows = np.arange(tgt_row0+1, acq_rowf, R)
    Nu = len(syn_rows)
    Sacq = np.empty((Nv*Nu*Nx, Nblk*Nc), 'F')

    sl_range = [sl] if sl>=0 else xrange(image.n_slice)

    for s in sl_range:
        Dy_idx = tgt_rows[:,None] + blk_offsets
        for r in range(R-1):
            # acs is shaped (nc, nsl, nacs, nx)
            tgt_pts = acs[:,s,tgt_rows+r,:].reshape(Nc, Nf*Nx)
            # this set of columns is shaped (Nf*Nx, Nc)
            Stgt[:,r*Nc:(r+1)*Nc] = tgt_pts.transpose()
        for n in xrange(Nf):
            blk_idx = Dy_idx[n]
            src_pts = acs[:,s,blk_idx,:].reshape(Nc*Nblk, Nx)
            # this set of rows is shaped (Nx, Nc*Nblk)
            Ssrc[n*Nx:(n+1)*Nx,:] = src_pts.transpose()
        if regularize:
            W = util.regularized_solve_lcurve(Ssrc, Stgt)
        else:
            Ssrc_pinv = np.linalg.pinv(Ssrc)
            W = np.dot(Ssrc_pinv, Stgt)
##         del Ssrc
##         del Stgt
##         del Ssrc_pinv
##         del src_pts
##         del tgt_pts

        #syn_rows = np.arange(tgt_row0+1, tgt_row0+1 + Nu, R)
        Dy_idx = syn_rows[:,None] + blk_offsets
        for n in xrange(Nu):
            blk_idx = Dy_idx[n]
            acq_pts = cdata[:,:,s,blk_idx,:].transpose(0,2,1,3).copy()
            acq_pts.shape = (Nc*Nblk, Nv*Nx)
            # this set of rows is shaped (Nv*Nx, Nc*Nblk)
            Sacq[n*Nv*Nx:(n+1)*Nv*Nx] = acq_pts.transpose()
        del acq_pts


        # Ssyn is shaped (Nu*Nv*Nx, (R-1)*Nc)
        Ssyn = np.dot(Sacq, W)
        for r in xrange(R-1):
            coil_data = Ssyn[:,r*Nc:(r+1)*Nc].reshape(Nu, Nv, Nx, Nc)
            srows = syn_rows + r
            print 'copying rows', srows
            cdata[:,:,s,srows,:] = coil_data.transpose(3, 1, 0, 2)
##         del Ssyn
##         del coil_data
    acs.shape = acs_shape
    del Ssrc
    del Stgt
    if not regularize:
        del Ssrc_pinv
    del src_pts
    del tgt_pts
    del Ssyn
    del coil_data

def basic_grappa_1D_prior(image, Nblk, sl=-1, lm=1e-2):
    R = int(image.accel)
    Nc = image.n_chan
    Nacs = int(image.n_acs)
    Nx = image.N1
    Nv = image.n_vol
    Nf = Nacs - (Nblk-1)*R
    Npe = image.n_pe
    acs = image.cacs_data
    acs_shape = acs.shape
    acs.shape = tuple([d for d in acs.shape if d > 1])
    cdata = image.cdata
    acs_lines = grappa_sampling(Npe, R, Nacs)[1]
    #W = np.empty((Nblk*Nc, (R-1)*Nc), 'F')

    model_err = np.zeros(image.n_slice)
    prior_err = np.zeros(image.n_slice)

    # if ky neighborhood is not symmetric, bias it towards earlier acquisition
    blk_offsets = np.arange(-((Nblk-1)/2), -((Nblk-1)/2)+Nblk)*R - 1

    # fitting stage
    Ssrc = np.empty((Nf*Nx, Nblk*Nc), 'F')
    Stgt = np.empty((Nf*Nx, (R-1)*Nc), 'F')
    tgt_row0 = R * ((Nblk-1)/2) + 1
    tgt_rows = np.arange(tgt_row0, tgt_row0 + Nf)

    # synthesis stage
    # true # of PE lines is last-sampled - first-sampled + 1
    pe_sampling = image.pe_sampling[:]
    pe_sampling.sort()
    acq_row0 = pe_sampling[0] # == 1
    acq_rowf = pe_sampling[-1]+1  # == last-sampled + 1
    syn_rows = np.arange(tgt_row0+1, acq_rowf, R)
    Nu = len(syn_rows)
    Sacq = np.empty((Nv*Nu*Nx, Nblk*Nc), 'F')

    id_cols = []
    pr_rows = []
    # these indices are the columns of L that will select the prior matching
    # section of Ssyn
    id_cols = [i for i in xrange(len(syn_rows)) if syn_rows[i] in acs_lines]
    # these indices are the rows of the ACS that will be used as prior info
    pr_rows = np.array([np.argwhere(acs_lines==syn_rows[i])[0][0]
                        for i in id_cols])
    ncols = Nx*len(id_cols)
    npriors = len(pr_rows)
    assert ncols==(Nx*Nacs/2), "i'm dumb"
    L = np.zeros((ncols, Nu*Nx), Sacq.dtype)
    L[:,slice(Nx*id_cols[0], Nx*(id_cols[-1]+1))] = np.diag(lm*np.ones(ncols))
    Pr = np.empty((npriors*Nx, Nc*(R-1)), 'F')

    sl_range = [sl] if sl>=0 else xrange(image.n_slice)
    for s in sl_range:
        Dy_idx = tgt_rows[:,None] + blk_offsets
        for r in range(R-1):
            # acs is shaped (nc, nsl, nacs, nx)
            tgt_pts = acs[:,s,tgt_rows+r,:].reshape(Nc, Nf*Nx)
            # this set of columns is shaped (Nf*Nx, Nc)
            Stgt[:,r*Nc:(r+1)*Nc] = tgt_pts.transpose()
        for n in xrange(Nf):
            blk_idx = Dy_idx[n]
            src_pts = acs[:,s,blk_idx,:].reshape(Nc*Nblk, Nx)
            # this set of rows is shaped (Nx, Nc*Nblk)
            Ssrc[n*Nx:(n+1)*Nx,:] = src_pts.transpose()
        Ssrc_pinv = np.linalg.pinv(Ssrc)
        print 'solving coefs'
        W = np.dot(Ssrc_pinv, Stgt)
##         del Ssrc
##         del Stgt
##         del Ssrc_pinv
##         del src_pts
##         del tgt_pts

        #syn_rows = np.arange(tgt_row0+1, tgt_row0+1 + Nu, R)
        Dy_idx = syn_rows[:,None] + blk_offsets
        for n in xrange(Nu):
            blk_idx = Dy_idx[n]
            acq_pts = cdata[:,:,s,blk_idx,:].transpose(0,2,1,3).copy()
            acq_pts.shape = (Nc*Nblk, Nv*Nx)
            # this set of rows is shaped (Nv*Nx, Nc*Nblk)
            Sacq[n*Nv*Nx:(n+1)*Nv*Nx] = acq_pts.transpose()
        del acq_pts

        # now with pinv(Sacq) and lm*L, form A,
        # with W and lm*Stgt form B,
        # solve Ax = B for x, x --> Ssyn
        # --------- construct new A matrix -----------
        Ap = np.linalg.pinv(Sacq)
        ap_rows = (slice(0, Ap.shape[0]), slice(None))
        l_rows = (slice(Ap.shape[0], Ap.shape[0]+L.shape[0]), slice(None))
        A = np.concatenate( (Ap, L), axis=0)
        del Ap
        Ap = A[ap_rows]
        L = A[l_rows]
        # --------- construct new B matrix -----------
        for r in range(R-1):
            prior_pts = acs[:,s,pr_rows+r,:].reshape(Nc, Nacs*Nx/2)
            Pr[:,r*Nc:(r+1)*Nc] = prior_pts.transpose()
        w_rows = (slice(0, W.shape[0]), slice(None))
        p_rows = (slice(W.shape[0], W.shape[0]+Pr.shape[0]), slice(None))
        B = np.concatenate( (W, lm*Pr), axis=0)
        del W
        W = B[w_rows]
##         Pr = B[p_rows]

        # Ssyn is shaped (Nu*Nv*Nx, (R-1)*Nc)
##         Ssyn = np.dot(Sacq, W)
        print 'solving regularized problem'
        Ssyn = np.linalg.lstsq(A, B, rcond=1e-5)[0]
        for r in xrange(R-1):
            coil_data = Ssyn[:,r*Nc:(r+1)*Nc].reshape(Nu, Nv, Nx, Nc)
            srows = syn_rows + r
            print 'copying rows', srows
            cdata[:,:,s,srows,:] = coil_data.transpose(3, 1, 0, 2)
##         del Ssyn
##         del coil_data

        #perr = np.dot(L, Ssyn) - Pr
        perr = Ssyn[id_cols[0]*Nx:(id_cols[-1]+1)*Nx,:] - Pr
        prior_err[s] = np.dot(perr.flat[:], perr.flat[:].conjugate()).real**0.5
        merr = np.dot(Ap, Ssyn) - W
        model_err[s] = np.dot(merr.flat[:], merr.flat[:].conjugate()).real**0.5
        
    del Ssrc
    del Stgt
    del Ssrc_pinv
    del src_pts
    del tgt_pts
    del Ssyn
    del coil_data
    del A
    del Ap
    del L
    del B
    del W
    del Pr
    acs.shape = acs_shape
    return model_err, prior_err

def basis_funcs(x, Ncx):
    fov = (x[1]-x[0])*len(x)
    return np.cos(2*np.pi*x*np.arange(Ncx)[:,None]/(2*fov))
##     return np.cos(2*np.pi*x*np.arange(Ncx)[:,None]/fov)
##     return np.exp(2j*np.pi*x*np.arange(-(Ncx/2), -(Ncx/2)+Ncx)[:,None]/fov)
def xspace_method(meth):
    def xformed_image_method(image, *args, **kwargs):
        print 'inverse transforming'
        util.ifft1(image.cdata, inplace=True, shift=True)
        util.ifft1(image.cacs_data, inplace=True, shift=True)
        fx = meth(image, *args, **kwargs)
        print 'forward transforming'
        util.fft1(image.cdata, inplace=True, shift=True)
        util.fft1(image.cacs_data, inplace=True, shift=True)
        if fx is not None:
            return fx
    return xformed_image_method



@xspace_method
def grappa_segmented_x(image, Nblk, Nseg, sl=-1):
    R = int(image.accel)
    Nc = image.n_chan
    Nacs = int(image.n_acs)
    Nx = image.N1/Nseg
    xsegs = [slice(x*Nx, (x+1)*Nx) for x in xrange(Nseg)]
    Nv = image.n_vol
    Nf = Nacs - (Nblk-1)*R
    Npe = image.n_pe
    acs = image.cacs_data
    acs.shape = tuple([d for d in acs.shape if d > 1])
    cdata = image.cdata

    # if ky neighborhood is not symmetric, bias it towards earlier acquisition
    blk_offsets = np.arange(-((Nblk-1)/2), -((Nblk-1)/2)+Nblk)*R - 1

    # fitting stage
    Ssrc = np.empty((Nf*Nx, Nblk*Nc), 'F')
    Stgt = np.empty((Nf*Nx, (R-1)*Nc), 'F')
    tgt_row0 = R * ((Nblk-1)/2) + 1
    tgt_rows = np.arange(tgt_row0, tgt_row0 + Nf)

    # synthesis stage
    # true # of PE lines is last-sampled - first-sampled + 1
    pe_sampling = image.pe_sampling[:]
    pe_sampling.sort()
    acq_row0 = pe_sampling[0] # == 1
    acq_rowf = pe_sampling[-1]+1  # == last-sampled + 1
    syn_rows = np.arange(tgt_row0+1, acq_rowf, R)
    Nu = len(syn_rows)
    Sacq = np.empty((Nv*Nu*Nx, Nblk*Nc), 'F')
    sl_range = [sl] if sl>=0 else xrange(image.n_slice)

    for s in sl_range:
        for g in range(Nseg):
            Dy_idx = tgt_rows[:,None] + blk_offsets
##             print 'Fitting...'
##             for row, tgt in zip(Dy_idx, tgt_rows):
##                 print row, '--->', [tgt+r for r in range(R-1)]
            for r in range(R-1):
                # acs is shaped (nc, nsl, nacs, nx)
                tgt_pts = acs[:,s,tgt_rows+r,xsegs[g]].reshape(Nc, Nf*Nx)
                # this set of columns is shaped (Nf*Nx, Nc)
                Stgt[:,r*Nc:(r+1)*Nc] = tgt_pts.transpose()
            for n in xrange(Nf):
                blk_idx = Dy_idx[n]
                src_pts = acs[:,s,blk_idx,xsegs[g]].reshape(Nc*Nblk, Nx)
                # this set of rows is shaped (Nx, Nc*Nblk)
                Ssrc[n*Nx:(n+1)*Nx,:] = src_pts.transpose()
            u,sv,vh = np.linalg.svd(Ssrc)
            Ssrc_pinv = np.linalg.pinv(Ssrc)
            W = np.dot(Ssrc_pinv, Stgt)
    
            #syn_rows = np.arange(tgt_row0+1, tgt_row0+1 + Nu, R)
            Dy_idx = syn_rows[:,None] + blk_offsets
            for n in xrange(Nu):
                blk_idx = Dy_idx[n]
                acq_pts = cdata[:,:,s,blk_idx,xsegs[g]].transpose(0,2,1,3).copy()
                acq_pts.shape = (Nc*Nblk, Nv*Nx)
                # this set of rows is shaped (Nv*Nx, Nc*Nblk)
                Sacq[n*Nv*Nx:(n+1)*Nv*Nx] = acq_pts.transpose()


            # Ssyn is shaped (Nu*Nv*Nx, (R-1)*Nc)
            Ssyn = np.dot(Sacq, W)
            for r in xrange(R-1):
                coil_data = Ssyn[:,r*Nc:(r+1)*Nc].reshape(Nu, Nv, Nx, Nc)
                srows = syn_rows + r
##                 print 'copying rows', srows
                cdata[:,:,s,srows,xsegs[g]] = coil_data.transpose(3, 1, 0, 2)
    del Ssrc
    del Stgt
    del Ssrc_pinv
    del W
    del src_pts
    del tgt_pts
    del Ssyn
    del coil_data
    del acq_pts

@xspace_method
def grappa_interpolated_x(image, Nblk, win_len, sl=-1):
    # find solutions for overlapping segments, and interpolate
    # to finer x-grid using those points
    R = int(image.accel)
    Nc = image.n_chan
    Nacs = int(image.n_acs)
    N1 = image.N1
    Nv = image.n_vol
    Nf = Nacs - (Nblk-1)*R
    Npe = image.n_pe
    acs = image.cacs_data
    acs.shape = tuple([d for d in acs.shape if d > 1])
    cdata = image.cdata

    # find the number of x segments and the degree of overlap corresponding
    # to a given window length 
    p = N1-win_len
    x_steps = win_len-1
    while float(p)/x_steps != p/x_steps and x_steps>0:
        x_steps -= 1
    Nseg = p/x_steps + 1
    xsegs = [slice(x*x_steps, x*x_steps+win_len) for x in xrange(Nseg)]
    Nx = win_len
    x_in = np.array([0] + [s.start + Nx/2 for s in xsegs] + [N1])
    x_out = np.linspace(0, float(N1-1), N1)
    # if ky neighborhood is not symmetric, bias it towards earlier acquisition
    blk_offsets = np.arange(-((Nblk-1)/2), -((Nblk-1)/2)+Nblk)*R - 1

    # fitting stage
    Ssrc = np.empty((Nf*Nx, Nblk*Nc), 'F')
    Stgt = np.empty((Nf*Nx, (R-1)*Nc), 'F')
    tgt_row0 = R * ((Nblk-1)/2) + 1
    tgt_rows = np.arange(tgt_row0, tgt_row0 + Nf)

    # synthesis stage
    # true # of PE lines is last-sampled - first-sampled + 1
    pe_sampling = image.pe_sampling[:]
    pe_sampling.sort()
    acq_row0 = pe_sampling[0] # == 1
    acq_rowf = pe_sampling[-1]+1  # == last-sampled + 1
    syn_rows = np.arange(tgt_row0+1, acq_rowf, R)
    Nu = len(syn_rows)
    Sacq = np.empty((Nv*Nu, Nblk*Nc), 'F')

    # the sparsely sampled coef array.. with buffer points at x=0 and x=N1
    W = np.zeros((Nseg+2, Nblk*Nc, (R-1)*Nc), 'F')
    sl_range = [sl] if sl>=0 else xrange(image.n_slice)

    for s in sl_range:
        for g in xrange(Nseg):
            Dy_idx = tgt_rows[:,None] + blk_offsets
##             print 'Fitting...'
##             for row, tgt in zip(Dy_idx, tgt_rows):
##                 print row, '--->', [tgt+r for r in range(R-1)]
            for r in range(R-1):
                # acs is shaped (nc, nsl, nacs, nx)
                tgt_pts = acs[:,s,tgt_rows+r,xsegs[g]].reshape(Nc, Nf*Nx)
                # this set of columns is shaped (Nf*Nx, Nc)
                Stgt[:,r*Nc:(r+1)*Nc] = tgt_pts.transpose()
            for n in xrange(Nf):
                blk_idx = Dy_idx[n]
                src_pts = acs[:,s,blk_idx,xsegs[g]].reshape(Nc*Nblk, Nx)
                # this set of rows is shaped (Nx, Nc*Nblk)
                Ssrc[n*Nx:(n+1)*Nx,:] = src_pts.transpose()
            Ssrc_pinv = np.linalg.pinv(Ssrc)
            W[g+1] = np.dot(Ssrc_pinv, Stgt)
        interpolator = interp1d(x_in, W, kind='cubic',
                                bounds_error=False, axis=0)
        Wx = interpolator(x_out)
        Dy_idx = syn_rows[:,None] + blk_offsets
        for x in xrange(N1):
            for n in xrange(Nu):
                blk_idx = Dy_idx[n]
                acq_pts = cdata[:,:,s,blk_idx,x].transpose(0,2,1).copy()
                acq_pts.shape = (Nc*Nblk, Nv)
                # this set of rows is shaped (Nv, Nc*Nblk)
                Sacq[n*Nv:(n+1)*Nv] = acq_pts.transpose()

            Ssyn = np.dot(Sacq, Wx[x])
            for r in xrange(R-1):
                coil_data = Ssyn[:,r*Nc:(r+1)*Nc].reshape(Nu, Nv, Nc)
                srows = syn_rows + r
##                 print 'copying rows', srows
                cdata[:,:,s,srows,x] = coil_data.transpose(2, 1, 0)
                

    del Ssrc
    del Stgt
    del Sacq
    del coil_data
    del Wx
    

@xspace_method
def grappa_smoothly_varying_x(image, Nblk, Ncx, sl=-1):
    R = int(image.accel)
    Nc = image.n_chan
    Nacs = int(image.n_acs)
    Nx = image.N1
    Nv = image.n_vol
    Nf = Nacs - (Nblk-1)*R
    Npe = image.n_pe
    acs = image.cacs_data
    acs.shape = tuple([d for d in acs.shape if d > 1])
    cdata = image.cdata

    xax = np.arange(Nx) * image.isize
    bfun = basis_funcs(xax, Ncx).transpose() # <-- returns (Ncx, Nx).T

    # if ky neighborhood is not symmetric, bias it towards earlier acquisition
    blk_offsets = np.arange(-((Nblk-1)/2), -((Nblk-1)/2)+Nblk)*R - 1


    # fitting stage...
    # Ssrc is block diag, with (Nf, Nblk*Nc) matrix at each x
##     Ssrc = np.empty((Nx*Nf, Nx*Nblk*Nc), 'F')
##     Ssubs = [Ssrc[x*Nf:(x+1)*Nf, x*Nblk*Nc:(x+1)*Nblk*Nc] for x in xrange(Nx)]
    src_pts = np.empty((Nx, Nf, Nblk*Nc), 'F')
##     Q = np.zeros((Nx, Nblk*Nc, Ncx*Nblk*Nc), 'F')
##     Qsubs = [Q[:, i, i*Ncx:(i+1)*Ncx] for i in xrange(Nblk*Nc)]
    Stgt = np.empty((Nx*Nf, (R-1)*Nc), 'F')
    tgt_row0 = R * ((Nblk-1)/2) + 1
    tgt_rows = np.arange(tgt_row0, tgt_row0 + Nf)
    SQ = np.empty((Nx*Nf, Ncx*Nc*Nblk), 'F')

    # synthesis stage
    # true # of PE lines is last-sampled - first-sampled + 1
    pe_sampling = image.pe_sampling[:]
    pe_sampling.sort()
    acq_row0 = pe_sampling[0] # == 1
    acq_rowf = pe_sampling[-1]+1  # == last-sampled + 1
    syn_rows = np.arange(tgt_row0+1, acq_rowf, R)
    Nu = len(syn_rows)
    acq_pts = np.empty((Nx, Nv*Nu, Nc*Nblk), 'F')
    sl_range = [sl] if sl>=0 else xrange(image.n_slice)

    for s in sl_range:
        print 'slice:', s
        # for each of Nx sections, Stgt has a (Nf, Nc*(R-1)) matrix
        for r in range(R-1):
            tgt_pts = acs[:,s,tgt_rows+r,:].transpose(2, 1, 0)
            Stgt[:,r*Nc:(r+1)*Nc] = tgt_pts.reshape(Nx*Nf, Nc)

        Dy_idx = tgt_rows[:,None] + blk_offsets
        print 'Fitting...'
        for row, tgt in zip(Dy_idx, tgt_rows):
            print row, '--->', [tgt+r for r in range(R-1)]
        # For each x, src_pts has a (Nf, Nc*Nblk) matrix
        for n in xrange(Nf):
            blk_idx = Dy_idx[n]
            src_pts[:,n,:] = acs[:,s,blk_idx,:].reshape(Nc*Nblk, Nx).transpose()

##         for qsub in Qsubs:
##             qsub[:] = bfun

        # now solve (Ssrc*Q) * H = Stgt

##         SQ = np.dot(Ssrc, Q)
        # fast SQ product
        for x in xrange(Nx):
            SQ[x*Nf:(x+1)*Nf] = (src_pts[x][:,:,None] * \
                                 bfun[x][None,None,:]).reshape(Nf, Nc*Nblk*Ncx)

        SQpinv = np.linalg.pinv(SQ)
        H = np.dot(SQpinv, Stgt)
        del SQpinv
        Dy_idx = syn_rows[:,None] + blk_offsets
        for n in xrange(Nu):
            blk_idx = Dy_idx[n]
            # original slice is (Nc, Nv, Nblk, Nx)
            pts = cdata[:,:,s,blk_idx,:].transpose(0, 2, 1, 3).copy()
            pts.shape = (Nc*Nblk, Nv, Nx)
            acq_pts[:,n*Nv:(n+1)*Nv,:] = pts.transpose(2, 1, 0)


    ##     Wp = np.dot(Q,H)
    ##     # fast QH product
        H.shape = (Nc*Nblk, Ncx, (R-1)*Nc)
    ##     for x in xrange(Nx):
    ##         Wp[x*Nblk*Nc:(x+1)*Nblk*Nc] = (H*bfun[x][None,:,None]).sum(axis=1)
    ##     #del H

        for x in xrange(Nx):
            Sacq_sub = acq_pts[x]
            #Wp_sub = Wp[x*Nblk*Nc:(x+1)*Nblk*Nc]
            Wp_sub = (H*bfun[x][None,:,None]).sum(axis=1)
            Ssyn = np.dot(Sacq_sub, Wp_sub)
            for r in xrange(R-1):
                coil_data = Ssyn[:,r*Nc:(r+1)*Nc].reshape(Nu, Nv, Nc)
                srows = syn_rows + r
                cdata[:,:,s,srows,x] = coil_data.transpose(2, 1, 0)
        del Ssyn
    

def ksp_grappa_2D(image, Ny_blk, Nx_blk, sl=-1):
    R = int(image.accel)
    Nc = image.n_chan
    Nacs = int(image.n_acs)
    Nx = image.N1
    Nv = image.n_vol
    Nfy = Nacs - (Ny_blk-1)*R
    Nfx = Nx - (Nx_blk-1)
    x_start = Nx_blk/2 # lower inclusive limit for x target points
    x_end = x_start + Nfx - 1 # upper limit, inclusive, for x target points
    x_slice = slice(x_start, x_end)
    Npe = image.n_pe
    acs = image.cacs_data
    acs.shape = tuple([d for d in acs.shape if d > 1])
    cdata = image.cdata
    
    # set up 2D neighborhood offsets
    xblk_offsets = np.arange(-(Nx_blk/2), -(Nx_blk/2)+Nx_blk)
    yblk_offsets = np.arange(-((Ny_blk-1)/2), -((Ny_blk-1)/2)+Ny_blk)*R - 1

    # fitting stage
    Ssrc = np.empty((Nfy*Nfx, Nc*Ny_blk*Nx_blk), 'F')
    Stgt = np.empty((Nfy*Nfx, (R-1)*Nc), 'F')
    tgt_row0 = R * ((Ny_blk-1)/2) + 1
    tgt_rows = np.arange(tgt_row0, tgt_row0 + Nfy)
    
    # synthesis stage
    pe_sampling = image.pe_sampling[:]
    pe_sampling.sort()
    acq_row0 = pe_sampling[0] # == 1
    acq_rowf = pe_sampling[-1]+1  # == last-sampled + 1
    syn_rows = np.arange(tgt_row0+1, acq_rowf, R)
    Nu = len(syn_rows)
    # make Sacq with full readout points, and assume circularity in kx direction
    Sacq = np.empty((Nu*Nx*Nv, Nc*Ny_blk*Nx_blk), 'F')

    Ssrc_fill = \
"""
int xidx, yidx, n, m, i, c, j;
for(n=0; n<Nfy; n++) {
  for(m=0; m<Nfx; m++) {
    for(c=0; c<Nc; c++) {
      for(i=0; i<Ny_blk; i++) {
        yidx = Dy_idx(n,i);
        for(j=0; j<Nx_blk; j++) {
          xidx = (m + x_start + xblk_offsets(j));
          Ssrc(n,m,c,i,j) = acs(c,s,yidx,xidx);
        }
      }
    }
  }
}
"""

    Sacq_fill = \
"""
int xidx, yidx, n, m, i, c, j, v;
for(n=0; n<Nu; n++) {
  for(m=0; m<Nx; m++) {
    for(v=0; v<Nv; v++) {
      for(c=0; c<Nc; c++) {
        for(i=0; i<Ny_blk; i++) {
          yidx = Dy_idx(n,i);
          for(j=0; j<Nx_blk; j++) {
            xidx = (m + xblk_offsets(j) + Nx) % Nx;
            Sacq(n,m,v,c,i,j) = cdata(c,v,s,yidx,xidx);
          }
        }
      }
    }
  }
}
"""
    syn_rows_fill = \
"""
using namespace blitz;
int m, c, v, r, srow;
for(m=0; m<(R-1); m++) {
  std::cout<<"filling rows: [";
  for(r=0; r<Nu; r++) std::cout<<syn_rows(r)+m<<", ";
  std::cout<<"]"<<std::endl;
  for(r=0; r<Nu; r++) {
    srow = syn_rows(r)+m;
    for(v=0; v<Nv; v++) {
      for(c=0; c<Nc; c++) {
        cdata(c,v,s,srow,Range::all()) = Ssyn(r,Range::all(),v,m,c);
      }
    }
  }
}
"""
    tgt_rows_fill = \
"""
using namespace blitz;
int m, c, r, trow;
for(m=0; m<(R-1); m++) {
  std::cout<<"usings rows: [";
  for(r=0; r<Nfy; r++) std::cout<<tgt_rows(r)+m<<", ";
  std::cout<<"]"<<std::endl;
  for(r=0; r<Nfy; r++) {
    trow = tgt_rows(r)+m;
    for(c=0; c<Nc; c++) {
      Stgt(r,Range::all(),m,c) = acs(c,s,trow,Range(x_start,x_end));
    }
  }
}
"""
    sl_range = [sl] if sl>=0 else xrange(image.n_slice)

    for s in sl_range:
        print 'reconstructing slice', s
        Dy_idx = tgt_rows[:,None] + yblk_offsets
##         print 'Fitting...'
##         for row, tgt in zip(Dy_idx, tgt_rows):
##             print row, '--->', [tgt+r for r in range(R-1)]        
        print 'filling Stgt'
        Stgt.shape = (Nfy, Nfx, (R-1), Nc)
        inline(tgt_rows_fill, ['R','tgt_rows','Nfy','Nc','acs',
                               'Stgt','x_start','x_end','s'],
               type_converters=blitz)
        Stgt.shape = (Nfy*Nfx, (R-1)*Nc)

        print 'filling Ssrc'
        Ssrc.shape = (Nfy, Nfx, Nc, Ny_blk, Nx_blk)
        inline(Ssrc_fill, ['Nc', 'Nfy', 'Nfx', 'Ny_blk', 'Nx_blk', 'Dy_idx',
                           'x_start', 'xblk_offsets', 'Ssrc', 'acs', 's'],
               type_converters=blitz)
        Ssrc.shape = (Nfy*Nfx, Nc*Ny_blk*Nx_blk)
##         Ssrc_pinv = np.linalg.pinv(Ssrc)
##         # W is shaped (Nc*Ny_blk*Nx_blk, (R-1)*Nc)
##         W = np.dot(Ssrc_pinv, Stgt)
        W = np.linalg.lstsq(Ssrc, Stgt)[0]
        Dy_idx = syn_rows[:,None] + yblk_offsets
        print 'filling Sacq'
        Sacq.shape = (Nu, Nx, Nv, Nc, Ny_blk, Nx_blk)
        inline(Sacq_fill, ['Nc', 'Nv', 'Nu', 'Nx', 'Ny_blk', 'Nx_blk',
                           'Dy_idx', 'xblk_offsets', 'Sacq', 'cdata', 's'],
               type_converters=blitz)
        print 'done'
        Sacq.shape = (Nu*Nx*Nv, Nc*Ny_blk*Nx_blk)
        
        # the row order of Sacq varies as y_idx, x_idx, vol
        # Ssyn will be shaped (Nu*Nx*Nv, (R-1)*Nc)
        assert np.abs(W.sum()) > 0, 'zero coefs'
        Ssyn = np.dot(Sacq, W)
        assert np.abs(Ssyn.sum()) > 0, 'zero synthesis'
        Ssyn.shape = (Nu,Nx,Nv,(R-1),Nc)
        print 'filling synth rows'
        inline(syn_rows_fill, ['Nc','Nv','Nu','syn_rows','R',
                               'cdata','Ssyn','s'],
               type_converters=blitz)
##         return Sacq, Ssyn, Ssrc, Stgt
        
#-------------- END --------------

Ssrc_fill = \
"""
int xidx, yidx, n, m, i, c, j;
for(n=0; n<Nfy; n++) {
  for(m=0; m<Nfx; m++) {
    for(c=0; c<Nc; c++) {
      for(i=0; i<Ny_blk; i++) {
        yidx = Dy_idx(n,i);
        for(j=0; j<Nx_blk; j++) {
          xidx = (m + x_start + xblk_offsets(j));
          Ssrc(n,m,c,i,j) = acs(c,s,yidx,xidx);
        }
      }
    }
  }
}
"""
Stgt_fill = \
"""
using namespace blitz;
int m, c, r, trow;
for(m=0; m<(R-1); m++) {
  std::cout<<"usings rows: [";
  for(r=0; r<Nfy; r++) std::cout<<tgt_rows(r)+m<<", ";
  std::cout<<"]"<<std::endl;
  for(r=0; r<Nfy; r++) {
    trow = tgt_rows(r)+m;
    for(c=0; c<Nc; c++) {
      Stgt(r,Range::all(),m,c) = acs(c,s,trow,Range(x_start,x_end));
    }
  }
}
"""
Sacq_fill = \
"""
int yidx, n, m, i, c, v;
for(n=0; n<Nu; n++) {
  for(m=0; m<Nx; m++) {
    for(v=0; v<Nv; v++) {
      for(c=0; c<Nc; c++) {
        for(i=0; i<Ny_blk; i++) {
          yidx = Dy_idx(n,i);
          Sacq(m,v,n,c,i) = cdata(c,v,s,yidx,m);
        }
      }
    }
  }
}
"""
Ssyn_fill = \
"""
using namespace blitz;
int m, c, v, r, srow;
for(m=0; m<(R-1); m++) {
  std::cout<<"filling rows: [";
  for(r=0; r<Nu; r++) std::cout<<syn_rows(r)+m<<", ";
  std::cout<<"]"<<std::endl;
  for(r=0; r<Nu; r++) {
    srow = syn_rows(r)+m;
    for(v=0; v<Nv; v++) {
      for(c=0; c<Nc; c++) {
        cdata(c,v,s,srow,Range::all()) = Ssyn(Range::all(),v,r,m,c);
      }
    }
  }
}
"""

def partial_sum_dft(w, kx_idc, N1):
    
    n_sums = w.shape[0]

    

def ksp_calib_hybrid_synth(image, Ny_blk, Nx_blk, sl=-1):
    R = int(image.accel)
    Nc = image.n_chan
    Nacs = int(image.n_acs)
    Nx = image.N1
    Nv = image.n_vol
    Nfy = Nacs - (Ny_blk-1)*R
    Nfx = Nx - (Nx_blk-1)
    x_start = Nx_blk/2 # lower inclusive limit for x target points
    x_end = x_start + Nfx - 1 # upper limit, inclusive, for x target points
    Npe = image.n_pe
    acs = image.cacs_data
    acs.shape = tuple([d for d in acs.shape if d > 1])
    cdata = image.cdata

    # set up 2D neighborhood offsets
    xblk_offsets = np.arange(-(Nx_blk/2), -(Nx_blk/2)+Nx_blk)
    yblk_offsets = np.arange(-((Ny_blk-1)/2), -((Ny_blk-1)/2)+Ny_blk)*R - 1

    # fitting stage
    Ssrc = np.empty((Nfy*Nfx, Nc*Ny_blk*Nx_blk), 'F')
    Stgt = np.empty((Nfy*Nfx, (R-1)*Nc), 'F')
    tgt_row0 = R * ((Ny_blk-1)/2) + 1
    tgt_rows = np.arange(tgt_row0, tgt_row0 + Nfy)

    # transpose this array, since we will be indexing over x
    Wpad = np.zeros((Nx, Nc, Ny_blk, (R-1)*Nc), 'F')
    Wpad_slice = [slice(None)] * len(Wpad.shape)
    kx0 = Nx/2 + xblk_offsets[0]
    kx1 = Nx/2 + xblk_offsets[-1] + 1
    Wpad_slice[0] = slice(kx0, kx1)
    Wpad_slice = tuple(Wpad_slice)
                    
    # synthesis stage
    pe_sampling = image.pe_sampling[:]
    pe_sampling.sort()
    acq_row0 = pe_sampling[0] # == 1
    acq_rowf = pe_sampling[-1]+1  # == last-sampled + 1
    syn_rows = np.arange(tgt_row0+1, acq_rowf, R)
    Nu = len(syn_rows)
    # make Sacq with full readout points, and assume circularity in kx direction
    Sacq = np.empty((Nu*Nx*Nv, Nc*Ny_blk), 'F')
    Ssyn = np.empty((Nx, Nv*Nu, Nc*(R-1)), 'F')

    sl_range = [sl] if sl>=0 else xrange(image.n_slice)
    W_list = []
    for s in sl_range:
        Dy_idx = tgt_rows[:,None] + yblk_offsets
        Stgt.shape = (Nfy, Nfx, (R-1), Nc)
        print 'filling Stgt'
        inline(Stgt_fill, ['R', 'tgt_rows', 'Nfy', 'Nc', 'acs',
                           'Stgt', 'x_start', 'x_end', 's'],
               type_converters=blitz)
        Stgt.shape = (Nfy*Nfx, (R-1)*Nc)

        Ssrc.shape = (Nfy, Nfx, Nc, Ny_blk, Nx_blk)
        print 'filling Ssrc'
        inline(Ssrc_fill, ['Nc', 'Nfy', 'Nfx', 'Ny_blk', 'Nx_blk', 'Dy_idx',
                           'x_start', 'xblk_offsets', 'Ssrc', 'acs', 's'],
               type_converters=blitz)
        Ssrc.shape = (Nfy*Nfx, Nc*Ny_blk*Nx_blk)
        W = np.linalg.lstsq(Ssrc, Stgt)[0]
        W.shape = (Nc, Ny_blk, Nx_blk, (R-1)*Nc)
        W_list.append(W)

    util.ifft1(cdata, inplace=True, axis=-1)
    cdata *= Nx
    for W, s in zip(W_list, sl_range):
        # W indexes in rows as (chan, yn_idx, xn_idx)
        # W indexes in cols as (m, chan)
        Wpad[Wpad_slice] = W.transpose(2, 0, 1, 3)
        Wx = util.ifft1(Wpad, shift=True, inplace=False, axis=0)
        # Now each W(x) index in rows as (chan, yn_idx)
        # Now each W(x) index in cols as (m, chan)
        Wx.shape = (Nx, Nc*Ny_blk, (R-1)*Nc)
        Wx *= Nx
        # want to repeatedly perform the linear combination of (Nc*Ny_blk)
        # points at all combinations of (Nu, Nx)
        # (Nu(x), Nc*Ny_blk) * W(x)
        Dy_idx = syn_rows[:,None] + yblk_offsets
        Sacq.shape = (Nx, Nv, Nu, Nc, Ny_blk)
        print 'filling acquisition rows, offsets', yblk_offsets
        inline(Sacq_fill, ['Nu', 'Nx', 'Nv', 'Nc', 'Ny_blk',
                           'Dy_idx', 'Sacq', 'cdata', 's'],
               type_converters=blitz)
        Sacq.shape = (Nx, Nv*Nu, Nc*Ny_blk)
        print 'synthesizing rows', syn_rows
        for xsyn, xacq, w in zip(Ssyn, Sacq, Wx):
            xsyn[:] = np.dot(xacq, w)
        Ssyn.shape = (Nx, Nv, Nu, R-1, Nc)
        print 'filling synth rows'
        inline(Ssyn_fill, ['Nc','Nv','Nu','syn_rows','R',
                           'cdata','Ssyn','s'],
               type_converters=blitz)
    util.fft1(cdata, inplace=True, axis=-1)
    
        
        
        
def find_weights(err, axis=0):
    # find weights based on error such that sum(weights, axis) = 1
    val = 1/err
    w = (1/err).sum(axis=axis)
    s = [slice(None)]*len(val.shape)
    s[axis] = None
    return val/w[s]
    
class GrappaSynthesize (Operation):
    params = (Parameter(name='nblocks', type='int', default=4),
              Parameter(name='sliding', type='bool', default=False),
              Parameter(name='n1_window', type='int', default=None),
              Parameter(name='floating_window', type='bool', default=False,
                        description="""
    Toggles whether the readout neighborhood window is always centered on the
    current column (floating), or whether the neighborhoods are fixed segments.
    """),
                        
              Parameter(name='ft', type='bool', default=False,
                        description="""
    fourier transform readout direction prior to GRAPPA fitting/synthesis
    """
                        ),
              Parameter(name='beloud', type='bool', default=False,
                        description="""
    Allow copious information about the GRAPPA process to be printed on screen
    """
                        ),
              )


    @ChannelAwareOperation
    def run(self, image):
        accel = int(image.accel)
        sub_data = image.cdata # shape (nc, nsl, n2, n1)
        acs = image.cacs_data
        # cheap fix for new shape of potentially multi-acquisition acs
        if len(acs.shape) > 4:
            acs = acs[:,0,:,:,:]
        # transpose to (nsl, nc, n2, n1)
        acs = acs.transpose(1, 0, 2, 3).copy()
        if self.ft:
            print "transforming readout..."
            util.ifft1(sub_data, inplace=True, shift=True)
            util.ifft1(acs, inplace=True, shift=True)

        n2_sampling = image.pe_sampling[:]
        n2_sampling.sort()

        for s in range(image.n_slice):
            N, e = grappa_coefs(acs[s], accel, self.nblocks,
                                sliding=self.sliding,
                                n1_window=self.n1_window,
                                loud=self.beloud,
                                fixed_window=not self.floating_window)
            # find weightings for each slide reconstruction based on the
            # errors of their fits (slide enumeration in axis=-2)
            w = find_weights(e, axis=-2)
            if self.beloud:
                print e
                print w.sum(axis=-2)
            Ssub = sub_data[:,:,s,:,:]
            Ssub[:] = grappa_synthesize(Ssub, N, n2_sampling,
                                        weights=w, loud=self.beloud,
                                        fixed_window=not self.floating_window)

        if self.ft:
            util.fft1(sub_data, inplace=True, shift=True)
            util.fft1(acs, inplace=True, shift=True)
        # by convention, set view to channel 0 
        image.use_membuffer(0)
            
            
class PreWhitenChannels (Operation):

    params = (
        Parameter(name='go_slow', type='bool', default=False,
                  description="""break up transform into smaller blocks"""),
        )

    @staticmethod
    def cov_mat(s, full=False):
        from neuroimaging.timeseries import algorithms as alg
        m,n = s.shape
        cm = np.zeros((m,m), 'D')
        _, csd_list = alg.multi_taper_csd(s, BW=5.0/n)
        for i in xrange(m):
            for j in xrange(i+1):
                cm[i,j] = np.trapz(csd_list[i][j], dx=1.0/n)
        if full:
            cm = cm + np.tril(cm, -1).T.conj()
        return cm

    @staticmethod
    def cov_mat_biased(s, full=False):
        m,n = s.shape
        cm = np.zeros((m,m), 'D')
        for i in xrange(m):
            for j in xrange(i+1):
                cm[i,j] = (s[i].conjugate() * s[j]).mean()
        if full:
            cm = cm + np.tril(cm, -1).T.conj()
        return cm
    
    @ChannelAwareOperation
    def run(self, image):
        if not hasattr(image, 'n_chan'):
            return
        from recon.scanners import siemens
        n_chan = image.n_chan
        dat = siemens.MemmapDatFile(image.path, n_chan)
        nz_scan = np.empty((image.n_chan, image.M1), 'F')
        nz_scan[:] = dat[:]['data']
##         del dat
##         image.nz_scan = nz_scan
        covariance = PreWhitenChannels.cov_mat(nz_scan)
        l, v = np.linalg.eigh(covariance, UPLO='L')
        W = np.dot(v, (l**(-1/2.))[:,None] * v.conjugate().T).astype('F')
        arr_names = ['cdata', 'cacs_data']
        arrs = [getattr(image, arr_name)
                for arr_name in arr_names
                if hasattr(image, arr_name)]
        if self.go_slow:
            print 'foo'
            for arr in arrs:
                for s in xrange(image.n_slice):
                    sl = arr[:,:,s,:,:].copy()
                    sl_shape = sl.shape
                    sl.shape = (sl_shape[0], np.prod(sl_shape[1:]))
                    arr[:,:,s,:,:] = np.dot(W, sl).reshape(sl_shape)
            del sl
        else:
            for arr, arr_name in zip(arrs, arr_names):
                arr_shape = arr.shape
                arr.shape = (arr.shape[0], np.prod(arr.shape[1:]))
                setattr(image, arr_name, np.dot(W, arr).reshape(arr_shape))
                del arr
        
                
                

                
##             cd_shape = image.cdata.shape
##             image.cdata.shape = (n_chan, np.prod(cd_shape[1:]))
##             acs_shape = image.cacs_data.shape
##             image.cacs_data.shape = (n_chan, np.prod(acs_shape[1:]))
##             cd = np.dot(W, image.cdata)
##             cd.shape = cd_shape
##             del image.cdata
##             image.cdata = cd
##             acs = np.dot(W, image.cacs_data)
##             acs.shape = acs_shape
##             del image.cacs_data
##             image.cacs_data = acs
        image.use_membuffer(0)
