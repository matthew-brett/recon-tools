import numpy as np, pylab as P
#from scipy import linalg as splinalg
from recon import imageio, util
# assume data shapes are (nchan, npe, nfe)

def smash_recon(S, a=2, acs_rep=1, restore_acs=False, validate=False,
                cols=[], n1_window=None):
    Ssub = sparsify_data(S, a=a, acs_rep=acs_rep)
    Nlm, err = smash_coefs(Ssub, a=a, acs_rep=acs_rep,
                           cols=cols, n1_window=n1_window)
    print "error for each solution:", err
    Ssyn = smash_synthesize(Ssub, Nlm, a=a,
                            acs_rep=acs_rep, restore_acs=restore_acs)
    if validate:
        Sref = S.sum(axis=0)
        n_gen = a-1
        n_pe = Sref.shape[-2]
        gen_lines, acs_lines, synth_harmonics = smash_lines_decomp(n_pe, a,
                                                                   acs_rep)[:3]
        redund = np.intersect1d(np.array(gen_lines), np.array(acs_lines))
        speedup = n_pe / float(len(gen_lines)+len(acs_lines)-redund.shape[0])
        syn_sl = [slice(None)]*2
        # subplot(n11), subplot(n12), ...
        eps = 1e-14
        plot_num = n_gen*100 + 11
        for i,m in enumerate(synth_harmonics):
            synth_lines = np.array(gen_lines) + m
            valid_idx = (synth_lines>=0) & (synth_lines<n_pe)
            synth_lines = synth_lines[valid_idx]
            syn_sl[-2] = synth_lines
            nrg_err = sum_sqrs(Sref[synth_lines] - Ssyn[synth_lines], axis=-1)
            nrg_sig = sum_sqrs(Sref[synth_lines], axis=-1)
            P.subplot(plot_num+i)
            P.semilogy(synth_lines, nrg_err+eps, label='err')
            P.semilogy(synth_lines, nrg_sig+eps, label='sig')
            if restore_acs:
                P.semilogy(acs_lines, np.zeros(len(acs_lines))+eps, 'ro')
            P.gca().legend()
            P.gca().set_title('error and signal energy, m=%d'%m)
        P.gcf().text(.5, 0, 'speedup=%1.2fx'%speedup, ha='center')
        P.show()
        
    return Ssyn

def grappa_for_real(epi, a=2, acs_rep=12, acs_offset=19, **kw):
    for sl in range(epi.n_slice):
        acs = epi.cacs_data[:,0,sl,:,:]
        Ssub = epi.cdata[:,0,sl,:,:]
        Ssub[:] = grappa_recon(Ssub, acs_blk=acs, a=a, acs_rep=acs_rep, **kw)
    epi.load_chan(0, copy=True)
        

def grappa_recon(S, acs_blk=None, a=2, acs_rep=1, blocks=0, slides=0,
                 avg=False, validate=False, restore_acs=False,
                 n1_window=None, acs_offset=0, rcond=-1):
    if acs_blk is None:
        acs_blk = S
    if not hasattr(acs_rep, '__getitem__'):
        acs_rep = [acs_rep]
    sig_plotted = np.zeros(a-1)
    for arep in acs_rep:
        n_pe = S.shape[-2]
        gen_lines, acs_lines, synth_harmonics = smash_lines_decomp(n_pe, a,
                                                                   arep)[:3]
        Ssub = sparsify_data(S, a=a, acs_rep=arep)
        if blocks and not slides:
            blk = np.arange(-((blocks-1)/2), -((blocks-1)/2) + blocks)
            N, e = grappa_coefs_blocks(acs_blk, a=a, acs_rep=arep, blocks=blk,
                                       avg=avg, acs_offset=acs_offset,
                                       rcond=rcond)[:2]
            Ssyn = grappa_synthesize_blocks(Ssub, N, a=a, acs_rep=arep)
        elif blocks and slides:
            N_list = []
            err_list = []
            for s in range(-(slides/2), -(slides/2)+slides):
            #for b in range(-blocks+1,1):
                blk = np.arange(-((blocks-1)/2)+s, -((blocks-1)/2)+blocks+s)
                N, e = grappa_coefs_blocks(acs_blk, a=a, acs_rep=arep,
                                           blocks=blk, avg=avg,
                                           acs_offset=acs_offset,
                                           rcond=rcond)[:2]
                N_list.append(N)
                err_list.append([e])
            errs = np.concatenate(err_list, axis=0)
            weights = find_weighting(errs)
            Ssyn = np.zeros_like(Ssub)
            for s in range(slides):
                Ssyn += grappa_synthesize_blocks(Ssub, N_list[s], a=a,
                                                 acs_rep=arep,
                                                 weights=weights[s])
            Ssyn[:,gen_lines,:] = Ssub[:,gen_lines,:]
        else:
            N = grappa_coefs_noblocks(acs_blk, a=a, acs_rep=arep,
                                      n1_window=n1_window,
                                      acs_offset=acs_offset, rcond=rcond)[0]
            Ssyn = grappa_synthesize(Ssub, N, a=a,
                                     acs_rep=arep,
                                     restore_acs=restore_acs)

        if validate:
            n_gen = a-1
            redund = np.intersect1d(np.array(gen_lines), np.array(acs_lines))
            speedup = n_pe / float(len(gen_lines)+len(acs_lines)-redund.shape[0])
            syn_sl = [slice(None)]*3
            # subplot(n11), subplot(n12), ...
            eps = 1e-14
            plot_num = n_gen*100 + 11
            for i,m in enumerate(synth_harmonics):
                synth_lines = np.array(gen_lines) + m
                valid_idx = (synth_lines>=0) & (synth_lines<n_pe)
                synth_lines = synth_lines[valid_idx]
                syn_sl[-2] = synth_lines
                nrg_err = sum_sqrs(S[syn_sl] - Ssyn[syn_sl], axis=-1)
                nrg_err = sum_sqrs(nrg_err, axis=0)
                nrg_sig = sum_sqrs(S[syn_sl], axis=-1)
                nrg_sig = sum_sqrs(nrg_sig, axis=0)
                P.subplot(plot_num+i)
                if not sig_plotted[i]:
                    P.semilogy(synth_lines, nrg_sig+eps, label='sig')
                    sig_plotted[i] = 1
                P.semilogy(synth_lines, nrg_err+eps, label='err@%1.2fx'%speedup)
                if restore_acs:
                    P.semilogy(acs_lines, np.zeros(len(acs_lines))+eps, 'ro')
                P.gca().legend()
                ttl = 'error and signal energy, m=%d'%m
                P.gca().set_title(ttl)
        #P.gcf().text(.5, 0, 'speedup=%1.2fx'%speedup, ha='center')
        #P.show()



    return Ssyn

def sum_sqrs(S, axis=0):
    sshape = list(S.shape)
    s_slice = [slice(None)]*len(sshape)
    dim_len = sshape.pop(axis)
    if sshape:
        ssq = np.zeros(sshape, 'd')
    else:
        ssq = np.zeros((1,), 'd')
    for c in range(dim_len):
        s_slice[axis] = c
        ssq += (S[s_slice].real**2 + S[s_slice].imag**2)
    np.sqrt(ssq, ssq)
    if sshape:
        ssq.shape = sshape
    else:
        ssq.shape = ()
    return ssq

def find_weighting(errs):
    """Given a list of Mx[KxLx...x]N errors, where M is the number of different
    fits and N is the number of right-hand-side solutions, compute a MxN list
    of combination weightings such that (weights.sum(axis=0) == 1).all() and
    weights[i,:]/weights[j,:] == errs[i,:]/errs[j,:] for any i,j
    """
    scale = 1.0/errs.sum(axis=0)
    return errs*scale
    

    
def smash_lines_decomp(n_pe, a, acs_rep):
    k0 = n_pe/2
    acs = a-1
    # acs lines
    acs_lines = range(k0-acs/2, k0+acs/2)
    if acs%2:
        acs_lines.append(k0+acs/2)
    # find first line.. such that sparse sampling does not sample k0 or any acs
    pe0 = min(acs_lines) - 1
    while(pe0 >= a): pe0 -= a
    # regular sparse pe acquisition
    lines = range(pe0, n_pe, a)

    synth_harmonics = [ (-1)**m * (m/2+1) for m in range(a-1) ]

    acs_reps = []
    for acs in acs_lines:
        acs_reps += [acs + a*((-1)**m * (m+1)/2) for m in range(acs_rep)]
    acs_lines = acs_reps
    acs_lines.sort()
    fit_lines = []
    # could do this with the intersection of acs_lines and sub sampled
    # lines, but in EPI acquisition this is not a realistic model
    acs_blk = range(acs_lines[0]-1, acs_lines[-1]+1)
    for m in synth_harmonics:

        dat_lines = [p for p in acs_blk if p-m in acs_blk]
        sys_lines = [p-m for p in dat_lines]
        
        fit_lines.append( (sys_lines, dat_lines) )
    
    return lines, acs_lines, synth_harmonics, fit_lines
    

def sparsify_data(S, a=2, acs_rep=1):
    n_pe = S.shape[-2]
    sub_lines, acs_lines = smash_lines_decomp(n_pe, a, acs_rep)[:2]
    Ssub = np.zeros_like(S)
    sub_samp = [slice(None)] * len(S.shape)
    sub_samp[-2] = sub_lines
    Ssub[sub_samp] = S[sub_samp]
    sub_samp[-2] = acs_lines
    Ssub[sub_samp] = S[sub_samp]
    return Ssub

def sparsify_epi_data(S, a=2, acs_rep=1):

    n_ch, n_pe, n_fe = S.shape
    sub_lines, acs_lines = smash_lines_decomp(n_pe, a, acs_rep)[:2]
    Ssub = np.zeros_like(S)
    Sacs = np.empty((n_ch, a*acs_rep, n_fe), S.dtype)
    sub_samp = [slice(None)] * len(S.shape)
    sub_samp[-2] = slice(sub_lines[0], sub_lines[-1]+1, a)
    Ssub[sub_samp] = S[sub_samp]
    acs_start = acs_lines[0] in sub_lines and acs_lines[0] or acs_lines[0]-1
    sub_samp[-2] = slice(acs_start, acs_lines[-1]+1, 1)
    Sacs[:] = S[sub_samp]
    return Ssub, Sacs

def smash_coefs(Ssub, a=2, acs_rep=1, cols=[], n1_window=None):
    """ Solve S_sub*Nlm = Sacs for Nlm shaped (accel_fac-1, [n_window], n_ch)
    """
    n_ch, n_pe, n1 = Ssub.shape
    (acs_lines, synth_harmonics,
     fit_lines) = smash_lines_decomp(n_pe, a, acs_rep)[1:]
    if cols and not n1_window:
        n1 = len(cols)
        n_win = 1
    elif n1_window:
        # make this the catch all case if cols and n1_window are specified
        if n1_window=='max':
            win_sz = n1
            n_win = 1
        elif n1_window=='min':
            # this is the number of rows in the acs block
            n_acs = acs_lines[-1] - (acs_lines[0]-1) + 1
            # this is the number of solutions by combining rows (worse case)
            n_soln = n_acs - max(synth_harmonics)
            # want min L such that L*n_soln >= n_ch
            win_sz = n_ch / n_soln
            if win_sz * n_soln < n_ch:
                win_sz += 1
            n_win = n1 - win_sz + 1
        else:
            assert type(n1_window)==type(1), "n1_window argument not understood: "+str(n1_window)
            win_sz = n1_window>0 and min(n1, n1_window) or 1
            n_win = n1 - win_sz + 1
    else:
        # choose all read-out points
        win_sz = n1
        n_win = 1
    
    dat_slice = [slice(None)] * len(Ssub.shape)
    dat_slice[-2] = acs_lines
    sys_slice = [slice(None)] * len(Ssub.shape)

    Nlm = np.zeros((a-1, n_win, n_ch), Ssub.dtype)
    err = np.zeros((a-1, n_win), 'd')
    for i,(m, rows) in enumerate(zip(synth_harmonics, fit_lines)):
        sys_rows, dat_rows = rows
        sys_slice[-2] = slice(sys_rows[0], sys_rows[-1]+1)
        dat_slice[-2] = slice(dat_rows[0], dat_rows[-1]+1)
        print "%dth harmonic: using"%m, sys_rows, "to fit to", dat_rows        
        assert len(sys_rows)==len(dat_rows), 'mismatch of sys_rows and dat_rows'
        nrow = len(sys_rows)
        assert nrow*win_sz >= n_ch, 'underdetermined problem, increase win size'
        print "problem shape:", (nrow*win_sz, n_ch), (n_ch, 1), (nrow*win_sz, 1)
        for w in range(n_win):
            dat_slice[-1] = slice(w, w+win_sz)
            sys_slice[-1] = slice(w, w+win_sz)
            # this is shaped (nrow, win_sz), reshape to (nrow*win_sz,)
            Sdat = Ssub[dat_slice].sum(axis=0).reshape((nrow*win_sz,)).copy()
            # this is shaped (n_ch, nrow, win_sz), rehsape to (n_ch,nrow*win_sz)
            Ssys = Ssub[sys_slice].reshape((n_ch, win_sz*nrow))
            Ssys = Ssys.T.copy()
            
            # so Ssys * Nlm = Sdat is a problem shaped:
            # (nrows*win_sz x n_ch) * (n_ch x 1) = (nrows*win_sz x 1)
            
            Nlm[i,w], e = lstsq(Ssys, Sdat)[:2]
            err[i,w] = e[0]
        
    return Nlm, err

def smash_synthesize(Ssub, Nlm, a=2, acs_rep=1, restore_acs=False):
    """ Synthesize the missing kspace rows in Ssub with coefficients in Nlm.
    """
    n_pe, n1 = Ssub.shape[-2:]
    gen_lines, acs_lines, synth_harmonics = smash_lines_decomp(n_pe,
                                                               a, acs_rep)[:-1]
    Ssyn = Ssub.sum(axis=0)


    n_win = Nlm.shape[1]
    win_sz = n1 - n_win + 1
    win_idx = [0,]*(win_sz/2) + range(0,n_win-1) + [n_win-1]*(win_sz/2+1)
    
    synth_slice = [slice(None)] * len(Ssyn.shape)
    gen_slice = [slice(None)] * len(Ssub.shape)

    if restore_acs:
        synth_slice[-2] = acs_lines
        acs_save = Ssyn[synth_slice].copy()

    for i,m in enumerate(synth_harmonics):
        synth_lines = np.array(gen_lines)+m
        valid_idx = (synth_lines>=0) & (synth_lines<n_pe)
        synth_lines = synth_lines[valid_idx]
        synth_slice[-2] = synth_lines
        gen_slice[-2] = synth_lines-m
        print "%dth harmonic: using"%m,synth_lines-m,"to generate",synth_lines
        for col in range(n1):
            w = win_idx[col]
            #print "column %d using window %d"%(col, w)
            gen_slice[-1] = slice(col, col+1)
            synth_slice[-1] = slice(col, col+1)
            Ssys = Ssub[gen_slice].transpose()
            Ssyn[synth_slice] = np.dot(Ssys, Nlm[i,w]).T

    if restore_acs:
        synth_slice[-2] = acs_lines; synth_slice[-1] = slice(None)
        Ssyn[synth_slice] = acs_save
    return Ssyn


def grappa_coefs_noblocks(acs_blk, a=2, acs_rep=1, n1_window=None,
                          acs_offset=0, rcond=-1):
    """Fit system lines to data lines to find tensor n(m,n_win,j,l)
    """
    n_ch, n_pe, n1 = acs_blk.shape
    (acs_lines, synth_harmonics,
     fit_lines) = smash_lines_decomp(n_pe, a, acs_rep)[1:]
    
    if n1_window:
        if n1_window=='max':
            win_sz = n1
            n_win = 1
        elif n1_window=='min':
            # this is the number of rows in the acs block
            n_acs = acs_lines[-1] - (acs_lines[0]-1) + 1
            # this is the number of solutions by combining rows (worse case)
            n_soln = n_acs - max(synth_harmonics)
            # want min L such that L*n_soln >= n_ch
            win_sz = n_ch / n_soln
            if win_sz * n_soln < n_ch:
                win_sz += 1
            n_win = n1 - win_sz + 1
        else:
            assert type(n1_window)==type(1), "n1_window argument not understood: "+str(n1_window)
            win_sz = n1_window>0 and min(n1, n1_window) or 1
            n_win = n1 - win_sz + 1
    else:
        # choose all read-out points
        win_sz = n1
        n_win = 1
    
    N = np.empty((a-1, n_win, n_ch, n_ch), acs_blk.dtype)
    err = np.empty((a-1, n_win, n_ch), 'd')
    sys_slice = [slice(None)]*3
    dat_slice = [slice(None)]*3
    for i,(m, rows) in enumerate(zip(synth_harmonics, fit_lines)):
        sys_rows, dat_rows = rows
        print "%dth harmonic: using"%m, sys_rows, "to fit to", dat_rows
        sys_slice[-2] = slice(sys_rows[0]-acs_offset, sys_rows[-1]-acs_offset+1)
        dat_slice[-2] = slice(dat_rows[0]-acs_offset, dat_rows[-1]-acs_offset+1)
        nrow = len(dat_rows)
        print "problem size:", (win_sz*nrow, n_ch), (n_ch, n_ch), (win_sz*nrow, n_ch)
        for w in range(n_win):
            sys_slice[-1] = slice(w, w+win_sz)
            dat_slice[-1] = slice(w, w+win_sz)
            # this is (win_sz, nrow, n_ch)
            s = acs_blk[sys_slice].reshape((n_ch, win_sz*nrow))
            s = s.T.copy()
            # this is (win_sz, nrow, n_ch)
            d = acs_blk[dat_slice].reshape((n_ch, win_sz*nrow))
            d = d.T.copy()
            if rcond<0:
                N[i,w], err[i,w] = lstsq(s, d)[:2]
            else:
                N[i,w] = lstsq(s, d, rcond=rcond)[0]
    return N, err

def grappa_coefs_blocks(acs_blk, a=2, acs_rep=1, blocks=None,
                        avg=False, rcond=-1, acs_offset=0):
    """Fit system lines to data lines to find tensor n(m,b,n1,j,l), using
    an EPI style ACS acquisition block.
    """

    # blocks currently hardwired to {-1,0,1,2}*A*dk
    if blocks is None:
        nblk = 4
        blocks = np.array([-1,0,1,2], 'i')
    else:
        blocks = np.asarray(blocks, dtype='i')
        nblk = blocks.shape[0]

    
    n_ch, n_pe, n1 = acs_blk.shape
    (gen_lines, acs_lines, synth_harmonics,
     fit_lines) = smash_lines_decomp(n_pe, a, acs_rep)

    # change later
    win_sz = n1
    n_win = 1

    N = np.empty((a-1, n_win, n_ch, nblk, n_ch), acs_blk.dtype)
    err = np.zeros((a-1, n_win, n_ch), 'd')
    sys_slice = [slice(None)]*3
    dat_slice = [slice(None)]*3
    # these numbers hardwired
    #sys = np.empty((n1, n_ch*nblk), acs_blk.dtype)
    #dat = np.empty((n1, n_ch), acs_blk.dtype)
    for i,(m, rows) in enumerate(zip(synth_harmonics, fit_lines)):
        # for a=2, acs_rep=1 this will say to fit k0-1 to k0 for m=1..
        # here I'll repeat over blocks also
        sys_rows, dat_rows = rows
        sys_rows = np.array([r for r in sys_rows if \
                             (r+blocks[0]*a in gen_lines and
                              r+blocks[-1]*a in gen_lines)])
        dat_rows = np.array([r+m for r in sys_rows])
        nrow = len(sys_rows)
        dat_slice[-2] = dat_rows-acs_offset
        # reshape (n_ch, nrow, N1) to (n_ch, nrow*N1) and transpose
        dat = acs_blk[dat_slice].reshape(n_ch, nrow*n1).transpose().copy()
        # the columns represent the block offsets,
        # the rows represent different fits
        sys_rows = np.concatenate([sys_rows[:,None] + b*a for b in blocks],
                                  axis=-1)
        print sys_rows, dat_rows
        sys = np.empty((nrow*n1, n_ch*nblk), acs_blk.dtype)
        # so if sys_rows is PxQ, the system matrix will be NxM where...
        # N = nrow*N1*P, M = N_ch*Q
        for n,r in enumerate(sys_rows):
            # row indexes are r
            sys_slice[-2] = r-acs_offset
            # reshape acs_blk[sys_slice] (n_ch, n_blk, N1) to (n_ch*n_blk, N1)
            s = acs_blk[sys_slice].reshape(n_ch*nblk, n1)
            # put (N1, n_ch*nblk) submatrix into sys
            sys[n*n1:(n+1)*n1] = s.transpose().copy()
        # sys is finally (nrow*N1, n_ch*nblk), dat is (nrow*N1, n_ch)...
        # solve Ax = b for (n_ch*nblk, n_ch) column vectors x
        if avg:
            # split this up into nrow different solutions, and average the
            # solutions with uniform weighting.
            soln = np.zeros((n_ch*nblk, n_ch), acs_blk.dtype)
            for n in range(nrow):
                r = lstsq(sys[n*n1:(n+1)*n1], dat[n*n1:(n+1)*n1], rcond=rcond)
                soln += r[0] #.reshape(n_ch, nblk, n_ch)
                if rcond<0:
                    err[i,0] += r[1]
                    print "matrix condition #:",r[-1][0]/r[-1][-1]
                else:
                    print "matrix condition #:",r[-1][0]/r[-1][-1], "improved condition #:",r[-1][0]/r[-1][r[-2]-1],"using %d columns"%r[-2]
            np.divide(soln, nrow, soln)
        else:
            r = lstsq(sys, dat, rcond=rcond)
            soln = r[0]
            if rcond<0:
                err[i,0] = r[1]
                print "matrix condition #:",r[-1][0]/r[-1][-1]
            else:
                print "matrix condition #:",r[-1][0]/r[-1][-1], "improved condition #:",r[-1][0]/r[-1][r[-2]-1],"using %d columns"%r[-2]
        
        N[i,0] = soln.reshape(n_ch, nblk, n_ch)
        #err[i,0] = r[1]
    return N, err

def grappa_synthesize(Ssub, N, a=2, acs_rep=1, restore_acs=False):
    n_pe, n1 = Ssub.shape[-2:]
    gen_lines, acs_lines, synth_harmonics = smash_lines_decomp(n_pe,
                                                               a, acs_rep)[:-1]

    Ssyn = Ssub.copy()

    n_win = N.shape[1]
    win_sz = n1 - n_win + 1
    win_idx = [0,]*(win_sz/2) + range(0,n_win-1) + [n_win-1]*(win_sz/2+1)
    print win_idx, len(win_idx), win_sz, n_win
    synth_slice = [slice(None)]*3
    gen_slice = [slice(None)]*3

    if restore_acs:
        synth_slice[-2] = acs_lines
        acs_save = Ssyn[synth_slice].copy()
    
    for i,m in enumerate(synth_harmonics):
        synth_lines = np.array(gen_lines)+m
        valid_idx = (synth_lines>=0) & (synth_lines<n_pe)
        synth_lines = synth_lines[valid_idx]
        synth_slice[-2] = synth_lines
        gen_slice[-2] = synth_lines-m
        print "%dth harmonic: using"%m,synth_lines-m,"to generate",synth_lines
        for col in range(n1):
            w = win_idx[col]
            synth_slice[-1] = col #slice(col, col+1)
            gen_slice[-1] = col #slice(col, col+1)
            s = Ssub[gen_slice].transpose().copy()
            Ssyn[synth_slice] = np.dot(s, N[i,w]).T
            # try to match energy in acquired lines and synth lines
            nrg_synth = np.sqrt((Ssyn[synth_slice].real**2 + \
                                 Ssyn[synth_slice].imag**2).sum())
            nrg_acq = np.sqrt((Ssub[gen_slice].real**2 + \
                               Ssub[gen_slice].imag**2).sum())
            #Ssyn[synth_slice] *= nrg_acq/nrg_synth
    
    if restore_acs:
        synth_slice[-2:] = (acs_lines, slice(None))
        Ssyn[synth_slice] = acs_save

    return Ssyn

def grappa_synthesize_blocks(Ssub, N, a=2, acs_rep=1, restore_acs=False,
                             weights=None):
    # blocks currently hardwired to {-1,0,1,2}*A*dk
    nblk = 4 # or N.shape[-2]
    blocks = np.array([-1,0,1,2], 'i')
    
    n_ch, n_pe, n1 = Ssub.shape
    gen_lines, acs_lines, synth_harmonics = smash_lines_decomp(n_pe,
                                                               a, acs_rep)[:-1]

    Ssyn = Ssub.copy()
    if weights is None:
        Nshape = N.shape
        weights = np.ones(Nshape[:2]+(N.shape[-1],), 'd')

    n_win = N.shape[1]
    win_sz = n1 - n_win + 1
    win_idx = [0,]*(win_sz/2) + range(0,n_win-1) + [n_win-1]*(win_sz/2+1)
    synth_slice = [slice(None)]*3
    gen_slice = [slice(None)]*3
    for i,m in enumerate(synth_harmonics):
        synth_lines = np.array(gen_lines)+m
        valid_idx = (synth_lines>=0) & (synth_lines<n_pe)
        synth_lines = synth_lines[valid_idx]
        synth_slice[-2] = synth_lines
        gen_lines = synth_lines - m
        gen_lines = np.concatenate([gen_lines[:,None] + b*a for b in blocks],
                                   axis=-1)
        # each row = gen_lines[i] lists the indices of the source points for
        # generating each synthesized line = synth_lines[i]
        print gen_lines, synth_lines
        for col in range(n1):
            w = win_idx[col]
            synth_slice[-1] = col
            gen_slice[-1] = col
            
            # want to create a matrix shaped (len(synth_lines), n_ch*nblk)
            # to multiply N[i,win_idx].reshape(n_ch*nblk, n_ch)
            gen_sys = []
            for r in gen_lines:
                # rectify the indices, by replacing invalid indices with
                # an index to zero-data
                np.putmask(r, r<0, synth_lines[0])
                np.putmask(r, r>=n_pe, synth_lines[0])
                gen_slice[-2] = r
                gen_sys.append(Ssub[gen_slice].reshape((1,n_ch*nblk)).copy())
            gen_sys = np.concatenate(gen_sys, axis=0)
            coefs = N[i,w].reshape(n_ch*nblk, n_ch)
            # weights[i,w] is an n_ch long array
            Ssyn[synth_slice] = (weights[i,w]*np.dot(gen_sys, coefs)).T

    return Ssyn    

def linear_interp_synth(Ssub, a=2):
    n_pe = Ssub.shape[1]
    Ssyn = Ssub.copy()
    gen_lines = smash_lines_decomp(n_pe, a, 1)[0]
    gen_slice_lo = [slice(None)]*3
    gen_slice_hi = [slice(None)]*3
    gen_slice_lo[-2] = slice(gen_lines[0], gen_lines[-2]+1, a)
    gen_slice_hi[-2] = slice(gen_lines[1], gen_lines[-1]+1, a)
    syn_slice = [slice(None)]*3
    for m in range(1,a):
        # find points bounded by sampled points
        syn_slice[-2] = slice(gen_lines[0]+m, gen_lines[-1], a)
        Ssyn[syn_slice] = ( float(a-m)/a*Ssub[gen_slice_lo] + \
                            float(m)/a*Ssub[gen_slice_hi] )
    return Ssyn
    

def smash_coefs_sanity(S, a=2, acs_rep=1):
    Ssub = sparsify_data(S, a=a, acs_rep=acs_rep)
    Nlm, err = smash_coefs(Ssub, a=a, acs_rep=acs_rep)
    n_ch, n_pe, n1 = S.shape
    (acs_lines, synth_harmonics,
     fit_lines) = smash_lines_decomp(n_pe, a, acs_rep)[1:]

    sys_slice = [slice(None)]*3
    err_slice = [slice(None)]*3

    for i, (m, rows) in enumerate(zip(synth_harmonics, fit_lines)):
        sys_rows, err_rows = rows
        nrow = len(sys_rows)
        sys_slice[-2] = slice(sys_rows[0], sys_rows[-1]+1)
        err_slice[-2] = slice(err_rows[0], err_rows[-1]+1)
        ref = S[err_slice].sum(axis=0)
        ref.shape = (nrow*n1,)

        sys = Ssub[sys_slice].reshape((n_ch, n1*nrow))
        sys = sys.T.copy()

        synth = np.dot(sys, Nlm[i,0])

        this_err = (ref - synth)
        this_err_l2 = np.array([np.dot(this_err, this_err.conjugate()).real])
        assert np.allclose(this_err_l2, err[i])
        
def grappa_coefs_sanity(S, a=2, acs_rep=1):
    Ssub = sparsify_data(S, a=a, acs_rep=acs_rep)    
    Nlm, err = grappa_coefs_noblocks(Ssub, a=a, acs_rep=acs_rep)
    n_ch, n_pe, n1 = S.shape
    (acs_lines, synth_harmonics,
     fit_lines) = smash_lines_decomp(n_pe, a, acs_rep)[1:]

    sys_slice = [slice(None)]*3
    err_slice = [slice(None)]*3

    for i, (m, rows) in enumerate(zip(synth_harmonics, fit_lines)):
        sys_rows, err_rows = rows
        nrow = len(sys_rows)
        sys_slice[-2] = slice(sys_rows[0], sys_rows[-1]+1)
        err_slice[-2] = slice(err_rows[0], err_rows[-1]+1)

        ref = Ssub[err_slice].reshape((n_ch, n1*nrow))
        ref = ref.T.copy()

        sys = Ssub[sys_slice].reshape((n_ch, n1*nrow))
        sys = sys.T.copy()

        synth = np.dot(sys, Nlm[i,0])
        this_err = synth-ref
        this_err_l2 = np.array([np.dot(v.conjugate(),v) for v in this_err.T]).real
        assert np.allclose(err[i,0], this_err_l2)

def regularized_solve(A, b, c=1e-4):
    """Solve Ax=b with Tikhonov regularization.. set up problem such that
    (A*A + cI)x = A*b
    """
    m,n = A.shape
    A2 = np.dot(A.conjugate().transpose().copy(), A)
    # A2 is shaped nxn, add c*I to diagonal
    A2.flat[0:n**2:n+1] += c
    b2 = np.dot(A.conjugate().transpose(), b)
    x = np.linalg.solve(A2, b2)
    # this should be nxn
    res = np.dot(A2, x) - b2
    err = np.array([np.dot(col, col) for col in res.T])
    return x, err

def load_data_prepd(fname, sl=0, N1=64):
    img = imageio.readImage(fname, N1=N1)
    cdata = np.empty((img.n_chan, img.n_pe, img.N1), 'F')
    cdata[:] = img.cdata[:,0,sl,:,:]
    del img.n_chan
    del img.cdata
    img.setData(cdata)
    return img

def least_squares_sanity():
    m = np.random.randint(12, high=1000)
    n = 12
    A = np.random.rand(m,n).astype('F')
    A.imag[:] = np.random.rand(m,n)
    B = np.random.rand(m,n).astype('F')
    B.imag[:] = np.random.rand(m,n)

    X, err = lstsq(A,B)[:2]
    Bp = np.dot(A,X)
    my_err = Bp - B
    my_l2_err = np.array([np.dot(col.conjugate(), col) for col in my_err.T]).real
    print my_l2_err, err
    assert np.allclose(my_l2_err, err)
    #return A,B,my_err

from numpy.core import zeros, intc, array, ravel, sum, transpose, \
     fastCopyAndTranspose, newaxis
from numpy.linalg.linalg import _makearray, _assertRank2, LinAlgError, \
     _commonType, _linalgRealType, _fastCopyAndTranspose, isComplexType, \
     _realType

from numpy.linalg import lapack_lite

fortran_int = intc
def lstsq(a, b, rcond=-1):
    """
    Return the least-squares solution to an equation.

    Solves the equation `a x = b` by computing a vector `x` that minimizes
    the norm `|| b - a x ||`.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Input equation coefficients.
    b : array_like, shape (M,) or (M, K)
        Equation target values.  If `b` is two-dimensional, the least
        squares solution is calculated for each of the `K` target sets.
    rcond : float, optional
        Cutoff for ``small`` singular values of `a`.
        Singular values smaller than `rcond` times the largest singular
        value are  considered zero.

    Returns
    -------
    x : ndarray, shape(N,) or (N, K)
         Least squares solution.  The shape of `x` depends on the shape of
         `b`.
    residues : ndarray, shape(), (1,), or (K,)
        Sums of residues; squared Euclidian norm for each column in
        `b - a x`.
        If the rank of `a` is < N or > M, this is an empty array.
        If `b` is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (K,).
    rank : integer
        Rank of matrix `a`.
    s : ndarray, shape(min(M,N),)
        Singular values of `a`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    Notes
    -----
    If `b` is a matrix, then all array results returned as
    matrices.

    Examples
    --------
    Fit a line, ``y = mx + c``, through some noisy data-points:

    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([-1, 0.2, 0.9, 2.1])

    By examining the coefficients, we see that the line should have a
    gradient of roughly 1 and cuts the y-axis at more-or-less -1.

    We can rewrite the line equation as ``y = Ap``, where ``A = [[x 1]]``
    and ``p = [[m], [c]]``.  Now use `lstsq` to solve for `p`:

    >>> A = np.vstack([x, np.ones(len(x))]).T
    >>> A
    array([[ 0.,  1.],
           [ 1.,  1.],
           [ 2.,  1.],
           [ 3.,  1.]])

    >>> m, c = np.linalg.lstsq(A, y)[0]
    >>> print m, c
    1.0 -0.95

    Plot the data along with the fitted line:

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'o', label='Original data', markersize=10)
    >>> plt.plot(x, m*x + c, 'r', label='Fitted line')
    >>> plt.legend()
    >>> plt.show()

    """
    import math
    a, _ = _makearray(a)
    b, wrap = _makearray(b)
    is_1d = len(b.shape) == 1
    if is_1d:
        b = b[:, newaxis]
    _assertRank2(a, b)
    m  = a.shape[0]
    n  = a.shape[1]
    n_rhs = b.shape[1]
    ldb = max(n, m)
    if m != b.shape[0]:
        raise LinAlgError, 'Incompatible dimensions'
    t, result_t = _commonType(a, b)
    real_t = _linalgRealType(t)
    bstar = zeros((ldb, n_rhs), t)
    bstar[:b.shape[0],:n_rhs] = b.copy()
    a, bstar = _fastCopyAndTranspose(t, a, bstar)
    s = zeros((min(m, n),), real_t)
    nlvl = max( 0, int( math.log( float(min(m, n))/2. ) ) + 1 )
    iwork = zeros((3*min(m, n)*nlvl+11*min(m, n),), fortran_int)
    if isComplexType(t):
        lapack_routine = lapack_lite.zgelsd
        lwork = 1
        rwork = zeros((lwork,), real_t)
        work = zeros((lwork,), t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, -1, rwork, iwork, 0)
        lwork = int(abs(work[0]))
        rwork = zeros((lwork,), real_t)
        a_real = zeros((m, n), real_t)
        bstar_real = zeros((ldb, n_rhs,), real_t)
        results = lapack_lite.dgelsd(m, n, n_rhs, a_real, m,
                                     bstar_real, ldb, s, rcond,
                                     0, rwork, -1, iwork, 0)
        lrwork = int(rwork[0])
        work = zeros((lwork,), t)
        rwork = zeros((lrwork,), real_t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, lwork, rwork, iwork, 0)
    else:
        lapack_routine = lapack_lite.dgelsd
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, -1, iwork, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, lwork, iwork, 0)
    if results['info'] > 0:
        raise LinAlgError, 'SVD did not converge in Linear Least Squares'
    resids = array([], t)
    if is_1d:
        x = array(ravel(bstar)[:n], dtype=result_t, copy=True)
        if results['rank'] == n and m > n:
            resids = array([np.linalg.norm(ravel(bstar)[n:], 2)**2])
    else:
        x = array(transpose(bstar)[:n,:], dtype=result_t, copy=True)
        if results['rank'] == n and m > n:
            resids = array([np.linalg.norm(v[n:], 2)**2 for v in bstar])
    st = s[:min(n, m)].copy().astype(_realType(result_t))
    return wrap(x), wrap(resids), results['rank'], st

