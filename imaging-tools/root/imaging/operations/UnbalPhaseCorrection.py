from Numeric import empty
from pylab import angle, conjugate, Float, arange, take, zeros, mean, floor, \
     pi, sqrt, ones, sum, find, Int, median, NewAxis, resize, matrixmultiply, bar, show, svd, sign, transpose, diag, putmask, plot, show
from imaging.operations import Operation
from imaging.util import ifft, apply_phase_correction, mod, unwrap1D, linReg, \
     shift

# some index names
EV = 0
ODD = 1
MSK_E = 2
MSK_O = 3
RES = 4

def maskedAvg(S, pixSize):
    """Takes a slice of angles
    The phase lines are unwrapped in the read-out direction, and then an
    attempt is made to correct for false offsets of n2pi brought about by
    the noise sensitivity in the unwrap algorithm. (This makes the mean
    and the standard deviation more representative of the real data)

    Returns:
    a) the average phase line along the rows
    b) an array that masks noisy regions (based on knowledge of the
       read gradient and the measured sample variance).
    c) the total residual from the best linear fit lines
    """
    # cut out values outside of known good range (ONLY KNOW ONE FOV?)
    lin_radius = 75.  #mm
    lin_pix = int(round(lin_radius/pixSize))
    nr,np = S.shape
    M = zeros((np,), Float)
    mask = ones((np,), Int)
    
    if lin_pix <= np/2:
        mask[0:(np/2-lin_pix)] = 0
        mask[(np/2+lin_pix):] = 0

    # work only with the unmasked region
    Sm = take(S, find(mask>0), axis=1)
    S0 = unwrap1D(Sm)
    
    b = empty((nr,),Float)
    b2 = empty((nr,),Float)
    m = empty((nr,),Float)
    res = empty((nr,),Float)
    ax = arange(len(S0[0]))+np/2-lin_pix

    # find the best linear fit for all lines, use this to:
    # a) get a sense (from sum(res)) of how good the data is
    # b) get all the y-intercepts in order to correct false n2pi jumps
    for r in range(nr):
        b[r] = S0[r,len(ax)/2]
        #(b[r], _, _) = linReg(arange(10), S0[r,np/2-5-ax[0]:np/2+5-ax[0]])
        (b2[r], m[r], res[r]) = linReg(arange(len(S0[0])), S0[r])

#### old method
##     # find which 2pi bracket the line falls into based on intercept
##     #c = floor((b-median(b))/2/pi + 0.5)
##     # find offset from supposed "normal" intercept (where most happen to be)
##     # this is failing in some sets where c is evenly distributed between two points:
##     # need to fix, perhaps by keeping track of the running distribution over slice or volume
##     #c = c - median(c)
####
    # shift the entire line by the appropriate number of 2*pi
    c = ((b+sign(b)*pi)/2/pi).astype(Int)
    S0 = S0 - 2*pi*c[:,NewAxis]
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    E = mean(S0)
    std = sqrt(sum((S0 - E)**2)/nr)

    if len(mask) > len(std):
        std = resize(std, (len(mask),))
        shift(std, 0, np/2-lin_pix)
        std = mask*std
#    from pylab import subplot, twinx
    putmask(mask, std>1, 0)

## #    plot(std)
## #    show()
    if len(mask) > len(E):
        M[np/2-lin_pix:np/2+lin_pix] = E
    else:
        M = E
    #ax1 = subplot(111)
##     for r in range(nr):
##         plot(ax, S0[r], color[r%7])
##         plot(ax, (arange(len(ax)))*m[r] + b2[r], color[r%7]+'--')
##     plot(ax, E, 'go')
##     #ax2 = twinx()
##     #plot(ax, ones(len(ax))*sum(res), 'bo')
##     #plot(std, 'bo')
##     #ax2.yaxis.tick_right()
##     show()
    return M, mask, sum(res)

def solve_phase(ev, odd, q_mask, z_mask):
    """let V = (beta_pr, beta, alpha_pr, alpha, c_pr, c)^T,
    we want to solve:
    phi(q,s,0) = q*(beta_pr + beta) + s*(alpha_pr + alpha) + c
    phi(q,s,1) = q*(beta_pr - beta) + s*(alpha_pr - alpha) + c
    with our overdetermined data.
    so P = [ev[0] odd[0] ev[1] odd[1] ... ev[Z] odd[Z]]^T
    A = [ ... ]
    and solve AV = P
    """
    BPR, B, ALPR, AL, EPR, E = (0,1,2,3,4,5)
    n_chunks = sum(z_mask)
    rows_in_chunk = sum(take(q_mask, find(z_mask)), axis=1)
    z_ind = find(z_mask)
    A = empty((sum(rows_in_chunk)*2, 6), Float)
    V = empty((sum(rows_in_chunk)*2, 1), Float)
    row_start, row_end = 0, 0
    for c in range(n_chunks):
        for s in [0, 1]:
            row_start = row_end
            row_end = row_start + rows_in_chunk[c]
            q_ind = find(q_mask[z_ind[c]])
            V[row_start:row_end,0] = s and take(odd[z_ind[c]], q_ind) or take(ev[z_ind[c]], q_ind)
            A[row_start:row_end,BPR] = q_ind
            A[row_start:row_end,B] = s and 2*q_ind or -2*q_ind
            A[row_start:row_end,ALPR] = z_ind[c]
            A[row_start:row_end,AL] = s and 2*z_ind[c] or - 2*z_ind[c]
            A[row_start:row_end,EPR] = 1
            A[row_start:row_end,E] = s and 2 or -2

##     f = open('matrix.txt', 'w')
##     for r in range(A.shape[0]):
##         f.write(str(A[r])+'\t'+str(V[r])+'\n')
##     f.close()
    [u,s,vt] = svd(A)
    P = matrixmultiply(transpose(vt), matrixmultiply(diag(1/s), \
                                      matrixmultiply(transpose(u), V)))

    return tuple(P)    

class UnbalPhaseCorrection (Operation):
    """This operation corrects Nyquist ghosting by finding "the best"
    pair of linear lines fitted to even and odd phase lines across all
    slices of a single reference scan. These two fitted lines are then
    used to correct ghosting in all slices/volumes of the image."""
    
    def run(self, image):
        if not image.ref_data:
            self.log("No reference volume, quitting")
            return
        if len(image.ref_vols) > 1:
            self.log("Could be performing Balanced Phase Correction!")

        refVol = image.ref_data[0]
        n_slice, n_pe, n_fe = refShape = refVol.shape
        take_order = arange(n_pe) + 1
        take_order[-1] = 0
        inv_ref = ifft(refVol)
        best = (0, 0, 0, 0, 1e6, 0)
        res = zeros((n_slice,), Float)
        z_mask, q_mask = zeros(n_slice), zeros((n_slice,n_fe))
        phs_even = empty((n_slice, n_fe), Float)
        phs_odd = empty((n_slice, n_fe), Float)
        for z in range(n_slice):    
            p = inv_ref[z]*conjugate(take(inv_ref[z], take_order))
           
            phs_even[z], mask_e, res_e = \
                   maskedAvg(angle(take(p, arange(2, n_pe, 2))), image.xsize)
            phs_odd[z], mask_o, res_o = \
                   maskedAvg(angle(take(p, arange(1, n_pe-1, 2))), image.xsize)
            
            #z_mask[z] = res_e+res_o < 2500.
            z_mask[z] = res_e+res_o < 2000
            res[z] = res_e+res_o
            q_mask[z] = mask_e*mask_o

            
        bar(arange(n_slice), res)
        show()
        
        print "solving for %d slices"%(sum(z_mask))
        #(beta_pr, beta, alpha_pr, alpha, e) = \
        #(beta_pr, beta, e_pr, e) = \
        (beta_pr, beta, alpha_pr, alpha, e_pr, e) = \
                  solve_phase(phs_even, phs_odd, q_mask, z_mask)

        print "computed coefficients:"
        print "\tBpr: %f, B: %f, Apr: %f, A: %f, e_pr: %f, e: %f"%(beta_pr, beta, alpha_pr, alpha, e_pr, e)
        from pylab import title
        for z in range(n_slice):
            plot(find(q_mask[z]), take(phs_even[z], find(q_mask[z])))
            plot(find(q_mask[z]), take(phs_odd[z], find(q_mask[z])))
            plot((arange(n_fe))*(beta_pr - 2*beta) + z*(alpha_pr - 2*alpha) + e_pr - 2*e, 'b--')
            plot((arange(n_fe))*(beta_pr + 2*beta) + z*(alpha_pr + 2*alpha) + e_pr + 2*e, 'g--')
            tstr = "slice %d"%(z)
            if z_mask[z]: tstr += " (selected)"
            title(tstr)
            show()
        for t in range(image.data.shape[0]):
            for z in range(n_slice):
                for r in range(0, n_pe, 2):
                    #ev_correction = (r)*(arange(n_fe)*beta_pr-z*alpha_pr) - (arange(n_fe)*beta + z*alpha) - (e_pr - e)

                    #ev_correction = (r-32)*(arange(n_fe)*beta_pr + e_pr) + (arange(n_fe)*beta + e) #good, no zdep
                    #ev_correction = (r)*(arange(n_fe)*beta_pr) + (arange(n_fe)*beta + e)
                    ev_correction = (r-32)*(arange(n_fe)*beta_pr + z*alpha_pr + e_pr) + (arange(n_fe)*beta + z*alpha + e)
                    #plot(ev_correction)
                    image.data[t,z,r,:] = apply_phase_correction(image.data[t,z,r], ev_correction)
                for r in range(1, n_pe, 2):
                    #odd_correction = (r)*(arange(n_fe)*beta_pr-z*alpha_pr) + (arange(n_fe)*beta + z*alpha) - (e_pr + e)
                    #odd_correction = (r-32)*(arange(n_fe)*beta_pr + e_pr) - (arange(n_fe)*beta + e) #good, no zdep
                    #odd_correction = (r)*(arange(n_fe)*beta_pr) - (arange(n_fe)*beta + e)
                    odd_correction = (r-32)*(arange(n_fe)*beta_pr + z*alpha_pr + e_pr) - (arange(n_fe)*beta + z*alpha + e) #good
                    #plot(odd_correction)
                    image.data[t,z,r,:] = apply_phase_correction(image.data[t,z,r], odd_correction)
