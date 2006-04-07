from Numeric import empty
from pylab import angle, conjugate, Float, arange, take, zeros, mean, floor, \
     pi, sqrt, ones, sum, find, Int, median, NewAxis, resize, matrixmultiply
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
    m = empty((nr,),Float)
    res = empty((nr,),Float)
    ax = arange(len(S0[0]))+np/2-lin_pix

    # find the best linear fit for all lines, use this to:
    # a) get a sense (from sum(res)) of how good the data is
    # b) get all the y-intercepts in order to correct false n2pi jumps
    for r in range(nr):
        (b[r], m[r], res[r]) = linReg(ax, S0[r])

    # find which 2pi bracket the line falls into based on intercept
    c = floor((b-median(b))/2/pi + 0.5)
    # find offset from supposed "normal" intercept (where most happen to be)
    c = c - median(c)
    # shift the entire line by the appropriate number of 2*pi
    S0 = S0 - 2*pi*c[:,NewAxis]
    #color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    E = mean(S0)
    std = sqrt(sum((S0 - E)**2)/nr)

    if len(mask) > len(std):
        std = resize(std, (len(mask),))
        shift(std, 0, np/2-lin_pix)
        std = mask*std
        
    if len(mask) > len(E):
        M[np/2-lin_pix:np/2+lin_pix] = E
    else:
        M = E    
    return M, mask, sum(res)

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
        for z in range(n_slice):    
            print "analyzing slice %d"%(z)
            p = inv_ref[z]*conjugate(take(inv_ref[z], take_order))
           
            phs_even, mask_e, res_e = \
                   maskedAvg(angle(take(p, arange(2, n_pe, 2))), image.xsize)
            phs_odd, mask_o, res_o = \
                   maskedAvg(angle(take(p, arange(1, n_pe-1, 2))), image.xsize)
            
            if (res_e + res_o < best[RES]):
                best = (phs_even, phs_odd, mask_e, mask_o, res_e+res_o)
            
        ev_ax = find(best[MSK_E] != 0)
        (b_even, m_even, res_even) = linReg(ev_ax, take(best[EV], ev_ax))
        od_ax = find(best[MSK_O] != 0)
        (b_odd, m_odd, res_odd) = linReg(od_ax, take(best[ODD], od_ax))

        (a, b) = (m_even*arange(n_fe)+b_even, \
                    m_odd*arange(n_fe)+b_odd)

        ### find beta, beta_pr with the following relationships: ###
        # beta_pr - 2*beta = a                                     #
        # beta_pr + 2*beta = b                                     #
        # --> beta = (b - a)/4                                     #
        # --> beta_pr = (b + a)/2                                  # 
        ###                                                      ###
        beta = (b - a)/4.
        beta_pr = (b + a)/2.

        B = empty((2,n_pe), Float)
        B[0], B[1] = beta_pr, beta
        T = empty((n_pe, 2))
        T[:,0] = arange(n_pe)-32
        T[:,1] = 1 - 2*mod(T[:,0], 2).astype(Int)
        Correction = matrixmultiply(T,B)

        # this is the "small array" version
        for vol in image.data:
            for slice in vol:
                slice[:] = apply_phase_correction(slice, Correction)

##         # this is the "large array" version        
##         image.data[:] = apply_phase_correction(image.data, Correction)

