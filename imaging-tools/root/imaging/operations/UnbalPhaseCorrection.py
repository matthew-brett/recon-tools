from pylab import angle, conjugate, Float, product, arange, take, zeros, \
     diff, cumsum, mean, asarray, putmask, floor, array, pi, plot, show, \
     ravel, where, sqrt, ones, subplot, twinx, Complex32, Complex, title, \
     sum, find, figure, Int, imshow, cm, reshape, resize, median, arctan2, \
     sin, cos, NewAxis
from imaging.operations import Operation
from imaging.util import ifft, apply_phase_correction, shift

EV = 0
ODD = 1
MSK_E = 2
MSK_O = 3
RES = 4

def mod(x,y):
    """ x - y*floor(x/y)
    
        For numeric arrays, x % y has the same sign as x while
        mod(x,y) has the same sign as y.
    """
    return x - y*floor(x*1.0/y)


#scipy's unwrap (pythonication of Matlab's routine)
def unwrap(p,discont=pi,axis=-1):
    """unwraps radian phase p by changing absolute jumps greater than
       discont to their 2*pi complement along the given axis.
    """
    p = asarray(p)
    nd = len(p.shape)
    dd = diff(p,axis=axis)
    slice1 = [slice(None,None)]*nd     # full slices
    slice1[axis] = slice(1,None)
    #ddmod = mod(dd+pi,2*pi)-pi
    ddmod = mod(dd+discont,2*pi)-discont
    #putmask(ddmod,(ddmod==-pi) & (dd > 0),pi)
    putmask(ddmod, (ddmod==-discont) & (dd > 0), discont)
    ph_correct = ddmod - dd;
    putmask(ph_correct,abs(dd)<discont,0)
    up = array(p,copy=1,typecode='d')
    up[slice1] = p[slice1] + cumsum(ph_correct,axis)
    return up

def unwrap2(p):
    if len(p.shape) < 2:
        p = reshape(p, (1, p.shape[0]))
    dd = diff(p)
    dd_wr = arctan2(sin(dd), cos(dd))
    uph = zeros(p.shape, Float)
    uph[:,0] = p[:,0]
    for col in range(dd.shape[-1]):
        uph[:,col+1] = uph[:,col] + dd_wr[:,col]
    return uph

def linReg(X, Y): 

    # solve for (b,m) = (crossing, slope)
    # let sigma = 1
    N = len(X)
    Sx = Sy = Sxx = Sxy = 0.
    for k in range(N):
	Sx += X[k]
	Sy += Y[k]
	Sxx += X[k]**2
	Sxy += X[k]*Y[k]
    
    delta = N*Sxx - Sx**2
    b = (Sxx*Sy - Sx*Sxy)/delta
    m = (N*Sxy - Sx*Sy)/delta
    res = sum((Y-(m*X+b))**2)
    return (b, m, res)
    

def maskedAvg(S, pixSize):
    """takes a slice of angles and returns the average per column
    after masking for noise (based on knowledge of the read gradient
    and the measured sample variance). Also gives the total residual
    of the best linear fit lines, in order to judge the most linear slice.
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
    
    # take st. dev. from "expected value" -- the mean,
    # or the line fitted to the linear region??
    
    Sm = take(S, find(mask>0), axis=1)
    S0 = unwrap(Sm)
    
    b = zeros((nr,),Float)
    m = zeros((nr,),Float)
    res = zeros((nr,),Float)
    midpt = int(S0.shape[1]/2)
    ax = arange(len(S0[0]))+np/2-lin_pix
    for r in range(nr):
        #(b[r], m[r], res[r]) = linReg(arange(10),S0[r,midpt-5:midpt+5])
        (b[r], m[r], res[r]) = linReg(ax, S0[r])
    #c = round(b/2/pi)
    c = floor((b-median(b))/2/pi + 0.5)
    c = c - median(c)
    S0 = S0 - 2*pi*c[:,NewAxis]
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
##     for r in range(nr):
##         plot(ax, S0[r], color[r%7])
##         plot(ax, ax*m[r]+2*pi*c[r], color[r%7]+'--')
##     show()
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

        #calculate phase corrections
        inv_ref = ifft(refVol)
        ref_phs = zeros(refShape, Float)

        # calculate mean phases on even and odd lines
        # also find best line through good region
        p = zeros(refShape, Complex)
        phs_even = zeros((n_slice, n_fe), Float)
        phs_odd = zeros((n_slice, n_fe), Float)
        
        best = (0, 0, 0, 0, 1e6, 0)
        for z in range(n_slice):    
            p[z] = inv_ref[z]*conjugate(take(inv_ref[z], take_order))
           
            phs_even[z], mask_e, res_e = maskedAvg(angle(take(p[z], arange(2, n_pe, 2))), image.xsize)
            phs_odd[z], mask_o, res_o = maskedAvg(angle(take(p[z], arange(1, n_pe-1, 2))), image.xsize)
            
            if (res_e + res_o < best[RES]):
                best = (phs_even[z], phs_odd[z], mask_e, mask_o, res_e+res_o)
            
##         plot(best[EV]*best[MSK_E])
##         plot(best[ODD]*best[MSK_O])

        ev_ax = find(best[MSK_E] != 0)
        (b_even, m_even, res_even) = linReg(ev_ax, take(best[EV], ev_ax))
        od_ax = find(best[MSK_O] != 0)
        (b_odd, m_odd, res_odd) = linReg(od_ax, take(best[ODD], od_ax))

        (aa, bb) = (m_even*arange(n_fe)+b_even, \
                    m_odd*arange(n_fe)+b_odd)
##         plot(aa, 'bo')
##         plot(bb, 'go')
##         show()

        ### find beta, beta_pr with the following relationships: ###
        # beta_pr - 2*beta = a                                     #
        # beta_pr + 2*beta = b                                     #
        # --> beta = (b - a)/4                                     #
        # --> beta_pr = (b + a)/2                                  # 
        ###                                                      ###
        beta = (bb - aa)/4.
        beta_pr = (bb + aa)/2.

 ##        plot(beta_pr)
##         plot(beta)
##         show()
##         for r in range(0, n_pe, 2):
##             plot((r-32)*beta_pr + beta)
##         show()
##         for r in range(1, n_pe, 2):
##             plot((r-32)*beta_pr - beta)
##         show()
        
        # apply the odd/even correction at odd/even lines
        for t in range(image.tdim):
            for z in range(image.zdim):
                for m in range(0, n_pe, 2):
                    image.data[t,z,m,:] = apply_phase_correction(image.data[t,z,m], ((m-32)*beta_pr + beta))
                for m in range(1, n_pe, 2):
                    image.data[t,z,m,:] = apply_phase_correction(image.data[t,z,m], ((m-32)*beta_pr - beta))

