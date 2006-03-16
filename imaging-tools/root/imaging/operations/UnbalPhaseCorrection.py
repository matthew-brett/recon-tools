#from FFT import inverse_fft
from pylab import angle, conjugate, sin, cos, Complex32, Float, Float32, product, arange, reshape, take, ones, pi, zeros, \
     diff, find, cumsum, mean, asarray, putmask, floor, array, plot, show
from imaging.operations import Operation
from imaging.util import shift, fft, ifft, apply_phase_correction



## def unwrap_phase(phase):
## #***********************************

## # Purpose: Unwrap phase in a single line of data.

## #   Unwrap phase values
##     len = phase.shape[-1]
##     if len < 2:
##         return 0.
##     phase_unwrapped = zeros(len).astype(Float32)
##     pm1 = phase[0]
##     wraps = 0.
##     phase_unwrapped[0] = phase[0]
##     sum = 0.
##     for i in range(1,len):
##         slope = phase[i] - phase[i-1]
##         if abs(slope) > pi:
## #           Must be a wrap.
##             if slope < 0:
##                 wraps = wraps + 2*pi
##             else:
##                 wraps = wraps - 2*pi
##         phase_unwrapped[i] = phase[i] + wraps
##         slopem1 = slope
##     return(phase_unwrapped.astype(Float32))


## #Matlab's unwrap, row-wise instead of column-wise
## def unwrap(phase, cutoff=170.*pi/180.):
##     #M,N = len(phase.shape) > 1 and phase.shape[-2:] or (1, phase.shape[-1])
##     dp = diff(phase)
##     dps = (dp+pi)%(2*pi) - pi
##     #for m in range(M):
##     for n in find((dps==-pi) & (dp>0)):
##         dps[n] = pi
##     dp_corr = dps - dp
##     #for m in range(M):
##     for n in find(abs(dp) < cutoff):
##         dp_corr[n] = 0
##     phase[1:] = phase[1:] + cumsum(dp_corr)
##     return phase


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
    ddmod = mod(dd+pi,2*pi)-pi
    putmask(ddmod,(ddmod==-pi) & (dd > 0),pi)
    ph_correct = ddmod - dd;
    putmask(ph_correct,abs(dd)<discont,0)
    up = array(p,copy=1,typecode='d')
    up[slice1] = p[slice1] + cumsum(ph_correct,axis)
    return up

def dumbThresh(matrix):
    p = matrix.copy()
    for row in p:
        putmask(row,abs(row) < .1*mean(abs(row)), 0j)
    return p

def linReg(X, Y):

    # solve for (b,m) = (crossing, slope)
    # let rho = 1
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
    return (b, m)
    


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
        for slice in range(n_slice):
            ref_phs[slice] = angle(inv_ref[slice]*conjugate(take(inv_ref[slice], take_order)))

        # calculate mean phases on even and odd lines
        # also find best line through good region
        phs_even = zeros((n_slice, n_fe), Float)
        phs_odd = zeros((n_slice, n_fe), Float)
        fk_even = zeros((n_slice, n_fe), Float)
        fk_odd = zeros((n_slice, n_fe), Float)
        for z in range(n_slice):    
            phs_even[z] = unwrap(mean(take(ref_phs[z], arange(0, n_pe, 2))), discont=pi)/2
            phs_odd[z] = unwrap(mean(take(ref_phs[z], arange(1, n_pe, 2))), discont=pi)/2
##             phs_even[z] = mean(take(ref_phs[z], arange(0, n_pe, 2)))/2
##             phs_odd[z] = mean(take(ref_phs[z], arange(1, n_pe, 2)))/2
	    
	    # let's say the 15 pts [20,35) are good
            b_even, m_even = linReg(arange(15)+20, phs_even[z,20:35])
            b_odd, m_odd = linReg(arange(15)+20, phs_odd[z,20:35])
	    # replace phs arrays with fake lines
            fk_even[z] = (arange(n_fe)*m_even + b_even)
            fk_odd[z] = (arange(n_fe)*m_odd + b_odd)
##             phs_even[z,30:] = fk_even[z,30:] 
##             phs_odd[z,30:] = fk_odd[z,30:]
            phs_even[z,:] = fk_even[z]
            phs_odd[z,:] = fk_odd[z]

        # apply the odd/even correction at odd/even lines
        for t in range(image.tdim):
            for z in range(image.zdim):
                plot(phs_even[z])
                plot(phs_odd[z])
#                plot(fk_even[z])
#                plot(fk_odd[z])
                for m in range(0, n_pe, 2):
                    image.data[t,z,m,:] = apply_phase_correction(image.data[t,z,m], -phs_even[z])
                for m in range(1, n_pe, 2):
                    image.data[t,z,m,:] = apply_phase_correction(image.data[t,z,m], -phs_odd[z])


        #correction = reshape(cos(ref_phs) - 1.0j*sin(ref_phs), origShape)
        #apply corrections
        #image.data[:] = fft(ifft(image.data)*correction).astype(Complex32)

