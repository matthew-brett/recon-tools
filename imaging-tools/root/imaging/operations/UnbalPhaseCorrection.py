from pylab import angle, conjugate, Float, product, arange, take, zeros, \
     diff, cumsum, mean, asarray, putmask, floor, array, pi 
from imaging.operations import Operation
from imaging.util import ifft, apply_phase_correction


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
            ref_phs[slice] = angle(inv_ref[slice] * \
                                   conjugate(take(inv_ref[slice], take_order)))

        # calculate mean phases on even and odd lines
        # also find best line through good region
        phs_even = zeros((n_slice, n_fe), Float)
        phs_odd = zeros((n_slice, n_fe), Float)
        fk_even = zeros((n_slice, n_fe), Float)
        fk_odd = zeros((n_slice, n_fe), Float)
        for z in range(n_slice):    
            phs_even[z] = unwrap(mean(take(ref_phs[z], \
                                           arange(0, n_pe, 2))), discont=pi)/2
            phs_odd[z] = unwrap(mean(take(ref_phs[z], \
                                          arange(1, n_pe, 2))), discont=pi)/2
	    
	    # let's say the 15 pts [20,35) are good
            b_even, m_even = linReg(arange(15)+20, phs_even[z,20:35])
            b_odd, m_odd = linReg(arange(15)+20, phs_odd[z,20:35])
	    # replace phs arrays with fake lines
            phs_even[z,:] = (arange(n_fe)*m_even + b_even)
            phs_odd[z,:] = (arange(n_fe)*m_odd + b_odd)

        # apply the odd/even correction at odd/even lines
        for t in range(image.tdim):
            for z in range(image.zdim):
                for m in range(0, n_pe, 2):
                    image.data[t,z,m,:] = apply_phase_correction(image.data[t,z,m], -phs_even[z])
                for m in range(1, n_pe, 2):
                    image.data[t,z,m,:] = apply_phase_correction(image.data[t,z,m], -phs_odd[z])

