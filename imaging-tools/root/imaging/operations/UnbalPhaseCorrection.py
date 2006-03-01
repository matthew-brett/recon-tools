#from FFT import inverse_fft
from pylab import angle, conjugate, sin, cos, Complex32, Float32, product, arange, reshape, take, ones, pi, zeros, \
     diff, find, cumsum
from imaging.operations import Operation
from imaging.util import shift, fft, ifft, apply_phase_correction



def unwrap_phase(phase):
#***********************************

# Purpose: Unwrap phase in a single line of data.

#   Unwrap phase values
    len = phase.shape[-1]
    if len < 2:
        return 0.
    phase_unwrapped = zeros(len).astype(Float32)
    pm1 = phase[0]
    wraps = 0.
    phase_unwrapped[0] = phase[0]
    sum = 0.
    for i in range(1,len):
        slope = phase[i] - phase[i-1]
        if abs(slope) > pi:
#           Must be a wrap.
            if slope < 0:
                wraps = wraps + 2*pi
            else:
                wraps = wraps - 2*pi
        phase_unwrapped[i] = phase[i] + wraps
#        if verbose:
#            print "%d %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f" % (i,phase[i],phase[i-1],slope,slopem1,wraps, phase_unwrapped[i])
        slopem1 = slope
    return(phase_unwrapped.astype(Float32))


## #Matlab's unwrap
## def unwrap(line, cutoff=4*pi/5):
##     m = line.shape[-1]
##     dp = diff(line)
##     dps = (dp+pi)%(2*pi) - pi
##     for n in find((dps==-pi) & (dp>0)):
##         dps[n] = pi
##     dp_corr = dps - dp
##     for n in find(abs(dp) < cutoff):
##         dp_corr[n] = 0
##     line[1:] = line[1:] + cumsum(dp_corr)
##     return line




class UnbalPhaseCorrection (Operation):

    def run(self, image):
        if not image.ref_data:
            self.log("No reference volume, quitting")
            return
        if len(image.ref_vols) > 1:
            self.log("Could be performing Balanced Phase Correction!")

        refVol = image.ref_data[0]
        origShape = refVol.shape
        #get total number of rows in data set (across slices)    
        nrows = product(origShape)/origShape[-1]
        #this will help shorten the code
        refVol = reshape(refVol, (nrows, origShape[-1]))
        take_order = arange(nrows) + 1
        take_order[-1] = 0

        #calculate phase corrections
        inv_ref = ifft(refVol)
        #neg = ones((nrows, 1)) - 2*reshape((arange(nrows))%2, (nrows, 1))
        ref_phs = zeros((nrows,origShape[-1]), Float32)
        for row in range(nrows):
            sign = (1-2*(row%2))
        #    sign = 1
            if row==nrows-1:
                ref_phs[row] = (sign*unwrap_phase(angle(inv_ref[row] * conjugate(inv_ref[0])))/2).astype(Float32)
            else:
                ref_phs[row] = (sign*unwrap_phase(angle(inv_ref[row] * conjugate(inv_ref[row+1])))/2).astype(Float32)
        #ref_phs = angle(inv_ref * conjugate(take(inv_ref, take_order)))/2
        correction = reshape(cos(ref_phs) - 1.0j*sin(ref_phs), origShape)
        #apply corrections
        image.data[:] = fft(ifft(image.data)*correction).astype(Complex32)

