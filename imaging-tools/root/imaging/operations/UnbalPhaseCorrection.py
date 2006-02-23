from FFT import inverse_fft
from pylab import angle, conjugate, sin, cos, Complex32, fft, product, arange, reshape, take
from imaging.operations import Operation

class UnbalPhaseCorrection (Operation):

    def run(self, image):
        if len(image.ref_vols) < 1:
            self.log("No reference volume, quitting")
            return
        if len(image.ref_vols) > 1:
            self.log("Could be performing Balanced Phase Correction!")
            refVol = image.ref_data[0]
        else:
            refVol = image.ref_data

        origShape = refVol.shape
        #get total number of rows in data set (across slices)    
        nrows = product(origShape)/origShape[-1]
        #this will help shorten the code
        refVol = reshape(refVol, (nrows, origShape[-1]))
        take_order = arange(nrows) + 1
        take_order[-1] = 0

        #calculate phase corrections
        inv_ref = inverse_fft(refVol)
        ref_phs = angle(inv_ref * conjugate(take(inv_ref, take_order)))/2
        correction = reshape(cos(ref_phs) - 1.0j*sin(ref_phs), origShape)
        #apply corrections
        image.data[:] = fft(inverse_fft(image.data)*correction).astype(Complex32)

        #set ref data back to previous shape
        refVol = reshape(refVol, origShape)
