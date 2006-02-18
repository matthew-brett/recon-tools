from FFT import inverse_fft
from pylab import angle, conjugate, sin, cos, Complex, Complex32, fft, product, arange, reshape, take, ones, sqrt, exp
from imaging.operations import Operation

class UnbalPhaseCorrection2 (Operation):

    def run(self, image):
        if len(image.ref_vols) < 1:
            self.log("No reference volume, quitting")
            return
        if len(image.ref_vols) > 1:
            self.log("Could be performing Balanced Phase Correction!")
            refVol = image.ref_data[0]
        else:
            refVol = image.ref_data

        (zdim, ydim, xdim) = refVol.shape
        rownum = 0
        for slice in range(zdim):
            for row in range(ydim):
                ref0 = inverse_fft(refVol[slice,row])
                ref1 = row == (ydim-1) and inverse_fft(refVol[slice,0]) or inverse_fft(refVol[slice,row+1])
                ###
                #was: if row%2 == 0:
                if 0 == 0:
                    #correction = sqrt(ref0)*sqrt(conjugate(ref1))/sqrt(abs(ref0))/sqrt(abs(ref1))
                    #ref_phs = angle(ref0*conjugate(ref1))/2
                    #ref_phs = angle(ref0)/2 - angle(ref1)/2
                    ref_phs = angle(sqrt(ref0)*sqrt(conjugate(ref1)))
                else:
                    #correction = sqrt(conjugate(ref0))*sqrt(ref1)/sqrt(abs(ref0))/sqrt(abs(ref1))
                    ref_phs = -angle(sqrt(ref0)*sqrt(conjugate(ref1)))
                ###
                correction = cos(ref_phs) - 1.0j*sin(ref_phs)

                for vol in range(image.data.shape[0]):
                    image.data[vol,slice,row] = fft(inverse_fft(image.data[vol,slice,row])*correction).astype(Complex32)
