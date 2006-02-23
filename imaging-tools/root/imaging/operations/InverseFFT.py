from FFT import inverse_fft2d
from pylab import Complex32
from imaging.util import shift
from imaging.operations import Operation

##############################################################################
class InverseFFT (Operation):
    "Perform an inverse 2D fft on each slice of each k-space volume."

    #-------------------------------------------------------------------------
    def run(self, image):
        n_pe, n_fe = image.data.shape[-2:]
        for volume in image.data:
            for slice in volume:
                shift(slice, 0 ,n_fe/2) 
                shift(slice, 1, n_pe/2)
                slice[:] = inverse_fft2d(slice).astype(Complex32)

        n_pe, n_fe = image.data.shape[-2:]
        shift(image.data, 0 ,n_fe/2) 
        shift(image.data, 1, n_pe/2)
