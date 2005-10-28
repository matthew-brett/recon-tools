from FFT import inverse_fft2d
from pylab import Complex32
from recon.util import shift
from recon.operations import Operation

##############################################################################
class InverseFFT (Operation):
    "Perform an inverse 2D fft on each slice of each k-space volume."

    #-------------------------------------------------------------------------
    def run(self, params, options, data):
        n_pe, n_fe = data.data_matrix.shape[-2:]
        for volume in data.data_matrix:
            for slice in volume:
                image = inverse_fft2d(slice)
                shift(image,0,n_fe/2) 
                shift(image,1,n_pe/2)  
                slice[:,:] = image.astype(Complex32)


