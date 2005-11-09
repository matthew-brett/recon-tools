from Numeric import empty
from FFT import inverse_fft
from pylab import zeros, pi, Complex32, arange, reshape, Float32, cos, sin, conjugate, take, fft
from imaging.operations import Operation

##############################################################################
class FixSliceTimeSkew (Operation):
    """
    Correct for variations in slice timing within a frame.  For example, if 6
    slices are acquired in the order [1,3,5,2,4,6] with a TR of 100ms per
    slice, slice 6 will be acquired 500ms after slice 1, i.e., with a
    time-skew of 500ms.

    Ideal (sinc) interpolation is implemented for each voxel by first padding
    the time-course f(i) of length N by the sequence f(N-1),f(N-2),...,f(0)
    such that there are no discontinuities when the padded sequence is thought
    of as wrapping around the unit circle.  The padded sequence is then
    Fourier transformed, phase shifted by an amount  corresponding to the time
    skew, and then inverse transformed to the temporal domain.
    """

    def run(self, options, data):
        nvol = data.nvol; nslice = data.nslice
        tc_rev = nvol - 1 - arange(nvol)
        tc_revm1 = nvol - 2 - arange(nvol-1)
        tc_pad = empty(2*nvol, Complex32)
        phs_shift = empty(2*nvol, Complex32)
        volumes = reshape(data.data_matrix,
            (nvol, nslice, data.n_pe_true*data.n_fe_true))
        volrange = pi*arange(nvol).astype(Float32)/nvol

        for slice in range(nslice):
            theta = (float(slice)/nslice)*volrange
            phs_shift[:nvol] = (cos(theta) + 1.0*sin(theta)).astype(Complex32)
            phs_shift[nvol+1:] = conjugate(take(phs_shift[1:nvol],tc_revm1))
            for vox in range(data.n_pe_true*data.n_fe_true):
                tc_pad[:nvol] = volumes[:,slice,vox]
                tc_pad[nvol:] = take(volumes[:,slice,vox],tc_rev)
                #tc_fft = fft(tc_pad)
                #tc_fft_phs = phs_shift*tc_fft
                #tc = inverse_fft(tc_fft_phs).astype(Complex32)
                tc = phs_shift*tc_pad
                volumes[:,slice,vox] = tc[:nvol]

        data.data_matrix[:] = reshape(volumes,
            (nvol, nslice, data.n_pe_true, data.n_fe_true))

