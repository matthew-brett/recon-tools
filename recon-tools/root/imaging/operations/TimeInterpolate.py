from FFT import inverse_fft
from Numeric import empty
from pylab import reshape, Complex32
from imaging.operations import Operation

##############################################################################
class TimeInterpolate (Operation):

    #-------------------------------------------------------------------------
    def run(self, options, data):
        if data.nseg != 2:
            print "TimeInterpolate:  non-segmented data, nothing to do."
            return
        pe_per_seg = data.n_pe_true/data.nseg
        old_vols = reshape(data.data_matrix,
          (data.nvol, data.nseg, data.nslice*pe_per_seg*data.n_fe_true))
        new_vols = empty(
          (2*data.nvol, data.nseg, data.nslice*pe_per_seg*data.n_fe_true),
          Complex32)

        # degenerate case for first and last volumes
        new_vols[0] = old_vols[0]
        new_vols[-1] = old_vols[-1]

        # interpolate interior volumes
        for oldvol in range(1,data.nvol):
            newvol = 2*oldvol

            new_vols[newvol-1,0] = \
              ((old_vols[oldvol-1,0] + old_vols[oldvol,0])/2.).astype(Complex32)
            new_vols[newvol-1,1] = old_vols[oldvol-1,1]

            new_vols[newvol,0] = old_vols[oldvol,0]
            new_vols[newvol,1] = \
              ((old_vols[oldvol-1,1] + old_vols[oldvol,1])/2.).astype(Complex32)

        data.data_matrix = reshape(new_vols,
          (2*data.nvol, data.nslice, data.n_pe_true, data.n_fe_true))
