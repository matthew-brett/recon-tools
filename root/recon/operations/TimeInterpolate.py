import numpy as N

from recon.operations import Operation, ChannelIndependentOperation

##############################################################################
class TimeInterpolate (Operation):

    #-------------------------------------------------------------------------
    @ChannelIndependentOperation
    def run(self, image):
        if image.nseg != 2:
            self.log("Operation only works on 2-shot data")
            return
        if image.n_vol < 2:
            self.log("Operation can't interpolate from a single volume")

        nv, ns, n_pe, n_fe = image.shape
        pe_per_seg = n_pe/image.nseg
        new_vols = N.empty((2*nv, ns ,n_pe,n_fe), N.complex64)

        seg1 = [slice(0,d) for d in image.shape[-3:]]
        seg1[-2] = image.seg_slicing(0)
        seg2 = [slice(0,d) for d in image.shape[-3:]]
        seg2[-2] = image.seg_slicing(1)
        # handle cases for first and last two volumes
        new_vols[0] = image[0]
        new_vols[1][seg1] = (image[0][seg1] + image[1][seg1])/2.0
        new_vols[1][seg2] = image[0][seg2]
        new_vols[-2][seg1] = image[-1][seg1]
        new_vols[-2][seg2] = (image[-1][seg2] + image[-2][seg2])/2.0
        new_vols[-1] = image[-1]
        # create volumes with an acquired segment and a mixture of two
        # surrounding segments
        for v in range(1,image.n_vol-1):
            new_vols[2*v][seg1] = image[v][seg1]
            new_vols[2*v][seg2] = (image[v-1][seg2] + image[v][seg2])/2.0
            
            new_vols[2*v+1][seg1] = (image[v][seg1] + image[v+1][seg1])/2.0
            new_vols[2*v+1][seg2] = image[v][seg2]
            
        image.setData(new_vols)
