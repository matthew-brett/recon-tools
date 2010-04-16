from recon.operations import Operation, ChannelAwareOperation
import numpy as np

class ReadoutWindow (Operation):

    @ChannelAwareOperation
    def run(self, image):
        Q3,Q2,Q1 = image.shape[-3:]
        ksize, jsize, isize = image.ksize, image.jsize, image.isize
        if Q1*isize == Q2*isize:
            print __class__, ": Field of view is already balanced"
            return
        q1_chop = (Q1*isize - Q2*jsize)/isize
        sl0_vox_center = np.array([image.idim/2, image.jdim/2, 0])
        vox_sizes = np.array([image.isize, image.jsize, image.ksize])
        M = image.orientation_xform.tomatrix()*vox_sizes
        r0 = np.array([image.x0, image.y0, image.z0])
        sl0_center = np.dot(M, sl0_vox_center) + r0
        new_slicing = (slice(None), slice(None),
                       slice(int(Q1/2 - q1_chop/2), int(Q1/2 + q1_chop/2)))
        if image.tdim:
            new_slicing = (slice(None),) + new_slicing
        
        image.setData(image[new_slicing].copy())
        # now update the r0 vector such that M*vox_center + r0 = sl0_center
        sl0_vox_center = np.array([image.idim/2, image.jdim/2, 0])
        r0 = sl0_center - np.dot(M, sl0_vox_center)
        image.x0, image.y0, image.z0 = r0
