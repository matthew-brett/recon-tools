"""
This module finds the plane-rotation necessary to put the image into one
of the six ANALYZE-defined orientations. Alternatively, the target recon_epi
can be used, which always makes the same rotation and matches the output of
the old recon_epi tool.

If the image is in a different plane than the orientation target, a warning
will arise and the image will not change.
"""

import numpy as N
from recon.operations import Operation, Parameter
from recon.imageio import ReconImage
from recon.analyze import xforms, canonical_orient
from recon import util

##############################################################################
class RotPlane (Operation):
    """
    The orientations in the operation are taken from ANALYZE orient codes
    and are left-handed. However, if the final image is to be NIFTI type,
    the rotation transform is updated (in the right-handed system).
    """

    params = (Parameter(name="orient_target", type="str", default=None,
                        description=
    """
    Final orientation of the image, taken from ANALYZE orient codes.
    Can be: radiological, transverse, coronal, coronal_flipped, saggital, and
    saggital_flipped. Also may be recon_epi."""),)

    def run(self, image):

        if (self.orient_target not in xforms.keys() + ["recon_epi",]):
            self.log("no xform available for %s"%self.orient_target)
            return
        if self.orient_target == "recon_epi":
            # always swap -x to +y, and -y to +x
            Ts = N.array([[ 0.,-1., 0.],
                          [-1., 0., 0.],
                          [ 0., 0., 1.],])
            dest_xform = N.dot(image.orientation_xform.tomatrix(),
                               N.linalg.inv(Ts))
        else:
            dest_xform = xforms.get(self.orient_target, None)
            Tr = image.orientation_xform.tomatrix()
            # Tr already maps [x,y,z]^T into [R,A,S] ...
            # Here I want to change coordinates with this relationship:
            # Tr*[x',y',z']^T = dest_xform*[x,y,z]^T
            # So inv(dest_xform)*Tr*[x',y',z'] = [x,y,z] = orientation of choice!
            Ts = N.dot(N.linalg.inv(dest_xform), Tr)
            
        if not Ts[-1,-1]:
            raise ValueError("This operation only makes in-plane rotations. "\
                             "EG you cannot use the saggital transform for "\
                             "an image in the coronal plane.")
        qform = util.Quaternion(M=dest_xform)
        image.orientation_xform = qform
        rot = compose_xform(Ts)
        ### do transform + book-keeping
        temp = rot(image[:])
        origin = N.array([image.x0, image.y0, image.z0])
        (image.x0, image.y0, image.z0) = tuple(N.dot(Ts, origin))
        # now want to reorder array scales and shapes, not change sign:
        Ts = abs(Ts.astype(N.int32))
        shape = N.array([image.idim, image.jdim, image.kdim])
        delta = N.array([image.isize, image.jsize, image.ksize])
        (image.isize, image.jsize, image.ksize) = tuple(N.dot(Ts, delta))
        new_cshape = (image.tdim and (image.tdim,) or ()) + \
                     tuple(util.reverse(N.dot(Ts,shape)))
        image.resize(new_cshape)
        image[:] = temp.copy()
        # name the orientation, for ANALYZE users
        image.orientation = canonical_orient(qform.tomatrix())

#-----------------------------------------------------------------------------
def compose_xform(mat):
    xform = lambda x: x
    if not mat[0,0]:
        xform = lambda x, g=xform: N.swapaxes(g(x), -1, -2)
    if (mat[1,0] < 0 or mat[1,1] < 0):
        # need to flip +y -> -y
        xform = lambda x, g=xform: reverse_y(g(x))
    if (mat[0,0] < 0 or mat[0,1] < 0):
        # need to flip +x -> -x
        xform = lambda x, g=xform: reverse_x(g(x))
    if mat[-1,-1] < 0:
        xform = lambda x, g=xform: reverse_z(g(x))
    return xform

#-----------------------------------------------------------------------------
def reverse_x(M):
    slices = [slice(0,d) for d in M.shape]
    slices[-1] = slice(M.shape[-1], None, -1)
    return M[slices]

#-----------------------------------------------------------------------------
def reverse_y(M):
    slices = [slice(0,d) for d in M.shape]
    slices[-2] = slice(M.shape[-2], None, -1)
    return M[slices]

#-----------------------------------------------------------------------------
def reverse_z(M):
    slices = [slice(0,d) for d in M.shape]
    slices[-3] = slice(M.shape[-3], None, -1)
    return M[slices]
