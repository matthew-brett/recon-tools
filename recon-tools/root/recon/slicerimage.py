from recon.imageio import ReconImage
from recon.nifti import NiftiImage
from recon.analyze import AnalyzeImage
from recon.util import reverse
import numpy as N

##############################################################################
class SlicerImage (ReconImage):
    """
    A wrapper of any ReconImage that provides slicing access and vox<->zyx
    transformations, and other helpful stuff
    """

    def __init__(self, image=None):
        if image is None:
            image = ReconImage(N.ones((4,4,4)), 1, 1, 1, 1)
        if not isinstance(image, ReconImage):
            image = ReconImage(N.asarray(image), 1, 1, 1, 1)
        for item in image.__dict__.items():
            self.__dict__[item[0]] = item[1]

        # get vox origin from zyx origin (can't use transform because
        # (these zyx are in the image's native space)
        self.r0 = N.array([self.z0, self.y0, self.x0])
        self.dr = N.array([self.ksize, self.jsize, self.isize])
        # orientation xform maps vx-space:xyz-space (Right,Anterior,Superior)
        # numpy arrays are in C-ordering, so I'm going to reverse the
        # rows and columns of the xform
        # Also, these xforms are right-handed, but ANALYZE parameter data
        # are left handed, so I'm going to reverse the sign of the row
        # mapping to x
        # (makes as if the images are right handed??)
        self.xform = self.orientation_xform.tomatrix()
        self.xform = reverse(reverse(self.xform,axis=-1),axis=-2)
        #self.xform[-1] = abs(self.xform[-1])
        # These variables describe which dimension a given standard slicing
        # (axial, saggital, coronal) slices in the data array
        self.ax, self.cor, self.sag = abs(N.dot(self.xform, N.arange(3)).astype(N.int32)).tolist()
        # IF the data is Analyze style, make a conversion of the r0 offset
        # to allow for a universal transformation formula.
        # keep this internal until later.. this is the r0 in zyx space

##         if not isinstance(image, NiftiImage):
##             # the NIFTI (r1,r2,r3) -> (x,y,z) transform is:
##             # xyz = N.dot(xform, dr*r) + r0_n
##             # the Analyze transform is:
##             # xyz = N.dot(xform, dr*r - r0_a)
##             # .... Then r0_n = -N.dot(xform, r0_a)
##             # (this should work for FidImages also)
##             self.r0 = -N.dot(self.xform, self.r0)
            
        self.prefilter = None
        self.vox_coords = self.zyx2vox([0,0,0])

    #-------------------------------------------------------------------------
    def zyx_coords(self, vox_coords=None):
        if vox_coords is not None:
            vox_coords = N.asarray(vox_coords)
        else:
            vox_coords = self.vox_coords
        # assume that vox_coords.shape[0] is 3, but pad dr and r0
        # with fake dims if necessary
        slices = (slice(None),) + (None,)*(len(vox_coords.shape)-1)
        dr = self.dr[slices]
        r0 = self.r0[slices]
        return N.dot(self.xform, vox_coords*dr) + r0
    #-------------------------------------------------------------------------
    def zyx2vox(self, zyx):
        zyx = N.asarray(zyx)
        ixform = N.linalg.inv(self.xform)
        r_img = N.dot(ixform, zyx - self.r0)
        return N.round(r_img/self.dr)
    #-------------------------------------------------------------------------
    def slicing(self):
        return [self.ax, self.cor, self.sag]
    #-------------------------------------------------------------------------
    def transverse_slicing(self, slice_idx):
        (ax, cor, sag) = self.slicing()
        x,y = {
            ax: (-1, -2), # x, y
            cor: (-1, -3), # x, z
            sag: (-2, -3), # y, z
        }.get(slice_idx)
        return x,y
    #-------------------------------------------------------------------------
    def extents(self):
        return [[min(*x),max(*x)]
                for x in zip(self.zyx_coords(vox_coords=(0,0,0)),
                             self.zyx_coords(vox_coords=self.shape))]
    
    #-------------------------------------------------------------------------
    def plane_xform(self, slice_idx):
        """
        Retrieve the submatrix that maps the plane selected by slice_idx
        into zyx space.
        """
        M = self.xform.copy()
        row_idx = range(3)
        col_idx = range(3)
        col = slice_idx
        row = (abs(M[:,col]) == 1).nonzero()[0]
        row_idx.remove(row)
        col_idx.remove(col)
        # do some numpy "fancy" slicing to find the sub-matrix
        Msub = M[row_idx][:,col_idx]
        return Msub
    #-------------------------------------------------------------------------
    def is_xpose(self, slice_idx):
        """
        Based on the plane transform, see if the data that is sliced in
        this direction should be transposed. This can be found by examining
        the submatrix that maps to the 2 dimensions of the slice.
        """
        Msub = self.plane_xform(slice_idx)
        # if abs(Msub) is not an identity xform, then slice needs xpose
        return not Msub[0,0]
    #-------------------------------------------------------------------------
    def data_xform(self, slice_idx, zyx_coords):
        """
        Given the a sliceplots slicing index and the current vox position,
        get a slice of data.
        """
        slicer = [slice(0,d) for d in self.shape]
        slicer[slice_idx] = self.zyx2vox(zyx_coords)[slice_idx]
        slicer = tuple(slicer)
        Msub = self.plane_xform(slice_idx)
        xform = compose_xform(Msub)
        return xform(self[slicer])
    #-------------------------------------------------------------------------
    def __getitem__(self, slicer):
        if self.prefilter:
            return self.prefilter(super(SlicerImage, self).__getitem__(slicer))
        else:
            return super(SlicerImage, self).__getitem__(slicer)


def compose_xform(M, prefilter=None):
    xform = prefilter or (lambda x: x)
    if not M[0,0]:
        xform = lambda x, g=xform: N.swapaxes(g(x), 0, 1)
    if M[0,0] < 0 or M[0,1] < 0:
        xform = lambda x, g=xform: reverse(g(x), axis=0)
    if M[1,0] < 0 or M[1,1] < 0:
        xform = lambda x, g=xform: reverse(g(x), axis=1)
    return xform
    
