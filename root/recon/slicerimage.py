from recon.imageio import ReconImage
from recon.util import decompose_rot, eulerRot
from recon.inplane_xforms import compose_xform, reverse
import numpy as np

def nearest_simple_transform(M):
    """Finds the closest transformation to M that is composed of only n*(PI/2)
    rotations and axis reflections.
    """
    # if det(M) < 0, then M has some reflections going on (which is still fine)
    tag = np.linalg.det(M) < 0 and -1 or 1
    M[:,2] *= tag
    (theta, psi, phi) = decompose_rot(M)
    angs = np.array([theta, psi, phi])
    angs = np.pi/2 * ((angs + np.sign(angs)*np.pi/4)/(np.pi/2)).astype('i')
    M2 = eulerRot(theta=angs[0], psi=angs[1], phi=angs[2])
    M[:,2] *= tag
    M2[:,2] *= tag
    return M2

##############################################################################
class SlicerImage (ReconImage):
    """
    A wrapper of any ReconImage that provides slicing access and vox<->zyx
    transformations, and other helpful stuff
    """

    def __init__(self, image=None):
        if image is None:
            image = ReconImage(np.ones((4,4,4)), 1, 1, 1, 1)
        if not isinstance(image, ReconImage):
            image = ReconImage(np.asarray(image), 1, 1, 1, 1)
        for (attr, val) in image.__dict__.items():
            self.__dict__[attr] = val

        # get vox origin from zyx origin (can't use transform because
        # (these zyx are in the image's native space)
        self.r0 = np.array([self.z0, self.y0, self.x0])
        self.dr = np.array([self.ksize, self.jsize, self.isize])
        # orientation xform maps vx-space:xyz-space (Right,Anterior,Superior)
        # numpy arrays are in C-ordering (zyx), so I'm going to reverse the
        # rows and columns of the xform to be in zyx order
        M = self.orientation_xform.tomatrix()
        self.xform = reverse(reverse(M, axis=-1), axis=-2)
        self.pxform = nearest_simple_transform(M)
        self.pxform = reverse(reverse(self.pxform,axis=-1),axis=-2)
        if not np.allclose(self.xform, self.pxform):
            print 'OBLIQUE IMAGE: NEED TO PLOT IN ROTATED "VOXEL" SPACE'
            sl0_vox_center = np.array([0, self.jdim/2, self.idim/2])
            sl0_center = np.dot(self.xform*self.dr, sl0_vox_center) + self.r0
            # reset this r0 to correspond to the rotated voxel box
            self.r0 = sl0_center - np.dot(self.pxform*self.dr, sl0_vox_center)
            # reset the xform to be the planar xform
            self.xform = self.pxform
            print self.r0
        # These variables describe which dimension a given standard slicing
        # (axial, sagittal, coronal) slices in the data array
        self.ax, self.cor, self.sag = abs(np.dot(self.pxform, np.arange(3)).astype(np.int32)).tolist()            
        self.prefilter = None
        self.vox_coords = self.zyx2vox([0,0,0])

    #-------------------------------------------------------------------------
    def zyx_coords(self, vox_coords=None):
        """Returns a point (z,y,x) based on either the SlicerImage's current
        state, or a supplied voxel coordinate. If given a (3xN) series of
        indices, zyx_coords() will return a (3xN) array of points.
        """
        if vox_coords is not None:
            vox_coords = np.asarray(vox_coords[-3:])
        else:
            vox_coords = self.vox_coords
        # assume that vox_coords.shape[0] is 3,
        # but pad r0 with fake dims if necessary
        slices = (slice(None),) + (None,)*(len(vox_coords.shape)-1)
        return np.dot(self.xform*self.dr, vox_coords) + self.r0[slices]
    #-------------------------------------------------------------------------
    def zyx2vox(self, zyx):
        """Returns the closest voxel indices given a point (z,y,x).
        """
        zyx = np.asarray(zyx, dtype=np.float64)
        #ixform = np.linalg.inv(self.xform)
        #r_img = np.dot(ixform, zyx - self.r0)
        xform = self.xform*self.dr
        vox = np.round(np.linalg.solve(xform, zyx-self.r0)).astype('i')
        for i in range(3): vox[i] = np.clip(vox[i], 0, self.shape[i]-1)
        return vox
        #return np.round(r_img/self.dr).astype(np.int32)
    #-------------------------------------------------------------------------
    def slicing(self):
        """Returns the dimension numbers of the axial, coronal, and sagittal
        planes.
        """
        return [self.ax, self.cor, self.sag]
    #-------------------------------------------------------------------------
    def transverse_slicing(self, slice_idx):
        """Given a slicing frame, slice_idx, return the lab axes for
        left-to-right and up-and-down (x and y)
        """
        (ax, cor, sag) = self.slicing()
        x,y = {
            ax: (-1, -2), # left-right, up-down = x, y
            cor: (-1, -3), # left-right, up-down = x, z
            sag: (-2, -3), # left-right, up-down = y, z
        }.get(slice_idx)
        return x,y
    #-------------------------------------------------------------------------
    def extents(self):
        """Return a list of [min,max] values for each lab axis z, y, x
        """
        # for oblique images, need to look at all 8 corners of the cube
        box = np.empty((8,3))
        box[0] = self.zyx_coords(vox_coords=(0,0,0))
        box[1] = self.zyx_coords(vox_coords=(0,0,self.idim))
        box[2] = self.zyx_coords(vox_coords=(0,self.jdim,0))
        box[3] = self.zyx_coords(vox_coords=(0,self.jdim,self.idim))
        box[4] = self.zyx_coords(vox_coords=(self.kdim,0,0))
        box[5] = self.zyx_coords(vox_coords=(self.kdim,0,self.idim))
        box[6] = self.zyx_coords(vox_coords=(self.kdim,self.jdim,0))
        box[7] = self.zyx_coords(vox_coords=(self.kdim,self.jdim,self.idim))
        return [[x.min(), x.max()] for x in box.T]
        
##         return [[min(*x),max(*x)]
##                 for x in zip(self.zyx_coords(vox_coords=(0,0,0)),
##                              self.zyx_coords(vox_coords=self.shape))]
    
    #-------------------------------------------------------------------------
    def plane_xform(self, slice_idx):
        """Retrieve the submatrix that maps the plane selected by slice_idx
        into zyx space.
        """
        M = self.pxform.copy()
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
        """Based on the plane transform, see if the data that is sliced in
        this direction should be transposed. This can be found by examining
        the submatrix that maps to the 2 dimensions of the slice.
        """
        Msub = self.plane_xform(slice_idx)
        # if abs(Msub) is not an identity xform, then slice needs xpose
        return not Msub[0,0]
    #-------------------------------------------------------------------------
    def data_xform(self, slice_idx, zyx_coords):
        """Given the a sliceplots slicing index and the current vox position,
        get a slice of data.
        """
        slicer = [slice(0,d) for d in self.shape]
        slicer[slice_idx] = self.zyx2vox(zyx_coords)[slice_idx]
        slicer = tuple(slicer)
        Msub = self.plane_xform(slice_idx)
        Msub = reverse(reverse(Msub, axis=-1), axis=-2)
        xform = compose_xform(Msub)
        return xform(self[slicer])
    #-------------------------------------------------------------------------
    def __getitem__(self, slicer):
        if self.prefilter:
            return self.prefilter(super(SlicerImage, self).__getitem__(slicer))
        else:
            return super(SlicerImage, self).__getitem__(slicer)

    
