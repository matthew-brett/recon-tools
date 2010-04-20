import numpy as np
from os import path
from recon.odict import odict
from recon import import_from
from recon.inplane_xforms import compose_xform
from recon.util import Quaternion, integer_ranges, scale_data

# module-private dict specifying available image readers
_readers = odict((
    ("analyze", ("recon.analyze","readImage")),
    ("nifti", ("recon.nifti","readImage")),
    ("fid", ("recon.scanners.varian","FidImage")),
    ("fdf", ("recon.scanners.varian","FDFImage")),
    ("siemens", ("recon.scanners.siemens", "SiemensImage"))))
available_readers = _readers.keys()

# module-private dict specifying available image writers
_writers = odict((
    ("analyze", ("recon.analyze","writeImage")),
    ("nifti-single", ("recon.nifti","writeImage")),
    ("nifti-dual", ("recon.nifti","writeDualImage")),
    ))

available_writers = _writers.keys()

recon_output2dtype = odict((
    ('magnitude', np.dtype(np.float32)),
    ('complex', np.dtype(np.complex64)),
    ('double', np.dtype(np.float64)),
    ('byte', np.dtype(np.int8)),
    ('ubyte', np.dtype(np.uint8)),
    ('short', np.dtype(np.int16)),
    ('ushort', np.dtype(np.uint16)),
    ('int', np.dtype(np.int32)),
    ('uint', np.dtype(np.uint32)),
    ))

# ReconTools default = idx 0
output_datatypes = recon_output2dtype.keys()

#-----------------------------------------------------------------------------
def get_dims(data):
    """
    Extract ndim, tdim, kdim, jdim, and idim from data shape.
    @return: (ndim, tdim, kdim, jdim, idim)
    """
    # tdim can be 0, but for kdim needs to be at least 1 for a 3D space
    shape = data.shape
    while shape[0] < 2:
        shape = shape[1:]
    ndim = len(shape)
    if ndim < 2 or ndim > 4:
        raise ValueError("data shape must be 2, 3, or 4 dimensional:",shape)
    shape = (1,)*(3-ndim) + shape
    ndim = len(shape)
    return (ndim,) + (0,)*(4-ndim) + shape

##############################################################################
class DataChunk (object):
    """
    A sub-iterator with N-dimensional data; can offer up further
    DataChunks with (N-1)-dimensional data
    """
    def __init__(self, data, num):
        self._data = data
        self.num = num
        self.shape = data.shape

    def __getitem__(self, slicer):
        return self._data[slicer]

    def __setitem__(self, slicer, data):
        self._data[slicer] = np.asarray(data).astype(self._data.dtype)
    
    def __iter__(self):
        if len(self.shape) > 1:
            iternum = 0
            while iternum < self.shape[0]:
                yield DataChunk(self._data[iternum], iternum)
                iternum += 1
            raise StopIteration
        else:
            raise ValueError("can't iterate through a 1D array")

##############################################################################
class ReconImage (object):
    """
    Interface definition for any image in Recon Tools.
    This class of images will be able to go through many of the available ops.

    This class of images can be exported to some medical imaging formats.
    
    Attributes:
      _data:  2, 3, or 4 dimensional matrix representing a slice, single
             volume, or a timecourse of volumes.
      ndim:  number of dimensions
      tdim:  number of volumes in a timecourse
      kdim:  number of slices per volume
      jdim:  number of rows per slice
      idim:  number of columns per row
      isize: spacial width of array columns
      jsize: spacial height of array rows
      ksize: spacial slice thickness (3rd dim of array)
      tsize: duration of each time-series volume (4th dim of array)
      x0: x coordinate of xyz offset
      y0: y coordinate of xyz offset
      z0: z coordinate of xyz offset
      orientation: name of the orientaion (coronal, axial, etc)
      orientation_xform: quaternion describing the orientation

    capabilities provided:
      volume/slice slicing
      fe/pe slicing
      __getitem__, __setitem__
      data xform (abs, real, imag, etc)
    """

    #-------------------------------------------------------------------------
    def __init__(self, data,
                 isize=1., jsize=1., ksize=1., tsize=1.,
                 offset=None, scaling=None,
                 orient_xform=None):
        """
        Construct a ReconImage with at least data, isize, jsize, ksize,
        and tsize known. Optional information are an offset 3-tuple
        specifying (x0,y0,z0), a Quaternion object representing the
        transformation of this data to neurological orientation
        (+X,+Y,+Z) = (Right,Anterior,Superior), and a name for the data's
        orientation (used for ANALYZE format output).
        """
        self.setData(data)
        self.isize, self.jsize, self.ksize, self.tsize = \
                    (isize, jsize, ksize, tsize)

        self.orientation_xform = orient_xform or Quaternion()

        # offset should be the (x,y,z) offset in xyz-space
        xform = self.orientation_xform.tomatrix()
        if offset is not None:
            (self.x0, self.y0, self.z0) = offset
        else:
            # assume that vox at (idim/2, jdim/2, kdim/2) is the origin
            # thus Trans*(i0,j0,k0)^T + (x0,y0,z0)^T = (0,0,0)^T
            (self.x0, self.y0, self.z0) = \
                      -np.dot(xform, np.array([self.isize*self.idim/2.,
                                               self.jsize*self.jdim/2.,
                                               self.ksize*self.kdim/2.]))
            

        self.scaling = scaling or 1.0

    #-------------------------------------------------------------------------
    def info(self):
        print "ndim =",self.ndim
        print "idim =",self.idim
        print "jdim =",self.jdim
        print "kdim =",self.kdim
        print "tdim =",self.tdim
        print "isize =",self.isize
        print "jsize =",self.jsize
        print "ksize =",self.ksize
        print "x0 =",self.x0
        print "y0 =",self.y0
        print "z0 =",self.z0
        print "data.shape =",self.data.shape
        print "data.dtype =",self.data.dtype

    #-------------------------------------------------------------------------
    def setData(self, data):
        """Inform self about dimension info from the data array. Assuming
        that new data is centered at the same location as the old data,
        update the origin.
        """
        from recon.slicerimage import SlicerImage
        if hasattr(self, 'shape'):
            old_shape = self.shape
        else:
            old_shape = data.shape[-3:]
        self.data = data
        self.ndim, self.tdim, self.kdim, self.jdim, self.idim = get_dims(data)
        self.shape = (self.tdim, self.kdim, self.jdim, self.idim)
        while self.shape[0] < 1:
            self.shape = self.shape[1:]
        self.data.shape = self.shape
##         try:
##             # this fails if isize,jsize,ksize reset before resizing
##             # the old_ctr_vox is mis-calculated
##             std_img = SlicerImage(self)
##             old_ctr_vox = std_img.zyx2vox((0,0,0))
##             ctr_ratio0 = old_ctr_vox/np.array(old_shape, dtype=np.float32)
            
##             new_shape = data.shape[-3:]
##             ctr_ratio1 = old_ctr_vox/np.array(new_shape, dtype=np.float32)

##             new_ctr_vox = old_ctr_vox * (ctr_ratio0 / ctr_ratio1)
##             off_center = std_img.zyx_coords(vox_coords=new_ctr_vox)
##             origin = [self.z0, self.y0, self.x0]
##             origin = map(lambda (x1,x2): x1 - x2, zip(origin, off_center))
##             self.z0, self.y0, self.x0 = origin
##         except:
##             pass
    #-------------------------------------------------------------------------
    def concatenate(self, image, axis=0, newdim=False):
        """Stitch together two images along a given axis, possibly
        creating a new dimension
        """
        self_sizes = (self.isize, self.jsize, self.ksize)
        image_sizes = (image.isize, image.jsize, image.ksize)

        # pixel sizes must match
        if self_sizes != image_sizes:
            raise ValueError(
              "won't concatenate images with different pixel sizes: %s != %s"%\
              (self_sizes, image_sizes))

        if newdim:
            newdata = np.asarray((self[:], image.data))
        else:
            if len(self.shape) > len(image.shape):
                newdata = np.concatenate((self[:], image[(None,)]))
            else:
                newdata = np.concatenate((self[:], image.data), axis)
        return self._subimage(newdata)

    #-------------------------------------------------------------------------
    def transform(self, new_mapping=None, transform=None, force=False):
        """Updates the voxel to real-space transform.

        There are two modes of usage--
        1) supply a new voxel to real mapping.
           In this case a voxel to voxel transform is found, and the image
           is rotated in-plane around the origin. Only transposes and
           reflections are supported. Image info is updated appropriately

        2) supply a real to real transform to apply to the current mapping.
           In this case the data is not updated, but the mapping is updated.

        """
        if new_mapping is None and transform is None:
            return
        if new_mapping is not None and transform is not None:
            print """
            The user must specify either a new mapping to convert to,
            or a transform to apply, but cannot specify both."""
            return
        if transform is not None:
            # this doesn't change the image array, it just updates the
            # transformation
            old_xform = self.orientation_xform.tomatrix()
            if isinstance(transform, Quaternion):
                transform = transform.tomatrix()
            dim_scale = np.array([self.isize, self.jsize, self.ksize])
            r0 = np.array([self.x0, self.y0, self.z0])
            origin_voxels = np.round(np.linalg.solve(old_xform*dim_scale, -r0))
            # now derive r0 again.. Tmap*(i,j,k)^T + r0^T = (x,y,z)^T
            r0 = -np.dot(transform*dim_scale, origin_voxels)
            self.x0, self.y0, self.z0 = r0
            self.orientation_xform = Quaternion(M=transform)
            return
        # else handle the new mapping
        from recon.slicerimage import nearest_simple_transform
        # Tr already maps [i,j,k]^T into [R,A,S] ...
        # Here I want to change coordinates with this relationship:
        # Tr*[i,j,k]^T = Tnew*[i',j',k']^T
        # so Tnew*(Tvx*[i,j,k]^T) = Tr*[i,j,k]^T
        # So inv(Tnew)*Tr*[i,j,k] = [i',j',k'] = orientation of choice!
        # The task is to analyze Tvx = (Tnew^-1 * Tr) to get rotation
        
        Tr = self.orientation_xform.tomatrix()
        Tvx = np.linalg.solve(new_mapping, Tr)
        Tvxp = nearest_simple_transform(Tvx)
        if not np.allclose(Tvxp, Tvx):
            # Tvxp might be a good choice in this case.. so could suggest
            # Tnew'*Tvxp = Tr
            # (Tvxp^T * Tnew'^T) = Tr^T
            # Tnew' = solve(Tvxp^T, Tr^T)^T
            Tnew_suggest = np.linalg.solve(Tvxp.T, Tr.T).T
            if not force:
                raise ValueError("""This method will not transform the data to
                the stated new mapping because the transformation cannot be
                accomplished through transposes and reflections. The closest new
                mapping you can perform is:\n"""+str(Tnew_suggest))
            else:
                print """It is not possible to rotate simply to the stated
                mapping; proceeding with this mapping:\n"""+str(Tnew_suggest)+\
                """\nbut lying about the final mapping."""
                #new_mapping = Tnew_suggest
                Tvx = Tvxp
        if not Tvx[-1,-1]:
            raise ValueError("This operation only makes in-plane rotations. "\
                             "EG you cannot use the sagittal transform for "\
                             "an image in the coronal plane.")
        if (Tvx==np.identity(3)).all():
            print "Already in place, not transforming"
            return
        # this is for simple mixing of indices
        Tvx_abs = np.abs(Tvx)
        r0 = np.array([self.x0, self.y0, self.z0])
        dvx_0 = np.array([self.isize, self.jsize, self.ksize])
        # mix up the voxel sizes
        dvx_1 = np.dot(Tvx_abs, dvx_0)
        (self.isize, self.jsize, self.ksize) = dvx_1
        
        dim_sizes = np.array(self.shape[-3:][::-1])
        # solve for (i,j,k) where Tr*(i,j,k)^T + r0 = (0,0,0)        
        # columns are i,j,k space, so scale columns by vox sizes
        vx_0 = np.linalg.solve(Tr*dvx_0, -r0)
        # transform the voxels --
        # can normalize to {0..I-1},{0..J-1},{0..K-1} due to periodicity        
        vx_1 = (np.dot(Tvx, vx_0) + dim_sizes) % dim_sizes
        r0_prime = -np.dot(new_mapping*dvx_1, vx_1)
        (self.x0, self.y0, self.z0) = r0_prime
        if self.shape[-1] != self.shape[-2]:
            func = compose_xform(Tvx, view=False, square=False)
            if self.tdim:
                new_shape = (self.tdim,)
            else:
                new_shape = ()
            new_shape += tuple(np.dot(Tvx_abs, self.shape[::-1]).astype('i'))[::-1]
            temp = func(self[:])
            self.resize(new_shape)
            self[:] = temp.copy()
            del temp
        else:
            func = compose_xform(Tvx, view=False)
            func(self[:])
        self.orientation_xform = Quaternion(M=new_mapping)

    #-------------------------------------------------------------------------
    def __iter__(self):
        "Handles iteration over the image--always yields a 3D DataChunk"
        # want to iterate over volumes, if tdim=0, then nvol = 1
        if len(self.shape) > 3:
            for t in range(self.tdim):
                yield DataChunk(self[t], t)
            raise StopIteration
        else:
            yield DataChunk(self[:], 0)
            raise StopIteration
    #-------------------------------------------------------------------------
    def __getitem__(self, slicer):
        if type(slicer) is type(()) and len(slicer) > self.ndim:
            nfakes = len(slicer)-self.ndim
            slicer = (None,)*(nfakes) + slicer[nfakes:]
        return self.data[slicer]
    #-------------------------------------------------------------------------
    def __setitem__(self, slicer, newdata):
        ndata = np.asarray(newdata)
        if ndata.dtype.char.isupper() and self.data.dtype.char.islower():
            print "warning: losing information on complex->real cast!"
        if type(slicer) is type(()) and len(slicer) > self.ndim:
            nfakes = len(slicer)-self.ndim
            slicer = (None,)*(nfakes) + slicer[nfakes:]
        self.data[slicer] = ndata.astype(self.data.dtype)
    #-------------------------------------------------------------------------
    def __mul__(self, a):
        self[:] = self[:]*a
    #-------------------------------------------------------------------------
    def __div__(self, a):
        self[:] = self[:]/a
    #-------------------------------------------------------------------------
    def _subimage(self, data):        
        return ReconImage(data,
                          self.isize, self.jsize, self.ksize, self.tsize,
                          offset=(self.x0, self.y0, self.z0),
                          scaling=self.scaling,
                          orient_xform=self.orientation_xform)

    #-------------------------------------------------------------------------
    def subImage(self, subnum):
        "returns subnum-th sub-image with dimension ndim-1"
        return self._subimage(self.data[subnum])

    #-------------------------------------------------------------------------
    def subImages(self):
        "yeilds all images of dimension self.ndim-1"
        if len(self.shape) < 2:
            raise StopIteration("can't iterate subdimensions of a 2D image")
        for subnum in xrange(self.shape[0]):
            yield self.subImage(subnum)

    #-------------------------------------------------------------------------
    def resize(self, newsize):
        """
        resize/reshape the data, non-destructively if the number of
        elements doesn't change
        """
        if np.product(newsize) == np.product(self.shape):
            self.data.shape = tuple(newsize)
        else:
            self.data.resize(tuple(newsize), refcheck=False)
        self.setData(self[:])

    #-------------------------------------------------------------------------
    def runOperations(self, opchain, logger=None):
        """
        This method runs the image object through a pipeline of operations,
        which are ordered inside the opchain list. ReconImage's method is
        a basic operations driver, and could be expanded in subclasses.
        """
        for operation in opchain:
            operation.log("Running")
            if operation.run(self) == -1:
                raise RuntimeError("critical operation failure")
            if logger is not None:
                logger.logop(operation)
    
    #-------------------------------------------------------------------------
    def writeImage(self, filestem, format_type="analyze",
                   datatype="magnitude", **kwargs):
        """
        Export the image object in a medical file format (ANALYZE or NIFTI).
        format_type is one of the internal file format specifiers, which
        are currently %s.
        possible keywords are:
        datatype -- a datatype identifier, supported by the given format
        targetdim -- number of dimensions per file
        filetype -- differentiates single + dual formats for NIFTI
        suffix -- over-ride default suffix style (eg volume0001)

        If necessary, a scaling is found for integer types
        """%(" ; ".join(available_writers))

        new_dtype = recon_output2dtype.get(datatype.lower(), None)
        if new_dtype is None:
            raise ValueError("Unsupported data type: %s"%datatype)

        # The image writing tool does scaling only to preserve dynamic range
        # when saving an as integer data type. Therefore specifying a scale
        # in the writeImage kwargs is not appropriate, since it would
        # cause the image writing logic to "unscale" the data first--it
        # is not to be used as a "gain" knob.
        try:
            kwargs.pop('scale')
        except KeyError:
            pass
        if new_dtype in integer_ranges.keys():
            scale = float(scale_data(self[:], new_dtype))
        else:
            scale = float(1.0)

        _write(self, filestem, format_type, scale=scale,
               dtype=new_dtype,**kwargs)

#-----------------------------------------------------------------------------
def get_reader(format):
    "Return an image file reader for the specified format."
    readerspec = _readers.get(format)
    if readerspec is None:
        raise ValueError("Reader '%s' not found.  Avaliable readers are: %s"%\
          (format, ", ".join(available_readers)))
    return import_from(*readerspec)
#-----------------------------------------------------------------------------
def readImage(filename, format=None, datatype=None, **kwargs):
    "Load an image in the specified format from the given filename."
    format_guess = {
        ".hdr": "nifti",
        ".img": "nifti",
        ".nii": "nifti",
        ".dat": "siemens",
        ".fid": "fid",
        ".fdf": "fdf"
    }
    if not path.exists(filename):
        raise IOError("This path is invalid: %s"%filename)
    if datatype and datatype not in recon_output2dtype.keys():
        raise ValueError("Unsupported data type: %s"%datatype)
    kwargs['target_dtype'] = recon_output2dtype.get(datatype, None)
    filestem,ext = clean_name(filename)
    if format is None:
        try:
            format = format_guess[ext]
        except KeyError:
            raise Exception("No format guessed for this extension: %s"%ext)
    # do a special test for nifti/analyze images, which can share a
    # common extension
    if format=="nifti":
        try:
            return get_reader(format)(filestem, **kwargs)
        except:
            return get_reader("analyze")(filestem, **kwargs)
    return get_reader(format)(filestem, **kwargs)

#-----------------------------------------------------------------------------
def get_writer(format_type):
    "Return an image file writer method for the Recon output type"
    writerspec = _writers.get(format_type)
    if writerspec is None:
        raise ValueError("Writer method for '%s' not found. "\
                         "Avaliable writers are: %s"%\
                         (format_type, ", ".join(available_writers)))
    return import_from(*writerspec)
#-----------------------------------------------------------------------------
def _write(image, filestem, format_type, dtype=None, targetdim=None,
           suffix=None, scale=1.0):
    """
    Given a RTools format_type description (eg, nifti-single), a numpy
    dtype in dtype, and a filename, write an appropriate file.
    Other keyword args:
      targetdim -- number of dimensions per output file (2, 3, or 4)
      suffix -- over-rides normal _volume000n suffix (for targetdim=3 only)
      scale -- scaling factor for integer output dtypes
    """

    Writer = get_writer(format_type)
    dimnames = {3: "volume", 2: "slice"}
    def images_and_names(image, stem, targetdim, suffix=None):
        # base case
        if targetdim >= image.ndim: return [(image, stem)]
        
        # recursive case
        subimages = tuple(image.subImages())
        if suffix is not None:
            substems = ["%s"%(stem,)+suffix%(i,) \
                        for i in range(len(subimages))]
        else:
            substems = ["%s_%s%04d"%(stem, dimnames[image.ndim-1], i)\
                        for i in range(len(subimages))]
        return _concatenate(
          [images_and_names(subimage, substem, targetdim)\
           for subimage,substem in zip(subimages, substems)])
    if suffix is not None: targetdim = 3
    if targetdim is None: targetdim = image.ndim
    for subimage, substem in images_and_names(image,filestem,targetdim,suffix):
        Writer(subimage, substem, dtype=dtype, scale=scale)

#-----------------------------------------------------------------------------
def clean_name(fname):
    pruned_exts = ['.nii', '.hdr', '.img', '.dat', '.fid', '.fdf']
    if path.splitext(fname.rstrip('/'))[-1] in pruned_exts:
        return path.splitext(fname.rstrip('/'))
    return fname.rstrip('.'), ''
#-----------------------------------------------------------------------------
def _concatenate(listoflists):
    "Flatten a list of lists by one degree."
    finallist = []
    for sublist in listoflists: finallist.extend(sublist)
    return finallist

