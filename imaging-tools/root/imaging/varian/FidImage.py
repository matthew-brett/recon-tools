"A class for interpreting image data in FID format"
import struct
import stat
import os.path
from Numeric import empty
from pylab import Complex32, Float32, Int16, Int32, pi, mlab, fft, fliplr,\
  zeros, fromstring, reshape, arange, take, floor, argmax, multiply, asarray
from imaging.util import shift
from imaging.imageio import BaseImage
from imaging.varian import tablib
from imaging.varian.ProcPar import ProcPar, ProcParImageMixin
from imaging.varian.FidFile import FidFile

FIDL_FORMAT = "fidl"
VOXBO_FORMAT = "voxbo"
ANALYZE_FORMAT = "analyze"
MAGNITUDE_TYPE = "magnitude"
COMPLEX_TYPE = "complex"

#-----------------------------------------------------------------------------
def complex_fromstring(data, numtype):
    return fromstring(
      fromstring(data, numtype).byteswapped().astype(Float32).tostring(),
      Complex32)

##############################################################################
class FidImage (BaseImage, ProcParImageMixin):

    """
    FidImage loads a FID formatted file from a Varian system. It knows how to
    reassemble volume data from various scanner sequences. A FidImage is both
    a BaseImage, giving it a clean, predictable interface appropriate for an
    image, and a ProcParImageMixin, giving it access to the parsed data from
    the procpar file. Additionally, FidImage loads more specific volume data
    such as reference and navigator data. FidImage calls on the FidFile class
    to interface with the actual data on-disk. FidImage contains a multitude
    of potentially useful image information that is freely accessible (due to
    Python's design), mainly assigned in the loadParams(...) and the
    loadData(...) methods.
    
    @cvar _procpar: holds the contents of the procpar file
    """
    #-------------------------------------------------------------------------
    def __init__(self, datadir, tr=None):
        self.loadParams(datadir, tr=tr)
        self.initializeData()
        self.loadData(datadir)

    #-------------------------------------------------------------------------
    def loadParams(self, datadir, tr=None):
        "calls superclass function, and potentially over-rides tr"
        ProcParImageMixin.loadParams(self, datadir)

        # manually override tr
        if tr > 0: self.tr = tr

    #-------------------------------------------------------------------------
    def logParams(self):
        "Report scan parameters to stdout."
        print "Phase encode table: ", self.petable_name
        print "Pulse sequence: %s" % self.pulse_sequence
        print "Number of volumes: %d" % self.nvol
        print "Number of slices: %d" % self.nslice
        print "Number of segments: %d" % self.nseg
        print "Number of navigator echoes per segment: %d" % self.nav_per_seg
        print "Number of phase encodes per slice (including any navigators echoes): %d" % self.n_pe
        print "Number of frequency encodes: %d" % self.n_fe_true
        print "Raw precision (bytes): ", self.datasize
        print "Number of reference volumes: %d" % len(self.ref_vols)
        print "Orientation: %s" % self.orient
        print "Pixel size (phase-encode direction): %7.2f" % self.xsize 
        print "Pixel size (frequency-encode direction): %7.2f" % self.ysize
        print "Slice thickness: %7.2f" % self.zsize

    #-------------------------------------------------------------------------
    def initializeData(self):
        "Allocate data matrices."
        nrefs = len(self.ref_vols)
        self.data = zeros(
          (self.nvol, self.nslice, self.n_pe_true, self.n_fe_true), Complex32)
        self.nav_data = zeros(
          (self.nvol, self.nslice, self.nav_per_slice, self.n_fe_true),
          Complex32)
        self.ref_data = zeros(
          (nrefs, self.nslice, self.n_pe_true, self.n_fe_true), Complex32)
        self.ref_nav_data = zeros(
          (nrefs, self.nslice, self.nav_per_slice, self.n_fe_true), Complex32)

    #-------------------------------------------------------------------------
    def _load_petable(self):
        """
        Read the phase-encode table and organize it into arrays which map
        k-space line number (recon_epi convention) to the acquisition order.
        These arrays will be used to read the data into the recon_epi as
        slices of k-space data.  Assumes navigator echo is aquired at the
        beginning of each segment.
        """
        nav_per_seg = self.nav_per_seg
        nseg = self.nseg
        nav_per_slice = self.nav_per_slice
        n_pe = self.n_pe
        pe_per_seg = self.pe_per_seg
        n_pe_true = self.n_pe_true
        pe_true_per_seg = n_pe_true/nseg
        nslice =  self.nslice
        pulse_sequence = self.pulse_sequence

        if pulse_sequence == "asems":
            line = range(self.n_pe)
            for i in range(self.n_pe):      # !!!!! WHAT IS THIS TOO ? !!!!!
                if i%2:
                     line[i] = self.n_pe - i/2 - 1
                else:
                     line[i] = i/2
        else:
            table_filename = os.path.join(tablib, self.petable_name)
            line = file(table_filename).readline().split()
            line = [int(i) for i in line]

        self.petable = empty(n_pe_true*nslice, Int16)
        i = 0 
        for slice in range(nslice):
            for pe in range(n_pe_true):
                seg = pe/pe_true_per_seg
                if pulse_sequence in ('epidw','epidw_se'):
                    offset = slice*n_pe + (seg+1)*nav_per_seg
                else:
                    offset = (slice + seg*nslice)*pe_per_seg \
                             - seg*pe_per_seg + (seg+1)*nav_per_seg
                # Find location of the pe'th phase-encode line in acquisition order.
                self.petable[i] = line.index(pe) + offset
                i += 1
            
        self.navtable = empty(nav_per_slice*nslice, Int16)
        if nav_per_seg == 1:    
            i = 0        
            # appears to not work if nav_per_seg > 1 ?
            for slice in range(nslice):
                for seg in range(nseg):
                    if pulse_sequence in ('epidw','epidw_se'):
                        offset = slice*n_pe + seg*pe_per_seg 
                    else:
                        offset = (slice + seg*nslice)*pe_per_seg
                    self.navtable[i] = offset
                    i += 1

    #-------------------------------------------------------------------------
    #### A note for using FidFiles in all volume readers ####
    #
    # FidFiles and DataBlocks:
    # The ONLY two things that should be understood about
    # the low-level fid file handling are how a FidFile
    # object addresses blocks, and how DataBlock objects
    # yield data. Every fid file has a FID Header followed
    # by a number (fidfile.nblocks) of data blocks. Each of
    # these data blocks is then made out of a Block Header
    # followed by a number (fidfile.ntraces) of traces.
    # Every trace has a certain number of points, and will
    # correspond to one phase encode line. How the volume
    # is split up among the blocks in the fid file
    # distinguishes the format of the file (see fidformat
    # switching logic in loadData(self, datadir))
    #
    # How a FidFile addresses blocks:
    #     block = fidfile.getBlock(block_number)
    # This statement yields a DataBlock object which represents
    # the (block_number+1)th block in the fid file
    #
    # How a DataBlock yields data:
    #     data = block.getData()
    # This statement returns all the data in the given block.
    # The particular volume reader method knows where to
    # put the block data into the volume it is constructing.
    #-------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------
    def _read_compressed_volume(self, fidfile, vol):
        """
        Reads one volume from a compressed FID file.
        @return: block of data with shape (nslice*n_pe, n_fe_true)
        """
        block = fidfile.getBlock(vol)
        bias = complex(block.lvl, block.tlt)
        volume = complex_fromstring(block.getData(), self.raw_typecode)
        volume = (volume - bias).astype(Complex32)
        return reshape(volume, (self.nslice*self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def _read_uncompressed_volume(self, fidfile, vol):
        """
        Reads one volume from an uncompressed FID file.
        @return: block of data with shape (nslice*n_pe, n_fe_true)
        """        
        volume = empty((self.nslice, self.n_pe*self.n_fe_true), Complex32)
        for slice_num, slice in enumerate(volume):
            block = fidfile.getBlock(self.nslice*vol + slice_num)
            bias = complex(block.lvl, block.tlt)
            slice[:] = complex_fromstring(block.getData(), self.raw_typecode)
            slice[:] = (slice - bias).astype(Complex32)
        return reshape(volume, (self.nslice*self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def _read_epi2fid_volume(self, fidfile, vol):
        """
        Reads one volume from an epi2fid FID file.
        @return: block of data with shape (nslice*n_pe, n_fe_true)
        """
        # procpar indicates navigator lines, but none are read in? huh
        volume = empty(
          (self.nslice, self.nseg, self.pe_per_seg, self.n_fe_true),Complex32)
        pe_true_per_seg = self.pe_per_seg - self.nav_per_seg
        for seg in range(self.nseg):
            for slice in range(self.nslice):
                for pe in range(pe_true_per_seg):
                    block = fidfile.getBlock(
                        self.nslice*(self.nvol_true*(seg*pe_true_per_seg + pe) + vol)
                        + slice)
                    volume[slice, seg, self.nav_per_seg+pe, :] = \
                                  complex_fromstring(block.getData(), self.raw_typecode) 
        return reshape(volume, (self.nslice*self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def _read_asems_ncsnn_volume(self, fidfile, vol):
        """
        Reads one volume from an asems_ncsnn FID file.
        @return: block of data with shape (nslice*n_pe, n_fe_true)
        """
        volume = empty((self.nslice, self.n_pe, self.n_fe_true), Complex32)
        for pe in range(self.n_pe):
            block = fidfile.getBlock(pe*self.nvol + vol)
            bias = complex(block.lvl, block.tlt)
            for slice, trace in enumerate(block):
                trace = complex_fromstring(trace, self.raw_typecode)
                volume[slice,pe] = (trace - bias).astype(Complex32)
        return reshape(volume, (self.nslice*self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def _read_asems_nccnn_volume(self, fidfile, vol):
        """
        Reads one volume from an asems_nccnn FID file.
        @return: block of data with shape (nslice*n_pe, n_fe_true)
        """
        volume = empty((self.nslice, self.n_pe, self.n_fe_true), Complex32)
        for pe in range(self.n_pe):
            for slice in range(self.nslice):
                block = fidfile.getBlock(
                  ((pe*self.nslice+slice)*self.nvol + vol))
                bias = complex(block.lvl, block.tlt)
                trace = complex_fromstring(block.getData(), self.raw_typecode)
                volume[slice,pe] = (trace - bias).astype(Complex32)
        return reshape(volume, (self.nslice*self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def loadData(self, datadir):
        """
        This method reads the data from a fid file into following VarianData
        attributes: 

        data: A rank 4 array containing time-domain data. This array is 
          dimensioned as data(nvol,nslice,n_pe_true,n_fe_true) where nvol 
          is the number of volumes, nslice is the number of slices per volume,
          n_pe_true is the number of phase-encode lines and n_fe_true is the
          number read-out points.

        nav_data: A rank 4 array containing time-domain data for the navigator
          echoes of the image data. This array is dimensioned as 
          nav_data(nvol,nslice,nav_per_slice,n_fe_true).

        ref_data: A rank 4 array containing time-domain reference scan data 
          (phase-encode gradients are kept at zero). This array is dimensioned 
          as ref_data(numrefs, nslice,n_pe_true,n_fe_true). 

        ref_nav_data: A rank 4 array containing time-domain data for the 
          navigator echoes of the reference scan data which is dimensioned as 
          ref_nav_data(numrefs, nslice,nav_per_slice,n_fe_true).
        """
        nav_per_slice = self.nav_per_slice
        n_pe = self.n_pe
        n_pe_true = self.n_pe_true
        n_fe = self.n_fe  
        n_fe_true = self.n_fe_true
        nslice =  self.nslice
        pulse_sequence = self.pulse_sequence
        nvol = self.nvol
        nvol_true = self.nvol_true
        numrefs = len(self.ref_vols)

        # open fid file
        fidfile = FidFile(os.path.join(datadir, "fid")) 

        ##############################
        ### Fid format is determined by the number of blocks and the number of
        ### traces per block.

        # this format has one block per volume
        blockstraces_compressed = (nvol_true, nslice*n_pe)
        # this format has one block per slice
        blockstraces_uncompressed = (nvol_true*nslice, n_pe)
        # this format has one block per phase encode (but in a weird order!)
        blockstraces_epi2fid = (nvol_true*nslice*n_pe_true, 1)
        # this format has one block per ???
        blockstraces_asems_ncsnn = (nvol_true*n_pe, nslice)
        # this format has one block per ???
        blockstraces_asems_nccnn = (nvol_true*nslice*n_pe, 1)

        # determine fid format based on number of blocks and traces per block
        blockstraces = (fidfile.nblocks, fidfile.ntraces)
        if blockstraces_compressed == blockstraces:
            fidformat = "compressed"
        elif blockstraces_uncompressed == blockstraces:
            fidformat = "uncompressed"
        elif blockstraces_epi2fid == blockstraces:
            fidformat = "epi2fid"
        elif blockstraces_asems_ncsnn == blockstraces:
            fidformat = "asems_ncsnn"
        elif blockstraces_asems_nccnn == fidDataLen:
            fidformat = "asems_nccnn"
        else:
            raise "unrecognized fid format, (nblocks, ntraces) = (%d,%d)"%\
                  blockstraces

        # choose volume reading method based on fid format
        volreader = {
          "compressed":   self._read_compressed_volume,
          "uncompressed": self._read_uncompressed_volume,
          "epi2fid":      self._read_epi2fid_volume,
          "asems_ncsnn":  self._read_asems_ncsnn_volume,
          "asems_nccnn":  self._read_asems_nccnn_volume
        }[fidformat]

        # determine if time reversal needs to be performed
        time_reverse = \
          (fidformat=="compressed" and pulse_sequence != "epi_se") or \
          (fidformat=="uncompressed" and \
           pulse_sequence not in ("epidw", "epidw_se"))
        time_rev = n_fe_true - 1 - arange(n_fe_true)

        # determine if phase encodes need reordering
        needs_pe_reordering = fidformat not in ("asems_ncsnn", "asems_nccnn")

        # load phase encode table
        if needs_pe_reordering: self._load_petable()

        for vol in range(nvol_true):

            # read the next image volume
            volume = volreader(fidfile, vol)

            # time-reverse the data
            if time_reverse:
                f = self.pe_per_seg*nslice
                for seg in range(self.nseg):
                    base = seg*f
                    for pe in range(base+1, base+f, 2): 
                        volume[pe] = take(volume[pe], time_rev)

            # Reorder data according to phase encode table and separate
            # k-space data from navigator echos
            if needs_pe_reordering:
                ksp_image = take(volume, self.petable)
                navigators = take(volume, self.navtable) 
            else:
                ksp_image = volume
                navigators = empty(0, Complex32)

            # reshape into final volume shape
            ksp_image = reshape(ksp_image, (nslice, n_pe_true, n_fe_true))
            navigators = reshape(navigators, (nslice, nav_per_slice, n_fe_true))

            # The time-reversal section above reflects the odd slices through
            # the pe-axis. This section puts all slices in the same orientation.
            if time_reverse:
                for slice in range(0,nslice,2):
                    for pe in range(n_pe_true):
                        ksp_image[slice,pe] = \
                          take(ksp_image[slice,pe], time_rev)
                    for pe in range(nav_per_slice):
                        navigators[slice,pe] = \
                          take(navigators[slice,pe], time_rev)

            # assign volume to the appropriate output matrix
            if vol in self.ref_vols:
                self.ref_data[vol] = ksp_image
                self.ref_nav_data[vol] = navigators
            else:
                self.data[vol-numrefs] = ksp_image
                self.nav_data[vol-numrefs] = navigators

        self.setData(self.data)

    #-------------------------------------------------------------------------
    def save(self, outfile, file_format, data_type):
        "Save the image data to disk."
        data = self.data

        print "Saving to disk (%s format). Please Wait"%file_format
        if file_format == ANALYZE_FORMAT:
            from imaging import analyze
            dtypemap = {
              MAGNITUDE_TYPE: analyze.FLOAT,
              COMPLEX_TYPE: analyze.COMPLEX }
            analyze.writeImage(self, outfile, dtypemap[data_type], 3)
        elif file_format == FIDL_FORMAT:
            f_img = open("%s.4dfp.img" % (outfile), "w")
            if data_type == MAGNITUDE_TYPE: data = abs(data)
            for volume in data:
                f_img.write(vol_transformer(volume).byteswapped().tostring())
        else: print "Unsupported output type: %s"%file_format

        # !!!!!!!!! WHERE IS THE CODE TO SAVE IN VOXBO FORMAT !!!!!!!!
