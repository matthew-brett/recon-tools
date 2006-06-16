"A class for interpreting image data in FID format"
import struct
import stat
import os.path
import sys
from Numeric import empty
from pylab import Complex32, Float32, Int16, Int32, pi, mlab, fft, fliplr,\
  zeros, fromstring, reshape, arange, take, floor, argmax, multiply, asarray
from imaging.util import shift
from imaging.imageio import BaseImage
from imaging.varian import tablib
from imaging.varian.ProcPar import ProcPar, ProcParImageMixin
from imaging.varian.FidFile import FidFile


#-----------------------------------------------------------------------------
def getPulseSeq(datadir):
    pp = ProcParImageMixin(datadir)
    ps = pp.pulse_sequence

    # Some sequences might require different treatment if conditions are true.
    # if mpflash has the flash_converted flag, then use 2DFFT instead of 3D
    if ps == "mp_flash3d":
        flag = (pp.flash_converted and 1 or 0, )
    # ... add more flags as necessary
    else:
        flag = None
    return (ps, flag)

#-----------------------------------------------------------------------------
def complex_fromstring(data, numtype):
    if sys.byteorder == "little":
        return fromstring(
            fromstring(data, numtype).byteswapped().astype(Float32).tostring(),
            Complex32)
    else:
        return fromstring(
	    fromstring(data,numtype).astype(Float32).tostring(), Complex32)

##############################################################################
class FidImage (BaseImage, ProcParImageMixin):

    """
    FidImage loads a FID formatted file from a Varian system. It knows how to
    reassemble volume data from various scanner sequences. A FidImage is both
    a BaseImage, giving it a clean, predictable interface appropriate for an
    image, and a ProcParImageMixin, giving it access to the parsed data from
    the procpar file. Additionally, FidImage loads more specific volume data
    such as reference and navigator data. FidImage calls on the FidFile class
    to interface with the actual data on-disk.
    """
    #-------------------------------------------------------------------------
    def __init__(self, datadir, tr=None):
        ProcParImageMixin.__init__(self, datadir, tr=tr)
        self.initializeData()
        self.loadData(datadir)

    #-------------------------------------------------------------------------
    def logParams(self):
        "Report scan parameters to stdout."
        print "Phase encode table: ", self.petable_name
        print "Pulse sequence: %s" % self.pulse_sequence
        print "Spinecho: %s" % self.spinecho
        print "Number of volumes: %d" % self.nvol
        print "Number of slices: %d" % self.nslice
        print "Number of segments: %d" % self.nseg
        print "Number of navigator echoes per segment: %d" % self.nav_per_seg
        print "Number of phase encodes per slice (including any navigators echoes): %d" % self.n_pe
        print "Number of frequency encodes: %d" % self.n_fe_true
        print "Raw precision (bytes): ", self.datasize
        print "Number of reference volumes: %d" % len(self.ref_vols)
        print "Orientation: %s" % self.orient
        print "Pixel size (phase-encode direction): %7.2f" % self.ysize 
        print "Pixel size (frequency-encode direction): %7.2f" % self.xsize
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
                if pulse_sequence == 'epidw':
                    offset = slice*n_pe + (seg+1)*nav_per_seg
                else:
                    offset = (slice + seg*nslice)*pe_per_seg \
                             - seg*pe_per_seg + (seg+1)*nav_per_seg
                # Find location of the pe'th phase-encode line in acquisition
                # order.
                self.petable[i] = line.index(pe) + offset
                i += 1
            
        self.navtable = empty(nav_per_slice*nslice, Int16)
        if nav_per_seg == 1:    
            i = 0        
            # appears to not work if nav_per_seg > 1 ?
            for slice in range(nslice):
                for seg in range(nseg):
                    if pulse_sequence == 'epidw':
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
                      self.nslice*(self.nvol_true*(seg*pe_true_per_seg + pe)\
                      + vol) + slice)
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
                if self.pulse_sequence == "gems" and self.n_transients>1:
                    volume[slice,pe] = trace
                else: volume[slice,pe] = (trace - bias).astype(Complex32)

                #volume[slice,pe] = trace.astype(Complex32)
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
    def _get_fidformat(self, fidfile):
        """
        Determine fid format from the number of blocks per volumen and the
        number of traces per block.  Known formats are:
          compressed
          uncompressed
          epi2fid
          asems_ncsnn
          asems_nccnn
        """
        n_pe = self.n_pe
        n_pe_true = self.n_pe_true
        nslice =  self.nslice
        nvol_true = self.nvol_true
        nblocks = fidfile.nblocks
        ntraces = fidfile.ntraces

        # compressed format has one block per volume
        if nblocks == nvol_true and ntraces == nslice*n_pe:
            return "compressed"

        # uncompressed format has one block per slice
        elif nblocks == nvol_true*nslice and ntraces == n_pe:
            return "uncompressed"

        # epi2fid format has one block per phase encode (but in a weird order!)
        elif nblocks == nvol_true*nslice*n_pe_true and ntraces == 1:
            return "epi2fid"

        # asems_ncsnn format has one block per ???
        elif nblocks == nvol_true*n_pe and ntraces == nslice:
            return "asems_ncsnn"

        # asems_nccnn format has one block per ???
        elif nblocks == nvol_true*nslice*n_pe and ntraces == 1:
            return "asems_nccnn"

        else:
            raise "unrecognized fid format, (nblocks, ntraces) = (%d,%d)"%\
                  (nblocks, ntraces)

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

        # determine fid format
        fidformat = self._get_fidformat(fidfile)
        print "fidformat=",fidformat

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
          pulse_sequence not in ("gems", "mp_flash3d", "box3d_v2") and \
          (fidformat=="compressed" and not(pulse_sequence == "epi"\
           and self.spinecho) or \
          (fidformat=="uncompressed" and pulse_sequence != "epidw"))
        time_rev = n_fe_true - 1 - arange(n_fe_true)
        if time_reverse: print "time reversing"

        # determine if phase encodes need reordering
        needs_pe_reordering = fidformat not in ("asems_ncsnn", "asems_nccnn") \
                              and pulse_sequence not in ("box3d_v2")

        # load phase encode table
        if needs_pe_reordering: self._load_petable()

        for vol in range(nvol_true):

            # read the next image volume
            volume = volreader(fidfile, vol)


            # reverse ENTIRE negative-gradient read (??)
            if vol in self.ref_vols and vol==1:
                volume[:] = take(volume, time_rev, axis=(len(volume.shape)-1))

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

##             # Make a correction for mpflash data
##             if pulse_sequence == "mp_flash3d" and not self.flash_converted:
##                 nline = int(n_fe_true/20)
##                 scale = 2*nline*n_fe_true
##                 for slice in ksp_image:
##                     slice[:] = (slice - (sum(slice[:nline,:].flat) + \
## 			   sum(slice[-nline:,:].flat))/scale).astype(Complex32) 

            # assign volume to the appropriate output matrix
            if vol in self.ref_vols:
                self.ref_data[vol] = ksp_image
                self.ref_nav_data[vol] = navigators
            else:
                self.data[vol-numrefs] = ksp_image
                self.nav_data[vol-numrefs] = navigators

        self.setData(self.data)
