"A class for interpreting image data in FID format"
import os.path
import sys
import numpy as N
from recon.util import euler2quat, qmult, eulerRot, Quaternion, reverse
from recon.scanners import ScannerImage
from recon.scanners.varian import tablib
from recon.scanners.varian.ProcPar import ProcPar, ProcParImageMixin
from recon.scanners.varian.FidFile import FidFile


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
        return N.fromstring(
            N.fromstring(data,numtype).byteswap().astype(N.float32).tostring(),
            N.complex64)
    else:
        return N.fromstring(
	    N.fromstring(data,numtype).astype(N.float32).tostring(),
            N.complex64)
    
##############################################################################
class FidImage (ScannerImage, ProcParImageMixin):

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
    def __init__(self, datadir, vrange=None, target_dtype=N.complex64):
        ProcParImageMixin.__init__(self, datadir, vrange=vrange)
        self.path = datadir
        self.initializeData()
        self.loadData(datadir)
        self.realizeOrientation()
        # scanner image will init ReconImage, which will set the data dims
        ScannerImage.__init__(self)
        ref_path = self.path.replace("data.fid", "ref_2.fid")
        if ref_path is not self.path and os.path.exists(ref_path):
            ref_fid = FidImage(ref_path)
            self.ref_data = N.array([self.ref_data[0], ref_fid.ref_data[0]])
        
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
        print "Pixel size (phase-encode direction): %7.2f" % self.dPE 
        print "Pixel size (frequency-encode direction): %7.2f" % self.dFE
        print "Slice thickness: %7.2f" % self.dSL

    #-------------------------------------------------------------------------
    def initializeData(self):
        "Allocate data matrices." # IF NEEDED! look at petable or something
        nrefs = len(self.ref_vols)
        self.data = N.zeros((self.nvol, self.nslice,
                             self.n_pe_true, self.n_fe_true), N.complex64)
        self.nav_data = N.zeros((self.nvol, self.nslice,
                                 self.nav_per_slice, self.n_fe_true),
                                N.complex64)
        self.ref_data = N.zeros((nrefs, self.nslice,
                                 self.n_pe_true, self.n_fe_true), N.complex64)
        self.ref_nav_data = N.zeros((nrefs, self.nslice,
                                     self.nav_per_slice, self.n_fe_true),
                                    N.complex64)
        
    #-------------------------------------------------------------------------
    def realizeOrientation(self):
        "Set up the orientation transform defined in the procpar"
        # Varian data is layed out with this improper rotation from
        # neurological oriented space:
        # from scanner to image-> map -y -> +y
        # (the object lies [+X, +Y, +Z] = [P, R, S] inside the scanner,
        #  so account for this rotation last)
        Qs_mat = N.asarray([[ 1., 0., 0.],
                            [ 0.,-1., 0.],
                            [ 0., 0., 1.],])
        Qr_mat = eulerRot(phi=N.pi/2)
        Qscanner = Quaternion(M=Qs_mat)
        Qrot = Quaternion(M=Qr_mat)

        phi,psi,theta = self.phi, self.psi, self.theta
        phi,theta,psi = map(lambda x: (N.pi/2)*int((x+N.sign(x)*45.)/90),
                                                 (phi,theta,psi))
        # find out the (closest) image plane
        if theta == 0 and psi == 0:
            self.image_plane = "axial"
        elif theta == N.pi/2 and psi != N.pi/2:
            self.image_plane = "coronal"
        elif theta == N.pi/2 and psi == N.pi/2:
            self.image_plane = "sagittal"
        else:
            self.image_plane = "oblique"
        # Varian system [R A S] vector seems to be [Y -X Z], and
        # the order of rotation in that space is
        # 1) theta degrees clockwise around Y
        # 2) psi degrees around X
        # 3) phi degrees around Z
        # The steps to rotate the reconstructed data are:
        # 1) flip Y to put into a right-handed system
        # 2) rotate in the reverse order as the Varian rotation
        # 3) make a final pi/2 rotation around Z to make [R A S] = [X Y Z]
        Qobl = Quaternion(M=N.dot(eulerRot(psi=theta),
                                  N.dot(eulerRot(theta=psi),
                                        eulerRot(phi=phi))))
        
        self.orientation_xform = qmult(Qrot, qmult(Qobl,Qscanner))
        # don't have orientation's name yet
        self.orientation = ""
            
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
        n_pe_true = self.n_pe_true
        table_filename = os.path.join(tablib, self.petable_name)
        # line is going to look like [0 2 4 ... 32 1 3 5 ... 63]
        # (eg: acquisition scheme for interleaved data)
        # want to return a list of integers that indexes ordered numbers
        # from the above list: [0 32 1 33 2 34 ... ]
        line = file(table_filename).readline().split()
        line = N.array([int(i) for i in line])
        all_pe = N.arange(n_pe_true)
        self.petable = N.array([(line==p).nonzero()[0][0] for p in all_pe])
##         if nav_per_seg:
##             # if line was [0 2 4 ... 32 1 3 5 ... 63] but there are
##             # navigators, the image data are in [1 3 5 ... 33 2 4 6 ... 64]
##             # (due to navigators being in rows 0 and 32)
##             # this adjustment adds the appropriate offset to each index
##             pe_per_seg = n_pe_true/nseg
##             offset = [N.repeat(n*nav_per_seg, pe_per_seg)
##                       for n in range(1,nseg+1)]
##             offset = N.concatenate(offset)
##             self.petable += offset[self.petable]

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
        volume = complex_fromstring(block.getData(), self.raw_dtype)
        volume = (volume - bias).astype(N.complex64)
        return N.reshape(volume, (self.nslice, self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def _read_uncompressed_volume(self, fidfile, vol):
        """
        Reads one volume from an uncompressed FID file.
        @return: block of data with shape (nslice*n_pe, n_fe_true)
        """        
        volume = N.empty((self.nslice, self.n_pe*self.n_fe_true), N.complex64)
        for sl_num, sl in enumerate(volume):
            block = fidfile.getBlock(self.nslice*vol + sl_num)
            bias = complex(block.lvl, block.tlt)
            sl[:] = complex_fromstring(block.getData(), self.raw_dtype)
            sl[:] = (sl - bias)
        return N.reshape(volume, (self.nslice, self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def _read_epi2fid_volume(self, fidfile, vol):
        """
        Reads one volume from an epi2fid FID file.
        @return: block of data with shape (nslice*n_pe, n_fe_true)
        """
        # procpar indicates navigator lines, but none are read in? huh
        volume = N.empty(
            (self.nslice, self.nseg, self.pe_per_seg, self.n_fe_true),N.complex64)
        pe_true_per_seg = self.pe_per_seg - self.nav_per_seg
        for seg in range(self.nseg):
            for sl in range(self.nslice):
                for pe in range(pe_true_per_seg):
                    block = fidfile.getBlock(
                      self.nslice*(self.nvol_true*(seg*pe_true_per_seg + pe)\
                      + vol) + sl)
                    volume[sl, seg, self.nav_per_seg+pe, :] = \
                      complex_fromstring(block.getData(), self.raw_dtype) 
        return N.reshape(volume, (self.nslice, self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def _read_asems_ncsnn_volume(self, fidfile, vol):
        """
        Reads one volume from an asems_ncsnn FID file.
        @return: block of data with shape (nslice*n_pe, n_fe_true)
        """
        volume = N.empty((self.nslice, self.n_pe, self.n_fe_true), N.complex64)
        for pe in range(self.n_pe):
            # should change to nvol_true?
            block = fidfile.getBlock(pe*self.nvol_true + vol)
            bias = complex(block.lvl, block.tlt)
            for sl, trace in enumerate(block):
                trace = complex_fromstring(trace, self.raw_dtype)
                if self.pulse_sequence == "gems" and self.n_transients>1:
                    volume[sl,pe] = trace
                else:
                    volume[sl,pe] = (trace - bias).astype(N.complex64)

        return volume

    #-------------------------------------------------------------------------
    def _read_asems_nccnn_volume(self, fidfile, vol):
        """
        Reads one volume from an asems_nccnn FID file.
        @return: block of data with shape (nslice*n_pe, n_fe_true)
        """
        volume = N.empty((self.nslice, self.n_pe, self.n_fe_true), N.complex64)
        for pe in range(self.n_pe):
            for sl in range(self.nslice):
                # should change to nvol_true?
                block = fidfile.getBlock(
                  ((pe*self.nslice+sl)*self.nvol_true + vol))
                bias = complex(block.lvl, block.tlt)
                trace = complex_fromstring(block.getData(), self.raw_dtype)
                volume[sl,pe] = (trace - bias).astype(N.complex64)
        return volume

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

        # asems_ncsnn format has one block per phase-encode per volume
        elif nblocks == nvol_true*n_pe and ntraces == nslice:
            return "asems_ncsnn"

        # asems_nccnn format has one block per phase-encode per slice per volume
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
        n_pe_true = self.n_pe_true
        n_pe = self.n_pe
        nav_per_seg = self.nav_per_seg
        n_fe = self.n_fe  
        n_fe_true = self.n_fe_true
        pulse_sequence = self.pulse_sequence
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

        time_reverse = pulse_sequence in ("epi","epidw","testbrs2") and \
                       fidformat == "compressed"

        #time_rev = n_fe_true - 1 - N.arange(n_fe_true)
        if time_reverse: print "time reversing"

        # determine if phase encodes need reordering
        needs_pe_reordering = fidformat not in ("asems_ncsnn", "asems_nccnn") \
                              and self.nseg > 1

        # load phase encode table
        if needs_pe_reordering: self._load_petable()

        for vnum, vol in enumerate(self.ref_vols+self.image_vols):
            # read the next image volume
            volume = volreader(fidfile, vol)

            (data,navdata,vidx) = vol in self.ref_vols and \
                                  (self.ref_data,self.ref_nav_data,vnum) or \
                                  (self.data,self.nav_data,vnum-numrefs)


            # strip off data into appropriate arrays
            # if nav_per_seg, navigator line is the 1st in a seg
            if nav_per_seg:
                nav_lines = range(0, n_pe, n_pe/self.nseg)
                vol_lines = range(0, n_pe)
                for nav_pt in nav_lines: vol_lines.remove(nav_pt)
                navdata[vidx] = N.take(volume, nav_lines, axis=-2)
                volume = N.take(volume, vol_lines, axis=-2)

            if time_reverse:
                volume[:,1::2,:] = reverse(volume[:,1::2,:], axis=-1)
            
            if needs_pe_reordering:
                data[vidx] = N.take(volume, self.petable, axis=-2)
            else:
                data[vidx] = volume
            
        
        
        # squeeze out single volume data
        self.data = N.squeeze(self.data)

