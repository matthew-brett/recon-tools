import struct
import stat
import os.path
from Numeric import empty
from pylab import (
  Complex32, Float32, Int16, Int32, pi, mlab, fft, fliplr, zeros, fromstring,
  reshape, arange, take, floor, argmax, multiply)
import file_io
import varian
from recon import petables
from recon.util import shift
from VolumeViewer import VolumeViewer

FIDL_FORMAT = "fidl"
VOXBO_FORMAT = "voxbo"
SPM_FORMAT = "spm"
MAGNITUDE_TYPE = "magnitude"
COMPLEX_TYPE = "complex"

#-----------------------------------------------------------------------------
def complex_fromstring(data, numtype):
    return fromstring(
      fromstring(data, numtype).byteswapped().astype(Float32).tostring(),
      Complex32)

##############################################################################
class FidImage (object):

    #-------------------------------------------------------------------------
    def __init__(self, datadir, options):
        self.options = options
        self.loadParams(os.path.join(datadir, "procpar"))
        self.loadData(os.path.join(datadir, "fid"))

    #-------------------------------------------------------------------------
    def logParams(self):
        "Report scan parameters to stdout."
        print "Phase encode table: ", self.petable_name
        print "Pulse sequence: %s" % self.pulse_sequence
        print "Number of volumes: %d" % self.nvol
        print "Number of slices: %d" % self.nslice
        print "Number of segments: %d" % self.nseg
        print "Number of navigator echoes per segment: %d" % self.nav_per_seg
        print "Number of phase encodes per slice (including navigators if present): %d" % self.n_pe
        print "Number of frequency encodes: %d" % self.n_fe_true
        print "Raw precision (bytes): ", self.datasize
        print "Number of volumes to skip: %d" %(self.nvol_true-self.nvol)
        print "Orientation: %s" % self.orient
        print "Pixel size (phase-encode direction): %7.2f" % self.xsize 
        print "Pixel size (frequency-encode direction): %7.2f" % self.ysize
        print "Slice thickness: %7.2f" % self.zsize

    #-------------------------------------------------------------------------
    def loadParams(self, procpar_file):
        """
        This method extracts scan related parameters from the procpar file.
        Some parameters may be overridden by the options from the command line
        and hence the options attribute is used here.
        """
        procpar = varian.procpar(procpar_file)
        self.n_fe = procpar.np[0]
        self.n_pe = procpar.nv[0]
        self.tr = procpar.tr[0]
        self.petable_name = procpar.petable[0]
        self.nvol_true = procpar.images[0]
        self.orient = procpar.orient[0]
        self.nvol = self.nvol_true - self.options.skip
        pulse_sequence = procpar.pslabel[0]
        
        if procpar.get("spinecho", ("",))[0] == "y":
            if pulse_sequence == 'epidw': pulse_sequence = 'epidw_se'
            else: pulse_sequence = 'epi_se'

        # try using procpar.cntr or procpar.image to know which volumes are reference
        if len(procpar.cntr) == self.nvol_true:
            is_imagevol = procpar.cntr
        elif len(procpar.image) == self.nvol_true:
            is_imagevol = procpar.image
        else: is_imagevol = [0] + [1]*nvol_true
        self.ref_vols = []
        self.image_vols = []
        for i, isimage in enumerate(is_imagevol):
            if isimage: self.image_vols.append(i)
            if not isimage: self.ref_vols.append(i)

        # determine number of slices, thickness, and gap
        if pulse_sequence == 'mp_flash3d':
            self.nslice = procpar.nv2[0]
            self.thk = 10.*procpar.lpe2[0]
            slice_gap = 0
        else:
            slice_positions = procpar.pss
            self.nslice = len(slice_positions)
            self.thk = procpar.thk[0]
            min = 10.*slice_positions[0]
            max = 10.*slice_positions[-1]
            nslice = self.nslice
            if nslice > 1:
                slice_gap = ((max - min + self.thk) - (nslice*self.thk))/(nslice - 1)
            else:
                slice_gap = 0

        # override tr with command-line option
        if self.options.TR > 0: self.tr = self.options.TR

        # Determine the number of navigator echoes per segment.
        # (Note: this CANNOT be the proper way to do this! -BH)
        self.nav_per_seg = self.n_pe%32 and 1 or 0

        # sequence-specific logic for determining pulse_sequence, petable and nseg
        # !!!!!! HEY BEN WHAT IS THE sparse SEQUENCE !!!!
        # Leon's "spare" sequence is really the EPI sequence with delay.
        if(pulse_sequence in ('epi','tepi','sparse','spare')):
            nseg = int(self.petable_name[-2])
        elif(pulse_sequence in ('epidw','Vsparse')):
            pulse_sequence = 'epidw'
            nseg = int(self.petable_name[-1])
        elif(pulse_sequence in ('epi_se','epidw_sb')):
            nseg = procpar.nseg[0]
            if self.petable_name.rfind('epidw') < 0:
                petable_name = "epi%dse%dk" % (self.n_pe, nseg)
            else:
                pulse_sequence = 'epidw'
        elif(pulse_sequence == 'asems'):
            nseg = 1
            petable = zeros(self.n_pe)
            for i in range(self.n_pe):      # !!!!! WHAT IS THIS TOO ? !!!!!
                if i%2:
                     petable[i] = self.n_pe - i/2 - 1
                else:
                     petable[i] = i/2
        else:
            print "Could not identify sequence: %s" % (pulse_sequence)
            sys.exit(1)

        self.nseg = nseg
        self.pulse_sequence = pulse_sequence

        # this quiet_interval may need to be added to tr in some way...
        #quiet_interval = procpar.get("dquiet", (0,))[0]
        self.n_fe_true = self.n_fe/2
        self.tr = nseg*self.tr
        self.nav_per_slice = nseg*self.nav_per_seg
        self.n_pe_true =  self.n_pe - self.nav_per_slice
        self.pe_per_seg = self.n_pe/nseg
        fov = procpar.lro[0]
        self.xsize = float(fov)/self.n_pe_true
        self.ysize = float(fov)/self.n_fe_true
        self.zsize = float(self.thk) + slice_gap

        self.datasize, self.raw_typecode = \
          procpar.dp[0]=="y" and (4, Int32) or (2, Int16)

        # calculate echo times
        f = 2.0*abs(procpar.gro[0])*procpar.trise[0]/procpar.gmax[0]
        self.echo_spacing = f + procpar.at[0]
        if(self.petable_name.find("alt") >= 0):
            self.echo_time = procpar.te[0] - f - procpar.at[0]
        else:
            self.echo_time = procpar.te[0] - floor(self.pe_per_seg)/2.0*self.echo_spacing
        self.pe_times = [self.echo_time + pe*self.echo_spacing \
                          for pe in range(self.pe_per_seg)]

    #-------------------------------------------------------------------------
    def _load_petable(self):
        """
        Read the phase-encode table and organize it into arrays which map k-space 
        line number (recon_epi convention) to the acquisition order. These arrays 
        will be used to read the data into the recon_epi as slices of k-space data. 
        Assumes navigator echo is aquired at the beginning of each segment
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

        petable_line = file(os.path.join(petables, self.petable_name))\
                       .readline().split()
        self.petable = zeros(n_pe_true*nslice).astype(int)
        i = 0 
        for slice in range(nslice):
            for pe in range(n_pe_true):
                seg = pe/pe_true_per_seg
                if pulse_sequence == 'epidw' or pulse_sequence == 'epidw_se':
                    offset = slice*n_pe + (seg+1)*nav_per_seg
                else:
                    offset = (slice + seg*nslice)*pe_per_seg \
                             - seg*pe_per_seg + (seg+1)*nav_per_seg
                # Find location of the pe'th phase-encode line in acquisition order.
                self.petable[i] = petable_line.index(str(pe)) + offset
                i += 1
            
        self.navtable = zeros(nav_per_slice*nslice).astype(int)
        if nav_per_seg != 0:    
            i = 0        
            for slice in range(nslice):
                for seg in range(nseg):
                    if pulse_sequence == 'epidw' or pulse_sequence == 'epidw_se':
                        offset = slice*n_pe + seg*pe_per_seg 
                    else:
                        offset = (slice + seg*nslice)*pe_per_seg
                    self.navtable[i] = offset
                    i += 1

    #-------------------------------------------------------------------------
    def _read_compressed_volume(self, fid, vol):
        block = fid.getBlock(vol)
        bias = complex(block.lvl, block.tlt)
        volume = complex_fromstring(block.getData(), self.raw_typecode)
        volume = (volume - bias).astype(Complex32)
        return reshape(volume, (self.nslice*self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def _read_uncompressed_volume(self, fid, vol):
        volume = empty((self.nslice, self.n_pe*self.n_fe_true), Complex32)
        for slice_num, slice in enumerate(volume):
            block = fid.getBlock(self.nslice*vol + slice_num)
            bias = complex(block.lvl, block.tlt)
            slice[:] = complex_fromstring(block.getData(), self.raw_typecode)
            slice[:] = (slice - bias).astype(Complex32)
        return reshape(volume, (self.nslice*self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def _read_epi2fid_volume(self, fid, vol):
        volume = empty(
          (self.nslice, self.nseg, self.pe_per_seg, self.n_fe_true),Complex32)
        pe_true_per_seg = self.pe_per_seg - self.nav_per_seg
        for seg in range(self.nseg):
            for slice in range(self.nslice):
                for pe in range(pe_true_per_seg):
                    block = fid.getBlock(
                      self.nslice*(self.nvol*(seg*pe_true_per_seg + pe) + vol)
                      + slice)
                    volume[slice, seg, self.nav_per_seg+pe, :] = \
                      complex_fromstring(block.getData(), self.raw_typecode)
        return reshape(volume, (self.nslice*self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def _read_asems_ncsnn_volume(self, fid, vol):
        volume = empty((self.nslice, self.n_pe, self.n_fe_true), Complex32)
        for pe in range(self.n_pe):
            block = fid.getBlock(pe*nvol + vol)
            bias = complex(block.lvl, block.tlt)
            for slice, trace in enumerate(block):
                trace = complex_fromstring(trace, self.raw_typecode)
                volume[slice,pe] = (trace - bias).astype(Complex32)
        return reshape(volume, (self.nslice*self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def _read_asems_nccnn_volume(self, fid, vol):
        volume = empty((self.nslice, self.n_pe, self.n_fe_true), Complex32)
        for pe in range(self.n_pe):
            for slice in range(self.nslice):
                block = fid.getBlock(((pe*self.nslice+slice)*self.nvol + vol))
                bias = complex(block.lvl, block.tlt)
                trace = complex_fromstring(block.getData(), self.raw_typecode)
                volume[slice,pe] = (trace - bias).astype(Complex32)
        return reshape(volume, (self.nslice*self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def loadData(self, fidfile):
        """
        This method reads the data from a fid file into following VarianData
        attributes: 

        data_matrix: A rank 4 array containing time-domain data. This array is 
          dimensioned as data_matrix(nvol,nslice,n_pe_true,n_fe_true) where nvol 
          is the number of volumes, nslice is the number of slices per volume,
          n_pe_true is the number of phase-encode lines and n_fe_true is the
          number read-out points.

        nav_data: A rank 4 array containing time-domain data for the navigator
          echoes of the image data. This array is dimensioned as 
          nav_data(nvol,nslice,nav_per_slice,n_fe_true).

        ref_data: A rank 4 array containing time-domain reference scan data 
          (phase-encode gradients are kept at zero). This array is dimensioned 
          as ref_data(nslice,n_pe_true,n_fe_true). 

        ref_nav_data: A rank 4 array containing time-domain data for the 
          navigator echoes of the reference scan data which is dimensioned as 
          ref_nav_data(nslice,nav_per_slice,n_fe_true).
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

        # allocate data matrices
        self.data_matrix = zeros(
          (nvol-numrefs, nslice, n_pe_true, n_fe_true), Complex32)
        self.nav_data = zeros(
          (nvol-numrefs, nslice, nav_per_slice, n_fe_true), Complex32)
        self.ref_data = zeros(
          (nslice, n_pe_true, n_fe_true), Complex32)
        self.ref_nav_data = zeros(
          (nslice, nav_per_slice, n_fe_true), Complex32)

        # open fid file
        fid = varian.fidfile(fidfile) 

        # determine fid type using fid file attributes nblocks and ntraces
        fidformat = {
          (nvol_true, nslice*n_pe):        "compressed",
          (nvol_true*nslice, n_pe):        "uncompressed",
          (nvol_true*nslice*n_pe_true, 1): "epi2fid",
          (nvol_true*n_pe, nslice):        "asems_ncsnn",
          (nvol_true*nslice*n_pe, 1):      "asems_nccnn"
        }.get( (fid.nblocks, fid.ntraces) )
        if fidformat is None:
          raise "unrecognized fid format, (nblocks, ntraces) = (%d,%d)"\
            (fid.nblocks, fid.ntraces)
        print "Fid Format:", fidformat

        # choose which method to use to read the volume based on the format
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
          (fidformat=="uncompressed" and pulse_sequence not in ("epidw", "epidw_se"))
        time_rev = n_fe_true - 1 - arange(n_fe_true)


        for vol in range(nvol_true-nvol, nvol_true):

            # read the next image volume
            volume = volreader(fid, vol)

            # time-reverse the data
            if time_reverse:
                f = self.pe_per_seg*nslice
                for seg in range(self.nseg):
                    base = seg*f
                    for pe in range(base+1, base+f, 2): 
                        volume[pe] = take(volume[pe], time_rev)

            # Reorder data according to phase encode table
            if fidformat != "epi2fid":
                self._load_petable()
                ksp_image = take(volume, self.petable)
                navigators = take(volume, self.navtable) 

            # reshape into final volume shape
            ksp_image = reshape(ksp_image, (nslice, n_pe_true, n_fe_true))
            navigators = reshape(navigators, (nslice, nav_per_slice, n_fe_true))

            # The time-reversal section above reflects the odd slices through
            # the pe-axis. This section puts all slices in the same orientation.
            if time_reverse:
                for slice in range(0,nslice,2):
                    for pe in range(n_pe_true):
                        ksp_image[slice,pe] = take(ksp_image[slice,pe], time_rev)
                    for pe in range(nav_per_slice):
                        navigators[slice,pe] = take(navigators[slice,pe], time_rev)

            # assign volume to the appropriate output matrix
            if vol in self.ref_vols:
                self.ref_data[:] = ksp_image
                self.ref_nav_data[:] = navigators
            else:
                self.data_matrix[vol-numrefs] = ksp_image
                self.nav_data[vol-numrefs] = navigators

        # shift data for easier fft'ing later
        shift(self.data_matrix, 0, n_fe_true/2)

    #-------------------------------------------------------------------------
    def save(self, outfile, file_format, data_type):
        """
        This function saves the image data to disk in a file specified on the command
        line. The file name is stored in the options dictionary using the outfile key.
        The output_data_type key in the options dictionary determines whether the data 
        is saved to disk as complex or magnitude data. By default the image data is 
        saved as magnitude data.
        """
        data_matrix = self.data_matrix
        VolumeViewer( abs(data_matrix), ("Time Point", "Slice", "Row", "Column"))

        n_pe_true = self.n_pe_true
        n_fe_true = self.n_fe_true
        nslice =  self.nslice
        xsize = self.xsize 
        ysize = self.ysize
        zsize = self.zsize

        if file_format == FIDL_FORMAT:
            f_img = open("%s.4dfp.img" % (outfile), "w")

        # Calculate proper scaling factor for SPM Analyze format by using the maximum value over all volumes.
        scl = 16383.0/abs(data_matrix).flat[argmax(abs(data_matrix).flat)]

        if data_type == MAGNITUDE_TYPE:
            hdr_datatype = "Short"
            vol_transformer = lambda v: multiply(scl, abs(v).astype(Float32))
        elif data_type == COMPLEX_TYPE:
            hdr_datatype = "Complex"
            vol_transformer = lambda v: v.astype(Complex32)

        # Save data to disk
        print "Saving to disk. Please Wait"
        for vol, volume in enumerate(data_matrix):
            if  file_format == SPM_FORMAT:
                outfile = "%s_%04d.img"%(outfile, vol)
                hdr = file_io.create_hdr(n_fe_true,n_pe_true,nslice,1,ysize,xsize,zsize,1.,0,0,0,hdr_datatype,64,1.,'analyze',outfile,0)
                file_io.write_analyze(outfile, hdr, vol_transformer(volume))

            elif file_format == FIDL_FORMAT:
                f_img.write(vol_transformer(volume).byteswapped().tostring())

            # !!!!!!!!! WHERE IS THE CODE TO SAVE IN VOXBO FORMAT !!!!!!!!
