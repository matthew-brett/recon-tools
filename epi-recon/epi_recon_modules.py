import sys
import string 
import os
from Numeric import *
import file_io
import FFT
import idl
import MLab
import struct
import math
import gc
import math_bic
from optparse import OptionParser, Option

FIDL_FORMAT = "fidl"
VOXBO_FORMAT = "voxbo"
SPM_FORMAT = "spm"
MAGNITUDE_TYPE = "magnitude"
COMPLEX_TYPE = "complex"

output_format_choices = (FIDL_FORMAT, VOXBO_FORMAT, SPM_FORMAT)
output_datatype_choices= (MAGNITUDE_TYPE, COMPLEX_TYPE)


#*****************************************************************************
class Operation (object):
    "Base class for data operations.  Doesn't do a whole lot right now."
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


#*****************************************************************************
class OperationManager (object):
    """
    This class is responsible for knowing which operations are available
    and retrieving them by name.  It should be a global singleton in the
    system. (Preferrably, an attribute of the EpiRecon tool class, once
    that class is implemented.)
    """

    # Exceptions used to indicate problems resolving Operations
    class InvalidOperationName (Exception): pass
    class ArgumentError (Exception): pass

    _operation_names = (
      "PhaseCorrelation",
      "fermi_filter",
      "fft_data")

    def getOperationNames(self):
        "@return list of valid operation names."
        return self._operation_names   

    def getOperation(self, opname):
        "@return the operation for the given name"
        return globals().get(opname, None)

    def resolveOperationSpec(self, opspec):
        """
        @return pair representing the operation and its args: (op, args)
        @param opspec: a string like 'opname: arg1=val1; arg2=val2; ...'
        """
        parts = opspec.split(":", 1)

        # resolve operation by name
        operation = self.getOperation(parts[0].strip())
        if not operation:
            raise self.InvalidOperationName(
              "Couldn't resolve operation name '%s'."%parts[0].strip())

        # parse operation arguments
        args = {}
        if len(parts) > 1:
            argstr = parts[1].strip()
            try:
                exec argstr in {}, args
            except:
                raise ArgumentError(
                  "Could not parse operation arguments '%s'"%argstr)

        return (operation, args)


#*****************************************************************************
class EpiReconOptionParser (OptionParser):
    "Parse command-line arguments to the epi_recon tool."

    #-------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        OptionParser.__init__(self, *args, **kwargs)
        self._opmanager = OperationManager()
        opnames = self._opmanager.getOperationNames()
        self.set_usage("usage: %prog [options] fid_file procpar output_image")
        self.add_options((
          Option( "", "--operation", dest="operations", action="append",
            metavar="<opspec>", help=\
            "A specification of an operation to perform, looking like: "\
            "'opname: arg=val; arg=val;'.  This option may be given multiple "\
            "times, the specified operations will be executed in the given "\
            "order.  Available operation names are {%s}.  If this option is "\
            "not given, all available operations will be performed in the "\
            "order shown with their default argument values."%"|".join(opnames)),
          Option( "-n", "--nvol", dest="nvol_to_read", type="int", default=0,
            action="store",
            help="Number of volumes within run to reconstruct." ),
          Option( "-s", "--frames-to-skip", dest="skip", type="int", default=0,
            action="store",
            help="Number of frames to skip at beginning of run." ),
          Option( "-f", "--file-format", dest="file_format", action="store",
            type="choice", default=FIDL_FORMAT, choices=output_format_choices,
            help="""{%s}
            fidl: save floating point file with interfile and 4D analyze headers.
            spm: Save individual image for each frame in analyze format.
            voxbo: Save in tes format."""%("|".join(output_format_choices)) ),
          Option( "-p", "--phs-corr", dest="phs_corr", default="", action="store",
            help="Dan, please describe the action of this option..."),
          Option( "-a", "--save-first", dest="save_first", action="store_true",
            help="Save first frame in file named 'EPIs.cub'." ),
          #Option( "-g", "--ignore-nav-echo", dest="ignore_nav", action="store_true",
          #  help="Do not use navigator in phase correction." ),
          Option( "-t", "--tr", dest="TR", type="float", action="store",
            help="Use the TR given here rather than the one in the procpar." ),
          Option( "-x", "--starting-frame-number", dest="sfn", type="int",
            default=0, action="store", metavar="<starting frame number>",
            help="Specify starting frame number for analyze format output." ),
          Option( "-b", "--flip-top-bottom", dest="flip_top_bottom",
            action="store_true", help="Flip image about the  horizontal axis." ),
          Option( "-l", "--flip-left-right", dest="flip_left_right",
            action="store_true", help="Flip image about the vertical axis." ),
          Option( "-q", "--flip-slices", dest="flip_slices", action="store_true",
            help="Reorders slices." ),
          Option( "-y", "--output-data-type", dest="output_data_type",
            type="choice", default=MAGNITUDE_TYPE, action="store",
            choices=output_datatype_choices,
            help="""{%s}
            Specifies whether output images should contain only magnitude or
            both the real and imaginary components (only valid for analyze
            format)."""%("|".join(output_datatype_choices)) )
        ))

    #-------------------------------------------------------------------------
    def resolveOperations(self, opspecs):
        """
        @return a list of operation pairs (operation, args).
        @param opspecs: list of operation specifiers.
        """
        operations = []
        for opspec in opspecs:
            operations.append(self._opmanager.resolveOperationSpec(opspec))
        return operations

    #-------------------------------------------------------------------------
    def getOptions(self):
        """
        Bundle command-line arguments and options into a single options
        object, including a resolved list of callable data operations.
        """
        options, args = self.parse_args()
        if len(args) != 3: self.error(
          "Expecting 3 arguments: fid_file procpar_file img_file" )

        # treat the raw args as named options
        options.fid_file, options.procpar_file, options.img_file = args

        # resolve operations
        options.operations = self.resolveOperations(options.operations)

        # is this necessary?
        #if options.save_first: options.ignore_nav = true

        return options


#*****************************************************************************
def get_params(options):
    """
    This function parses the procpar file and returns parameter values related
    to the scan. Some parameters may be overridden by the options from the
    command line and hence the "options" list is an argument to this function.
    """
    params = {}       # Create empty dictionary.
    infile = open(options.procpar_file,'r')
    nvol_images = '0'
    nvol_image = '0'
    lc_spin = ''
    while 1:
        line = infile.readline()
        if(line == ''):
            break
        keywords = string.split(line)
        if (keywords[0] == 'thk'):
            line = infile.readline()
            words = string.split(line)
            thk = words[1]
            params['thk'] = string.atof(thk)
        elif (keywords[0] + " 3" == "te 3"):
            line = infile.readline()
            words = string.split(line)
            te = words[1]
            params['te'] = te
        elif (keywords[0] == 'pss'):
            line = infile.readline()
            words = string.split(line)
            nslice = words[0]
            len = int(nslice)
            pss = words[1:]
            params['pss'] = pss
            params['nslice'] = string.atoi(nslice)
        elif (keywords[0] == 'lro'):
            line = infile.readline()
            words = string.split(line)
            ffov = 10.*float(words[1])
            fov = "%f" % ffov
            params['fov'] = fov
        elif (keywords[0] == 'lpe2'):
            line = infile.readline()
            words = string.split(line)
            flpe2 = 10.*float(words[1])
            lpe2 = "%f" % flpe2
            params['lpe2'] = lpe2
        elif (keywords[0] == 'dquiet'):
            line = infile.readline()
            words = string.split(line)
            fquiet_interval = float(words[1])
            quiet_interval = "%f" % fquiet_interval
            params['quiet_interval'] = quiet_interval
        elif (keywords[0] == 'np'):
            line = infile.readline()
            words = string.split(line)
            np = words[1]
            params['n_fe'] = string.atoi(np)
        elif (keywords[0] == 'nseg'):
            line = infile.readline()
            words = string.split(line)
            nseg = words[1]
            params['nseg'] = nseg
        elif (keywords[0] == 'dp'):
            line = infile.readline()
            words = string.split(line)
            dp = words[1]
            params['dp'] = dp
        elif (keywords[0] == 'nv'):
            line = infile.readline()
            words = string.split(line)
            nv = words[1]
            params['n_pe'] = string.atoi(nv)
        elif (keywords[0] == 'nv2'):
            line = infile.readline()
            words = string.split(line)
            nv2 = words[1]
            params['nv2'] = nv2
        elif (keywords[0] == 'pslabel'):
            # Pulse sequence label
            line = infile.readline()
            words = string.split(line)
            pulse_sequence = words[1][1:-1]
            if pulse_sequence == 'spare':
#               Leon's "spare" sequence is really the EPI sequence with delay.
                pulse_sequence = "epi" 
            params['pulse_sequence'] = pulse_sequence
        elif (keywords[0] == 'tr'):
            line = infile.readline()
            words = string.split(line)
            tr = words[1]
            params['tr'] = tr
        elif (keywords[0] == 'spinecho'):
            line = infile.readline()
            words = string.split(line)
            lc_spin = words[1]
        elif (keywords[0]+" 1" == 'at 1'):
            line = infile.readline()
            words = string.split(line)
            at = words[1]
            params['at'] = at
        elif (keywords[0]+" " == 'gmax '):
            line = infile.readline()
            words = string.split(line)
            gmax = words[1]
            params['gmax'] = gmax
        elif (keywords[0]+" " == 'gro '):
            line = infile.readline()
            words = string.split(line)
            gro = words[1]
            params['gro'] = gro
        elif (keywords[0]+" " == 'trise '):
            line = infile.readline()
            words = string.split(line)
            trise  = words[1]
            params['trise'] = trise 
        elif (keywords[0] == 'petable'):
            line = infile.readline()
            words = string.split(line)
            petable = words[1]
            params['petable'] = petable[1:-1]
        elif (keywords[0] == 'cntr'):
            line = infile.readline()
            words = string.split(line)
            nvol_images = words[0]
            params['nvol'] = nvol_images
        elif (keywords[0] == 'image'):
            line = infile.readline()
            words = string.split(line)
            nvol_image = words[0]
        elif (keywords[0] == 'images'):
            line = infile.readline()
            words = string.split(line)
            nvol_images = words[1]
        elif (keywords[0] == 'orient'):
            line = infile.readline()
            words = string.split(line)
            orient = words[1]
            params['orient'] = orient
    infile.close()

    if "y" in lc_spin:
        if params['pulse_sequence'] == 'epidw':
            params['pulse_sequence'] = 'epidw_se'
        else:
            params['pulse_sequence'] = 'epi_se'

    if pulse_sequence == 'mp_flash3d':
        params['nslice'] = params['nv2']
        ns = int(params['nslice'])
        params['thk'] = float(params['lpe2'])
        gap = 0
    else:
        slice_pos = params['pss']
        th = float(thk)
        ns = int(nslice)
        min = 10.*float(slice_pos[0])
        max = 10.*float(slice_pos[ns-1])
        if ns > 1:
            gap = ((max - min + th) - (ns*th))/(ns - 1)
        else:
            gap = 0
    params['gap'] = gap

#   Coding of number of frames varies with the sequence.
    if nvol_images > 0:
#       This is the most reliable.
        nvol = nvol_images
    else:
        nvol = nvol_image
    params['nvol'] = nvol
    params['nvol'] = string.atoi(params['nvol'])


    if options.TR > 0:
        params['tr'] = options.TR
    params['n_fe_true'] = params['n_fe']/2
    n_pe = params['n_pe']
    n_fe = params['n_fe'] 


    # Determine the number of navigator echoes per segment.
    if params['n_pe'] % 32:
        params['nav_per_seg'] = 1
    else:
        params['nav_per_seg'] = 0

    if(params.has_key('nvol') == 0):
        params['nvol'] = 1

    if(options.nvol_to_read > 0):
        params['nvol'] = options.nvol_to_read
    else:
        params['nvol'] = params['nvol'] - options.skip


    if(pulse_sequence == 'epi' or pulse_sequence == 'tepi'):
        pulse_sequence = 'epi'
        nseg = string.atoi(params['petable'][-2])
    elif(pulse_sequence == 'epidw' or pulse_sequence == 'Vsparse'):
        pulse_sequence = 'epidw'
        nseg = string.atoi(params['petable'][-1])
        if(params.has_key('dquiet')):
            TR = TR + string.atof(params['quiet_interval'])
    elif(pulse_sequence == 'epi_se'):
        if string.rfind(params['petable'],'epidw') < 0:
            petable = "epi%dse%dk" % (n_pe,nseg)
        else:
            pulse_sequence = 'epidw'
        nseg = string.atoi(params['nseg'])
    elif(pulse_sequence == 'epidw_sb'):
        if string.rfind(params['petable'],'epidw') < 0:
            petable = "epi%dse%dk" % (n_pe,nseg)
        else:
            pulse_sequence = 'epidw'
        nseg = string.atoi(params['nseg'])
    elif(pulse_sequence == 'epidw_se'):
        nseg = string.atoi(params['nseg'])
    elif(pulse_sequence == 'asems'):
        nseg = 1
        petable = "epi%dse%dk" % (n_pe,nseg)
        petable = "64alt"
        params['petable'] = petable
        params['petable'] = ""  #!!! WHAT? OBTAIN petable THEN JUST SET TO "" ? !!!
        petab = zeros(n_pe)
        for i in range(n_pe):      # !!!!! WHAT IS THIS TOO ? !!!!!
            if i%2:
                 petab[i] = n_pe - i/2 - 1
            else:
                 petab[i] = i/2
    elif(pulse_sequence == 'sparse'): # !!!!!! HEY BEN WHAT IS THE sparse SEQUENCE !!!!
        nseg = string.atoi(params['petable'][-2])
        pulse_sequence = 'epi'
        TR = TR + string.atof(params['quiet_interval'])
    else:
        print "Could not identify sequence: %s" % (pulse_sequence)
        sys.exit(1)

    params['tr'] = nseg*string.atof(params['tr'])
    params['nseg'] = nseg
    params['n_pe_true'] =  params['n_pe'] - nseg*params['nav_per_seg']
    params['xsize'] = string.atof(params['fov'])/params['n_pe_true']
    params['ysize'] = string.atof(params['fov'])/params['n_fe_true']
    params['zsize'] = string.atof(params['thk']) + string.atof(params['gap'])

    if(params['dp'] == '"y"'):
        params['datasize'] = 4
        params['num_type'] = Int32
    else:
        params['datasize'] = 2
        params['num_type'] = Int16
    return params


#*****************************************************************************
def initialize_data(params):
    "Allocate and initialize data arrays."
    nvol = params['nvol']
    nslice = params['nslice']
    n_nav = params['nseg']*params['nav_per_seg']
    n_pe_true = params['n_pe_true']
    n_fe_true = params['n_fe_true']
    class Data (object): pass
    data = Data()
    data.data_matrix = zeros((nvol, nslice, n_pe_true, n_fe_true)).astype(Complex32)
    data.nav_data = zeros((nvol, nslice, n_nav, n_fe_true)).astype(Complex32)
    data.ref_data = zeros((nslice, n_pe_true, n_fe_true)).astype(Complex32)
    data.ref_nav_data = zeros((nslice, n_nav, n_fe_true)).astype(Complex32)
    return data


#*****************************************************************************
def get_data(params, options):
    """
    This function reads the data from a fid file into an object with the
    following attributes: 

    data_matrix: A rank 4 array containing time-domain data. This array is 
      dimensioned as data_matrix(nvol,nslice,n_pe_true,n_fe_true) where nvol 
      is the number of volumes, nslice is the number of slices per volume,
      n_pe_true is the number of phase-encode lines and n_fe_true is the
      number read-out points. Indices begun at 1. !!!! THE RANGE FOR THE PE
      ENCODES IS ..... (do similar stuff below) !!!!

    nav_data: A rank 4 array containing time-domain data for the navigator
      echoes of the image data. This array is dimensioned as 
      nav_data(nvol,nslice,n_nav,n_fe_true) where n_nav is the 
      number navigator echoes per slice. Indices begun at 1.

    ref_data: A rank 4 array containing time-domain reference scan data 
      (phase-encode gradients are kept at zero). This array is dimensioned 
      as ref_data(nslice,n_pe_true,n_fe_true). 

    ref_nav_data: A rank 4 array containing time-domain data for the 
      navigator echoes of the reference scan data which is dimensioned as 
      ref_nav_data(nslice,n_nav,n_fe_true) where n_nav is the number 
      navigator echoes per slice. Indices begun at 1.
    """
    data = initialize_data(params)
    data_matrix = data.data_matrix
    nav_data = data.nav_data
    ref_data = data.ref_data
    ref_nav_data = data.ref_nav_data
    nav_per_seg = params['nav_per_seg']
    nseg = params['nseg']
    n_nav = nseg*nav_per_seg
    n_pe = params['n_pe']
    pe_per_seg = n_pe/nseg
    n_pe_true = params['n_pe_true']
    pe_true_per_seg = n_pe_true/nseg
    n_fe = params['n_fe']  
    n_fe_true = params['n_fe_true']
    nslice =  params['nslice']
    ydim = n_fe_true
    xdim = n_pe_true
    pulse_sequence = params['pulse_sequence']
    nvol = params['nvol']
    num_type = params['num_type']
    datasize = params['datasize']
    main_hdr = 32
    sub_hdr = 28
    line_len_data = datasize*n_fe
    line_len = datasize*n_fe + sub_hdr
    slice_len = n_fe*n_pe*datasize

    # Compute the lengths of the different FID file according to format.
    len_asems_ncsnn = main_hdr + nvol*n_pe*(sub_hdr + nslice*n_fe*datasize)
    len_asems_nccnn = main_hdr + nvol*n_pe*nslice*(sub_hdr + n_fe*datasize)
    len_compressed = main_hdr + nvol*(sub_hdr + nslice*n_fe*n_pe*datasize)
    len_uncompressed = main_hdr + nvol*nslice*(sub_hdr + slice_len)
    len_epi2fid = main_hdr + nvol*nslice*(n_pe - nseg)*line_len

    # Open the actual FID file to determine its length
    f_fid = open(options.fid_file,"r")
    f_fid.seek(0,2)
    file_length = f_fid.tell()
    f_fid.close()

    # Determine the file format by comparing the computed and actual file lengths.
    if len_compressed == file_length:
        fid_type = "compressed"
    elif len_uncompressed == file_length:
        fid_type = "uncompressed"
    elif len_epi2fid == file_length:
        fid_type = "epi2fid"
    elif len_asems_ncsnn == file_length:
        fid_type = "asems_ncsnn"
    elif len_asems_nccnn == file_length:
        fid_type = "asems_nccnn"
    else:
        print "Cannot recognize fid format, exiting."
        sys.exit(1)
    print "FID file format: ", fid_type, "\n"


    # Read the phase-encode table and organize it into arrays which map k-space 
    # line number (recon_epi convention) to the acquisition order. These arrays 
    # will be used to read the data into the recon_epi as slices of k-space data. 
    # Assumes navigator echo is aquired at the beginning of each segment
    petab = zeros(n_pe_true*nslice).astype(int)
    navtab = zeros(nav_per_seg*nseg*nslice).astype(int)
    petable_file = "recon_epi_pettabs" + "/" + params['petable']
    f_pe = open(petable_file)   # Open recon_epi petable file               
    petable_lines = f_pe.readlines()
    petable_lines[0] = string.split(petable_lines[0])
    index_1 = 0 
    for slice in range(nslice):
        for pe in range(n_pe_true):
            seg = pe/pe_true_per_seg
            if pulse_sequence == 'epidw' or pulse_sequence == 'epidw_se':
                offset = slice*n_pe + (seg+1)*nav_per_seg
            else:
                offset = (slice + seg*nslice)*pe_per_seg - seg*pe_per_seg + (seg+1)*nav_per_seg
            # Find location of the pe'th phase-encode line in acquisition order.
            petab[index_1] = petable_lines[0].index(str(pe)) + offset
            index_1 = index_1 + 1
        
    if nav_per_seg != 0:    
        index_2 = 0        
        for slice in range(nslice):
            for seg in range(nseg):
                if pulse_sequence == 'epidw' or pulse_sequence == 'epidw_se':
                    offset = slice*n_pe + seg*pe_per_seg 
                else:
                    offset = (slice + seg*nslice)*pe_per_seg
                navtab[index_2] = offset
                index_2 = index_2 + 1

    f_pe.close()


    # Open FID file.
    f_fid = open(options.fid_file,"r") 

    # We use no data within the main header. Move the current file position past it.
    f_fid.seek(main_hdr)

    frame = options.sfn - 1
    ref_vol = 0
    ksp_vol = 0
    for vol in range(nvol):
        frame = frame + 1
        if frame == 1 and options.skip > 0: #!!!! NEED TO TEST THIS SECTION !!!!
            # Skip data.
            if fid_type == 'compressed':
                # Skip phase data and skip blocks.
                pos = skip*(sub_hdr + nslice*n_fe*n_pe*datasize)
            elif fid_type == 'uncompressed':
                pos = skip*nslice*(sub_hdr + slice_len)
            elif fid_type == 'epi2fid':
                pos = skip*nslice*(n_pe_true)*(sub_hdr + line_len_data)
            elif fid_type == 'asems_ncsnn':
                pos = 0
            else:
                print "Unsupported pulse sequence."
                sys.exit(1)
            f_fid.seek(pos,1)

        # Read the next block according to the file format.
        bias = zeros(nslice).astype(Complex32)
        if fid_type == "compressed":
            block_len = nslice*n_fe*n_pe*datasize
            shdr = struct.unpack('>hhhhlffff',f_fid.read(sub_hdr))
            bias[:] = complex(shdr[7],shdr[8])
            blk = fromstring(f_fid.read(block_len),num_type).byteswapped().astype(Float32).tostring()
            blk = fromstring(blk,Complex32)
            blk = reshape(blk,(nslice*n_pe,n_fe_true))
            if pulse_sequence == 'epi_se':
                time_reverse = 0
            else:
                time_reverse = 1
        elif fid_type == "uncompressed":
            blk = zeros((nslice,n_pe*n_fe_true)).astype(Complex32)
            for slice in range(nslice):
                shdr = struct.unpack('>hhhhlffff',f_fid.read(sub_hdr))
                bias[slice] = complex(shdr[7],shdr[8])
                blk_slc = fromstring(f_fid.read(slice_len),num_type).byteswapped().astype(Float32).tostring()
                blk[slice,:] = fromstring(blk_slc,Complex32)
            blk = reshape(blk,(nslice*n_pe,n_fe_true))
            if pulse_sequence == 'epidw' or pulse_sequence == 'epidw_se':
                time_reverse = 0
            else:
                time_reverse = 1
        elif fid_type == "epi2fid":
            blk = zeros((nslice,nseg,n_pe/nseg,n_fe_true)).astype(Complex32)
            for seg in range(nseg):
                for slice in range(nslice):
                    for pe in range(n_pe/nseg - nav_per_seg):
                        position = ((pe + seg*(n_pe/nseg - nav_per_seg))*nvol*nslice + vol*nslice + slice)*line_len + main_hdr
                        f_fid.seek(position,0)
                        shdr = struct.unpack('>hhhhlffff',f_fid.read(sub_hdr))
                        bias[slice] = complex(0.,0.)
                        blk_line = fromstring(f_fid.read(line_len_data),num_type).byteswapped().astype(Float32).tostring()
                        blk_line = fromstring(blk_line,Complex32)
                        blk[slice,seg,pe + nav_per_seg,:] = blk_line
            blk = reshape(blk,(nslice*n_pe,n_fe_true))
            time_reverse = 0
        elif fid_type == "asems_ncsnn":
            blk = zeros((nslice,n_pe,n_fe_true)).astype(Complex32)
            for pe in range(n_pe):
                shdr = struct.unpack('>hhhhlffff',f_fid.read(sub_hdr))
                bias1 = complex(shdr[7],shdr[8])
                for slice in range(nslice):
                    # if weird_asems_ncsnn:
                    position = (pe*nvol+vol)*(nslice*line_len_data + sub_hdr) + slice*line_len_data + sub_hdr + main_hdr
                    f_fid.seek(position,0)
                    blk_line = fromstring(f_fid.read(line_len_data),num_type).byteswapped().astype(Float32).tostring()
                    blk_line = fromstring(blk_line,Complex32)
                    bias[slice] = complex(0.,0.)
                    blk[slice,pe,:] = (blk_line - bias1).astype(Complex32)
            blk = reshape(blk,(nslice*n_pe,n_fe_true))
            time_reverse = 0
        elif fid_type == "asems_nccnn":
            blk = zeros((nslice,n_pe,n_fe_true)).astype(Complex32)
            for pe in range(n_pe):
                shdr = struct.unpack('>hhhhlffff',f_fid.read(sub_hdr))
                bias1 = complex(shdr[7],shdr[8])
                for slice in range(nslice):
                    # if weird_asems_ncsnn:
                    position = (pe*nslice+slice)*nvol*line_len + vol*line_len + sub_hdr + main_hdr
                    f_fid.seek(position,0)
                    blk_line = fromstring(f_fid.read(line_len_data),num_type).byteswapped().astype(Float32).tostring()
                    blk_line = fromstring(blk_line,Complex32)
                    bias[slice] = complex(0.,0.)
                    blk[slice,pe,:] = (blk_line - bias1).astype(Complex32)
            blk = reshape(blk,(nslice*n_pe,n_fe_true))
            time_reverse = 0
        else:
            print "Unknown type of fid file."
            sys.exit(1)

        # Time-reverse the data
        time_rev = n_fe_true - 1 - arange(n_fe_true)
        n_pe_per_seg = n_pe/nseg  
        if time_reverse:
            fact1 = n_pe_per_seg*nslice
            for seg in range(nseg):
                lim1 = seg*fact1
                for pe in range(n_pe_per_seg*nslice): 
                    if (pe % 2):
                        pe = pe + lim1
                        blk[pe,:] = take(blk[pe,:],time_rev)

        # Reorder the data according to the phase encode table read from "petable".
        ksp_image = zeros((nslice*n_pe_true,n_fe_true)).astype(Complex32)
        navigators = zeros((nslice*n_nav,n_fe_true)).astype(Complex32)
        if fid_type == "epi2fid":
            ksp_image = reshape(blk,(nslice,n_pe_true,n_fe_true))
        else:
            ksp_image = take(blk,petab)
            navigators = take(blk,navtab) 


        # Remove bias. Should work if all slices have same bias. 
        # !!!!! TRY WITH AND WITHOUT THIS BIAS SUBTRACTION LATER !!!!!
        ksp_image = reshape(ksp_image, (nslice, n_pe_true, n_fe_true))
        navigators = reshape(navigators, (nslice, n_nav, n_fe_true))
        for slice in range(nslice):
            ksp_image[slice,:,:] = (ksp_image[slice,:,:] - bias[slice]).astype(Complex32)
            navigators[slice,:,:] = (navigators[slice,:,:] - bias[slice]).astype(Complex32)

        # The time-reversal section above reflects the odd slices (acquistion order
        # implied) through the pe-axis. This section puts all slices in the same 
        # conventional orientation.
        if time_reverse and nav_per_seg > 0:
            for slice in range(0,nslice,2):
                for pe in range(n_pe_true):
                    ksp_image[slice,pe,:] = take(ksp_image[slice,pe,:],time_rev)


        if vol == 0:
            ref_data[:,:,:] = ksp_image[:,:,:]
            ref_nav_data[:,:,:] = navigators[:,:,:]
        else:
            data_matrix[vol,:,:,:] = ksp_image[:,:,:]
            nav_data[vol,:,:,:] = navigators[:,:,:]

    f_fid.close()
    return data


#*****************************************************************************
def fermi_filter(params, options, data):
    ksp_data = data.data_matrix
    nav_per_seg = params['nav_per_seg']
    nseg = params['nseg']
    n_pe = params['n_pe']
    n_pe_true = params['n_pe_true']
    n_fe = params['n_fe']
    n_fe_true = params['n_fe_true']
    nslice =  params['nslice']
    nvol = params['nvol']
   
    # Create Fermi filter kernel.
    Fermi = zeros((n_pe_true,n_fe_true)).astype(Float32)
    cutoff = 0.9
    for j in range(n_pe_true/2):
        r2 = j/(0.5*n_pe_true)
        for k in range(n_fe_true/2):
            r3 = k/(0.5*n_fe_true)
            r = r2*r2 + r3*r3
            filter = 1.0/(1.0 + exp((r-cutoff)*50.0))
            Fermi[n_pe_true/2  -j,n_fe_true/2-1+k] = filter
            Fermi[n_pe_true/2-1+j,n_fe_true/2  -k] = filter
            Fermi[n_pe_true/2  -j,n_fe_true/2  -k] = filter
            Fermi[n_pe_true/2-1+j,n_fe_true/2-1+k] = filter

    # Fermi Filter the data.
    for vol in range(nvol):
        for slice in range(nslice):
            for pe in range(n_pe_true):
                    ksp_data[vol,slice,pe,:] = ksp_data[vol,slice,pe,:]*Fermi[pe,:]


#*****************************************************************************
def fft_data(params, options, data):
    nav_per_seg = params['nav_per_seg']
    nseg = params['nseg']
    n_pe = params['n_pe']
    n_pe_true = params['n_pe_true']
    n_fe = params['n_fe']
    n_fe_true = params['n_fe_true']
    nslice =  params['nslice']
    nvol = params['nvol']

    print "Taking FFTs. Please Wait"
    for vol in range(nvol):
        tmp_vol = zeros((nslice,n_pe_true,n_fe_true)).astype(Complex32)
        for slice in range(nslice):
            ksp = zeros((n_pe_true,n_fe_true)).astype(Complex32)
            ksp[0:n_pe_true,0:n_fe_true] = data.data_matrix[vol,slice,:,:]
            image = FFT.inverse_fft2d(ksp)

            # The following shift is required because the frequency space data were not
            # put into standard ordering for FFT.
            idl.shift(image,0,n_fe_true/2) 
            idl.shift(image,1,n_pe_true/2)  
         
            # Reorder the slices from inferior to superior.
            if nslice % 2: midpoint = nslice/2 + 1
            else: midpoint = nslice/2
            if slice < midpoint:
                if options.flip_slices: z = 2*slice
                else: z = nslice - 2*slice - 1
            else:
                if options.flip_slices: z = 2*(slice - midpoint) + 1
                else: z = nslice - 2*(slice - midpoint) - 2
            tmp_vol[z,:,:] = image[:,:].astype(Complex32)
        data.data_matrix[vol,:,:,:] = tmp_vol[:,:,:]


#*****************************************************************************
def save_image_data(data_matrix, params, options):
    """
    This function saves the image data to disk in a file specified on the command
    line. The file name is stored in the options dictionary using the img_file key.
    The output_data_type key in the options dictionary determines whether the data 
    is saved to disk as complex or magnitude data. By default the image data is 
    saved as magnitude data.
    """
    nav_per_seg = params['nav_per_seg']
    nseg = params['nseg']
    n_pe = params['n_pe']
    n_pe_true = params['n_pe_true']
    n_fe = params['n_fe']
    n_fe_true = params['n_fe_true']
    nslice =  params['nslice']
    pulse_sequence = params['pulse_sequence']
    nvol = params['nvol']
    xsize = params['xsize'] 
    ysize = params['ysize']
    zsize = params['zsize']

    # Setup output file names.
    if options.file_format == SPM_FORMAT:
        period = string.rfind(options.img_file,".img")
        if period < 0:
            period = string.rfind(options.img_file,".hdr")
    elif options.file_format == FIDL_FORMAT:
        period = string.rfind(options.img_file,".4dfp.img")
        if period < 0:
            period = string.rfind(options.img_file,".4dfp.ifh")
    elif options.file_format == VOXBO_FORMAT:
        period = string.rfind(options.img_file,".tes")
    if period < 0: img_stem = options.img_file
    else: img_stem = options.img_file[:period]
    if options.file_format == FIDL_FORMAT:
        options.img_file     = "%s.4dfp.img" % (img_stem)
        img_file_hdr = "%s.4dfp.hdr" % (img_stem)
        img_file_ifh = "%s.4dfp.ifh" % (img_stem)
        f_img = open(options.img_file,"w")
    if options.phs_corr != "": frame_start = 1
    else: frame_start = 0

    # Calculate proper scaling factor for SPM Analyze format by using the maximum value over all volumes.
    scl = 16383.0/abs(data_matrix).flat[argmax(abs(data_matrix).flat)]

    # Save data to disk
    print "Saving to disk. Please Wait"
    tmp_vol = zeros((nslice,n_pe_true,n_fe_true)).astype(Complex32)
    vol_rng = range(frame_start, nvol)
    for vol in vol_rng:
        if options.save_first and vol == 0:  #!!!! DO WE NEED THIS SECTION !!!!!
            img = zeros((nslice,n_fe_true,n_pe_true)).astype(Float32)
            for slice in range(nslice):
                if flip_left_right:
                    img[slice,:,:] = MLab.fliplr(abs(data_matrix[vol,slice,:,:])).astype(Float32)
                else:
                    img[slice,:,:] = abs(data_matrix[vol,slice,:,:]).astype(Float32)
            file_io.write_cub(img,"EPIs.cub",n_fe_true,n_pe_true,nslice,ysize,xsize,zsize,0,0,0,"s",params)
        if  options.file_format == SPM_FORMAT:
            # Open files for this volume.
            options.img_file = "%s_%04d.img" % (img_stem,vol)
            # Write to disk.
            if options.output_data_type == MAGNITUDE_TYPE:
                tmp_vol = abs(data_matrix[vol,:,:,:]).astype(Float32)
                tmp_vol = multiply(scl,tmp_vol)
                hdr = file_io.create_hdr(n_fe_true,n_pe_true,nslice,1,ysize,xsize,zsize,1.,0,0,0,'Short',64,1.,'analyze',options.img_file,0)
            elif options.output_data_type == COMPLEX_TYPE:
                hdr = file_io.create_hdr(n_fe_true,n_pe_true,nslice,1,ysize,xsize,zsize,1.,0,0,0,'Complex',64,1.,'analyze',options.img_file,0)
            file_io.write_analyze(options.img_file,hdr,tmp_vol[:,:,:].astype(Float32)) #!!! NOT SAVING AS COMPLEX WHEN 'complex' IS SET !!!
        elif file_format == FIDL_FORMAT:
            # Write to disk. !!!!!!! SHOULDN'T THERE BE OPTION FOR MAG OR COMPLEX SAVE HERE !!!!!!!
            f_img.write(abs(data_matrix[vol,:,:,:]).astype(Float32).byteswapped().tostring())

        # !!!!!!!!! WHERE IS THE CODE TO SAVE IN VOXBO FORMAT !!!!!!!!


#*****************************************************************************
def save_ksp_data(complex_data, params, options):   # !!! FINISH AND TEST !!!
    """
    This function saves the k-space data to disk in a file specified on the
    command line with an added suffix of "_ksp". The k-space data is saved
    with the slices in the acquisition order rather than the spatial order.
    """
    nav_per_seg = params['nav_per_seg']
    nseg = params['nseg']
    n_pe = params['n_pe']
    n_pe_true = params['n_pe_true']
    n_fe = params['n_fe']
    n_fe_true = params['n_fe_true']
    nslice =  params['nslice']
    pulse_sequence = params['pulse_sequence']
    nvol = params['nvol']
    xsize = params['xsize'] 
    ysize = params['xsize']
    zsize = params['zsize']

    # Setup output file names.
    if options.file_format == SPM_FORMAT:
        period = string.rfind(options.img_file,".img")
        if period < 0:
            period = string.rfind(options.img_file,".hdr")
    elif options.file_format == FIDL_FORMAT:
        period = string.rfind(options.img_file,".4dfp.img")
        if period < 0:
            period = string.rfind(options.img_file,".4dfp.ifh")
    elif options.file_format == VOXBO_FORMAT:
        period = string.rfind(options.img_file,".tes")
    if period < 0:
        img_stem = options.img_file
    else:
        img_stem = options.img_file[:period]
    if options.file_format == FIDL_FORMAT:
        options.img_file     = "%s.4dfp.img" % (img_stem)
        img_file_hdr = "%s.4dfp.hdr" % (img_stem)
        img_file_ifh = "%s.4dfp.ifh" % (img_stem)
        f_img = open(options.img_file,"w")

    # Save data to disk
    print "Saving to disk. Please Wait"
    vol_rng = range(nvol)
    for vol in vol_rng:
        if  options.file_format == SPM_FORMAT:
            # Open files for this volume.
            ksp_files = "%s_%04d_ksp.img" % (img_stem)
            # Write to disk.
            hdr = file_io.create_hdr(n_fe_true,n_pe_true,nslice,1,xsize,ysize,zsize,1.,0,0,0,'Complex',64,1.,'analyze',ksp_files,0)
            file_io.write_analyze(ksp_files,hdr,complex_data[vol,:,:,:].astype(Complex32))
        elif file_format == FIDL_FORMAT:
            # Write to disk.
            f_img.write(complex_data[vol,:,:,:].astype(Complex32).byteswapped().tostring())


#*****************************************************************************
class PhaseCorrelation (Operation):
    # Default values for toy test args
    type = "nonlinear"
    foo = 15

    def run(self, params, options, data):
        print "PhaseCorrelation operation:  test args [type=%s, foo=%d]"\
          %(self.type, self.foo)
        ref_data = data.ref_data
        ksp_data = data.data_matrix
        ref_nav_data = data.ref_nav_data
        ksp_nav_data = data.nav_data
        nvol = params['nvol']
        nslice = params['nslice']
        nseg = params['nseg']
        nav_per_seg = params['nav_per_seg']
        n_pe = params['n_pe']
        n_pe_true = params['n_pe_true']
        n_fe = params['n_fe']
        n_fe_true = params['n_fe_true']
        te = string.atof(params['te'])
        trise = string.atof(params['trise'])
        gro = string.atof(params['gro'])
        gmax = string.atof(params['gmax'])
        at = string.atof(params['at'])


    # !!!!!!!!!!!!! IS THIS CONDITIONAL NEEDED. AFTERALL "alt" AS A DESIGNATION FOR A petable_file IS USED NOWHERE ELSE IN THE CODE
    # !!!!!!!!!!!!! UNCOMMENT PARTS THAT ARE NEEDED 
    #    if(string.find(petable_file,"alt") >= 0):
    #        time0 = te - 2.0*abs(gro)*trise/gmax - at
    #    else:
    #        time0 = te - (math.floor(nv/nseg)/2.0)*((2.0*abs(gro)*trise)/gmax + at)
    #    time1 = 2.0*abs(gro)*trise/gmax + at
    #    if nav_per_seg > 0:
    #        print "Data acquired with navigator echo time of %f" % (time0)
    #        print "Data acquired with echo spacing of %f" % (time1)

        # Compute correction for Nyquist ghosts.
        # First and/or last block contains phase correction data.  Process it.
        phasecor_phs = zeros((nslice, n_pe_true, n_fe_true)).astype(Float)
        phasecor_ftmag = zeros((nslice, n_pe_true, n_fe_true)).astype(Float32)
        phasecor_total = zeros((nslice, n_pe_true, n_fe_true)).astype(Complex32)
        tmp = zeros(n_fe_true).astype(Complex32)
        temp = zeros(n_fe_true).astype(Complex32)
        # Compute point-by-point phase correction
        for slice in range(nslice):
            for pe in range(n_pe_true):
                tmp[:] = ref_data[slice,pe,:].astype(Complex32)
                idl.shift(tmp, 0, n_fe_true/2)  # IS THIS NEEDED? NOTICE THIS IS n_fe_true
                ft_blk = FFT.inverse_fft(tmp)
                idl.shift(ft_blk, 0, n_fe/4)  # IS THIS NEEDED? NOTICE THIS IS n_fe
                msk = where(equal(ft_blk.real, 0.0), 1.0, 0.0)
                phs = (1.0 - msk)*arctan(ft_blk.imag/(ft_blk.real + msk))
                phasecor_ftmag[slice,pe,:] = abs(ft_blk).astype(Float32)

                # Create mask for threshold of zero for magnitudes.
                phasecor_ftmag_abs = abs(phasecor_ftmag[slice,pe,:])
                mag_msk = where(phasecor_ftmag_abs > 0.0, 1.0, 0.0) 

                # Convert to 4 quadrant arctan and set cor to zero for low magnitude voxels.
                pos_msk = where(phs > 0.0, 1.0, 0.0)
                msk1 = pos_msk*where(ft_blk.imag < 0, math.pi, 0.0)   # Re > 0, Im < 0
                msk2 = (1.0 - pos_msk)*where(ft_blk.imag < 0, 2.0*math.pi, math.pi) # Re < 0, Im < 0
                phs = mag_msk*(phs + msk1 + msk2) 
                phasecor_phs[slice,pe,:] = phs[:].astype(Float32)

        # Apply the phase correction to the image data.
        pe_per_seg = n_pe/nseg
        dphs = zeros(n_pe_true).astype(Float32)
        for vol in range(nvol):
            for slice in range(nslice):
    # I HAVE COMMENTED THE SECTIONS INVOLVING THE NAV ECHO WHICH WOULD APPEAR DON'T HAVE A LOT OF IMPACT.
    # I CAN PUT THIS BACK IN LATER AFTER THE REST IS WORKING CORRECTLY.
    #            for seg in range(nseg):
    #                if nseg > 0:
    #                    # The first data frame is a navigator echo, compute the difference  in 
    #                    # phase (dphs) due to B0 inhomogeneities using navigator echo.
    #                    tmp[:] = ksp_nav_data[vol,slice,seg,:] 
    #                    idl.shift(tmp,0,n_fe/4)
    #                    nav_echo = (FFT.inverse_fft(tmp)).astype(Complex32)
    #                    idl.shift(nav_echo,0,n_fe/4)
    #                    nav_mag = abs(nav_echo)
    #                    msk = where(equal(nav_echo.real,0.),1.,0.)
    #                    phs = (1.0 - msk)*arctan(nav_echo.imag/(nav_echo.real + msk))
    #
    #                    # Convert to 4 quadrant arctan.
    #                    pos_msk = where(phs > 0,1,0)
    #                    msk1 = pos_msk*where(nav_echo.imag < 0, math.pi, 0)
    #                    msk2 = (1 - pos_msk)*where(nav_echo.imag < 0, 2.0*math.pi, math.pi)
    #                    msk  = where((msk1 + msk2) == 0,1,0)
    #                    phs = phs + msk1 + msk2
    #
    #                    # Create mask for threshold of MAG_THRESH for magnitudes.
    #                    mag_msk = where(nav_mag > 0.0, 1, 0)
    #                    nav_phs = mag_msk*phs
    #                    dphs = (phasecor_phs[slice,seg,:] - nav_phs)
    #                    msk1 = where(dphs < -math.pi, 2.0*math.pi, 0)
    #                    msk2 = where(dphs > math.pi, -2.0*math.pi, 0)
    #                    nav_mask = where(nav_mag > 0.75*MLab.max(nav_mag), 1.0, 0.0)
    #                    dphs = dphs + msk1 + msk2  # This partially corrects for field inhomogeneity.
    #
    #                for pe in range(n_pe_true/nseg):
    #                    # Calculate the phase correction.
    #                    time = time0 + pe*time1
    #                    if nseg > 0 and not ignore_nav:
    #                        theta = -(phasecor_phs[slice,pe,:] - dphs*time/time0)
    #                        msk1 = where(theta < 0.0, 2.0*math.pi, 0)
    #                        theta = theta + msk1
    #                        scl = cos(theta) + 1.0j*sin(theta)
    #                        msk = where(nav_mag == 0.0, 1, 0)
    #                        mag_ratio = (1 - msk)*phasecor_ftmag[slice,pe,:]/(nav_mag + msk)
    #                        msk1 = (where((mag_ratio > 1.05), 0.0, 1.0))
    #                        msk2 = (where((mag_ratio < 0.95), 0.0, 1.0))
    #                        msk = msk1*msk2
    #                        msk = (1 - msk) + msk*mag_ratio
    #                        cor = scl*msk
    #                    else:
    #                        theta = -phasecor_phs[slice,pe,:]
    #                        cor = cos(theta) + 1.j*sin(theta)
    #                    phasecor_total[slice,pe,:] = theta.astype(Float32)
    #                    
    #                    # Do the phase correction.
    #                    tmp[:] = ksp_data[vol,slice,pe,:] 
    #                    idl.shift(tmp, 0, n_fe/4)
    #                    echo = FFT.inverse_fft(tmp)
    #                    idl.shift(echo, 0, n_fe/4)
    #
    #                    # Shift echo time by adding phase shift.
    #                    echo = echo*cor
    #
    #                    idl.shift(echo, 0, n_fe/4)
    #                    tmp = (FFT.fft(echo)).astype(Complex32)
    #                    idl.shift(tmp, 0, n_fe/4)
    #                    ksp_data[vol, slice, pe, :] = tmp

                for pe in range(n_pe_true):
                    theta = -phasecor_phs[slice,pe,:]
                    cor = cos(theta) + 1.0j*sin(theta)
                    phasecor_total[slice,pe,:] = theta.astype(Float32)

                    # Do the phase correction.
                    tmp[:] = ksp_data[vol,slice,pe,:] 
                    idl.shift(tmp, 0, n_fe/4)
                    echo = FFT.inverse_fft(tmp)
                    idl.shift(echo, 0, n_fe/4)

                    # Shift echo time by adding phase shift.
                    echo = echo*cor
                    idl.shift(echo, 0, n_fe/4)
                    tmp = (FFT.fft(echo)).astype(Complex32)
                    idl.shift(tmp, 0, n_fe/4)
                    ksp_data[vol, slice, pe, :] = tmp
