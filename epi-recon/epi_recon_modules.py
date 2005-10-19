import sys
import string 
import os
from Numeric import *
import file_io
import struct
from ConfigParser import SafeConfigParser
from types import TypeType, BooleanType
from optparse import OptionParser, Option
from VolumeViewer import VolumeViewer
from FFT import inverse_fft, inverse_fft2d
from pylab import (
  pi, mlab, fft, frange, fliplr, amax, meshgrid, floor, zeros, squeeze,
  fromstring)
from prompt import prompt

FIDL_FORMAT = "fidl"
VOXBO_FORMAT = "voxbo"
SPM_FORMAT = "spm"
MAGNITUDE_TYPE = "magnitude"
COMPLEX_TYPE = "complex"

output_format_choices = (FIDL_FORMAT, VOXBO_FORMAT, SPM_FORMAT)
output_datatype_choices= (MAGNITUDE_TYPE, COMPLEX_TYPE)


#-----------------------------------------------------------------------------
def shift(matrix, axis, shft):
    """
    axis: Axis of shift: 0=x (rows), 1=y (columns),2=z
    shft: Number of pixels to shift.
    """
    dims = matrix.shape
    ndim = len(dims)
    if ndim == 1:
        tmp = zeros((shft),matrix.typecode())
        tmp[:] = matrix[-shft:]
        matrix[-shft:] = matrix[-2*shft:-shft]
        matrix[:shft] = tmp
    elif ndim == 2:
        ydim = dims[0]
        xdim = dims[1]
        tmp = zeros((shft),matrix.typecode())
        new = zeros((ydim,xdim),matrix.typecode())
        if(axis == 0):
            for y in range(ydim):
                tmp[:] = matrix[y,-shft:]
                new[y,shft:] =  matrix[y,:-shft]
                new[y,:shft] = matrix[y,-shft:]
            matrix[:,:] = new[:,:]
        elif(axis == 1):
            for x in range(xdim):
                new[shft:,x] =  matrix[:-shft,x]
                new[:shft,x] = matrix[-shft:,x]
            matrix[:,:] = new[:,:]
    elif ndim == 3:
        zdim = dims[0]
        ydim = dims[1]
        xdim = dims[2]
        new = zeros((zdim,ydim,xdim),matrix.typecode())
        if(axis == 0):
            tmp = zeros((zdim,ydim,shft),matrix.typecode())
            tmp[:,:,:] = matrix[:,:,-shft:]
            new[:,:,shft:] =  matrix[:,:,:-shft]
            new[:,:,:shft] = matrix[:,:,-shft:]
        elif(axis == 1):
            tmp = zeros((zdim,shft,xdim),matrix.typecode())
            tmp[:,:,:] = matrix[:,-shft:,:]
            new[:,shft:,:] =  matrix[:,:-shft,:]
            new[:,:shft,:] = matrix[:,-shft:,:]
        elif(axis == 2):
            tmp = zeros((shft,ydim,xdim),matrix.typecode())
            tmp[:,:,:] = matrix[-shft:,:,:]
            new[shft:,:,:] =  matrix[:-shft,:,:]
            new[:shft,:,:] = matrix[-shft:,:,:]
        matrix[:,:,:] = new[:,:,:]
    else:
        print "shift() only support 1D, 2D, and 3D arrays."
        sys.exit(1)


#-----------------------------------------------------------------------------
def shifted_fft(a):
    tmp = a.copy()
    shift_width = a.shape[0]/2
    shift(tmp, 0, shift_width)
    tmp = fft(tmp)
    shift(tmp, 0, shift_width)
    return tmp


#-----------------------------------------------------------------------------
def shifted_inverse_fft(a):
    tmp = a.copy()
    shift_width = a.shape[0]/2
    shift(tmp, 0, shift_width)
    tmp = inverse_fft(tmp)
    shift(tmp, 0, shift_width)
    return tmp


#-----------------------------------------------------------------------------
def atan2(a):
    x = a.real; y=a.imag
    msk = where(equal(x, 0.0), 1.0, 0.0)
    phs = (1.0 - msk)*arctan(y/(x + msk))
    pos_msk = where(phs > 0.0, 1.0, 0.0)
    msk1 = pos_msk*where(y < 0, pi, 0.0)   # Re > 0, Im < 0
    msk2 = (1.0 - pos_msk)*where(y < 0, 2.0*pi, pi) # Re < 0, Im < 0
    return phs + msk1 + msk2


#*****************************************************************************
class EmptyObject (object):
    "Takes whatever attributes are passed as keyword args when initialized."
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


#*****************************************************************************
def bool_valuator(val):
    if type(val)==BooleanType: return val
    lowerstr = val.lower()
    if lowerstr == "true": return True
    elif lowerstr == "false": return False
    else: raise ValueError(
      "Invalid boolean specifier '%s'. Must be either 'true' or 'false'."%\
      lowerstr)


#*****************************************************************************
class Parameter (object):
    """
    Specifies a named, typed parameter for an Operation.  Is used for
    documentation and during config parsing.
    """

    # map type spec to callable type constructor
    _type_map = {
      "str":str,
      "bool":bool_valuator,
      "int":int,
      "float":float,
      "complex":complex}

    #-------------------------------------------------------------------------
    def __init__(self, name, type="str", default=None, description=""):
        self.name=name
        if not self._type_map.has_key(type):
            raise ValueError("type must be one of %s"%self._type_map.keys())
        self.valuator=self._type_map[type]
        self.default=default

    #-------------------------------------------------------------------------
    def valuate(self, valspec): return self.valuator(valspec)


#*****************************************************************************
class Operation (object):
    "Base class for data operations."

    class ConfigError (Exception): pass

    params=()

    #-------------------------------------------------------------------------
    def __init__(self, **kwargs): self.configure(**kwargs)

    #-------------------------------------------------------------------------
    def configure(self, **kwargs):
        #print self.__class__.__name__, "params=", self.params
        for p in self.params:
            self.__dict__[p.name] = p.valuate(kwargs.pop(p.name, p.default))

        # All valid args should have been popped of the kwargs dict at this
        # point.  If any are left, it means they are not valid parameters for
        # this operation.
        leftovers = kwargs.keys()
        if leftovers:
            raise self.ConfigError("Invalid parameter '%s' for operation %s"%
              (leftovers[0], self.__class__.__name__))

    #-------------------------------------------------------------------------
    def run(self, params, options, data): pass


#*****************************************************************************
class OperationManager (object):
    """
    This class is responsible for knowing which operations are available
    and retrieving them by name.  It should be a global singleton in the
    system. (Preferrably, an attribute of the EpiRecon tool class, once
    that class is implemented.)
    """

    class InvalidOperationName (Exception): pass

    #-------------------------------------------------------------------------
    def getOperationNames(self):
        "@return list of valid operation names."
        return tuple([name for name,obj in globals().items()
          if type(obj)==TypeType and issubclass(obj, Operation)])

    #-------------------------------------------------------------------------
    def getOperation(self, opname):
        "@return the operation for the given name"
        operation = globals().get(opname, None)
        if not operation:
            raise self.InvalidOperationName("Operation '%s' not found."%opname)
        return operation


#*****************************************************************************
class OrderedConfigParser (SafeConfigParser):
    "Config parser which keeps track of the order in which sections appear."

    #-------------------------------------------------------------------------
    def __init__(self, defaults=None):
        SafeConfigParser.__init__(self, defaults=defaults)
        import odict
        self._sections = odict.odict()


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
          Option( "-c", "--config", dest="config", type="string",
            default="recon.cfg", action="store",
            help="Name of the config file describing operations and operation"\
            " parameters."),
          Option( "-r", "--vol-range", dest="vol_range", type="string", default=":",
            action="store",
            help="Which volumes to reconstruct.  Format is start:end, where "\
            "either start or end may be omitted, indicating to start with the "\
            "first or end with the last respectively.  The index of the first "\
            "volume is 0.  The default value is a single colon with no start "\
            "or end specified, meaning process all volumes."),
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
          Option( "-t", "--tr", dest="TR", type="float", action="store",
            help="Use the TR given here rather than the one in the procpar." ),
          Option( "-x", "--starting-frame-number", dest="sfn", type="int",
            default=0, action="store", metavar="<starting frame number>",
            help="Specify starting frame number for analyze format output." ),
          Option( "-l", "--flip-left-right", dest="flip_left_right",
            action="store_true", help="Flip image about the vertical axis." ),
          Option( "-y", "--output-data-type", dest="output_data_type",
            type="choice", default=MAGNITUDE_TYPE, action="store",
            choices=output_datatype_choices,
            help="""{%s}
            Specifies whether output images should contain only magnitude or
            both the real and imaginary components (only valid for analyze
            format)."""%("|".join(output_datatype_choices)) )
        ))

    #-------------------------------------------------------------------------
    def configureOperations(self, configfile):
        """
        @return a list of operation pairs (operation, args).
        @param configfile: filename of operations config file.
        """
        config = OrderedConfigParser()
        config.read(configfile)
        return [
          (self._opmanager.getOperation(opname), dict(config.items(opname)))
          for opname in config.sections()]

    #-------------------------------------------------------------------------
    def parseVolRange(self, vol_range):
        parts = vol_range.split(":")
        if len(parts) < 2: self.error(
          "The specification of vol-range must contain a colon separating "\
          "the start index from the end index.")
        try: vol_start = int(parts[0] or 0)
        except ValueError: self.error(
          "Bad vol-range start index '%s'.  Must be an integer."%parts[0])
        try: vol_end = int(parts[1] or -1)
        except ValueError: self.error(
          "Bad vol-range end index '%s'. Must be an integer."%parts[1])
        return vol_start, vol_end

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

        # parse vol-range
        options.vol_start, options.vol_end = \
          self.parseVolRange(options.vol_range)

        # configure operations
        options.operations = self.configureOperations(options.config)

        #options.ignore_nav = false

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
            params['thk'] = float(thk)
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
            params['nslice'] = int(nslice)
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
            params['n_fe'] = int(np)
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
            params['n_pe'] = int(nv)
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
    params['nvol'] = int(params['nvol'])


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
        nseg = int(params['petable'][-2])
    elif(pulse_sequence == 'epidw' or pulse_sequence == 'Vsparse'):
        pulse_sequence = 'epidw'
        nseg = int(params['petable'][-1])
        if(params.has_key('dquiet')):
            TR = TR + float(params['quiet_interval'])
    elif(pulse_sequence == 'epi_se'):
        if string.rfind(params['petable'],'epidw') < 0:
            petable = "epi%dse%dk" % (n_pe,nseg)
        else:
            pulse_sequence = 'epidw'
        nseg = int(params['nseg'])
    elif(pulse_sequence == 'epidw_sb'):
        if string.rfind(params['petable'],'epidw') < 0:
            petable = "epi%dse%dk" % (n_pe,nseg)
        else:
            pulse_sequence = 'epidw'
        nseg = int(params['nseg'])
    elif(pulse_sequence == 'epidw_se'):
        nseg = int(params['nseg'])
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
        nseg = int(params['petable'][-2])
        pulse_sequence = 'epi'
        TR = TR + float(params['quiet_interval'])
    else:
        print "Could not identify sequence: %s" % (pulse_sequence)
        sys.exit(1)

    params['tr'] = nseg*float(params['tr'])
    params['nseg'] = nseg
    params['n_pe_true'] =  params['n_pe'] - nseg*params['nav_per_seg']
    params['xsize'] = float(params['fov'])/params['n_pe_true']
    params['ysize'] = float(params['fov'])/params['n_fe_true']
    params['zsize'] = float(params['thk']) + float(params['gap'])
    params["te"] = float(params['te'])
    params["trise"] = float(params['trise'])
    params["gro"] = float(params['gro'])
    params["gmax"] = float(params['gmax'])
    params["at"] = float(params['at'])

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
    return EmptyObject(
      data_matrix = zeros((nvol, nslice, n_pe_true, n_fe_true), Complex32),
      nav_data = zeros((nvol, nslice, n_nav, n_fe_true), Complex32),
      ref_data = zeros((nslice, n_pe_true, n_fe_true), Complex32),
      ref_nav_data = zeros((nslice, n_nav, n_fe_true), Complex32))


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
            ref_data[:] = ksp_image
            ref_nav_data[:] = navigators
        else:
            data_matrix[vol] = ksp_image
            nav_data[vol] = navigators

    f_fid.close()
    return data


#*****************************************************************************
def save_image_data(data_matrix, params, options):
    """
    This function saves the image data to disk in a file specified on the command
    line. The file name is stored in the options dictionary using the img_file key.
    The output_data_type key in the options dictionary determines whether the data 
    is saved to disk as complex or magnitude data. By default the image data is 
    saved as magnitude data.
    """
    VolumeViewer(
      abs(data_matrix),
      ("Time Point", "Slice", "Row", "Column"))
    nav_per_seg = params['nav_per_seg']
    nseg = params['nseg']
    n_pe = params['n_pe']
    n_pe_true = params['n_pe_true']
    n_fe = params['n_fe']
    n_fe_true = params['n_fe_true']
    nslice =  params['nslice']
    pulse_sequence = params['pulse_sequence']
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
    vol_rng = range(frame_start, params['nvol'])
    for vol in vol_rng:
        if options.save_first and vol == 0:  #!!!! DO WE NEED THIS SECTION !!!!!
            img = zeros((nslice,n_fe_true,n_pe_true)).astype(Float32)
            for slice in range(nslice):
                if flip_left_right:
                    img[slice,:,:] = fliplr(abs(data_matrix[vol,slice,:,:])).astype(Float32)
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
    vol_rng = range(params['nvol'])
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
class PhaseCorrection (Operation):

    #-------------------------------------------------------------------------
    def run(self, params, options, data):
        ref_data = data.ref_data
        ksp_data = data.data_matrix
        nslice = params['nslice']
        n_pe_true = params['n_pe_true']
        n_fe = params['n_fe']
        n_fe_true = params['n_fe_true']

        # Compute correction for Nyquist ghosts.
        # First and/or last block contains phase correction data.  Process it.
        phasecor_phs = zeros((nslice, n_pe_true, n_fe_true)).astype(Float)

        # Compute point-by-point phase correction
        for slice, phscor_slice in zip(ref_data, phasecor_phs):
            for pe, theta in zip(slice, phscor_slice):
                # Convert inverse fourier transformed pe to 4 quadrant arctan.
                theta[:] = atan2(shifted_inverse_fft(pe))

        # Apply the phase correction to the image data.
        for volume in ksp_data:
            for slice, phscor_slice in zip(volume, phasecor_phs):
                for pe, theta in zip(slice, phscor_slice):
                    correction = cos(-theta) + 1.0j*sin(-theta)

                    # Shift echo time by adding phase shift.
                    echo = shifted_inverse_fft(pe)*correction
                    pe[:] = shifted_fft(echo).astype(Complex32)


#*****************************************************************************
class SegmentationCorrection (Operation):
    """
    Correct for the Nyquist ghosting in segmented scans due to mismatches
    between segments.
    """
    
    #-------------------------------------------------------------------------
    def get_times(self, params):
        te = params["te"]
        gro = params["gro"]
        trise = params["trise"]
        gmax = params["gmax"]
        at = params["at"]

        if(string.find(params["petable"],"alt") >= 0):
            time0 = te - 2.0*abs(gro)*trise/gmax - at
        else:
            time0 = te - (floor(params["nv"]/params["nseg"])/2.0)*\
                         ((2.0*abs(gro)*trise)/gmax + at)
        time1 = 2.0*abs(gro)*trise/gmax + at
        print "Data acquired with navigator echo time of %f" % (time0)
        print "Data acquired with echo spacing of %f" % (time1)
        return time0, time1

    #-------------------------------------------------------------------------
    def arctan(self, phs, nav_echo):
        msk = where(equal(nav_echo.real,0.),1.,0.)
        phs = (1.0 - msk)*arctan(nav_echo.imag/(nav_echo.real + msk))
        pos_msk = where(phs > 0,1,0)
        msk1 = pos_msk*where(nav_echo.imag < 0, pi, 0)
        msk2 = (1 - pos_msk)*where(nav_echo.imag < 0, 2.0*pi, pi)
        msk  = where((msk1 + msk2) == 0,1,0)
        return phs + msk1 + msk2

    #-------------------------------------------------------------------------
    def compute_phase_difference(self): pass

    #-------------------------------------------------------------------------
    def correct_slice(self, slice, nav_slice): pass

    #-------------------------------------------------------------------------
    def run(self, params, options, data):
        ksp_data = data.data_matrix
        ksp_nav_data = data.nav_data
        nseg = params['nseg']
    #    n_pe_true = params['n_pe_true']
    #    n_fe = params['n_fe']

        time0, time1 = self.get_times(params)
        print "SegmentationCorrection::correct_slice, nseg:",params["nseg"]," nav_per_seg:",params["nav_per_seg"],"ksp_data shape:",ksp_data.shape,"ksp_nav_data shape",ksp_nav_data.shape

        for volume, nav_volume in zip(ksp_data, ksp_nav_data):
            for slice, nav_slice in zip(volume, nav_volume):
                self.correct_slice(slice, nav_slice)
            #    for seg in range(nseg):
            #        if nseg > 0:
            #            # The first data frame is a navigator echo, compute the difference in
            #            # phase (dphs) due to B0 inhomogeneities using navigator echo.
            #            #tmp[:] = ksp_nav_data[vol,slice,seg,:] 
            #            #shift(tmp,0,n_fe/4)
            #            #nav_echo = (inverse_fft(tmp)).astype(Complex32)
            #            #shift(nav_echo,0,n_fe/4)
            #            nav_echo = shifted_inverse_fft(ksp_nav_data[vol,slice,seg,:])
            #
            #            # Convert to 4 quadrant arctan.
            #            phs = self.arctan(phs, nav_echo)
            
            #            # Create mask for threshold of MAG_THRESH for magnitudes.
            #            nav_mag = abs(nav_echo)
            #            mag_msk = where(nav_mag > 0.0, 1, 0)
            #            nav_phs = mag_msk*phs
            #            dphs = (phasecor_phs[slice,seg,:] - nav_phs)
            #            msk1 = where(dphs < -pi, 2.0*pi, 0)
            #            msk2 = where(dphs > pi, -2.0*pi, 0)
            #            nav_mask = where(nav_mag > 0.75*amax(nav_mag), 1.0, 0.0)
            #            dphs = dphs + msk1 + msk2  # This partially corrects for field inhomogeneity.
            #
            #        for pe in range(n_pe_true/nseg):
            #            # Calculate the phase correction.
            #            time = time0 + pe*time1
            #            if nseg > 0 and not ignore_nav:
            #                theta = -(phasecor_phs[slice,pe,:] - dphs*time/time0)
            #                msk1 = where(theta < 0.0, 2.0*pi, 0)
            #                theta = theta + msk1
            #                scl = cos(theta) + 1.0j*sin(theta)
            #                msk = where(nav_mag == 0.0, 1, 0)
            #                mag_ratio = (1 - msk)*phasecor_ftmag[slice,pe,:]/(nav_mag + msk)
            #                msk1 = (where((mag_ratio > 1.05), 0.0, 1.0))
            #                msk2 = (where((mag_ratio < 0.95), 0.0, 1.0))
            #                msk = msk1*msk2
            #                msk = (1 - msk) + msk*mag_ratio
            #                cor = scl*msk
            #            else:
            #                theta = -phasecor_phs[slice,pe,:]
            #                cor = cos(theta) + 1.j*sin(theta)

            #            echo = shifted_inverse_fft(ksp_data[vol,slice,pe,:])
            #            echo = echo*cor
            #            ksp_data[vol, slice, pe, :] = shifted_fft(echo)
            #                    
            #            # Do the phase correction.
            #            #tmp[:] = ksp_data[vol,slice,pe,:] 
            #            #shift(tmp, 0, n_fe/4)
            #            #echo = inverse_fft(tmp)
            #            #shift(echo, 0, n_fe/4)
            #
            #            # Shift echo time by adding phase shift.
            #            #echo = echo*cor
            #
            #            #shift(echo, 0, n_fe/4)
            #            #tmp = (fft(echo)).astype(Complex32)
            #            #shift(tmp, 0, n_fe/4)
            #            #ksp_data[vol, slice, pe, :] = tmp

    def old_version(self):
#       This block contains image data. First, calculate the phase correction including
#       the navigator echo.
        dphs = zeros(N_pe).astype(Float32)
        for slice in range(nslice):
#           correction for all echos.
            for seg in range(nseg):
                ii = line_index(slice,seg,pe_per_seg)
                jj = line_index(slice,seg,N_pe_true/nseg)
                for pe in range(N_pe/nseg):
#                   Calculate the phase correction.
                    time = time0 + pe*time1
###                    if pe == 0 and pulse_sequence == 'epi':
                    if pe == 0 and n_nav_echo > 0:
#                       The first data frame is a navigator echo, compute the difference  in 
#                       phase (dphs) due to B0 inhomogeneities using navigator echo.
                        tmp[:] = blk[pe+ii,:]
                        shift(tmp,0,N_fe/4)
                        nav_echo = (inverse_fft(tmp - bias[slice])).astype(Complex32)
                        shift(nav_echo,0,N_fe/4)
                        nav_mag = abs(nav_echo)
                        msk = where(equal(nav_echo.real,0.),1.,0.)
                        phs = (1.-msk)*arctan(nav_echo.imag/(nav_echo.real+msk))

                        # Convert to 4 quadrant arctan.
                        pos_msk = where(phs>0,1,0)
                        msk1 = pos_msk*where(nav_echo.imag<0,math.pi,0)
                        msk2 = (1-pos_msk)*where(nav_echo.imag<0,2.*math.pi,math.pi)
                        msk  = where((msk1+msk2) == 0,1,0)
                        phs = phs + msk1 + msk2

#                       Create mask for threshold of MAG_THRESH for magnitudes.
                        mag_msk = where(nav_mag>MAG_THRESH,1,0)
                        nav_phs = mag_msk*phs
                        dphs = (phasecor_phs[ii,:] - nav_phs)
                        msk1 = where(dphs<-math.pi, 2.*math.pi,0)
                        msk2 = where(dphs> math.pi,-2.*math.pi,0)
                        nav_mask = where(nav_mag>.75*MLab.max(nav_mag),1.,0.)
                        dphs = dphs + msk1 + msk2  # This partially corrects for field inhomogeneity.
                        nav_save[slice,seg,:] = (dphs*(time1/time0)).astype(Float32)
                    else:
                        if n_nav_echo > 0 and not ignore_nav:
                            theta = -(phasecor_phs[pe+ii,:] - dphs*time/time0)
                            msk1 = where(theta<0.,2.*math.pi,0)
                            theta = theta + msk1
                            scl = cos(theta) + 1.j*sin(theta)
                            msk = where(nav_mag==0.,1,0)
                            mag_ratio = (1-msk)*phasecor_ftmag[pe+ii,:]/(nav_mag + msk)
                            msk1 = (where((mag_ratio>1.05),0.,1.))
                            msk2 = (where((mag_ratio<.95),0.,1.))
                            msk = msk1*msk2
                            msk = (1-msk) + msk*mag_ratio
                            cor = scl*msk
                        else:
                            theta = -phasecor_phs[pe+ii,:]
                            cor = cos(theta) + 1.j*sin(theta)
 
                        # Do the phase correction.
                        tmp[:] = blk[pe+ii,:]
                        shift(tmp,0,N_fe/4)
                        echo = inverse_fft(tmp - bias[slice])
                        shift(echo,0,N_fe/4)

                        # Shift echo time by adding phase shift.
                        echo = echo*cor

                        shift(echo,0,N_fe/4)
                        tmp = (fft(echo)).astype(Complex32)
                        shift(tmp,0,N_fe/4)
                        blk_cor[vol-1,pe-n_nav_echo+jj,:] = tmp



#*****************************************************************************
def fermi_filter(rows, cols, cutoff, trans_width):
    """
    @return a Fermi filter kernel.
    @param cutoff: distance from the center at which the filter drops to 0.5.
      Units for cutoff are percentage of radius.
    @param trans_width: width of the transition.  Smaller values will result
      in a sharper dropoff.
    """
    from matplotlib.mlab import frange, meshgrid
    row_end = (rows-1)/2.0; col_end = (cols-1)/2.0
    row_vals = frange(-row_end, row_end)**2
    col_vals = frange(-col_end, col_end)**2
    X, Y = meshgrid(row_vals, col_vals)
    return 1/(1 + exp((sqrt(X + Y) - cutoff*cols/2.0)/trans_width))


#*****************************************************************************
class FermiFilter (Operation):
    "Apply a Fermi filter to the data."

    params=(
      Parameter(name="cutoff", type="float", default=0.95,
        description="distance from the center at which the filter drops to "\
        "0.5.  Units for cutoff are percentage of radius."),
      Parameter(name="trans_width", type="float", default=0.3,
        description="transition width.  Smaller values will result in a "\
        "sharper dropoff."))

    #-------------------------------------------------------------------------
    def run(self, params, options, data):
        rows, cols = data.data_matrix.shape[-2:]
        kernel = fermi_filter(
          rows, cols, self.cutoff, self.trans_width).astype(Float32)
        for volume in data.data_matrix:
          for slice in volume: slice *= kernel


#*****************************************************************************
class InverseFFT (Operation):
    "Perform an inverse 2D fft on each slice of each k-space volume."

    #-------------------------------------------------------------------------
    def run(self, params, options, data):
        n_pe, n_fe = data.data_matrix.shape[-2:]
        for volume in data.data_matrix:
            for slice in volume:
                image = inverse_fft2d(slice)
                shift(image,0,n_fe/2) 
                shift(image,1,n_pe/2)  
                slice[:,:] = image.astype(Complex32)


#*****************************************************************************
class ReorderSlices (Operation):
    "Reorder image slices from inferior to superior."

    params=(
      Parameter(name="flip_slices", type="bool", default=False,
        description="Flip slices during reordering."),)

    #-------------------------------------------------------------------------
    def run(self, params, options, data):
        nslice = data.data_matrix.shape[-3]

        # Reorder the slices from inferior to superior.
        midpoint = nslice/2 + (nslice%2 and 1 or 0)
        tmp = mlab.zeros_like(data.data_matrix[0])
        #print "midpoint=",midpoint
        for volume in data.data_matrix:
            # if I can get the slice indices right, these two lines can replace
            # the nine lines which follow them!
            #tmp[:midpoint] = self.flip_slices and volume[::2] or volume[::-2]
            #tmp[midpoint:] = self.flip_slices and volume[1::2] or volume[-2::-2]
            for i, slice in enumerate(volume):
                if i < midpoint:
                    if self.flip_slices: z = 2*i
                    else: z = nslice - 2*i - 1
                else:
                    if self.flip_slices: z = 2*(i - midpoint) + 1
                    else: z = nslice - 2*(i - midpoint) - 2
                #print i, z
                tmp[z] = slice
            volume[:] = tmp


#*****************************************************************************
def fi_phase_corr(params, options, data):
    "Correct for Nyquist ghosting due to field inhomogeneity."

    # Read the inhomogeneity field-map data from disk (Calculated using compute_fmap).
   
    # Loop over each phase-encode line of uncorrected data S'_mn.
       
        # Perform FFT of S'_mn with respect to n (frequency-encode direction) to
        # obtain S'_m(x).

        # Loop over each point x.

            # Calculate the perturbation kernel K_mm'(x) at x by: 
            # (1) Performing FFT with respect to y' of exp(i*2*pi*phi(x,y')*Delta t)
            #     and exp(i*2*pi*phi(x,y')*Delta t).
            # (2)  

            # Invert perturbation operator to obtain correction operator at x.

            # Apply correction operator at x to the distorted data S'_m(x) to
            # obtain the corrected data S_m(x).

        # Perform an inverse FFT in the read direction to obtain the corrected
        # k-space data S_mn.


