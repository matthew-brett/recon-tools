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

FIDL_FORMAT = "fidl"
VOXBO_FORMAT = "voxbo"
SPM_FORMAT = "spm"
MAGNITUDE_TYPE = "magnitude"
COMPLEX_TYPE = "complex"

output_format_choices = (FIDL_FORMAT, VOXBO_FORMAT, SPM_FORMAT)
output_datatype_choices= (MAGNITUDE_TYPE, COMPLEX_TYPE)


#*****************************************************************************
class OperationManager (object):
    """
    This class is responsible for knowing which operations are available
    and retrieving them by name.  It should be a global singleton in the
    system.
    """
    _operation_names = (
      "phase_corr",
      "fermi_filter",
      "fft_data")
    def getOperationNames(self):
        return self._operation_names   
    def getOperation(self, opname):
        return globals().get(opname, None)
opmanager = OperationManager()


#*****************************************************************************
def get_options():
    """
    Parse command-line arguments and options, including the list of operations
    to perform on the data.
    """
    from optparse import OptionParser, Option
    opnames = opmanager.getOperationNames()
    optparser = OptionParser(
      "usage: %prog [options] fid_file procpar output_image", option_list=(
      Option( "-o", "--operations", dest="operations", default="all",
        action="store",
        help="""
        A comma delimited list (no whitespace) of named operations to perform
        in the given order.  Available operations are {%s}.  By default all
        available operations will be performed in the order shown."""\
        %"|".join(opnames)),
      Option( "-n", "--nvol", dest="nvol_to_read", type="int", default=0,
        action="store",
        help="Number of volumes within run to reconstruct." ),
      Option( "-s", "--frames-to-skip", dest="skip", type="int", default=0,
        action="store",
        help="Number of frames to skip at beginning of run." ),
      Option( "-f", "--file-format", dest="file_format", action="store",
        type="choice", default=FIDL_FORMAT,
        choices=output_format_choices,
        help="""{%s}
        fidl: save floating point file with interfile and 4D analyze headers.
        spm: Save individual image for each frame in analyze format.
        voxbo: Save in tes format."""%("|".join(output_format_choices)) ),
      Option( "-p", "--phs-corr", dest="phs_corr", default="", action="store",
        help="Dan, please describe the action of this option..."),
      Option( "-a", "--save-first", dest="save_first", action="store_true",
        help="Save first frame in file named 'EPIs.cub'." ),
      Option( "-g", "--ignore-nav-echo", dest="ignore_nav", action="store_true",
        help="Do not use navigator in phase correction." ),
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

    options, args = optparser.parse_args()
    if len(args) != 3: optparser.error(
      "Expecting 3 arguments: fid_file procpar_file img_file" )
    options.fid_file, options.procpar_file, options.img_file = args

    # THIS SHOULD ONLY TEMPORARALLY BE HARD-CODED.
    options.n_echo = 2

    # is this necessary?
    if options.save_first: options.ignore_nav = true

    # resolve operations
    operations = []
    if options.operations == "all": options.operations = opnames
    else: options.operations = options.operations.split(",")
    for opname in options.operations:
        operation = opmanager.getOperation(opname)
        if not operation: optparser.error(
          "Invalid operation name: '%s'\n"\
          "Valid operation names are {%s}."%\
          (opname,"|".join(opnames)))
        operations.append(operation)
    options.operations = operations

    return options


#*****************************************************************************
def get_params( options ):
    """
    This function gets many parameters related to the scan. Some parameters
    may be overridden by the options from the command line and hence the
    "options" list is an argument to this function.
    """

    # Get parameters contained in the Varian procpar file. 
    params = file_io.parse_procpar(options.procpar_file,0)

    pulse_sequence = params['pulse_sequence']
    if options.TR > 0:
        params['tr'] = options.TR
    params['n_fe'] = string.atoi(params['np'])
    params['n_fe_true'] = params['n_fe']/2
    params['n_pe'] = string.atoi(params['nv'])
    n_pe = params['n_pe'] 

    # Determine n_nav_echo the number of navigator echoes per segment.
    if n_pe % 32:
        n_nav_echo = 1
    else:
        n_nav_echo = 0
        ignore_nav = 1

    params['nslice'] = string.atoi(params['nslice'])
    nslice = params['nslice']
    if(params.has_key('nvol')):
        params['nvol'] = string.atoi(params['nvol'])
    else:
        params['nvol'] = 1
    if(options.nvol_to_read > 0):
        params['nvol'] = options.nvol_to_read
    else:
        params['nvol'] = params['nvol'] - options.skip

    slice_pos = params['pss']
    thk = string.atof(params['thk'])
    min = 10.*string.atof(slice_pos[0])
    max = 10.*string.atof(slice_pos[nslice-1])
    if nslice > 1:
        gap = ((max - min + thk) - (nslice*thk))/(nslice - 1)
    else:
        gap = 0.
    params['gap'] = "%f" % gap

    params['orient'] =  params['orient'][1:-1]

    if(params.has_key('dwell')):
        dwell_time = string.atof(params['dwell'])

    if(pulse_sequence == 'epi'):
        nseg = string.atoi(params['petable'][-2])
    elif(pulse_sequence == 'tepi'):
        pulse_sequence = 'epi'   # "tepi" sequence is the same as the "epi" sequence.
        nseg = string.atoi(petable[-2])
    elif(pulse_sequence == 'epidw' or pulse_sequence == 'Vsparse'):
        pulse_sequence = 'epidw'
        nseg = string.atoi(petable[-1])
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
        params['petable'] = ""     #!!!!! WHAT IS THIS ? AFTER OBTAINING petable IT IS THEN JUST SENT TO NULL !!!!
        petab = zeros(n_pe)
        for i in range(n_pe):      # !!!!! WHAT IS THIS TOO ? !!!!!
            if i%2:
                 petab[i] = n_pe - i/2 - 1
            else:
                 petab[i] = i/2
    elif(pulse_sequence == 'sparse'):
        nseg = string.atoi(params['petable'][-2])
        pulse_sequence = 'epi'
        TR = TR + string.atof(params['quiet_interval'])
    else:
        print "Could not identify sequence: %s" % (pulse_sequence)
        sys.exit(1)

    params['tr'] = nseg*string.atof(params['tr'])
    params['nseg'] = nseg
    params['n_nav_echo'] = n_nav_echo
    params['n_pe_true'] =  params['n_pe'] - nseg*n_nav_echo
    params['xsize'] =  string.atof(params['fov'])/(string.atof(params['nv']) - nseg*n_nav_echo)
    params['ysize'] = params['xsize']
    params['zsize'] = string.atof(params['thk']) + string.atof(params['gap'])

    if(params['dp'] == '"y"'):
        datasize = 4
        params['num_type'] = Int32
    else:
        datasize = 2
        params['num_type'] = Int16

    # Parameters for compressed fid files.
    nseg = params['nseg']
    n_pe = params['n_pe']
    n_fe = params['n_fe']  
    nslice =  params['nslice']
    nvol = params['nvol'] 
    params['main_hdr'] = 32
    params['sub_hdr'] = 28
    params['line_len_data'] = datasize*n_fe
    params['line_len'] = params['line_len_data'] + params['sub_hdr']
    params['slice_len'] = n_fe*n_pe*datasize
    params['block_len'] = params['slice_len']*nslice

    # Compute the lengths of the different FID file according to format.
    len_asems_ncsnn = params['main_hdr'] + nvol*n_pe*(params['sub_hdr'] + nslice*n_fe*datasize)
    len_asems_nccnn = params['main_hdr'] + nvol*n_pe*nslice*(params['sub_hdr'] + n_fe*datasize)
    len_compressed = params['main_hdr'] + nvol*(params['sub_hdr'] + params['block_len'])
    len_uncompressed = params['main_hdr'] + nvol*nslice*(params['sub_hdr'] + params['slice_len'])
    len_epi2fid = params['main_hdr'] + nvol*nslice*(n_pe - nseg)*params['line_len']
    f_fid = open(options.fid_file,"r")

    # Open the actual FID file length
    f_fid.seek(0,2)
    file_length = f_fid.tell()
    f_fid.close()

    # Determine the file format by comparing the computed and actual file lengths.
    if len_compressed == file_length:
        fid_type = "compressed"
    elif len_uncompressed == file_length:
        fid_type = "uncompressed"
        params['block_len'] = nslice*(params['slice_len'] + params['sub_hdr'])
    elif len_epi2fid == file_length:
        fid_type = "epi2fid"
        phase_correct = 0
    elif len_asems_ncsnn == file_length:
        fid_type = "asems_ncsnn"
        phase_correct = 0
    elif len_asems_nccnn == file_length:
        fid_type = "asems_nccnn"
        params['block_len'] = params['line_len']
    else:
        print "Cannot recognize fid format, exiting."
        print "len_file: %d" % file_length
        print "\nlen_compressed: %d" % len_compressed
        print "len_uncompressed: %d" % len_uncompressed
        print "len_epi2fid: %d" % len_epi2fid
        print "len_asems_ncsnn: %d" % len_asems_ncsnn
        sys.exit(1)

    params['fid_type'] = fid_type

    return params


#*****************************************************************************
def get_data( context ):
    """
    This function reads the data from a fid file into the following arrays: 

    ksp_data: A rank 4 array containing time-domain data. This array is 
      dimensioned as ksp_data(nvol,nslice,n_pe_true,n_fe_true) where nvol 
      is the number of volumes, nslice is the number of slices per volume,
      n_pe_true is the number of phase-encode lines and n_fe_true is the
      number read-out points. Indices begun at 1. !!!! THE RANGE FOR THE PE
      ENCODES IS ..... (do similar stuff below) !!!!

    imag_nav_data: A rank 4 array containing time-domain data for the 
      navigator echoes of the image scan data which is dimensioned as 
      imag_nav_data(nvol,nslice,n_echo,n_fe_true) where n_echo is the 
      number navigator echoes per slice. Indices begun at 1.

    ref_data: A rank 4 array containing time-domain reference scan data 
      (phase-encode gradients are kept at zero). This array is dimensioned 
      as ref_data(nslice,n_pe_true,n_fe_true). 

    ref_nav_data: A rank 4 array containing time-domain data for the 
      navigator echoes of the reference scan data which is dimensioned as 
      ref_nav_data(nslice,n_echo,n_fe_true) where n_echo is the number 
      navigator echoes per slice. Indices begun at 1.
    """
    params = context.params
    options = context.options
    ksp_data = context.ksp_data
    ksp_nav_data = context.ksp_nav_data
    ref_data = context.ref_data
    ref_nav_data = context.ref_nav_data

    n_nav_echo = params['n_nav_echo']
    nseg = params['nseg']
    n_pe = params['n_pe']
    n_pe_true = params['n_pe_true']
    n_fe = params['n_fe']  
    n_fe_true = params['n_fe_true']
    nslice =  params['nslice']
    ydim = n_fe_true
    xdim = n_pe_true
    pulse_sequence = params['pulse_sequence']
    nvol = params['nvol']
    fid_type = params['fid_type']
    num_type = params['num_type']
    main_hdr = 32
    sub_hdr = 28 
    line_len_data = params['line_len_data']
    line_len = params['line_len']
    slice_len = params['slice_len']
    block_len = params['block_len']

    # !!!!!!!!!!!!!! IS THIS SMALL SECTION STILL NEEDED !!!!!!!!!!!!!!
    if(params.has_key('nvol')):
        nvol = params['nvol']
    else:
        nvol = 1

    # Read the phase-encode table and organize it into arrays which map k-space 
    # line number (recon_epi convention) to the collection order. These arrays 
    # will be used to read the data into the recon_epi as slices of k-space data. 
    # Assumes navigator echo is aquired at the beginning of each segment
    if len(params['petable']) > 0:
        petab = zeros(n_pe_true*nslice).astype(int)
        navtab = zeros(n_nav_echo*nseg*nslice).astype(int)
        petable_file = "recon_epi_pettabs" + "/" + params['petable']
        # Open recon_epi petable file
        f_pe = open(petable_file)                  
        petable_lines = f_pe.readlines()
        pe_per_seg = n_pe/nseg
 
        for seg in range(nseg):
            petable_lines[seg] = string.split(petable_lines[seg])
            for slice in range(nslice):
                for pe in range(n_pe_true):
                    index_1 = slice*n_pe_true + pe
                    if pulse_sequence == 'epidw' or pulse_sequence == 'epidw_se':
                        offset = slice*n_pe + seg*pe_per_seg 
                    else:
                        offset = (slice + seg*nslice)*pe_per_seg
                    try:
                        petab[index_1] = petable_lines[seg].index(str(pe)) + offset 
                    except:
                        pass
            
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
    else:
        petable_file = ""

    # Open FID file.
    f_fid = open(options.fid_file,"r") 

    # We use no data within the main header. Move the current file position past it.
    f_fid.seek(main_hdr)

    frame = options.sfn - 1
    ref_vol = 0
    ksp_vol = 0
    for vol in range(nvol):
        frame = frame + 1
        if frame == 1 and options.skip > 0:   #!!!!! NEED TO TEST THIS CONDITIONAL SECTION !!!!!
            # Skip data.
            if fid_type == 'compressed':
                # Skip phase data and skip blocks.
                pos = skip*(sub_hdr + block_len)
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
            shdr = struct.unpack('>hhhhlffff',f_fid.read(sub_hdr))
            bias[:] = complex(shdr[7],shdr[8])
            blk = fromstring(f_fid.read(block_len),num_type).byteswapped().astype(Float32).tostring()
            blk = fromstring(blk,Complex32)
            blk = reshape(blk,(nslice*n_pe,ydim))
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
            blk = reshape(blk,(nslice*n_pe,ydim))
            if pulse_sequence == 'epidw' or pulse_sequence == 'epidw_se':
                time_reverse = 0
            else:
                time_reverse = 1
        elif fid_type == "epi2fid":
            blk = zeros((nslice,nseg,n_pe/nseg,n_fe_true)).astype(Complex32)
            for seg in range(nseg):
                for slice in range(nslice):
                    for pe in range(n_pe/nseg-n_nav_echo):
                        position = ((pe + seg*(n_pe/nseg-n_nav_echo))*nvol*nslice + vol*nslice + slice)*line_len + main_hdr
                        f_fid.seek(position,0)
                        shdr = struct.unpack('>hhhhlffff',f_fid.read(sub_hdr))
                        bias[slice] = complex(0.,0.)
                        blk_line = fromstring(f_fid.read(line_len_data),num_type).byteswapped().astype(Float32).tostring()
                        blk_line = fromstring(blk_line,Complex32)
                        blk[slice,seg,pe+n_nav_echo,:] = blk_line
            blk = reshape(blk,(nslice*n_pe,ydim))
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
            blk = reshape(blk,(nslice*n_pe,ydim))
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
            blk = reshape(blk,(nslice*n_pe,ydim))
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
        navigators = zeros((nslice*options.n_echo,n_fe_true)).astype(Complex32)
        if pulse_sequence == 'epi':
            if fid_type == "epi2fid":
                ksp_image = reshape(blk,(nslice,n_pe_true,n_fe_true))
            else:
                ksp_image = take(blk,petab)
                navigators = take(blk,navtab) 
        elif pulse_sequence == 'epidw' or pulse_sequence == 'epidw_se':
            ksp_image = take(blk,petab)
            navigators = take(blk,navtab)
        elif pulse_sequence == 'epi_se':
            for slice in range(nslice):
                for seg in range(nseg):
                    ii = index = seg*pe_per_seg
                    for pe in range(ydim/2):
                        pe_out = petab[pe + n_nav_echo + seg*pe_per_seg]
                        pe_in = pe + ii
                        ksp_image[slice,pe_out,:] = blk[slice,pe_in,:]
        elif pulse_sequence == 'asems':
            blk = reshape(blk,(nslice,ydim,xdim))
            for slice in range(nslice):
                for pe in range(n_pe_true):
                    pe_out = pe
                    ksp_image[slice,pe_out,:] = blk[slice,pe,:]

        # Remove bias. Should work if all slices have same bias. 
        # !!!!! TRY WITH AND WITHOUT THIS BIAS SUBTRACTION LATER !!!!!
        ksp_image = reshape(ksp_image, (nslice, n_pe_true, n_fe_true))
        navigators = reshape(navigators, (nslice, options.n_echo, n_fe_true))
        for slice in range(nslice):
            ksp_image[slice,:,:] = (ksp_image[slice,:,:] - bias[slice])\
                                   .astype(Complex32)
            navigators[slice,:,:] = (navigators[slice,:,:] - bias[slice])\
                                    .astype(Complex32)

        # Place the appropriate data from blk into td_data, td_nav_data, ref_data and 
        # ref_nav_data arrays.
        if vol == 0:
            ref_data[:,:,:] = ksp_image[:,:,:]
            ref_nav_data[:,:,:] = navigators[:,:,:]
        else:
            ksp_data[vol,:,:,:] = ksp_image[:,:,:]
            ksp_nav_data[vol,:,:,:] = navigators[:,:,:]

    f_fid.close()

    return 


#*****************************************************************************
def fermi_filter( context ):
    ksp_data = context.ksp_data
    params = context.params

    n_nav_echo = params['n_nav_echo']
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
def fft_data( context ):
    params = context.params
    options = context.options
    ksp_data = context.ksp_data

    n_nav_echo = params['n_nav_echo']
    nseg = params['nseg']
    n_pe = params['n_pe']
    n_pe_true = params['n_pe_true']
    n_fe = params['n_fe']
    n_fe_true = params['n_fe_true']
    nslice =  params['nslice']
    ydim = n_fe_true
    pulse_sequence = params['pulse_sequence']
    nvol = params['nvol']
    fid_type = params['fid_type']

    if n_fe_true > n_pe_true: 
        xdim_out = n_fe_true
        ydim_out = n_fe_true
    else:
        xdim_out = n_pe_true
        ydim_out = n_pe_true

    # !!!! WHY IS THE CHECKERBOARD STUFF NEEDED !!!!
    checkerboard_mask = zeros((xdim_out,ydim_out)).astype(Float)
    line = zeros(ydim_out)
    for y in range(ydim_out):
        if y % 2:
            line[y] = 1
        else:
            line[y] = -1
    for x in range(xdim_out):
        if x % 2:
            checkerboard_mask[x,:] = line
        else:
            checkerboard_mask[x,:] = -line
    checkerboard_mask_1 = -checkerboard_mask[:,:]

    slice_order = zeros(nslice)

    print "Taking FFTs. Please Wait"
    for vol in range(nvol):
        for slice in range(nslice):
            # Take Fourier transform
            ksp = zeros((xdim_out,ydim_out)).astype(Complex32)
            ksp[0:n_pe_true,0:n_fe_true] = ksp_data[vol,slice,:,:]
            image = FFT.inverse_fft2d(ksp)
# DO WE REALLY WANT TO KEEP THIS CHECKERBOARD FILTER. ONLY IMPORTANT FOR COMPLEX VALUED IMAGE NOT MAGNITUDE IMAGE ?
#            image.real = image.real*checkerboard_mask
#            image.imag = image.imag*checkerboard_mask_1
            # !!!! WHAT IS ALL THIS SHIFTING ABOUT? !!!!
            if pulse_sequence == 'epi':
                if((slice) % 2 and fid_type != "epi2fid"):
                    idl.shift(image,0,ydim_out/2)
                else:
                    pass
                    idl.shift(image,0,ydim_out/2-1)    # Take out Menon one-voxel shift. JJO 1/22/03
                idl.shift(image,1,xdim_out/2)  
            elif pulse_sequence == 'epidw' or pulse_sequence == 'epidw_se':
                idl.shift(image,0,ydim_out/2-1)
                idl.shift(image,1,xdim_out/2)  
            elif pulse_sequence == 'asems':
                idl.shift(image,0,ydim_out/2)
                idl.shift(image,1,xdim_out/2)  
            image = transpose(image)

            if pulse_sequence == 'epi':
                if(slice % 2 and fid_type != "epi2fid"):
                    image = MLab.flipud(image)

            if options.flip_left_right:
                image = MLab.fliplr(image)
            if options.flip_top_bottom:
                image = MLab.flipud(image)

            # Reorder the slices from inferior to superior.
            if nslice % 2:
                midpoint = nslice/2 + 1
            else:
                midpoint = nslice/2
            if slice < midpoint:
                if options.flip_slices:
                    z = 2*slice
                else:
                    z = nslice-2*slice-1
            else:
                if  options.flip_slices:
                    z = 2*(slice-midpoint) + 1
                else:
                    z = nslice-2*(slice-midpoint)-2
            context.imag_data_complex[vol,z,:,:] = image[:,:].astype(Complex32)


#*****************************************************************************
def magnitude( complex_data ):
    "Take magnitude of the complex valued image."
    magnitude = abs(complex_data)
    # !!!! WHAT'S THIS SCALING ABOUT !!!!
    scale = 16383.0/magnitude.flat[argmax(magnitude.flat)]
    return multiply( scale, magnitude )


#*****************************************************************************
def save_data( context ):
    params = context.params
    options = context.options
    imag_data_mag = magnitude( context.imag_data_complex )
    n_nav_echo = params['n_nav_echo']
    nseg = params['nseg']
    n_pe = params['n_pe']
    n_pe_true = params['n_pe_true']
    n_fe = params['n_fe']
    n_fe_true = params['n_fe_true']
    nslice =  params['nslice']
    ydim = n_fe_true
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
    
    if n_fe_true > n_pe_true: 
        xdim_out = n_fe_true
        ydim_out = n_fe_true
    else:
        xdim_out = n_pe_true
        ydim_out = n_pe_true

    if options.phs_corr:
        frame_start = 1
    else:
        frame_start = 0

    # Save data to disk
    print "Saving to disk. Please Wait"
    vol_rng = range(frame_start, nvol)
    for vol in vol_rng:
        frame = vol + 1
        if options.save_first and vol == 0:
            img = zeros((nslice,ydim_out,xdim_out)).astype(Float32)
            for slice in range(nslice):
                if flip_left_right:
                    img[slice,:,:] = MLab.fliplr(abs(
                      imag_data_mag[vol,slice,:,:])).astype(Float32)
                else:
                    img[slice,:,:] = \
                      abs(imag_data_mag[vol,slice,:,:]).astype(Float32)
            file_io.write_cub(img,"EPIs.cub",xdim_out,ydim_out,nslice,xsize,
              ysize,zsize,0,0,0,"s",params)

        if  options.file_format == SPM_FORMAT:
            # Open files for this frame.
            options.img_file = "%s_%04d.img" % (img_stem,frame)
            # Write to disk.
            if options.output_data_type == MAGNITUDE_TYPE:
                hdr = file_io.create_hdr(xdim_out,ydim_out,nslice,1,xsize,
                  ysize,zsize,1.,0,0,0,'Short',64,1.,'analyze',
                  options.img_file,0)
            elif  options.output_data_type == MAGNITUDE_TYPE:
                hdr = file_io.create_hdr(xdim_out,ydim_out,nslice,1,xsize,
                  ysize,zsize,1.,0,0,0,'Complex',64,1.,'analyze',
                  options.img_file,0)
            file_io.write_analyze(
              options.img_file,hdr,imag_data_mag[vol,:,:,:])

        elif file_format == FIDL_FORMAT:
            # Write to disk.
            f_img.write(imag_data_mag[vol,:,:,:].astype(Float32)\
              .byteswapped().tostring())

    # Save navigator echo corrections.
    if options.file_format == VOXBO_FORMAT:
        # Write to disk.
        f_img = open(options.img_file,"w")
        file_io.write_tes(abs(imag_data_mag),options.img_file,xdim_out,
          ydim_out,nslice,nvol,xsize,ysize,zsize,0,0,0,TR,"s",params)

    elif options.file_format == FIDL_FORMAT:
        f_img.close()
        file_io.write_ifh(img_file_ifh,xdim_out,ydim_out,nslice,nvol,
          xsize,ysize,zsize)
        hdr = file_io.create_hdr(xdim_out,ydim_out,nslice,nvol,xsize,
          ysize,zsize,TR,0,0,0,'Float',32,1.,'analyze',options.img_file,1)
        file_io.write_analyze_header(img_file_hdr,hdr)


#*****************************************************************************
def phase_corr( context ):
    params = context.params
    options = context.options
    ksp_data = context.ksp_data
    ksp_nav_data = context.ksp_nav_data
    ref_data = context.ref_data
    ref_nav_data = context.ref_nav_data

    nvol = params['nvol']
    nslice = params['nslice']
    nseg = params['nseg']
    n_nav_echo = params['n_nav_echo']
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
#    if n_nav_echo > 0:
#        print "Data acquired with navigator echo time of %f" % (time0)
#        print "Data acquired with echo spacing of %f" % (time1)

    # Compute correction for Nyquist ghosts.
    # First and/or last block contains phase correction data.  Process it.
    phasecor_phs = zeros((nslice, n_pe_true, n_fe_true)).astype(Float)
    phasecor_ftmag = zeros((nslice, n_pe_true, n_fe_true)).astype(Float32)
    phasecor_total = zeros((nslice, n_pe_true, n_fe_true)).astype(Complex32)
    tmp = zeros(n_fe_true).astype(Complex32)
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
            pos_msk = where(phs > 0, 1.0, 0.0)
            msk1 = pos_msk*where(ft_blk.imag < 0, math.pi, 0.0)   # Re > 0, Im < 0
            msk2 = (1 - pos_msk)*where(ft_blk.imag < 0, 2.0*math.pi, math.pi) # Re < 0, Im < 0
            phs = mag_msk*(phs + msk1 + msk2) 
            phasecor_phs[slice,pe,:] = phs[:].astype(Float32)


#   Apply the phase correction to the image data.
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
