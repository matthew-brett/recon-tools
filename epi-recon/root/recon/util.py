import sys
import string 
import os
from Numeric import *
import file_io
import struct
from Numeric import empty
from FFT import inverse_fft
from pylab import pi, mlab, fft, fliplr, zeros, fromstring
from recon import petables, FIDL_FORMAT, VOXBO_FORMAT, SPM_FORMAT,MAGNITUDE_TYPE, COMPLEX_TYPE
from VolumeViewer import VolumeViewer


#-----------------------------------------------------------------------------
def shift(matrix, axis, shift):
    """
    axis: Axis of shift: 0=x (rows), 1=y (columns), 2=z (slices), etc...
    shift: Number of pixels to shift.
    """
    dims = matrix.shape
    ndim = len(dims)
    if axis >= ndim: raise ValueError("bad axis %s"%axis)
    axis_dim = ndim - 1 - axis

    # construct slices
    slices = [slice(0,d) for d in dims]
    slices_new1 = list(slices)
    slices_new1[axis_dim] = slice(shift, dims[axis_dim])
    slices_old1 = list(slices)
    slices_old1[axis_dim] = slice(0, -shift)
    slices_new2 = list(slices)
    slices_new2[axis_dim] = slice(0, shift)
    slices_old2 = list(slices)
    slices_old2[axis_dim] = slice(-shift, dims[axis_dim])

    # apply slices
    new = empty(dims, matrix.typecode())
    new[tuple(slices_new1)] = matrix[tuple(slices_old1)]
    new[tuple(slices_new2)] = matrix[tuple(slices_old2)]
    matrix[...] = new


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


#*****************************************************************************
class EmptyObject (object):
    "Takes whatever attributes are passed as keyword args when initialized."
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


#*****************************************************************************
def get_params(options):
    """
    This function parses the procpar file and returns parameter values related
    to the scan. Some parameters may be overridden by the options from the
    command line and hence the "options" list is an argument to this function.
    """
    import varian
    params = EmptyObject()
    procpar = varian.procpar(options.procpar_file)
    params.thk = procpar.thk[0]
    params.pss = procpar.pss
    params.nslice = len(procpar.pss)
    params.n_fe = procpar.np[0]
    params.n_pe = procpar.nv[0]
    params.tr = procpar.tr[0]
    params.petable = procpar.petable[0]
    params.nvol = procpar.images[0]
    params.te = procpar.te[0]
    params.gro = procpar.gro[0]
    params.trise = procpar.trise[0]
    params.gmax = procpar.gmax[0]
    params.at = procpar.at[0]
    params.orient = procpar.orient[0]
    pulse_sequence = procpar.pslabel[0]
    
    if procpar.get("spinecho", ("",))[0] == "y":
        if pulse_sequence == 'epidw': pulse_sequence = 'epidw_se'
        else: pulse_sequence = 'epi_se'

    # try using procpar.cntr or procpar.image to know which volumes are reference

    if pulse_sequence == 'mp_flash3d':
        params.nslice = procpar.nv2[0]
        params.thk = 10.*procpar.lpe2[0]
        slice_gap = 0
    else:
        slice_pos = params.pss
        min = 10.*slice_pos[0]
        max = 10.*slice_pos[-1]
        nslice = params.nslice
        if nslice > 1:
            slice_gap = ((max - min + params.thk) - (nslice*params.thk))/(nslice - 1)
        else:
            slice_gap = 0

    if options.TR > 0: params.tr = options.TR
    params.n_fe_true = params.n_fe/2
    n_pe = params.n_pe
    n_fe = params.n_fe 

    # Determine the number of navigator echoes per segment.
    params.nav_per_seg = params.n_pe%32 and 1 or 0

    if(options.nvol_to_read > 0):
        params.nvol = options.nvol_to_read
    else:
        params.nvol = params.nvol - options.skip

    # sequence-specific logic for determining pulse_sequence, petable and nseg
    # !!!!!! HEY BEN WHAT IS THE sparse SEQUENCE !!!!
    # Leon's "spare" sequence is really the EPI sequence with delay.
    if(pulse_sequence in ('epi','tepi','sparse','spare')):
        nseg = int(params.petable[-2])
    elif(pulse_sequence in ('epidw','Vsparse')):
        pulse_sequence = 'epidw'
        nseg = int(params.petable[-1])
    elif(pulse_sequence in ('epi_se','epidw_sb')):
        nseg = procpar.nseg[0]
        if string.rfind(params.petable,'epidw') < 0:
            petable = "epi%dse%dk" % (n_pe,nseg)
        else:
            pulse_sequence = 'epidw'
    elif(pulse_sequence == 'asems'):
        nseg = 1
        petab = zeros(n_pe)
        for i in range(n_pe):      # !!!!! WHAT IS THIS TOO ? !!!!!
            if i%2:
                 petab[i] = n_pe - i/2 - 1
            else:
                 petab[i] = i/2
    else:
        print "Could not identify sequence: %s" % (pulse_sequence)
        sys.exit(1)

    params.nseg = nseg
    params.pulse_sequence = pulse_sequence
    
    # this quiet_interval may need to be added to tr in some way...
    #quiet_interval = procpar.get("dquiet", (0,))[0]
    params.tr = nseg*params.tr
    params.nav_per_slice = nseg*params.nav_per_seg
    params.n_pe_true =  params.n_pe - params.nav_per_slice
    fov = procpar.lro[0]
    params.xsize = float(fov)/params.n_pe_true
    params.ysize = float(fov)/params.n_fe_true
    params.zsize = float(params.thk) + slice_gap

    params.datasize, params.num_type = \
      procpar.dp[0]=="y" and (4, Int32) or (2, Int16)

    return params


#*****************************************************************************
def initialize_data(params):
    "Allocate and initialize data arrays."
    nvol = params.nvol
    nslice = params.nslice
    n_nav = params.nav_per_slice
    n_pe_true = params.n_pe_true
    n_fe_true = params.n_fe_true
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
    nav_per_seg = params.nav_per_seg
    nseg = params.nseg
    n_nav = params.nav_per_slice
    n_pe = params.n_pe
    pe_per_seg = n_pe/nseg
    n_pe_true = params.n_pe_true
    pe_true_per_seg = n_pe_true/nseg
    n_fe = params.n_fe  
    n_fe_true = params.n_fe_true
    nslice =  params.nslice
    ydim = n_fe_true
    xdim = n_pe_true
    pulse_sequence = params.pulse_sequence
    nvol = params.nvol
    num_type = params.num_type
    datasize = params.datasize
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
    navtab = zeros(n_nav*nslice).astype(int)
    petable_file = os.path.join(petables, params.petable)
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
    # shift data for easy fft'ing
    shift(data_matrix, 0, n_fe_true/2)
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
    n_pe = params.n_pe
    n_pe_true = params.n_pe_true
    n_fe = params.n_fe
    n_fe_true = params.n_fe_true
    nslice =  params.nslice
    pulse_sequence = params.pulse_sequence
    xsize = params.xsize 
    ysize = params.ysize
    zsize = params.zsize

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
    vol_rng = range(frame_start, params.nvol)
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
    n_pe = params.n_pe
    n_pe_true = params.n_pe_true
    n_fe = params.n_fe
    n_fe_true = params.n_fe_true
    nslice =  params.nslice
    pulse_sequence = params.pulse_sequence
    xsize = params.xsize 
    ysize = params.xsize
    zsize = params.zsize

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
    vol_rng = range(params.nvol)
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
