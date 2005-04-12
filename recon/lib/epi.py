from Numeric import zeros,convolve,argmax,Float,Float32,Complex,Complex32,arange,array,where,greater,nonzero,put,take,ones,shape,sin,cos,exp,dot,argmin,reshape,conjugate,Int16,Int32,fromstring,transpose,equal,arctan,sum,cross_correlate,sqrt,arctan2
from math import pi
from idl import shift
from FFT import fft,inverse_fft
from  file_io import parse_procpar,dump_image,write_ifh,create_hdr,write_tes
from imageio import write_analyze, write_analyze_header
import sys
import string
import struct
import MLab

FIDL_FORMAT = "fidl"
VOXBO_FORMAT = "voxbo"
SPM_FORMAT = "spm"
MAG_THRESH = 0.
NONE = "none"
SINC = "sinc"
LINEAR = "linear"

# Types of navigator echo corrections:
NERD = "nerd"
PARTIAL_DORK = "partial_dork"
FULL_DORK = "full_dork"

#-----------------------------------------------------------------------------
def dump_complex_image(image,filename_stem,N_pe_true,N_fe_true,nseg,nslice,n_nav_echo,petab,reorder_segments,reorder_dump_slices,pulse_sequence,dump_phase,dump_mag):
    """
    Write a complex image to disk.  Called only when debug=1
    """

    s = shape(image)
    ydim = s[0]/nslice
    n_nav = n_nav_echo
    pe_per_seg = ydim/nseg
    image = reshape(image,(nslice*ydim,N_fe_true))
    if dump_phase:
        phs = phase(image)
    if reorder_segments:
        magout = zeros((nslice,ydim,N_fe_true)).astype(Float)
        tmp_phs = zeros((nslice,ydim,N_fe_true)).astype(Float)
        for slice in range(nslice):
            for seg in range(nseg):
                ii = line_index(slice,nslice,seg,nseg,pe_per_seg,pulse_sequence,n_nav)
                for pe in range(ydim/nseg-n_nav):
                    pe_in = pe + ii
                    if nseg > 1:
                        pe_out = petab[pe + n_nav + seg*pe_per_seg] - seg*n_nav
                    else:
                        pe_out = pe
                    if dump_phase:
                        tmp_phs[slice,pe_out,:] = phs[pe_in,:]
                    magout[slice,pe_out,:] = abs(image[pe_in,:])
    else:
         if (image.typecode() == 'F') or (image.typecode() == 'D'):
#            Complex numbers.
             if dump_phase:
                 tmp_phs = phs
             if dump_mag:
                 magout = abs(image)
         else:
             if dump_mag:
                 magout = image
    if dump_phase:
        dump_image(filename_stem+"_phs.4dfp.img",tmp_phs,N_fe_true,ydim,nslice,1,1,1,1,reorder_dump_slices,0)
    if dump_mag:
        dump_image(filename_stem + "_mag.4dfp.img",magout,N_fe_true,ydim,nslice,1,1,1,1,reorder_dump_slices,0)

    return(0)


#--------------------------------------------
def Fermi_filter(N_pe_true,N_fe_true,cutoff):
#--------------------------------------------
    """
    Compute Fermi filter for windowing k-space data.
    """
    Fermi_filter = zeros((N_pe_true,N_fe_true)).astype(Float32)
    F_FRAC = 0.9 # never used except to set cutoff below
    cutoff = F_FRAC
    for j in range(N_pe_true/2):
        r2 = j/(.5*N_pe_true)
        for k in range(N_fe_true/2):
            r3 = k/(.5*N_fe_true)
            r = r2*r2 + r3*r3
            filter = 1./(1. + exp((r-cutoff)*50.))
            Fermi_filter[N_pe_true/2  -j,N_fe_true/2-1+k] = filter
            Fermi_filter[N_pe_true/2-1+j,N_fe_true/2  -k] = filter
            Fermi_filter[N_pe_true/2  -j,N_fe_true/2  -k] = filter
            Fermi_filter[N_pe_true/2-1+j,N_fe_true/2-1+k] = filter
    return(Fermi_filter)


#-------------------------------------------------------------------------------------------
def fix_time_skew(img_vol,xdim,ydim,zdim,tdim,xdim_out,ydim_out,nslice,nseg,slice_order,options,pulse_sequence):
#-------------------------------------------------------------------------------------------
    """
    Correct for variations in slice timing within a frame.  For example, if 6 slices
    are acquired in the order [1,3,5,2,4,6] with a TR of 100ms per slice, slice 6
    will be acquired 500ms after slice 1, i.e., with a time-skew of 500ms.

    Ideal (sinc) interpolation is implemented for each voxel by first padding the
    time-course f(i) of length N by the sequence f(N-1),f(N-2),...,f(0) such that
    there are no discontinuities when the padded sequence is thought of as wrapping
    around the unit circle.  The padded sequence is then Fourier transformed, phase
    shifted by an amount  corresponding to the time skew, and then inverse 
    transformed to the temporal domain.
    """

    print "Correcting for slice timing skew."
    tc_pad = zeros(2*tdim).astype(Complex32)
    tc_rev = tdim - 1 - arange(tdim)
    tc_revm1 = tdim - 2 - arange(tdim-1)
    img_vol = reshape(img_vol,(tdim,zdim,ydim_out*xdim_out))
    for slice in range(nslice):
        sys.stdout.write(".")
        sys.stdout.flush()
        if options.time_interp == None:
            theta = pi*(float(slice_order[slice])/float(nslice*nseg))*arange(tdim)/tdim
        else:
            theta = pi*float(slice_order[slice]/float(nslice))*arange(tdim).astype(Float32)/tdim
        if(pulse_sequence == 'sparse'):
            theta = theta*(options.TR - 1.)/options.TR   # Adjust for silent period.
        phs_shift = zeros((2*tdim)).astype(Complex32)
        phs_shift[:tdim].real = cos(theta).astype(Float32)
        phs_shift[:tdim].imag = sin(theta).astype(Float32)
        phs_shift[tdim+1:] = conjugate(take(phs_shift[1:tdim],tc_revm1))
        for vox in range(xdim_out*ydim_out):
            if options.output_data_type == 'mag':
                tc_pad[:tdim].real = img_vol[:,slice,vox].astype(Float32)
                tc_pad[tdim:].real = take(img_vol[:,slice,vox],tc_rev).astype(Float32)
                tc_pad[:].imag = 0.
            elif options.output_data_type == 'complex':
                tc_pad[:tdim] = img_vol[:,slice,vox]
                tc_pad[tdim:] = take(img_vol[:,slice,vox],tc_rev)
            tc_fft = fft(tc_pad)
            tc_fft_phs = phs_shift*tc_fft
            tc = inverse_fft(tc_fft_phs).astype(Complex32)
            if options.output_data_type == 'mag':
                img_vol[:,slice,vox] = abs(tc[:tdim])
            elif options.output_data_type == 'complex':
                img_vol[:,slice,vox] = tc[:tdim]
    img_vol = reshape(img_vol,(tdim,zdim,ydim_out,xdim_out))
    print "\n"
    return(0)

    
#----------------------------------------------------------------------
def identify_pulse_sequence(params,pulse_sequence,options,tdim,nvol_r,nvol,N_pe):
#----------------------------------------------------------------------
    """
    Determine type of pulse sequence and salient parameters from the contents 
    of the procpar file.
    """
    petab = 0
    if(pulse_sequence == 'epi'):
        petable = params['petable']
        nseg = string.atoi(petable[-2])
    elif(pulse_sequence == 'tepi'):
        pulse_sequence = 'epi'   # "tepi" sequence is the same as the "epi" sequence.  
        petable = params['petable']
        nseg = string.atoi(petable[-2])
    elif pulse_sequence == 'epidw' or pulse_sequence == 'Vsparse':
        pulse_sequence = 'epidw'
        petable = params['petable']
        nseg = string.atoi(petable[-1])
        if(params.has_key('dquiet')):
            options.TR = options.TR + string.atof(params['quiet_interval'])
    elif(pulse_sequence == 'epi_se'): 
        if string.rfind(petable,'epidw') < 0:
            petable = "epi%dse%dk" % (N_pe,nseg)
        else:
            pulse_sequence = 'epidw'
        nseg = string.atoi(params['nseg'])
    elif(pulse_sequence == 'epidw_se'):
        petable = params['petable']
        nseg = string.atoi(params['nseg'])
    elif(pulse_sequence == 'asems'):
        options.phase_correct = 0
        nvol_r = nvol
        tdim = nvol_r
        nseg = 1
        petable = "epi%dse%dk" % (N_pe,nseg)
        petable = "64alt"
        params['petable'] = petable
        params['petable'] = ""
        petab = zeros(N_pe) 
        for i in range(N_pe):
            if i%2:
                 petab[i] = N_pe - i/2 - 1
            else:
                 petab[i] = i/2
    elif(pulse_sequence == 'sparse'):
        petable = params['petable']
        nseg = string.atoi(petable[-2])
        pulse_sequence = 'epi'
        options.TR = options.TR + string.atof(params['quiet_interval'])
    else:
        print "Could not identify sequence: %s" % (pulse_sequence)
        sys.exit(1)
    
    return(nseg,petable,petab)


#-------------------------------------------------------------------------
def line_index(slice,nslice,seg,nseg,pe_per_seg,pulse_sequence,n_nav_echo):
#-------------------------------------------------------------------------
    """
    Return index to the fid file from the slice, segment.
    """
    if pulse_sequence == 'epidw' or pulse_sequence == 'epidw_se':
        if n_nav_echo == 0:
            index = slice*nseg*pe_per_seg + seg*pe_per_seg
        else:
            index = (slice + seg*nslice)*pe_per_seg
    else:
        index = (slice + seg*nslice)*pe_per_seg
    return index


#-------------------------------------
def get_procpar(procpar_file,options):
    """
    Extract parameters of interest from the procpar file.
    Uses file_io.parse_procpar for reading procpar.

    Refers to the following values from file_io.parse_procpar:
      pulse_sequence
      tr
      np
      nv
      dp
      nslice
      nvol
      te
      trise
      gro
      max
      at
      pss
      thk
      gap
      orient
      dwell

    Modifies in-place:
      tr
      gap
    """

    params = parse_procpar(procpar_file,0)
    pulse_sequence = params['pulse_sequence']

    if options.TR > 0:
        params['tr'] = options.TR
    N_fe = string.atoi(params['np'])
    N_pe = string.atoi(params['nv'])
    N_fe_true = N_fe/2

    if(params['dp'] == '"y"'):
        datasize = 4
        num_type = Int32
    else:
        datasize = 2
        num_type = Int16
    nslice = string.atoi(params['nslice'])

    if(params.has_key('nvol')):
        nvol = string.atoi(params['nvol'])
    else:
        nvol = 1

    # determine number of volumes to read
    if options.nvol_to_read > 0:
        nvol_to_read = options.nvol_to_read + 1
    else:
        nvol_to_read = nvol - options.frames_to_skip

    te = string.atof(params['te'])
    trise = string.atof(params['trise'])
    gro = string.atof(params['gro'])
    gmax = string.atof(params['gmax'])
    at = string.atof(params['at'])
    ydim = N_fe_true
    if pulse_sequence == 'asems':
        tdim = nvol_to_read
    else:
        tdim = nvol_to_read - 1
    zdim = nslice
    slice_pos = params['pss']
    thk = string.atof(params['thk'])
    min = 10.*string.atof(slice_pos[0])
    max = 10.*string.atof(slice_pos[nslice-1])
    if nslice > 1:
        gap = ((max - min + thk) - (nslice*thk))/(nslice - 1)
    else:
        gap = 0.
    params['gap'] = "%f" % gap
    nv = string.atoi(params['nv'])
    orient =  params['orient'][1:-1]
    if(params.has_key('dwell')):
        dwell_time = string.atof(params['dwell'])
    else:
        dwell_time = -1.

    return(params,pulse_sequence,N_fe,N_pe,N_fe_true,datasize,num_type,nslice,nvol,nvol_to_read,te,trise,gro,gmax,at,ydim,tdim,zdim,slice_pos,thk,gap,nv,orient,dwell_time)

#----------------
def phase(image):
#----------------
    """
    Compute phase of a complex image.  This function should be superseded by the
    arctan2 function in scipy and is retained for historical reasons.
    """
#    msk = where(equal(image.real,0.),1.,0.)
#    phs = ((1.-msk)*arctan(image.imag/(image.real+msk))).astype(Float)
#    pos_msk = where(phs>0,1,0)
#    msk1 = pos_msk*where(image.imag<0,pi,0)
#    msk2 = (1-pos_msk)*where(image.imag<0,2.*pi,pi)
#    msk  = where((msk1+msk2) == 0,1,0)
#    phs = (phs + msk1 + msk2).astype(Float)
    return(arctan2(image.imag,image.real))

#------------------------------------------------------------------------------------
def read_block(f_fid,fid_type,MAIN_HDR,SUB_HDR,SLICE_LEN,LINE_LEN,LINE_LEN_DATA,N_pe,n_nav_echo,N_fe_true,nseg,nslice,nvol,N_pe_true,pulse_sequence,bias,num_type,vol):
#------------------------------------------------------------------------------------
    """
    Read one block of data from the fid file.  One block consists of a single 
    segment of k-space and includes all navigator echos.  Blocks are accessed in 
    the order of frames then slices  within each frame regardless of how the data
    are stored in the fid.
    """
    BLOCK_LEN = SLICE_LEN*nslice
    if fid_type == "compressed":
        shdr = struct.unpack('>hhhhlffff',f_fid.read(SUB_HDR))
        bias[:] = complex(shdr[7],shdr[8])
        blk = fromstring(f_fid.read(BLOCK_LEN),num_type).byteswapped().astype(Float32).tostring()
        blk = fromstring(blk,Complex32)
        blk = reshape(blk,(nslice*N_pe,N_fe_true))
        if pulse_sequence == 'epi_se':
            time_reverse = 0
        else:
            time_reverse = 1
    elif fid_type == "uncompressed":
        blk = zeros((nslice,N_pe*N_fe_true)).astype(Complex32)
        for slice in range(nslice):
            shdr = struct.unpack('>hhhhlffff',f_fid.read(SUB_HDR))
            bias[slice] = complex(shdr[7],shdr[8])
            blk_slc = fromstring(f_fid.read(SLICE_LEN),num_type).byteswapped().astype(Float32).tostring()
            blk[slice,:] = fromstring(blk_slc,Complex32)
        blk = reshape(blk,(N_pe*nslice,N_pe_true))
        if pulse_sequence == 'epidw' or pulse_sequence == 'epidw_se':
            time_reverse = 0
        else:
            time_reverse = 1
    elif fid_type == "epi2fid":
        blk = zeros((nslice,nseg,N_pe/nseg,N_fe_true)).astype(Complex32)
        for seg in range(nseg):
            for slice in range(nslice):
                for pe in range(N_pe/nseg-n_nav_echo):
                    position = ((pe + seg*(N_pe/nseg-n_nav_echo))*nvol*nslice + vol*nslice + slice)*LINE_LEN + MAIN_HDR
                    f_fid.seek(position,0)
                    shdr = struct.unpack('>hhhhlffff',f_fid.read(SUB_HDR))
                    bias[slice] = complex(0.,0.)
                    blk_line = fromstring(f_fid.read(LINE_LEN_DATA),num_type).byteswapped().astype(Float32).tostring()
                    blk_line = fromstring(blk_line,Complex32)
                    blk[slice,seg,pe+n_nav_echo,:] = blk_line
        blk = reshape(blk,(N_pe*nslice,N_pe_true))
        time_reverse = 0
    elif fid_type == "asems_ncsnn":
        blk = zeros((nslice,N_pe,N_fe_true)).astype(Complex32)
        for pe in range(N_pe):
            shdr = struct.unpack('>hhhhlffff',f_fid.read(SUB_HDR))
            bias1 = complex(shdr[7],shdr[8])
            for slice in range(nslice):
                position = (pe*nvol+vol)*(nslice*LINE_LEN_DATA + SUB_HDR) + slice*LINE_LEN_DATA + SUB_HDR + MAIN_HDR
                f_fid.seek(position,0)
                blk_line = fromstring(f_fid.read(LINE_LEN_DATA),num_type).byteswapped().astype(Float32).tostring()
                blk_line = fromstring(blk_line,Complex32)
                bias[slice] = complex(0.,0.)
                blk[slice,pe,:] = (blk_line - bias1).astype(Complex32)
        blk = reshape(blk,(N_pe*nslice,N_pe_true))
        time_reverse = 0
    elif fid_type == "asems_nccnn":
        blk = zeros((nslice,N_pe,N_fe_true)).astype(Complex32)
        for pe in range(N_pe):
            shdr = struct.unpack('>hhhhlffff',f_fid.read(SUB_HDR))
            bias1 = complex(shdr[7],shdr[8])
            for slice in range(nslice):
                position = (pe*nslice+slice)*nvol*LINE_LEN + vol*LINE_LEN + SUB_HDR + MAIN_HDR
                f_fid.seek(position,0)
                blk_line = fromstring(f_fid.read(LINE_LEN_DATA),num_type).byteswapped().astype(Float32).tostring()
                blk_line = fromstring(blk_line,Complex32)
                bias[slice] = complex(0.,0.)
                blk[slice,pe,:] = (blk_line - bias1).astype(Complex32)
        blk = reshape(blk,(N_pe*nslice,N_pe_true))
        time_reverse = 0
    else:
        print "Unknown type of fid file."
        sys.exit(1)

    return(blk,time_reverse)


#------------------------------------
def read_pe_table(petable_file,N_pe):
#------------------------------------
    """
    Read the phase encode table specified in the procpar.  This ascii file defines
    the order in which lines of k-space are written.
    """
    f_pe = open(petable_file)
    petable =f_pe.readlines()
    f_pe.close()
    i = 0
    i0 = -1
    i1 = -1
    for line in petable:
        line = string.lstrip(line)
        s = string.split(line)
        if len(s) > 0:
            if s[0] == 't1':
                i0 = i
            elif (s[0] == '=') and (i0 >= 0):
                i0 = i
            elif (line[0] == 't') and (i1 == -1):
                i1 = i-1
        i = i + 1
    if i1 < 0:
        i1 = i - 1
    s = ""
    for i in range(i1-i0+1):
        s = s + petable[i+i0]
    s = string.split(s)
    petab = zeros(N_pe)
    i = 0
    for ss in s:
        if (not ss == 't1') and (not ss == '='):
            petab[i] = string.atoi(ss)
            i = i + 1
    tmin = MLab.min(petab)
    petab = petab - tmin

    return(petab)


#-----------------------------------------------------------------------------------
def reorder_kspace(pulse_sequence,blk_interp,Fermi,ksp_image,N_fe_true,N_pe_true,nseg,nslice,xdim,ydim,zdim,tdim,vol,petab,n_nav_echo,fid_type,pe_per_seg):
#-----------------------------------------------------------------------------------
    """
    Reorder k-space according to the rules in the phase-encode table read from disk.
    """
    if pulse_sequence=='epi':
        if fid_type == "epi2fid":
            blk_interp = reshape(blk_interp,(tdim,zdim,N_pe_true,N_fe_true))
            for slice in range(nslice):
                for pe in range(ydim):
                    ksp_image[slice,pe,:] = blk_interp[vol,slice,pe,:]
                # Filter the data.
                for y in range(ydim):
                    ksp_image[slice,y,:] = ksp_image[slice,y,:]*Fermi[y,:]
        else:
            for slice in range(nslice):
                for seg in range(nseg):
                    ii = line_index(slice,nslice,seg,nseg,N_pe_true/nseg,pulse_sequence,n_nav_echo)
                    for pe in range(N_pe_true/2):
                        pe_in = pe + ii
                        pe_out = petab[pe + n_nav_echo + seg*(pe_per_seg)] - 1
                        if seg % 2 and nslice % 2:
                            # time_reverse second segment.
                            ksp_image[slice,pe_out,:] = take(blk_interp[vol,pe_in,:],time_rev)
                        else:
                            ksp_image[slice,pe_out,:] = blk_interp[vol,pe_in,:]
                    # Filter the data.
                    for y in range(ydim):
                        ksp_image[slice,y,:] = ksp_image[slice,y,:]*Fermi[y,:]
    elif pulse_sequence=='epidw' or pulse_sequence == 'epidw_se':
            for slice in range(nslice):
                for seg in range(nseg):
                    ii = line_index(slice,nslice,seg,nseg,ydim/nseg,pulse_sequence,n_nav_echo)
                    for pe in range(ydim/nseg):
                        pe_in = pe + ii
                        pe_out = petab[pe + seg*N_pe_true/nseg]
                        if n_nav_echo > 0:
                            if seg % 2 and nslice % 2:
                                # time_reverse second segment.
                                ksp_image[slice,pe_out,:] = take(blk_interp[vol,pe_in,:],time_rev)
                            else:
                                ksp_image[slice,pe_out,:] = blk_interp[vol,pe_in,:]
                        else:
                            ksp_image[slice,pe_out,:] = blk_interp[vol,pe_in,:]
                    # Filter the data.
                    for y in range(ydim):
                        ksp_image[slice,y,:] = ksp_image[slice,y,:]*Fermi[y,:]
    elif pulse_sequence == 'epi_se':
        for slice in range(nslice):
            for seg in range(nseg):
                ii = line_index(slice,nslice,seg,nseg,N_pe_true/nseg,pulse_sequence,n_nav_echo)
                for pe in range(ydim/2):
                    pe_out = petab[pe + n_nav_echo + N_pe_true]
                    pe_in = pe + ii
                    ksp_image[slice,pe_out,:] = blk_interp[vol,pe_in,:]
                # Filter the data.
                for pe in range(ydim):
                    ksp_image[slice,pe,:] = ksp_image[slice,pe,:]*Fermi[y,:]
    elif pulse_sequence == 'asems':
        blk_interp = reshape(blk_interp,(tdim,zdim,ydim,xdim))
        for slice in range(nslice):
            for pe in range(N_pe_true):
                pe_out = pe
                ksp_image[slice,pe_out,:] = blk_interp[vol,slice,pe,:]
            # Filter the data.
            for pe in range(ydim):
                ksp_image[slice,pe,:] = ksp_image[slice,pe,:]*Fermi[pe,:]

    else:
        print "Unknown pulse sequence in '"'reorder_kspace:'"' %s" % pulse_sequence
        return(-1)
    return(0)


#------------------------------------------------------------------------------------
def reorder_slices(image,ksp_vol,ksp_image,slice_order,options,xdim_out,ydim_out,zdim,N_pe,nslice,slice,vol,pulse_sequence,img_vol,fid_type):
#------------------------------------------------------------------------------------
    """
    Reorder slices such that the slice one is the most inferior slice. Images are
    shifted as required for each sequence and are flipped as appropriate for each 
    output image format.
    """
    if pulse_sequence == 'epi':
        if((slice) % 2 and fid_type != "epi2fid"):
            shift(image,0,ydim_out/2)
        else:
            shift(image,0,ydim_out/2-1)# Take out Menon one-voxel shift. JJO 1/22/03
        shift(image,1,xdim_out/2)
    elif pulse_sequence == 'epidw' or pulse_sequence == 'epidw_se':
        shift(image,0,ydim_out/2-1)
        shift(image,1,xdim_out/2)
    elif pulse_sequence == 'asems':
        shift(image,0,ydim_out/2)
        shift(image,1,xdim_out/2)
    image = transpose(image)

    if pulse_sequence == 'epi':
        if(slice % 2 and fid_type != "epi2fid"):
            image = MLab.flipud(image)

    if options.flip_left_right:
        image = MLab.fliplr(image)
    if options.flip_top_bottom:
        image = MLab.flipud(image)
    if options.phase_cor_nav_file and options.phase_correct:
        phasecor_total = reshape(phasecor_total,(zdim,N_pe,xdim))


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
        if options.lcksp:
            ksp_vol[vol,z,:,:] = ksp_image[slice,:,:].astype(Complex32)
        if (options.phase_cor_nav_file and (vol ==1)):
            f_phscor_nav.write((phasecor_total[z,:,:].astype(Float32)).byteswapped().tostring())
    else:
        if options.flip_slices:
            z = 2*(slice-midpoint) + 1
        else:
            z = nslice-2*(slice-midpoint)-2
        if options.lcksp:
            ksp_vol[vol,z,:,:] = ksp_image[slice,:,:].astype(Complex32)
        if options.phase_cor_nav_file and options.phase_correct:
            f_phscor_nav.write((phasecor_total[z,:,:].astype(Float32)).byteswapped().tostring())
    slice_order[z] = slice
    if options.field_map_file:
        img_vol.real[vol,z,:,:] = image
        img_vol.imag[vol,z,:,:] = 0.
    else:
        img_vol[vol,z,:,:] = image[:,:].astype(Complex32)

    return(0)


#-----------------------------------
def unwrap_phase(phase,len,verbose):
#-----------------------------------
    """
    Unwrap phase in a single line of data.
    """
    if len < 2:
        return 0.
    phase_unwrapped = zeros(len).astype(Float)
    wraps = 0.
    phase_unwrapped[0] = phase[0]
    for i in range(1,len): 
        slope = phase[i] - phase[i-1]
        if abs(slope) > pi:
            # Must be a wrap.
            if slope < 0:
                wraps = wraps + 2*pi
            else:
                wraps = wraps - 2*pi
        phase_unwrapped[i] = phase[i] + wraps
        if verbose:
            print "%d %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f" % (i,phase[i],phase[i-1],slope,slopem1,wraps, phase_unwrapped[i])
        slopem1 = slope
    return(phase_unwrapped)


#-----------------------------------------------------------------------------------
def detect_fid_format(filename,nframe,nseg,nslice,N_pe,N_fe,datasize,time_interp,phase_correct,pulse_sequence):
#-----------------------------------------------------------------------------------
    """
    #   Purpose:  Determine format of fid file from its length.
    #    nframe: Number of frames specified in procpar file.
    #    nslice: Number of slices.
    #    N_pe: Number of phase encodes including navigator echoes.
    #    N_fe: Number of frequency codes. 
    #    datasize: Number of bytes per value.
    """
    MAIN_HDR= 32
    SUB_HDR= 28

#   Compute number of bytes per frame
    len_asems_ncsnn = N_pe*(SUB_HDR + 2*nslice*N_fe*datasize)
    len_asems_nccnn = N_pe*nslice*(SUB_HDR + 2*N_fe*datasize)
    len_compressed = SUB_HDR + 2*nslice*N_fe*N_pe*datasize
    len_uncompressed = nslice*(SUB_HDR + 2*N_fe*N_pe*datasize)
    len_epi2fid = nslice*(N_pe-nseg)*(SUB_HDR + 2*datasize*N_fe)

    f_fid = open(filename,"r")
    f_fid.seek(0,2)
    file_length = f_fid.tell() - MAIN_HDR
    nframe_new = nframe
###print "len_file: %d\nlen_compressed: %d\nlen_uncompressed: %d" % (file_length,nframe*len_compressed,nframe*len_uncompressed)
    f_fid.seek(MAIN_HDR)
 
#   First, assume that the specified number of frames is correct.
    if nframe*len_compressed == file_length:
        fid_format = "compressed"
        print "fid file is in compressed format."
    elif nframe*len_uncompressed == file_length:
        fid_format = "uncompressed"
        print "fid file is in uncompressed format."
    elif nframe*len_epi2fid == file_length:
        print "*** fid file has been converted by epi2fid. ***"
        fid_format = "epi2fid"
        phase_correct = 0
        if time_interp != NONE:
            print "*** Interpolation cannot be done on a fid file in this format. ***"
            time_interp = NONE
    elif nframe*len_asems_ncsnn == file_length:
        fid_format = "asems_ncsnn"
        phase_correct = 0
        time_interp = NONE
        print "fid file is in asems_ncsnn format."
    elif nframe*len_asems_nccnn == file_length:
        fid_format = "asems_nccnn"
        print "fid file is in asems_nccnn format."
    else:
#       That didn't work, so estimate the number of frames for each format and
#       see if it is an integer .
        if file_length % len_compressed == 0:
            fid_format = "compressed"
            nframe_new = file_length/len_compressed
            print "fid file is in compressed format."
        elif file_length % len_uncompressed == 0:
            fid_format = "uncompressed"
            nframe_new = file_length/len_uncompressed
            print file_length,len_uncompressed
            print "fid file is in uncompressed format."
        elif file_length % len_epi2fid == 0:
            print "*** fid file has been converted by epi2fid. ***"
            nframe_new = file_length/len_epi2fid
            fid_format = "epi2fid"
            phase_correct = 0
            if time_interp != NONE:
                print "*** Interpolation cannot be done on a fid file in this format. ***"
                time_interp = NONE
            print "fid file is in epi2fid format."
        elif file_length % len_asems_ncsnn == 0:
            fid_format = "asems_ncsnn"
            nframe_new = file_length/len_asems_ncsnn
            phase_correct = 0
            time_interp = NONE
            print "fid file is in asems_ncsnn format."
        elif file_length % len_asems_nccnn == 0:
            fid_format = "asems_nccnn"
            nframe_new = file_length/len_asems_nccnn
            print "fid file is in asems_nccnn format."
        else:
#           Can'e identify format from its length, try using type of pulse sequence. I 
#           didn't do this first because at some point I didns't trust he correspondence 
#           between this procpar value and the fid format.
            if pulse_sequence == 'epi' or pulse_sequence == 'tepi' or pulse_sequence == 'sparse' or pulse_sequence == 'epidw' or pulse_sequence == 'Vsparse' or pulse_sequence == 'epi_se' or pulse_sequence == epidw_se:
                fid_format = 'compressed'
                nframe_new = file_length/len_compressed 
#                print "file_length % len_compressed: ",file_length % len_compressed
#                print "file_length % len_uncompressed: ",file_length % len_uncompressed
#                print "file_length % len_epi2fid: ",file_length % len_epi2fid
#                print "file_length % len_asems_ncsnn: ",file_length % len_asems_ncsnn
#                print "file_length % len_asems_nccnn: ",file_length % len_asems_nccnn
            else:
                print "*** Error: Cannot recognize fid format. ***"
                return(-1,-1)

    return nframe_new,fid_format


#---------------------------------------------------------------------------------------
class Recon_IO:
#   Writes non-debug images to disk.
    def __init__(self,img_file,options):
#       Setup output file names.
        if options.file_format == SPM_FORMAT:
            period = string.rfind(img_file,".img")
            if period < 0:
                period = string.rfind(img_file,".hdr")
        elif options.file_format == FIDL_FORMAT:
            period = string.rfind(img_file,".4dfp.img")
            if period < 0:
                period = string.rfind(img_file,".4dfp.ifh")
        elif options.file_format == VOXBO_FORMAT:
            period = string.rfind(img_file,".tes")
        else:
            print "Invalid image format: %s" % options.file_format
            period = -1
        if period < 0: 
            self.img_stem = img_file
        else:
            self.img_stem = img_file[:period]
        if options.file_format == FIDL_FORMAT:
            self.img_file     = "%s.4dfp.img" % (self.img_stem)
            self.img_file_hdr = "%s.4dfp.hdr" % (self.img_stem)
            self.img_file_ifh = "%s.4dfp.ifh" % (self.img_stem)
            if  options.lcphs: 
                self.phs_file     = "%s_phs.4dfp.img" % (self.img_stem)
                self.phs_file_hdr = "%s_phs.4dfp.hdr" % (self.img_stem)
                self.phs_file_ifh = "%s_phs.4dfp.ifh" % (self.img_stem)
                f_phase = open(phs_file,"w")
            if options.lcksp:    
                self.ksp_mag_file     = "%s_ksp_mag.4dfp.img" % (self.img_stem)
                self.ksp_mag_file_hdr = "%s_ksp_mag.4dfp.hdr" % (self.img_stem)
                self.ksp_mag_file_ifh = "%s_ksp_mag.4dfp.ifh" % (self.img_stem)
                self.ksp_phs_file     = "%s_ksp_phs.4dfp.img" % (self.img_stem)
                self.ksp_phs_file_hdr = "%s_ksp_phs.4dfp.hdr" % (self.img_stem)
                self.ksp_phs_file_ifh = "%s_ksp_phs.4dfp.ifh" % (self.img_stem)
                self.f_ksp_mag = open(self.ksp_mag_file,"w")
                self.f_ksp_phs = open(self.ksp_phs_file,"w")
            self.f_img = open(img_file,"w")
        elif options.file_format == VOXBO_FORMAT:
            self.img_file     = "%s.tes" % (self.img_stem)

#   --------------------------------------------------------
    def write_image_frame(self,img_vol,phs_vol,options,vol,xdim_out,ydim_out,nslice,xsize,ysize,zsize):
#   --------------------------------------------------------
#       Write images in image space to disk.
        frame = vol + 1
        if options.file_format == SPM_FORMAT:
            # Open files for this frame.
            img_file = "%s_%04d.img" % (self.img_stem,frame)
            # Write to disk.
            if options.output_data_type == 'mag':
                hdr = create_hdr(xdim_out,ydim_out,nslice,1,xsize,ysize,zsize,1.,0,0,0,'Short',64,1.,'analyze',img_file,0)
            elif options.output_data_type == 'complex':
                hdr = create_hdr(xdim_out,ydim_out,nslice,1,xsize,ysize,zsize,1.,0,0,0,'Complex',64,1.,'analyze',img_file,0)
            write_analyze(img_file,hdr,img_vol[vol,:,:,:])
            if options.lcphs:
                phs_file = "%s_phs_%04d.img" % (self.img_stem,frame)
                hdr = create_hdr(xdim_out,ydim_out,nslice,1,xsize,ysize,zsize,1.,0,0,0,'Float',32,1.,'analyze',phs_file,0)
                write_analyze(phs_file,hdr,phs_vol[vol,:,:,:])
        elif options.file_format == FIDL_FORMAT:
            # Write to disk.
            self.f_img.write(img_vol[vol,:,:,:].astype(Float32).byteswapped().tostring())
            if options.lcphs:
                self.f_phase.write(phs_vol[vol,:,:,:].astype(Float32).byteswapped().tostring())

#   -----------------------------------------------------------------------------------
    def write_ksp_frame(self,ksp_vol,options,frame,N_fe,N_pe,nslice,xsize,ysize,zsize):
#   -----------------------------------------------------------------------------------
#       Write images in kspace to disk.
        ksp_phs,ksp_mag = polar(ksp_vol[frame,:,:,:])
        if options.file_format == SPM_FORMAT:
            ksp_mag_file = "%s_ksp_mag_%04d.img" % (self.img_stem,frame)
            ksp_phs_file = "%s_ksp_phs_%04d.img" % (self.img_stem,frame)
            hdr = create_hdr(N_fe,N_pe,nslice,1,xsize,ysize,zsize,1.,0,0,0,'Float',32,1.,'analyze',ksp_mag_file,0)

            write_analyze(ksp_mag_file,hdr,ksp_mag)
            hdr = create_hdr(N_fe,N_pe,nslice,1,xsize,ysize,zsize,1.,0,0,0,'Float',32,1.,'analyze',ksp_phs_file,0)
            write_analyze(ksp_phs_file,hdr,ksp_phs)
        elif options.file_format == FIDL_FORMAT:
            self.f_ksp_mag.write(ksp_mag.astype(Float32).byteswapped().tostring())
            self.f_ksp_phs.write(ksp_phs.astype(Float32).byteswapped().tostring())

#   -----------------------------------------------------------------------------------------------
    def write_image_final(self,options,xdim_out,ydim_out,nslice,tdim,xsize,ysize,zsize,tr,N_fe,N_pe,img_vol,params):
#   -----------------------------------------------------------------------------------------------
        if options.file_format == VOXBO_FORMAT:
            # Write to disk.
            f_img = open(self.img_file,"w")
            print self.img_file
            write_tes(abs(img_vol),self.img_file,xdim_out,ydim_out,nslice,tdim,xsize,ysize,zsize,0,0,0,tr,"s",params)
        elif options.file_format == FIDL_FORMAT:
            self.f_img.close()
            write_ifh(self.img_file_ifh,xdim_out,ydim_out,nslice,tdim,xsize,ysize,zsize)
            hdr = create_hdr(xdim_out,ydim_out,nslice,tdim,xsize,ysize,zsize,tr,0,0,0,'Float',32,1.,'analyze',self.img_file,1)
            write_analyze_header(self.img_file_hdr,hdr)
            print "Images written to %s" % (self.img_file)
            if options.lcphs:
                f_phase.close()
                write_ifh(self.phs_file_ifh,xdim_out,ydim_out,nslice,tdim,xsize,ysize,zsize)
                hdr = create_hdr(xdim_out,ydim_out,nslice,tdim,xsize,ysize,zsize,tr,0,0,0,'Float',32,1.,'analyze',phs_file,1)
                write_analyze_header(self.phs_file_hdr,hdr)
                print "Phase data written to %s" % (phs_file)
            if options.lcksp:
                self.f_ksp_mag.close()
                write_ifh(self.ksp_mag_file_ifh,N_fe,N_pe,nslice,tdim,xsize,ysize,zsize)
                write_ifh(self.ksp_phs_file_ifh,N_fe,N_pe,nslice,tdim,xsize,ysize,zsize)
                self.f_ksp_phs.close()


#------------------------------------------------------------------------------------------
def compute_linear_offset(phase_data,nslice,nseg,N_pe_seg,N_fe,N_nav,options,mode=0):
#------------------------------------------------------------------------------------------
    """
    The central pixel of each line drifts linearly with line number. This routine 
    uses the reference phase data to estimate the linear drift.
    """

# mode: Sets method used.
#       0: Do a two stage estimate for first two lines, then use odd-even differences to compute a 
#          per-line offset.  Accumulate these to form final offsets.
#       1: Used for refining estimate.


#   First compute the slope of the phase difference between two adjacent lines.
    THRESH = 0.00 # Need for threshold eliminated by weighted least squares in linfit. 
    f0 = zeros((nslice,nseg,N_pe_seg),Float)
    correction = zeros((N_fe),Complex)
    dphi = zeros((N_fe),Complex)
    dphi_phs = zeros((N_fe),Float)
    dphi_phs_sum = zeros((N_fe),Float)
    d_sum = zeros((N_fe),Float)
    line = zeros((N_fe),Complex)
    linem1 = zeros((N_fe),Complex)
    linem2 = zeros((N_fe),Complex)
    linea = zeros((N_fe),Complex)
    lineb = zeros((N_fe),Complex)
    ramps = zeros((N_fe),Float)
    ramps[:N_fe/2] = arange(N_fe/2)
    ramps[N_fe/2:] = -(N_fe/2 - arange(N_fe/2))
    dmsk = zeros((N_fe),Float)
    dmsk[:N_fe/4] = 1.
    dmsk[-N_fe/4:] = 1.
    nzd = nonzero(dmsk)
    if options.debug:
        dphi_phs_ref = zeros((nslice,nseg,N_pe_seg,N_fe),Float)
        ddphi_phs_ref = zeros((nslice,nseg,N_pe_seg,N_fe),Float)
    for slice in range(nslice):
        for seg in range(nseg):
            linem1[:] = inverse_fft(phase_data[slice,seg,0,:])
            line[:] = inverse_fft(phase_data[slice,seg,1,:])
            dphi_phs_sum[:] = 0.
            d_sum[:] = 0.
            for pe in range(N_pe_seg-1):
                if pe == 0:
                    linem2[:] = inverse_fft(phase_data[slice,seg,0,:])
                    line[:] = inverse_fft(phase_data[slice,seg,1,:])
                elif nseg == 1:
                    linem2[:] = line[:]
                    line[:] = inverse_fft(phase_data[slice,seg,pe+1,:])
                else:
                    linem2[:] = linem1[:]
                    linem1[:] = line[:]
                    line[:] = inverse_fft(phase_data[slice,seg,pe+1,:])

                ma = abs(line)
                mb = abs(linem2)
                d = ma*mb
                msk = d > THRESH*d[argmax(d)]
                d = d/sum(d)
                nz = nonzero(msk)
#               Take difference between the phase at points on adjacent lines.
                dphi = msk*line*conjugate(linem2)/(d + (1 - msk)) # Subtract angles
                dphi_phs[:] = arctan2(dphi.imag,dphi.real)

                if pe == 0 and mode == 0:
                    # Get rough estimate of misalignment between first and second line.
                    offset_even = 0.
                    xcor = cross_correlate(abs(phase_data[slice,seg,0,:]),abs(phase_data[slice,seg,1,:]),1)
                    offset_odd = -2.*pi*(N_fe/2 - argmax(xcor.flat))/float(N_fe)

                    # Refine the estimate by correcting rough estimate and remeasuring.
                    correction.real = cos(ramps*offset_even).astype(Float)
                    correction.imag = sin(ramps*offset_even).astype(Float)
                    linea[:] = inverse_fft(phase_data[slice,seg,0,:])*correction

                    correction.real = cos(ramps*offset_odd).astype(Float)
                    correction.imag = sin(ramps*offset_odd).astype(Float)
                    lineb[:] = inverse_fft(phase_data[slice,seg,1,:])*correction
                    d = abs(linea*lineb)

                    dphi = linea*conjugate(lineb) # Subtract angles
                    dphi_phs[:] = arctan2(dphi.imag,dphi.real)
                    (f00,intercept) = fit_line(take(ramps,nzd),take(dphi_phs[:],nzd),take(dmsk,nzd))
                    f0[slice,seg,0] = offset_even 
                    f0[slice,seg,1] = offset_odd  + f00
                else:
                    if mode == 0:
                        (f00,intercept) = fit_line(take(ramps,nzd),take(dphi_phs[:],nzd),take(dmsk,nzd))
                        if nseg == 2:
                            f0[slice,seg,pe+1] =  f0[slice,seg,pe-1] - f00
                        else:
                            f0[slice,seg,pe+1] =  f0[slice,seg,pe] - f00
                    elif mode == 1 and pe > 2:
                        dphi_phs_sum = dphi_phs_sum + dphi_phs/float(N_pe_seg-3.)
                        d_sum = d_sum + d/float(N_pe_seg-3.)

                if options.debug:
                    dphi_phs_ref[slice,seg,pe+1,:] = dphi_phs[:]
                    ddphi_phs_ref[slice,seg,pe+1,:] = d_sum

            if mode == 1:
                (f00,intercept) = fit_line(take(ramps,nzd),take(dphi_phs_sum[:],nzd),take(dmsk,nzd))
                f0[slice,seg,0] = 0.
                for pe in range(N_pe_seg-1):
                    f0[slice,seg,pe+1] =  f0[slice,seg,pe] - f00/2
    
#           Filter offset coefficients.
            f0[slice,seg,:] = filter_coefficient(f0[slice,seg,:],N_pe_seg,N_nav)

    if options.debug:
        dphi_phs_ref = reshape(dphi_phs_ref,(nslice,nseg*N_pe_seg,N_fe))
        ddphi_phs_ref = reshape(ddphi_phs_ref,(nslice,nseg*N_pe_seg,N_fe))
        dump_image("dphi_phs_ref.4dfp.img",dphi_phs_ref,N_fe,nseg*N_pe_seg,nslice,1,1,1,1,0,0)
        dump_image("ddphi_phs_ref.4dfp.img",ddphi_phs_ref,N_fe,nseg*N_pe_seg,nslice,1,1,1,1,0,0)

    return(f0)

#---------------------------------------------------------------------------------------
def compute_linear_phase_corr(ref_data,bias,nslice,nseg,N_pe_seg,N_fe,N_nav,options):
#---------------------------------------------------------------------------------------

    ref_data = reshape(ref_data,(nslice,nseg,N_pe_seg,N_fe))
    dphi_ref_N_dc = zeros((nslice,nseg,N_fe)).astype(Complex)
    phi_ref_dc = zeros((nslice,nseg,N_fe)).astype(Complex)
    dphi_ref_N_dc.real[:] = 1.
    phi_ref_dc.real[:] = 1.
    dork_data = (phi_ref_dc,dphi_ref_N_dc)
    f0_ref = zeros((nslice,nseg,N_pe_seg),Float)
    if options.debug:
        phase_data_ref = zeros((nslice,nseg,N_pe_seg,N_fe),Complex)
        phase_data_corr = zeros((nslice,nseg,N_pe_seg,N_fe),Complex)
        phase_data_ref = zeros((nslice,nseg,N_pe_seg,N_fe),Float)

#   Compute the linear parameters of the phase error for the reference scan.
    f0_ref = compute_linear_offset(ref_data,nslice,nseg,N_pe_seg,N_fe,N_nav,options)

#   Correct for offset errors with no phase correction.
    theta_corr = zeros((nslice,nseg,N_pe_seg),Complex)
    theta_corr.real[:] = 1.
    offset_corr = compute_phase_coefficients(f0_ref,nslice,nseg,N_pe_seg,N_fe,options)
    phase_data_corr = correct_phase_errors_linear(ref_data,bias,offset_corr,theta_corr,nslice,nseg,nseg*N_pe_seg,N_fe,N_nav,dork_data,1.,1.,options,1)

#   Refine the offset correction.
    phase_data_corr = reshape(phase_data_corr,(nslice,nseg,N_pe_seg,N_fe))
    mask = zeros((nslice,nseg,N_pe_seg,N_fe),Float)
    mask[:,:,:,N_fe/4:3*N_fe/4] = 1
    f0_ref2 = compute_linear_offset(mask*phase_data_corr,nslice,nseg,N_pe_seg,N_fe,N_nav,options,0)
    f0_ref = f0_ref + f0_ref2

#   Compute the final correction terms excluding the navigator echo stuff.
    offset_corr = compute_phase_coefficients(f0_ref,nslice,nseg,N_pe_seg,N_fe,options)

#   Compute the line-to-line phase correction from the offset-corrected data.
    phase_data_corr = correct_phase_errors_linear(ref_data,bias,offset_corr,theta_corr,nslice,nseg,nseg*N_pe_seg,N_fe,N_nav,dork_data,1.,1.,options,1)
    phase_data_corr = reshape(phase_data_corr,(nslice,nseg,N_pe_seg,N_fe))
    for slice in range(nslice):
        for seg in range(nseg):
            dtht = phase_data_corr[slice,seg,:,N_fe/2]
            theta_corr[slice,seg,:] = conjugate(dtht)/abs(dtht)

#   Save data for Pfeuffer's dork data.  First line is the difference in phase between the
#   navigator echo and the dc line in the refrence data, the second line is the dc line in the
#   reference data.
    if N_nav > 0:
#        phi_ref_N[slice,seg,:] = phase_data_corr[slice,seg,0,:]/abs(phase_data_corr[slice,seg,0,:])
        phi_ref_dc[slice,seg,:] = phase_data_corr[slice,seg,N_nav,:]/abs(phase_data_corr[slice,seg,N_nav,:])
        dphi_ref_N_dc[slice,seg,:] = phase_data_corr[slice,seg,0,:]*conjugate(phase_data_corr[slice,seg,N_nav,:])/abs(phase_data_corr[slice,seg,N_nav,:]*phase_data_corr[slice,seg,0,:])
    dork_data = (phi_ref_dc,dphi_ref_N_dc)

    if options.debug:
        tmp_image = correct_phase_errors_linear(ref_data,bias,offset_corr,theta_corr,nslice,nseg,nseg*N_pe_seg,N_fe,N_nav,dork_data,1.,1.,options,0)
        tmp_image = reshape(tmp_image,(nslice,nseg,N_pe_seg-N_nav,N_fe))
#       Flip and join segments for display.
        if nseg == 2:
            tmp = zeros((nslice,nseg,N_pe_seg-N_nav,N_fe),Complex)
            for islc in range(nseg*nslice):
                slcin = islc/nseg
                segin = islc % nseg
                slcout = islc % nslice
                segout = islc/nslice
                if segout:
                    tmp[slcout,segout,:,:] = tmp_image[slcin,segin,:,:]
                else:
                    for pe in range(N_pe_seg-N_nav):
                        tmp[slcout,segout,pe,:] = tmp_image[slcin,segin,N_pe_seg-N_nav-pe-1,:]
        else:
            tmp = tmp_image
        dump_image("phasedata_lincor_final.4dfp.img",abs(tmp),(N_pe_seg-N_nav)*nseg,N_fe,nslice,1,1,1,1,0,0)
        dump_image("dtheta_ref.4dfp.img",arctan2(theta_corr.imag,theta_corr.real),nseg*N_pe_seg,nslice,1,1,1,1,1,0,0)

    if options.debug:
        dump_image("phase_data_mag_ref.4dfp.img",abs(ref_data),N_fe,nseg*N_pe_seg,nslice,1,1,1,1,0,0)
        dump_image("phase_data_phs_ref.4dfp.img",phase(ref_data),N_fe,nseg*N_pe_seg,nslice,1,1,1,1,0,0)
        dump_image("phase_data_corr_mag.4dfp.img",abs(phase_data_corr),N_fe,nseg*N_pe_seg,nslice,1,1,1,1,0,0)
        dump_image("phase_data_corr_phs.4dfp.img",arctan2(phase_data_corr.imag,phase_data_corr.real),N_fe,nseg*N_pe_seg,nslice,1,1,1,1,0,0)
        dump_image("f0_ref.4dfp.img"    ,f0_ref,nseg*N_pe_seg,nslice,1,1,1,1,1,0,0)
        dump_image("f0_ref2.4dfp.img"    ,f0_ref2,nseg*N_pe_seg,nslice,1,1,1,1,1,0,0)

    return(offset_corr,theta_corr,dork_data)


#------------------------------------------------------
def compute_phase_coefficients(f0,nslice,nseg,N_pe_seg,N_fe,options):
#------------------------------------------------------

    ramps = zeros((N_fe),Float)
    ramps[:N_fe/2] =  arange(N_fe/2)
    ramps[N_fe/2:] = -(N_fe/2 - arange(N_fe/2))
    offset_corr = zeros((nslice,nseg,N_pe_seg,N_fe),Complex)
    f0 = reshape(f0,(nslice,nseg,N_pe_seg))
    for slice in range(nslice):
        for seg in range(nseg):
            for pe in range(N_pe_seg):
                offset_corr.real[slice,seg,pe,:] = cos(ramps*f0[slice,seg,pe]).astype(Float)
                offset_corr.imag[slice,seg,pe,:] = sin(ramps*f0[slice,seg,pe]).astype(Float)
    return(offset_corr)


#-------------------------------------------------------------------------------------
def correct_phase_errors_linear(bk,bias,offset_corr,theta_corr,nslice,nseg,N_pe,N_fe,N_nav,dork_data,te,time1,options,lc_calc=0,):
#-------------------------------------------------------------------------------------

#   lc_calc = 1 implies that returned array will include the navigator echos because scatter 
#               correction is being calculated.

    N_pe_true = N_pe - nseg*N_nav
    if lc_calc:
        N_pe_local = N_pe
    else:
        N_pe_local = N_pe_true
    N_pe_seg = N_pe_local/nseg
    blk = zeros((nslice,nseg,N_pe_local/nseg,N_fe),Complex)
    bk = reshape(bk,(nslice,nseg,N_pe/nseg,N_fe))
    phi_ref_dc,dphi_ref_N_dc = dork_data
    for slice in range(nslice):
        for seg in range(nseg):
#           First do the offset correction
            for pe in range(N_pe_seg):
                x = inverse_fft(bk[slice,seg,pe,:]-bias[slice])
                cfl = offset_corr[slice,seg,pe,:]*x
                if lc_calc:
                    blk[slice,seg,pe,:] = fft(cfl)
                else:
                    if pe == 0:
                        if N_nav == 0:
                            phi_img_dc = blk[slice,seg,0,:]
                        else:
                            phi_img_N = blk[slice,seg,0,:]
                        blk[slice,seg,0,:] = fft(cfl)
                    elif pe == N_nav:
                        phi_img_dc = blk[slice,seg,N_nav,:]
                        blk[slice,seg,pe-N_nav,:] = fft(cfl)
                    elif pe > N_nav:
                        blk[slice,seg,pe-N_nav,:] = fft(cfl)
            if not lc_calc:
#               Correct line-to-line phase offset. Do Pfeuffer correction if commanded.
                if options.navcor_type == NERD:
                    nav_cor = zeros((N_fe),Complex)
                    nav_cor.real[:] = 1.
                elif options.navcor_type == PARTIAL_DORK:
                    timex = time1*float((N_pe_seg - 1))/2.
                    domega =  phi_img_dc*conjugate(phi_ref_dc[slice,seg,:])
                    domega = domega/abs(domega)
                elif options.navcor_type == FULL_DORK:
                    timex = time1*float((N_pe_seg))/2.
                    domega = dphi_ref_N_dc[slice,seg,:]*phi_img_dc*conjugate(phi_img_N)
                    domega = domega/abs(domega)
                    dphi = pow(conjugate(domega),te/timex)
                    dphi = dphi*phi_ref_dc[slice,seg,:]*conjugate(phi_img_dc)/abs(phi_img_dc)
                    dphi = pow(dphi,1./N_pe_seg)
                else:
                    nav_cor = zeros((N_fe),Complex)
                    nav_cor.real[:] = 1.
                for pe in range(N_pe_seg):
                    if options.navcor_type == PARTIAL_DORK:
                        nav_cor = (pow(domega,(time1*float(pe))/te))
                    elif options.navcor_type == FULL_DORK:
                        nav_cor = conjugate(dphi*pow(domega,float(pe)/float(N_pe_seg)))
                    blk[slice,seg,pe,:] = theta_corr[slice,seg,pe+N_nav]*nav_cor[N_fe/2]*blk[slice,seg,pe,:]
    bk = reshape(bk.astype(Complex32),(nslice,N_pe,N_fe))
    blk = reshape(blk.astype(Complex32),(nslice,N_pe_local,N_fe))

    if options.debug:
#       Save uncorrected and corrected data.
        dump_image("blk_mag.4dfp.img",abs(bk),N_fe,N_pe,nslice,1,1,1,1,0,0)
        bk_phs = phase(bk)
        dump_image("blk_phs.4dfp.img",bk_phs,N_fe,N_pe,nslice,1,1,1,1,0,0)
        dump_image("blk_corrected_mag.4dfp.img",abs(blk[:,:,:]),N_fe,N_pe_local,nslice,1,1,1,1,0,0)
        blk_phs = phase(blk[:,:,:])
        dump_image("blk_corrected_phs.4dfp.img",blk_phs,N_fe,N_pe_local,nslice,1,1,1,1,0,0)

    blk = reshape(blk.astype(Complex32),(nslice*N_pe_local,N_fe))
    return(blk)



#----------------------------------------------
def filter_coefficient(f0,N_pe_seg,N_nav):
#----------------------------------------------
#
# Extract forward and reverse time readout directions for each segment.
    idx = arange(N_pe_seg-N_nav)
    idx_fit = arange((N_pe_seg-N_nav)/2)
    idx_out = arange((N_pe_seg-N_nav)/2)
    x = arange((N_pe_seg-N_nav)/2)
    corrected = arange(N_pe_seg/2).astype(Float)
    weights = ones(((N_pe_seg-N_nav)/2),Float)
    outweights = idx_fit*(idx_fit-1)/2.
    linear_corr = zeros((N_pe_seg),Float)

#   Extract data for the even lines while ignoring navigator echoes.
    fit_data = take(f0[N_nav:],2*idx_fit)
#   Fit a straight line to each segment
    me,be = fit_line(x,fit_data,weights)
#   Subtract the linear trend from each point and store in the output.
    corrected[:] = me*x + be
    put(linear_corr[N_nav:],2*idx_out,corrected)

#   Now do the odd lines.
    fit_data = take(f0[N_nav:],2*idx_fit+1)
#   Fit a straight line to each segment
    mo,bo = fit_line(x,fit_data,weights)
    corrected[:] = mo*x + bo
    put(linear_corr[N_nav:],2*idx_out+1,corrected)

#   Store the results for output.
    f0_filt = zeros((N_pe_seg),Float)
    f0_filt[:N_nav] = f0[:N_nav]
    f0_filt = f0_filt + linear_corr

    return(f0_filt)
                                       

#-----------------------------------------------------------------------
def compute_menon_phase_correction(blk,nslice,nseg,N_pe,N_fe_true,bias,nvol_r,options,n_nav_echo,petab,reorder_segments,reorder_dump_slices,pulse_sequence):
#-----------------------------------------------------------------------

#   Compute point-by-point phase correction copied from epi2fid.c

    # Compute correction for Nyquist ghosts.
    # First and/or last block contains phase correction data.  Process it.
    N_pe_true = N_pe - nseg*n_nav_echo
    correction = zeros((nslice*N_pe,N_fe_true)).astype(Float)
    dork_data = zeros((nslice,nseg,2,N_fe_true)).astype(Float)
    phasedat_phs = zeros((nslice*N_pe,N_fe_true)).astype(Float32)
    phasedat_mag = zeros((nslice*N_pe,N_fe_true)).astype(Float32)
    phasecor_ftmag = zeros((nslice*N_pe,N_fe_true)).astype(Float32)
    phs_correction = zeros((nslice*N_pe,N_fe_true)).astype(Float32)
    slopes = zeros((nslice,nseg)).astype(Float32)
    phs_test = zeros((N_pe,N_fe_true)).astype(Complex32)
    tmp     = zeros((N_fe_true)).astype(Complex32)
    if options.debug:
        dump_complex_image(blk,"phasedata",N_pe_true,N_fe_true,nseg,nslice,n_nav_echo,petab,reorder_segments,reorder_dump_slices,pulse_sequence,0,1)
    # Compute point-by-point phase correction
    for pe in range(N_pe*nslice):
        slice = pe/N_pe
        seg = ((pe - slice*N_pe)*nseg)/N_pe
        pep = pe - slice*N_pe - seg*N_pe/nseg
        tmp[:] = (blk[pe,:] - bias[slice]).astype(Complex32)
        shift(tmp,0,N_fe_true/2)
        ft_blk = inverse_fft(tmp)
        shift(ft_blk,0,N_fe_true/2)
        msk = where(equal(ft_blk.real,0.),1.,0.)
        phs = (1.-msk)*arctan(ft_blk.imag/(ft_blk.real+msk))
        phasecor_ftmag[pe,:] = abs(ft_blk).astype(Float32)
        phasedat_mag[pe,:] = abs(blk[pe,:]).astype(Float32)

        # Create mask for threshold of MAG_THRESH for magnitudes.
        phasecor_ftmag_abs = abs(phasecor_ftmag[pe,:])
        mag_thresh = .2*phasecor_ftmag_abs.flat[argmax(phasecor_ftmag_abs.flat)]
        mag_thresh = MAG_THRESH
        mag_msk = where(phasecor_ftmag_abs>mag_thresh,1.,0.)

        # Convert to 4 quadrant arctan and set cor to zero for low magnitude voxels.
        pos_msk = where(phs>0,1.,0.)
        msk1 = pos_msk*where(ft_blk.imag<0,pi,0.)   # Re > 0, Im < 0
        msk2 = (1-pos_msk)*where(ft_blk.imag<0,2.*pi,pi) # Re < 0, Im < 0
        phs = mag_msk*(phs + msk1 + msk2)
        phs_correction[pe,:] = phs[:].astype(Float32)
        if pep == 0:
            dork_data[slice,seg,0,:] = phs
        elif pe == 1:
            dork_data[slice,seg,1,:] = phs

    if options.debug:
        dump_image("phsdata_cor.4dfp.img",phasedat_mag,N_fe_true,N_pe,nslice,1,1,1,1,reorder_dump_slices,0)
        tmp_image = zeros((nslice*N_pe_true,N_fe_true)).astype(Complex32)
        for slice in range(nslice):
            # Correction for all echos.
            for seg in range(nseg):
                ii = line_index(slice,nslice,seg,nseg,N_pe/nseg,pulse_sequence,n_nav_echo)
                jj = slice*N_pe_true + seg*(N_pe_true/nseg)
                for pe in range(N_pe/nseg):
                    theta = -phs_correction[pe+ii,:]
                    cor = cos(theta) + 1.j*sin(theta)

                    # Do the phase correction.
                    tmp[:] = blk[pe+ii,:]
                    shift(tmp,0,N_fe_true/2)
                    echo = inverse_fft(tmp - bias[slice])
                    shift(echo,0,N_fe_true/2)

                    # Shift echo time by adding phase shift.
                    echo = echo*cor

                    shift(echo,0,N_fe_true/2)
                    tmp = (fft(echo)).astype(Complex32)
                    shift(tmp,0,N_fe_true/2)
                    if pe > 0:
                        tmp_image[pe-n_nav_echo+jj,:] = tmp
        dump_complex_image(tmp_image,"phasedata_cor_final",N_pe_true,N_fe_true,nseg,nslice,n_nav_echo,petab,0,reorder_dump_slices,pulse_sequence,0,1)
    phasedat_phs = phase(blk)

    return(phs_correction,phasecor_ftmag,dork_data)


#-------------------------------------------------------------------------------
def compute_menon_navigator_correction(blk,tmp,slice,seg,pe,ii,nseg,N_pe,N_fe_true,bias,phs_correction,dork_data):
#-------------------------------------------------------------------------------

#   The first line is a navigator echo, compute the difference  in 
#   phase (dphs) due to B0 inhomogeneities using navigator echo.
    tmp[:] = blk[ii,:]
    shift(tmp,0,N_fe_true/2)
    nav_echo = (inverse_fft(tmp - bias[slice])).astype(Complex32)
    shift(nav_echo,0,N_fe_true/2)
    nav_mag = abs(nav_echo)
    phs = phase(nav_echo)
#   Create mask for threshold of MAG_THRESH for magnitudes.
    dphs = zeros((2,N_fe_true),Float)
    mag_msk = where(nav_mag>MAG_THRESH,1,0)
    nav_phs = mag_msk*phs
    slice1 = ii/N_pe
    seg1 = ((ii - slice1*N_pe)*nseg)/N_pe
    dphs[0,:] = (dork_data[slice1,seg1,0,:] - nav_phs)
    msk1 = where(dphs[0,:]<-pi, 2.*pi,0)
    msk2 = where(dphs[0,:] > pi,-2.*pi,0)
    dphs[0,:] = dphs[0,:] + msk1 + msk2 

# Now do the same thing for the ky=0 line in k-space.
    tmp[:] = blk[ii+1,:] # Only true for two-shot sequence.
    shift(tmp,0,N_fe_true/2)
    echo = (inverse_fft(tmp - bias[slice])).astype(Complex32)
    shift(echo,0,N_fe_true/2)
    mag = abs(echo)
    phs = phase(echo)

#    mag_msk = where(mag>MAG_THRESH,1,0)
    phs = mag_msk*phs
    dphs[1,:] = (dork_data[slice1,seg1,0,:] - phs)
    msk1 = where(dphs[1,:]<-pi, 2.*pi,0)
    msk2 = where(dphs[1,:] > pi,-2.*pi,0)
    dphs[1,:] = dphs[1,:] + msk1 + msk2 

    return(dphs,nav_mag)


#-------------------------------------------------------------------------------------
def correct_phs_menon(blk,phs_correction,dphs,n_nav_echo,slice,N_pe,N_fe,pe,ii,tmp,bias,phasecor_total,time0,time1,te,nav_mag,phasecor_ftmag,options):
#-------------------------------------------------------------------------------------

#   Correct phase of data using the method Ravi Menon used in epi2fid.
#   The full_dork and partial_dork methods correct for respiratory artifacts.
#   See Pfeuffer et al., "Correction of physiologically induced global off-resonance effects in dynamic echo-planar and spiral functional imaging," MRM 47:344-353 (2003)

    if n_nav_echo > 0 and not options.ignore_nav:
        if options.navcor_type == NERD:
            nav_scl = (time0 + pe*time1)/time0
            nav_cor = nav_scl*dphs[0,:]
        elif options.navcor_type == PARTIAL_DORK:
            nav_scl = pe*time1/te
            nav_cor = nav_scl*dphs[0]
        elif options.navcor_type == FULL_DORK:
            timex = time1*float((N_pe - 1))/2.
            domega = (dphs[1,:] - dphs[0,:])/timex
            dphi = (te*dphs[0,:] - (te - timex)*dphs[1,:])/timex
            nav_cor = dphi + pe*time1*domega
        theta = -(phs_correction[pe+ii,:] - nav_cor[N_fe/4])
        msk1 = where(theta<0.,2.*pi,0)
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
        theta = -phs_correction[pe+ii,:]
        cor = cos(theta) + 1.j*sin(theta)
    phasecor_total[pe+ii,:] = theta.astype(Float32)

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
    return(tmp)

#****************
def polar(image):
#****************

# Compute phase from a complex number.

#    msk = where(equal(image.real,0.),1.,0.)
#    phs = ((1.-msk)*arctan(image.imag/(image.real+msk))).astype(Float)
#    pos_msk = where(phs>0,1,0)
#    msk1 = pos_msk*where(image.imag<0,pi,0)
#    msk2 = (1-pos_msk)*where(image.imag<0,2.*pi,pi)
#    msk  = where((msk1+msk2) == 0,1,0)
#    phs = (phs + msk1 + msk2).astype(Float)
#                dphi_phs[:] = arctan2(dphi.imag,dphi.real)

    return(arctan2(phs.imag,phs.real),abs(image))


#*******************
def fit_line(x,y,w):
#*******************

# w: weights defined for each x.


# Purpose: Fit straight line to data.

#   Use weighted least squares.
    N = len(x)

    wx = w*x
    sumw = sum(w)
    sumwx = sum(wx)
    sumwxsq = dot(wx,wx)
    sumwy0 = sum(w*y)
    b = 0.
    for iter in range(10):
#       Iteratively determine intercept.
        wy = w*(y - b)
        sumwy = sum(wy)
        sumwxy = dot(wx,wy)
        m = (N*sumwxy - sumwx*sumwy)/(N*sumwxsq - sumwx**2)
        b = (sumwy0 - m*sumwx)/sumw
    resid = y - m*x - b
#        print "%d m: %8.6f, b: %8.6f" % (i,slope,b)
#    intercept = (sumwy - slope*sumwx)/sum(w)
#    for i in range(N):
#        print "%d %6.3f %6.3f  %6.3f %6.3f %6.3f" % (i,x[i],y[i],w[i],slope,intercept)

    return((m,b))

