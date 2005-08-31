"""
Tool: recon-epi
Perform MRI image reconstruction.
Read data from fid, phase correct, take FFT, and write to disk.

svn-Id: $Id$
"""

# constant symbols representing values for some command-line flags
# These should move to a globally accessible spot (like the recon package),
# then be referred to instead of using the string values directly in
# client code.

FIDL_FORMAT = "fidl"
VOXBO_FORMAT = "voxbo"
SPM_FORMAT = "spm"
NONE = "none"
SINC = "sinc"
LINEAR = "linear"

# Types of navigator echo corrections:
NERD = "nerd"
NO_DORK = "no_dork"
PARTIAL_DORK = "partial_dork"
FULL_DORK = "full_dork"

#-------------------- parse command-line options -------------------------
from optparse import OptionParser
optparser = OptionParser("usage: %prog [options] fid_file procpar output_image")
optparser.add_option( "-n", "--nvol", action="store", dest="nvol_to_read", type="int",
  default=0, help="Number of volumes within run to reconstruct." )
optparser.add_option( "-s", "--frames-to-skip", action="store", dest="frames_to_skip", type="int",
  default=0, help="Number of frames to skip at beginning of run." )
optparser.add_option( "-k", "--save-kspace", action="store_true", dest="lcksp",
  help="Save k-space magnitude and phase." )
optparser.add_option( "-p", "--save-phase", action="store_true", dest="lcphs",
  help="Save phase as well as magnitude of reconstructed images." )
optparser.add_option( "-f", "--file-format", action="store", dest="file_format",
  type="choice", choices=(FIDL_FORMAT, VOXBO_FORMAT, SPM_FORMAT), default=FIDL_FORMAT,
  help="""{fidl | voxbo | spm}
  fidl: save floating point file with interfile and 4D analyze headers.
  spm: Save individual image for each frame in analyze format.
  voxbo: Save in tes format.""" )
optparser.add_option( "-N", "--nav-cor", action="store", dest="navcor_type",
  type="choice", choices=(NERD, NO_DORK, PARTIAL_DORK, FULL_DORK), default=NERD,
  help="""{no_dork | nerd | partial_dork | full_dork}
  nerd: The Near-DORK correction originally used in epi2fid and recon_epi.
     Valid only for the two-shot Ravi sequence.
  partial_dork: The partial DORK correction that does not need a navigator echo.
     Works for any sequence.
  full_dork: The full DORK correction that corrects for frequency and phase
     error. Only available for sequences with a navigator echo, such as the
     Ravi Menon sequence.""" )
optparser.add_option( "-v", "--save-first", action="store_true", dest="save_first",
  help="Save first frame in file named 'EPIs.cub'." )
optparser.add_option( "-g", "--ignore-nav-echo", action="store_true", dest="ignore_nav",
  help="Do not use navigator in phase correction." )
optparser.add_option( "-t", "--time-interp", action="store", dest="time_interp",
  type="choice", choices=(NONE, SINC, LINEAR), default=NONE,
  help="{none | sinc | linear} Interpolate to increase temporal resolution by a factor equal to the number of shots in the acquistions." )
optparser.add_option( "-w", "--fix-time-skew", action="store_true", dest="lc_fixskew",
  help="Correct for varying acquisition times of slices within a frame." )
optparser.add_option( "-z", "--zoom-by-two", action="store_true", dest="zoom",
  help="Spatially zoom images by a factor of two." )
optparser.add_option( "-c", "--skip-phase-corr", action="store_false", dest="phase_correct",
  default=True,
  help="Do not apply phase correction. (use when reconstructing fid files processed with epi2fid)" )
optparser.add_option( "-l", "--phase-last", action="store_true", dest="phase_last",
  help="Use phase information stored at the end of the data file." )
optparser.add_option( "-d", "--debug", action="store_true", dest="debug",
  help="Save images of correction arrays." )
optparser.add_option( "-o", "--phase-corr-total", action="store", dest="phase_cor_nav_file",
  help="Save phase correction nav to the specified file." )
optparser.add_option( "-i", "--linear-phase-corr", action="store_true", dest="linear_phase_corr",
  help="Use linear fit to compute Nyquist ghost correction." )
optparser.add_option( "-e", "--tr", action="store", dest="TR", type="float",
  help="Use the TR given here rather than the one in the procpar." )
optparser.add_option( "-x", "--starting-frame-number", action="store", dest="sfn",
  type="int", metavar="<starting frame number>", default=0,
  help="Specify starting frame number for analyze format output." )
optparser.add_option( "-m", "--field-map", action="store", dest="field_map_file",
  help="File containing asems field map computed with compute fmap.  This file is used for undistorting linear one-shot sequences." )
optparser.add_option( "-u", "--undistortion-matrix", action="store", dest="undistortion_file",
  help="File containing undistortion matrix computed by compute_fmap.  (used to undistort the reconstructed images)" )
optparser.add_option( "-b", "--flip-top-bottom", action="store_true", dest="flip_top_bottom",
  help="Flip image about the  horizontal axis." )
optparser.add_option( "-j", "--flip-left-right", action="store_true", dest="flip_left_right",
  help="Flip image about the vertical axis." )
optparser.add_option( "-q", "--flip-slices", action="store_true", dest="flip_slices",
  help="Reorders slices." )
optparser.add_option( "-y", "--output-data-type", action="store", dest="output_data_type",
  type="choice", choices=("mag", "complex"), default="mag",
  help="{mag | complex}  Specifies whether output images should contain only magnitude or both the real and imaginary components.  (only valid for analyze format)" )


def run():
    options, args = optparser.parse_args()
    if len(args) != 3: optparser.error( "Expecting 3 arguments" )
    fid_file, procpar_file, img_file = args

    # account for dependencies among option values
    if options.debug:
        options.save_phase_cor = True
    options.linear_phase_corr = options.field_map_file or options.undistortion_file or options.linear_phase_corr
    if options.linear_phase_corr:
        options.ignore_nav = 1
    if options.field_map_file and options.undistortion_file:
        optparser.error( 'Specify either "--fmap" or "--undistortion-matrix" but not both.' )

    #------------------ done parsing command-line options --------------------

    import sys
    import os
    from Numeric import Float, zeros, arange, Float32, Complex32, take, reshape, fromstring, Complex, argmax, multiply
    from  file_io import create_hdr,dump_image,read_file,write_ifh,write_analyze_header
    import FFT
    import struct
    import math
    import gc
    import math_bic
    import recon
    from recon.lib.epi import phase,detect_fid_format,Fermi_filter,line_index,get_procpar,reorder_slices,reorder_kspace,read_block,fix_time_skew,read_pe_table,identify_pulse_sequence,Recon_IO,compute_menon_phase_correction,compute_menon_navigator_correction,correct_phs_menon,compute_linear_phase_corr,correct_phase_errors_linear

    reorder_segments = 0 # constant:  not settable via command-line options
    reorder_dump_slices = 0 # constant:  not settable via command-line options

    # Include path in screen-dumps for troubleshooting purposes.
    print sys.argv[0]
    print os.getcwd()

    # Intialize output variables.
    io = Recon_IO(img_file,options)

    #-------------- parse procpar -------------------
    (params,pulse_sequence,N_fe,N_pe,N_fe_true,datasize,num_type,nslice,nvol,nvol_r,te,trise,gro,gmax,at,ydim,tdim,zdim,slice_pos,thk,gap,nv,orient,dwell_time) = get_procpar(procpar_file,options)

    # Get pulse sequence info.
    (nseg,petable,petab) = identify_pulse_sequence(params,pulse_sequence,options,tdim,nvol_r,nvol,N_pe)

    # Identify type of fid file.
    nvol_new,fid_type = detect_fid_format(fid_file,nvol,nseg,nslice,N_pe,N_fe/2,datasize,options.time_interp,options.phase_correct,pulse_sequence)

    if nvol_r > nvol_new:
        print "****** The fid file contains fewer frames than are specified by the procpar file. ******"
        print "Changing the number of frames from %d to %d" % (nvol,nvol_new)
        nvol = nvol_new
        nvol_r = nvol_new
        tdim = nvol_new
    vol_rng = range(nvol_r)

    if N_pe % 32:
        # Must be a navigator echo.
        n_nav_echo = 1
        print "Data acquired with navigator echos."
    else:
        if params.has_key('lcnav'):
            if params['lcnav'] == "1":
                n_nav_echo = 1
                N_pe = N_pe + nseg
            else:
                print "No navigator echos."
                n_nav_echo = 0
                options.ignore_nav = 1
                if options.navcor_type == NO_DORK or options.navcor_type == FULL_DORK: 
                    options.navcor_type = NO_DORK
                    print "*** Navigator echo options deleted. ***"
        else:
            print "No navigator echos."
            n_nav_echo = 0
            options.ignore_nav = 1

    xsize =  float(params['fov'])/(float(params['nv']) - nseg*n_nav_echo)
    ysize = xsize
    zsize = float(params['thk']) + float(params['gap'])

    # Parameters for compressed fid files. 
    SUB_HDR= 28
    LINE_LEN_DATA = datasize*N_fe
    LINE_LEN = LINE_LEN_DATA + SUB_HDR
    SLICE_LEN = N_fe*N_pe*datasize
    BLOCK_LEN = SLICE_LEN*nslice

    if nseg == 1 and options.time_interp != NONE:
        print " *** Data from single-shot sequences cannot be time-interpolated. ***"
        options.time_interp = NONE

    print "Phase encode table: ",petable

    options.TR = nseg*float(params['tr'])
    N_pe_true = N_pe - nseg*n_nav_echo
    xdim = N_pe_true

    #if len(params['petable']) > 0:
    petable_file = os.path.join( recon.petable_lib, petable )
    petab = read_pe_table( petable_file, N_pe )

    pe_per_seg = N_pe/nseg

    params['nseg'] = nseg
    date = os.popen("date").readlines()
    params['date'] = date[0]
    rev_string = "$Id$"
    rev_string = rev_string.split()
    params['recon_rev'] = rev_string[2] + " " + rev_string[3]

    if petable_file.find("alt") >= 0:
        time0 = te - 2.0*abs(gro)*trise/gmax - at
    else:
        time0 = te - (math.floor(nv/nseg)/2.0)*((2.0*abs(gro)*trise)/gmax + at)
    time1 = 2.0*abs(gro)*trise/gmax + at

    # Open files.
    f_fid = open(fid_file,"r")
    if options.phase_cor_nav_file:
        f_phscor_nav = open(phase_cor_nav_file,"w")

    # Get main header and decode it.
    MAIN_HDR= 32
    mhdr = struct.unpack('>llllllhhl',f_fid.read(MAIN_HDR))
    ebytes = mhdr[3]      # number of bytes per element

    print ""
    print "Pulse sequence: %s" % params['pulse_sequence']
    print "Number of frequency encodes: %d" % N_fe_true
    print "Number of phase encodes: %d" % N_pe
    print "Bytes per sample: %d" % ebytes
    print "Number of slices: %d" % nslice
    print "Number of frames: %d" % tdim
    print "Total number of volumes acquired (image & phase): %d" % nvol
    print "Number of segments: %d" % nseg
    print "Number of frames to skip: %d" % options.frames_to_skip
    print "Orientation: %s" % orient
    print "Pixel size: %7.2f" % xsize
    print "Slice thickness: %7.2f" % zsize
    if options.lc_fixskew:
        print "Correcting for skewed slice acquisition times."
    if options.time_interp == NONE:
        print "Full Fourier reconstruction."
    elif options.time_interp == SINC:
        print "Sinc-interpolating adjacent half-Fourier acquisitions to double temporal resolution."
    elif options.time_interp == LINEAR:
        print "Linearly interpolating adjacent half-Fourier acquisitions to double temporal resolution."
    if n_nav_echo > 0:
        print "Data acquired with navigator echo time of %f" % time0
        print "Data acquired with echo spacing of %f" % time1
    if(params.has_key('dwell')):
        print "Dwell time: %f" % dwell_time
    print ""

    # Read field map if specified.
    if options.field_map_file:
        fmap_data = read_file(options.field_map_file)
        shift = (xdim*dwell_time/(2.*math.pi))*fmap_data['image']
        pixel_pos = zeros((zdim,ydim,xdim)).astype(Float)
        for z in range(zdim):
            for y in range(ydim):
                pixel_pos[z,y,:] = arange(xdim).astype(Float)
        pixel_pos = pixel_pos + shift

    # Create Fermi filter kernel.
    Fermi = Fermi_filter(N_pe_true,N_fe_true,0.9)

    phs_vol = zeros((zdim,ydim,xdim)).astype(Float32)
    tmp     = zeros((N_fe_true)).astype(Complex32)
    blk_cor = zeros((tdim,zdim*N_pe_true,N_fe_true),Complex)

    time_rev = N_fe_true - 1 - arange(N_fe_true)
    frame = options.sfn - 1
    if not options.phase_correct:
        print "\nleading data."
    else:
        print "\nPhase-correcting the data"
    for vol in vol_rng:
        frame = frame + 1
        if frame == 1 and options.frames_to_skip > 0:
            # Skip data.
            if fid_type == 'compressed':
                # Skip phase data and skip blocks.
                pos = options.frames_to_skip*(SUB_HDR + BLOCK_LEN)
            elif fid_type == 'uncompressed':
                pos = options.frames_to_skip*nslice*(SUB_HDR + SLICE_LEN)
            elif fid_type == 'epi2fid':
                pos = options.frames_to_skip*nslice*(N_pe_true)*(SUB_HDR + LINE_LEN_DATA)
            elif fid_type == 'asems_ncsnn':
                pos = 0
            else:
                print "Unsupported pulse sequence."
                sys.exit(1)
            f_fid.seek(pos,1)
        sys.stdout.write(".")
        sys.stdout.flush()

        # Get block header and unpack it.
    #    scale   = shdr[0]       # scaling factor
    #    status  = shdr[1]       # status of data in block
    #    index   = shdr[2]       # block index
    #    spare3  = shdr[3]       # reserved for future use
    #    ctcount = shdr[4]       # completed transients in fids
    #    lpval   = shdr[5]       # left phase in phasefile
    #    rpval   = shdr[6]       # right phase in phasefile
    #    lvl     = shdr[7]       # level drift correction
    #    tlt     = shdr[8]       # tilt drift correction

        # Read the next block.
        bias = zeros(nslice).astype(Complex32)
        (blk,time_reverse) = read_block(f_fid,fid_type,MAIN_HDR,SUB_HDR,SLICE_LEN,LINE_LEN,LINE_LEN_DATA,N_pe,n_nav_echo,N_fe_true,nseg,nslice,nvol,N_pe_true,pulse_sequence,bias,num_type,vol)

        if time_reverse:
            # Time-reverse the data.
            for pe in range(N_pe*nslice):
                if(pe % 2):
                    blk[pe,:] = take(blk[pe,:],time_rev)
        fname1 = "blk_mag_%d.4dfp.img" % (vol)
        dump_image(fname1,abs(reshape(blk,(nslice*N_pe,N_fe_true))),N_fe_true,N_pe,nslice,1,1,1,1,0,0)

        if vol== 0 and options.phase_correct:  
            # Compute correction for Nyquist ghosts.
            # First and/or last block contains phase correction data.  Process it.
            if options.phase_last:
                # Read phase data from last block.
                cur_fileptr = f_fid.tell()
                f_fid.seek(0,2)
                end_fileptr = f_fid.tell()
                lastblk = end_fileptr - (BLOCK_LEN+SUB_HDR)
                f_fid.seek(lastblk,0) # Move to last block. f.seek(*,2) does not work.
                shdr = struct.unpack('>hhhhlffff',f_fid.read(SUB_HDR))
                bias[:] = complex(shdr[7],shdr[8]).astype(Complex32)
                blk = fromstring(f_fid.read(BLOCK_LEN),num_type).byteswapped().astype(Float32).tostring()
                blk = fromstring(blk,Complex32)
                blk = reshape(blk,(N_pe*zdim,xdim))
                f_fid.seek(cur_fileptr,0) # Return to original block.
                for pe in range(N_pe*nslice):
                    if(pe % 2):
                        blk[pe,:] = take(blk[pe,:],time_rev)
            if options.linear_phase_corr:
    #           Compute linear phase correction.
                offset_corr,theta_corr,dork_data = compute_linear_phase_corr(blk,bias,nslice,nseg,N_pe/nseg,N_fe_true,n_nav_echo,options)
            else:
    #           Compute point-by-point phase correction copied from epi2fid.c
                phs_correction,phasecor_ftmag,dork_data = compute_menon_phase_correction(blk,nslice,nseg,N_pe,N_fe_true,bias,nvol_r,options,n_nav_echo,petab,reorder_segments,reorder_dump_slices,pulse_sequence)

        elif options.phase_correct: 
            # This block contains image data. First, calculate the phase correction including
            # the navigator echo. 
    #        if options.debug:
    #            dump_image("phsdata_img_mag.4dfp.img",blk,N_fe_true,N_pe,nslice,1,1,1,1,reorder_dump_slices,0)
            dphs = zeros(N_pe).astype(Float32)
            if options.linear_phase_corr:
                blk_cor[vol-1,:,:] = correct_phase_errors_linear(blk,bias,offset_corr,theta_corr,nslice,nseg,N_pe,N_fe_true,n_nav_echo,dork_data,te,time1,options)
            else:
                for slice in range(nslice):
                    # correction for all echos.
                    for seg in range(nseg):
                        ii = line_index(slice,nslice,seg,nseg,pe_per_seg,pulse_sequence,n_nav_echo)
                        jj = line_index(slice,nslice,seg,nseg,N_pe_true/nseg,pulse_sequence,n_nav_echo)
                        for pe in range(N_pe/nseg):
                            if pe == 0 and n_nav_echo > 0 and not options.linear_phase_corr and not options.ignore_nav:
                                # The first line is a navigator echo, compute the difference  in 
                                # phase (dphs) due to B0 inhomogeneities using navigator echo.
                                dphs,nav_mag = compute_menon_navigator_correction(blk,tmp,slice,seg,pe,ii,nseg,N_pe,N_fe_true,bias,phs_correction,dork_data)
                            else:
    #                           Correct the phase.
                                if pe == 0:
                                    nav_mag = 0
                                if not options.linear_phase_corr: # and pe > n_nav_echo-1:
    #                                blk_cor[vol-1,pe-n_nav_echo+jj,:] = blk_cor[vol-1,pe+ii,:]
    #                            else:
    #                                print slice,seg,pe,pe-n_nav_echo+jj,pe+ii,shape(blk_cor) 
                                    blk_cor[vol-1,pe-n_nav_echo+jj,:] = correct_phs_menon(blk,phs_correction,dphs,n_nav_echo,slice,N_pe,N_fe,pe,ii,tmp,bias,time0,time1,te,nav_mag,phasecor_ftmag,options)

            if options.debug:
                dump_image("data_uncor.4dfp.img",abs(reshape(blk,(nslice*N_pe,N_fe_true))),N_fe_true,N_pe,nslice,1,1,1,1,0,0)
                phs_test = zeros((N_pe,N_fe_true),Complex)
                f_phsdata = open("data_cor.4dfp.img","w")
                for slice in range(nslice):
                    for seg in range(nseg):
                        ii = line_index(slice,nslice,seg,nseg,pe_per_seg,pulse_sequence,n_nav_echo)
                        for pe in range(N_pe/nseg - n_nav_echo):
                            jj = pe + (pe_per_seg-n_nav_echo)*(slice + seg*nslice)
                            phs_test[pe+seg*pe_per_seg,:] = blk_cor[vol-1,jj,:]
                    x= (abs(phs_test)).astype(Float32).byteswapped().tostring()
                    f_phsdata.write(x)
                f_phsdata.close()
                write_ifh("data_cor.4dfp.ifh",N_fe_true,N_pe,zdim,tdim,xsize,ysize,zsize)
                hdr = create_hdr(N_fe_true,N_pe,zdim,1,xsize,ysize,zsize,1.,0,0,0,'Float',32,1.,'analyze',"data_cor.4dfp.img",1)
                write_analyze_header("data_cor.4dfp.hdr",hdr)
        else:
    #       Phase correction was skipped.
            print "Skipping phase correction."
            blk = reshape(blk,(zdim,N_pe,N_fe_true))
            for slice in range(nslice):
                blk[slice,:,:] = (blk[slice,:,:] - bias[slice]).astype(Complex32)
            blk = reshape(blk,(zdim*N_pe,N_fe_true))
            if pulse_sequence == 'asems':
                blk_cor[vol,:,:] = blk[:,:]
            elif fid_type == "epi2fid" or pulse_sequence == 'epidw' or pulse_sequence == 'epidw_se':
                blk_cor = reshape(blk_cor,(tdim,nseg,zdim,N_pe_true/nseg,N_fe_true))
                blk = reshape(blk,(nseg,zdim,N_pe/nseg,N_fe_true))
                for seg in range(nseg):
                    for slice in range(nslice):
                        blk_cor[vol-1,seg,slice,:,:] = blk[seg,slice,n_nav_echo:,:]
            else:
                print "This combination of options and pulse sequence is not supported."
                sys.exit(1)
    print "\n"
    #del blk
    gc.enable()
    gc.collect()
    blk_cor = reshape(blk_cor,(tdim,nslice*nseg,N_pe_true/nseg,N_fe_true))

    if options.time_interp == LINEAR:
        print "Interpolating frames"
        blk_cor = reshape(blk_cor,(tdim,nseg,zdim*N_pe_true/nseg*N_fe_true))
        tdim_new = 2*tdim
        blk_interp = zeros((tdim_new,nseg,zdim*N_pe_true/nseg*N_fe_true)).astype(Complex32)
        for seg in range(nseg):
            blk_interp[0,0,:] = blk_cor[0,0,:]
            blk_interp[0,1,:] = blk_cor[0,1,:]
            blk_interp[1,0,:] = (.5*(blk_cor[0,0,:] + blk_cor[1,0,:])).astype(Complex32)
            blk_interp[1,1,:] = blk_cor[1,1,:]
            for t in range(1,tdim-1):
                sys.stdout.write(".")
                sys.stdout.flush()
                blk_interp[2*t,0,:] = blk_cor[t,0,:]
                blk_interp[2*t,1,:] = (.5*(blk_cor[t,1,:] + blk_cor[t-1,1,:])).astype(Complex32)
                blk_interp[2*t+1,0,:] = (.5*(blk_cor[t,0,:] + blk_cor[t+1,0,:])).astype(Complex32)
                blk_interp[2*t+1,1,:] = blk_cor[t,1,:]
            blk_interp[2*tdim-2,0,:] = (.5*(blk_cor[tdim-2,0,:] + blk_cor[tdim-1,0,:])).astype(Complex32)
            blk_interp[2*tdim-2,1,:] = blk_cor[tdim-2,1,:]
            blk_interp[2*tdim-1,0,:] = blk_cor[tdim-1,0,:]
            blk_interp[2*tdim-1,1,:] = blk_cor[tdim-1,1,:]
        tdim = tdim_new
        options.TR = options.TR/2.
        print "\n"
        del blk_cor
    elif options.time_interp == NONE:
        blk_interp = blk_cor
        blk_interp = reshape(blk_interp,(tdim,zdim,N_fe_true*N_pe_true))
        tdim_new = tdim
    else:
        print "*** Invalid interpolation mode in recon_epi ***"
        sys.exit(1)
    gc.collect()

    print "Taking FFTs."
    blk_interp = reshape(blk_interp,(tdim,zdim*N_pe_true,N_fe_true))
    slice_order = zeros(nslice)
    if options.zoom:
        xdim_out = xdim*2
        ydim_out = ydim*2
    else:
        xdim_out = xdim
        ydim_out = ydim
    ksp_image = zeros((zdim,N_pe_true,N_fe_true)).astype(Complex)
    checkerboard_mask = zeros((N_pe_true,N_fe_true)).astype(Float)
    line = zeros(N_fe_true)
    for x in range(N_fe_true):
        if x % 2:
            line[x] = 1
        else:
            line[x] = -1
    for y in range(N_pe_true):
        if y % 2:
            checkerboard_mask[y,:] = line
        else:
            checkerboard_mask[y,:] = -line
    checkerboard_mask_1 = -checkerboard_mask[:,:]
    img_vol = zeros((tdim,zdim,ydim_out,xdim_out)).astype(Complex32)
    if options.lcksp:
        ksp_vol = zeros((tdim,zdim,N_pe_true,N_fe_true)).astype(Complex32)
    else:
        ksp_vol = 0
    ksp = zeros((ydim_out,xdim_out)).astype(Complex32)
    for vol in range(tdim):
        sys.stdout.write(".")
        sys.stdout.flush()

        # Reorder the data according to the phase encode table read from "petable".
        reorder_kspace(pulse_sequence,blk_interp,Fermi,ksp_image,N_fe_true,N_pe_true,nseg,nslice,xdim,ydim,zdim,tdim,vol,petab,n_nav_echo,fid_type,pe_per_seg)

        for slice in range(nslice):
            # Take Fourier transform
            if options.zoom:
                ksp[:ydim,:xdim] = ksp_image[slice,:,:]
                image = FFT.inverse_fft2d(ksp)
            else:
                image = FFT.inverse_fft2d(ksp_image[slice,:,:])
            image.real = image.real*checkerboard_mask
            image.imag = image.imag*checkerboard_mask_1

            # Correct for distortion.
            if options.field_map_file:
                image.real = abs(image)
                image.imag = 0.
                image = math_bic.resample_phase_axis(abs(image),pixel_pos[z,:,:]).astype(Float32)

    #       Reorder k-space, flip and reorder slices.
            reorder_slices(image,ksp_vol,ksp_image,slice_order,options,xdim_out,ydim_out,zdim,N_pe,nslice,slice,vol,pulse_sequence,img_vol,fid_type)

        if options.lcksp:
            io.write_ksp_frame(ksp_vol,options,vol,N_fe_true,N_pe_true,nslice,xsize,ysize,zsize)
    print "\n"
    del blk_interp
    gc.collect()

    if  options.lcphs:
        phs_vol = phase(img_vol) - math.pi
    if options.output_data_type == 'mag':
        img_vol = abs(img_vol)
        
    blk_interp = 1
    gc.collect()  # Clean up memory before allocating more.

    # Scale output image.
    if options.output_data_type == 'mag':
        scl = 16383./img_vol.flat[argmax(img_vol.flat)]
        img_vol = multiply(scl,img_vol)

    if options.lc_fixskew:
        # Now correct for slice-timing skew.
        fix_time_skew(img_vol,xdim_out,ydim_out,zdim,tdim,xdim_out,ydim_out,nslice,nseg,slice_order,options,pulse_sequence)

    print "Saving to disk."
    for vol in range(tdim):
        sys.stdout.write(".")
        sys.stdout.flush()
        io.write_image_frame(img_vol,phs_vol,options,vol,xdim_out,ydim_out,nslice,xsize,ysize,zsize)
    print "\n"
                
    io.write_image_final(options,xdim_out,ydim_out,nslice,tdim,xsize,ysize,zsize,options.TR,N_fe_true,N_pe,img_vol,params)
    f_fid.close()

    sys.exit(0)
