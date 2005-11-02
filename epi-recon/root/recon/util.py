import sys
import string 
import os
from Numeric import *
import file_io
import struct
from Numeric import empty
from FFT import inverse_fft
from pylab import pi, mlab, fft, fliplr, zeros, fromstring
from recon import FIDL_FORMAT, VOXBO_FORMAT, SPM_FORMAT, MAGNITUDE_TYPE, COMPLEX_TYPE
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

#-----------------------------------------------------------------------------
def save_image_data(options, data):
    """
    This function saves the image data to disk in a file specified on the command
    line. The file name is stored in the options dictionary using the img_file key.
    The output_data_type key in the options dictionary determines whether the data 
    is saved to disk as complex or magnitude data. By default the image data is 
    saved as magnitude data.
    """
    data_matrix = data.data_matrix
    VolumeViewer(
      abs(data_matrix),
      ("Time Point", "Slice", "Row", "Column"))
    n_pe = data.n_pe
    n_pe_true = data.n_pe_true
    n_fe = data.n_fe
    n_fe_true = data.n_fe_true
    nslice =  data.nslice
    pulse_sequence = data.pulse_sequence
    xsize = data.xsize 
    ysize = data.ysize
    zsize = data.zsize

    # Setup output file names.

    # where's the dot
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

    # what's the stem
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
    vol_rng = range(frame_start, data.nvol)
    for vol in vol_rng:
        if options.save_first and vol == 0:  #!!!! DO WE NEED THIS SECTION !!!!!
            img = zeros((nslice,n_fe_true,n_pe_true)).astype(Float32)
            for slice in range(nslice):
                if flip_left_right:
                    img[slice,:,:] = fliplr(abs(data_matrix[vol,slice,:,:])).astype(Float32)
                else:
                    img[slice,:,:] = abs(data_matrix[vol,slice,:,:]).astype(Float32)
            file_io.write_cub(img,"EPIs.cub",n_fe_true,n_pe_true,nslice,ysize,xsize,zsize,0,0,0,"s",data)
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
def save_ksp_data(options, data):   # !!! FINISH AND TEST !!!
    """
    This function saves the k-space data to disk in a file specified on the
    command line with an added suffix of "_ksp". The k-space data is saved
    with the slices in the acquisition order rather than the spatial order.
    """
    complex_data = data.data_matrix
    n_pe = data.n_pe
    n_pe_true = data.n_pe_true
    n_fe = data.n_fe
    n_fe_true = data.n_fe_true
    nslice =  data.nslice
    pulse_sequence = data.pulse_sequence
    xsize = data.xsize 
    ysize = data.xsize
    zsize = data.zsize

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
    vol_rng = range(data.nvol)
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
