#!/usr/bic/python/bin/python
#!/usr/bin/python

import sys
import string
import os
from LinearAlgebra import *
from Numeric import *
import struct
import MLab
import file_io
import idl
import FFT


#*********************
def invert_xform(xfm):
#*********************

# Invert affine 4x4 transformation matrix.

    r = xfm[0:3,0:3]
    rm1 = inverse(r)
    dx = xfm[:3,3]
    dxm1 = -matrixmultiply(rm1,dx)
    inv = zeros((4,4)).astype(Float)
    inv[:3,:3] = rm1
    inv[:3,3] = dxm1
    inv[3,3] = 1.

    return inv


#*************************************************
def clip_sinuses(input_file,output_file,skip,lcv):
#*************************************************

# Usage: clip_sinuses input_image output_image
# Extract brain and then sets image to zero for inferior non-brain regions.

    dot = string.find(input_file,".img")
    if(dot < 0):
        dot = string.find(input_file,".hdr")
    if dot < 0:
        stem = input_file
    else:
        stem = input_file[0:dot]
    
#     Extract brains.
    brain_file = stem + "_brain.img"
    cmd = "bet %s %s" % (input_file,brain_file)
    file_io.exec_cmd(cmd,skip,lcv)
    
    brain_data = file_io.read_file(brain_file)
    brain_hdr = brain_data['header']
    brain_image = brain_data['image']

    data = file_io.read_file(input_file)
    hdr = data['header']
    image = data['image']

    brain_data = file_io.read_file(brain_file)
    brain_hdr = brain_data['header']
    brain_image = brain_data['image']

    xdim = brain_hdr['xdim']
    ydim = brain_hdr['ydim']
    zdim = brain_hdr['zdim']

    brain_image = reshape(brain_image,(zdim,ydim,xdim))
    image = reshape(image,(zdim,ydim,xdim))

    profile = sum(brain_image,2)
    profile_mask = where(greater(profile,0.),1.,0.)

    ramp = ydim -1 - arange(ydim)
    min = zeros(zdim)
    for slice in range(zdim):
        min[slice] = argmax(ramp*profile_mask[slice,:])
    minmin = min[argmin(min)]

    ydim = ydim - minmin
    img = zeros((zdim,ydim,xdim)).astype(Float32)
    img[:,:,:] = brain_image[:,minmin:,:]

    hdr['ydim'] = ydim
    file_io.write_analyze(output_file,hdr,img)

    return ydim



#************************************************************
def create_transform(dx,dy,dz,roll,pitch,yaw,xscl,yscl,zscl):
#************************************************************

# Create transformation matrix.

    D2R = math.pi/180.
    A = identity(3).astype(Float)
    B = identity(3).astype(Float)

    B[1,1] =  cos(pitch*D2R)
    B[1,2] =  sin(pitch*D2R)
    B[2,1] = -sin(pitch*D2R)
    B[2,2] =  cos(pitch*D2R)
    AX = matrixmultiply(A,B)
    
    B = identity(3).astype(Float)
    B[0,0] =  cos(roll*D2R)
    B[0,2] =  sin(roll*D2R)
    B[2,0] = -sin(roll*D2R)
    B[2,2] =  cos(roll*D2R)
    A = matrixmultiply(AX,B)

    B = identity(3).astype(Float)
    B[0,0] =  cos(yaw*D2R)
    B[0,1] =  sin(yaw*D2R)
    B[1,0] = -sin(yaw*D2R)
    B[1,1] =  cos(yaw*D2R)
    AX = matrixmultiply(A,B)

    B = identity(3).astype(Float)
    B[0,0] = xscl
    B[1,1] = yscl
    B[2,2] = zscl
    A = transpose(matrixmultiply(AX,B))

    offset = zeros(3).astype(Float)
    offset[0] = dx
    offset[1] = dy
    offset[2] = dz
    off1 = matrixmultiply(A,offset)

    xfm = identity(4).astype(Float)
    xfm[:3,:3] = A
    xfm[:3,3] = off1[:]

    return xfm
    
#********************************************
def resample_phase_axis(input_image,pixel_pos):
#********************************************

# Purpose: Resample along phase encode axis of epi images. It is assumed that the phase encode axis is the last (fastest varying) axis in the input image.

# Inputs: input_image: Epi image to be resampled.
#         pixel_pos: Image of resampled pixel positions.

    shp = input_image.shape
    ndim = len(shp)
    xdim = shp[1]
    if ndim == 2:
        ydim = shp[0]
        output_image = zeros((ydim,xdim)).astype(input_image.typecode())
    elif ndim == 1:
        ydim = 1
        output_image = zeros((xdim)).astype(input_image.typecode())
    else:
        print 'Resample phase axis can only handle 1D or 2D input arrays.'
        sys.exit(1)

    delta = zeros((xdim)).astype(Float)
    for y in range(ydim):
        if ndim == 1:
            vals = input_image[:]
            x = pixel_pos[:]
        elif ndim == 2:
            vals = input_image[y,:]
            x = pixel_pos[y,:]
        ix = clip(floor(x).astype(Int),0,xdim-2)
        delta = x - ix
        if ndim == 1:
            output_image[:] = (1.-delta)*take(vals,ix) + delta*take(vals,ix+1)
        elif ndim == 2:
            output_image[y,:] = (1.-delta)*take(vals,ix) + delta*take(vals,ix+1)
        x1 = take(vals,ix)
        x2 = take(vals,ix+1)
#        if y == 27 and z==0:
#            for i in range(xdim):
#                print "%d x: %7.3f, ix: %d, delta: %5.3f, epi: %8.0f, out: %8.0f, x1: %8.0f, x2: %8.0f" % (i,x[i],ix[i],delta[i],abs(input_image[y,i]),abs(output_image[y,i]),abs(x1[i]),abs(x2[i]))

    return output_image

#*******************************************
def inverse_fft3d(image,verbose,save_kspace):
#*******************************************

# Invert affine 4x4 transformation matrix.

    shp = shape(image)
    xdim = shp[2]
    ydim = shp[1]
    zdim = shp[0]

    kimg = zeros((zdim,ydim,xdim)).astype(Complex)

#   First transform along the axial dimension.
    tmp = zeros((zdim)).astype(Complex)
    for y in range(ydim):
        if verbose:
            sys.stdout.write(".")
            sys.stdout.flush()
        for x in range(xdim):
            tmp[:] = image[:,y,x]
            ktmp = FFT.inverse_fft(tmp)
            kimg[:,y,x] = ktmp[:]

    if save_kspace:
###        file_io.dump_image("kspace_mag.4dfp.img",abs(kimg),xdim,ydim,zdim,1,1,1,1,0,0)
        img_file = "kspace_mag.img"
        hdr = file_io.create_hdr(xdim,ydim,zdim,1,1.,1.,1.,1.,0,0,0,'Short',16,1.,'analyze',img_file,0)
        file_io.write_analyze(img_file,hdr,abs(kimg))

#   Now do an in-plane 2D fft.
    tmp = zeros((ydim,xdim)).astype(Complex)
    for z in range(zdim):
        if verbose:
            sys.stdout.write(".")
            sys.stdout.flush()
        tmp[:,:] = kimg[z,:,:]
        image = FFT.inverse_fft2d(tmp)
        kimg[z,:,:] = image
    sys.stdout.write("\n")


    return kimg

#**************************
def median_filter(image,N):
#**************************

# Filter an image with a median filter of order NxN where N is odd.

    if (N % 2) == 0:
        print "Order of median filter must be odd."
        return(1)
    median_pt = int((N*N)/2)
    ctr = int(N/2)
    shp = image.shape
    ndim = len(shp)
    xdim = shp[2]
    ydim = shp[1]
    zdim = shp[0]
    idx = 0
    if ndim == 4:
        tdim =  shp[idx]
        idx = idx + 1
    else:
        tdim = 1
    if ndim >= 3:
        zdim = shp[idx]
        idx = idx + 1
    else:
        zdim = 1
    if ndim >= 2:
        ydim = shp[idx]
        xdim = shp[idx+1]
    if ndim < 2:
        print "Image dimension must be at least 2 in module <median_filter>"
        sys.exit(1)

    image = reshape(image,(tdim*zdim,ydim,xdim))

    img = zeros((tdim*zdim,ydim,xdim)).astype(image.typecode())
    subm = zeros((N,N)).astype(image.typecode())
    for tz in range(tdim*zdim):
        img[tz,0,:] = 0.
        img[tz,-1:,:] = 0.
        for y in range(ydim-N):
            img[tz,y,0] = 0.
            img[tz,y,xdim-1] = 0.
            for x in range(xdim-N):
                subm[:,:] = image[tz,y+1:y+N+1,x+1:x+N+1]
                s = sort(subm.flat)
                img[tz,y+ctr+1,x+ctr+1] = s[median_pt]
    reshape(img,(tdim,zdim,ydim,xdim))
        


    return(img)
