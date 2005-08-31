#!/raid0/despo/jjo/python/bin/python

# Program: fid_to_flat
# Purpose: Sort data in fid, take FFT, and write to disk.

import sys
import string
import os
import file_io
from Numeric import *
import FFT
import idl
import MLab

MAIN_HDR= 24
SUB_HDR= 56
MAIN_HDR= 24
SUB_HDR= 56

MAIN_HDR= 32
SUB_HDR= 28

if(len(sys.argv) < 3):
    print "Usage: fid_to_flat fid_file procpar output_image  [-nvol_to_read nvol_to_read -mag_file mag_file -phase_file phase_file]"
    sys.exit(1)

fid_file = sys.argv[1]
procpar_file = sys.argv[2]
img_file = sys.argv[3]

lcmag = 0
lcphs = 0
iarg = 0
nvol_to_read = 1
for arg in sys.argv:
    if arg == "-mag_file":
        lcmag = 1
        mag_file = sys.argv[iarg+1]
    elif arg == "-phase_file":
        lcphs = 1
        phs_file = sys.argv[iarg+1]
    elif arg == "-nvol_to_read":
        nvol_to_read = string.atoi(sys.argv[iarg+1])
    iarg = iarg + 1

params = file_io.parse_procpar(procpar_file,0)
print params

N_fe = string.atoi(params['np'])
N_pe = string.atoi(params['nv'])
if(params['dp'] == '"y"'):
    datasize = 4
    num_type = Int32
else:
    datasize = 2
    num_type = Int16
nslice = string.atoi(params['nslice'])
if(params.has_key('gain')):
    nvol = string.atoi(params['nvol']) - 1
else:
    nvol = 1
if(nvol_to_read > 0):
    nvol_r = nvol_to_read
else:
    nvol_r = nvol

print ""
print "Number of frequency encodes: %d" % (N_fe)
print "Number of phase encodes: %d" % (N_pe)
print "Bytes per sample: %d" % (datasize)
print "Number of slices: %d" % (nslice)
print "Number of volumes: %d" % (nvol)
print ""
if params['pulse_sequence'] == "epi":
    N_pe = N_pe - 2
xdim = N_fe/2
ydim = N_pe
zdim = nslice
tdim = nvol_r
xsize =  string.atof(params['fov'])/(string.atof(params['nv'])-2)
ysize = xsize
zsize = string.atof(params['thk']) + string.atof(params['gap'])

LINE_LEN_DATA = datasize*N_fe
LINE_LEN = LINE_LEN_DATA + SUB_HDR
SLICE_LEN = nslice*LINE_LEN
VOL_LEN = nvol*SLICE_LEN
f_fid = open(fid_file)
f_img = open(img_file,"w")
if(lcmag):
    f_mag = open(mag_file,"w")
if(lcphs):
    f_phase = open(phs_file,"w")
for vol in range(nvol_r):
    print "Processing volume #%d" % (vol)
    img_vol = zeros((zdim,ydim,xdim)).astype(Float32)
    mag_vol = zeros((zdim,ydim,xdim)).astype(Float32)
    phs_vol = zeros((zdim,ydim,xdim)).astype(Float32)
    for slice in range(nslice):
        sys.stdout.write(".")
        sys.stdout.flush()
        pe = -1
        position = (vol*nslice + slice)*LINE_LEN + MAIN_HDR + SUB_HDR
        f_fid.seek(position,0)
        x = fromstring(f_fid.read(LINE_LEN_DATA),num_type).byteswapped().astype(Float32).tostring()
        ksp_image = fromstring(x,Complex32)
        for pe in range(N_pe-1):
            position = ((pe+1)*nvol*nslice + vol*nslice + slice)*LINE_LEN + MAIN_HDR + SUB_HDR
            f_fid.seek(position,0)
            x = fromstring(f_fid.read(LINE_LEN_DATA),num_type).byteswapped().astype(Float32).tostring()
            x = fromstring(x,Complex32)
            ksp_image = concatenate((ksp_image,x),1)
        ksp_image = reshape(ksp_image,(xdim,ydim))
###        ksp_image[:,0:7] = 0.
        image = FFT.inverse_fft2d(ksp_image)
        image = abs(image).astype(Float32)
        idl.shift(image,0,ydim/2)
        idl.shift(image,1,xdim/2)
        image = transpose(image)
        image = MLab.flipud(transpose(image))

        if(lcphs):
            mask = where(equal(ksp_image.real,0.),1.,0.)
            ksp_phase = (1. - mask[:,:])*(ksp_image.imag/(ksp_image.real + mask[:,:]))
            ksp_phase = arctan(ksp_phase)
            ksp_phase = ksp_phase.astype(Float32)
        if(lcmag):
            ksp_image = abs(ksp_image)
###        ksp_phase = MLab.flipud(ksp_phase.astype(Float32))
###        ksp_image = MLab.flipud(abs(ksp_image))

        img_vol[slice,:,:] = MLab.fliplr(transpose(image[:,:]))
#        if(slice < nslice/2):
#            img_vol[2*slice,:,:] = image[:,:]
#            if(lcmag):
#                mag_vol[2*slice,:,:] = ksp_image[:,:]
#            if(lcphs):
#                phs_vol[2*slice,:,:] = ksp_phase[:,:]
#        else:
#            img_vol[2*(slice-nslice/2)+1,:,:] = image[:,:]
#            if(lcmag):
#                mag_vol[2*(slice-nslice/2)+1,:,:] = ksp_image[:,:]
#            if(lcphs):
#                phs_vol[2*(slice-nslice/2)+1,:,:] = ksp_phase[:,:]
            

#   Write magnitude and phase of this slice to disk.
#    if(lcmag):
#        f_mag.write(mag_vol.byteswapped().tostring())
#    if(lcphs):
#        f_phase.write(phs_vol.byteswapped().tostring())
    f_img.write(img_vol.byteswapped().tostring())

f_fid.close()
f_img.close()
if(lcmag):
    f_mag.close()
if(lcphs):
    f_phase.close()

dot = string.find(img_file,".")
if(dot >= 0):
    ifh_file = img_file[0:dot] + ".4dfp.ifh"
else:
    ifh_file = img_file + ".4dfp.ifh"
file_io.write_ifh(ifh_file,xdim,ydim,zdim,tdim,xsize,ysize,zsize)

print "Images written to %s" % (img_file)
if(lcmag):
    dot = string.find(mag_file,".")
    if(dot >= 0):
        ifh_file = mag_file[0:dot] + ".4dfp.ifh"
    else:
        ifh_file = mag_file + ".4dfp.ifh"
    file_io.write_ifh(ifh_file,xdim,ydim,zdim,tdim,xsize,ysize,zsize)
    print "Magnitude data written to %s" % (mag_file)

if(lcphs):
    dot = string.find(phs_file,".")
    if(dot >= 0):
        ifh_file = phs_file[0:dot] + ".4dfp.ifh"
    else:
        ifh_file = phs_file + ".4dfp.ifh"
    file_io.write_ifh(ifh_file,xdim,ydim,zdim,tdim,xsize,ysize,zsize)
    print "Phase data written to %s" % (phs_file)
