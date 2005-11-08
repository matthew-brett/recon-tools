import sys
import string
import os
from Numeric import *
import struct
from MLab import fliplr, flipud


#***********************
def file_type(filename):
#***********************

# Determine type of image file.


    dot = string.find(filename,".img")
    if(dot > 0):
        hdrname = filename[0:dot]+".hdr"
        ifhname = filename[0:dot]+".ifh"
    else:
        dot = string.find(filename,".hdr")
        if(dot > 0):
            hdrname = filename
            ifhname = filename
        else:
            dot = string.find(filename,".ifh")
            if(dot > 0):
                ifhname = filename
            else:
                hdrname = filename + '.hdr'
                ifhname = filename + '.ifh'
    if os.access(ifhname,0):
        testname = ifhname
    else:
        testname = filename
    if not os.access(testname,0):
        return -1
    infile = open(testname,'r')
    line1 = infile.readline()
    line2 = infile.readline()
    infile.seek(38940)
    id_buffer = infile.read(100)
    filetype = -1
    words1 = string.split(string.lstrip(line1))
    if len(words1) > 0: 
        interfile = words1[0]
    else:
        interfile = ""

    if(line1 == "VB98\n" and line2 == "CUB1\n"):
        filetype = "cub"
    elif(line1 == "VB98\n" and line2 == "TES1\n"):
        filetype = "tes"
    elif interfile == "INTERFILE":
        filetype = "4dfp"
    elif id_buffer[0:4] == "GEMS":
        filetype = "ge_data"
    elif os.access(hdrname,0):
        filetype = "analyze"
    else:
        print "Invalid file type"
        return(-1)
    infile.close()

    return filetype


#***********************
def read_file(filename):
#***********************

# Read image file.

    hdr = read_header(filename)
    if hdr == -1:
        return(-1)
    datatype = hdr['datatype']
    xdim = hdr['xdim']
    ydim = hdr['ydim']
    zdim = hdr['zdim']
    tdim = hdr['tdim']

    if(datatype == 'Byte'): 
        numtype = Int8
        new_numtype = Float32
    elif(datatype == 'Short'): 
        numtype = Int16
        new_numtype = Float32
    elif(datatype == 'Integer'): 
        numtype = Int32
        new_numtype = Float32
    elif(datatype == 'Float'): 
        numtype = Float32
        new_numtype = Float32
    elif(datatype == 'Double'): 
        numtype = Float64
        new_numtype = Float32
    elif(datatype == 'Complex'): 
        numtype = Complex32
        new_numtype = Complex
    else: 
        print "Unknown data type, aborting ..."
        sys.exit(1)
    length = xdim*ydim*zdim*tdim*hdr['data_length']
    datatype = hdr['datatype']

    if  hdr['filetype'] < 0:
        print "Could not find header for %s" % filename
        return -1
    if hdr['filetype'] == 'analyze':
        f_img = open(hdr['imgfile'],"r")
        if hdr['swap']:
            image = fromstring(f_img.read(length),numtype).byteswapped().astype(new_numtype)
        else:
            image = fromstring(f_img.read(length),numtype).astype(new_numtype)
#       Flip images left to right and top to bottom.
        img = zeros((tdim,zdim,ydim,xdim)).astype(new_numtype)
        jmg = zeros((ydim,xdim)).astype(new_numtype)
        image = reshape(image,(tdim,zdim,ydim,xdim))
        for t in range(tdim):
            for z in range(zdim):
                jmg[:,:] = fliplr(image[t,z,:,:])
                img[t,z,:,:] = flipud(jmg)
        image = img
    elif hdr['filetype'] == '4dfp':
        dot = string.rfind(filename,".4dfp.img")
        if dot < 0:
            dot = string.rfind(filename,".4dfp.ifh")
        if dot < 0:
            imgfile = filename + ".4dfp.img"
        else:
            imgfile = filename[:dot] + ".4dfp.img"
        f_img = open(imgfile,"r")
        image = fromstring(f_img.read(length),numtype).byteswapped().astype(Float)
    elif hdr['filetype'] == 'cub':
#       Advance file pointer to the end of the header.
        f_img = open(filename,'r')
        while 1:
            line = f_img.readline()
            if line == '\f\n': 
                break
        image = fromstring(f_img.read(len),numtype).byteswapped().astype(Float)
        image = reshape(image,(zdim,ydim,xdim))
#       Flip images left to right and top to bottom.
        img = zeros((zdim,ydim,xdim)).astype(Float)
        jmg = zeros((ydim,xdim)).astype(Float)
        for z in range(zdim):
            jmg[:,:] = fliplr(image[z,:,:])
            img[z,:,:] = flipud(jmg)
        image = img
    elif hdr['filetype'] == 'tes':
        f_img = open(filename,'r')
        while 1:
            line = f_img.readline()
            if line == '\f\n': 
                break

#       Read mask.
        mask = zeros((zdim*ydim*xdim)).astype(Int0)
        mask[:] = fromstring(f_img.read(xdim*ydim*zdim),Int0)
        idx = nonzero(mask)
        nidx = shape(idx)[0]

#       Read image.
        image = zeros((tdim,zdim*ydim*xdim)).astype(Float)
        length = tdim*hdr['data_length']
        for i in range(nidx):
            image[:,idx[i]] = fromstring(f_img.read(length),numtype).byteswapped().astype(Float)
    else:
        print "read_file error: Image type %s is not supported." % hdr['filetype']
        sys.exit(1)

    f_img.close()
    if tdim > 1:
        image = reshape(image,(tdim,zdim,ydim,xdim))
    else:
        image = reshape(image,(zdim,ydim,xdim))

    return {'header':hdr,'image':image}


#*********************************
def write_file(filename,filedata):
#*********************************

#   Write image file

    hdr = filedata['header']
    image = filedata['image']

    xdim = hdr['xdim']
    ydim = hdr['ydim']
    zdim = hdr['zdim']
    tdim = hdr['tdim']
    xsize = hdr['xsize']
    ysize = hdr['ysize']
    zsize = hdr['zsize']
    x0 = hdr['x0']
    y0 = hdr['y0']
    z0 = hdr['z0']
    if(hdr.has_key('TR')):
        TR = hdr['TR']
    else:
        TR = 0.
    if(hdr.has_key('typecode')):
        typecode = hdr['typecode']
    else:
        typecode = image.typecode()

    params = {}

    if hdr['filetype'] == 'analyze':
        write_analyze(filename,hdr,image)
    elif hdr['filetype'] == 'cub':
        write_cub(image,filename,xdim,ydim,zdim,xsize,ysize,zsize,x0,y0,z0,typecode,params)
    elif hdr['filetype'] == 'tes':
        typecode = 's'
        write_tes(image,filename,xdim,ydim,zdim,tdim,xsize,ysize,zsize,x0,y0,z0,TR,typecode,params)
    elif hdr['filetype'] == '4dfp': 
        dot = string.rfind(filename,".4dfp.img")
        if dot < 0:
            dot = string.rfind(filename,".4dfp.ifh")
        if dot < 0:
            dot = string.rfind(filename,".img")
            if dot < 0:
                img_file = filename + ".4dfp.img"
                hdr_file = filename + ".4dfp.ifh"
            else:
                img_file = filename[:dot] + ".4dfp.img"
                hdr_file = filename[:dot] + ".4dfp.ifh"
        else:
            img_file = filename[:dot] + ".4dfp.img"
            hdr_file = filename[:dot] + ".4dfp.ifh"
        write_ifh(hdr_file,xdim,ydim,zdim,tdim,xsize,ysize,zsize)
        f = open(img_file,'w')
        f.write(image.astype(Float32).byteswapped().tostring())
        f.close()
    else:
        print "write_file error: Image type %s is not supported." % hdr['filetype']
        sys.exit(1)

#*************************
def read_header(filename):
#*************************

# Read image header.

    type = file_type(filename)
    if type == -1:
        return(-1)
    xdim = -1
    ydim = -1
    zdim = -1
    tdim = -1
    xsize = -1.
    ysize = -1.
    zsize = -1.
    x0 = 0
    y0 = 0
    z0 = 0
    TR = 0.
    orientation = ""
    data_length = 0
    imgfile = ""
    num_voxels = 0
    swap = 0
    if(type == "cub"):
        whole_header = []
        infile = open(filename,'r')
        while 1:
            line = infile.readline()
            whole_header = whole_header + [line]
            if (string.find(line,'DataType:') == 0):
                words = string.split(line)
                datatype = words[1]
                if(datatype == 'Byte'): data_length = 1
                elif(datatype == 'Integer'): 
                    data_length = 2
                    datatype = 'Short'
                elif(datatype == 'Float'): data_length = 4
                elif(datatype == 'Double'): data_length = 8
                else: data_length = 4
            elif (string.find(line,'VoxDims(XYZ):') == 0):
                words = string.split(line)
                xdim = int(words[1])
                ydim = int(words[2])
                zdim = int(words[3])
                tdim = 1
                num_voxels = xdim*ydim*zdim
            elif string.find(line,"VoxSizes(XYZ):") >= 0:
                words = string.split(line)
                xsize = float(words[1])
                ysize = float(words[2])
                zsize = float(words[3])
            elif line == '\f\n': 
                start_binary = infile.tell()
                break
        infile.close()
    elif(type == "tes"):
        whole_header = []
        infile = open(filename,'r')
        while 1:
            line = infile.readline()
            whole_header = whole_header + [line]
            if (string.find(line,'DataType:') == 0):
                words = string.split(line)
                datatype = words[1]
                if(datatype == 'Byte'): data_length = 1
                elif(datatype == 'Integer'): 
                    data_length = 2
                    datatype = 'Short'
                elif(datatype == 'Float'): data_length = 4
                elif(datatype == 'Double'): data_length = 8
                else: data_length = 4
            elif (string.find(line,'VoxDims(TXYZ):') == 0):
                words = string.split(line)
                tdim = int(words[1])
                xdim = int(words[2])
                ydim = int(words[3])
                zdim = int(words[4])
                num_voxels = xdim*ydim*zdim*tdim
            elif string.find(line,"VoxSizes(XYZ):") >= 0:
                words = string.split(line)
                xsize = float(words[1])
                ysize = float(words[2])
                zsize = float(words[3])
            elif string.find(line,"TR(msecs):") >= 0:
                words = string.split(line)
                TR = float(words[1])
            elif string.find(line,"orientation:") >= 0:
                words = string.split(line)
                orientation = words[1]
            elif line == '\f\n': 
                start_binary = infile.tell()
                break
        infile.close()
    elif(type == "analyze"):
        start_binary = -1
        whole_header = []
        dot = string.find(filename,".img")
        if(dot < 0):
            dot = string.find(filename,".hdr")
        if dot < 0:
            hdrname = filename + '.hdr'
        else:
            hdrname = filename[0:dot]+".hdr"
        hdr = read_analyze_header(hdrname)
        return(hdr)
    elif(type == "4dfp"):
        start_binary = -1
        whole_header = []
        datatype = 'Float'
        dot = string.rfind(filename,".4dfp.img")
        if dot < 0:
            dot = string.rfind(filename,".4dfp.ifh")
        if dot < 0:
            ifhfile = filename + ".4dfp.ifh"
        else:
            ifhfile = filename[:dot] + ".4dfp.ifh"
        infile = open(ifhfile,'r')
        lines = infile.readlines()
        for line in lines:
            words = string.split(line)
            whole_header = whole_header + [line]
            if (string.find(line,'number of dimensions') == 0):
                if len(words) == 5:
                    ndim = words[4]
            elif (string.find(line,'matrix size [1]') == 0):
                if len(words) == 5:
                    xdim = int(words[4])
            elif (string.find(line,'matrix size [2]') == 0):
                if len(words) == 5:
                    ydim = int(words[4])
            elif (string.find(line,'matrix size [3]') == 0):
                if len(words) == 5:
                    zdim = int(words[4])
            elif (string.find(line,'matrix size [4]') == 0):
                if len(words) == 5:
                    tdim = int(words[4])
            elif (string.find(line,'scaling factor (mm/pixel) [1]') == 0):
                if len(words) == 6:
                    xsize = float(words[5])
            elif (string.find(line,'scaling factor (mm/pixel) [2]') == 0):
                if len(words) == 6:
                    ysize = float(words[5])
            elif (string.find(line,'scaling factor (mm/pixel) [3]') == 0):
                if len(words) == 6:
                    zsize = float(words[5])
            elif (string.find(line,'slice thickness (mm/pixel)') == 0):
                if len(words) == 5:
                    zsize = float(words[4])
            elif (string.find(line,'number format') == 0):
                if len(words) == 4:
                    datatype = words[3]
                else:
                    datatype = "float"
                if(datatype == 'byte'): 
                    data_length = 1
                    datatype = "Byte"
                elif(datatype == 'int'): 
                    data_length = 2
                    datatype = "Short"
                elif(datatype == 'float'): 
                    data_length = 4
                    datatype = "Float"
                elif(datatype == 'double'): 
                    data_length = 8
                    datatype = "Double"
                elif(datatype == 'complex'): 
                    data_length = 8
                    datatype = "Complex"
                else: 
                    data_length = 4
                    datatype = "Float"
    elif(type == "fdf"):
        whole_header = []
        while 1:
            line = infile.readline()
            whole_header = whole_header + [line]
            if (string.find(line,'storage =') == 0):
                words = string.split(line)
                datatype = words[3][1:-2]
            elif (string.find(line,'bits =') == 0):
                words = string.split(line)
                data_length = words[3][:-1]
            elif (string.find(line,'matrix') == 0):
                words = string.split(line)
                xdim = int(words[3][2:-1])
                ydim = int(words[4][:-2])
                num_voxels = xdim*ydim*zdim
            elif string.find(line,"slice_no =") >= 0:
                words = string.split(line)
                slice = int(words[3][:-1])
            elif string.find(line,"slices =") >= 0:
                words = string.split(line)
                zdim = int(words[3][:-1])
            elif string.find(line,"span[] =") >= 0:
                words = string.split(line)
                zdim = int(words[3][:-1])
            elif line == '\f\n': 
                start_binary = infile.tell()
                break
        infile.close()
        num_voxels = xdim*ydim*zdim*tdim
    else:
        print "Unknown file type.\n"
        return -1

    hdr = {'xdim':xdim,'ydim':ydim,'zdim':zdim,'tdim':tdim,'xsize':xsize,'ysize':ysize,'zsize':zsize,'x0':x0,'y0':y0,'z0':z0,'TR':TR,'orientation':orientation,'whole_header':whole_header,'start_binary':start_binary,'data_length':data_length,'num_voxels':num_voxels,'datatype':datatype,'filetype':type,'imgfile':imgfile,'swap':swap}
 
    return hdr


#**************************
def exec_cmd(cmd,skip,lcv,exit_on_error=False):
#**************************

# Execute unix command and handle errors.
    if(skip):
        return
    if(lcv):
        print "\n" +  cmd
    status = os.system(cmd)
    if(status):
        print "\n****** Error occurred while executing: ******"
        print cmd
        print "Aborting procedure\n\n"
        if exit_on_error: sys.exit(1)


#*************************************************************
def write_ifh(filename,xdim,ydim,zdim,tdim,xsize,ysize,zsize):
#*************************************************************

# Write fidl 4dfp header file

    suffix = ".4dfp.ifh"
    dot = string.rfind(filename,".4dfp.img")
    if dot < 0:
        dot = string.rfind(filename,".4dfp.ifh")
    if dot < 0:
        dot = string.rfind(filename,".img")
    if dot < 0:
        dot = string.rfind(filename,".hdr")
    if dot < 0:
        ifhname = filename + ".4dfp.ifh"
    else:
        ifhname = filename[:dot] + suffix

    f = open(ifhname,"w")
    f.write("INTERFILE                         := \n")
    f.write("version of keys                   := 3.3\n")
    f.write("image modality                    := \n")
    f.write("originating system                := \n")
    f.write("conversion program                := \n")
    f.write("program version                   := fidl revision\n")
    f.write("program date                      := \n")
    f.write("original institution              := \n")
    f.write("name of data file                 := \n")
    f.write("patient ID                        := \n")
    f.write("study date                        := \n")
    f.write("number format                     := float\n")
    f.write("number of bytes per pixel         :=        0\n")
    f.write("orientation                       :=        0\n")
    f.write("time series flag                  :=        0\n")
    f.write("number of dimensions              :=        4\n")
    f.write("matrix size [1]                   :=           %d\n" % (xdim))
    f.write("matrix size [2]                   :=           %d\n" % (ydim))
    f.write("matrix size [3]                   :=           %d\n" % (zdim))
    f.write("matrix size [4]                   :=           %d\n" % (tdim))
    f.write("scaling factor (mm/pixel) [1]     :=       %f\n" % (xsize))
    f.write("scaling factor (mm/pixel) [2]     :=       %f\n" % (ysize))
    f.write("scaling factor (mm/pixel) [3]     :=       %f\n" % (zsize))
    f.write("slice thickness (mm/pixel)        :=       %f\n" % (zsize))
    f.write("matrix initial element [1]        := \n")
    f.write("matrix initial element [2]        := \n")
    f.write("matrix initial element [3]        := \n\n\n")
    f.write("global minimum                    :=       0.00000\n")
    f.write("global maximum                    :=       0.00000\n")
    f.write("Gaussian field smoothness         :=       0.00000\n")
    f.write("mri parameter file name           := \n")
    f.write("mri sequence file name            := \n")
    f.write("mri sequence description          := \n")
    f.write("paradigm format                   := 0x\n")

    return

#**************************
def write_params(params,f):
#**************************

# Write parameters to TES/CUB headers

    if(params.has_key('tr')):
        if(params.has_key('nseg')):
#           Mult-shot acquisition.  TR is for one segment only.
            str = "MRI_TR(secs): %f\n" % (float(params['nseg'])*float(params['tr']))
            f.write(str)
        else:
            f.write("MRI_TR(secs):	%s\n" % params['tr'])
    if(params.has_key('dp')):
        f.write("MRI_Precision:	%s\n" % params['dp'])
    if(params.has_key('orient')):
        str = params['orient']
        f.write("MRI_Orientation:	%s\n" % str)
    if(params.has_key('petable')):
        f.write("MRI_PhaseEncodeTable:	%s\n" % params['petable'])
    if(params.has_key('te')):
        f.write("MRI_EchoTime(sec):	%s\n" % params['te'])
    if(params.has_key('nvol')):
        f.write("MRI_NumberofVolumes:	%s\n" % params['nvol'])
    if(params.has_key('np')):
        f.write("MRI_NumberFreqEncodeSamples:	%s\n" % params['np'])
    if(params.has_key('pss')):
        f.write("MRI_SlicePositions(cm):	%s\n" % params['pss'])
    if(params.has_key('nv')):
        f.write("MRI_NumberPhaseEncodes:	%s\n" % params['nv'])
    if(params.has_key('thk')):
        f.write("MRI_SliceThickness(mm):	%s\n" % params['thk'])
    if(params.has_key('pulse_sequence')):
        f.write("MRI_PulseSequence:	%s\n" % params['pulse_sequence'])
    if(params.has_key('gap')):
        f.write("MRI_SliceGap(mm):	%s\n" % params['gap'])
    if(params.has_key('nslice')):
        f.write("MRI_NumberofSlices:	%s\n" % params['nslice'])
    if(params.has_key('nv2')):
        f.write("MRI_nv2:	%s\n" % params['nv2'])
    if(params.has_key('at')):
        f.write("MRI_ADC_OnTime(sec):	%s\n" % params['at'])
    if(params.has_key('gro')):
        f.write("MRI_ReadoutGradient(G/cm):	%s\n" % params['gro'])
    if(params.has_key('gmax')):
        f.write("MRI_MaxGradient(G/cm):	%s\n" % params['gmax'])
    if(params.has_key('trise')):
        f.write("MRI_GradientRiseTime(sec):	%s\n" % params['trise'])
    if(params.has_key('fov')):
        f.write("MRI_FieldofView_lro(cm):	%s\n" % params['fov'])
    if(params.has_key('recon_rev')):
        f.write("ReconstructionRevision:	%s\n" % params['recon_rev'])
    if(params.has_key('Date')):
        f.write("DateCreated:	%s\n" % params['date'])
    if(params.has_key('lpe')):
        f.write("MRI_FieldofView_lpe(cm):	%s\n" % params['lpe'])
    if(params.has_key('gain')):
        f.write("MRI_Gain:	%s\n" % params['gain'])
    return 0

#***********************
def make_header(params):
#***********************

# Create header containing procpar parameters

    hdr = {}

    N_fe = int(params['np'])
    N_pe = int(params['nv'])
    nslice = int(params['nslice'])
    if(params.has_key('nvol')):
        nvol = int(params['nvol'])
    else:
        nvol = 1
    hdr['xdim'] = N_fe/2
    if params['pulse_sequence'] == 'epi':
        hdr['ydim'] = N_pe - 2
    else:
        hdr['ydim'] = N_pe
    hdr['tdim'] = nvol - 1
    hdr['zdim'] = nslice
    slice_pos = params['pss']
    hdr['thk'] = float(params['thk'])
    min = 10.*float(slice_pos[0])
    max = 10.*float(slice_pos[nslice-1])
    thk = float(params['thk'])
    gap = ((max - min + thk) - (nslice*thk))/(nslice - 1)
    hdr['xsize'] =  float(params['fov'])/(float(params['nv'])-2)
    hdr['ysize'] = hdr['xsize']
    hdr['zsize'] = float(params['thk']) + gap
    hdr['gap'] = "%f" % gap
    hdr['te'] = float(params['te'])
    hdr['trise'] = float(params['trise'])
    hdr['gro'] = float(params['gro'])
    hdr['gmax'] = float(params['gmax'])
    hdr['at'] = float(params['at'])
    hdr['pulse_sequence'] = params['pulse_sequence']
    hdr['TR'] = 2*float(params['tr'])
    hdr['nv'] = int(params['nv'])
    hdr['orient'] =  params['orient'][1:-1]
    return hdr
    
#********************************************************************************
def write_tes(image,filename,xdim,ydim,zdim,tdim,xsize,ysize,zsize,x0,y0,z0,TR,typecode,params):
#********************************************************************************

# Write TES file.

# image:	Image to be written.
# filename:	File name to be written to.
# xdim:		X Dimension
# ydim:		Y Dimension
# zdim:		Z Dimension
# tdim:		T Dimension
# xsize: 	X voxel size in mm.
# ysize: 	Y voxel size in mm.
# zsize: 	Z voxel size in mm.
# x0:		X origin
# y0:		Y origin
# z0:		Z origin
# typecode: Python typecode for image number format (d,f,l,i,c)
# params:   Dictionary of variables to be written to user header.

    f = open(filename,"w")
    f.write("VB98\nTES1\n")

    image_typecode = image.typecode()
    if typecode == "d":
        f.write("DataType:	Double\n")
        if(image_typecode != typecode):
            img = image.astype(Float64)
            jmg = zeros((ydim,xdim)).astype(Float64)
    elif typecode == "f":
        f.write("DataType:	Float\n")
        if(image_typecode != typecode):
            img = image.astype(Float32)
            jmg = zeros((ydim,xdim)).astype(Float32)
    elif typecode == "l":
        f.write("DataType:	Long\n")
        if(image_typecode != typecode):
            img = image.astype(Int32)
            jmg = zeros((ydim,xdim)).astype(Int32)
    elif typecode == "s":
        f.write("DataType:	Integer\n")
        if(image_typecode != typecode):
            scl = 16383./image.flat[argmax(image.flat)]
            img = (scl*image).astype(Int16)
            jmg = zeros((ydim,xdim)).astype(Int16)
    elif typecode == "c":
        f.write("DataType:	Byte\n")
    else:
        print "Invalid type code: %s" % typecode
        return -1

    f.write("VoxDims(TXYZ):	%d	%d	%d	%d\n" % (tdim,xdim,ydim,zdim))
    f.write("VoxSizes(XYZ):	%f	%f	%f\n" % (xsize,ysize,zsize))
    f.write("Origin(XYZ):	%d	%d	%d\n" % (x0,y0,z0))
    xTR = 100*int(TR*10+.5)
    f.write("TR(msecs):	%d\n" % (xTR))

    write_params(params,f)

#   Write end-of-header mark.
    f.write("\f\n")

#   Write the mask.
    mask = ones((zdim*ydim*xdim),Int0)
    f.write(mask.tostring())

#   Flip images left to right and top to bottom.
    for t in range(tdim):
        for z in range(zdim):
            jmg[:,:] = fliplr(img[t,z,:,:])
            img[t,z,:,:] = flipud(jmg)

#   Now write the data. 
#   Note that numpy index ordering is the same as in C - last index varies fastest.
    img = reshape(img,(tdim,zdim,ydim,xdim))
    for z in range(zdim):
        for y in range(ydim):
            for x in range(xdim):
                f.write(img[:,z,y,x].astype(typecode).byteswapped().tostring())

    if image_typecode != image.typecode():
        image = image.astype(image_typecode)

    f.close()
    return 0

#***************************************************************************************
def write_cub(image,filename,xdim,ydim,zdim,xsize,ysize,zsize,x0,y0,z0,typecode,params):
#***************************************************************************************

# Write cub image.

# image:	Image to be written.
# filename:	File name to be written to.
# xdim:		X Dimension
# ydim:		Y Dimension
# zdim:		Z Dimension
# xsize: 	X voxel size in mm.
# ysize: 	Y voxel size in mm.
# zsize: 	Z voxel size in mm.
# x0:		X origin
# y0:		Y origin
# z0:		Z origin
# typecode: Python typecode for image number format (d,f,l,i,c)
# params:   Dictionary of variables to be written to user header.

    f = open(filename,"w")
    f.write("VB98\nCUB1\n")

    image_typecode = image.typecode()
    if typecode == "d":
        f.write("DataType:	Double\n")
        if(image_typecode != typecode):
            img = image.astype(Float64)
    elif typecode == "f":
        f.write("DataType:	Float\n")
        if(image_typecode != typecode):
            img = image.astype(Float32)
    elif typecode == "l":
        f.write("DataType:	Long\n")
        if(image_typecode != typecode):
            img = image.astype(Int32)
    elif typecode == "s":
        f.write("DataType:	Integer\n")
        if(image_typecode != typecode):
            scl = 16383./image.flat[argmax(image.flat)]
            img = (scl*image).astype(Int16)
    elif typecode == "c":
        f.write("DataType:	Byte\n")
    else:
        print "Invalid type code: %s" % typecode
        return -1

    f.write("VoxDims(XYZ):	%d	%d	%d\n" % (xdim,ydim,zdim))
    f.write("VoxSizes(XYZ):	%f	%f	%f\n" % (xsize,ysize,zsize))
    f.write("Origin(XYZ):	%d	%d	%d\n" % (x0,y0,z0))

    write_params(params,f)

#   Write end-of-header mark.
    f.write("\f\n")

#   Flip images left to right and top to bottom.
    img = zeros((zdim,ydim,xdim)).astype(Float)
    jmg = zeros((ydim,xdim)).astype(Float)
    for z in range(zdim):
        jmg[:,:] = fliplr(image[z,:,:])
        img[z,:,:] = flipud(jmg)
    image = img

#   Now write the data. 
#   Note that numpy index ordering is the same as in C - last index varies fastest.
    f.write(image.astype(typecode).byteswapped().tostring())

    if image_typecode != image.typecode():
        image = image.astype(image_typecode)

    f.close()

#*************************************
def write_analyze(filename,hdr,image):
#*************************************

# Write analyze image.

    dot = string.find(filename,".img")
    if(dot < 0):
        dot = string.find(filename,".hdr")
    if dot < 0:
        hdrname = filename + '.hdr'
        imgfile = filename + '.img'
    else:
        hdrname = filename[0:dot]+".hdr"
        imgfile = filename[0:dot]+".img"

    xdim = hdr['xdim']
    ydim = hdr['ydim']
    zdim = hdr['zdim']
    tdim = hdr['tdim']

#   Flip images left to right and top to bottom.
    if hdr['datatype'] == "Complex":
        img = zeros((tdim,zdim,ydim,xdim)).astype(Complex)
        jmg = zeros((ydim,xdim)).astype(Complex)
    else:
        img = zeros((tdim,zdim,ydim,xdim)).astype(Float)
        jmg = zeros((ydim,xdim)).astype(Float)
    image = reshape(image,(tdim,zdim,ydim,xdim))
    for t in range(tdim):
        for z in range(zdim):
            jmg[:,:] = fliplr(image[t,z,:,:])
            img[t,z,:,:] = flipud(jmg)
    if tdim == 1:
        img = reshape(img,(zdim,ydim,xdim))

#   First write the image file.
    if hdr['datatype'] != "Complex":
        min = img.flat[argmin(img.flat)]
        max = img.flat[argmax(img.flat)]
        if max < -min:
            max = -min
        if max == 0.:
            max = 1.e20
    scl = 1.
    if hdr['datatype'] == "Byte":
        if max > 255.:
            scl = 255./max
        img = (scl*img).astype(Int8)
    elif hdr['datatype'] == "Short":
        if max > 32767.:
            scl = 32767./max
        img = (scl*img).astype(Int16)
    elif hdr['datatype'] == "Integer":
        if max > 2147483648.:
            scl = 2147483648./max
        img = (scl*img).astype(Int32)
    elif hdr['datatype'] == "Float":
        img = img.astype(Float32)
    elif hdr['datatype'] == "Double":
        img = img.astype(Float64)
    elif hdr['datatype'] == "Complex":
        img = img.astype(Complex32)

    f_img = open(imgfile,"w")
    if hdr['swap']:
        f_img.write(img.byteswapped().tostring())
    else:
        f_img.write(img.tostring())
    f_img.close()
    write_analyze_header(hdrname,hdr)


#***********************
def dump_procpar(filename):
#***********************

# Dump procpar parameters to screen.

    params = parse_procpar(filename,0)
    print params

    N_fe = int(params['np'])
    N_pe = int(params['nv'])
    nslice = int(params['nslice'])
    if(params.has_key('nvol')):
        nvol = int(params['nvol'])
    else:
        nvol = 1
    pulse_sequence = params['pulse_sequence']
    slice_pos = params['pss']
    thk = float(params['thk'])
    if pulse_sequence != 'mp_flash3d':
        min = 10.*float(slice_pos[0])
        max = 10.*float(slice_pos[nslice-1])
        if nslice > 1:
            gap = ((max - min + thk) - (nslice*thk))/(nslice - 1)
        else:
            gap = 0
    else:
        gap = 0
    params['gap'] = gap
    if(params.has_key('fov') and params.has_key('nv')):
        pixsize =  float(params['fov'])/(float(params['np']))
    else:
        pixsize = -1.

    print "Number of frequency encodes: %d" % (N_fe/2)
    print "Number of phase encodes: %d" % (N_pe)
    print "Number of slices: %d" % (nslice)
    print "Number of frames (including phase correction blocks): %d" % (nvol)
    print "Slice thickness: %f" % (thk)
    print "Inter-slice gap: %f" % (gap)
    print "Pixel size: %f" % (pixsize)

    if(params.has_key('tr')):
        if(params.has_key('nseg')):
#           Mult-shot acquisition.  TR is for one segment only.
            str = "TR in secs: %f\n" % (float(params['nseg'])*float(params['tr']))
            sys.stdout.write(str)
        else:
            sys.stdout.write("TR in secs:	%s\n" % params['tr'])
    if(params.has_key('dp')):
        sys.stdout.write("Precision:	%s\n" % params['dp'])
    if(params.has_key('orient')):
        str = params['orient']
        sys.stdout.write("Orientation:	%s\n" % str)
    if(params.has_key('petable')):
        sys.stdout.write("Phase encode table:	%s\n" % params['petable'])
    if(params.has_key('te')):
        sys.stdout.write("Echo time in seconds:	%s\n" % params['te'])
    if(params.has_key('pss')):
        sys.stdout.write("Slice positions in cm: %s\n" % params['pss'])
    if(params.has_key('pulse_sequence')):
        sys.stdout.write("Pulse sequence:	%s\n" % params['pulse_sequence'])
    if(params.has_key('nv2')):
        sys.stdout.write("nv2:	%s\n" % params['nv2'])
    if(params.has_key('at')):
        sys.stdout.write("ADC on time in seconds:	%s\n" % params['at'])
    if(params.has_key('gro')):
        sys.stdout.write("Readout gradient  in G/cm (gro):	%s\n" % params['gro'])
    if(params.has_key('sw')):
        sys.stdout.write("(sw):	%s\n" % params['sw'])
    if(params.has_key('gmax')):
        sys.stdout.write("Maximum gradient  in G/cm (gmax):	%s\n" % params['gmax'])
    if(params.has_key('trise')):
        sys.stdout.write("Gradient rise time  in seconds (trise):	%s\n" % params['trise'])
    if(params.has_key('fov')):
        sys.stdout.write("Field of view (lro) in cm:	%s\n" % params['fov'])
    if(params.has_key('recon_rev')):
        sys.stdout.write("Reconstruction code Revision:	%s\n" % params['recon_rev'])
    if(params.has_key('Date')):
        sys.stdout.write("Date created:	%s\n" % params['date'])
    if(params.has_key('lpe')):
        sys.stdout.write("Field of view (lpe) in cm (lpe):	%s\n" % params['lpe'])
    if(params.has_key('gain')):
        sys.stdout.write("Gain:	%s\n" % params['gain'])
    if(params.has_key('asym_time')):
        sys.stdout.write("Echo time asymmetry (asym_time):	%s\n" % params['asym_time'])
    if(params.has_key('theta')):
        theta = float(params['theta'])
        sys.stdout.write("theta: %6.2f degrees\n" % theta)
    if(params.has_key('phi')):
        phi = float(params['phi'])
        sys.stdout.write("phi: %6.2f degrees\n" % phi)
    if(params.has_key('psi')):
        psi = float(params['psi'])
        sys.stdout.write("psi: %6.2f degrees\n" % psi)
    if(params.has_key('dwell')):
        dwell = 1000.*float(params['dwell'])
        sys.stdout.write("Dwell time: %6.3f msec\n" % dwell)

#***************************************************************************************
def dump_image(filename,image,xdim,ydim,zdim,tdim,xsize,ysize,zsize,reorder,flip_slices):
#***************************************************************************************

# Quick save of image array to a file.
# Reorder slices from acquisition order to slice order if reorder=1
# Flip slices from top to bottom if flip = 1

    dot = string.rfind(filename,".4dfp.img")
    if dot < 0:
        dot = string.rfind(filename,".4dfp.ifh")
    if dot < 0:
        imgname = filename + ".4dfp.img"
        ifhname = filename + ".4dfp.ifh"
        hdrname = filename + ".4dfp.hdr"
    else:
        imgname = filename[:dot] + ".4dfp.img"
        ifhname = filename[:dot] + ".4dfp.ifh"
        hdrname = filename[:dot] + ".4dfp.hdr"

    if tdim > 1:
        img = reshape(image,(tdim,zdim,ydim,xdim))
    else:
        img = reshape(image,(zdim,ydim,xdim))
    img_vol = zeros((zdim,ydim,xdim)).astype(Float32)
    if reorder:
        midpoint = zdim/2
        for slice in range(zdim):
            if slice < midpoint:
                if flip_slices:
                    z = 2*slice
                else:
                    z = zdim-2*slice-1
                img_vol[z,:,:] = img[slice,:,:].astype(Float32)
            else:
                if flip_slices:
                    z = 2*(slice-midpoint) + 1
                else:
                    z = zdim-2*(slice-midpoint)-2
                img_vol[z,:,:] = img[slice,:,:].astype(Float32)
    else:
        for slice in range(zdim):
            img_vol = img.astype(Float32)

    f_image = open(imgname,"w")
    f_image.write(img_vol.byteswapped().tostring())
    f_image.close()
    write_ifh(ifhname,xdim,ydim,zdim,tdim,xsize,ysize,zsize)

    hdr = create_hdr(xdim,ydim,zdim,tdim,xsize,ysize,zsize,1.,0,0,0,'Float',32,1.,'analyze',imgname,1)
    write_analyze_header(hdrname,hdr)

#*****************************************
def get_matlab_matrix(file_name,mat_name):
#*****************************************

# Get matlab matrix contents.

    cmd = "dump_mat %s -get_dims %s" % (file_name,mat_name)
    words = string.split((os.popen(cmd).readlines())[0])
    nrows = int(words[1])
    ncols = int(words[2])
    size = nrows*ncols*8
    cmd = "dump_mat %s -dump_binary %s" % (file_name,mat_name)
    matrix = transpose(reshape(fromstring(os.popen(cmd).read(size),Float64),(ncols,nrows)))

    return matrix

#*****************************************
def get_matlab_matrix_dims(file_name,mat_name):
#*****************************************

# Get matlab matrix dimensions and return in a two-element list.

    cmd = "dump_mat %s -get_dims %s" % (file_name,mat_name)
    words = string.split((os.popen(cmd).readlines())[0])
    nrows = int(words[1])
    ncols = int(words[2])
    dims = [nrows,ncols]

    return dims


#***************************************************
def modify_matlab_matrix(file_name,mat_name,matrix):
#***************************************************

# Get matlab matrix contents.

    cmd = "dump_mat %s -get_dims %s" % (file_name,mat_name)
    words = string.split((os.popen(cmd).readlines())[0])
    nrows_file = int(words[1])
    ncols_file = int(words[2])

    shp = shape(matrix)
    nrows = shp[0]
    ncols = shp[1]
    if (nrows != nrows_file) or (ncols != ncols_file):
        print "*** Incompatible matrices in modify_matlab_matix.  Aborting ... ***"
        return 1
    matrix = transpose(matrix)


    cmd = "modify_mat %s %s" % (file_name,mat_name)
    f = os.popen(cmd,"w").write(matrix.astype(Float64).tostring())
    if f != None:
        print "Error writing %s to %s." % (file_name,mat_name)
        return(1)

    return 0


#*****************************
def read_ucb_xform(file_name):
#*****************************

# Read transformation matrix and parameters from a .xfm file.

    f_in = open(file_name,"r")
    lines = f_in.readlines()
    f_in.close()
    tdim = (len(lines) - 6)/4

    words = string.split(lines[2])
    xdim_in = int(words[0])
    ydim_in = int(words[1])
    zdim_in = int(words[2])
    words = string.split(lines[3])
    xdim_out = int(words[0])
    ydim_out = int(words[1])
    zdim_out = int(words[2])
    words = string.split(lines[4])
    xsize_in = float(words[0])
    ysize_in = float(words[1])
    zsize_in = float(words[2])
    words = string.split(lines[5])
    xsize_out = float(words[0])
    ysize_out = float(words[1])
    zsize_out = float(words[2])

    if tdim == 1:
        matrix = zeros((4,4)).astype(Float)
        for row in range(4):
            words = string.split(lines[6+row])
            matrix[row][0] = float(words[0])
            matrix[row][1] = float(words[1])
            matrix[row][2] = float(words[2])
            matrix[row][3] = float(words[3])
    else:
        matrix = zeros((tdim,4,4)).astype(Float)
        for t in range(tdim):
            for row in range(4):
                words = string.split(lines[6+row])
                matrix[t][row][0] = float(words[0])
                matrix[t][row][1] = float(words[1])
                matrix[t][row][2] = float(words[2])
                matrix[t][row][3] = float(words[3])
        
    xfm = {'xdim_in':xdim_in,'ydim_in':ydim_in,'zdim_in':zdim_in,'xsize_in':xsize_in,'ysize_in':ysize_in,'zsize_in':zsize_in,'xdim_out':xdim_out,'ydim_out':ydim_out,'zdim_out':zdim_out,'xsize_out':xsize_out,'ysize_out':ysize_out,'zsize_out':zsize_out,'matrix':matrix,'tdim':tdim}

    return xfm

#**********************************
def write_ucb_xform(file_name,xfm):
#**********************************

# Write transformation matrix and parameters to a .xfm file.

    f= open(file_name,"w")

    f.write("XFORM_UCB\n")
    f.write("1.0\n")
    f.write("%d %d %d\n" % (xfm['xdim_in'],xfm['ydim_in'],xfm['zdim_in']))
    f.write("%d %d %d\n" % (xfm['xdim_out'],xfm['ydim_out'],xfm['zdim_out']))
    f.write("%e %e %e\n" % (xfm['xsize_in'],xfm['ysize_in'],xfm['zsize_in']))
    f.write("%e %e %e\n" % (xfm['xsize_out'],xfm['ysize_out'],xfm['zsize_out']))

    matrix = xfm['matrix']
    dims = shape(matrix)
    if len(dims) == 2:
        for row in range(4):
            f.write("%e %e %e %e\n" % (matrix[row,0],matrix[row,1],matrix[row,2],matrix[row,3]))
    else:
        tdim = dims[0]
        for t in range(tdim):
            for row in range(4):
                f.write("%e %e %e %e\n" % (matrix[t,row,0],matrix[t,row,1],matrix[t,row,2],matrix[t,row,3]))
        
    f.close()

#*******************************************
def get_analyze_file_names(filename,suffix):
#*******************************************

#   Returns the string "[stem]_suffix.img" and "[stem]_suffix.hdr"
    dot = string.find(filename,".img")
    if(dot < 0):
        dot = string.find(filename,".hdr")
    if dot < 0:
        stem = filename
    else:
        stem = filename[0:dot]
    imgname = "%s.img" % stem
    hdrname = "%s.hdr" % stem

    return {'imgname':imgname,'hdrname':hdrname,'stem':stem}

#*********************************
def read_analyze_header(filename):
#*********************************


# 0: i       int sizeof_hdr
# 1: 10s     char data_type[10]
# 2: 18s     char db_name[18]
# 3: i       int extents
# 4: h       short int session_error
# 5: c       char regular
# 6: c       char hkey_un0
# 7: 8h      short int dim[8]
# 8: 4s      char vox_units[4]
# 9: 8s      char cal_units[8]
# 10: h       short int unused1
# 11: h       short int datatype
# 12: h       short int bitpix
# 13: h       short int dim_un0
# 14: 8f      float pixdim[8]
# 22: f       float vox_offset
# 23: f       float funused1
# 24: f       float funused2
# 25: f       float funused3
# 26: f       float cal_max
# 27: f       float cal_min
# 28: i       int compressed
# 29: i       int verified
# 30: 2i      int glmax, glmin
# 32: 80s     char descrip[80]
# 33: 24s     char aux_file[24]
# 34: c       char orient
# 35: 10s     char originator[10]
# 36: 10s     char generated[10]
# 37: 10s     char scannum[10]
# 38: 10s     char patient_id[10]
# 39: 10s     char exp_date[10]
# 40: 10s     char exp_time[10]
# 41: 3s      char hist_un0[3]
# 42: i       int views
# 43: i       int vols_added
# 44: i       int start_field
# 45: i       int field_skip
# 46: 2i      int omax,omin
# 48: 2i      int smax,smin

    format = "i 10s 18s i h c c 8h 4s 8s h h h h 8f f f f f f f i i 2i 80s 24s c 3h 4s 10s 10s 10s 10s 10s 3s i i i i 2i 2i"
    f = open(filename,'r')
    fmt = "<" + format
    vhdr = struct.unpack(fmt,f.read(348))
    f.close()

    length = vhdr[0]
    if length == 348:
        swap = 0
    else:
        swap = 1
        fmt = ">" + format
        f = open(filename,'r')
        vhdr = struct.unpack(fmt,f.read(348))
        f.close()

    whole_header = ''
    start_binary = -1
    orientation = 'trans'
    xdim = vhdr[8]
    ydim = vhdr[9]
    zdim = vhdr[10]
    tdim = vhdr[11]
    num_voxels = xdim*ydim*zdim*tdim
    datatype = vhdr[18]
    xsize = vhdr[22]
    ysize = vhdr[23]
    zsize = vhdr[24]
    TR = vhdr[25]
    scale_factor = vhdr[30]
    x0 = vhdr[42]
    y0 = vhdr[43]
    z0 = vhdr[44]

    if(datatype == 1): 
        data_length = 1
        datatype = 'Bit'
    elif(datatype == 2): 
        data_length = 1
        datatype = 'Byte'
    elif(datatype == 4): 
        data_length = 2
        datatype = 'Short'
    elif(datatype == 8): 
        data_length = 4
        datatype = 'Integer'
    elif(datatype == 16): 
        data_length = 4
        datatype = 'Float'
    elif(datatype == 32): 
        data_length = 8
        datatype = 'Complex'
    elif(datatype == 64): 
        data_length = 8
        datatype = 'Double'
    elif(datatype == 'Integer'): data_length = 2
    elif(datatype == 'Float'): data_length = 4
    elif(datatype == 'Double'): data_length = 8
    elif(datatype == 'Complex'): data_length = 8
    else: data_length = 4

    hdr = {'xdim':xdim,'ydim':ydim,'zdim':zdim,'tdim':tdim,'xsize':xsize,'ysize':ysize,'zsize':zsize,'x0':x0,'y0':y0,'z0':z0,'TR':TR,'orientation':orientation,'whole_header':whole_header,'start_binary':start_binary,'data_length':data_length,'num_voxels':num_voxels,'datatype':datatype,'filetype':'analyze','swap':swap,'scale_factor':scale_factor}

    return(hdr)


#**************************************
def write_analyze_header(filename,hdr):
#**************************************

    xdim = hdr['xdim']
    ydim = hdr['ydim']
    zdim = hdr['zdim']
    tdim = hdr['tdim']
    TR = hdr['TR']
    xsize = hdr['xsize']
    ysize = hdr['ysize']
    zsize = hdr['zsize']
    x0 = hdr['x0']
    y0 = hdr['y0']
    z0 = hdr['z0']
    if(hdr.has_key('scale_factor')):
        scale_factor = hdr['scale_factor']
    else:
        scale_factor = 1.
    swap = hdr['swap']
 
    datatype = hdr['datatype']
    if(datatype == 'Byte'): 
        data_type = 2
        bitpix = 8
    elif(datatype == 'Short'): 
        data_type = 4
        bitpix = 16
    elif(datatype == 'Integer'): 
        data_type = 8
        bitpix = 32
    elif(datatype == 'Float'): 
        data_type = 16
        bitpix = 32
    elif(datatype == 'Complex'): 
        data_type = 32
        bitpix = 64
    elif(datatype == 'Double'): 
        data_type = 64
        bitpix = 64
    else: 
        data_type = 4
        bitpix = 16

    if tdim > 1:
        ndim = 4
    else:
        ndim = 3

    cd = " "
    sd = " "
    hd = 0
    id = 0
    fd = 0.

    format = "i 10s 18s i h 1s 1s 8h 4s 8s h h h h 8f f f f f f f i i 2i 80s 24s 1s 3h 4s 10s 10s 10s 10s 10s 3s i i i i 2i 2i"
    if swap:
        format = ">" + format
    else:
        format = "<" + format
    
    bhdr = struct.pack(format,348, sd, sd, id, hd, cd, cd, ndim, xdim, ydim, zdim, tdim, hd, hd, hd, sd, sd, hd, data_type, bitpix, hd, fd, xsize, ysize, zsize, TR, fd, fd, fd, fd, scale_factor, fd, fd, fd, fd, id, id, id, id, sd, sd, cd, x0, y0, z0, sd, sd, sd, sd, sd, sd, sd, id, id, id, id, id, id, id, id)

    f = open(filename,'w')
    f.write(bhdr)
    f.close()


#*************************************************************************
def create_hdr(xdim,ydim,zdim,tdim,xsize,ysize,zsize,TR,x0,y0,z0,data_type,data_length,scale_factor,filetype,imgfile,swap):
#*************************************************************************

    whole_header = ''
    start_binary = -1
    orientation = 'trans'
    num_voxels = xdim*ydim*zdim*tdim
    hdr = {'xdim':xdim,'ydim':ydim,'zdim':zdim,'tdim':tdim,'xsize':xsize,'ysize':ysize,'zsize':zsize,'x0':x0,'y0':y0,'z0':z0,'TR':TR,'orientation':orientation,'whole_header':whole_header,'start_binary':start_binary,'data_length':data_length,'num_voxels':num_voxels,'datatype':data_type,'filetype':type,'imgfile':imgfile,'swap':swap,'scale_factor':scale_factor}

    return(hdr)
