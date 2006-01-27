from pylab import randn, amax, Int8, Int16, Int32, Float32, Float64, Complex32
import struct

from imaging.imageio import BaseImage

# maximum numeric range for some smaller data types
maxranges = {
  Int8:  255.,
  Int16: 32767.,
  Int32: 2147483648.}

class AnalyzeImage (BaseImage):

    #-------------------------------------------------------------------------
    def __init__(self, filestem): pass


    #-------------------------------------------------------------------------
    def read_header(self, filename):

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

    #-------------------------------------------------------------------------
    def read_file(self, filename):

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


##############################################################################
class AnalyzeWriter (object):
    """
    """

    # (datatype, bitpix) for each Analyze datatype
    # datatype is a bit flag into a datatype byte of the Analyze header
    # bitpix is the number of bits per pixel (or voxel, as the case may be)
    BYTE = (2,8)
    SHORT = (4,16)
    INTEGER = (8,32)
    FLOAT = (16,32)
    COMPLEX = (32,64)
    DOUBLE = (64,64)

    # map Numeric typecode to Analyze datatype
    typecode2datatype = {
      Int8: BYTE,
      Int16: SHORT,
      Int32: INTEGER,
      Float32: FLOAT,
      Float64: DOUBLE,
      Complex32: COMPLEX}

    # map Analyze datatype to Numeric typecode
    datatype2typecode = dict([(v,k) for k,v in typecode2datatype.items()])

    #-------------------------------------------------------------------------
    def __init__(self, image, datatype=None, byteswap=False):
        self.image = image
        self.datatype = \
          datatype or self.typecode2datatype[image.data.typecode()]
        self.byteswap = byteswap

    #-------------------------------------------------------------------------
    def write(self, filestem):
        "Write ANALYZE format header, image file pair."
        headername, imagename = "%s.hdr"%filestem, "%s.img"%filestem
        self.write_header(headername)
        self.write_image(imagename)

    #-------------------------------------------------------------------------
    def write_header(self, filename):
        "Write ANALYZE format header (.hdr) file."
        image = self.image
        ndim, tdim, zdim, ydim, xdim = get_dims(image.data)
        scale_factor = getattr(image, "scale_factor", 1. )
        tr = getattr(image, "tr", 0.)
        datatype, bitpix = self.datatype
        cd = sd = " "
        hd = id = 0
        fd = 0.

        format = "%si 10s 18s i h 1s 1s 8h 4s 8s h h h h 8f f f f f f f i i 2i 80s 24s 1s 3h 4s 10s 10s 10s 10s 10s 3s i i i i 2i 2i"%\
                 (self.byteswap and ">" or "<")
        # why is TR used...?
        binary_header = struct.pack(
          format, 348, sd, sd, id, hd, cd, cd, ndim, xdim, ydim, zdim, tdim,
          hd, hd, hd, sd, sd, hd, datatype, bitpix, hd, fd, image.xsize,
          image.ysize, image.zsize, tr, fd, fd, fd, fd, scale_factor,
          fd, fd, fd, fd, id, id, id, id, sd, sd, cd, image.x0, image.y0,
          image.z0, sd, sd, sd, sd, sd, sd, sd, id, id, id, id, id, id, id, id)
        f = open(filename,'w')
        f.write(binary_header)
        f.close()

    #-------------------------------------------------------------------------
    def write_image(self, filename):
        "Write ANALYZE format image (.img) file."
        imagedata = self.image.data

        # if requested datatype does not correspond to image datatype, cast
        if self.datatype != self.typecode2datatype[imagedata.typecode()]:
            typecode = self.datatype2typecode[self.datatype]

            # Make sure image values are within the range of the desired datatype
            if self.datatype in (self.BYTE, self.SHORT, self.INTEGER):
                maxval = amax(abs(imagedata).flat)
                if maxval == 0.: maxval = 1.e20
                maxrange = maxranges[typecode]

                # if out of desired bounds, perform scaling
                if maxval > maxrange: imagedata *= (maxrange/maxval)

            # cast image values to the desired datatype
            imagedata = imagedata.astype( typecode )

        if self.datatype != self.COMPLEX: imagedata = abs(imagedata)

        # perform byteswap if requested
        if self.byteswap: imagedata = imagedata.byteswapped()

        # Write the image file.
        f = file( filename, "w" )
        f.write( imagedata.tostring() )
        f.close()


