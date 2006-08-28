import struct
import os, sys
from pylab import empty, Complex, Int16, floor, Float, array, repeat, arange, \
     zeros, exp
from odict import odict


datafileheader = odict((
    ('nblocks', 'l'), # num data blocks
    ('ntraces', 'l'), # num traces per block
    ('num_els', 'l'), # num simple elements per trace (2x # complex pts)
    ('ebytes', 'l'),  # 2 for 16-bit int, 4 for all others
    ('tbytes', 'l'),  # set to num_els*ebytes
    ('bbytes', 'l'),  # set to ntraces*tbytes + nbheaders*sizeof(datafileheader)
    ('vers_id', 'h'), # ?
    ('status', 'h'),  # try 0
    ('nbheaders', 'l') # num block headers per data block (should be 1)
))

datafileheader_formats = datafileheader.values()

datablockheader = odict((
    ('scale', 'h'),
    ('status', 'h'),
    ('index', 'h'), # set this
    ('mode', 'h'),
    ('ctcount', 'l'),
    ('lpval', 'f'),
    ('rpval', 'f'),
    ('lvl', 'f'), # set to 0
    ('tlt', 'f')  # set to 0
))

datablockheader_formats = datablockheader.values()

def backrot(sig, n):
    rsig = empty(sig.shape, sig.typecode())
    rsig[:len(sig)-n] = sig[n:]
    rsig[len(sig)-n:] = sig[:n]
    return rsig

def fieldvalues(struct_field, indications):
    defaults = {'i': 0, 'h': 0, 'f': 0., \
                'c': '\0', 's': '', 'B': 0, 'l': 0}
    #struct_names = struct_field.keys()
    #struct_formats = struct_field.values()
    return [indications.get(name) or defaults[struct_field[name]] \
            for name in struct_field.keys()]
    

def writeFid(m, name):
    # write a fid file in uncompressed format
    (nslice, ny, nx) = m.shape[-3:]
    oshape = m.shape
    T = len(m.shape) > 3 and m.shape[0] or 1
    m = reshape(m, (T,nslice,ny,nx))
    blockheadersize = struct.calcsize('='+' '.join(datablockheader_formats))
    file_header = {
        'nblocks': T*nslice,
        'ntraces': ny,
        'num_els': nx*2,
        'ebytes': 2,
        'tbytes': 2*(nx*2),
        'bbytes': ny*(2*(nx*2)) + blockheadersize,
        'nbheaders': 1
    }

    # fid's are expected to be big-endian
    fheader_str = struct.pack('>'+' '.join(datafileheader_formats),
                              *tuple(fieldvalues(datafileheader, file_header)))
    
    flip_my_data = sys.byteorder=='little'
    block_header = {
        'scale': 1,
        'index': 0,
    }

    f = open(name+'.fid', 'w')
    f.write(fheader_str)
    i_cnt = 1
    for vol in m:
        for sl in vol:
            block_header['index'] = i_cnt
            i_cnt += 1
            bheader_str = struct.pack('>'+' '.join(datablockheader_formats),
                                      *tuple(fieldvalues(datablockheader,
                                                         block_header)))
            f.write(bheader_str)
            if flip_my_data: f.write(sl.byteswapped().tostring())
            else: f.write(sl.tostring())

    f.close()
    m = reshape(m, oshape)
    return

def sawtoothVolSeries(dims):
    if len(dims) != 4:
        print "only makes a volumetric time series of 4 dimensions"
        return

    (T, nslice, ny, nx) = dims
    half_per = int(T/6.)
    rise = arange(half_per)/float(half_per) - 0.5
    sig = array((rise.tolist() + (-rise).tolist())*7)
    real_ts = sig[:T] + arange(T)*.0025 #add a little ramp
    imag_ts = zeros(T)#backrot(real_ts, int(T/10.))

    vol_series = empty((T,nslice,ny,2*nx), Int16)
    for s in range(nslice):
        for y in range(ny):
            for x in range(nx):
                vol_series[:,s,y,2*x] = (real_ts*32767.).astype(Int16)
                vol_series[:,s,y,2*x+1] = (imag_ts*32767.).astype(Int16)
    writeFid(vol_series)
##     blockheadersize = struct.calcsize('='+' '.join(datablockheader_formats))
##     file_header = {
##         'nblocks': T*nslice,
##         'ntraces': ny,
##         'num_els': nx*2,
##         'ebytes': 2,
##         'tbytes': 2*(nx*2),
##         'bbytes': ny*(2*(nx*2)) + blockheadersize,
##         'nbheaders': 1
##     }

##     # fid's are expected to be big-endian
##     fheader_str = struct.pack('>'+' '.join(datafileheader_formats),
##                               *tuple(fieldvalues(datafileheader, file_header)))
    
##     block_header = {
##         'scale': 1,
##         'index': 0,
##     }

##     f = open('fake.fid', 'w')
##     f.write(fheader_str)
##     i_cnt = 1
##     for vol in vol_series:
##         for sl in vol:
##             block_header['index'] = i_cnt
##             i_cnt += 1
##             bheader_str = struct.pack('>'+' '.join(datablockheader_formats),
##                                       *tuple(fieldvalues(datablockheader,
##                                                          block_header)))
##             f.write(bheader_str)
##             f.write(sl.byteswapped().tostring())

##     f.close()
    return real_ts    

## datafileheader = odict((
##     ('nblocks', 'l'), # num data blocks
##     ('ntraces', 'l'), # num traces per block
##     ('num_els', 'l'), # num simple elements per trace (2x # complex pts)
##     ('ebytes', 'l'),  # 2 for 16-bit int, 4 for all others
##     ('tbytes', 'l'),  # set to num_els*ebytes
##     ('bbytes', 'l'),  # set to ntraces*tbytes + nbheaders*sizeof(datafileheader)
##     ('vers_id', 'h'), # ?
##     ('status', 'h'),  # try 0
##     ('nbheaders', 'l') # num block headers per data block (should be 1)
## ))

## datafileheader_formats = datafileheader.values()

## datablockheader = odict((
##     ('scale', 'h'),
##     ('status', 'h'),
##     ('index', 'h'), # set this
##     ('mode', 'h'),
##     ('ctcount', 'l'),
##     ('lpval', 'f'),
##     ('rpval', 'f'),
##     ('lvl', 'f'), # set to 0
##     ('tlt', 'f')  # set to 0
## ))
