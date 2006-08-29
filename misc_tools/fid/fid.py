"""
This module implements writeFid(), which writes a 3- or 4-D array in various
FID formats
"""

import struct, os, sys
from Numeric import empty, Complex, Int16, floor, Float, array, arange, zeros,\
     exp, reshape, Complex32, Float32, Int32, Int8, swapaxes, NewAxis
from odict import odict

datatype2numbytes = {
    Int8: 1,
    Int16: 2,
    Int32: 4,
    Float32: 4,
    Complex32: 4,
    Complex: 8,
    Float: 8,
}

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

def fieldvalues(struct_field, indications):
    defaults = {'i': 0, 'h': 0, 'f': 0., \
                'c': '\0', 's': '', 'B': 0, 'l': 0}
    return [indications.get(name) or defaults[struct_field[name]] \
            for name in struct_field.keys()]
    

def writeFid(m, name, newtype=Int16, biases=None, format='uncompressed'):
    # write a fid file in any format, will destroy m
    (nslice, ny, nx) = m.shape[-3:]
    oshape = m.shape
    T = len(m.shape) > 3 and m.shape[0] or 1
    
    # convert into (blocks x traces) dimensions
    if format=='compressed':
        nblocks = T
        ntraces = nslice*ny
        m = reshape(m, (T,nslice*ny*nx))
    elif format=='ncsnn':
        nblocks = T*ny
        ntraces = nslice
        m = reshape(swapaxes(swapaxes(m,1,2),0,1), (T*ny,nslice*nx))
    elif format=='uncompressed':
        nblocks = T*nslice
        ntraces = ny
        m = reshape(m, (T*nslice,ny*nx))
    else raise ValueError("%s: not an available format"%format)

    # put biases back into raw data, allows us to truncate
    if biases is None: biases = zeros((nblocks,),Complex32)

    if m.typecode() in ('D','F'):
        m = (m + biases[:,NewAxis])
        (nblocks,tracedim) = m.shape
        tracedim = tracedim*2
        nx = nx*2
        m_fix = empty((nblocks,tracedim),newtype)
        m_fix[:,::2] = m.real.astype(newtype)
        m_fix[:,1::2] = m.imag.astype(newtype)
        m = m_fix
    else:
        m = (m + biases[:,NewAxis]).astype(newtype)
    
    ebytes = datatype2numbytes[newtype]
    blockheadersize = struct.calcsize('='+' '.join(datablockheader_formats))
    
    file_header = {
        'nblocks': nblocks,
        'ntraces': ntraces,
        'num_els': nx,
        'ebytes': ebytes,
        'tbytes': ebytes*(nx),
        'bbytes': ntraces*(ebytes*nx) + blockheadersize,
        'nbheaders': 1
    }

    # fid's are expected to be big-endian
    fheader_str = struct.pack('>'+' '.join(datafileheader_formats),
                              *tuple(fieldvalues(datafileheader, file_header)))
    
    flip_my_data = sys.byteorder=='little'
    block_header = {
        'scale': 1,
        'index': 0,
        'lvl': 0.0,
        'tlt': 0.0,
    }

    f = open(name+'.fid', 'w')
    f.write(fheader_str)
    i_cnt = 1
    for block in m:
        #for sl in vol:
        block_header['index'] = i_cnt
        block_header['lvl'] = biases.astype(Complex32)[i_cnt-1].real
        block_header['tlt'] = biases.astype(Complex32)[i_cnt-1].imag
        i_cnt += 1
        bheader_str = struct.pack('>'+' '.join(datablockheader_formats),
                                  *tuple(fieldvalues(datablockheader,
                                                     block_header)))
        f.write(bheader_str)
        if flip_my_data: f.write(block.byteswapped().tostring())
        else: f.write(block.tostring())

    f.close()
    return


def backrot(sig, n):
    rsig = empty(sig.shape, sig.typecode())
    rsig[:len(sig)-n] = sig[n:]
    rsig[len(sig)-n:] = sig[:n]
    return rsig

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
    return real_ts    



