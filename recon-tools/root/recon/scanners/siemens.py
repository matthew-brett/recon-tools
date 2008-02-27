from recon.scanners import _HeaderBase, _BitMaskedWord, ScannerImage, ReconImage
import os, struct, sys
import numpy as N

def sinc_kernel(Tr, T0, Tf, M1, N1):
    (Tr, T0, Tf) = map(float, (Tr, T0, Tf))
    As = Tf + Tr - (T0**2)/Tr
    nr = int( (N1-1)*(Tr**2 - T0**2)/(2*Tr*As) )
    nf = int( (N1-1)*(2*Tf*Tr + Tr**2 - T0**2)/(2*Tr*As) )
    r1 = N.arange(0, nr)
    r2 = N.arange(nr, nf)
    r3 = N.arange(nf, int(N1))
    t = N.zeros(int(N1))
    t[r1] = N.power( T0**2 + 2.0*r1*Tr*As/(N1-1), 0.5)
    t[r2] = r2*As/(N1-1) + Tr/2.0 + T0**2/(2.0*Tr)
    t[r3] = (Tf + 2*Tr) - N.power(2.0*Tr*(As + T0**2/(2.0*Tr) - r3*As/(N1-1)), 0.5)

    dt = (2.0*Tr + Tf - 2.0*T0)/float(M1-1)
    s_idx = N.outer( (t-T0)/dt, N.ones(M1) ) - N.arange(M1)
    s_idx = N.outer(N.ones(M1), (t-T0)/dt) - N.arange(M1)[:,None]
    return N.sinc(s_idx).astype(N.complex64)

def load_agems(dat, agems, kshape, snc_xform):
    # kspshape is something like (12,2,nsl,npe,M1) .. but actually
    # there are two echos, not two volumes 
    # But, we still want to stack the data up in order of channels
    dat_dtype = dat.mmap.dtype
    dtype = dat_dtype['data'].base
    nchan = kshape[0]
    nvol = kshape[1]
    nsl = kshape[2]
    nline = kshape[3]
    M1 = snc_xform is None and kshape[-1] or snc_xform.shape[0]
    N1 = snc_xform is None and kshape[-1] or snc_xform.shape[1]
    cshape = (N.product(kshape[1:-1]), N1)
    chan = N.empty(cshape, dtype)
    for c in range(nchan):
        if snc_xform is not None:
            chan[:] = N.dot(dat[c::nchan]['data'], snc_xform)
        else:
            chan[:] = dat[c::nchan]['data']
        # now I've got one channel of data ordered like:
        # (n_pe, n_vol, n_slice)
        chan.shape = (nline, nvol, nsl, N1)
        agems[c] = chan.transpose((1, 2, 0, 3))
        chan.shape = cshape
    del chan
    

def load_epi(dat, epi, ref, kshape, snc_xform):
    dat_dtype = dat.mmap.dtype
    dtype = dat_dtype['data'].base
    nchan = kshape[0]
    nline = kshape[-2]
    N1 = snc_xform is None and kshape[-1] or snc_xform.shape[1]
    M1 = snc_xform is None and kshape[-1] or snc_xform.shape[0]
    dshape = [d for d in kshape]
    rshape = [d for d in kshape]
    # assuming EPI has 3 ref lines!!
    dshape[-2:] = [kshape[-2]-3, int(N1)]
    rshape[-2:] = [3, int(N1)]

    # get one slice of data, find out what's ref and what's reversed, etc
    blks = dat[N.arange(nline)*nchan]['hdr']
    epi_lines = []
    ref_lines = []
    rev_lines = []
    for n,b in enumerate(blks):
        mdh = MDH(b)
        if mdh.info0.REFLECT:
            rev_lines.append(n)
        if mdh.info0.PHSCOR:
            ref_lines.append(n)
        else:
            epi_lines.append(n)
    cshape = (dat.nblocks/nchan, N1)
    chan = N.empty(cshape, dtype)
##     epi = MemmapArray(tuple(dshape), dtype)
##     ref = MemmapArray(tuple(rshape), dtype)
##     epi = N.empty(tuple(dshape), dtype)
##     ref = N.empty(tuple(rshape), dtype)
    for c in range(nchan):
        print "grabbing chan", c
        if snc_xform is not None:
            chan[:] = N.dot(dat[c::nchan]['data'], snc_xform)
            #chan[:] = N.dot(dat[cshape[0]*c:cshape[0]*(c+1)]['data'], snc_xform)
        else:
            chan[:] = dat[c::nchan]['data']
            #chan[:] = dat[cshape[0]*c : cshape[0]*(c+1)]['data']
            print chan.sum()
        chan.shape = kshape[1:-1] + (N1,)
        chan[:,:,rev_lines,:] = chan[:,:,rev_lines,::-1]
        epi[c] = chan[:,:,epi_lines,:]
        ref[c] = chan[:,:,ref_lines,:]
        chan.shape = cshape
    del chan
    #return epi, ref
    #return None, None


class SiemensImage(ScannerImage):
    nr = 0
    nd = 0

    def __init__(self, filestem, vrange=None, target_dtype=None,
                 ksp_shape=None, N1=64, scan='epi', **kwargs):

        if ksp_shape is None:
            raise AttributeError('ksp_shape must be defined')
        self._deal_with_kwargs(kwargs)
        self.n_chan, self.n_vol, self.n_slice, self.n_pe = ksp_shape[:-1]
        self.n_fe = N1
        self.pe0 = -self.n_pe/2
        self.tr = 1
        self.sampstyle='linear'
        self.nseg = 1
        self.path = filestem+'.dat'
        
        # get data (and possibly ref)
        dat = MemmapDatFile(filestem+'.dat', ksp_shape)
        skern = None if (N1 == ksp_shape[-1]) else \
                sinc_kernel(self.Tr, self.T0, self.Tf, ksp_shape[-1], N1)
        
        if scan == 'epi':
            self.dscratch = os.path.abspath('dscratch%d'%SiemensImage.nd)
            SiemensImage.nd += 1
            self.rscratch = os.path.abspath('rscratch%d'%SiemensImage.nr)
            SiemensImage.nr += 1
            os.system('touch %s'%self.dscratch)
            os.system('touch %s'%self.rscratch)
            dshape = [n for n in ksp_shape]
            dshape[-2:] = [dshape[-2] - 3, int(N1)]
            rshape = [n for n in ksp_shape]
            rshape[-2:] = [3, int(N1)]
            self.cdata = N.memmap(self.dscratch, shape=tuple(dshape),
                                  dtype=N.complex64, mode='r+')
            self.cref_data = N.memmap(self.rscratch, shape=tuple(rshape),
                                      dtype=N.complex64, mode='r+')
            load_epi(dat, self.cdata, self.cref_data, ksp_shape, skern)
            self.data = self.cdata[0]
            self.ref_data = self.cref_data[0]
        elif scan == 'agems':
            self.dscratch = os.path.abspath('dscratch%d'%SiemensImage.nd)
            SiemensImage.nd += 1
            os.system('touch %s'%self.dscratch)
            dshape = [n for n in ksp_shape]
            dshape[-1] = int(N1)
            self.cdata = N.memmap(self.dscratch, shape=tuple(dshape),
                                  dtype=N.complex64, mode='r+')
            load_agems(dat, self.cdata, ksp_shape, skern)
            
        del dat.mmap
        del dat

        a = N.array( range(1, self.n_slice, 2) + range(0, self.n_slice, 2) )
        b = N.array( range(1, self.n_slice, 2) + range(0, self.n_slice, 2) )
        b.sort()
        self.acq_order = N.array([(p==b).nonzero()[0][0] for p in a])

        # this hack avoids MPL bugs for now!!
        self.data = N.empty(self.cdata.shape[1:], self.cdata.dtype)
        if hasattr(self, 'cref_data'):
            self.ref_data = N.empty(self.cref_data.shape[1:],
                                    self.cref_data.dtype)
        self.load_chan(0)
        self.combined = False

        ScannerImage.__init__(self)
        
    def _deal_with_kwargs(self, kwargs):
        atts = ScannerImage.necessary_params
        for att in atts:
            a = kwargs.has_key(att) and kwargs.pop(att) or 0
            setattr(self, att, a)
        for k,v in kwargs.items():
            setattr(self, k, v)

    def load_chan(self, cnum):
        self.cdata.flush()
        #self.setData(self.cdata[cnum])
        self.data[:] = self.cdata[cnum]
        if hasattr(self, 'cref_data'):
            self.cref_data.flush()
            self.ref_data[:] = self.cref_data[cnum]

    def runOperations(self, opchain, logger=None):
        """
        Run the opchain through for each channel of data
        """
        for c in range(self.n_chan):
            self.load_chan(c)
            ReconImage.runOperations(self, opchain, logger=logger)
            #super(SiemensImage, self).runOperations(opchain)
        self.load_chan(0)
        
            
    def run_op(self, op_obj):
        for c in range(self.n_chan):
            self.load_chan(c)
            op_obj.run(self)
        self.load_chan(0)

    def combine_channels(self):
        d = N.zeros(self.cdata.shape[-4:], self.data.dtype.char.lower())
        for chan in self.cdata:
            d += N.power(chan.real, 2.0) + N.power(chan.imag, 2.0)
        self.setData(N.sqrt(d))
        self.combined = True

##############################################################################
###########################   FILE LEVEL CLASSES   ###########################
##############################################################################

class MDH (_HeaderBase):
    """
    MDH blocks contain header and data portions. The header contains
    information about the indexing of the data, the data type, and some
    metadata. The data constitutes a single line or row of k-space.

    There are five indexing counters in the MDH header. The order in which
    these number advance will change with scan type. This list is in their
    logical ordering:
    1) echo number -- groups by echo train (eg, two partitions for AGEMS)
    2) channel number -- groups by receiving channel
    3) volume number (repitition) -- groups by volume
    4) slice number  -- groups by slice (in acquisition order)
    5) row number -- groups by k-space row
    """
    HEADER_FMT = "<IiIIIIIHHHHHHHHHHHHHHHHHHHHfIHH8s8sfffffffHH"
    HEADER_FIELD_NAMES = (
        "flags_and_dma",
        "measUID",
        "scan_counter",
        "time_stamp",
        "PMU_time_stamp",
        "eval_info0",
        "eval_info1",
        "samps_in_scan",
        "used_chans",
        "line",
        "acq",
        "slice",
        "partition",
        "echo", 
        "phase",
        "rep",
        "set",
        "seg",
        "idA",
        "idB",
        "idC",
        "idD",
        "idE",
        "prezero",
        "postzero",
        "ksp_center",
        "coil_select",
        "readout_offcenter",
        "time_since_RF",
        "ksp_center_line",
        "ksp_center_part",
        "unused3",
        "unused4",
        "sag_pos",
        "cor_pos",
        "tran_pos",
        "quatW",
        "quatI",
        "quatJ",
        "quatK",
        "chan", 
        "ptab_pos",
    )

    class _EvalInfo0 (_BitMaskedWord):
        RTFEEDBACK = property( lambda self: self._word & 0x1 != 0 )
        HPFEEDBACK = property( lambda self: self._word & 0x2 != 0 )
        ONLINE = property( lambda self: self._word & 0x4 != 0 )
        OFFLINE = property( lambda self: self._word & 0x8 != 0 )
        SYNCDATA = property( lambda self: self._word & 0x10 != 0 )
        LASTSCANINCONCAT = property( lambda self: self._word & 0x100 != 0 )
        RAWDATACORRECTION = property( lambda self: self._word & 0x200 != 0 )
        LASTSCANINMEAS = property( lambda self: self._word &0x400 != 0 )
        SCANSCALEFACTOR = property( lambda self: self._word & 0x1000 != 0 )
        SCNDHADAMARDPULSE = property( lambda self: self._word & 0x2000 != 0 )
        REFPHASETABSCAN = property( lambda self: self._word & 0x4000 != 0 )
        PHASETABSCAN = property( lambda self: self._word & 0x8000 != 0 )
        D3FFT = property( lambda self: self._word & 0x10000 != 0 )
        SIGNREV = property( lambda self: self._word & (0x1 << 17) != 0 )
        PHASEFFT = property( lambda self: self._word & (0x1 << 18) != 0 )
        SWAPPED = property( lambda self: self._word & (0x1 << 19) != 0 )
        POSTSHAREDLINE = property( lambda self: self._word & (0x1 << 20) != 0 )
        PHSCOR = property( lambda self: self._word & (0x1 << 21) != 0 )
        PATREF = property( lambda self: self._word & (0x1 << 22) != 0 )
        PATREF_IMG = property( lambda self: self._word & (0x1 << 23) != 0 )
        REFLECT = property( lambda self: self._word & (0x1 << 24) != 0 )
        NOISEADJ = property( lambda self: self._word & (0x1 << 25) != 0 )
        SHARENOW = property( lambda self: self._word & (0x1 << 26) != 0 )
        LASTMEASUREDLINE = property( lambda self: self._word & (0x1 << 27) != 0)
        FIRSTSCANINSLICE = property( lambda self: self._word & (0x1 << 28) != 0)
        LASTSCANINSLICE = property( lambda self: self._word & (0x1 << 29) != 0 )
        TREFFBEGIN = property( lambda self: self._word & (0x1 << 30) != 0 )
        TREFFEND = property( lambda self: self._word & (0x1 << 31) != 0 )
    class _EvalInfo1 (_BitMaskedWord):
        FIRSTSCANINBLADE = property( lambda self: self._word & (0x1 << 8) != 0 )
        LASTSCANINBLADE = property( lambda self: self._word & (0x1 << 9) != 0 )
        LASTBLADEINTR = property( lambda self: self._word & (0x1 << 10) != 0 )
        RETROLASTPHASE = property( lambda self: self._word & (0x1 << 13) != 0 )
        RETROENDOFMEAS = property( lambda self: self._word & (0x1 << 14) != 0 )
        RETROREPEATHEART = property( lambda self: self._word & (0x1<<15) != 0 )
        RETROREPEATLASTHEART = property( lambda self: self._word & (0x1<<16)!=0)
        RETROABORTSCAN = property( lambda self: self._word & (0x1 << 17) != 0 )
        RETROLASTHEARTBEAT = property( lambda self: self._word & (0x1<<18)!=0 )
        RETRODUMMYSCAN = property( lambda self: self._word & (0x1 << 19) != 0 )
        RETROARRDETDISABLED = property( lambda self: self._word & (0x1<<20)!=0)

    def __init__(self, hdr_string):
        _HeaderBase.__init__(self, hdr_string)
        self.info0 = MDH._EvalInfo0(self.eval_info0)
        self.info1 = MDH._EvalInfo1(self.eval_info1)

class MemmapDatFile:                      
    def __init__(self, fname, shape):
        #file.__init__(self, fname)
        fp = open(fname)
        self.main_header_size = struct.unpack("<i", fp.read(4))[0]
        fp.seek(self.main_header_size)
        # need to grab a header and find out the samps_per_scan in order
        # to construct the dtype((N.float32, samps_per_scan*2))

        hdr_dtype = N.dtype('S128')

        
        first_mdh = MDH(fp.read(128))
        nsamps = first_mdh.samps_in_scan
        blk_dtype = N.dtype((N.complex64, nsamps))

        block_size = 128 + nsamps*2*4
        fp.close()
        #stat = os.stat(fname)
        #self.nblocks = int((stat.st_size - self.main_header_size)/block_size)
        #self.nblocks -= 2
        self.nblocks = N.product(shape[:-1])
        dat_type = N.dtype({'names':['hdr', 'data'], 'formats':[hdr_dtype, blk_dtype]})

        self.mmap = N.memmap(fname, dtype=dat_type,
                             offset=self.main_header_size,
                             shape=(self.nblocks,),
                             mode='r')


    def __iter__(self):
        for bnum in xrange(self.nblocks):
            yield self.mmap[b]

    def __getitem__(self, slicer):
        return self.mmap[slicer]

    def __setitem__(self, slicer, item):
        raise IOError('this is a read-only memmap')

import os
class MemmapArray(N.memmap):

    ntmps = 0
    
    def __new__(subtype, shape, dtype):
        fname = 'scratch%d'%MemmapArray.ntmps
        if not os.path.exists(fname):
            os.system('touch %s'%fname)
        MemmapArray.ntmps += 1
##         data = N.memmap.__new__(subtype, fname,
##                                 dtype=dtype, shape=shape, mode='r+')
        data = N.memmap(fname, dtype=dtype, shape=shape, mode='r+')
        data = data.view(subtype)
        data.f = os.path.abspath(fname)
        return data

    def __array_finalize__(self, obj):
        N.memmap.__array_finalize__(self, obj)
##         if hasattr(obj, 'f'):
##             self.f = getattr(obj, 'f')
##         if hasattr(obj, '_mmap'):
##             self._mmap = obj._mmap

##     def __del__(self):
##         if type(self.base) is N.memmap:
##             try:
##                 self.base.__del__()
##                 os.unlink(self.f)
##             except ValueError:
##                 print "almost unlinked file in use"
##         else:
##             print "almost unlinked file in use"
                
