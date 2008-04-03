from recon.scanners import _HeaderBase, _BitMaskedWord, ScannerImage, \
     ReconImage, CachedReadOnlyProperty
from recon.util import fft2d, ifft2d, checkerline, MemmapArray
import os, struct, sys, re
import numpy as N

def sinc_kernel(Tr, T0, Tf, dt, M1, N1):
    (Tr, T0, Tf) = map(float, (Tr, T0, Tf))
    #As = Tf + Tr - (T0**2)/Tr
    Bs = N1 / (Tf + Tr - (T0**2)/Tr)
    nr = int( Bs*(Tr**2 - T0**2)/(2*Tr) )
    nf = int( Bs*(2*Tf*Tr + Tr**2 - T0**2)/(2*Tr) )
    r1 = N.arange(0, nr+1)
    r2 = N.arange(nr+1, nf+1)
    r3 = N.arange(nf+1, int(N1))
##     r1 = N.arange(0, nr)
##     r2 = N.arange(nr, nf)
##     r3 = N.arange(nf, int(N1))
    t = N.zeros(int(N1))
    t[r1] = N.power( T0**2 + 2.0*r1*Tr/Bs, 0.5)
    t[r2] = r2/Bs + Tr/2.0 + T0**2/(2.0*Tr)
    t[r3] = (Tf + 2*Tr) - N.power(2.0*Tr*(Tf + Tr - T0**2/(2.0*Tr) - r3/Bs), 0.5)
    print t
    # should use the DWELL TIME from siemens header
    s_idx = N.outer(N.ones(M1), (t-T0)/dt) - N.arange(M1)[:,None]
    return N.sinc(s_idx).astype(N.complex64)


## def sinc_kernel(Tr, T0, Tf, dt, M1, N1):
##     (Tr, T0, Tf) = map(float, (Tr, T0, Tf))
##     As = Tf + Tr - (T0**2)/Tr
##     nr = int( N1*(Tr**2 - T0**2)/(2*Tr*As) )
##     nf = int( N1*(2*Tf*Tr + Tr**2 - T0**2)/(2*Tr*As) )
##     r1 = N.arange(0, nr+1)
##     r2 = N.arange(nr+1, nf+1)
##     r3 = N.arange(nf+1, int(N1))
##     t = N.zeros(int(N1))
##     t[r1] = N.power( T0**2 + 2.0*r1*Tr*As/N1, 0.5)
##     t[r2] = r2*As/N1 + Tr/2.0 + T0**2/(2.0*Tr)
##     t[r3] = (Tf + 2*Tr) - N.power(2.0*Tr*(As + T0**2/(2.0*Tr) - r3*As/N1), 0.5)
##     s_idx = N.outer(N.ones(M1), (t-T0)/dt) - N.arange(M1)[:,None]
##     return N.sinc(s_idx).astype(N.complex64)

def load_agems(dat, agems, kshape, N1):
    # kspshape is something like (12,2,nsl,npe,M1) .. but actually
    # there are two echos, not two volumes 
    # But, we still want to stack the data up in order of channels
    dat_dtype = dat.mmap.dtype
    dtype = dat_dtype['data'].base
    (nchan, nvol, nsl, nline, M1) = kshape
    cshape = (N.product(kshape[1:-1]),M1)
    chan = N.empty(cshape, dtype)
    for c in range(nchan):
        chan[:] = dat[c::nchan]['data']
        # now I've got one channel of data ordered like:
        # (n_pe, n_vol, n_slice)
        chan.shape = (nline, nvol, nsl, M1)
        if (N1 == M1):
            agems[c] = chan.transpose((1, 2, 0, 3))
        else:
            chan = ifft2d(chan.transpose((1, 2, 0, 3)))
            agems[c] = fft2d(chan[...,M1/2-N1/2:M1/2+N1/2])
        chan.shape = cshape
    del chan
    

def load_epi(dat, epi, ref, kshape, snc_xform,
             dwell_time, echo_spc, refoffset,
             vrange=None, scl=1.):
## def load_epi(dat, epi, ref, kshape, snc_xform, vrange=None, scl=1.):
    dat_dtype = dat.mmap.dtype
    dtype = dat_dtype['data'].base
    nchan = kshape[0]
    nvol = kshape[1]
    nsl = kshape[2]
    nline = kshape[3]
    N1 = snc_xform is None and kshape[-1] or snc_xform.shape[1]
    M1 = snc_xform is None and kshape[-1] or snc_xform.shape[0]
    if not vrange:
        vrange = (0, nvol-1)
    blks_per_vol = dat.nblocks/nvol
    start_block = vrange[0]*blks_per_vol
    end_block = (vrange[1]+1)*blks_per_vol
    new_nvol = vrange[1] - vrange[0] + 1
    
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
    cshape = ((end_block-start_block)/nchan, N1)
    chan = N.empty(cshape, dtype)

##     vol_idx = N.indices((nsl,nline-3,M1), dtype=N.float32)
##     vol_phs_corr = N.empty((new_nvol,nsl,nline,M1), N.complex64)
##     phs = refoffset[:,None,None] * \
##           (vol_idx[1]*echo_spc + vol_idx[2]*dwell_time)
##     print dwell_time, echo_spc, refoffset
##     print phs.max()
##     vol_phs_corr[:,:,3:,:] = N.exp(-1.j*refoffset[:,None,None] *\
##                                    (vol_idx[1]*echo_spc + \
##                                     vol_idx[2]*dwell_time))[None,:,:,:]
##     # do not adjust ref lines for now
##     vol_phs_corr[:,:,:3,:] = 1.
##     vol_phs_corr.shape = (new_nvol*nsl*nline,M1)

    #sl_idx = N.indices((nline-3,M1), dtype=N.float32)
    sl_idx = N.indices((nline,M1), dtype=N.float32)
    vol_phs_corr = N.empty((new_nvol,nsl,nline, M1), N.complex64)
    ro_offset = mdh.readout_offcenter/1000. # offcenter in meters
    print ro_offset
    gma = 2*N.pi*42575575. # larmor in rad/tesla
    Gx = 0.013979846426031827 # grad strength in tesla/m
    #Gx = 0.015643178038097248
    #phs = (gma*Gx*ro_offset) * (sl_idx[0]*echo_spc + checkerline(64)[:,None]*sl_idx[1]*dwell_time)
    phs = (gma*Gx*ro_offset) * (sl_idx[0]*echo_spc - checkerline(nline)[:,None]*sl_idx[1]*dwell_time)
    vol_phs_corr[:] = N.exp(1.j*phs)[None,None,:,:]
    #vol_phs_corr[:,:,3:,:] = N.exp(-1.j*phs)[None,None,:,:]
    #vol_phs_corr[:,:,:3,:] = 1.
    vol_phs_corr.shape = (new_nvol*nsl*nline,M1)
    for c in range(nchan):
        print "grabbing chan", c
        # slicing from 1st block of vrange[0] volume plus channel-offset,
        # every nchan blocks until we've gotten all specified volumes
        slicer = slice(start_block + c, end_block, nchan)
        if snc_xform is not None:
##             chan[:] = scl*N.dot(vol_phs_corr*dat[slicer]['data'], snc_xform)
            chan[:] = scl*N.dot(dat[slicer]['data'], snc_xform)
        else:
##             chan[:] = scl*vol_phs_corr*dat[slicer]['data']
            chan[:] = scl*dat[slicer]['data']
        chan.shape = (new_nvol, nsl, nline, N1)
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

##     def __init__(self, filestem, vrange=None, target_dtype=None,
##                  ksp_shape=None, N1=64, scan='epi', **kwargs):

##         if ksp_shape is None:
##             raise AttributeError('ksp_shape must be defined')
##         self._deal_with_kwargs(kwargs)
##         self.n_chan, self.n_vol, self.n_slice, self.n_pe = ksp_shape[:-1]
##         self.n_fe = N1
##         self.pe0 = -self.n_pe/2
##         self.tr = 1
##         self.sampstyle='linear'
##         self.nseg = 1
##         self.path = filestem+'.dat'
        
##         # get data (and possibly ref)
##         dat = MemmapDatFile(filestem+'.dat', ksp_shape)
##         skern = None if (N1 == ksp_shape[-1]) else \
##                 sinc_kernel(self.Tr, self.T0, self.Tf, ksp_shape[-1], N1)
        
##         if scan == 'epi':
##             self.dscratch = os.path.abspath('dscratch%d'%SiemensImage.nd)
##             SiemensImage.nd += 1
##             self.rscratch = os.path.abspath('rscratch%d'%SiemensImage.nr)
##             SiemensImage.nr += 1
##             os.system('touch %s'%self.dscratch)
##             os.system('touch %s'%self.rscratch)
##             dshape = [n for n in ksp_shape]
##             dshape[-2:] = [dshape[-2] - 3, int(N1)]
##             rshape = [n for n in ksp_shape]
##             rshape[-2:] = [3, int(N1)]
##             self.cdata = N.memmap(self.dscratch, shape=tuple(dshape),
##                                   dtype=N.complex64, mode='r+')
##             self.cref_data = N.memmap(self.rscratch, shape=tuple(rshape),
##                                       dtype=N.complex64, mode='r+')
##             load_epi(dat, self.cdata, self.cref_data, ksp_shape, skern)
##             self.data = self.cdata[0]
##             self.ref_data = self.cref_data[0]
##         elif scan == 'agems':
##             self.dscratch = os.path.abspath('dscratch%d'%SiemensImage.nd)
##             SiemensImage.nd += 1
##             os.system('touch %s'%self.dscratch)
##             dshape = [n for n in ksp_shape]
##             dshape[-1] = int(N1)
##             self.cdata = N.memmap(self.dscratch, shape=tuple(dshape),
##                                   dtype=N.complex64, mode='r+')
##             load_agems(dat, self.cdata, ksp_shape, skern)
            
##         del dat.mmap
##         del dat

##         self.acq_order = N.array( range(1, self.n_slice, 2) + range(0, self.n_slice, 2) )

##         self.use_membuffer()
##         #self.load_chan(0)
##         self.combined = False

##         ScannerImage.__init__(self)

    def __init__(self, filestem, vrange=None, target_dtype=None, **kwargs):
        self.path = filestem+'.dat'
        self.__dict__.update(parse_siemens_hdr(self.path))
        if kwargs.has_key('N1'):
            self.N1 = kwargs['N1']
        self.n_fe = self.N1

        # fake a couple things still
        if not hasattr(self, 'n_refs'):
            self.n_refs = 0
        else:
            self.n_refs += 1
        if self.isagems:
            self.n_vol *= self.n_echo
            self.T_pe = 0

        ksp_shape = (self.n_chan, self.n_vol, self.n_slice,
                     self.n_pe+self.n_refs, self.M1)

        dat = MemmapDatFile(self.path, ksp_shape)
        if self.isepi:
            if vrange:
                self.n_vol = vrange[1] - vrange[0] + 1
            
##             refoffset = N.linspace(0, 2.75*2*N.pi*30.637, self.n_slice,
##                                    endpoint=False)
##             refoffset = refoffset[self.acq_order]
            skern = None if (self.N1 == self.M1) else \
                    sinc_kernel(self.T_ramp, self.T0, self.T_flat, self.delT,
                                self.M1, self.N1)
##             skern = sinc_kernel(self.T_ramp, self.T0, self.T_flat, self.delT,
##                                 self.M1, self.N1)
            self.dscratch = os.path.abspath('dscratch%d'%SiemensImage.nd)
            SiemensImage.nd += 1
            self.rscratch = os.path.abspath('rscratch%d'%SiemensImage.nr)
            SiemensImage.nr += 1
            os.system('touch %s'%self.dscratch)
            os.system('touch %s'%self.rscratch)
            dshape = (self.n_chan, self.n_vol, self.n_slice,
                      self.n_pe, self.n_fe)
            rshape = (self.n_chan, self.n_vol, self.n_slice,
                      self.n_refs, self.n_fe)
            self.cdata = MemmapArray(dshape, N.complex64)
            self.cref_data = MemmapArray(rshape, N.complex64)
##             self.cdata = N.memmap(self.dscratch, shape=dshape,
##                                   dtype=N.complex64, mode='r+')
##             self.cref_data = N.memmap(self.rscratch, shape=rshape,
##                                       dtype=N.complex64, mode='r+')
            load_epi(dat, self.cdata, self.cref_data, ksp_shape, skern,
                     self.delT/1e6, (self.T_flat+2*self.T_ramp)/1e6, None,
                     vrange=vrange, scl=64*128.)
##             load_epi(dat, self.cdata, self.cref_data, ksp_shape, skern,
##                      vrange=vrange, scl=64*128.)
        elif self.isagems:
            self.dscratch = os.path.abspath('dscratch%d'%SiemensImage.nd)
            SiemensImage.nd += 1
            os.system('touch %s'%self.dscratch)
            dshape = [d for d in ksp_shape]
            dshape[-1] = self.N1
            self.cdata = MemmapArray(tuple(dshape), N.complex64)
##             self.cdata = N.memmap(self.dscratch, shape=tuple(dshape),
##                                   dtype=N.complex64, mode='r+')
            load_agems(dat, self.cdata, ksp_shape, self.N1)

        del dat.mmap
        del dat
        
        if self.acq_order.shape[0] != self.n_slice:
            self.acq_order = N.array( range(1, self.n_slice, 2) +
                                      range(0, self.n_slice, 2) )

        self.use_membuffer()
        self.combined = False

        ScannerImage.__init__(self)
    
    def _deal_with_kwargs(self, kwargs):
        atts = ScannerImage.necessary_params
        for att in atts:
            a = kwargs.has_key(att) and kwargs.pop(att) or 0
            setattr(self, att, a)
        for k,v in kwargs.items():
            setattr(self, k, v)

    @CachedReadOnlyProperty
    def pe0(self):
        return -self.n_pe/2
    @CachedReadOnlyProperty
    def sampstyle(self):
        return 'linear'
    @CachedReadOnlyProperty
    def echo_time(self):
        return self.te[0]/1e6
    @CachedReadOnlyProperty
    def asym_times(self):
        if self.isagems:
            return N.array(self.te)/1e6
        else:
            return []
    @CachedReadOnlyProperty
    def theta(self):
        return 0.
    @CachedReadOnlyProperty
    def psi(self):
        return 0.
    @CachedReadOnlyProperty
    def phi(self):
        return 0.
    @CachedReadOnlyProperty
    def petable(self):
        return N.arange(self.n_pe)
    @CachedReadOnlyProperty
    def dFE(self):
        return self.fov_x/self.n_fe
    @CachedReadOnlyProperty
    def dPE(self):
        return self.fov_y/self.n_pe
    @CachedReadOnlyProperty
    def delT(self):
        # return this in micro-secs
        return self.dwell_time/1e3
    
    def use_membuffer(self, chan=0):
        # this hack avoids MPL bugs for now!!
        self.data = N.empty(self.cdata.shape[1:], self.cdata.dtype)
        self.data[:] = self.cdata[chan]
        self.setData(self.data)
        if hasattr(self, 'cref_data'):
            self.ref_data = N.empty(self.cref_data.shape[1:],
                                    self.cref_data.dtype)
            self.ref_data[:] = self.cref_data[chan]
        self.combined = False

    def load_chan(self, cnum):
        self.cdata.flush()
        self.setData(self.cdata[cnum])
        #self.data = self.cdata[cnum]
        if hasattr(self, 'cref_data'):
            self.cref_data.flush()
            self.ref_data = self.cref_data[cnum]
        self.combined = False

    def runOperations(self, opchain, logger=None):
        """
        Run the opchain through for each channel of data
        """
        for c in range(self.n_chan):
            self.load_chan(c)
            ReconImage.runOperations(self, opchain, logger=logger)
            #super(SiemensImage, self).runOperations(opchain)
        #self.load_chan(0)
        self.use_membuffer()
        
            
    def run_op(self, op_obj):
        for c in range(self.n_chan):
            self.load_chan(c)
            op_obj.run(self)
        #self.load_chan(0)
        self.use_membuffer()

    def combine_channels(self):
        dtype = self.data.dtype
        if type(self.data) is not N.memmap:
            del self.data
        d = N.zeros(self.cdata.shape[-4:], dtype.char.lower())
        for chan in self.cdata:
            d += N.power(chan.real, 2.0) + N.power(chan.imag, 2.0)
        self.setData(N.sqrt(d))
        self.combined = True

    def epi_trajectory(self, pe0=None):
        """
        This method is helpful for computing T[n2] in the artifact
        correction algorithms.
        Returns:
        a) the ksp trajectory (-1 or +1) of each row (alpha)
        b) the index of each row in acq. order in its segment (beta)
        c) the index of each row, based on the number of rows in a segment,
           which is [-M2/2, M2/2)
        d) the ksp trajectory (-1 or +1) of each reference line
        """
        M = self.shape[-2]
        if not pe0:
            pe0 = self.pe0
        n2 = N.arange(M) + pe0
        a = N.empty(M, N.int32)
        for n in range(self.nseg):
            a[n:M:2*self.nseg] = 1
            a[n+self.nseg:M:2*self.nseg] = -1
        b = ((N.arange(float(M))+pe0)/float(self.nseg)).astype(N.int32)
        aref = N.power(-1, N.arange(self.n_refs)+1)
        return (a, b, n2, aref)

    @CachedReadOnlyProperty
    def n_ramp(self):
        (Tr, T0, Tf) = map(float, (self.T_ramp, self.T0, self.T_flat))
        N1 = self.shape[-1]
        As = Tf + Tr - (T0**2)/Tr
        return int( N1*(Tr**2 - T0**2)/(2*Tr*As) )

    @CachedReadOnlyProperty
    def n_flat(self):
        (Tr, T0, Tf) = map(float, (self.T_ramp, self.T0, self.T_flat))
        N1 = self.shape[-1]
        As = Tf + Tr - (T0**2)/Tr
        return int( N1*(2*Tf*Tr + Tr**2 - T0**2)/(2*Tr*As) )

##     @CachedReadOnlyProperty
##     def n_ramp(self):
##         (Tr, T0, Tf) = map(float, (self.T_ramp, self.T0, self.T_flat))
##         N1 = self.shape[-1]
##         #As = Tf + Tr - (T0**2)/Tr
##         Bs = 1/(Tf + Tr - T0**2/Tr) * N1
##         #Bs = 1/(2*Tr + Tf - 2*T0) * N1 # cycles / microsec        
##         return int( Bs*(Tr**2 - T0**2)/(2*Tr) )

##     @CachedReadOnlyProperty
##     def n_flat(self):
##         (Tr, T0, Tf) = map(float, (self.T_ramp, self.T0, self.T_flat))
##         N1 = self.shape[-1]
##         #As = Tf + Tr - (T0**2)/Tr
##         Bs = 1/(Tf + Tr - T0**2/Tr) * N1
##         #Bs = 1/(2*Tr + Tf - 2*T0) * N1 # cycles / microsec        
##         return int( Bs*(2*Tf*Tr + Tr**2 - T0**2)/(2*Tr) )
    
##     def t_n1(self):
##         (Tr, T0, Tf) = map(float, (self.T_ramp, self.T0, self.T_flat))
##         N1 = self.shape[-1]
##         As = Tf + Tr - (T0**2)/Tr
##         nr = self.n_ramp
##         nf = self.n_flat
##         print nr, nf
##         r1 = N.arange(0, nr+1)
##         r2 = N.arange(nr+1, nf+1)
##         r3 = N.arange(nf+1, int(N1))
##         t = N.zeros(int(N1))
##         t[r1] = N.power( T0**2 + 2.0*r1*Tr*As/N1, 0.5)
##         t[r2] = r2*As/N1 + Tr/2.0 + T0**2/(2.0*Tr)
##         t[r3] = (Tf + 2*Tr) - N.power(2.0*Tr*(As + T0**2/(2.0*Tr) - r3*As/N1), 0.5)
##         return t

    def t_n1(self):
        (Tr, T0, Tf) = map(float, (self.T_ramp, self.T0, self.T_flat))
        N1 = self.shape[-1]
        #As = Tf + Tr - (T0**2)/Tr
        Bs = 1/(Tf + Tr - T0**2/Tr) * N1
        #Bs = 1/(2*Tr + Tf - 2*T0) * N1 # cycles / microsec
        nr = self.n_ramp
        nf = self.n_flat
        print nr, nf
        r1 = N.arange(0, nr+1)
        r2 = N.arange(nr+1, nf+1)
        r3 = N.arange(nf+1, int(N1))
        t = N.zeros(int(N1))
        t[r1] = N.power( T0**2 + 2.0*r1*Tr/Bs, 0.5 )
        t[r2] = r2/Bs + Tr/2.0 + T0**2/(2.0*Tr)
        t[r3] = (Tf + 2*Tr) - N.power(2.0*Tr*(Tf + Tr - T0**2/(2*Tr) - r3/Bs), 0.5)
        return t
    
    def seg_slicing(eslf, n):
        """
        This method returns a list of indices corresponding to segment n,
        to be used in separating segments from recombined kspace
        """
        pass

    def __del__(self):
        del self.cdata
        os.unlink(self.dscratch)
        if hasattr(self, 'cref_data'):
            del self.cref_data
            os.unlink(self.rscratch)

##############################################################################
###########################   FILE LEVEL CLASSES   ###########################
##############################################################################

def parse_siemens_hdr(fname):
    fp = open(fname, 'r')
    hdr_len = struct.unpack("<i", fp.read(4))[0]
    hdr_str = fp.read(hdr_len)
    fp.close()

    hdr_dict = {}
    long_re = re.compile('[-0-9]+')
    float_re = re.compile('[-0-9]+.[-0-9]+')
    

    def find_long(substr, offset):
        match = long_re.search(substr, offset)
        s = match.group().split('-')
        if len(s) > 1:
            return -1 * int(s[1])
        else:
            return int(s[0])
    
    def find_dec(substr, offset):
        match = float_re.search(substr, offset)
        s = match.group().split('-')
        if len(s) > 1:
            return -1.0 * float(s[1])
        else:
            return float(s[0])
        
    search_strs = {
        '<ParamString."SequenceString">':
        {'<ParamLong."BaseResolution">': ('N1', find_long),
         '<ParamLong."PhaseEncodingLines">': ('n_pe', find_long),
         '<ParamLong."NSlc">': ('n_slice', find_long),
         '<ParamLong."NEco">': ('n_echo', find_long),
         '<ParamLong."NSeg">': ('nseg', find_long),
         '<ParamDouble."ReadFoV">': ('fov_x', find_dec),
         '<ParamDouble."PhaseFoV">': ('fov_y', find_dec),
         '<ParamDouble."TR">': ('tr', find_dec),
         },
        
        '<ParamFunctor."ROFilterFunctor">':
        {'<ParamLong."NColMeas">': ('M1', find_long),
         '<ParamLong."NChaMeas">': ('n_chan', find_long),
         },

        '### ASCCONV BEGIN ###':
        {'sRXSPEC.alDwellTime': ('dwell_time', find_dec),
         'sAdjData.sAdjVolume.sPosition.dSag': ('dSag', find_dec),
         'sAdjData.sAdjVolume.sPosition.dCor': ('dCor', find_dec),
         'sAdjData.sAdjVolume.sPosition.dTra': ('dTra', find_dec),
         'sAdjData.sAdjVolume.sNormal.dSag': ('sagNorm', find_dec),
         'sAdjData.sAdjVolume.sNormal.dCor': ('corNorm', find_dec),
         'sAdjData.sAdjVolume.sNormal.dTra': ('traNorm', find_dec),
         'sAdjData.sAdjVolume.dInPlaneRot': ('pln_rot', find_dec),
         }
    }
    epi_search_strs = {
        '<ParamFunctor."adjroftregrid">':
        {'<ParamLong."RampupTime">': ('T_ramp', find_long),
         '<ParamLong."FlattopTime">': ('T_flat', find_long),
         '<ParamLong."DelaySamplesTime">': ('T0', find_long),
         '<ParamDouble."ADCDuration">': ('ro_period', find_dec),
         },
        
        '<ParamFunctor."EPIPhaseCorrPE">':
        {'<ParamLong."EchoSpacing_us">': ('T_pe', find_long),
         '<ParamLong."NSeg">': ('n_refs', find_long),
         }
    }
        
        
    
    f = re.compile('<ParamString."SequenceString">')
    m = f.search(hdr_str)
    fseq = re.compile('\w+')
    mseq = fseq.search(hdr_str, m.end())
    hdr_dict['pslabel'] = mseq.group()
    isepi = (mseq.group().find('epfid') >= 0)
    isagems = mseq.group() == 'fm2d2r'
    hdr_dict['isepi'] = isepi
    hdr_dict['isagems'] = isagems
    if isepi:
        search_strs.update(epi_search_strs)
    last_pos = 0
    for item, substrs in search_strs.items():
        f = re.compile(item)
        m = f.search(hdr_str)
        for hdr_att, parse_info in substrs.items():
            att_name, parse_func = parse_info
            fsub = re.compile(hdr_att)
            msub = fsub.search(hdr_str, m.end())
            hdr_dict[att_name] = msub is not None and \
                                 parse_func(hdr_str, offset=msub.end()) or 0

    # still need to get n_vol, TE, and slice acq order
            
    m = re.compile('<ParamLong."NRepMeas">').search(hdr_str)
    hdr_dict['n_vol'] = find_long(hdr_str, m.end())

    m = re.compile('<ParamArray."TE">').search(hdr_str)
    te = []
    mte = long_re.search(hdr_str, m.end())
    te.append(float(mte.group()))
    # if it's agems, do it twice
    if isagems:
        mte = long_re.search(hdr_str, mte.end())
        te.append(float(mte.group()))
    hdr_dict['te'] = te

    m = re.compile('<ParamArray."SpacingBetweenSlices">').search(hdr_str)
    hdr_dict['dSL'] = find_dec(hdr_str, m.end())

    m = re.compile('<ParamArray."AnatomicalSliceNo">').search(hdr_str)
    acq_order = []
    # this matches { } and { 2 } etc
    sl_re = re.compile(r"\{+[- 0-9]+\}+")
    for mi in sl_re.finditer(hdr_str, m.end()):
        #strip away ' ', '{', '}'
        s = mi.group().strip('{} ')
        if not s:
            acq_order.append(0)
            continue
        s = s.split('-')
        if len(s) > 1:
            break
        else:
            acq_order.append(int(s[0]))
    hdr_dict['acq_order'] = N.array(acq_order)

    return hdr_dict

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
    3) volume number (repetition) -- groups by volume
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
            yield self.mmap[bnum]

    def __getitem__(self, slicer):
        return self.mmap[slicer]

    def __setitem__(self, slicer, item):
        raise IOError('this is a read-only memmap')
