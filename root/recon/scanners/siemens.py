from recon.scanners import _HeaderBase, _BitMaskedWord, ScannerImage, \
     ReconImage, CachedReadOnlyProperty
from recon.util import TempMemmapArray, Quaternion, eulerRot, real_euler_rot
from recon.scanners.siemens_utils import strip_ascconv, header_length, \
     condense_array, header_string
import os, struct, sys, re, stat
import numpy as np


def load_gre(dat, arr, interp_xform, scl=1.):
    # kspshape is something like (nchan,2,nsl,npe,M1) .. but actually
    # there are two echos, not two volumes 
    # But, we still want to stack the data up in order of channels

    # line -> pe
    # echo -> vol
    # slice -> slice
    # chan -> chan
    for d in dat:
        m = MDH(d['hdr'])
        arr[m.chan, m.echo, m.slice, m.line] = interp_xform(d['data'])
    if scl != 1.0:
        arr *= scl


class SiemensImage(ScannerImage):
    def __init__(self, filestem, vrange=None, use_mmap=False,
                 osamp=True, **kwargs):
        # this is required for now
        osamp = True
        self.path = os.path.abspath(os.path.splitext(filestem)[0]+'.dat')
        self.__dict__.update(parse_siemens_hdr(self.path))

        if osamp:
            self.N1 *= 2
            self.fov_x *= 2
        # this should go by the wayside
        self.Nc = self.N1/2+1

        # fake/fix a couple things still
        if self.acq_order.shape[0] != self.n_slice:
            self.acq_order = np.array( range(1, self.n_slice, 2) +
                                       range(0, self.n_slice, 2) )

        if self.n_refs: # this may be irrelevant
            # nrefs reports as 2 when it is really 3
            self.n_refs += 1

        if not hasattr(self, 'echo_spacing'):
            self.echo_spacing = 0
        if not hasattr(self, 'accel') or not self.accel:
            self.accel = 1
        if self.isagems:
            self.n_vol = self.n_echo
        if self.isgrs:
            self.n_slice *= self.n_part
        if vrange:
            self.n_vol = vrange[1] - vrange[0] + 1
        else:
            vrange = (0, self.n_vol-1)
        
        # this is a little bit magicky, but since I've switched to
        # learning much about the data in the load phase, it's become
        # appropriate to make this functionality a method of the Image object
        loud = kwargs.pop('loud', False)
        self._load_data(use_mmap, vrange, loud=loud)
    
        self.use_membuffer()
        self.combined = False

        ScannerImage.__init__(self, offset=(self.x0, self.y0, self.z0))



    @CachedReadOnlyProperty
    def echo_time(self):
        return self.te[0]/1e6
    @CachedReadOnlyProperty
    def asym_times(self):
        if self.isagems:
            return np.array(self.te)/1e6
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
        return np.arange(self.n_pe)
    @CachedReadOnlyProperty
    def dFE(self):
        return self.fov_x/self.n_fe
    @CachedReadOnlyProperty
    def dPE(self):
        return self.fov_y/self.n_pe
    @CachedReadOnlyProperty
    def delT(self):
        # return this in secs
        return self.dwell_time/1e9
    @CachedReadOnlyProperty
    def T_pe(self):
        # return this in secs
        return self.echo_spacing/1e6
    
    def use_membuffer(self, chan=0):
        self.load_chan(chan, copy=True)

    def load_chan(self, chan, copy=False):
        for arr_name in ('cdata', 'cref_data', 'cacs_data', 'cacs_ref_data'):
            if hasattr(self, arr_name) and getattr(self, arr_name) is not None:
                carr = getattr(self, arr_name)
                try:
                    carr.flush()
                except:
                    # was not a np.memmap type
                    pass
                if copy:
                    arr = np.empty(carr.shape[1:], carr.dtype)
                    arr[:] = carr[chan]
                else:
                    arr = carr[chan]
                # I'm not sure how important this special case is...
                if arr_name=='cdata':
                    self.setData(arr)
                else:
                    setattr(self, arr_name[1:], arr)
                    
        self.combined = False
        self.chan = chan   

    def combine_channels(self):
        dtype = self.data.dtype
        if type(self.data) is not np.memmap:
            del self.data
        d = np.zeros(self.cdata.shape[-4:], dtype.char.lower())
        for n,chan in enumerate(self.cdata):
            d += np.power(self.channel_gains[n]*chan.real, 2.0) + \
                 np.power(self.channel_gains[n]*chan.imag, 2.0)
        self.setData(np.sqrt(d))
        self.combined = True
        self.chan = None

    @CachedReadOnlyProperty
    def n_ramp(self):
        if not self.ramp_samp:
            return 0
        (Tr, T0, Tf) = map(float, (self.T_ramp, self.T0, self.T_flat))
        T0 = float(self.T0)
        N1 = self.N1
        As = (Tf + Tr - T0**2/Tr)/2.
        return int( (self.Nc-1)*(Tr**2 - T0**2)/(2*Tr*As) )


    @CachedReadOnlyProperty
    def n_flat(self):
        if not self.ramp_samp:
            return int(self.N1)-1
        (Tr, T0, Tf) = map(float, (self.T_ramp, self.T0, self.T_flat))
        N1 = self.N1
        T0 = float(self.T0)
        As = (Tf + Tr - T0**2/Tr)/2.
        # when T0 == Tr, then this evaluates to (Nc-1)*2 == N1 (too big by 1)
        return min(int(N1)-1,
                   int( (self.Nc-1)*(2*Tr*Tf + Tr**2 - T0**2)/(2*Tr*As) ))
    
    def t_n1(self):
        (Tr, T0, Tf) = map(float, (self.T_ramp, self.T0, self.T_flat))
        N1 = self.N1
        Nc = self.Nc
        As = (Tf + Tr - T0**2/Tr)
        nr = self.n_ramp
        nf = self.n_flat
        r1 = np.arange(0, nr+1)
        r2 = np.arange(nr+1, nf+1)
        r3 = np.arange(nf+1, int(N1))
        t = np.zeros(int(N1))

        t[r1] = np.power( T0**2 + 2.0*r1*Tr*As/(N1-1), 0.5)
        t[r2] = r2*As/(N1-1) + Tr/2. + T0**2/(2.*Tr)
        t[r3] = (Tf+2*Tr) - np.power(2.*Tr*(Tf+Tr-T0**2/(2.*Tr) - r3*As/(N1-1)), 0.5)

        return t

    def sinc_kernel(self):
        # make a M1->N1 sinc-interpolation kernel, where each column
        # interpolates the raw signal to t[n1]

        # this really needs to be re-computed to fit the values at hand,
        # since there is some rounding error in the siemens header
        delT = (2*self.T_ramp + self.T_flat - 2*self.T0)/float(self.M1-1)
        #delT = self.delT * 1e6
        t = (self.t_n1() - self.T0)/delT
        print t
        m1 = np.arange(self.M1)
        s_idx = t[None,:] - m1[:,None]
        return np.sinc(s_idx).astype(np.complex64)




    def _load_data(self, use_mmap, vrange, loud=True):            

        array_creator = TempMemmapArray if use_mmap else np.empty
        ref_sampling = []
        ref_rev = []

        pe_sampling = []
        pe_rev = []

        acs_sampling = []
        acs_rev = []

        acs_ref_sampling = []
        acs_ref_rev = []

        vol_start, vol_end = vrange

        #hdr = sm.parse_siemens_hdr(p)
        n_chan = self.n_chan #hdr['n_chan']
        n_slice = self.n_slice #hdr['n_slice']

        # start by memory mappping ALL blocks.. this first pass should
        # only scan through a small section of blocks anyway.
        d = MemmapDatFile(self.path)
        nz_scans = 0
        acs_ref_counter = 0
        epi_ref_counter = 0

        # FIRST PASS TO LEARN THE DATA LAYOUT
        for b in d[::n_chan]:
            m = MDH(b['hdr'])
            if m.info0.NOISEADJ:
                nz_scans += 1
                continue
            if m.info0.PATREF:
                if m.slice > 0:
                    continue
                if m.info0.PHSCOR:
                    acs_ref_sampling.append(m.line)
                    if m.info0.REFLECT:
                        # here record WHICH ref it was,
                        # since all lines==ksp_center_line
                        acs_ref_rev.append(acs_ref_counter)
                    acs_ref_counter += 1

                else:
                    acs_ref_counter = 0
                    acs_sampling.append(m.line)
                    if m.info0.REFLECT:
                        acs_rev.append(m.line)
            else:
                if m.info0.PHSCOR:
                    ref_sampling.append(m.line)
                    if m.info0.REFLECT:
                        ref_rev.append(epi_ref_counter)
                    epi_ref_counter += 1
                else:
                    epi_ref_counter = 0
                    if m.line not in pe_sampling:
                        pe_sampling.append(m.line)
                    if m.info0.REFLECT:
                        pe_rev.append(m.line)
                if m.info0.FIRSTSCANINSLICE:
                    ksp_center_line = m.ksp_center_line
                if m.info0.LASTSCANINSLICE:
                    break

        upper_half = max(pe_sampling) - ksp_center_line
        lower_half = ksp_center_line - min(pe_sampling)
        if loud:
            if acs_sampling:
                print 'ACS sampling pattern:', acs_sampling, acs_rev
                print 'ACS ref scans:', acs_ref_sampling, acs_ref_rev
            print 'PE sampling pattern:', pe_sampling, pe_rev
            print 'PE ref scans:', ref_sampling, ref_rev
            print ksp_center_line, 'out of', len(pe_sampling), 'sampled lines'
            print 'upper half length:', upper_half
            print 'lower half length:', lower_half
        # is this the correct criterion? in a normal scan with
        # N2 phase encodes, center is N2/2, and
        # lower half == N2/2, upper half == (N2-1-N2/2)) == N2/2-1
        if abs(upper_half - lower_half) > 1:
            epi_offset = upper_half - lower_half
            pe_sampling = [pe + epi_offset for pe in pe_sampling]
            pe_rev = [pe + epi_offset for pe in pe_rev]
            ksp_center_line += epi_offset
            if loud:
                print 'sampling in zero filled grid:', pe_sampling
                print 'reverse gradient lines:', pe_rev
                print 'ksp_center_line:', ksp_center_line
        else:
            epi_offset = 0

        # ALLOCATE MEMORY AND READ IN DATA

        n_vol = self.n_vol #hdr['n_vol']
        n_slice = self.n_slice #hdr['n_slice']
        N1 = self.N1 #hdr['M1'] # keep oversampling
        
        n_pe_sampled = len(pe_sampling)
        n_pe_ref = len(ref_sampling)

        data = array_creator((n_chan, n_vol, n_slice, self.n_pe, N1), 'F')
        ref = array_creator((n_chan, n_vol, n_slice, n_pe_ref, N1), 'F') \
              if n_pe_ref else None

        if acs_sampling:
            n_acs = int(self.n_acs)
            n_acs_reps = len(acs_sampling)/n_acs
            n_acs_ref = len(acs_ref_sampling)/n_acs_reps
            # truncate lists to remove duplicate acquisitions
            acs_sampling = acs_sampling[:n_acs]
            acs_ref_sampling = acs_ref_sampling[:n_acs_ref]
            acs_rev = acs_rev[: (len(acs_rev)/n_acs_reps) ]
            acs_ref_rev = acs_ref_rev[: (len(acs_ref_rev)/n_acs_reps) ]

            # make these plain old arrays, they are not big
            acs_data = np.empty((n_chan, n_acs_reps,
                                 n_slice, n_acs, N1), 'F')
            acs_ref = np.empty((n_chan, n_acs_reps,
                                n_slice, n_acs_ref, N1), 'F')
            print acs_ref.shape

            acs_offset = -acs_sampling[0]
            acs_ref_lu = {}
            acs_samp_lu = {}
        else:
            n_acs = n_acs_reps = n_acs_ref = 0
        epi_ref_lu = {}

        # COIL NOISE SAMPLES + ACS samples + Image samples
        nblocks = nz_scans + \
                  n_chan*n_slice*n_acs_reps*(n_acs + n_acs_ref) + \
                  n_chan*n_vol*n_slice*(n_pe_sampled + n_pe_ref)
        print '# of blocks in dat file:', nblocks
        del d
        d = MemmapDatFile(self.path, nblocks=nblocks)
        acq2anat = self.acq_order

        for b in d[nz_scans:]:
            m = MDH(b['hdr'])
            s = acq2anat[m.slice]
            if m.info0.PATREF:
                # all ref scans for a slice are read in first
                if m.info0.PHSCOR:
                    try:
                        n = acs_ref_lu[(m.chan, s)]
                    except KeyError:
                        acs_ref_lu[(m.chan, s)] = 0
                        n = 0
                    acs_rep = n // n_acs_ref
                    r = n % n_acs_ref
                    # still should reflect ref echoes here??
                    if m.info0.REFLECT:
                        acs_ref[m.chan, acs_rep, s, r, :] = b['data'][::-1]
                    else:
                        acs_ref[m.chan, acs_rep, s, r, :] = b['data']
                    # increment lookup counter
                    n += 1
                    acs_ref_lu[(m.chan, s)] = n
                # afterwards, reset the ref scan counter to 0 until the next slice
                else:
                    # The channel, slice, line numbers repeat for every
                    # ACS acquisition. So if we've seen this combo before,
                    # then the dictionary tells us which acq index we're on
                    try:
                        n = acs_samp_lu[(m.chan, s, m.line)]
                    except KeyError:
                        acs_samp_lu[(m.chan, s, m.line)] = 0
                        n = 0
                    acs_data[m.chan, n, s, m.line + acs_offset, :] = b['data']
                    n += 1
                    acs_samp_lu[(m.chan, s, m.line)] = n
            else:
                if m.rep < vol_start:
                    continue
                if m.rep > vol_end:
                    break
                if m.info0.PHSCOR:
                    try:
                        n = epi_ref_lu[(m.chan, m.rep, s)]
                    except KeyError:
                        epi_ref_lu[(m.chan, m.rep, s)] = 0
                        n = 0
                    # still should reflect ref echoes here??
                    if m.info0.REFLECT:
                        ref[m.chan, m.rep, s, n, :] = b['data'][::-1]
                    else:
                        ref[m.chan, m.rep, s, n, :] = b['data']
                    n += 1
                    epi_ref_lu[(m.chan, m.rep, s)] = n
                else:
                    if self.isagems or self.isgre:
                        v = m.echo
                    else:
                        v = m.rep - vol_start
                    data[m.chan, v, s, m.line + epi_offset, :] = b['data']
        ############## LEARNED AND LOADED INFORMATION ########################
        self.cdata = data
        self.cref_data = ref
        self.pe_sampling = pe_sampling
        self.pe_reflected = pe_rev
        self.ref_reflected = ref_rev
        # leave these as empty lists if there is no ACS
        self.acs_sampling = acs_sampling
        self.acs_reflected = acs_rev
        self.acs_ref_reflected = acs_ref_rev
        # but only assign these if there is ACS
        if acs_sampling:
            self.cacs_data = acs_data
            self.cacs_ref_data = acs_ref


##############################################################################
###########################   FILE LEVEL CLASSES   ###########################
##############################################################################
def parse_siemens_hdr(fname):

    """
    lRepetitions                             = 19 (= nvol - 1)
    sKSpace.lBaseResolution                  = 64
    sKSpace.lPhaseEncodingLines              = 128
    sKSpace.ucMultiSliceMode                 = 0x2 (interleaved)
    tSequenceFileName                        = "%CustomerSeq%\ep2d_bold_bigfov"
    sRXSPEC.alDwellTime[0]                   = 2800
    alTR[0]                                  = 2500000
    lContrasts                               = 2
    alTE[0]                                  = 4920
    alTE[1]                                  = 7380
    sSliceArray.asSlice[0].sPosition.dTra    = -39.375
    sSliceArray.asSlice[0].sNormal.dTra      = 1
    sSliceArray.asSlice[0].dThickness        = 3
    sSliceArray.asSlice[0].dPhaseFOV         = 240
    sSliceArray.asSlice[0].dReadoutFOV       = 240
    sSliceArray.asSlice[0].dInPlaneRot       = 2.051034897e-010
    ...
    sSliceArray.lSize                        = 22
    sFastImaging.lEchoSpacing                = 420
    sPat.lAccelFactPE                        = 1
    sPat.lRefLinesPE                         = 24
    asCoilSelectMeas[0].aFFT_SCALE[0].flFactor = 1.26665
    asCoilSelectMeas[0].aFFT_SCALE[0].bValid = 1
    asCoilSelectMeas[0].aFFT_SCALE[0].lRxChannel = 1
    ...

    more info...
    sSliceArray.ucMode (I THINK this is slice acq orde -- can it be OR'd??)
    enum SeriesMode
    {
      ASCENDING   = 0x01,
      DESCENDING  = 0x02,
      INTERLEAVED = 0x04
    };
    
    sKSpace.unReordering
    enum Reordering
    {
      REORDERING_LINEAR    = 0x01,
      REORDERING_CENTRIC   = 0x02,
      REORDERING_LINE_SEGM = 0x04,
      REORDERING_PART_SEGM = 0x08,
      REORDERING_FREE_0    = 0x10,
      REORDERING_FREE_1    = 0x20,
      REORDERING_FREE_2    = 0x40,
      REORDERING_FREE_3    = 0x80
    };
    sKSpace.ucPhasePartialFourier
    sKSpace.ucSlicePartialFourier
    enum PartialFourierFactor
    {
      PF_HALF = 0x01,
      PF_5_8  = 0x02,
      PF_6_8  = 0x04,
      PF_7_8  = 0x08,
      PF_OFF  = 0x10
    };
    sKSpace.ucMultiSliceMode
    enum MultiSliceMode
    {
      MSM_SEQUENTIAL  = 0x01,
      MSM_INTERLEAVED = 0x02,
      MSM_SINGLESHOT  = 0x04
    };

    slice spacing ( can do with position arrays )
    slice order / acq order

    missing:
    n_pe_acq (can this be derived from n_pe and accel? may be 31 for accel=2)
    n_ref == nsegmeas + 1 (or is that a coincidence???)
    n_part -- do I care?
    rampsamp info
    """
    hdr_dict = {}
##     hdrlen = header_length(fname)
##     hdr_str = open(fname, 'r').read(4+hdrlen)
    hdr_str = header_string(fname)
    asc_dict = strip_ascconv(hdr_str)
    chan_scales = condense_array(asc_dict, 'asCoilSelectMeas[0].aFFT_SCALE')
    gains = chan_scales['flFactor']
    chans = chan_scales['lRxChannel'].astype('i') # usually [1, 2, 3, 4, ...]
    hdr_dict['n_chan'] = gains.shape[0]
    hdr_dict['channel_gains'] = gains[chans-1]
    hdr_dict['n_echo'] = int(asc_dict['lContrasts'])
    hdr_dict['n_vol'] = int(1 + asc_dict.get('lRepetitions', 0))
    hdr_dict['n_slice'] = int(asc_dict.get('sSliceArray.lSize', 1))
    hdr_dict['n_partition'] = int(asc_dict.get('sKSpace.lPartitions', 1))
    # I've seen set down as 127.. I don't think it could hurt to enforce 2**n
    # .. doing ceil(log2(n_pe)) is much more strong than round(log2(n_pe))
    # .. which one??
    n_pe = int(asc_dict['sKSpace.lPhaseEncodingLines'])
    n_pe = int(2**np.ceil(np.log2(n_pe)))
    hdr_dict['n_pe'] = n_pe
    hdr_dict['N1'] = int(asc_dict['sKSpace.lBaseResolution'])
    hdr_dict['M1'] = hdr_dict['n_fe'] = 2*hdr_dict['N1']
    hdr_dict['fov_x'] = asc_dict['sSliceArray.asSlice[0].dReadoutFOV']
    hdr_dict['fov_y'] = asc_dict['sSliceArray.asSlice[0].dPhaseFOV']
    hdr_dict['tr'] = asc_dict['alTR[0]']
    hdr_dict['te'] = []
    for e in range(hdr_dict['n_echo']):
        hdr_dict['te'].append(asc_dict['alTE[%d]'%e])
    hdr_dict['dwell_time'] = asc_dict['sRXSPEC.alDwellTime[0]']
    hdr_dict['echo_spacing'] = asc_dict.get('sFastImaging.lEchoSpacing', 0.0)
    hdr_dict['accel'] = asc_dict.get('sPat.lAccelFactPE', 1)
    hdr_dict['n_acs'] = asc_dict.get('sPat.lRefLinesPE', 0)
    pslabel = asc_dict['tSequenceFileName'].split('\\')[-1]
    hdr_dict['isepi'] = (pslabel.find('ep2d') >= 0)
    hdr_dict['isgre'] = (pslabel=='gre')
    hdr_dict['isagems'] = (pslabel=='gre_field_mapping')
    hdr_dict['isgrs'] = (pslabel.find('grs3d') >= 0)
    hdr_dict['pslabel'] = pslabel

    # now decode some bit encoded fields
    slices = np.arange(hdr_dict['n_slice'])
    slicing_mode = asc_dict['sSliceArray.ucMode']
    if slicing_mode == 4:
        # interleaved
        if hdr_dict['n_slice']%2:
            hdr_dict['acq_order'] = np.concatenate((slices[0::2], slices[1::2]))
        else:
            hdr_dict['acq_order'] = np.concatenate((slices[1::2], slices[0::2]))
    elif slicing_mode == 2:
        # descending
        hdr_dict['acq_order'] = slices[::-1]
    else:
        # ascending
        hdr_dict['acq_order'] = slices

    sampstyle = asc_dict['sKSpace.unReordering']
    hdr_dict['sampstyle'] = {1:'linear',2:'centric'}.get(sampstyle, 'unknown')
    
    partial_fourier = asc_dict['sKSpace.ucPhasePartialFourier']
    # factors are encoded in increments of n/8 starting at 4
    partial_numerator = {1: 4, 2: 5, 4: 6, 8: 7, 16: 8 }.get(partial_fourier)
    hdr_dict['pe0'] = int(round(hdr_dict['n_pe']*(1/2. - partial_numerator/8.)))

    # Now get rampsamp info, n_pe_acq (in case of accel>1), n_ref
    fields_by_section = {
        '<ParamFunctor."rawobjprovider">': [ ('<ParamLong."RawLin">', # raw str
                                              'n_pe_acq', # our name
                                              int) ], # how to convert
        '<ParamFunctor."adjroftregrid">': [ ('<ParamLong."RampupTime">',
                                             'T_ramp',
                                             float),
                                            ('<ParamLong."FlattopTime">',
                                             'T_flat',
                                             float),
                                            ('<ParamLong."DelaySamplesTime">',
                                             'T0',
                                             float),
                                            ('<ParamDouble."ADCDuration">',
                                             'adc_period',
                                             float) ],
        '<ParamFunctor."EPIPhaseCorrPE">': [ ('<ParamLong."NSeg">',
                                              'n_refs',
                                              int) ]
        }
    for section, parse_info in fields_by_section.items():
        p = hdr_str.find(section)
        valid = p >= 0
        if valid:
            s = hdr_str[p:]
        for (field, val_name, evalfunc) in parse_info:
            if not valid:
                hdr_dict[val_name] = evalfunc('0')
                continue
            p = s.find(field)
            s = s[(p + len(field)):]
            p = s.find('\n')
            val = s[:p].split()[-2]
            hdr_dict[val_name] = evalfunc(val)

    # see if we have should forge a plausible gradient shape
    if not hdr_dict['adc_period'] and (hdr_dict['isepi'] or hdr_dict['isgrs']):
        # time resolution for gradient events is 5 microsec..
        # calculate flat time as dt * M1
        # cut echo spacing time into 3 even parts: ramp_up + flat + ramp_dn
        adc = (hdr_dict['M1']-1)*hdr_dict['dwell_time']/1e3
        Tpe = hdr_dict['echo_spacing']
        # 1) make flat time long enough to support adc period
        # 2) make echo-spacing - flat_time be evenly split in two
        npts = int(adc/5)
        if npts%2:
            npts += 1
        else:
            npts += 2
        flat = npts*5
        ramps = (Tpe-flat)/2
        hdr_dict['T_flat'] = flat
        hdr_dict['T_ramp'] = ramps
        hdr_dict['T0'] = ramps
        hdr_dict['adc_period'] = adc
    hdr_dict['ramp_samp'] = (hdr_dict['T0'] < hdr_dict['T_ramp'])
    
       
    # now get slice thickness order, x0, y0, z0
    # (or is it i0, j0, k0 ??)
            
    slice_arrays = condense_array(asc_dict, 'sSliceArray.asSlice')
    ns = hdr_dict['n_slice']
    pos_tra = slice_arrays.get('sPosition.dTra', np.zeros(ns))
    pos_cor = slice_arrays.get('sPosition.dCor', np.zeros(ns))
    pos_sag = slice_arrays.get('sPosition.dSag', np.zeros(ns))
    if ns > 1:
        hdr_dict['dSL'] = np.sqrt( (pos_tra[1]-pos_tra[0])**2 + \
                                   (pos_cor[1]-pos_cor[0])**2 + \
                                   (pos_sag[1]-pos_sag[0])**2 )
        hdr_dict['slice_thick'] = slice_arrays['dThickness'][0]
        hdr_dict['slice_gap'] = hdr_dict['dSL'] - hdr_dict['slice_thick']
    else:
        hdr_dict['dSL'] = hdr_dict['slice_thick'] = hdr_dict['slice_gap'] = 1.


##     norm_tra = slice_arrays.get('sNormal.dTra', np.zeros(ns))
##     norm_cor = slice_arrays.get('sNormal.dCor', np.zeros(ns))
##     norm_sag = slice_arrays.get('sNormal.dSag', np.zeros(ns))

##     # the normal tells us the angles phi and theta in the rotation,
##     # psi is found literally in the header
##     normal = np.array([norm_sag[0], norm_cor[0], norm_tra[0]])

##     theta = np.arccos(normal[2])
##     phi = np.arccos(-normal[1]/np.sin(theta))
##     psi = slice_arrays.get('dInPlaneRot', np.zeros(ns))[0]
    
##     m = real_euler_rot(phi=phi, theta=theta, psi=psi)

    dat = MemmapDatFile(fname, nblocks=1)
    mdh = MDH(dat[0]['hdr'])
    in_plane_rot = slice_arrays.get('dInPlaneRot', np.zeros(ns))[0]
    # this is the "rotated" voxel to world mapping
    xform = Quaternion(i=mdh.quatI, j=mdh.quatJ, k=mdh.quatK)
    # this is the voxel transform in the vox coordinates
    m = xform.tomatrix()
    # maybe this is irrelevant? maybe the rotation matrix is just
    # x pe fe sl
    # L x  x  x
    # P x  x  x
    # S x  x  x
##     rot = eulerRot(phi=in_plane_rot)
##     m = np.dot(m, rot)
    # m is now the transform from (j,i,k) to (L,P,S) (DICOM coords)
    # since (R,A,S) = (-L,-P,S), just multiply two axes by -1 and swap them
    m_ijk = np.zeros_like(m)
    m_ijk[:,0] = -m[:,1]
    m_ijk[:,1] = -m[:,0]
    m_ijk[:,2] = m[:,2]
    m_ijk[:2] *= -1
    
    # now m*(n_fe/2, n_pe/2, 0) + r0 = (-pos_sag[0], -pos_cor[0], pos_tra[0])
    kdim = hdr_dict['n_slice']; ksize = hdr_dict['dSL']
    jdim = hdr_dict['n_pe']; jsize = hdr_dict['fov_y']/jdim
    # account for oversampling.. n_fe is already 2X base resolution
    idim = hdr_dict['n_fe']; isize = 2.0*hdr_dict['fov_x']/idim
    dim_scale = np.array([isize, jsize, ksize])
    sl0_center = np.array([-pos_sag[0], -pos_cor[0], pos_tra[0]])
    # want sl0_center = M*(i=idim/2,j=jdim/2,k=0) + r0
    sl0_vox_center = np.array([idim/2, jdim/2, 0])
    r0 = sl0_center - np.dot(m_ijk*dim_scale, sl0_vox_center)
##     print m*dim_scale, sl0_vox_center
##     print sl0_center, r0
    # would have to re-adjust one of the r0 components when oversampling
    # is eliminated
    hdr_dict['orientation_xform'] = Quaternion(M=m_ijk)
    hdr_dict['x0'] = r0[0]; hdr_dict['y0'] = r0[1]; hdr_dict['z0'] = r0[2]
    # faking this for now..
    hdr_dict['nseg'] = 1
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

    _shape = None

    def __init__(self, fname, nblocks=0):
        fp = open(fname, 'rb')
        self.main_header_size = header_length(fname) #struct.unpack("<i", fp.read(4))[0]
        fp.seek(self.main_header_size)
        # need to grab a header and find out the samps_per_scan in order
        # to construct the dtype((np.float32, samps_per_scan*2))

        mdh_sz = MDH.header_size()
        hdr_dtype = np.dtype('S%d'%mdh_sz)

        first_mdh = MDH(fp.read(mdh_sz))
        nsamps = first_mdh.samps_in_scan
        blk_dtype = np.dtype((np.complex64, nsamps))

        block_size = mdh_sz + nsamps*2*4
        fp.close()
        if nblocks:
            self.nblocks = nblocks
        else:
            self.nblocks = int((os.stat(fname)[stat.ST_SIZE] - self.main_header_size)/block_size)
        
        dat_type = np.dtype({'names':['hdr', 'data'], 'formats':[hdr_dtype, blk_dtype]})

        self.mmap = np.memmap(fname, dtype=dat_type,
                              offset=self.main_header_size,
                              shape=(self.nblocks,),
                              mode='r')
        self.shape = self.mmap.shape

    def _shape_setter(self, new_shape):
        try:
            self.mmap.shape = new_shape
        except:
            print 'Invalid shape'
        self._shape = self.mmap.shape

    def _shape_getter(self):
        return self._shape

            
    shape = property(fget=_shape_getter, fset=_shape_setter)

    def __iter__(self):
        for bnum in xrange(self.nblocks):
            yield self.mmap[bnum]

    def __getitem__(self, slicer):
        return self.mmap[slicer]

    def __setitem__(self, slicer, item):
        raise IOError('this is a read-only memmap')
