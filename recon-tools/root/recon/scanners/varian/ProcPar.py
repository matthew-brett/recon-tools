"Interface to the procpar file in Varian scan output"
from os.path import join as pjoin
from itertools import imap, ifilter
from tokenize import generate_tokens
import numpy as N
import token as tokids


##############################################################################
class StaticObjectDictMixin (object):
    """
    When mixed-in with a class which supports the dict interface,
    this mixin will make it read-only and also add the ability to
    access its values as attributes.
    """
    def __setitem__( self, key, val ):
        raise AttributeError, 'object is read-only'

    def __setattr__( self, attname, val ):
        raise AttributeError, 'object is read-only'

    def __getattr__( self, attname ):
        try:
            return self[attname]
        except KeyError:
            raise AttributeError, attname


#-----------------------------------------------------------------------------
def advanceTo( tokens, stopid ):
    collection = []
    for tokid, tok in tokens:
        collection.append( (tokid, tok) )
        if tokid == stopid: break
    return collection

#-----------------------------------------------------------------------------
def advanceBy( tokens, count ):
    collection = []
    for tokid, tok in tokens:
        if count < 1: break
        if tokid == tokids.NEWLINE: continue
        collection.append( (tokid, tok) )
        count -= 1
    return collection

#-----------------------------------------------------------------------------
def cast( toktup ):
    tokid, tok = toktup
    if   tokid == tokids.STRING: return tok[1:-1]
    elif tokid == tokids.NUMBER:
        characteristic = tok[0]=="-" and tok[1:] or tok
        if characteristic.isdigit(): return int(tok)
        else: return float(tok)
    else: 
        print tokids.tok_name[tokid], tok
        return tokids.tok_name[tokid]


##############################################################################
class ProcPar (StaticObjectDictMixin, dict):
    """
    A read-only representation of the named records found in a Varian procpar
    file.  Record values can be accessed by record name in either dictionary
    style or object attribute style.  A record value is either a single element
    or a tuple of elements.  Each element is either a string, int, or float.
    """

    def __init__( self, filename ):
        # just need a mutable object to hang the negative flag on
        class foo: pass
        state = foo()

        # flag indicating when we've parse a negative sign
        state.isneg = False

        # stream of each element's first two items (tokenid and token) from
        # the raw token stream returned by the tokenize module's
        # generate_tokens function
        tokens = imap(lambda t: t[:2], generate_tokens(file(filename).readline))

        # filter out negative ops (but remember them in case they come before
        # a number)
        def negfilter( toktup ):
            tokid, tok = toktup
            if tok == "-":
                state.isneg = True
                return False
            return True
        tokens = ifilter( negfilter, tokens )

        # add back the negative sign for negative numbers
        def negnums( toktup ):
            tokid, tok = toktup
            extra = ""
            if state.isneg:
                state.isneg = False
                if tokid == tokids.NUMBER: extra = "-"
            return tokid, "%s%s"%(extra,tok)
        tokens = imap( negnums, tokens )

        class KeyValPairs:
            def __iter__( self ): return self
            def next( self ):
                toks = advanceTo( tokens, tokids.NAME )
                if not toks or not toks[-1][0] == tokids.NAME:
                    raise StopIteration
                else: name = toks[-1][1]
                rest = advanceTo( tokens, tokids.NEWLINE )
                numvalues = int( tokens.next()[1] )
                values = tuple( map( cast, advanceBy( tokens, numvalues ) ) )
                return name, values
        
        self.update( dict( [pair for pair in KeyValPairs()] ) )


##############################################################################
class CachedReadOnlyProperty (property):
    _id = 0
    def __init__(self, getter, doc):
        key = CachedReadOnlyProperty._id
        def cached_getter(self):
            if not hasattr(self, "_propvals"): self._propvals = {}
            return self._propvals.setdefault(key, getter(self))
        CachedReadOnlyProperty._id += 1
        property.__init__(self, fget=cached_getter, doc=doc)

##############################################################################
class ProcParImageMixin (object):
    """
    Knows how to extract basic MR image parameters from a Varian procpar file.
    
    ProcParImageMixin keeps a local ProcPar object, and then redefines some
    ProcPar parameters into useful image properties.
    """

    n_fe = CachedReadOnlyProperty(lambda self: self._procpar.np[0], "")

    n_fe_true = CachedReadOnlyProperty(lambda self: self.n_fe/2, "")

    isepi = CachedReadOnlyProperty(
        lambda self: self.pulse_sequence.find('epidw') != -1 or \
                     self.pulse_sequence.find('testbrs') != -1, "")

    isravi = CachedReadOnlyProperty(
        lambda self: not self.isepi and \
        self.pulse_sequence.find('epi') != -1, "")

    ismultislice = CachedReadOnlyProperty(
        lambda self: self.pulse_sequence in ('gems', 'asems', 'asems_mod'), "")

    ismpflash_like = CachedReadOnlyProperty(
        lambda self: self.pulse_sequence in ('box3d_slab', 'box3d_v2',
                                             'mp_flash3d'), "")
    
    # oops, nv is screwed up for half-space data! try nv/2 + over-scan
    n_pe = CachedReadOnlyProperty(lambda self: self.isepi and \
                        self._procpar.fract_ky[0]+self._procpar.nv[0]/2 or \
                        self._procpar.nv[0], "")
    
    n_pe_true = CachedReadOnlyProperty(
        lambda self: self.n_pe - self.nav_per_slice, "")

    pe0 = CachedReadOnlyProperty(
        lambda self: -self._procpar.get("fract_ky", [self.n_pe/2,])[0], "")

    orient = CachedReadOnlyProperty(lambda self: self._procpar.orient[0], "")

    spinecho = CachedReadOnlyProperty(
        lambda self: self._procpar.get("spinecho", ("n",))[0] == "y", "")

    # seems that this should count as true if field simply exists
    flash_converted = CachedReadOnlyProperty(
        lambda self: 
        getattr(self._procpar, "flash_converted", ("foo",))[0] != "foo", "")

    acq_cycles = CachedReadOnlyProperty(
        lambda self: getattr(self._procpar, "acqcycles", (None,))[0], "")

    n_transients = CachedReadOnlyProperty(
        lambda self: getattr(self._procpar, "ct", (-1,))[0], "")

    def _get_pulse_sequence(self):
        pslabel = self._procpar.pslabel[0]
        if pslabel == "Vsparse": pslabel = "epidw"
        elif self.spinecho and pslabel== 'epidw':
            pslabel = "epi%dse%dk" % (self.n_pe, self.nseg)
        return pslabel
    pulse_sequence = CachedReadOnlyProperty(_get_pulse_sequence, "")

    def _get_petable_name(self):
        petable = self._procpar.petable[0]
        if self.isepi and self.spinecho and petable.rfind('epidw') < 0:
            petable = "epi%dse%dk" % (self.n_pe, self.nseg)
        return petable
    petable_name = CachedReadOnlyProperty(_get_petable_name, "")

    asym_times = CachedReadOnlyProperty(
        lambda self: getattr(self._procpar, "asym_time", ()), "")

    # seems to be that acq_cycles either equals num volumes, or
    # num slices (in the case num volumes = 1)
    mpflash_vols = CachedReadOnlyProperty(
        lambda self: (self.acq_cycles != self.nslice and self.acq_cycles \
                      or 1), "")

    def _get_nvol_true(self):
        if self.isepi or self.isravi:
            return len(self.is_imagevol)
        if self.ismpflash_like:
            return self.mpflash_vols
        else:
            return self.acq_cycles
    
    nvol_true = CachedReadOnlyProperty(_get_nvol_true, "")

    def _get_is_imagevol(self):
        procpar = self._procpar
        if self.isepi and hasattr(procpar, "image"):
            return procpar.image
        elif self.isravi and hasattr(procpar, "cntr"):
            return procpar.cntr
        else:
            return [1]*self.nvol_true

    is_imagevol = CachedReadOnlyProperty(_get_is_imagevol, "")

    # image volumes are shown by a "1" index in the is_imagevol list
    # ref volumes are shown by numbers != 1
    ref_vols = CachedReadOnlyProperty(
        lambda self:\
        [i for i, isimage in enumerate(self.is_imagevol) if isimage!=1], "")

    image_vols = CachedReadOnlyProperty(
        lambda self:\
        [i for i, isimage in enumerate(self.is_imagevol) \
         if (isimage==1 and i in self.vrange)], "")

    nvol = CachedReadOnlyProperty(lambda self: len(self.vrange), "")

    slice_positions = CachedReadOnlyProperty(
        lambda self: 10.*N.asarray(self._procpar.pss), "")

    gss = CachedReadOnlyProperty(lambda self: self._procpar.gss[0], "")

    ### Not exactly in procpar, but useful still
    ### 
    acq_order = CachedReadOnlyProperty(
        lambda self:
        N.asarray((self.ismpflash_like and range(self.nslice) or \
                   (range(self.nslice-1,-1,-2) +
                    range(self.nslice-2,-1,-2)))), "")

    def _get_slice_gap(self):
        if self.pulse_sequence == "mp_flash3d" or self.nslice < 2: return 0.
        spos = self.slice_positions
        return ((max(spos) - min(spos) + self.thk)
               - (self.nslice*self.thk))/(self.nslice - 1)
    slice_gap = CachedReadOnlyProperty(_get_slice_gap, "")

    nslice = CachedReadOnlyProperty(
        lambda self: self.ismpflash_like and \
        self._procpar.nv2[0] or \
        len(self.slice_positions), "")

    thk = CachedReadOnlyProperty(
        lambda self: self.pulse_sequence == "mp_flash3d" and \
        10.*self._procpar.lpe2[0]/self.nslice or \
        self._procpar.thk[0], "")

    # If there is an extra (odd) line per segment, then it is navigator (??)
    nav_per_seg = CachedReadOnlyProperty(
        lambda self: (self.n_pe/self.nseg)%2, "")

    pe_per_seg = CachedReadOnlyProperty(lambda self: self.n_pe/self.nseg, "")

    nav_per_slice = CachedReadOnlyProperty(
        lambda self: self.nseg*self.nav_per_seg, "")

    def _get_nseg(self):
        # this is very crude, but "nseg" in procpar doesn't reflect reality!
        if self.pulse_sequence in ('epi','tepi','sparse','spare','mp_flash3d'):
            return int(self.petable_name[-2])
        elif self.pulse_sequence in ('epidw', 'Vsparse','testbrs2') and\
          not self.spinecho:
            return int(self.petable_name[-1])
        elif self.pulse_sequence in ('epi','epidw') and self.spinecho:
            return self._procpar.nseg[0]
        elif self.ismultislice or self.ismpflash_like:
            return 1
        else: raise ValueError(
              "Could not identify sequence: %s" % (self.pulse_sequence))
    nseg = CachedReadOnlyProperty(_get_nseg, "")

    sampstyle = CachedReadOnlyProperty(lambda self:
                                       (self.petable_name.find('cen')>0 or \
                                        self.petable_name.find('alt')>0) and \
                                       "centric" or "linear", "")

    tr = CachedReadOnlyProperty(lambda self: self.nseg*self._procpar.tr[0], "")

    phi = CachedReadOnlyProperty(lambda self: self._procpar.phi[0], "")

    psi = CachedReadOnlyProperty(lambda self: self._procpar.psi[0], "")

    theta = CachedReadOnlyProperty(lambda self: self._procpar.theta[0], "")

    delT = CachedReadOnlyProperty(lambda self: 1./self._procpar.sw[0], "")

    dFE = CachedReadOnlyProperty(
        lambda self: 10.*float(self._procpar.lro[0])/self.n_fe_true, "")

    dPE = CachedReadOnlyProperty(
        lambda self: 10.*float(self._procpar.lpe[0])/self.n_pe_true, "")

    dSL = CachedReadOnlyProperty(
        lambda self: float(self.thk) + self.slice_gap, "")

    datasize = CachedReadOnlyProperty(
        lambda self: self._procpar.dp[0]=="y" and 4 or 2, "")

    raw_dtype = CachedReadOnlyProperty(
        lambda self: self._procpar.dp[0]=="y" and N.int32 or N.int16, "")

    echo_factor = CachedReadOnlyProperty(
        lambda self: 2.0*abs(self._procpar.gro[0])\
                     *self._procpar.trise[0]/self._procpar.gmax[0], "")

    def _get_echo_time(self):
        if self.petable_name.find("alt") != -1:
            return self._procpar.te[0] - self.echo_factor - self._procpar.at[0]
        else:
            return self._procpar.te[0] - \
                   N.floor(self.pe_per_seg)/2.0*self.echo_spacing
    echo_time = CachedReadOnlyProperty(_get_echo_time, "")

    echo_spacing = CachedReadOnlyProperty(
        lambda self: self.echo_factor + self._procpar.at[0], "")

    pe_times = CachedReadOnlyProperty(
        lambda self: N.asarray(
                        [self.echo_time + pe*self.echo_spacing \
                         for pe in range(self.pe_per_seg)]), "")

    # time between beginning of one PE scan and the next (was dwell_time)
    T_pe = CachedReadOnlyProperty(
	lambda self: getattr(self._procpar, "at_calc", (None,))[0], "")

    # this quiet_interval may need to be added to tr in some way...
    quiet_interval = CachedReadOnlyProperty(
        lambda self: self._procpar.get("dquiet", (0,))[0], "")

    #-------------------------------------------------------------------------
    def __init__(self, datadir, vrange=None):
        procpar = self._procpar = ProcPar(pjoin(datadir, "procpar"))
        # allow manual override of some values
        #self._tr= tr
        if vrange:
            skip = len(self.ref_vols)
            # if vrange[1] is -1 or too big, set it to nvol_true
            # if it is 0 (for unknown reason), will go to img vol 1
            vend = vrange[1] in range(self.nvol_true-skip) \
                   and vrange[1]+1+skip or self.nvol_true
            # if it is past the last volume, make it one spot less
            # else, make it vrange[0]+skip (even if that == 0)
            if vrange[0] in range(vend-skip):
                vstart = vrange[0] + skip
            else:
                vstart = vend - 1
            self.vrange = range(vstart,vend)
        else: self.vrange = range(len(self.ref_vols), self.nvol_true)

##############################################################################
if __name__ == "__main__":
    import pprint
    p = ProcPar( "procpar" )
    pprint.pprint( p )
