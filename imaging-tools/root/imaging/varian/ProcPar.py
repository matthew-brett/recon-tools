"Interface to the procpar file in Varian scan output"
from os.path import join as pjoin
from itertools import imap, ifilter
from tokenize import generate_tokens
from pylab import Int16, Int32, asarray, floor
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
    Knows how to extract basic MRIImage parameters from a Varian procpar file.
    
    ProcParImageMixin keeps a local ProcPar object, and then redefines some
    ProcPar parameters into image parameters.
    """

    n_fe = CachedReadOnlyProperty(lambda self: self._procpar.np[0], "")

    n_fe_true = CachedReadOnlyProperty(lambda self: self.n_fe/2, "")

    isepi = CachedReadOnlyProperty(
        lambda self: self.pulse_sequence.find("epi") != -1, "")

    # oops, nv is screwed up for half-space data!
    n_pe = CachedReadOnlyProperty(lambda self: self.isepi and \
                                  self._procpar.nf[0] or \
                                  self._procpar.nv[0], "")
    
    n_pe_true = CachedReadOnlyProperty(
        lambda self: self.n_pe - self.nav_per_slice, "")

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

    nvol_true = CachedReadOnlyProperty(
        lambda self: self.asym_times and len(self.asym_times) or \
                     not self.isepi and self.mpflash_vols or \
                     self._procpar.images[0], "")

    def _get_is_imagevol(self):
        # try using procpar.cntr or procpar.image to know which volumes are
        # reference
        procpar = self._procpar
        if hasattr(procpar, "cntr") and len(procpar.cntr) == self.nvol_true:
            return procpar.cntr
        elif hasattr(procpar, "image") and len(procpar.image) == self.nvol_true:
            return procpar.image
        else: return [1]*self.nvol_true
    is_imagevol = CachedReadOnlyProperty(_get_is_imagevol, "")

    ref_vols = CachedReadOnlyProperty(
        lambda self:\
        [i for i, isimage in enumerate(self.is_imagevol) if not isimage], "")

    image_vols = CachedReadOnlyProperty(
        lambda self:\
        [i for i, isimage in enumerate(self.is_imagevol) if isimage], "")

    nvol = CachedReadOnlyProperty(
        lambda self: self.nvol_true - len(self.ref_vols), "")

    slice_positions = CachedReadOnlyProperty(
        lambda self: 10.*asarray(self._procpar.pss), "")

    def _get_slice_gap(self):
        if self.pulse_sequence == "mp_flash3d" or self.nslice < 2: return 0.
        spos = self.slice_positions
        return ((max(spos) - min(spos) + self.thk)
               - (self.nslice*self.thk))/(self.nslice - 1)
    slice_gap = CachedReadOnlyProperty(_get_slice_gap, "")

    nslice = CachedReadOnlyProperty(
        lambda self: self.pulse_sequence in ("mp_flash3d", "box3d_v2") and \
                     self._procpar.nv2[0] or len(self.slice_positions), "")

    thk = CachedReadOnlyProperty(
        lambda self: self.pulse_sequence == "mp_flash3d" and\
                     10.*self._procpar.lpe2[0] or self._procpar.thk[0], "")

    nav_per_seg = CachedReadOnlyProperty(
        lambda self: self.n_pe%32 and 1 or 0, "")

    pe_per_seg = CachedReadOnlyProperty(lambda self: self.n_pe/self.nseg, "")

    nav_per_slice = CachedReadOnlyProperty(
        lambda self: self.nseg*self.nav_per_seg, "")

    def _get_nseg(self):
        # !!!!!! HEY BEN WHAT IS THE sparse SEQUENCE !!!!
        # Leon's "spare" sequence is really the EPI sequence with delay.
        if self.pulse_sequence in ('epi','tepi','sparse','spare'):
            return int(self.petable_name[-2])
        elif self._procpar.pslabel[0] in ('epidw', 'Vsparse') and\
          not self.spinecho:
            return int(self.petable_name[-1])
        elif self.pulse_sequence in ('epi','epidw') and self.spinecho:
            return self._procpar.nseg[0]
        elif self.pulse_sequence in ('gems', 'mp_flash3d', 'box3d_v2', 'asems'): return 1
        else: raise ValueError(
              "Could not identify sequence: %s" % (self.pulse_sequence))
    nseg = CachedReadOnlyProperty(_get_nseg, "")

    tr = CachedReadOnlyProperty(lambda self: self.nseg*self._procpar.tr[0], "")

    x0 = CachedReadOnlyProperty(lambda self: 0., "")

    y0 = CachedReadOnlyProperty(lambda self: 0., "")

    z0 = CachedReadOnlyProperty(lambda self: 0., "")

    xsize = CachedReadOnlyProperty(
        lambda self: 10.*float(self._procpar.lro[0])/self.n_fe_true, "")

    ysize = CachedReadOnlyProperty(
        lambda self: 10.*float(self._procpar.lpe[0])/self.n_pe_true, "")

    zsize = CachedReadOnlyProperty(
        lambda self: float(self.thk) + self.slice_gap, "")

    tsize = CachedReadOnlyProperty(lambda self: self.tr, "")

    datasize = CachedReadOnlyProperty(
        lambda self: self._procpar.dp[0]=="y" and 4 or 2, "")

    raw_typecode = CachedReadOnlyProperty(
        lambda self: self._procpar.dp[0]=="y" and Int32 or Int16, "")

    echo_factor = CachedReadOnlyProperty(
        lambda self: 2.0*abs(self._procpar.gro[0])\
                     *self._procpar.trise[0]/self._procpar.gmax[0], "")

    def _get_echo_time(self):
        if self.petable_name.find("alt") != -1:
            return self._procpar.te[0] - self.echo_factor - self._procpar.at[0]
        else:
            return self._procpar.te[0] - \
              floor(self.pe_per_seg)/2.0*self.echo_spacing
    echo_time = CachedReadOnlyProperty(_get_echo_time, "")

    echo_spacing = CachedReadOnlyProperty(
        lambda self: self.echo_factor + self._procpar.at[0], "")

    pe_times = CachedReadOnlyProperty(
        lambda self: asarray(
          [self.echo_time + pe*self.echo_spacing\
           for pe in range(self.pe_per_seg)]), "")

    # time between beginning of one PE scan and the next (was dwell_time)
    T_pe = CachedReadOnlyProperty(
	lambda self: getattr(self._procpar, "at_calc", (None,))[0], "")

    # this quiet_interval may need to be added to tr in some way...
    quiet_interval = CachedReadOnlyProperty(
        lambda self: self._procpar.get("dquiet", (0,))[0], "")

    #-------------------------------------------------------------------------
    def __init__(self, datadir, tr=None):
        procpar = self._procpar = ProcPar(pjoin(datadir, "procpar"))
        # allow manual override of tr value
        self._tr = tr


##############################################################################
if __name__ == "__main__":
    import pprint
    p = ProcPar( "procpar" )
    pprint.pprint( p )
