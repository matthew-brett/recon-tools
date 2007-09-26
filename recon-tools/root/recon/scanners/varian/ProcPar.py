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
    """
    advances the parser until the next stopid (eg tokids.NEWLINE), while
    collecting tokens along the way
    """
    collection = []
    for tokid, tok in tokens:
        collection.append( (tokid, tok) )
        if tokid == stopid: break
    return collection

#-----------------------------------------------------------------------------
def advanceBy( tokens, count ):
    """
    advances the parser count times, collecting the tokens
    """
    collection = []
    for tokid, tok in tokens:
        if count < 1: break
        if tokid == tokids.NEWLINE: continue
        collection.append( (tokid, tok) )
        count -= 1
    return collection

#-----------------------------------------------------------------------------
def cast( toktup ):
    """
    returns the token as a string, integer, or float based on the token ID
    """
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
                # find the next NAME token
                toks = advanceTo( tokens, tokids.NAME )
                if not toks or not toks[-1][0] == tokids.NAME:
                    raise StopIteration
                else: name = toks[-1][1]
                # scan to the end of this line
                rest = advanceTo( tokens, tokids.NEWLINE )
                # the number of values is given by the first
                # token of the next line
                numvalues = int( tokens.next()[1] )
                # collect and cast the next N tokens
                values = tuple( map( cast, advanceBy( tokens, numvalues ) ) )
                return name, values
        
        self.update( dict( [pair for pair in KeyValPairs()] ) )


##############################################################################
class CachedReadOnlyProperty (property):
    _id = 0
    def __init__(self, getter):
        key = CachedReadOnlyProperty._id
        def cached_getter(self):
            if not hasattr(self, "_propvals"): self._propvals = {}
            return self._propvals.setdefault(key, getter(self))
        CachedReadOnlyProperty._id += 1
        property.__init__(self, fget=cached_getter, doc=getter.__doc__)

##############################################################################
class ProcParImageMixin (object):
    """
    Knows how to extract basic MR image parameters from a Varian procpar file.
    
    ProcParImageMixin keeps a local ProcPar object, and then redefines some
    ProcPar parameters into useful image properties.

    The class CachedReadOnlyProperty is used in decorator form to allow
    the methods to be accessible as read-only object attributes.
    """

    #-------------------------------------------------------------------------
    def __init__(self, datadir, vrange=None):
        self._procpar = ProcPar(pjoin(datadir, "procpar"))
        self._set_vrange(vrange)
    #-------------------------------------------------------------------------
    def _set_vrange(self, vrange):
        if vrange:
            skip = len(self.ref_vols)
            # if vrange[1] is -1 or too big, set it to nvol_true
            # if it is 0 (for unknown reason), will go to img vol 1
            vend = vrange[1] in range(self.nvol_true-skip) \
                   and vrange[1]+1+skip or self.nvol_true
            max_valid = self.nvol_true - skip
            vend = (vrange[1] >= 0 and vrange[1] < max_valid) \
                   and (vrange[1] + 1 + skip) or self.nvol_true
            # if it is past the last volume, make it one spot less
            # else, make it vrange[0]+skip (even if that == 0)
            max_valid = vend - skip
            if vrange[0] >=0 and vrange[0] < max_valid:
                vstart = vrange[0] + skip
            else:
                vstart = vend - 1
            self.vrange = range(vstart,vend)
        else:
            self.vrange = range(len(self.ref_vols), self.nvol_true)
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def n_fe(self):
        "number of read-out points"
        return self._procpar.np[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def n_fe_true(self):
        "number of complex-valued frequence encode points"
        return self.n_fe / 2
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    #nv is screwed up for half-space data, so first try nv/2 + over-scan
    def n_pe(self):
        "number of phase-encodes (scan rows)"
        return self.isepi and \
               self._procpar.fract_ky[0] + self._procpar.nv[0]/2 or \
               self._procpar.nv[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty    
    def n_pe_true(self):
        "number of phase-encodes not counting navigator scans"
        return self.n_pe - self.nav_per_slice
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def nslice(self):
        return self.ismpflash_like and self._procpar.nv2[0] or \
               len(self.slice_positions)
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def nvol(self):
        "the number of image voluems in the fid file"
        return len(self.vrange)
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def nvol_true(self):
        "the number of total volumes in the fid file"
        if self.isepi or self.isravi:
            return len(self.is_imagevol)
        if self.ismpflash_like:
            return self.mpflash_vols
        else:
            return self.acq_cycles    
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def nseg(self):
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
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def pe_per_seg(self):
        return self.n_pe / self.nseg
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    # If there is an extra (odd) line per segment, then it is navigator (??)
    def nav_per_seg(self):
        return (self.n_pe / self.nseg)%2
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def nav_per_slice(self):
        return self.nseg * self.nav_per_seg
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty    
    def isepi(self):
        return self.pulse_sequence.find('epidw') > -1 or \
               self.pulse_sequence.find('testbrs') > -1
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def isravi(self):
        return not self.isepi and self.pulse_sequence.find('epi') > -1
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def ismultislice(self):
        return self.pulse_sequence in ('gems', 'asems', 'asems_mod')
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def ismpflash_like(self):
        return self.pulse_sequence in ('box3d_slab', 'box3d_v2', 'mp_flash3d')
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def pe0(self):
        "value of the first phase encode row (not necessarily -M/2)"
        if hasattr(self._procpar, 'fract_ky'):
            return -self._procpar.fract_ky[0]
        else:
            return -self.n_pe/2        
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def orient(self):
        "name of image orientation given by procpar"
        return self._procpar.orient[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def spinecho(self):
        return self._procpar.get('spinecho', ('n',))[0] == 'y'
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    # this takes no value, so count as true if field simply exists
    def flash_converted(self):
        return self._procpar.get('flash_converted', ('foo',))[0] != 'foo'
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def acq_cycles(self):
        return self._procpar.get('acqcycles', (None,))[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    # this is used to see if a GEMS has had its DC removed already
    def n_transients(self):
        return self._procpar.get('ct', (-1,))[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def pulse_sequence(self):
        pslabel = self._procpar.pslabel[0]
        if pslabel == "Vsparse": pslabel = "epidw"
        elif self.spinecho and pslabel== 'epidw':
            pslabel = "epi%dse%dk" % (self.n_pe, self.nseg)
        return pslabel
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def petable_name(self):
        petable = self._procpar.petable[0]
        if self.isepi and self.spinecho and petable.rfind('epidw') < 0:
            petable = "epi%dse%dk" % (self.n_pe, self.nseg)
        return petable
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def asym_times(self):
        "the TE delay for asems scans, eg: (0, 0.001875)"
        return self._procpar.get('asym_time', ())
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    # seems to be that acq_cycles either equals num volumes, or
    # num slices (in the case num volumes = 1)
    def mpflash_vols(self):
        return self.acq_cycles != self.nslice and self.acq_cycles or 1
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def is_imagevol(self):
        procpar = self._procpar
        if self.isepi and hasattr(procpar, "image"):
            return procpar.image
        elif self.isravi and hasattr(procpar, "cntr"):
            return procpar.cntr
        else:
            return [1]*self.nvol_true
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    # image volumes are shown by a "1" index in the is_imagevol list
    # ref volumes are shown by numbers != 1
    def ref_vols(self):
        "a list representing which volumes are reference volumes"
        return [i for i,isimage in enumerate(self.is_imagevol) if isimage != 1]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def image_vols(self):
        "a list representing which volumes are image volumes"
        return [i for i,isimage in enumerate(self.is_imagevol)
                if (isimage == 1 and i in self.vrange)]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def slice_positions(self):
        """the slice positions relative to isocenter, in acquisition order"""
        return 10. * N.asarray(self._procpar.pss)\
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def gss(self):
        return self._procpar.gss[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    #Not exactly in procpar, but useful still
    def acq_order(self):
        "the mapping of acquisition sequence to physical slice ordering"
        return N.asarray( (self.ismpflash_like and range(self.nslice) or \
                           (range(self.nslice-1,-1,-2) +
                            range(self.nslice-2,-1,-2))) )
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def slice_gap(self):
        "gap between slice selective bandpasses"
        if self.pulse_sequence == "mp_flash3d" or self.nslice < 2:
            return 0.
        spos = self.slice_positions
        return ((max(spos) - min(spos) + self.thk)
                - (self.nslice*self.thk)) / (self.nslice - 1)
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def thk(self):
        "slice select bandpass thickness"
        return self.pulse_sequence == 'mp_flash3d' and \
               10. * self._procpar.lpe2[0] / self.nslice or \
               self._procpar.thk[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def sampstyle(self):
        "says whether samping was centric or linear"
        petable = self.petable_name
        if petable.find('cen') > -1 or petable.find('alt') > -1:
            return 'centric'
        else:
            return 'linear'
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def tr(self):
        return self.nseg * self._procpar.tr[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def phi(self):
        return self._procpar.phi[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def psi(self):
        return self._procpar.psi[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def theta(self):
        return self._procpar.theta[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def delT(self):
        "sample period"
        return 1. / self._procpar.sw[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def dFE(self):
        "frequency-encode resolution"
        return (10. * self._procpar.lro[0]) / self.n_fe_true
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def dPE(self):
        "phase-encode resolution"
        return (10. * self._procpar.lpe[0]) / self.n_pe_true
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def dSL(self):
        "slice resolution"
        return float(self.thk + self.slice_gap)
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def datasize(self):
        "number of bytes per sample"
        return self._procpar.dp[0] == 'y' and 4 or 2
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def raw_dtype(self):
        return self._procpar.dp[0] == 'y' and N.int32 or N.int16
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def echo_factor(self):
        return 2.0*abs(self._procpar.gro[0]) * \
               self._procpar.trise[0] / self._procpar.gmax[0]    
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def echo_time(self):
        if self.petable_name.find("alt") != -1:
            return self._procpar.te[0] - self.echo_factor - self._procpar.at[0]
        else:
            return self._procpar.te[0] - \
                   N.floor(self.pe_per_seg)/2.0*self.echo_spacing        
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def echo_spacing(self):
        return self.echo_factor + self._procpar.at[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def pe_times(self):
        return N.asarray([self.echo_time + pe*self.echo_spacing
                          for pe in range(self.pe_per_seg)])
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    # time between beginning of one PE scan and the next (was dwell_time)
    def T_pe(self):
        return self._procpar.get('at_calc', (None,))[0]
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    # this quiet_interval may need to be added to tr in some way...
    def quiet_interval(self):
        return self._procpar.get('dquiet', (0,))[0]


##############################################################################
if __name__ == "__main__":
    import pprint
    p = ProcPar( "procpar" )
    pprint.pprint( p )
