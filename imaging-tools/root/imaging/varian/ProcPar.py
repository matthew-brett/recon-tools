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
    A read-only representation of the named records found in a Varian procpar file.
    Record values can be accessed by record name in either dictionary style or object
    attribute style.  A record value is either a single element or a tuple of elements.
    Each element is either a string, int, or float.
    """

    def __init__( self, filename ):
        class foo: pass # just need a mutable object to hang the negative flag on
        state = foo()
        state.isneg = False # flag indicating when we've parse a negative sign

        # stream of each element's first two items (tokenid and token) from the raw
        # token stream returned by the tokenize module's generate_tokens function
        tokens = imap( lambda t: t[:2], generate_tokens( file( filename ).readline ) )

        # filter out negative ops (but remember them in case they come before a number)
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
                if not toks or not toks[-1][0] == tokids.NAME: raise StopIteration
                else: name = toks[-1][1]
                rest = advanceTo( tokens, tokids.NEWLINE )
                numvalues = int( tokens.next()[1] )
                values = tuple( map( cast, advanceBy( tokens, numvalues ) ) )
                return name, values
        
        self.update( dict( [pair for pair in KeyValPairs()] ) )


##############################################################################
class ProcParImageMixin (object):
    "Knows how to extract basic Image parameters from a procpar file"

    #-------------------------------------------------------------------------
    def loadParams(self, datadir):
        "Adapt procpar params into image parameters."
        procpar = self._procpar = ProcPar(pjoin(datadir, "procpar"))
        self.n_fe = procpar.np[0]
        self.n_pe = procpar.nv[0]
        self.tr = procpar.tr[0]
        self.petable_name = procpar.petable[0]
        self.orient = procpar.orient[0]
        pulse_sequence = procpar.pslabel[0]

        if hasattr(procpar, "asym_time"):
            self.asym_times = procpar.asym_time
            self.nvol_true = len(self.asym_times)
        else:
            self.asym_times = ()
            self.nvol_true = procpar.images[0]

        if procpar.get("spinecho", ("",))[0] == "y":
            if pulse_sequence == 'epidw': pulse_sequence = 'epidw_se'
            else: pulse_sequence = 'epi_se'

        # try using procpar.cntr or procpar.image to know which volumes are reference
        if hasattr(procpar, "cntr") and len(procpar.cntr) == self.nvol_true:
            is_imagevol = procpar.cntr
        elif hasattr(procpar, "image") and len(procpar.image) == self.nvol_true:
            is_imagevol = procpar.image
        else: is_imagevol = [1]*self.nvol_true
        self.ref_vols = []
        self.image_vols = []
        for i, isimage in enumerate(is_imagevol):
            if isimage: self.image_vols.append(i)
            if not isimage: self.ref_vols.append(i)
        self.nvol = self.nvol_true - len(self.ref_vols)

        # determine number of slices, thickness, and gap
        if pulse_sequence == 'mp_flash3d':
            self.nslice = procpar.nv2[0]
            self.thk = 10.*procpar.lpe2[0]
            slice_gap = 0
        else:
            slice_positions = procpar.pss
            self.nslice = len(slice_positions)
            self.thk = procpar.thk[0]
            min = 10.*slice_positions[0]
            max = 10.*slice_positions[-1]
            nslice = self.nslice
            if nslice > 1:
                slice_gap = ((max - min + self.thk) - (nslice*self.thk))/(nslice - 1)
            else:
                slice_gap = 0

        # Determine the number of navigator echoes per segment.
        self.nav_per_seg = self.n_pe%32 and 1 or 0

        # sequence-specific logic for determining pulse_sequence, petable and nseg
        # !!!!!! HEY BEN WHAT IS THE sparse SEQUENCE !!!!
        # Leon's "spare" sequence is really the EPI sequence with delay.
        if(pulse_sequence in ('epi','tepi','sparse','spare')):
            nseg = int(self.petable_name[-2])
        elif(pulse_sequence in ('epidw','Vsparse')):
            pulse_sequence = 'epidw'
            nseg = int(self.petable_name[-1])
        elif(pulse_sequence in ('epi_se','epidw_sb')):
            nseg = procpar.nseg[0]
            if self.petable_name.rfind('epidw') < 0:
                petable_name = "epi%dse%dk" % (self.n_pe, nseg)
            else:
                pulse_sequence = 'epidw'
        elif(pulse_sequence == 'asems'):
            nseg = 1
        else:
            print "Could not identify sequence: %s" % (pulse_sequence)
            sys.exit(1)

        self.nseg = nseg
        self.pulse_sequence = pulse_sequence

        # this quiet_interval may need to be added to tr in some way...
        #quiet_interval = procpar.get("dquiet", (0,))[0]
        self.n_fe_true = self.n_fe/2
        self.tr = nseg*self.tr
        self.nav_per_slice = nseg*self.nav_per_seg
        self.n_pe_true =  self.n_pe - self.nav_per_slice
        self.pe_per_seg = self.n_pe/nseg
        fov = procpar.lro[0]
        self.x0 = 0.
        self.y0 = 0.
        self.z0 = 0.
        self.xsize = 10.*float(fov)/self.n_pe_true
        self.ysize = 10.*float(fov)/self.n_fe_true
        self.zsize = float(self.thk) + slice_gap

        self.datasize, self.raw_typecode = \
          procpar.dp[0]=="y" and (4, Int32) or (2, Int16)

        # calculate echo times
        f = 2.0*abs(procpar.gro[0])*procpar.trise[0]/procpar.gmax[0]
        self.echo_spacing = f + procpar.at[0]
        if(self.petable_name.find("alt") >= 0):
            self.echo_time = procpar.te[0] - f - procpar.at[0]
        else:
            self.echo_time = procpar.te[0] - floor(self.pe_per_seg)/2.0*self.echo_spacing
        self.pe_times = asarray([self.echo_time + pe*self.echo_spacing \
                          for pe in range(self.pe_per_seg)])




##############################################################################
if __name__ == "__main__":
    import pprint
    p = ProcPar( "procpar" )
    pprint.pprint( p )
