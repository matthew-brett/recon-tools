import glob
import sys
import token as tokids
import struct
from os import path
from itertools import imap, ifilter
from tokenize import generate_tokens

import numpy as N
from recon.util import qmult, eulerRot, Quaternion, reverse, normalize_angle, \
     checkerline
from recon.scanners import ScannerImage, _HeaderBase, _BitMaskedWord, \
     tablib, CachedReadOnlyProperty
from recon.imageio import ReconImage
from recon.analyze import canonical_orient

#-----------------------------------------------------------------------------
def getPulseSeq(datadir):
    pp = ProcParImageMixin(datadir)
    ps = pp.pulse_sequence

    # Some sequences might require different treatment if conditions are true.
    # if mpflash has the flash_converted flag, then use 2DFFT instead of 3D
    #if ps == "mp_flash3d":
    if pp.ismpflash_like:
        flag = (pp.flash_converted and 1 or 0, )
    # ... add more flags as necessary
    else:
        flag = None
    return (ps, flag)

#-----------------------------------------------------------------------------
def complex_fromstring(data, numtype):
    if sys.byteorder == "little":
        return N.fromstring(
            N.fromstring(data,numtype).byteswap().astype(N.float32).tostring(),
            N.complex64)
    else:
        return N.fromstring(
	    N.fromstring(data,numtype).astype(N.float32).tostring(),
            N.complex64)

def cast_to_complex(cplx_float, cplx_integer, bias=None):
    if bias is not None:
        cplx_float.real[:] = cplx_integer['real'] - bias.real
        cplx_float.imag[:] = cplx_integer['imag'] - bias.imag
    else:
        cplx_float.real[:] = cplx_integer['real']
        cplx_float.imag[:] = cplx_integer['imag']

##############################################################################
########################   PROCPAR PARSING CLASSES   #########################
##############################################################################
class ProcParImageMixin (object):
    """
    Knows how to extract basic MR image parameters from a Varian procpar file.
    
    ProcParImageMixin keeps a local ParsedProcPar object, which is a giant
    dictionary full of procpar values, and using various logic, remaps that
    information to useful image properties.

    Since most properties require some logic to compute and may also look back
    to other computed values, the class CachedReadOnlyProperty is used to
    hide the function call, and let the return values be accessible as
    read-only object attributes. This wrapping is made easier on the eye by
    using the @CachedReadOnlyProperty decorator on each method-value.
    
    This class is designed to be extended by FidImage, which holds the
    actual image data from the FID file.
    """

    #-------------------------------------------------------------------------
    def __init__(self, datadir, vrange=None):
        # the ProcPar class is a giant dictionary full of procpar values
        self._procpar = ParsedProcPar(path.join(datadir, "procpar"))
        self._set_vrange(vrange)
    #-------------------------------------------------------------------------
    def _set_vrange(self, vrange):
        if vrange:
            skip = len(self.ref_vols)
            # if vrange[1] is -1 or too big, set it to n_vol_true
            # if it is 0 (for unknown reason), will go to img vol 1
            vend = vrange[1] in range(self.n_vol_true-skip) \
                   and vrange[1]+1+skip or self.n_vol_true
            max_valid = self.n_vol_true - skip
            vend = (vrange[1] >= 0 and vrange[1] < max_valid) \
                   and (vrange[1] + 1 + skip) or self.n_vol_true
            # if it is past the last volume, make it one spot less
            # else, make it vrange[0]+skip (even if that == 0)
            max_valid = vend - skip
            if vrange[0] >=0 and vrange[0] < max_valid:
                vstart = vrange[0] + skip
            else:
                vstart = vend - 1
            self.vrange = range(vstart,vend)
        else:
            self.vrange = range(len(self.ref_vols), self.n_vol_true)
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
    def n_slice(self):
        return self.ismpflash_like and self._procpar.nv2[0] or \
               len(self.slice_positions)
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def n_vol(self):
        "the number of image voluems in the fid file"
        return len(self.vrange)
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def n_vol_true(self):
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
            try:
                nseg = int(self.petable_name[-2])
            except:
                nseg = 1
            return nseg
        elif self.pulse_sequence in ('epidw', 'Vsparse','testbrs2') and\
          not self.spinecho:
            try:
                nseg = int(self.petable_name[-1])
            except:
                nseg = 1
            return nseg
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
        pslabel = self.pulse_sequence
        return pslabel.find('epidw') > -1 or pslabel.find('testbrs') > -1
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def isravi(self):
        pslabel = self.pulse_sequence        
        return not self.isepi and pslabel.find('epi') > -1
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def ismultislice(self):
        pslabel = self.pulse_sequence        
        return pslabel in ('gems', 'asems', 'asems_mod')
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def ismpflash_like(self):
        pslabel = self.pulse_sequence        
        return pslabel in ('box3d_slab', 'box3d_v2', 'mp_flash3d')
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def pe0(self):
        "value of the first phase encode row (not necessarily -M/2)"
        if hasattr(self._procpar, 'fract_ky'):
            return -self._procpar.fract_ky[0]
        else:
            return -self.n_pe_true/2        
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
        return self._procpar.pslabel[0]
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
        return self.acq_cycles != self.n_slice and self.acq_cycles or 1
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def is_imagevol(self):
        procpar = self._procpar
        if self.isepi and hasattr(procpar, "image"):
            return procpar.image
        elif self.isravi and hasattr(procpar, "cntr"):
            return procpar.cntr
        else:
            return [1]*self.n_vol_true
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
        return N.asarray( (self.ismpflash_like and range(self.n_slice) or \
                           (range(self.n_slice-1,-1,-2) +
                            range(self.n_slice-2,-1,-2))) )
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def slice_gap(self):
        "gap between slice selective bandpasses"
        #if self.pulse_sequence == "mp_flash3d" or self.n_slice < 2:
        if self.ismpflash_like or self.n_slice < 2:
            return 0.
        spos = self.slice_positions
        return ((max(spos) - min(spos) + self.thk)
                - (self.n_slice*self.thk)) / (self.n_slice - 1)
    #-------------------------------------------------------------------------
    @CachedReadOnlyProperty
    def thk(self):
        "slice select bandpass thickness"
##         return self.pulse_sequence == 'mp_flash3d' and \
        return self.ismpflash_like and \
               10. * self._procpar.lpe2[0] / self.n_slice or \
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
class ParsedProcPar (dict):
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

        # These methods help navigate the file based on tokens
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
        
        def cast( toktup ):
            """
            casts token as a string, integer, or float based on the token ID
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

    # overloading these methods makes this class read-only, and allows
    # items to be accessed as attributes
    def __setitem__( self, key, val ):
        raise AttributeError, 'object is read-only'

    def __setattr__( self, attname, val ):
        raise AttributeError, 'object is read-only'

    def __getattr__( self, attname ):
        try:
            return self[attname]
        except KeyError:
            raise AttributeError, attname

##############################################################################
class FidImage (ScannerImage, ProcParImageMixin):

    """
    FidImage loads a FID format file from a Varian system. It knows how to
    reassemble volume data from various scanner sequences. A FidImage is both
    a ScannerImage, giving it a clean, predictable interface, and a
    ProcParImageMixin, giving it access to the parsed data from
    the procpar file. Additionally, FidImage loads more specific scan data
    such as reference and navigator data. FidImage calls on the FidFile class
    to interface with the actual data on-disk.
    """
    #-------------------------------------------------------------------------
    def __init__(self, filestem, vrange=None, target_dtype=N.complex64):
        datadir = filestem+".fid"
        self.path = datadir
        ProcParImageMixin.__init__(self, datadir, vrange=vrange)
        self.initializeData()
        self.loadData(datadir)
        self.realizeOrientation()
        # scanner image will init ReconImage, which will set the data dims
        if not self.vrange:
            # this is a badly behaved FidImage.. might be a BRS
            return
        ScannerImage.__init__(self)
        ref_path = filestem.replace("_data", "_ref_2")
        if ref_path is not filestem and path.exists(ref_path+".fid"):
            ref_fid = FidImage(ref_path)
            self.ref_data = N.array([self.ref_data[0], ref_fid.ref_data[0]])
        
    #-------------------------------------------------------------------------
    def logParams(self):
        "Report scan parameters to stdout."
        print "Phase encode table: ", self.petable_name
        print "Pulse sequence: %s" % self.pulse_sequence
        print "Spinecho: %s" % self.spinecho
        print "Number of volumes: %d" % self.n_vol
        print "Number of slices: %d" % self.n_slice
        print "Number of segments: %d" % self.nseg
        print "Number of navigator echoes per segment: %d" % self.nav_per_seg
        print "Number of phase encodes per slice (including any navigators echoes): %d" % self.n_pe
        print "Number of frequency encodes: %d" % self.n_fe_true
        print "Raw precision (bytes): ", self.datasize
        print "Number of reference volumes: %d" % len(self.ref_vols)
        print "Orientation: %s" % self.orient
        print "Pixel size (phase-encode direction): %7.2f" % self.dPE 
        print "Pixel size (frequency-encode direction): %7.2f" % self.dFE
        print "Slice thickness: %7.2f" % self.dSL

    #-------------------------------------------------------------------------
    def initializeData(self):
        "Allocate data matrices." # IF NEEDED! look at petable or something
        nrefs = len(self.ref_vols)
        self.data = N.zeros((self.n_vol, self.n_slice,
                             self.n_pe_true, self.n_fe_true), N.complex64)
        self.nav_data = N.zeros((self.n_vol, self.n_slice,
                                 self.nav_per_slice, self.n_fe_true),
                                N.complex64)
        self.ref_data = N.zeros((nrefs, self.n_slice,
                                 self.n_pe_true, self.n_fe_true), N.complex64)
        self.ref_nav_data = N.zeros((nrefs, self.n_slice,
                                     self.nav_per_slice, self.n_fe_true),
                                    N.complex64)
        
    #-------------------------------------------------------------------------
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
        # the n2 vector is always this
        n2 = N.arange(M) + pe0        
        if self.sampstyle == "centric":
            if self.nseg > 2:
                raise NotImplementedError("centric sampling not implemented for nseg > 2")
            a = checkerline(M)
            a[:M/2] *= -1
            # offset the pe ordering by pe0, which may or may not be -M/2
            b = N.arange(M) + pe0
            b[:-pe0] = abs(b[:-pe0] + 1)
        else:
            a = N.empty(M, N.int32)
            for n in range(self.nseg):
                a[n:M:2*self.nseg] = 1
                a[n+self.nseg:M:2*self.nseg] = -1
            b = N.floor((N.arange(float(M))+pe0)/float(self.nseg)).astype(N.int32)
        aref = a
        return (a, b, n2, aref)
    
    #-------------------------------------------------------------------------
    def seg_slicing(self, n):
        """
        This method returns a list of indices corresponding to segment n,
        to be used in separating segments from recombined kspace
        """
        if n >= self.nseg:
            return self.seg_slicing(n-1)
        pe_per_seg = self.n_pe_true/self.nseg
        seg_group = range( n*pe_per_seg, (n+1)*pe_per_seg )
        # find the current row indices of this seg group
        seg_rows = N.array([(n==self.petable).nonzero()[0][0]
                            for n in seg_group])
        # this is listed in acq order, and might be backwards (eg centric mode)
        seg_rows.sort()
        return seg_rows
    
    #-------------------------------------------------------------------------
    def realizeOrientation(self):
        "Set up the orientation transform defined in the procpar"
        # Varian data is layed out with this improper rotation from
        # neurological oriented space:
        # from scanner to image-> map -y -> +y
        # (the object then lies [+X, +Y, +Z] = [P, R, S] inside the scanner,
        #  so account for this rotation last)
        Qs_mat = N.asarray([[ 1., 0., 0.],
                            [ 0.,-1., 0.],
                            [ 0., 0., 1.],])
        Qr_mat = eulerRot(phi=N.pi/2)
        Qscanner = Quaternion(M=Qs_mat)
        Qrot = Quaternion(M=Qr_mat)

        phi,psi,theta = self.phi, self.psi, self.theta
        ang = N.array([self.phi, self.psi, self.theta])
        ang = normalize_angle(N.pi/2 * \
                              ((ang + N.sign(ang)*45.)/90).astype(N.int32))
        phi, psi, theta = ang.tolist()
        # find out the (closest) image plane
        if theta == 0 and psi == 0:
            self.image_plane = "axial"
        elif abs(theta) == N.pi/2 and psi == 0:
            self.image_plane = "coronal"
        elif abs(theta) == N.pi/2 and abs(psi) == N.pi/2:
            self.image_plane = "sagittal"
        else:
            self.image_plane = "oblique"
        # Varian system [R A S] vector seems to be [Y -X Z], and
        # the order of rotation in that space is
        # 1) theta degrees clockwise around Y
        # 2) psi degrees around X
        # 3) phi degrees around Z
        # The steps to rotate the reconstructed data are:
        # 1) flip Y to put into a right-handed system
        # 2) rotate in the reverse order as the Varian Euler rotation
        # 3) make a final pi/2 rotation around Z to make [R A S] = [X Y Z]
        Qobl = Quaternion(M=N.dot(eulerRot(psi=theta),
                                  N.dot(eulerRot(theta=psi),
                                        eulerRot(phi=phi))))
        
        self.orientation_xform = qmult(Qrot, qmult(Qobl,Qscanner))
        # find out if this orientation has a "name" in ANALYZE format
        self.orientation = canonical_orient(self.orientation_xform.tomatrix())
            
    #-------------------------------------------------------------------------
    def _load_petable(self):
        """
        Read the phase-encode table and organize it into arrays which map
        k-space line number (recon_epi convention) to the acquisition order.
        These arrays will be used to read the data into the recon_epi as
        slices of k-space data.  Assumes navigator echo is aquired at the
        beginning of each segment.
        """
        nav_per_seg = self.nav_per_seg
        nseg = self.nseg
        n_pe_true = self.n_pe_true
        table_filename = path.join(tablib, self.petable_name)
        # line is going to look like [0 2 4 ... 32 1 3 5 ... 63]
        # (eg: acquisition scheme for interleaved data)
        # want to return a list of integers that indexes ordered numbers
        # from the above list: [0 32 1 33 2 34 ... ]
        line = file(table_filename).readline().split()
        line = N.array([int(i) for i in line])
        all_pe = N.arange(n_pe_true)
        self.petable = N.array([(line==p).nonzero()[0][0] for p in all_pe])

    #-------------------------------------------------------------------------
    #### A note for using FidFiles in all volume readers ####
    #
    # FidFiles and DataBlocks:
    # Varian raw data comes in a "fid" file, with a (hdr, data) style block
    # layout. Each fid file has nblocks number of blocks, and each block has
    # ntraces number of traces (read-outs). FidFile objects are smartly
    # memory mapped "fid" files. You can index into a FidFile like an array,
    # and it will yield a (hdr, data) pair. These are the known block layouts
    # of the fids:
    #-------------------------------------------------------------------------
    def _read_compressed_volume(self, fidfile, vol):
        """
        Reads one volume from a compressed FID file.
        @return: block of data with shape (n_slice*n_pe, n_fe_true)
        """
        (hdr, data) = fidfile[vol]
        bias = N.array([hdr.lvl + 1.j*hdr.tlt], N.complex64)
        volume = N.empty((self.n_slice*self.n_pe*self.n_fe_true), N.complex64)
        cast_to_complex(volume[:], data, bias=bias)
        volume.shape = (self.n_slice, self.n_pe, self.n_fe_true)
        return volume
    
    #-------------------------------------------------------------------------
    def _read_uncompressed_volume(self, fidfile, vol):
        """
        Reads one volume from an uncompressed FID file.
        @return: block of data with shape (n_slice*n_pe, n_fe_true)
        """        
        volume = N.empty((self.n_slice, self.n_pe*self.n_fe_true), N.complex64)
        # oddly enough, it is much faster to get one block at a time!!
        for sl_num, sl in enumerate(volume):
            (hdr, data) = fidfile[self.n_slice*vol + sl_num]
            bias = N.array([hdr.lvl + 1.j*hdr.tlt], N.complex64)
            cast_to_complex(sl, data, bias=bias)
        volume.shape = (self.n_slice, self.n_pe, self.n_fe_true)
        return volume

    #-------------------------------------------------------------------------
    ### This format is quite poorly understood!
    def _read_epi2fid_volume(self, fidfile, vol):
        """
        Reads one volume from an epi2fid FID file.
        @return: block of data with shape (n_slice*n_pe, n_fe_true)
        """
        # procpar indicates navigator lines, but none are read in? huh
        volume = N.empty(
            (self.n_slice, self.nseg, self.pe_per_seg, self.n_fe_true),N.complex64)
        pe_true_per_seg = self.pe_per_seg - self.nav_per_seg
        for seg in range(self.nseg):
            for sl in range(self.n_slice):
                for pe in range(pe_true_per_seg):
                    (hdr, data) = fidfile[self.n_slice*(self.n_vol_true*(seg*pe_true_per_seg + pe) + vol) + sl]
                    volume[sl,seg,self.nav_per_seg+pe].real[:] = data['real']
                    volume[sl,seg,self.nav_per_seg+pe].imag[:] = imag['real']
        return N.reshape(volume, (self.n_slice, self.n_pe, self.n_fe_true))

    #-------------------------------------------------------------------------
    def _read_asems_nccnn_volume(self, fidfile, vol):
        """
        Reads one volume from an asems_nccnn FID file.
        @return: block of data with shape (n_slice*n_pe, n_fe_true)
        """
        volume = N.empty((self.n_pe, self.n_slice*self.n_fe_true), N.complex64)
        # There's a block for each phase encode, each block has n_slice
        # read-outs. For the 1st volume of a 2 volume data set, the block
        # slicing is [0, 2, 4, ..., 2*(n_pe-1)]
        vol_sl = slice(vol, self.n_vol_true*self.n_pe, self.n_vol_true)
        (hdr, data) = fidfile[vol_sl]
        bias = N.array( [[h.lvl, h.tlt] for h in hdr], N.float32).transpose()
        bias.shape = (2,self.n_pe,1)
        if self.pulse_sequence == "gems" and self.n_transients>1:
            volume.real[:] = data['real']
            volume.imag[:] = data['imag']
        else:
            volume.real[:] = data['real'] - bias[0]
            volume.imag[:] = data['imag'] - bias[1]

        volume.shape = (self.n_pe, self.n_slice, self.n_fe_true)
        return volume.transpose((1,0,2))

    #-------------------------------------------------------------------------
    def _read_asems_nscsn_volume(self, fidfile, vol):
        """
        Reads one volume from an asems_nccnn FID file.
        @return: block of data with shape (n_slice*n_pe, n_fe_true)
        """
        volume = N.empty( (self.n_slice*self.n_pe, self.n_fe_true), N.complex64)
        # There are n_vol*n_slice*n_pe blocks, each with one single trace
        # For 3 volumes shaped (4,6,n_fe), indexing for vol=0, for example is:
        # [0,1,2,3,12,13,14,15,24,25,26,27,36,37,38,39,...]
        # this will be computed by a list [0,1,...,n_slice]*n_pe + offset_array
        offset = N.repeat(N.arange(self.n_pe), self.n_slice) * \
                 (self.n_vol_true*self.n_slice) + vol*self.n_slice
        s_idx = N.array( range(self.n_slice) * self.n_pe )
        vol_sl = s_idx + offset
        (hdr, data) = fidfile[vol_sl]
        bias = N.array( [[h.lvl, h.tlt] for h in hdr], N.float32).transpose()
        bias.shape = (2,self.n_slice*self.n_pe,1)
        if self.pulse_sequence=='mp_flash3d' and self.flash_converted:
            volume.real[:] = data.real[:]
            volume.imag[:] = data.imag[:]
        else:
            volume.real[:] = data.real[:] - bias[0]
            volume.imag[:] = data.imag[:] - bias[1]
        volume.shape = (self.n_slice, self.n_pe, self.n_fe_true)
        return volume
    
    #-------------------------------------------------------------------------
    def _get_fidformat(self, fidfile):
        """
        Determine fid format from the number of blocks per volumen and the
        number of traces per block.  Known formats are:
          compressed
          uncompressed
          epi2fid
          asems_ncsnn
          asems_nccnn
        """
        n_pe = self.n_pe
        n_pe_true = self.n_pe_true
        n_slice =  self.n_slice
        n_vol_true = self.n_vol_true
        nblocks = fidfile.nblocks
        ntraces = fidfile.ntraces

        # compressed format has one block per volume
        if nblocks == n_vol_true and ntraces == n_slice*n_pe:
            return "compressed"

        # uncompressed format has one block per slice
        elif nblocks == n_vol_true*n_slice and ntraces == n_pe:
            return "uncompressed"

        # epi2fid format has one block per phase encode (but in a weird order!)
        #elif nblocks == n_vol_true*n_slice*n_pe_true and ntraces == 1:
        #    return "epi2fid"

        # asems_nccnn format has one block per phase-encode per volume
        # this is the normal format for ASEMS and GEMS
        elif nblocks == n_vol_true*n_pe and ntraces == n_slice:
            return "asems_nccnn"

        # asems_nscsn format has one block per phase-encode per segment
        # volume per slice. This can happen with processed MP_FLASH
        elif nblocks == n_vol_true*n_slice*n_pe and ntraces == 1:
            return "asems_nscsn"

        else:
            raise "unrecognized fid format, (nblocks, ntraces) = (%d,%d)"%\
                  (nblocks, ntraces)

    #-------------------------------------------------------------------------
    def loadData(self, datadir):
        """
        This method reads the data from a fid file into following VarianData
        attributes: 

        data: A rank 4 array containing time-domain data. This array is 
          dimensioned as data(n_vol,n_slice,n_pe_true,n_fe_true) where n_vol 
          is the number of volumes, n_slice is the number of slices per volume,
          n_pe_true is the number of phase-encode lines and n_fe_true is the
          number read-out points.

        nav_data: A rank 4 array containing time-domain data for the navigator
          echoes of the image data. This array is dimensioned as 
          nav_data(n_vol,n_slice,nav_per_slice,n_fe_true).

        ref_data: A rank 4 array containing time-domain reference scan data 
          (phase-encode gradients are kept at zero). This array is dimensioned 
          as ref_data(numrefs, n_slice,n_pe_true,n_fe_true). 

        ref_nav_data: A rank 4 array containing time-domain data for the 
          navigator echoes of the reference scan data which is dimensioned as 
          ref_nav_data(numrefs, n_slice,nav_per_slice,n_fe_true).
        """
        n_pe_true = self.n_pe_true
        n_pe = self.n_pe
        nav_per_seg = self.nav_per_seg
        n_fe = self.n_fe  
        n_fe_true = self.n_fe_true
        pulse_sequence = self.pulse_sequence
        numrefs = len(self.ref_vols)

        # open fid file
        fidfile = FidFile(path.join(datadir, "fid")) 

        # determine fid format
        fidformat = self._get_fidformat(fidfile)
        print "fidformat=",fidformat

        # choose volume reading method based on fid format
        volreader = {
          "compressed":   self._read_compressed_volume,
          "uncompressed": self._read_uncompressed_volume,
          #"epi2fid":      self._read_epi2fid_volume,
          "asems_nccnn":  self._read_asems_nccnn_volume,
          "asems_nscsn":  self._read_asems_nscsn_volume
        }[fidformat]

        time_reverse = pulse_sequence in ("epi","epidw","testbrs2") and \
                       fidformat == "compressed"

        #time_rev = n_fe_true - 1 - N.arange(n_fe_true)
        if time_reverse: print "time reversing"

        # determine if phase encodes need reordering 
        needs_pe_reordering = self.nseg > 1
        
        # load phase encode table
        if needs_pe_reordering: self._load_petable()

        for vnum, vol in enumerate(self.ref_vols+self.image_vols):
            # read the next image volume
            volume = volreader(fidfile, vol)

            (data,navdata,vidx) = vol in self.ref_vols and \
                                  (self.ref_data,self.ref_nav_data,vnum) or \
                                  (self.data,self.nav_data,vnum-numrefs)


            # strip off data into appropriate arrays
            # if nav_per_seg, navigator line is the 1st in a seg
            if nav_per_seg:
                nav_lines = range(0, n_pe, n_pe/self.nseg)
                vol_lines = range(0, n_pe_true)
                for nav_pt in nav_lines: vol_lines.remove(nav_pt)
                navdata[vidx] = volume[:,nav_lines,:]
                voltmp = volume[:,vol_lines,:].copy()
                volume = voltmp

            if time_reverse:
                revtmp = volume[:,1::2,::-1].copy()
                volume[:,1::2,:] = revtmp
            
            if needs_pe_reordering:
                data[vidx] = N.take(volume, self.petable, axis=-2)
            else:
                data[vidx] = volume

        #del fidfile


##############################################################################
###########################   FILE LEVEL CLASSES   ###########################
##############################################################################
class _SharedHeaderStatus (_BitMaskedWord):
    """
    This class digests the first six bits of status word that are common
    to the FID main header and block headers. Each flag is a "property"
    object whose getter returns a Boolean.
    """
    S_DATA = property( lambda self: self._word&0x1 !=0,
        doc="0=no data, 1=data" )
    S_SPEC = property( lambda self: self._word&0x2 !=0,
        doc="0=FID, 1=spectrum" )
    S_32 = property( lambda self: self._word&0x4 !=0,
        doc="0=16-bit int, 1=32-bit int, (only used when S_FLOAT=0" )
    S_FLOAT = property( lambda self: self._word&0x8 !=0,
        doc="0=integer, 1=floating point" )
    S_COMPLEX = property( lambda self: self._word&0x10 !=0,
        doc="0=real, 1=complex" )
    S_HYPERCOMPLEX = property( lambda self: self._word&0x20 !=0,
        doc="1=hypercomplex" )

##############################################################################
class HypercomplexBlockHeader (_HeaderBase):
    "This class represents the extra header that might be present at a block."
    HEADER_FMT = ">hhhhlffff"
    HEADER_FIELD_NAMES = (
        "s_spare1",     # spare short word
        "status",       # status word for block header
        "s_spare2",     # spare short word
        "s_spare3",     # spare short word
        "l_spare1",     # spare long word
        "lpval1",       # 2D-f2 left phase
        "rpval1",       # 2D-f2 right phase
        "f_spare1",     # spare float word
        "f_spare2"      # spare float word
    )

    #-------------------------------------------------------------------------
    class _Status (_SharedHeaderStatus):
        "knows how to digest the hypercomplex block header status word"
        U_HYPERCOMPLEX = property( lambda self: self._word&0x2 !=0,
            doc="1=hypercomplex block structure" )

    #-------------------------------------------------------------------------
    def __init__( self, hdr ):
        if issubclass(type(hdr), file):
            _HeaderBase.__init__(self, hdr.read(self.header_size))
        else:
            _HeaderBase.__init__(self, hdr)
        self.status = self._Status( self.status )

##############################################################################
class BlockHeader (_HeaderBase):
    "This class represents the basic block of the FID file."
    HEADER_FMT = ">hhhhlffff"
    HEADER_FIELD_NAMES = (
        "scale",        # scaling factor
        "status",       # status of data in block
        "index",        # block index
        "mode",         # mode of data in block
        "ctcount",      # ct value for FID
        "lpval",        # f2 (2D-f1) left phase in phasefile
        "rpval",        # f2 (2D-f1) right phase in phasefile
        "lvl",          # level drift correction
        "tlt"           # tilt drift correction
    )

    #-------------------------------------------------------------------------
    class _Status (_SharedHeaderStatus):
        "knows how to digest remaining bits of the block header status word"
        MORE_BLOCKS = property( lambda self: self._word&0x80 !=0,
            doc="0=absent, 1=present" )
        NP_CMPLX = property( lambda self: self._word&0x100 !=0,
            doc="0=real, 1=complex" )
        NF_CMPLX = property( lambda self: self._word&0x200 !=0,
            doc="0=real, 1=complex" )
        NI_CMPLX = property( lambda self: self._word&0x400 !=0,
            doc="0=real, 1=complex" )
        NI2_CMPLX = property( lambda self: self._word&0x800 !=0,
            doc="0=real, 1=complex" )

    #-------------------------------------------------------------------------
    class _Mode (_BitMaskedWord):
        "knows how to interpret each bit of the block header mode word"
        NP_PHMODE = property( lambda self: self._word&0x1!=0)
        NP_AVMODE = property( lambda self: self._word&0x2 !=0)
        NP_PWRMODE = property( lambda self: self._word&0x4 !=0)
        NF_PHMODE = property( lambda self: self._word&0x10!=0)
        NF_AVMODE = property( lambda self: self._word&0x20 !=0)
        NF_PWRMODE = property( lambda self: self._word&0x40 !=0)
        NI_PHMODE = property( lambda self: self._word&0x100 !=0)
        NI_AVMODE = property( lambda self: self._word&0x200 !=0)
        NI_PWRMODE = property( lambda self: self._word&0x400 !=0)
        NI2_PHMODE = property( lambda self: self._word&0x1000 !=0)
        NI2_AVMODE = property( lambda self: self._word&0x2000 !=0)
        NI2_PWRMODE = property( lambda self: self._word&0x4000 !=0)

    #-------------------------------------------------------------------------
    def __init__( self, hdr, nheaders ):
        if issubclass(type(hdr), file):
            _HeaderBase.__init__(self, hdr.read(self.header_size))
            if nheaders == 2:
                self.hyperheader = HypercomplexBlockHeader( hdr )
        else:
            _HeaderBase.__init__(self, hdr[:self.header_size])
            if nheaders == 2:
                self.hyperheader = HypercomplexBlockHeader(hdr[self.header_size:])
        
        self.status = self._Status( self.status )
        self.mode = self._Mode( self.mode )



##############################################################################
class FidFile (_HeaderBase):
    """
    This class extends the basic HeaderBase by yielding DataBlocks. It is
    also a "file" object, so that it can seek into itself to find the
    location of the DataBlock in the file.
    """

    HEADER_FMT = ">llllllhhl"
    HEADER_FIELD_NAMES = (
        "nblocks",      # number of blocks in file
        "ntraces",      # number of traces per block
        "np",           # number of elements per trace
        "ebytes",       # number of bytes per element
        "tbytes",       # number of bytes per trace
        "bbytes",       # number of bytes per block
        "vers_id",      # software version, file_id status bits
        "status",       # status of whole file
        "nheaders"     # number of block header per block
    )

    #-------------------------------------------------------------------------
    class FileHeaderStatus (_SharedHeaderStatus):
        "knows how to digest remaining bits of the file header status word"
        S_ACQPAR = property( lambda self: self._word&0x80 !=0,
            doc="0=not Acqpar, 1=Acqpar" )
        S_SECND = property( lambda self: self._word&0x100 !=0,
            doc="first FT, 1=second FT" )
        S_TRANSF = property( lambda self: self._word&0x200 !=0,
            doc="0=regular, 1=transposed" )
        S_NP = property( lambda self: self._word&0x800 !=0,
            doc="1=np dimension is active" )
        S_NF = property( lambda self: self._word&0x1000 !=0,
            doc="1=nf dimension is active" )
        S_NI = property( lambda self: self._word&0x2000 !=0,
            doc="1=ni dimension is active" )
        S_NI2 = property( lambda self: self._word&0x4000 !=0,
            doc="1=ni2 dimension is active" )

    #-------------------------------------------------------------------------
    def __init__( self, name ):
        "always open read-only"
        #file.__init__( self,  name )
        f = open(name)
        _HeaderBase.__init__(self, f.read(self.header_size))
        f.close()
        # convert status byte into a FileHeaderStatus object
        self.status = self.FileHeaderStatus( self.status )

        
        block_data_size = self.ntraces*self.tbytes
        # number of complex elements
        num_elem = block_data_size/(self.ebytes*2)
        dt = self.ebytes == 2 and N.int16 or N.int32
        if self.ebytes == 2:
            elem_dtype = N.dtype(('>u4',
                                  {'real':('>i2',0), 'imag':('>i2',2)}))
        else:
            elem_dtype = N.dtype(('>S8',
                                  {'real':('>i4',0), 'imag':('>i4',4)}))
        
        block_header_size = self.bbytes - block_data_size
        #hdr_dtype = N.dtype('S%d'%block_header_size)
        hdr_dtype = N.dtype('>V%d'%block_header_size)
        blk_dtype = N.dtype((elem_dtype, num_elem))
        #blk_dtype = N.dtype((elem_dtype, (self.ntraces,num_elem/self.ntraces)))
        dat_dtype = N.dtype({'names':['hdr','data'],
                             'formats':[hdr_dtype,blk_dtype]})

        self.mmap = N.memmap(name, dtype=dat_dtype, offset=self.header_size,
                             shape=(self.nblocks,), mode='r')

        
      
    #-------------------------------------------------------------------------
    def __iter__(self):
        "yields blocks in FidFile"
        for bnum in xrange(self.nblocks):
            yield self.mmap[b]

    def __getitem__(self, slicer):
        "gets (hdr, data) pairs according to slice spec"
        d = self.mmap[slicer]
        hdata = d['hdr']
        if type(hdata) is not N.void:
            hdrs = []
            for h in hdata:
                hdrs.append(BlockHeader(str(h), self.nheaders))
        else:
            hdrs = BlockHeader(str(hdata), self.nheaders)
        return (hdrs, d['data'])

    def __setitem__(self, slicer, item):
        raise IOError("this is a read-only memmap")

##############################################################################
###########################   FDF IMAGE CLASSES   ############################
##############################################################################
def slicefilename(slicenum): return "image%04d.fdf"%slicenum
#-----------------------------------------------------------------------------
def string_valuator(strstr):
    return strstr.replace('"', "")
##############################################################################
class FDFImage (ReconImage, ProcParImageMixin):

    #-------------------------------------------------------------------------
    def __init__(self, datadir):
        self.datadir = datadir
        ProcParImageMixin.__init__(self, datadir)
        self.loadData()
 
    #-------------------------------------------------------------------------
    def loadDimSizes(self):
        "@return (isize, jsize, ksize)"
        if not (self.image_vols or self.kdim): return (0.,0.,0.)
        header = FDFHeader(file(path.join(self.datadir, slicefilename(1))))
        x, y, z = header.roi
        xpix, ypix = header.matrix
        return ( x/xpix, y/ypix, z )

    #-------------------------------------------------------------------------
    def loadData(self):
        volumes = []
        self.kdim = len(self._procpar.pss)
        for volnum in self.image_vols:
            slices = []
            for slicenum in range(self.kdim):
                filename = slicefilename(volnum*self.kdim + slicenum + 1)
                slices.append(FDFFile(path.join(self.datadir, filename)).data)
            volumes.append(N.asarray(slices))
        self.setData(N.asarray(volumes))

    #-------------------------------------------------------------------------
    def save(self, outputdir): self.writeImage(path.join(outputdir, "image"))

#-----------------------------------------------------------------------------
def string_valuator(strstr):
    return strstr.replace('"', "")

##############################################################################
class FDFHeader (object):

    #-------------------------------------------------------------------------
    def __init__(self, infile):
        infile.readline() # magic string
        lineno = 1
        while True:
            lineno += 1
            line = infile.readline().strip()
            if not line: break

            # extract data type, variable name, and value from input line
            itemtype, therest = line.split(" ", 1)
            name, value = therest.split("=", 1)
            name = name.strip()
            value = value.strip()
            if name[-2:] == "[]":
                name = name[:-2]
                islist = True
            else: islist = False

            # get rid of unused syntactic elements
            name = name.replace("*", "")
            value = value.replace(";", "")
            value = value.replace("{", "")
            value = value.replace("}", "")

            # determine which valuator to use based on data type
            item_valuator = {
                "int":   int,
                "float": float,
                "char":  string_valuator
            }.get(itemtype)
            if item_valuator is None:
                raise ValueError( "unknown data type '%s' at header line %d"\
                  %(itemtype, lineno))

            # valuate value items
            if islist:
                value = tuple([item_valuator(item) for item in value.split(",")])
            else:
                value = item_valuator(value)
            setattr(self, name, value)


##############################################################################
class FDFFile (object):

    #-------------------------------------------------------------------------
    def __init__(self, filename):
        self.infile = file(filename)
        self.loadHeader()
        self.loadData()

    #-------------------------------------------------------------------------
    def loadHeader(self):
        self.header = FDFHeader(self.infile)
      
    #-------------------------------------------------------------------------
    def loadData(self):
        # advance file to beginning of binary data (demarcated by a null byte)
        while self.infile.read(1) != "\x00": pass

        datatype = {"integer": "int", "float": "float"}\
          .get(self.header.storage, "Float")
        typecode = getattr(N, "%s%d"%(datatype, self.header.bits))
        shape = [int(d) for d in self.header.matrix]
        self.data = N.fromstring(self.infile.read(),
                                 typecode).byteswap().resize(shape)
