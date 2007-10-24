"""
This module describes a type of MR Image that comes from a scanner's
raw output file. It will have certain properties that a general
processed image (eg Analyze, NIFTI) may not, but that are necessary for
artifact correction techniques.

This module also has helper code to describe two basic things that
will come up in these scanner files:
   * Bit-masked words, where individual bits mark off flags or fields.
     These objects have a number of "property" objects, whose "getters"
     check the data word against some bitmask, and indicating whether the
     flag is on or off, or giving the value of the field (see example below)

   * Data headers, which are just chunks of bytes that are broken up into
     a number of primary types (ints, longs, floats, etc). A generic data
     header class is provided (_HeaderBase), which given field names and
     struct composition, will decode the header and define attributes named
     by the field names. Typically _HeaderBase will be extended by an
     object that also yields data. A header may include bit-masked words.
"""

import numpy as N
import struct
from os import path
try:
    from recon.imageio import ReconImage
    from recon import util
except ImportError:
    # allow import of this module: scanners.varian is needed by setup.py
    # define a null class
    class ReconImage (object):
        def __init__(self):
            pass

tablib = path.join(path.split(__file__)[0], "tablib")

class ScannerImage (ReconImage):
    """
    Interface for image data originating from an MRI scanner.
    This class of objects should be able to go through all of the operations,
    especially the artifact correction operations.

    In order to be used with certain Operations, this class guarantees the
    definition of certain system/scanning parameters.

    """

    necessary_params = dict((
        ('T_pe', 'time to scan from k(y=a,x=0) to k(y=a+1,x=0)'),
        ('phi', 'euler angle about z'),
        ('psi', 'euler angle about y'),
        ('theta', 'euler angle about x'),
        ('delT', 'sampling dwell time'),
        ('echo_time', 'time-to-echo'),
        ('asym_times', 'list of te times in an asems scan'),
        ('acq_order', 'order that the slices were acquired'),
        ('nseg', 'number of sampling segments'),
        ('sampstyle', 'style of sampling: linear, centric, interleaved'),
        ('pe0', 'value of the first sampled pe line (normally -N2/2)'),
        ('tr', 'time series step size'),
        ('dFE', 'frequency-encode step size (in mm)'),
        ('dPE', 'phase-encode step size (in mm)'),
        ('dSL', 'slice direction step size (in mm)'),
        ('path', 'path of associated scanner data file'),
    ))

    def __init__(self):
        # should set up orientation info
        self.check_attributes()
        if not hasattr(self, "orientation_xform"):
            self.orientation_xform = util.Quaternion()
        if not hasattr(self, "orientation"):
            self.orientation = ""
        ReconImage.__init__(self, self.data, self.dFE,
                            self.dPE, self.dSL, self.tr,
                            orient_xform=self.orientation_xform,
                            orient_name=self.orientation)
                            

    # may change this this ksp_trajectory and handle different cases
    def epi_trajectory(self, pe0=None):
        M = self.shape[-2]
        # sometimes you want to force pe0
        if not pe0:
            pe0 = self.pe0
        if self.sampstyle == "centric":
            if self.nseg > 2:
                raise NotImplementedError("centric sampling not implemented for nseg > 2")
            a = util.checkerline(M)
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
        return (a, b) 

    def check_attributes(self):
        for key in ScannerImage.necessary_params.keys():
            if not hasattr(self, key):
                raise AttributeError("This is not a complete ScannerImage, "\
                                     "missing parameter %s"%key)
            

    
##############################################################################
class _HeaderBase (object):
    """
    This class is an all-purpose header representation using the built-in
    "struct" module. A subclass must implement:
    HEADER_FMT (struct format string),
    HEADER_FIELD_NAMES (an iterable of names for each of the fields)

    Anything that is a _HeaderBase will have those names as attributes.
    """

    #-------------------------------------------------------------------------
    def __init__( self, file_handle ):
        
        # read and validate header
        header_size = struct.calcsize(self.HEADER_FMT)
        header = file_handle.read( header_size )
        assert len(header) == header_size, \
            "Bad header size: expected %d but got %d" % \
            (header_size, len(header))

        # translate header bytes according to format string
        header_fields = struct.unpack( self.HEADER_FMT, header )
        assert len(header_fields) == len(self.HEADER_FIELD_NAMES), \
            "Wrong number of header fields for format '%s':\n" \
            "  expected %d but got %d" % (self.HEADER_FMT,
                                          len(self.HEADER_FIELD_NAMES),
                                          len(header_fields))

        # add header fields to self
        header_dict = dict( zip( self.HEADER_FIELD_NAMES, header_fields) )
        self.__dict__.update( header_dict )

##############################################################################
class _BitMaskedWord (object):
    """
    Represents a data word comprised of several arbitrary-length bitmasked
    encodings. This class only provides a handy to-string method. Any real
    subclass implementations should use a built-in Python "property" to
    represent each flag or field, For example, if bits 2-3 are a small number:

    ZeroToThree = property(lambda self: (self._word & 0xC) >> 2,
                           doc="retrieves a short int from bits 2-3")

    The word argument must be some size of integer.
    """
    def __init__( self, word ):
        self._word = word
	
    def __str__( self ):
        names = dir( self.__class__ )
        props = [name for name in names if \
                 isinstance( getattr( self.__class__, name ), property )]
        return "(%s)" % (", ".join( ["%s=%s"%(name,getattr( self, name )) \
                                     for name in props ] ))
