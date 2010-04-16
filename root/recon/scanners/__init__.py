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
        ('petable', 'acquisition order of PE rows'),
        ('sampstyle', 'style of sampling: linear, centric, interleaved'),
        ('pe0', 'value of the first sampled pe line (normally -N2/2)'),
        ('tr', 'time series step size'),
        ('dFE', 'frequency-encode step size (in mm)'),
        ('dPE', 'phase-encode step size (in mm)'),
        ('dSL', 'slice direction step size (in mm)'),
        ('path', 'path of associated scanner data file'),
        ('n_vol', 'number of volumes'),
        ('n_slice', 'number of slices (or pts in the z-direction)'),
        ('n_pe', 'number of phase encodes'),
        ('n_fe', 'number of frequency encodes'),
        ('data', 'the data array'),
    ))

    def __init__(self, **kwargs):
        # should set up orientation info
        if not hasattr(self, "orientation_xform"):
            self.orientation_xform = util.Quaternion()
        if not hasattr(self, "petable"):
            self.petable = N.arange(self.n_pe)
        self.check_attributes()
        ReconImage.__init__(self, self.data, isize=self.dFE, jsize=self.dPE,
                            ksize=self.dSL, tsize=self.tr,
                            orient_xform=self.orientation_xform, **kwargs)
                            

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
        raise NotImplementedError
    
    def seg_slicing(self, n):
        """
        This method returns a list of indices corresponding to segment n,
        to be used in separating segments from recombined kspace
        """
        raise NotImplementedError

    def check_attributes(self):
        for key in ScannerImage.necessary_params.keys():
            if not hasattr(self, key):
                raise AttributeError("This is not a complete ScannerImage, "\
                                     "missing parameter %s"%key)
            

##############################################################################
class CachedReadOnlyProperty (property):
    """
    This kind of object is a Python property, but with cached results
    and no getter. It can be used as a function decorator.
    """
    _id = 0
    def __init__(self, getter):
        key = CachedReadOnlyProperty._id
        def cached_getter(self):
            if not hasattr(self, "_propvals"): self._propvals = {}

            if self._propvals.has_key(key):
                a = self._propvals[key]
                return a
            else:
                a = getter(self)
                self._propvals[key] = a
                return a
        CachedReadOnlyProperty._id += 1
        property.__init__(self, fget=cached_getter, doc=getter.__doc__)
    
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
    def __init__( self, hdr_string ):
        
        # read and validate header
        assert len(hdr_string) == self.header_size(), \
            "Bad header size: expected %d but got %d" % \
            (self.header_size(), len(hdr_string))

        # translate header bytes according to format string
        header_fields = struct.unpack( self.HEADER_FMT, hdr_string )
        assert len(header_fields) == len(self.HEADER_FIELD_NAMES), \
            "Wrong number of header fields for format '%s':\n" \
            "  expected %d but got %d" % (self.HEADER_FMT,
                                          len(self.HEADER_FIELD_NAMES),
                                          len(header_fields))

        # add header fields to self
        header_dict = dict( zip( self.HEADER_FIELD_NAMES, header_fields) )
        self.__dict__.update( header_dict )

    @classmethod
    def header_size(cls):
        return struct.calcsize(cls.HEADER_FMT)

    def __str__(self):
        """prints the values of any HEADER_FIELD_NAMES items..
        potentially could scan for BitMaskedWord items in self.__dict__
        and print them too.
        """
        s1 = ''
        s2 = ''
        for name in self.HEADER_FIELD_NAMES:
            s1 += name + ' = ' + str(getattr(self, name)) + '\n'
        for key in self.__dict__.keys():
            obj = getattr(self, key)
            if isinstance(obj, _BitMaskedWord):
                s2 += 'Bit Masked Word %s contents:\n'%key
                s2 += str(obj) + '\n\n'
        if s2:
            s1 += '+'*40 + '\n'
            s1 += s2
        return s1

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

