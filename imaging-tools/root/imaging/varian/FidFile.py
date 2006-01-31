"Describes classes to interpret a FID file"
from __future__ import generators # this allows it to work in Python-2.2+
import struct


##############################################################################
class _FlagsByte (object):
    """
    Represents a byte with a named flag for each bit.
    Must be subclassed with a property for each flag bit.
    """
    def __init__( self, byte ):
        self._byte = byte

    def __str__( self ):
        names = dir( self.__class__ )
        props = [name for name in names if \
            isinstance( getattr( self.__class__, name ), property )]
        return "(%s)" % (", ".join( ["%s=%s"%(name,getattr( self, name )) \
            for name in props ] ))


##############################################################################
class _SharedHeaderStatus (_FlagsByte):
    """
    knows how to interpret the first six bits of the file and block
    headers' status byte
    """
    S_DATA = property( lambda self: self._byte&0x1 !=0,
        doc="0=no data, 1=data" )
    S_SPEC = property( lambda self: self._byte&0x2 !=0,
        doc="0=FID, 1=spectrum" )
    S_32 = property( lambda self: self._byte&0x4 !=0,
        doc="0=16-bit int, 1=32-bit int, (only used when S_FLOAT=0" )
    S_FLOAT = property( lambda self: self._byte&0x8 !=0,
        doc="0=integer, 1=floating point" )
    S_COMPLEX = property( lambda self: self._byte&0x10 !=0,
        doc="0=real, 1=complex" )
    S_HYPERCOMPLEX = property( lambda self: self._byte&0x20 !=0,
        doc="1=hypercomplex" )


##############################################################################
class _HeaderBase (object):
    """
     subclasses must implement:
        HEADER_FMT,
        HEADER_SIZE,
        HEADER_FIELD_NAMES
    """

    #-------------------------------------------------------------------------
    def __init__( self, file_handle ):
        
        # read and validate header
        header = file_handle.read( self.HEADER_SIZE )
        assert len(header) == self.HEADER_SIZE, \
            "Bad header size: expected %s but got %s" % \
            (self.HEADER_SIZE, len(header))

        header_fields = struct.unpack( self.HEADER_FMT, header )
        assert len(header_fields) == len(self.HEADER_FIELD_NAMES), \
            "Wrong number of header fields for format '%s':\n" \
            "  expected %s but got %s" % \
            (len(self.HEADER_FIELD_NAMES), len(header_fields))

        # add header fields to self
        header_dict = dict( zip( self.HEADER_FIELD_NAMES, header_fields) )
        self.__dict__.update( header_dict )



##############################################################################
class _BlockHeaderBase (_HeaderBase):
    HEADER_FMT = ">hhhhlffff"
    HEADER_SIZE = struct.calcsize( HEADER_FMT )


##############################################################################
class BlockHeader (_BlockHeaderBase):
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
        "knows how to interpret each bit of the block header status byte"
        MORE_BLOCKS = property( lambda self: self._byte&0x80 !=0,
            doc="0=absent, 1=present" )
        NP_CMPLX = property( lambda self: self._byte&0x100 !=0,
            doc="0=real, 1=complex" )
        NF_CMPLX = property( lambda self: self._byte&0x200 !=0,
            doc="0=real, 1=complex" )
        NI_CMPLX = property( lambda self: self._byte&0x400 !=0,
            doc="0=real, 1=complex" )
        NI2_CMPLX = property( lambda self: self._byte&0x800 !=0,
            doc="0=real, 1=complex" )

    #-------------------------------------------------------------------------
    class _Mode (_FlagsByte):
        "knows how to interpret each bit of the block header mode byte"
        NP_PHMODE = property( lambda self: self._byte&0x1!=0)
        NP_AVMODE = property( lambda self: self._byte&0x2 !=0)
        NP_PWRMODE = property( lambda self: self._byte&0x4 !=0)
        NF_PHMODE = property( lambda self: self._byte&0x10!=0)
        NF_AVMODE = property( lambda self: self._byte&0x20 !=0)
        NF_PWRMODE = property( lambda self: self._byte&0x40 !=0)
        NI_PHMODE = property( lambda self: self._byte&0x100 !=0)
        NI_AVMODE = property( lambda self: self._byte&0x200 !=0)
        NI_PWRMODE = property( lambda self: self._byte&0x400 !=0)
        NI2_PHMODE = property( lambda self: self._byte&0x1000 !=0)
        NI2_AVMODE = property( lambda self: self._byte&0x2000 !=0)
        NI2_PWRMODE = property( lambda self: self._byte&0x4000 !=0)

    #-------------------------------------------------------------------------
    def __init__( self, file_handle ):
        super( BlockHeader, self ).__init__( file_handle )
        self.status = self._Status( self.status )
        self.mode = self._Mode( self.mode )


##############################################################################
class HypercomplexBlockHeader (_BlockHeaderBase):
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
        """
        knows how to interpret each bit of the hypercomplex block
        header status byte
        """
        U_HYPERCOMPLEX = property( lambda self: self._byte&0x2 !=0,
            doc="1=hypercomplex block structure" )

    #-------------------------------------------------------------------------
    def __init__( self, file_handle ):
        super( BlockHeader, self ).__init__( file_handle )
        self.status = self._Status( self.status )


##############################################################################
class DataBlock (BlockHeader):

    #-------------------------------------------------------------------------
    def __init__( self, fidfile ):
        super( DataBlock, self ).__init__( fidfile )
        self._fidfile = fidfile
        self._data = None
        self.header_size = fidfile.block_header_size
        if fidfile.nbheaders == 2:
            self.hyperheader = HypercomplexBlockHeader( fidfile )

    #-------------------------------------------------------------------------
    def __iter__( self ):
        "yield next trace"
        for trace_num in xrange(self._fidfile.ntraces):
            yield self._fidfile.read(self._fidfile.tbytes)

    #-------------------------------------------------------------------------
    def getData(self):
        if self._data is None:
            self._data = \
              self._fidfile.read(self._fidfile.bbytes - self.header_size)
        return self._data


##############################################################################
class FidFile (file, _HeaderBase ):
    HEADER_FMT = ">llllllhhl"
    HEADER_SIZE = struct.calcsize( HEADER_FMT )
    HEADER_FIELD_NAMES = (
        "nblocks",      # number of blocks in file
        "ntraces",      # number of traces per block
        "np",           # number of elements per trace
        "ebytes",       # number of bytes per element
        "tbytes",       # number of bytes per trace
        "bbytes",       # number of bytes per block
        "vers_id",      # software version, file_id status bits
        "status",       # status of whole file
        "nbheaders"     # number of block header per block
    )

    #-------------------------------------------------------------------------
    class FileHeaderStatus (_SharedHeaderStatus):
        "knows how to interpret each bit of the file header status byte"
        S_ACQPAR = property( lambda self: self._byte&0x80 !=0,
            doc="0=not Acqpar, 1=Acqpar" )
        S_SECND = property( lambda self: self._byte&0x100 !=0,
            doc="first FT, 1=second FT" )
        S_TRANSF = property( lambda self: self._byte&0x200 !=0,
            doc="0=regular, 1=transposed" )
        S_NP = property( lambda self: self._byte&0x800 !=0,
            doc="1=np dimension is active" )
        S_NF = property( lambda self: self._byte&0x1000 !=0,
            doc="1=nf dimension is active" )
        S_NI = property( lambda self: self._byte&0x2000 !=0,
            doc="1=ni dimension is active" )
        S_NI2 = property( lambda self: self._byte&0x4000 !=0,
            doc="1=ni2 dimension is active" )

    #-------------------------------------------------------------------------
    def __init__( self, name ):
        "always open read-only"
        file.__init__( self,  name )
        _HeaderBase.__init__(self, self )
        
        # convert status byte into a FileHeaderStatus object (it knows how
        # interpret each status bit)
        self.status = self.FileHeaderStatus( self.status )
        self.block_data_size = self.ntraces*self.tbytes
        self.block_header_size = self.bbytes - self.block_data_size

    #-------------------------------------------------------------------------
    def __iter__(self):
        "yield next block"
        self.seekBlock(0)
        for block_num in xrange( self.nblocks ): yield DataBlock( self )

    #-------------------------------------------------------------------------
    def seekBlock(self, bnum):
        "Seek to the beginning of block bnum (index starts a zero)."
        self.seek(self.HEADER_SIZE + bnum*self.bbytes)

    #-------------------------------------------------------------------------
    def getBlock(self, bnum):
        "Return a single block (index starts a zero)."
        self.seekBlock(bnum)
        return DataBlock(self)
