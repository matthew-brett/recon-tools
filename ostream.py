"provide C++ stream-like objects for output."
import sys

##############################################################################
class outstream (object):

    def __init__(self, outf=sys.stdout, mode="w" ):
        # if a string, assume it's a filename
        if isinstance( outf, str ): self._outf = file( outf, mode )
        elif hasattr( outf, "write" ): self._outf = outf
        else: raise TypeError( "outf must either be a string or have a write method" )

    def close(self):
        # never close sys.stdout or sys.stderr
        if hasattr( self._outf, "close" ):
            if (hasattr( sys, "stdout" ) and self._outf != sys.stdout) or \
               (hasattr( sys, "stderr" ) and self._outf != sys.stderr):
                self._outf.close()
    
    def __del__(self):
        self.close()

    def write( self, x ):
        if not isinstance( x, str ): x = str( x )
        self._outf.write( x )
        return self

    def __lshift__( self, text ):
        return self.write( text )

    def flush(self):
        if hasattr( self._outf, "flush" ): self._outf.flush()


##############################################################################
class outstring (outstream):
    def __init__(self ):
        import cStringIO
        outf = cStringIO.StringIO()
        super( outstream, self).__init__( outf )
    def getvalue(self):
        return self._outf.getvalue()


##############################################################################
class multicaster (object):
    """
    multicaster provides outstream-like functionality for writing
    to multiple destinations.  The constructor accepts a list (or tuple)
    of open files, outstreams, multicasters, or strings (representing
    a filename, which will be opened for writing).  Then writing can
    be performed with the << operator.
    Eg:
        from outstream import *
        import sys

        logfilename = "/tmp/log.txt"
        mcast = multicaster( (sys.stdout, logfilename) )

        # the following will write to sys.stdout and the log file
        mcast << "'Twas brillig and the slithy toves\n"
    """

    def  __init__( self, outfs=(sys.stdout,), mode="w" ):
        """
        takes a tuple of output file handles or instances of an outstream
        """
        import sets
        outf_set = sets.Set()
        for outf in outfs:

            # string, so interpret as a file name and open
            if isinstance(outf, str):
                outf_set.add( file( outf, mode ) )
            # something with a write method
            elif hasattr( outf, "write" ):
                outf_set.add( outf )
            # multicaster instance
            elif isinstance(outf, multicaster):
                outf_set.union_update( outf._outf_set )
            # unknown type
            else:
                raise TypeError( "Arg %s has invalid type %s for Multicaster" % (outf,type(outf)) )
        self._outf_set = outf_set

    def close( self ):
        for outf in self._outf_set:
            # never close sys.stdout or sys.stderr
            if hasattr( outf, "close" ):
                if (hasattr( sys, "stdout" ) and outf != sys.stdout) or \
                   (hasattr( sys, "stderr" ) and outf != sys.stderr):
                    outf.close()

    def write( self, x ):
        if not isinstance( x, str ): x = str( x )
        for outf in self._outf_set:
            outf.write( x )
        return self
    
    def __lshift__( self, x ):
        return self.write( x )

    def flush( self ):
        for outf in self._outf_set:
            if hasattr( outf, "flush" ): outf.flush()


# this can be used in place of a stream for code
# that needs a stream, in cases where you don't
# want anything written
nullstream = multicaster( () )
