"Defines a tool for extracting values from a Varian procpar file."

# configure command line options
from optparse import OptionParser
parser = OptionParser( usage=\
"""usage: %prog procpar [param, param, ...]
  Where filename is the name of the procpar file, followed by
  zero or more parameter names.  If no parameter names are given,
  all parameters in the file will be displayed.""" )

def cli():
    import sys
    options, args = parser.parse_args()

    def fail( mesg="" ):
        print mesg
        parser.print_help()
        sys.exit(0)

    if not args: fail( "Filename required." )
    filename = args[0]
    import os.path
    if not os.path.exists( filename ): fail( "File not found: %s"%filename )
    import varian
    try:
        procpar = varian.procpar( filename )
    except: fail( "Error parsing procpar file." )

    import pprint
    if len( args ) == 1: params = procpar.keys()
    else: params = args[1:]
    params.sort()
    for param in params:
        if procpar.has_key( param ):
            valuestr = "\n".join( map( str, procpar[param] ) )
        else: valuestr = "PARAMETER NOT FOUND"
        print "%s: %s"% (param, valuestr)

