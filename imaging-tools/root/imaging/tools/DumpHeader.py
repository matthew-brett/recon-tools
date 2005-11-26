import sys
from optparse import OptionParser
from imaging.varian.FidFile import FidFile

# configure command line options
parser = OptionParser( usage="usage: %prog [options] fidfile" )
parser.add_option( "-b", "--blocknum",
    help="dump header for a particular data block",
    metavar="<blocknum>" )

def cli():
    options, args = parser.parse_args()

    # get the filename
    if not args:
        parser.print_help()
        sys.exit(0)
    else: filename = args[0]

    # open fidfile
    try:
        fh = FidFile( filename )
    except IOError, e:
        print e
        parser.print_help()
        sys.exit(0)

    # dump the file header
    if options.blocknum is None:
        for fieldname in fh.HEADER_FIELD_NAMES:
            print "%s:"%fieldname, getattr( fh, fieldname )

    # or dump a block header
    else:
        bnum = int(options.blocknum)
        if bnum < 0 or bnum >= fh.nblocks:
            print "Bad blocknum %s.  File has %s blocks."%(bnum,fh.nblocks) 
            parser.print_help()
            sys.exit(0)
        block = fh.getBlock( bnum )
        for fieldname in block.HEADER_FIELD_NAMES:
            print "%s:"%fieldname, getattr( block, fieldname )
 
