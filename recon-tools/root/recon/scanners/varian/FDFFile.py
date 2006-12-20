import pylab
from Numeric import fromstring

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

        datatype = {"integer": "Int", "float": "Float"}\
          .get(self.header.storage, "Float")
        typecode = getattr(pylab, "%s%d"%(datatype, self.header.bits))
        shape = [int(d) for d in self.header.matrix]
        self.data = fromstring(self.infile.read(), typecode)\
                    .byteswapped().resize(shape)
