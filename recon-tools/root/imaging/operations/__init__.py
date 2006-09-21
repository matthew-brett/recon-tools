import sys
from os.path import dirname, basename, join
from glob import glob
from types import TypeType, BooleanType, TupleType

from recon.util import import_from

##############################################################################
def bool_valuator(val):
    if type(val)==BooleanType: return val
    lowerstr = val.lower()
    if lowerstr == "true": return True
    elif lowerstr == "false": return False
    else: raise ValueError(
      "Invalid boolean specifier '%s'. Must be either 'true' or 'false'."%\
      lowerstr)

def tuple_valuator(val):
    if not val: return ()
    if type(val) is TupleType: return tuple(val)
    # else it is a stringified tuple ill-gotten from an oplist (bah!)
    def try_int(el):
        try:
            return int(el)
        except ValueError:
            return
    return tuple([try_int(el) for el in val if try_int(el)])


##############################################################################
class Parameter (object):
    """
    Specifies a named, typed parameter for an Operation.  Is used for
    documentation and during config parsing.
    @cvar name: Text name of the parameter
    @cvar type: Data type of the parameter (can be str, bool, int, float, or
               complex)
    @cvar default: Default value 
    @cvar description: Optional description
    """

    # map type spec to callable type constructor
    _type_map = {
      "str":str,
      "bool":bool_valuator,
      "int":int,
      "float":float,
      "complex":complex,
      "list":list,
      "tuple":tuple_valuator}

    #-------------------------------------------------------------------------
    def __init__(self, name, type="str", default=None, description=""):
        self.name=name
        if not self._type_map.has_key(type):
            raise ValueError("type must be one of %s"%self._type_map.keys())
        self.valuator=self._type_map[type]
        self.default=default
        self.description=description

    #-------------------------------------------------------------------------
    def valuate(self, valspec): 
        """
        Evaluates valspec (a string) to the appropriate value according to
        the type of self.
        """
        # don't valuate None, especially not as a string
        if valspec is None:
            return valspec
        else: return self.valuator(valspec)



##############################################################################
class Operation (object):
    """
    The Operation class prototypes an abstract operation, which has parameters,
    and can be run on an image. In the actual implementation of operations, run
    will be overloaded, and params may or may not be filled out with Parameter
    objects. In initializing any of these subclasses, the constructor here is
    called, which assigns parameter values based on its keyword arguments.
    
    @cvar params: A list of Parameter objects describing which configuration
                  parameters this operations will accept.
    """

    class ConfigError (Exception): pass

    params=()

    #-------------------------------------------------------------------------
    def __init__(self, **kwargs): self.configure(**kwargs)

    #-------------------------------------------------------------------------
    def configure(self, **kwargs):
        """
        Assign values to this Operation's Parameters based on provided keyword
        arguments.  The keyword is the parameter name and the keyword value is
        a string specifying the parameter value.  These strings are transformed
        into values by the Parameter's valuate method.  Parameters with no
        keyword value provided here will receive their declared default value.

        A ConfigError is raised if unanticipated parameters are found (ie
        parameters which are not declared in the Operation's params list).
        """
        for p in self.params:
            self.__dict__[p.name] = p.valuate(kwargs.pop(p.name, p.default))

        # All valid args should have been popped off the kwargs dict at this
        # point.  If any are left, it means they are not valid parameters for
        # this operation.
        leftovers = kwargs.keys()
        if leftovers:
            raise self.ConfigError("Invalid parameter '%s' for operation %s"%
              (leftovers[0], self.__class__.__name__))

    #-------------------------------------------------------------------------
    def log(self, message):
        print "[%s]: %s"%(self.__class__.__name__, message)

    #-------------------------------------------------------------------------
    def run(self, image): 
        "This is how the Recon system calls an operation to process an image"
        pass


##############################################################################
class RunLogger (object):
    """
    RunLogger is a simple class which can write info about an operation to an
    output stream. The log will be made executable in a future release.
    """

    # what command is used to run the executable log
    _magic_string = "#!/usr/bin/env python\n"\
                    "from recon.tools.Rerun import Rerun\n"\
                    "if __name__==\"__main__\": Rerun(__file__).run()\n"

    _start_string = "## BEGIN OPS LOG\n"
    _end_string = "## END OPS LOG\n"
    #-------------------------------------------------------------------------
    def __init__(self, ostream=sys.stdout):
        self.ostream = ostream
        print >> self.ostream, self._magic_string
        print >> self.ostream, self._start_string

    #-------------------------------------------------------------------------
    def _format_doc(self, doc):
        for line in (doc or "").splitlines():
            line = line.strip()
            if line: print >> self.ostream, "##", line

    #-------------------------------------------------------------------------
    def logop(self, operation):
        """
        Writes name and parameters of operations, in the same format as the
        config file.  The intention is to record the provenance of analyses
        and facilitate reproduction of results.
        """
        self._format_doc(operation.__class__.__doc__)
        print >> self.ostream, "#[%s]"%operation.__class__.__name__
        for parameter in operation.params:
            self._format_doc(parameter.description)
            paramval = str(getattr(operation, parameter.name))
            if paramval.find("%")>0:
                paramval = paramval.split("%")[0]+"%%"+paramval.split("%")[1]
            print >> self.ostream, "#%s = %s"%(parameter.name, paramval)
        print >> self.ostream

    #-------------------------------------------------------------------------
    def setExecutable(self):
        # check to see if we're using abnormal file streams, but nothing
        # done about general file permissions
        import os
        if self.ostream.name not in ('/dev/null', '<stdout>'):
            os.system('chmod +x %s'%(self.ostream.name))
            
        
##############################################################################
class OperationManager (object):
    """
    This class is responsible for knowing which operations are available
    and retrieving them by name.  It should be a global singleton in the
    system.
    """
    class InvalidOperationName (Exception): pass
    class DuplicateOperationName (Exception): pass

    #-------------------------------------------------------------------------
    def __init__(self):
        self._op_index = {}
        self._load_operation_index()

    #-------------------------------------------------------------------------
    def _load_operation_index(self):
        """
        Find and index by classname all Operation subclasses declared in any
        module in the recon.operations package.
        """
        for name, obj in self._get_operation_modules():
            if type(obj)==TypeType and issubclass(obj, Operation) \
                   and obj is not Operation:
                if self._op_index.has_key(name):
                    raise self.DuplicateOperationName(name)
                self._op_index[name] = obj

    #-------------------------------------------------------------------------
    def _get_operation_modules(self):
        """
        Find and import all modules in the recon.operation package.
        Return a list of the found module objects.
        """
        opfiles = glob(join(dirname(__file__), "*.py"))
        opmodules = []
        for opfile in opfiles:
            opmodname = basename(opfile).split(".")[0]
            full_opmodname = "recon.operations.%s"%opmodname
            opmodules.append((opmodname,
                              import_from(full_opmodname, opmodname)))
        return opmodules

    #-------------------------------------------------------------------------
    def getOperationNames(self):
        "@return: an alphabetically sorted list of all Operation class names."
        names = self._op_index.keys()
        names.sort()
        return names

    #-------------------------------------------------------------------------
    def getOperation(self, opname):
        """
        Retrieve the Operation class object with the given opname.
        @return: the Operation class with the given name.
        """
        operation = self._op_index.get(opname, None)
        if not operation:
            raise self.InvalidOperationName("Operation '%s' not found."%opname)
        return operation
