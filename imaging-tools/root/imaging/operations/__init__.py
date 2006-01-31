import sys
from os.path import dirname, basename, join
from glob import glob
from types import TypeType, BooleanType


##############################################################################
def bool_valuator(val):
    if type(val)==BooleanType: return val
    lowerstr = val.lower()
    if lowerstr == "true": return True
    elif lowerstr == "false": return False
    else: raise ValueError(
      "Invalid boolean specifier '%s'. Must be either 'true' or 'false'."%\
      lowerstr)


##############################################################################
class Parameter (object):
    """
    Specifies a named, typed parameter for an Operation.  Is used for
    documentation and during config parsing.
    @cvar name: Text name of the parameter
    @car type: Data type of the parameter
    @cvar default: Default value 
    @cvar description: Optional description
    """

    # map type spec to callable type constructor
    _type_map = {
      "str":str,
      "bool":bool_valuator,
      "int":int,
      "float":float,
      "complex":complex}

    #-------------------------------------------------------------------------
    def __init__(self, name, type="str", default=None, description=""):
        """
	Sets up a named, typed Parameter with an optional descriptive tag. Such parameters are used in 	some image operations
	"""
	self.name=name
        if not self._type_map.has_key(type):
            raise ValueError("type must be one of %s"%self._type_map.keys())
        self.valuator=self._type_map[type]
        self.default=default
        self.description=description

    #-------------------------------------------------------------------------
    def valuate(self, valspec): 
	"""
	Evaluates to an appropriate value, given self's type. For instance, if self is an int, valuate returns 	
	an int from valspec.
	"""
	return self.valuator(valspec)


##############################################################################
class Operation (object):
    """
    The Operation class prototypes an abstract operation, which has parameters, and can be run on an image. In 
    the actual implementation of operations, run will be overloaded, and params may or may not be filled out with 
    Parameter objects. In initializing any of these subclasses, the constructor here is called, which passes control 
    to the configure method, which sets up Parameter values from specifications found in the config file.
    
    Operation gives Recon a systematic interface to any operation.
    
    @cvar params: A list of Parameter objects. This gives an Operation
    a chance to have any number of parameters, despite having a standard
    interface with the system. params is filled in by the subclass, and values
    are given in the config file.
    
    """

    class ConfigError (Exception): pass

    params=()

    #-------------------------------------------------------------------------
    def __init__(self, **kwargs): self.configure(**kwargs)

    #-------------------------------------------------------------------------
    def configure(self, **kwargs):
    	"""
        This method fills in values of Parameter objects in the params list by popping values from the 	kwargs 
	dict, using the Parameter name as a keyword. If the keyword is not in the dict, the default value of the 
	Parameter is used. If there are are unused items in the dict (ie unanticipated parameters), an exception 
	is raised.
	
	@param kwargs: a dict of parameters in {name: val} format
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
     RunLogger is a simple class which can write info about an operation to an output stream. The log will be 
     executable in the future.
    """

    # what command is used to run the executable log
    _magic_string = "#!/usr/bin/env runops"

    #-------------------------------------------------------------------------
    def __init__(self, ostream=sys.stdout):
        self.ostream = ostream
        print >> self.ostream, self._magic_string

    #-------------------------------------------------------------------------
    def _format_doc(self, doc):
        for line in (doc or "").splitlines():
            line = line.strip()
            if line: print >> self.ostream, "#", line

    #-------------------------------------------------------------------------
    def logop(self, operation):
        """
	Writes name and parameters of operations, in the same format as the config file.
	The intention is to create record to facilitate reproduction of results.
	"""
	
	self._format_doc(operation.__class__.__doc__)
        print >> self.ostream, "[%s]"%operation.__class__.__name__
        for parameter in operation.params:
            self._format_doc(parameter.description)
            paramval = getattr(operation, parameter.name)
            print >> self.ostream, "%s = %s"%(parameter.name, paramval)
        print >> self.ostream


##############################################################################
class OperationManager (object):
    """
    This class is responsible for knowing which operations are available
    and retrieving them by name.  It should be a global singleton in the
    system.
    
    @cvar _op_index: contains mapping from opname to opclass
    """
    class InvalidOperationName (Exception): pass
    class DuplicateOperationName (Exception): pass

    #-------------------------------------------------------------------------
    def __init__(self):
        """
	The constructor for OperationManager creates a list of all available imaging operation classes (scans 
	for Python classes which are subclasses of Operation in imaging.operations). The list is indexed by 
	the operation's name as a string. 
	"""
	self._op_index = {}
        self._load_operation_index()

    #-------------------------------------------------------------------------
    def _load_operation_index(self):
	"""
	Steps through the returned list of (name, obj) pairs from _get_operation_modules, adding a new
	{name: obj} dictionary entry into _op_index for any subclass of Operation
	"""
        for opmodule in self._get_operation_modules():
            for name, obj in opmodule.__dict__.items():
                if type(obj)==TypeType and issubclass(obj, Operation) \
                  and obj is not Operation:
                    if self._op_index.has_key(name):
                        raise self.DuplicateOperationName(name)
                    self._op_index[name] = obj

    #-------------------------------------------------------------------------
    def _get_operation_modules(self):
	"""
	Finds and imports all Python modules in imaging/operation. Returns a list of the return value of 
	__import__(...) for each module.
	"""
	opfiles = glob(join(dirname(__file__), "*.py"))
        opmodules = []
        for opfile in opfiles:
            opmodname = basename(opfile).split(".")[0]
            full_opmodname = "imaging.operations.%s"%opmodname
            opmodules.append(__import__(full_opmodname,{},{},[opmodname]))
        return opmodules

    #-------------------------------------------------------------------------
    def getOperationNames(self):
        "@return: a sorted set of keys from the _op_index dictionary."
        names = self._op_index.keys()
        names.sort()
        return names

    #-------------------------------------------------------------------------
    def getOperation(self, opname):
        """
	queries _op_index for an operation class, given the opname
	@return: the operation class for the given name
	"""
        operation = self._op_index.get(opname, None)
        if not operation:
            raise self.InvalidOperationName("Operation '%s' not found."%opname)
        return operation
