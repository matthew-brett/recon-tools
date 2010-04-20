__version__ = "0.5"

from numpy.testing import *
from os import path
import warnings
import struct
import tempfile

# nosetester is a relatively new feature of numpy
try:
    test = nosetester.NoseTester().test
except NameError:
    def test(level=1):
        mod_file = path.split(path.split(__file__)[0])[-1]
        NumpyTest(mod_file).testall(level=level)
try:
    import scipy
    bench = nosetester.NoseTester().bench
except ImportError:
    def bench(**kw):
        warnings.warn("Some benchmarks require scipy to be installed")
except NameError:
    def bench(**kw):
        warnings.warn("older scipy detected, not implementing benchmarks")

def set_temp_dir(p):
    if path.exists(p):
        tempfile.tempdir = p

def visual(t):
    """A decorator for nose testing--use this for plotting tests.
    """
    t.visual = True
    return t

def find_extensions(build=False):
    """This method walks through the code to grab and execute any
    "export_extension" functions it finds. This is intended for distutils,
    but can also be helpful on the fly if build=True.
    """
    import sys
    warnings.filterwarnings('ignore')
    exts = {}
    # why this??
    import recon
    def get_ext(base, dirname, fnames):
        for fnm in fnames:
            if fnm.endswith('.py'):
                pth = dirname.split(base_path)[1]
                if fnm != '__init__.py':
                    mod = fnm[:-3]
                else:
                    mod = ''
                modstr = 'recon'+pth.replace(path.sep,'.')
                if mod:
                    modstr += '.'+mod
                try:
                    ext_getter = import_from(modstr, 'export_extension')
                    e = ext_getter(build=build)
                    exts.update( {modstr: e} )
                except AttributeError:
                    continue
                except ImportError:
                    continue
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
        

    base_path = path.abspath(path.split(__file__)[0])
    path.walk(base_path, get_ext, base_path)
    warnings.resetwarnings()
    return exts.values()


##############################################################################
### Since the util.py module imports from here and there, this is a spot   ###
### for non-numerical utilities that were previously defined in util.      ###
##############################################################################
# struct byte order constants
NATIVE = "="
LITTLE_ENDIAN = "<"
BIG_ENDIAN = ">"

def struct_format(byte_order, elements):
    return byte_order+" ".join(elements)
    
def struct_unpack(infile, byte_order, elements):
    format = struct_format(byte_order, elements)
    return struct.unpack(format, infile.read(struct.calcsize(format)))

def struct_pack(byte_order, elements, values):
    format = struct_format(byte_order, elements)
    return struct.pack(format, *values)

#-----------------------------------------------------------------------------
def import_from(modulename, objectname):
    "Import and return objectname from modulename."
    module = __import__(modulename, globals(), locals(), (objectname,))
    return getattr(module, objectname)
#-----------------------------------------------------------------------------
def loads_extension_on_call(ext_name, namespc_dict):
    """This function requires a C extension (named ext_name) that this
    module builds. If the extension is not already loaded, this decorator
    will try to load it before ungating the function. If there is a problem
    loading the extension, then the function will be redefined to do nothing.
    """
    def dec(func):
        ext_loaded = namespc_dict.get(ext_name+'_loaded', False)
        if not ext_loaded:
            try:
                exec 'import '+ext_name in namespc_dict
                namespc_dict[ext_name+'_loaded'] = True
            except:
                warnings.warn("The C extension %s must be built before using this function."%ext_name)
                def new_func(*args, **kwargs):
                    raise ImportError("The function %s passes until the appropriate extension is installed"%func)
                    pass
                return new_func
        
        return func
    return dec
