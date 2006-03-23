"""
The imaging.conf package exists to contain configuration data files.  It does
not contain any python modules.  The package defines a single function which
can retrieve these config files by name.
"""
from os.path import dirname, join, exists

#-----------------------------------------------------------------------------
def getOplistFileName(name):
    "Resolves the given config name into a full file path to that config file"
    confdir = dirname(__file__)
    filename = join(confdir, name)
    if not exists(filename):
        raise ValueError("File %s not found in config directory %s"\
                         %(name,confdir))
    return filename
