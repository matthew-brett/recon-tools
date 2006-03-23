from ConfigParser import SafeConfigParser
from optparse import OptionParser
from odict import odict
from imaging.util import import_from

tool_map = odict((
  ("dumpheader", "DumpHeader"),
  ("fdf2img", "Fdf2Img"),
  ("getparam", "GetParam"),
  ("imaging-doc", "ImagingDoc"),
  ("recon", "Recon"),
  ("viewimage", "ViewImageTool")))
tool_names = tool_map.keys()

#-----------------------------------------------------------------------------
def getToolByName(toolname):
    if not tool_map.has_key(toolname):
        raise ValueError("No tool called '%s'."%toolname)
    classname = tool_map[toolname]
    modulename = "imaging.tools.%s"%classname
    try:
        return import_from(modulename, classname)
    except ImportError:
        raise RuntimeError("Tool class %s.%s not found."%(modulename,classname))


##############################################################################
class OrderedConfigParser (SafeConfigParser):
    """
    Config parser which keeps track of the order in which sections appear.
    
    This class extends ConfigParser just slightly, giving it an ordered
    dictionary to store sections, so that operations listed in the config file
    are performed in the correct order.  Otherwise, the functionality is that
    of ConfigParser.
    """

    #-------------------------------------------------------------------------
    def __init__(self, defaults=None):
        SafeConfigParser.__init__(self, defaults=defaults)
        self._sections = odict()


##############################################################################
class ConsoleTool (OptionParser):
    command_name = ""
    description = ""
