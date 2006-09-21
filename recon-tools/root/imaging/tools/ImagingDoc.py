"Defines a command-line to the recon-doc tool."
from optparse import Option
 
import recon
import recon.conf
from recon.util import import_from
from recon.tools import tool_names, getToolByName, ConsoleTool
from recon.operations import OperationManager


##############################################################################
class ImagingDoc (ConsoleTool):
    "Command-line documentation for the recon-tools package."

    usage= "usage: %prog [options]\n ", __doc__

    _opmanager = OperationManager()

    options = (

      Option("-v", "--version", dest="show_version",
        default=False, action="store_true",
        help="Show which version of recon-tools is used."),

      Option("-t", "--tools", dest="show_tools",
        default=False, action="store_true",
        help="List command-line tools provided by the recon-tools package."),

      Option("-u", "--tool", dest="tool_name", type="string",
        default=None, action="store",
        help="Show tool usage for the named tool (case sensitive)."),

      Option("-o", "--operations", dest="show_operations",
        default=False, action="store_true",
        help="List operations defined in the recon-tools package."),

      Option("-d", "--operation", dest="operation_name", type="string",
        default=None, action="store",
        help="Describe the named recon-tools operation (case sensitive)."))

    #-------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        ConsoleTool.__init__(self, *args, **kwargs)
        self.set_usage("usage: %prog [options]")
        self.add_options(self.options)

    #-------------------------------------------------------------------------
    def showVersion(self):
        "Print which version of recon-tools is in use."
        print recon.__version__

    #-------------------------------------------------------------------------
    def showTools(self):
        "List all command-line tools in the recon-tools system."
        for toolname in tool_names:
            print toolname
            print getToolByName(toolname).description

    #-------------------------------------------------------------------------
    def toolUsage(self, tool_name):
        "Show usage string for the given command-line tool."
        toolclass = getToolByName(tool_name)
        tool = toolclass(prog=tool_name)
        print tool.format_help()

    #-------------------------------------------------------------------------
    def showOperations(self):
        "List all Operations known to the recon-tools system."
        opnames = self._opmanager.getOperationNames()
        for opname in opnames:
            operation = self._opmanager.getOperation(opname)
            print opname
            print operation.__doc__

    #-------------------------------------------------------------------------
    def describeOperation(self, opname):
        "Describe the given operation."
        operation = self._opmanager.getOperation(opname)
        print opname
        print operation.__doc__
        #for param in operation.params:
        #    print param.describe()
        
    #-------------------------------------------------------------------------
    def run(self):

        # Parse command-line options.
        options, _ = self.parse_args()

        if options.show_version: self.showVersion()
        if options.show_tools: self.showTools()
        if options.tool_name: self.toolUsage(options.tool_name)
        if options.show_operations: self.showOperations()
        if options.operation_name:
            self.describeOperation(options.operation_name)

