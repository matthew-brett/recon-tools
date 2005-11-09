import os
import varian.lib.ProcPar
import varian.lib.FidFile
import varian.lib.FidImage
import varian.lib.FDFImage

tablib = os.path.join(__path__[0], "tablib")
procpar = varian.lib.ProcPar.Parser
fidfile = varian.lib.FidFile.FidFile
fidimage = varian.lib.FidImage.FidImage
fdfimage = varian.lib.FDFImage

# keep the namespace tidy
del varian, os
