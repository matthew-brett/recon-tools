import numpy as N
try:
    from recon.imageio import ReconImage
    from recon import util
except ImportError:
    # allow import of this module: scanners.varian is needed by setup.py
    # define a null class
    class ReconImage (object):
        def __init__(self):
            pass

class ScannerImage (ReconImage):
    """
    Interface for image data originating from an MRI scanner.
    This class of objects should be able to go through all of the operations,
    especially the artifact correction operations.

    In order to be used with certain Operations, this class guarantees the
    definition of certain system/scanning parameters.

    """

    necessary_params = dict((
        ('T_pe', 'time to scan from k(y=a,x=0) to k(y=a+1,x=0)'),
        ('phi', 'euler angle about z'),
        ('psi', 'euler angle about y'),
        ('theta', 'euler angle about x'),
        ('delT', 'sampling dwell time'),
        ('echo_time', 'time-to-echo'),
        ('asym_times', 'list of te times in an asems scan'),
        ('acq_order', 'order that the slices were acquired'),
        ('nseg', 'number of sampling segments'),
        ('sampstyle', 'style of sampling: linear, centric, interleaved'),
        ('pe0', 'value of the first sampled pe line (normally -N2/2)'),
        ('tr', 'time series step size'),
        ('dFE', 'frequency-encode step size (in mm)'),
        ('dPE', 'phase-encode step size (in mm)'),
        ('dSL', 'slice direction step size (in mm)'),
        ('path', 'path of associated scanner data file'),
    ))

    def __init__(self):
        # should set up orientation info
        self.check_attributes()
        if not hasattr(self, "orientation_xform"):
            self.orientation_xform = util.Quaternion()
        if not hasattr(self, "orientation"):
            self.orientation = ""
        ReconImage.__init__(self, self.data, self.dFE,
                            self.dPE, self.dSL, self.tr,
                            orient_xform=self.orientation_xform,
                            orient_name=self.orientation)
                            

    # may change this this ksp_trajectory and handle different cases
    def epi_trajectory(self, pe0=None):
        M = self.shape[-2]
        # sometimes you want to force pe0
        if not pe0:
            pe0 = self.pe0
        if self.sampstyle == "centric":
            if self.nseg > 2:
                raise NotImplementedError("centric sampling not implemented for nseg > 2")
            a = util.checkerline(M)
            a[:M/2] *= -1
            # offset the pe ordering by pe0, which may or may not be -M/2
            b = N.arange(M) + pe0
            b[:-pe0] = abs(b[:-pe0] + 1)
        else:
            a = N.empty(M, N.int32)
            for n in range(self.nseg):
                a[n:M:2*self.nseg] = 1
                a[n+self.nseg:M:2*self.nseg] = -1
                b = N.floor((N.arange(float(M))+pe0)/float(self.nseg)).astype(N.int32)
        return (a, b) 

    def check_attributes(self):
        for key in ScannerImage.necessary_params.keys():
            if not hasattr(self, key):
                raise AttributeError("This is not a complete ScannerImage, "\
                                     "missing parameter %s"%key)
            

    
    
