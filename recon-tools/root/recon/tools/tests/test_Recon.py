from numpy.testing import *
import os
from os import path
from glob import glob
set_package_path()
from recon.tools.Rerun import Rerun
restore_path()



class test_reconstruction(NumpyTestCase):
    """
    This test reconstructs:
       * asems/fieldmaps
       * epidw 1shot, 2shot interleaved, 2shot centric
       * brs
       * non-square matrix epidw with FixTimeSkew
       * GEMS
       * MP_FLASH
    """

    def setUp(self):
        self.pwd = path.split(__file__)[0]
        self.recon_list = glob(self.pwd+'/*.log')
        # do this so that all the asems log-scripts get run first!
        self.recon_list.sort()
        self._fixpaths()

    def _fixpaths(self):
        fname_str = '#filename = '
        clean_ups = []
        for log in self.recon_list:
            f = open(log)
            s = ''
            for ln in f.readlines():
                if ln.find('filename =') >= 0 or ln.find('fmap_file =') >= 0:
                    n = ln.rfind('/data')
                    post = ln[n:]
                    pre = ln[:ln.find('=')+1]
                    #print "fixing:", ln
                    s += pre + ' ' + self.pwd + post

                else:
                    s += ln

            f.close()
            open(log, 'w').write(s)

    def _cleanfiles(self):
        outfiles = glob(self.pwd+'/data/*img') + \
                   glob(self.pwd+'/data/*.nii') + \
                   glob(self.pwd+'/data/*.hdr')
        for ofile in outfiles:
            os.remove(ofile)

    def test_Recon(self, level=10):
        not_installed_msg = "The test data package is not installed, don't test"
        try:
            assert path.exists(path.join(self.pwd, 'data')), not_installed_msg
        except:
            return
        for fname in self.recon_list:
            Rerun(fname).run()
        self._cleanfiles()
    
def generalize_logs():
    log_list = glob('*.log')
    print log_list
    for log in log_list:
        f = open(log)
        s = ''
        for ln in f.readlines():
            if ln.find('filename =') >= 0 or ln.find('fmap_file =') >= 0:
                n = ln.rfind('/data')
                post = ln[n:]
                pre = ln[:ln.find('=')+1]
                print "fixing:", ln
                s += pre + ' .' + post

            else:
                s += ln

        f.close()
        open(log, 'w').write(s)


if __name__ == '__main__':
    generalize_logs()
