import os
from numpy.testing import NumpyTest, NumpyTestCase
import numpy as N
from recon import nifti
from recon import imageio

class test_nifti(NumpyTestCase):

    def setUp(self):
        self.image = nifti.readImage('filtered_func_data', vrange=(0,1))


    def test_datatypes(self):
        image = self.image._subimage(2140.*N.random.rand(*self.image.shape).astype(N.float32))
        imscale = 1.0
        print "max value: ", image[:].max()
        for datatype, dt in imageio.recon_output2dtype.items():
            if dt not in nifti.dtype2datatype.keys():
                continue
            image.writeImage('out', format_type='nifti-single',
                             datatype=datatype)
            image2 = nifti.readImage('out')
            imscale2 = image2.scaling
            msg = "failed at dt=%s"%dt
            N.testing.assert_almost_equal(image2[:]*imscale2,
                                          image[:]*imscale,decimal=-1,
                                          err_msg=msg)
        os.remove('out.nii')

if __name__=='__main__':
    NumpyTest(__file__.rstrip(".py")).test()
