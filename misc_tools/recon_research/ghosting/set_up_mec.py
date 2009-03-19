
def mec_setup(mec, fpath, gpath='', fmin_func='fmin',
              axes=range(4), l=1.0, cons_func=None):
    mec.push(dict(fpath=fpath, gpath=gpath))
    mec.execute('from recon import imageio, util')
    mec.execute('from recon.operations.ReorderSlices import ReorderSlices')
    mec.execute('from ghosting import search_driver_full_standalone as sdf')
    mec.execute('epi = imageio.readImage(fpath, vrange=(0,0), N1=128, do_rev=False, use_mmap=False)')
    mec.execute('ReorderSlices().run(epi)')
    mec.execute('import numpy as np')
    mec.execute('from scipy.optimize import brent')
    if gpath:
        mec.execute("mask = np.fromstring(open(gpath).read(), 'd')")
        mec.execute('mask.shape = (22,64,128)')
    else:
        mec.push(dict(mask=None))
    mec.execute('grad = util.grad_from_epi(epi)')
    mec.execute('from ghosting import search')
    mec.execute('from ghosting.search_driver_full_standalone import evalND_full_wrap, corr_func')
    mec.execute('from ghosting.eddy_corr_utils import '+fmin_func)
    #mec.execute('func = sdf.evalND_full_wrap')
    #mec.execute('axes = range(4)')
    mec.push(dict(axes=axes, l=l))
    if cons_func:
        mec.push_function(dict(cons=cons_func))
    else:
        mec.push(dict(cons=None))
    #mec.execute('coefs = np.zeros(4)')
    #mec.execute('args=(axes, epi, grad, coefs, 0, 0, 0, 1.0, None, gmask, None)')
    
