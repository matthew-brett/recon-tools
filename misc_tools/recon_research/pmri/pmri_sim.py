from recon import util
from recon.operations.InverseFFT import InverseFFT
from recon.operations.ReadoutWindow import ReadoutWindow
from recon.pmri import grappa_recon as gr
from ghosting.fwd_calc import perturb_image_noghosts_constFE, distortion_kernel
import numpy as np
import os
from recon.imageio import readImage
from ghosting.eddy_corr_ana import plot_rows_of_slices, arr2str
from glob import glob

def grappa_sampling(n_pe, a, n_acs):
    n_samp_lines = (n_pe-1)/a
    samp_lines = np.arange(1, a * n_samp_lines, a)
    k0 = 1 + a * (n_samp_lines/2)
    acs_lines = np.arange(k0-n_acs/2, k0+n_acs/2)
    return samp_lines, acs_lines

    

def grappa_sample_gre(gre, accel, n_acs=24, xleave_acs=False,
                      distort=False, fmap=None, **dist_kw):
    """down-samples a full res gradient echo image to simulate a
    GRAPPA image.

    Parameters
    __________
    gre : ReconImage
        fully sampled image
    accel : int
        corresponds to the number of skipped phase encodes
    n_acs : int, optional
        number of ACS lines to simulate
    distort : {True, False}, optional
        apply simulated B0 inhomogeneity phase distortion to ACS and data
    fmap : ndarray, optional
        B0 field phase map
    **dist_kw : optional
        keyword arguments to use in "perturb_image_noghosts_constFE"
    

    Returns
    _______
    None

    Notes
    _____
    The gre image will:
     * downsampled in gre.cdata
     * have new arrays appended--'cacs_data' (a TempMemmapArray) and 'acs_data'
     * have a new attribute 'accel'
     * have a new attribute 'n2_sampling'
    """

    full_shape = list(gre.cdata.shape)
    n_chan, n_pe = full_shape[0], full_shape[3]
    acs_shape = full_shape[:]
    acs_shape[3] = n_acs
    acs_shape[1] = 1

    cdata_creator = np.empty if not isinstance(gre.cdata, util.TempMemmapArray)\
                    else util.TempMemmapArray
    
    #cacs_data = util.TempMemmapArray(tuple(acs_shape), gre.cdata.dtype)
    cacs_data = cdata_creator(tuple(acs_shape), gre.cdata.dtype)
    samp_lines,acs_lines = grappa_sampling(n_pe, accel, n_acs)
    acs_sl = [slice(None)] * len(full_shape)
    acs_sl[1] = slice(0,1)
    acs_sl[3] = slice(acs_lines[0], acs_lines[-1]+1, 1)
    if distort:
        print 'distorting ACS data'
        # backup can be TempMemmapArray in any case.. this is sensible
        cdata_bkp = util.TempMemmapArray(gre.cdata.shape, gre.cdata.dtype)
        cdata_bkp[:] = gre.cdata
        cdata_bkp.flush()
        gre.cdata = np.asarray(gre.cdata[acs_sl].copy())
        # if simulating interleaved ACS acquisition, then distort the ACS
        # block in the same manner as the data, with R=accel
        if xleave_acs:
            dist_kw['accel'] = accel
        acs_offset = samp_lines.tolist().index(acs_lines[0])
        perturb_image_noghosts_constFE(gre.cdata, fmap,
                                       pe_offset=acs_offset,
                                       n2_freqs=acs_lines - n_pe/2,
                                       **dist_kw)
        cacs_data[:] = gre.cdata
        #gre.cdata = util.TempMemmapArray(cdata_bkp.shape, cdata_bkp.dtype)
        gre.cdata = cdata_creator(cdata_bkp.shape, cdata_bkp.dtype)
        gre.cdata[:] = cdata_bkp
        del cdata_bkp
    else:
        cacs_data[:] = gre.cdata[acs_sl].reshape(acs_shape)
    try:
        cacs_data.flush()
    except:
        pass

    if distort:
        print 'distorting parallel acquired data'
        # now apply a simulated distortion for parallel acquisition..
        # acq. always starts on row=1, so set pe_offset=-1/R (such that tn[1]=0)
        dist_kw['accel'] = accel
        perturb_image_noghosts_constFE(gre.cdata[:], fmap,
                                       pe_offset=-1.0/accel, **dist_kw)
        
    n2_sampling = slice(samp_lines[0], samp_lines[-1]+1, accel)
    clear_sl = [slice(None)] * len(full_shape)
    for start in range(accel):
        if start != samp_lines[0]:
            clear_sl[-2] = slice(start, n_pe, accel)
            gre.cdata[clear_sl] = 0.

    gre.cacs_data = cacs_data
    gre.n2_sampling = n2_sampling
    gre.accel = accel
    gre.n_acs = n_acs
    gre.use_membuffer(0)
    return

def grappa_recon_sim(nblocks, *args, **kwargs):
    grappa_sample_gre(*args, **kwargs)
    gre = args[0]
    #gr.GrappaSynthesize(nblocks=nblocks, sliding=False, ft=False).run(gre)
    gr.basic_grappa_1D(gre, nblocks)
    InverseFFT().run(gre)
    gre.combine_channels()
    if not (gre.isize==gre.jsize or gre.idim==gre.jdim):
        ReadoutWindow().run(gre)
    return
    
def error_img(path, pct=True):
    dr, fn = os.path.split(os.path.abspath(path))
    ref_path = os.path.join(dr, 'reference.nii')
    ref = readImage(ref_path)
    img = readImage(path)
    rrms = (((ref[:]-img[:])**2).sum() / (ref[:]**2).sum()) ** 0.5
    if pct:
        return img._subimage(100*np.abs(ref[:] - img[:])/ref[:]), rrms
    else:
        return img._subimage(np.abs(ref[:] - img[:])), rrms

def plot_errs(plist, sl=10, mask=None, pct=True):
    p_diffs = []
    p_names = []
    for p in plist:
        eimg, rrms = error_img(p, pct=pct)
        p_diffs.append(eimg[sl])
        rrms_str = arr2str(np.array([rrms]))[0]
        #p_diffs.append(error_img(p, pct=pct)[sl])
        if mask is not None:
            p_diffs[-1] *= mask
        p_names.append(os.path.splitext(os.path.split(p)[-1])[0]+ ': '+rrms_str)
    return plot_rows_of_slices(p_diffs, p_names)

def plot_all_errs(directory, sl=10, mask=None, pct=True):
    plist = glob(os.path.join(directory, '*.nii'))
    plist.remove(os.path.join(directory, 'reference.nii'))
    plot_errs(plist, sl=sl, mask=mask, pct=pct)
                 

def undistort_grappa(img, fmap, Tl=500., mask=True):
    if mask:
        m = np.ones_like(fmap)
        np.putmask(m, fmap==0, 0)
    else:
        m = None
    n_chan, n_vol, n_slice, N2, N1 = img.cdata.shape
    R = float(img.accel)
    n2_ax = np.linspace(-N2/2, N2/2-1, N2)
    t_n2 = (-1/R + np.arange(N2)/R) * Tl * 1e-6
    print t_n2
    util.ifft1(img.cdata, inplace=True, shift=True)
    for s in xrange(n_slice):
        print 'undistorting slice', s
        fm = distortion_kernel(fmap[s], t_n2, n2_ax, m=(m[s] if mask else None))
        for q1 in xrange(N1):
            S = img.cdata[:,:,s,:,q1].transpose(2,0,1).copy()
            S.shape = (N2,n_chan*n_vol)
            St = util.regularized_solve(fm[q1], S, 2.0)
##             S = img.cdata[:,:,s,:,q1].copy()
##             S.shape = (n_chan*n_vol,N2)
##             if (fm[q1]*fm[q1].conjugate()).real.sum() == 0:
##                 St = S
##             else:
##                 [u,s,vt] = np.linalg.svd(fm[q1], 0, 1)
##                 St = np.array([util.regularized_solve_lcurve(fm[q1],Sc,u=u,s=s)
##                                for Sc in S])
##             img.cdata[:,:,s,:,q1] = St.reshape(n_chan, n_vol, N2)
            St.shape = (N2, n_chan, n_vol)
            img.cdata[:,:,s,:,q1] = St.transpose(1,2,0)
            
    util.fft1(img.cdata, inplace=True, shift=True)

def undistort_reg(img, fmap, Tl=500., mask=True, sl=-1):
    if mask:
        m = np.ones_like(fmap)
        np.putmask(m, fmap==0, 0)
    else:
        m = None
    n_chan, n_vol, n_slice, N2, N1 = img.cdata.shape
    n2_ax = np.linspace(-N2/2, N2/2-1, N2)
    t_n2 = np.arange(N2) * Tl * 1e-6
    print t_n2
    util.ifft1(img.cdata, inplace=True, shift=True)
    srange = xrange(n_slice) if sl < 0 else [sl]
    for s in srange:
        print 'undistorting slice', s
        fm = distortion_kernel(fmap[s], t_n2, n2_ax, m=(m[s] if mask else None))
        for q1 in xrange(N1):
            S = img.cdata[:,:,s,:,q1].transpose(2,0,1).copy()
            S.shape = (N2,n_chan*n_vol)
            St = util.regularized_solve(fm[q1], S, 2)
##             S = img.cdata[:,:,s,:,q1].copy()
##             S.shape = (n_chan*n_vol,N2)
##             if (fm[q1]*fm[q1].conjugate()).real.sum() == 0:
##                 St = S
##             else:
##                 [u,sv,vt] = np.linalg.svd(fm[q1], 0, 1)
##                 St = np.array([util.regularized_solve_lcurve(fm[q1],Sc,u=u,s=sv,vt=vt)
##                                for Sc in S])
##             img.cdata[:,:,s,:,q1] = St.reshape(n_chan, n_vol, N2)
            St.shape = (N2, n_chan, n_vol)
            img.cdata[:,:,s,:,q1] = St.transpose(1,2,0)
    util.fft1(img.cdata, inplace=True, shift=True)
    
