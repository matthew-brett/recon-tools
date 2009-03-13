from os import path
import numpy as np
import pylab as P
import eddy_corr_utils as eutils
from recon import util
from recon.operations.InverseFFT import InverseFFT
from recon.operations.ReadoutWindow import ReadoutWindow
# methods for comparing different coefficient combinations--
# s1, s1s11, s1s21, s1a0, s1s11s21, s1s11a0, s1s21a0
TAGS = ['s1', 's1s11', 's1s21', 's1a0', 's1s11s21', 's1s11a0', 's1s21a0']
def load_coefs(prefix):
    global TAGS
    cfs = ()
    for t in TAGS:
        cf = np.fromstring(open(path.join(prefix,'cf_'+t)+'.dat').read(), 'd')
        cf.shape = (12,22,4)
        cfs = cfs + (cf,)
    return cfs

def deghost_combos(epi, cf_s1, cf_s1s11, cf_s1s21, cf_s1a0,
                   cf_s1s11s21, cf_s1s11a0, cf_s1s21a0):
    tags = ['s1', 's1s11', 's1s21', 's1a0', 's1s11s21', 's1s11a0', 's1s21a0']
    vols = []
    distorted = np.empty(epi.cdata.shape, epi.cdata.dtype)
    distorted[:] = epi.cdata
    grad = util.grad_from_epi(epi)
    for cf in (cf_s1, cf_s1s11, cf_s1s21, cf_s1a0, cf_s1s11s21, cf_s1s11a0, cf_s1s21a0):
        eutils.deghost_image(epi, grad, cf, 1.0)
        InverseFFT().run(epi)
        epi.combine_channels()
        ReadoutWindow().run(epi)
        d = epi.data.copy()
        vols.append(d)
        epi.cdata[:] = distorted
        epi.use_membuffer(0)
    return tags, vols

def ediff(epi, ref):
    return 100. * ( (np.abs(ref[:]) - np.abs(epi[:]))**2 ).sum(axis=0)/(np.abs(ref[:])**2).sum(axis=0)

def eratio(epi, ref):
    return 10*(np.log10((np.abs(epi[:])**2).sum(axis=0)) - np.log10((np.abs(ref[:])**2).sum(axis=0)))

def adiff(epi, ref):
    return 100. * np.abs(np.abs(ref[:])-np.abs(epi[:])).sum(axis=0)/np.abs(ref[:]).sum(axis=0)

def sdiff(epi, ref):
    r = (np.abs(ref[:])**2).sum(axis=0)
    e = (np.abs(epi[:])**2).sum(axis=0)
    return 100. * (r - e)/r

def g_decibel_diff(epi, ref, gmask):
    g_nrg = ((epi[:]*gmask)**2).sum(axis=-1).sum(axis=-1)
    r_nrg = ((ref[:]*gmask)**2).sum(axis=-1).sum(axis=-1)
    return 10*(np.log10(g_nrg) - np.log10(r_nrg))

def ghost_decibel_by_slice(ref, vols, tags, gmask):
    for v,t in zip(vols, tags):
        P.plot(g_decibel_diff(v, ref, gmask), label=t)
    P.legend()
    return P.gca()


def arr2str(a):
    aup = 10*a
    pwr = 1
    while (aup.max()<=10):
        aup *= 10
        pwr += 1
    aup = np.round(aup).astype('i')
    s = [str(p) for p in aup]
    s2 = []
    for p in s:
        if len(p)>1:
            sn = p[:-1]+'.'+p[-1]
        else:
            sn = '0.'+p
        if pwr>1:
            sn += 'e-'+str(pwr-1)
        s2.append(sn)
    #s = [(p[:-1] + '.' + p[-1]) if len(p)>1 else '0.'+p for p in s]
    return s2


def plot_diffs(diffs, names, max_col=4):
    dpi = 50.
    nd = len(diffs)
    wpix = hpix = 128.
    #ht = hpix + 30 # + buffer for text
    nrow = np.ceil(nd/float(max_col))
    ht = nrow*(hpix + 30) # + buffer for text
    #wd = wpix*len(diffs) + 100 # + buffer for colorbar
    ncol = min(nd, max_col)
    wd = wpix*ncol + 100 # + buffer for colorbar
    figsize = (wd/dpi, ht/dpi)
    fig = P.figure(figsize=figsize, dpi=dpi)
    mxval = max([d.max() for d in diffs])
    mnval = min([d.min() for d in diffs])
    norm = P.normalize(vmin=mnval, vmax=mxval)
    for n in range(nd):
        row = nrow - np.ceil((n+1)/float(max_col))
        col = n%ncol
        ax = fig.add_axes([col*wpix/wd, row*(hpix+30)/ht, wpix/wd, hpix/ht])
        ax.imshow(diffs[n], origin='lower', interpolation='nearest',
                  norm=norm, aspect='auto',
                  cmap=P.cm.spectral)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_frame_on(False)
        fig.text((col+0.5)*wpix/wd, (row*(hpix+30)+hpix+5)/ht,
                 names[n], ha='center')
    print mxval
    cbar = np.multiply.outer(np.linspace(mnval,mxval,100), np.ones(5))
    #ax = fig.add_axes([(len(diffs)*wpix+10)/wd, .1, 30/wd, 100/158.])
    ax = fig.add_axes([(ncol*wpix+10)/wd, .1, 30/wd, 100/158.])
    ax.imshow(cbar, origin='lower', cmap=P.cm.spectral)#, norm=norm)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(True)
    ax.yaxis.set_ticks(np.linspace(0,100,8))
    ax.yaxis.set_ticklabels(arr2str(np.linspace(mnval,mxval,8)))
    ax.yaxis.set_ticks_position('right')
    return fig
    
