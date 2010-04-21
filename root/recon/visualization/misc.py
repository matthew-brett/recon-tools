import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp

def transpose_table(l):
    try:
        if not hasattr(l, '__iter__') or not hasattr(l[0], '__iter__'):
            raise
    except:
        raise ValueError('input is not a nested iterable')
    nrow = len(l)
    ncol = len(l[0])
    nl = []
    try:
        for i in xrange(ncol):
            nl.append( [l[r][i] for r in xrange(nrow)] )
    except IndexError:
        raise ValueError('input is not table-like, dims are inconsistent')
    return nl

def rows_of_plots(n_row, arr_list, img_extent,
                  ax_lims=None, labels=None, cmaps=None, norm=True,
                  transpose_args=False, fig=None):
    """ Take a possibly nested list of 2D arrays and plot them in nrow rows,
    row-by-row

    If there are more rows in the list than n_row, attempt to overlay the
    images.

    Parameters
    ----------
    n_row : int
        number of rows in the figure
    arr_list : iterable
        the arrays to plot.. the length of each row is taken as n_col
    img_extent : iterable
        the extent of the image array box: [xmin, xmax, ymin, ymax]
    ax_lims : iterable
        the limits of the image plot (may extent or crop plot box)
        [xmin, xmax, ymin, ymax]
    labels : iterable
        labels for each plot
    cmaps : iterable
        colormaps for each row or overlay
    transpose_args : bool
        swap rows for columns instead
        
    """
    if len(arr_list) != n_row:
        n_col = len(arr_list)
        arr_list = [arr_list]
    else:
        n_col = len(arr_list[0])

    if labels:
        if len(labels) != n_row:
            labels = [labels]
    else:
        labels = [ [None]*n_col ]*n_row
    cmaps = [ [c]*n_col for c in cmaps] if cmaps \
            else [ [pp.cm.gray]*n_col ]*n_row
    ax_lims = ax_lims or img_extent[:]

    if norm is not False:
        if norm is True:
            mx = np.asarray(arr_list).max()
            mn = np.asarray(arr_list).min()
            norm = mpl.colors.normalize(mn, mx)
        print 'norm min, max:', norm.vmin, norm.vmax
    else:
        norm = None

    if transpose_args:
        arr_list = transpose_table(arr_list)
        labels = transpose_table(labels)
        cmaps = transpose_table(cmaps)
##         old_arr_list = arr_list[:]; old_labels = labels[:]; old_cmaps = cmaps[:]
##         arr_list = []; labels = []; cmaps = []
##         for i in xrange(n_col):
##             arr_list.append( [ old_arr_list[r][i] for r in xrange(n_row) ] )
##             labels.append( [old_labels[r][i] for r in xrange(n_row) ] )
##             cmaps.append( [old_cmaps[r][i] for r in xrange(n_row) ] )
        n_row = len(arr_list)
        n_col = len(arr_list[0])
    print n_row, n_col
    print len(arr_list), len(arr_list[0])
    plot_height = ax_lims[1] - ax_lims[0]; plot_width = ax_lims[3] - ax_lims[2]
    # normalize to 1" for the maximum dim
    mx_dim = float(max(plot_height, plot_width))
    plot_height /= mx_dim
    plot_width /= mx_dim

    if not fig:

        fig_width = n_col * plot_width
        fig_height = n_row * (plot_height + .25)

        fig = pp.figure(figsize = (fig_width*2, fig_height*2))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=.95,
                            wspace=0, hspace=.15)
    
    

    for r, row_list, row_labels, row_cmaps in zip(xrange(n_row), arr_list,
                                                  labels, cmaps):
        ax_nums = xrange(1 + r*n_col, 1 + (r+1)*n_col)
        for x, arr, txt, cmap in zip(ax_nums, row_list, row_labels, row_cmaps):
            print 'making plot', x
            #sax = mpl.axes.Subplot(fig, n_row, n_col, x)
            ax = fig.add_subplot(n_row, n_col, x)
            ax.imshow(arr, extent=img_extent, origin='upper',
                      cmap=cmap, interpolation='nearest', norm=norm)
            ax.axis('off')
            ax.set_xlim(ax_lims[:2]); ax.set_ylim(ax_lims[2:])
            if txt:
##                 bb = ax.get_position()
##                 tpos = ( (bb.x1+bb.x0)/2, bb.y1 + .05)
##                 print 'adding text at', tpos
##                 fig.text(tpos[0], tpos[1], txt, size=8,
##                          ha='center', weight='bold')
                tpos = ( (ax_lims[0] + ax_lims[1])/2., ax_lims[-1] * 1.05 )
                ax.text(tpos[0], tpos[1], txt, size=8,
                        ha='center', weight='bold')
    return fig


def find_image_threshold(arr, percentile=90., debug=False):
    nbins = 200
    bsizes, bpts = np.histogram(arr.flatten(), bins=nbins)
    # heuristically, this should show up near the middle of the
    # second peak of the intensity histogram
    start_pt = np.abs(bpts - arr.max()/2.).argmin()
    db = np.diff(bsizes[:start_pt])
##     zcross = np.argwhere((db[:-1] < 0) & (db[1:] >= 0)).flatten()[0]
    bval = bsizes[1:start_pt-1][ (db[:-1] < 0) & (db[1:] >= 0) ].min()
    zcross = np.argwhere(bval==bsizes).flatten()[0]
    thresh = (bpts[zcross] + bpts[zcross+1])/2.
    # interpolate the percentile value from the bin edges
    bin_lo = int(percentile * nbins / 100.0)
    bin_hi = int(round(percentile * nbins / 100.0 + 0.5))
    p_hi = percentile - bin_lo # proportion of hi bin
    p_lo = bin_hi - percentile # proportion of lo bin
##     print bin_hi, bin_lo, p_hi, p_lo
    pval = bpts[bin_lo] * p_lo + bpts[bin_hi] * p_hi
    if debug:
        import matplotlib as mpl
        import matplotlib.pyplot as pp
        f = pp.figure()
        ax = f.add_subplot(111)
        ax.hist(arr.flatten(), bins=nbins)
        l = mpl.lines.Line2D([thresh, thresh], [0, .25*bsizes.max()],
                             linewidth=2, color='r')
        ax.add_line(l)
        ax.xaxis.get_major_formatter().set_scientific(True)
        f = pp.figure()
        norm = pp.normalize(0, pval)
        ax = f.add_subplot(211)
        plot_arr = arr
        while len(plot_arr.shape) > 2:
            plot_arr = plot_arr[plot_arr.shape[0]/2]
        ax.imshow(plot_arr, norm=norm)
        ax = f.add_subplot(212)
        simple_mask = (plot_arr < thresh)
        ax.imshow(np.ma.masked_array(plot_arr, mask=simple_mask), norm=norm)
        pp.show()
    
    return thresh, pval


