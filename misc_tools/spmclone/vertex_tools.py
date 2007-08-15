import numpy as N
import pylab as P
#import sys
from recon import util
from odict import odict

## def segment_polys(pts):
##     polygroups = []
##     polygroups.append([pts[0]])
##     for pt in pts[1:]:
##         for n, group in enumerate(polygroups):
##             # if the point is in this group, add it to the group and stop iter
##             if has_neighbor_in(pt, group):
##                 group.append(pt)
##                 # check to see if it's also in another group, if so merge
##                 # groups (a pt can't belong to 2 groups!)
##                 del_list = []
##                 for m in range(n+1, len(polygroups)):
##                     if has_neighbor_in(pt, polygroups[m]):
##                         polygroups[n] = polygroups[n] + polygroups[m]
##                         #print "merging group",m,"with group",n
##                         del_list.append(m)
##                 for m in (N.array(del_list) - N.arange(len(del_list))):
##                     polygroups.remove(polygroups[m])
##                 break
##             # otherwise if it was the last group tried, make a new group
##             elif group is polygroups[-1]:
##                 polygroups.append([pt])
##                 break
##         #sys.stdout.flush()
##     return polygroups

def segment_polys(pts):
    # an empty list to be filled with hashes keyed by (x,y) points
    polygroups = []
    # make an initial group consisting of the first point
    # the actual list of pts will be polygroups[p].keys()
    #polygroups.append( dict( (pts[0], 1) ) )
    polygroups.append(odict( [ (pts[0], 1) ] ) )
    #polygroups[0][pts[0]] = 1

    for pt in pts[1:]:
        for n, group in enumerate(polygroups):
            # if the point is in this group, add it to the group and stop iter
            if has_neighbor_in(pt, group):
                group[pt] = len(group.keys())+1
                # check to see if it's also in another group, if so merge
                # groups (a pt can't belong to 2 groups!)
                del_list = []
                for m in range(n+1, len(polygroups)):
                    if has_neighbor_in(pt, polygroups[m]):
                        polygroups[n].update(polygroups[m])
                        del_list.append(m)
                for m in (N.array(del_list) - N.arange(len(del_list))):
                    polygroups.remove(polygroups[m])
                break
            # otherwise if it was the last group tried, make a new group
            elif group is polygroups[-1]:
                polygroups.append( odict( [ (pt, 1)] ) )
                #polygroups.append( odict() )
                #polygroups[n+1][pt] = 1
                #polygroups.append( {pt: 1} )
                break
    return [pgrp.keys() for pgrp in polygroups]
        
def has_neighbor_in(pt, hashtab):
##     # make a hash for speed?
##     hashtab = dict(zip(group, xrange(1,len(group)+1)))
    x,y = pt
    if hashtab.get((x-1,y-1), False) or \
           hashtab.get((x,y-1), False) or \
           hashtab.get((x+1,y-1), False) or \
           hashtab.get((x-1,y), False) or \
           hashtab.get((x+1,y), False) or \
           hashtab.get((x-1,y+1), False) or \
           hashtab.get((x,y+1), False) or \
           hashtab.get((x+1,y+1), False):
        return True
    return False

## def has_neighbor_in(pt, group):
## ##     # make a hash for speed?
##     hashtab = dict(zip(group, xrange(1,len(group)+1)))
##     x,y = pt
##     if hashtab.get((x-1,y-1), False) or \
##            hashtab.get((x,y-1), False) or \
##            hashtab.get((x+1,y-1), False) or \
##            hashtab.get((x-1,y), False) or \
##            hashtab.get((x+1,y), False) or \
##            hashtab.get((x-1,y+1), False) or \
##            hashtab.get((x,y+1), False) or \
##            hashtab.get((x+1,y+1), False):
##         return True
##     return False

def get_edge_polys(mask):
    if not mask.sum():
        return []

    cmpmask = N.ones(mask.shape, N.int32) ^ mask
    edgeslr = N.zeros(mask.shape, N.int32)
    #edgeslr[:,1:] = N.diff(mask, axis=-1) & N.diff(cmpmask, axis=-1)
    edgeslr[:,1:] = mask[:,1:]*N.diff(mask, axis=-1)
    edgeslr[:,:-1] |= mask[:,:-1]*N.diff(cmpmask, axis=-1)
    edgesud = N.zeros(mask.shape, N.int32)
    edgesud[1:,:] = mask[1:,:]*N.diff(mask, axis=-2)
    edgesud[:-1,:] |= mask[:-1,:]*N.diff(cmpmask, axis=-2)
    #edgesud[1:,:] |= N.diff(cmpmask, axis=-2)
    edges = edgesud | edgeslr
    #return edges
    #nz = N.transpose(edges).nonzero()
    nz = edges.nonzero()
    # pts[0] will have the y value of the lowest row in the object,
    # and the x value of the minimum x of that row.
    # this should follow for each of the groups
    pts = zip(nz[1],nz[0])
    pgroups = segment_polys(pts)
    edgepolys = []
    for group in pgroups:
        edgepolys.append(edge_walk3(group))
    return edgepolys
            


def edge_walk(pts):
    unsorted_pts = [complex(r,i) for r,i in pts]
    unsorted_hash = dict(zip(unsorted_pts[1:], range(len(unsorted_pts[1:]))))
    sorted_pts = []
    seed = unsorted_pts[0]
    sorted_pts.append(seed)
    def isin(p, A):
        r = A.get(p, -1)
        return r > -1
    dwalk = 1.j # prefer to walk up, then left (try to go clockwise)
    clkws = False
    ddiag = 1. + 1.j
    nturns = 0
    while unsorted_hash.keys():
        if ddiag == (1. + 1.j):
            if isin(seed + dwalk, unsorted_hash):
                nturns = 0
                seed = seed + dwalk
                sorted_pts.append(seed)
                unsorted_hash.pop(seed)
                if not clkws:
                    clkws = True
                # and continue preferrentially in this direction
                continue
            # otherwise try a new direction
            nturns += 1
            dwalk *= 1.j
            # if we're not going clockwise yet, short circuit this rotation
            if dwalk in [1., -1.j] and not clkws:
                nturns = 4
        # if we've made a revolution in the crosswise direction, try
        # skipping around diagonally
        if nturns == 4.0:
            if isin(seed + ddiag, unsorted_hash):
                seed = seed + ddiag
                sorted_pts.append(seed)
                unsorted_hash.pop(seed)
                # start over with crosswise directions
                #dwalk = -1.0
                dwalk = 1.j
                nturns = 0
                ddiag = 1.0 + 1.j
                # continue searching with crosswise directions
                continue
            ddiag *= 1.j
            # if we've made one complete revolution here, we've either
            # finished or walked out onto a line. If the latter, back off
            # and go in a new direction
            if ddiag == (1.0 + 1.j):
                if not clkws:
                    # not even started, so clearly anticlockwise
                    # is the only direction to go?
                    clkws = True
                    dwalk = -1.j
                    nturns = 0
                    ddiag = 1.0 + 1.j
                    continue
                else:
                    sorted_pts.pop(-1)
                    seed = sorted_pts[-1]
                    dwalk = 1.j
                    nturns = 0
    if unsorted_hash.keys():
        print "untouched points:",unsorted_hash.keys()            
    return [(r.real, r.imag) for r in sorted_pts]

def edge_walk2(pts):
    unsorted_pts = [complex(r,i) for r,i in pts]
    unsorted_hash = dict(zip(unsorted_pts[1:], range(len(unsorted_pts[1:]))))
    sweeps = [(1+1j), 1.0, (1-1j), -1j, (-1-1j), -1.0, (-1+1j), 1j]
    #sweeps.reverse()
    rot_gen = {}
    for n in range(0,-len(sweeps),-1):
        rot_gen[sweeps[n]] = sweeps[n-1]
    sorted_pts = []
    seed = unsorted_pts[0]
    sorted_pts.append(seed)
    def isin(p, A):
        r = A.get(p, -1)
        return r > -1
    dwalk = 1.j # prefer to walk up, then left (try to go clockwise)
    circle = 1.j
    nturns = 0
    while unsorted_hash.keys():
        if nturns < 8:
            if isin(seed + dwalk, unsorted_hash):
                nturns = 0
                seed = seed + dwalk
                sorted_pts.append(seed)
                unsorted_hash.pop(seed)
                continue
            else:
                nturns += 1
                dwalk = rot_gen[dwalk]
        else:
            sorted_pts.pop(-1)
            seed = sorted_pts[-1]
            dwalk = 1.j
            nturns = 0

    if unsorted_hash.keys():
        print "untouched points:",unsorted_hash.keys()            
    return [(r.real, r.imag) for r in sorted_pts]
        
def edge_walk3(pts):
    unsorted_pts = [complex(r,i) for r,i in pts[1:]]
    sorted_pts = [pts[0]]
    while unsorted_pts:
        z = abs(N.asarray(unsorted_pts) - complex(*sorted_pts[-1]))
        if (z > 2**0.5).all():
            break
        k = (z==z.min()).nonzero()[0][0]
        tpl = (unsorted_pts[k].real, unsorted_pts[k].imag)
        sorted_pts.append(tpl)
        unsorted_pts.pop(k)
    return sorted_pts

def test_polys(mask, axis=0):
    from matplotlib.patches import Polygon
    if axis != 0:
        mask = N.swapaxes(mask, axis, 0)
    for s,sl in enumerate(mask):
        if sl.sum():
            polys = get_edge_polys(sl)
            ax = P.subplot(111)
            ax.imshow(sl, interpolation='nearest')
            for p in polys:
                ax.add_patch(Polygon(p, facecolor=(0.2, 0.8, 0.0), alpha=0.4))
            P.title("slice %d, axis %d"%(s, axis))
            P.show()
    if axis != 0:
        mask = N.swapaxes(mask, axis, 0)

def test_edges(mask, axis=0):
    from matplotlib.patches import Polygon
    if axis != 0:
        mask = N.swapaxes(mask, axis, 0)
    for sl in mask:
        if abs(sl).sum():
            edges = get_edge_polys(sl)
            ax = P.subplot(111)
            #ax.imshow(sl, interpolation='nearest', cmap=P.cm.gray)
            ax.hold(True)
            ax.imshow(edges, interpolation='nearest')
            #P.colorbar()
            P.show()
    if axis != 0:
        mask = N.swapaxes(mask, axis, 0)

