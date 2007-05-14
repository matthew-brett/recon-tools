import numpy as N
import pylab as P
#import sys
from recon import util

def segment_polys(pts):
    polygroups = []
    polygroups.append([pts[0]])
    for pt in pts[1:]:
        for n, group in enumerate(polygroups):
            # if the point is in this group, add it to the group and stop iter
            if has_neighbor_in(pt, group):
                group.append(pt)
                # check to see if it's also in another group, if so merge
                # groups (a pt can't belong to 2 groups!)
                del_list = []
                for m in range(n+1, len(polygroups)):
                    if has_neighbor_in(pt, polygroups[m]):
                        polygroups[n] = polygroups[n] + polygroups[m]
                        #print "merging group",m,"with group",n
                        del_list.append(m)
                for m in (N.array(del_list) - N.arange(len(del_list))):
                    polygroups.remove(polygroups[m])
                break
            # otherwise if it was the last group tried, make a new group
            elif group is polygroups[-1]:
                polygroups.append([pt])
                break
        #sys.stdout.flush()
    return polygroups

def has_neighbor_in(pt, group):
    # make a hash for speed?
    hashtab = dict(zip(group, xrange(1,len(group)+1)))
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

def get_edge_polys(mask):
    if not mask.sum():
        return []

    cmpmask = N.ones(mask.shape, N.int32) ^ mask
    edgeslr = N.zeros(mask.shape, N.int32)
    #edgeslr[:,1:] = N.diff(mask, axis=-1) & N.diff(cmpmask, axis=-1)
    edgeslr[:,1:] = mask[:,1:]*N.diff(mask, axis=-1)
    edgeslr[:,:-1] += mask[:,:-1]*N.diff(cmpmask, axis=-1)
    edgesud = N.zeros(mask.shape, N.int32)
    edgesud[1:,:] = mask[1:,:]*N.diff(mask, axis=-2)
    edgesud[:-1,:] += mask[:-1,:]*N.diff(cmpmask, axis=-2)
    #edgesud[1:,:] |= N.diff(cmpmask, axis=-2)
    edges = edgesud | edgeslr
    #return edges
    nz = edges.nonzero()
    pts = zip(nz[1],nz[0])
    pgroups = segment_polys(pts)
    edgepolys = []
    for group in pgroups:
        edgepolys.append(edge_walk(group))
    return edgepolys
            


def edge_walk(pts):
    unsorted_pts = [complex(r,i) for r,i in pts]
    unsorted_hash = dict(zip(unsorted_pts[1:], range(len(unsorted_pts[1:]))))
    sorted_pts = []
    seed = unsorted_pts[0]
    sorted_pts.append(seed)
    #unsorted_hash.pop(seed)    
    def isin(p, A):
        r = A.get(p, -1)
        return r > -1
    dwalk = -1.0 # prefer to walk left
    ddiag = 1. + 1.j
    nturns = 0
    while unsorted_hash.keys():
        if ddiag == (1. + 1.j):
            if isin(seed + dwalk, unsorted_hash):
                nturns = 0
                seed = seed + dwalk
                sorted_pts.append(seed)
                unsorted_hash.pop(seed)
                # and continue preferrentially in this direction
                continue
            # otherwise try a new direction
            nturns += 1
            dwalk *= 1.j
        # if we've made a revolution in the crosswise direction, try
        # skipping around diagonally
        if nturns == 4.0:
            if isin(seed + ddiag, unsorted_hash):
                seed = seed + ddiag
                sorted_pts.append(seed)
                unsorted_hash.pop(seed)
                # start over with crosswise directions
                dwalk = -1.0
                nturns = 0
                ddiag = 1.0 + 1.j
                # continue searching with crosswise directions
                continue
            ddiag *= 1.j
            # if we've made one complete revolution here, we've either
            # finished or walked out onto a line. If the latter, back off
            # and go in a new direction
            if ddiag == (1.0 + 1.j):
                sorted_pts.pop(-1)
                seed = sorted_pts[-1]
                dwalk = -1.0
                nturns = 0
    #if unsorted_hash.keys():
    #    print "untouched points:",unsorted_hash.keys()            
    return [(r.real, r.imag) for r in sorted_pts]
        

