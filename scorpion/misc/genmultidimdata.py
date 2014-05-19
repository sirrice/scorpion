"""
Messatfa - Comparative Study of Clustering Methods
multidimensional data generation

Basic Steps
1) Generate attribute definitions 
   - # attributes
   - name
   - type
   - value range
   - can default to [0,1] for everything
2) Define c clusters
   - split each attribute into c nonoverlapping attributes
   - each cluster doesn't overlap any range
3) Generate
   - same number of tuples per cluster

Extensions
- support values, not just tuple density

"""
import sys
import random
import numpy as np

random.seed(0)
np.random.seed(0)

class Attr(object):
    def __init__(self, name, vals=(0.,100.)):
        self.name = name
        self.vals = vals

    def ranges(self, c):
        mv = self.vals[0]
        vrange = self.vals[1] - self.vals[0]
        rsize = vrange / c
        return [Attr(self.name, vals=(mv+rsize*i, mv+rsize*(i+1)))
                for i in xrange(c)]
        
    def random(self, **kwargs):
        return random.uniform(*self.vals)

    def __str__(self):
        if self.vals[0] != 0 or self.vals[1] != 100: 
            return '(%3d-%3d)' % (self.vals[0], self.vals[1])
        return '(   -   )'


class ValAttr(Attr):

    def ranges(self, c):
        return [ValAttr(self.name, vals=self.vals) for i in xrange(c)]

    def random(self, bad=False):
        mv = self.vals[0]
        vrange = self.vals[1] - mv
        if bad:
            return random.uniform(vrange*0.4+mv, vrange*0.5+mv)#self.vals[1])
        return random.uniform(mv, vrange*0.2+mv)


           
class CDef(object):
    def __init__(self, attrs):
        self.attrs = attrs

    def random(self, **kwargs):
        return tuple(attr.random(**kwargs) for attr in self.attrs)


def gen_attributes(nattrs):
    """
    @param nattrs number of attribute
    """
    attrs = []
    for i in xrange(nattrs):
        attr = 'a_%d' % i
        attrs.append(Attr(attr, vals=(0, 100.)))
    attrs.append(ValAttr('val', vals=(0., 100.)))
    return attrs

def gen_cluster_defs(attrs, c):
    attr_ranges = []
    for attr in attrs:
        attr_ranges.append(attr.ranges(c))

    map(random.shuffle, attr_ranges)
    clusters = []
    for i in xrange(c):
        cdef = CDef(tuple(ar[i] for ar in attr_ranges))
        clusters.append(cdef)

    return clusters

def gen_dirty_cdefs(attrs, c):
    attr_ranges = []
    for attr in attrs:
        attr_ranges.append(attr.ranges(c))

    clusters = []
    for i in xrange(c):
        nattrs = random.randint(1, len(attrs))
        tmp = range(len(attrs))
        random.shuffle(tmp)
        tmp = sorted(tmp[:nattrs])

        cdef_attrs = []
        for idx in xrange(len(attrs)):
            if idx in tmp:
                cdef_attrs.append(random.choice(attr_ranges[idx]))
            else:
                cdef_attrs.append(attrs[idx])
        cdef = CDef(cdef_attrs)
        clusters.append(cdef)

    return clusters

       

def gen_data(cdefs, n, **kwargs):
    for cdef in cdefs[:len(cdefs)-1]:
       for i in xrange(n):
            yield cdef.random()
    cdef = cdefs[-1]
    for i in xrange(n):
        yield cdef.random(**kwargs)

def print_cdefs(cdefs):
    for idx, cdef in enumerate(cdefs):
        params = (idx == len(cdefs)-1 and 'Bad' or 'Good', 
                  '\t'.join(map(str, cdef.attrs)))
        print >>sys.stderr, '%s\t%s' % params

def print_data(idx, pts):
    for pt in pts:
        print '%d\t%s' % (idx, '\t'.join(map(lambda v: '%f' % v, pt)))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print "Generates and prints points in tab delimitied format"
        print "Usage:"
        print "\tpython genmultdimdata.py nattrs nclusters pts/cluster [niters]"
        exit()

    nattrs = int(sys.argv[1])
    nclusters = int(sys.argv[2])
    npts = int(sys.argv[3])
    niter = len(sys.argv) > 4 and int(sys.argv[4]) or 2


    # want to generate good and bad iterations using the same
    # clusters.
    # We should pick good and bad clusters and 
    # generate good and bad values for those clusters

    attrs = gen_attributes(nattrs)
    cdefs = gen_cluster_defs(attrs, nclusters)
    cdefs = gen_dirty_cdefs(attrs, nclusters)

    print_cdefs(cdefs)
    
    print 'iter\t%s' % '\t'.join(map(lambda a: a.name, attrs))
    for idx in xrange(niter):
        print_data(idx, gen_data(cdefs, npts, bad=(idx < niter/2)))
