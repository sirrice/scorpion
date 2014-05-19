import math
import pdb
import random
import numpy as np
import sys
import time
sys.path.extend(['.', '..'])

from itertools import chain, repeat
from collections import defaultdict
from rtree.index import Index as RTree
from rtree.index import Property as RProp
from operator import mul, and_, or_

from ..util import rm_attr_from_domain, get_logger
from ..util.table import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *

_logger = get_logger()


class AdjacencyGraph(object):
    def __init__(self, clusters, partitions_complete=True):
        self.partitions_complete = partitions_complete
        self.graph = defaultdict(set)
        self.cid = 0
        self.clusters = []
        self.id2c = dict()
        self.c2id = dict()
        self._rtree = None  # internal datastructure
        self._ndim = None

        self.bulk_init(clusters)

    def to_json(self):
        data = {
                'clusters' : [c and c.__dict__ or None for c in self.clusters],
                'id2c' : [(key, c.__dict__) for key, c in self.id2c.items()],
                'c2id' : [(c.__dict__, val) for c, val in self.c2id.items()],
                'graph' : [(key.__dict__, [val.__dict__ for val in vals]) for key, vals in self.graph.itemsiter()],
                'cid' : self.cid,
                '_ndim' : self._ndim,
                '_rtreename' : 'BLAH'
                }
        return json.dumps(data)

    def from_json(self, encoded):
        data = json.loads(encoded)
        self.clusters = [c and Cluster.from_dict(c) or None for c in data['clusters']]
        self.id2c = dict([(key, Cluster.from_dict(val)) for key, val in data['id2c']])
        self.c2id = dict([(Cluster.from_dict(key), val) for key, val in data['c2id']])
        self.graph = dict([(Cluster.from_dict(key), map(Cluster.from_dict, vals)) for key, vals in data['graph']])
        self.cid = data['cid']
        self._ndim = data['_ndim']
        self._rtree = None

    def setup_rtree(self, ndim, clusters=None):
        if self._rtree:
            return self._rtree

        self._ndim = ndim
        if not ndim:
            class k(object):
                def __init__(self, graph):
                    self.graph = graph
                def insert(self, *args, **kwargs):
                    pass
                def delete(self, *args, **kwargs):
                    pass
                def intersection(self, *args, **kwargs):
                    return xrange(len(self.graph.clusters))
            self._rtree = k(self)
            return self._rtree
 

        p = RProp()
        p.dimension = max(2, ndim)
        p.dat_extension = 'data'
        p.idx_extension = 'index'

        if clusters:
            gen_func = ((i, self.bbox_rtree(c, enlarge=0.00001), None) for i, c in enumerate(clusters))
            self._rtree = RTree(gen_func, properties=p)
        else:
            self._rtree = RTree(properties=p)
        return self._rtree

    def bbox_rtree(self, cluster, enlarge=0.):
        bbox = cluster.bbox
        lower, higher = map(list, bbox)
        if self._ndim == 1:
            lower.append(0)
            higher.append(1)

        if enlarge != 1.:
            lower = [v - enlarge for v in lower]
            higher = [v + enlarge for v in higher]

        bbox = lower + higher
        return bbox

    def insert_rtree(self, idx, cluster):
        self.setup_rtree(len(cluster.bbox[0]))
        self._rtree.insert(idx,self.bbox_rtree(cluster))
        return cluster

    def remove_rtree(self, idx, cluster):
        self.setup_rtree(len(cluster.bbox[0]))
        self._rtree.delete(idx, self.bbox_rtree(cluster))
        return cluster

    def search_rtree(self, cluster):
        self.setup_rtree(len(cluster.bbox[0]))
        bbox = self.bbox_rtree(cluster, enlarge=0.00001)
        res = [self.clusters[idx] for idx in self._rtree.intersection(bbox)]
        return filter(bool, res)

    def bulk_init(self, clusters):
        if clusters:
            self.setup_rtree(len(clusters[0].bbox[0]), clusters)

        self.clusters.extend(clusters)
        for cid, c in enumerate(clusters):
            self.id2c[cid] = c
            self.c2id[c] = cid

        for idx, c in enumerate(clusters):
            for n in self.search_rtree(c):
                if self.c2id[n] <= idx: continue
                if c.discretes_contains(n) and box_completely_contained(c.bbox, n.bbox): continue
                if not c.adjacent(n, 0.8): continue
                self.graph[c].add(n)
                self.graph[n].add(c)



    def insert(self, cluster):
        if cluster in self.graph:
            return

        self.graph[cluster] = set()
        #for o in self.search_rtree(cluster):
        for o in self.graph.keys():
            if cluster == o:
                continue
            if cluster.adjacent(o, 0.8) or (volume(intersection_box(cluster.bbox, o.bbox)) > 0 and not cluster.contains(o)):
                self.graph[cluster].add(o)
                self.graph[o].add(cluster)
        

        cid = len(self.clusters)
        self.clusters.append(cluster)
        self.id2c[cid] = cluster
        self.c2id[cluster] = cid
        self.insert_rtree(cid, cluster)

    def remove(self, cluster):
        if cluster not in self.graph:
            return

        try:
            for neigh in self.graph[cluster]:
                if not neigh == cluster:
                    self.graph[neigh].remove(cluster)
        except:
            pdb.set_trace()
        del self.graph[cluster]

        cid = self.c2id[cluster]
        self.remove_rtree(cid, cluster)
        del self.c2id[cluster]
        del self.id2c[cid]
        self.clusters[cid] = None

    def neighbors(self, cluster):
        if not self.partitions_complete:
            return filter(bool, self.clusters)

        if cluster in self.graph:
            return self.graph[cluster]

        ret = set()
        intersects = self.search_rtree(cluster)
        for key in filter(cluster.adjacent, intersects):
            if box_completely_contained(key.bbox, cluster.bbox):
                continue
            ret.update(self.graph[key])
        return ret



if __name__ == '__main__':
    c1 = Cluster([], 0, [], discretes={'moteid':[0]})
    c2 = Cluster([], 0, [], discretes={'moteid':[0,1]})
#    pdb.set_trace()
    graph = AdjacencyGraph([c1, c2])
    print graph.neighbors(c1)
