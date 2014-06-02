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
from sklearn.neighbors import NearestNeighbors, DistanceMetric

from ..util import rm_attr_from_domain, get_logger
from ..util.table import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *

_logger = get_logger()

class FeatureMapper(object):
  """
  For discrete features
  """

  def __init__(self, domain, cont_dists):
    self.feature_mappers = {}
    self.cont_dists = cont_dists
    self.ranges = { col: cont_dists[col].max - cont_dists[col].min for col in cont_dists.keys() if cont_dists[col] }
    for attr in domain:
      if 'Contin' not in str(attr):
        self.feature_mappers[attr.name] = self.get_feature_mapper(attr)

  def get_feature_mapper(self, attr):
    name = attr.name
    val2idx = {v:idx for idx,v in enumerate(attr.values)}
    return val2idx

  def __attrs__(self):
    return self.feature_mappers.keys()
  attrs = property(__attrs__)

  def nvals(self, name):
    return len(self.feature_mappers.get(name, []))

  def __call__(self, cluster, name):
    vals = cluster.discretes.get(name, None)
    mapping = self.feature_mappers.get(name, {})
    if vals is None: 
      return np.ones(len(mapping))

    vect = np.zeros(len(mapping)).astype(int)
    for v in vals:
      vect[v] = 1
    return vect

class AdjacencyGraph(object):
  def __init__(self, clusters, domain, cont_dists):
    """
    Args
      domain: orange table domain
    """
    self.feature_mapper = FeatureMapper(domain, cont_dists)
    self.versions = []
    self.insert_buf = clusters or []

    if self.insert_buf:
      self.new_version()

  def new_version(self):
    v = AdjacencyVersion(self.insert_buf, self.feature_mapper)
    self.versions.append(v)
    self.insert_buf = []


  def insert(self, cluster):
    for v in self.versions:
      if v.contains(cluster):
        return
    self.insert_buf.append(cluster)

  def remove(self, cluster):
    for v in self.versions:
      if v.remove(cluster):
        return True
    return False

  def neighbors(self, cluster):
    ret = []
    for v in self.versions:
      ret.extend(v.neighbors(cluster))
    return ret


class AdjacencyVersion(object):

  def __init__(self, clusters, feature_mapper):
    #self.partitions_complete = partitions_complete
    self.cid = 0
    self.disc_idxs = {}
    self.feature_mapper = feature_mapper
    self.radius = .2
    self.metric = 'hamming'

    self._rtree = None  # internal datastructure
    self._ndim = None
    self.clusters = []
    self.id2c = dict()
    self.c2id = dict()
    self.bulk_init(clusters)

  def to_json(self):
    data = {
            'clusters' : [c and c.__dict__ or None for c in self.clusters],
            'id2c' : [(key, c.__dict__) for key, c in self.id2c.items()],
            'c2id' : [(c.__dict__, val) for c, val in self.c2id.items()],
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
        gen_func = ((i, self.bbox_rtree(c, enlarge=0.005), None) for i, c in enumerate(clusters))
        self._rtree = RTree(gen_func, properties=p)
    else:
        self._rtree = RTree(properties=p)
    return self._rtree

  def bbox_rtree(self, cluster, enlarge=0.):
    cols = cluster.cols
    bbox = cluster.bbox
    lower, higher = map(list, bbox)
    if self._ndim == 1:
        lower.append(0)
        higher.append(1)

    if enlarge != 0:
      for idx in xrange(len(lower)):
        col = cols[idx]
        rng = enlarge * self.feature_mapper.ranges[col]
        lower[idx] -= rng
        higher[idx] += rng

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
    bbox = self.bbox_rtree(cluster, enlarge=0.005)
    return self._rtree.intersection(bbox)
    res = [self.clusters[idx] for idx in self._rtree.intersection(bbox)]
    return filter(bool, res)

  def bulk_init(self, clusters):
    if clusters:
      self.setup_rtree(len(clusters[0].bbox[0]), clusters)

    self.clusters.extend(clusters)
    for cid, c in enumerate(clusters):
      self.id2c[cid] = c
      self.c2id[c] = cid
    
    for dim in self.feature_mapper.attrs:
      Xs = []
      for cidx, c in enumerate(clusters):
        Xs.append(self.feature_mapper(c, dim))
      idx = NearestNeighbors(radius=self.radius, algorithm='ball_tree', metric=self.metric)
      self.disc_idxs[dim] = idx
      self.disc_idxs[dim].fit(np.array(Xs))

  def contains(self, cluster):
    return cluster in self.c2id

  def remove(self, cluster):
    if cluster in self.c2id:
      cid = self.c2id[cluster]
      self.remove_rtree(cid, cluster)
      del self.c2id[cluster]
      del self.id2c[cid]
      self.clusters[cid] = None
      return True
    return False


  def neighbors(self, cluster):
    ret = None
    for name, vals in cluster.discretes.iteritems():
      vect = self.feature_mapper(cluster, name)
      dists, idxs = self.disc_idxs[name].radius_neighbors(vect, radius=self.radius)
      idxs = set(idxs[0].tolist())

      if ret is None:
        ret = idxs
      else:
        #ret.intersection_update(idxs)
        ret.update(idxs)
      if not ret: return []

    idxs = self.search_rtree(cluster)
    if ret is None:
      ret = set(idxs)
    else:
      ret.intersection_update(set(idxs))

    return filter(bool, [self.clusters[idx] for idx in ret])


  """
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
    """




