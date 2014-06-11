import math
import pdb
import random
import numpy as np
import sys
import time
import Orange
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
      if attr.var_type == Orange.feature.Type.Discrete:
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
  """
  Stores versions of adjacency graphs and manages insert buffering
  """
  def __init__(self, clusters, domain, cont_dists):
    """
    Args
      domain: orange table domain
    """
    self.feature_mapper = FeatureMapper(domain, cont_dists)
    self.versions = []
    self.insert_bufs = defaultdict(set)

    if clusters:
      self.insert(clusters)
      self.sync()

  def sync(self):
    """
    apply all of the pending inserts and deletes to the existing
    versions
    """
    for idx, v in enumerate(self.versions):
      buf = self.insert_bufs[idx]
      if not buf: continue
      buf.update(v.clusters)
      self.versions[idx] = AdjacencyVersion(self.feature_mapper)
      self.versions[idx].bulk_init(list(buf))
      self.insert_bufs[idx] = set()

  def new_version(self, clusters=None):
    clusters = clusters or []
    v = AdjacencyVersion(self.feature_mapper)
    self.versions.append(v)

  def ensure_version(self, version):
    if version is None: return
    while len(self.versions) <= version:
      self.new_version()
    return self.versions[version]

  def __len__(self):
    return len(self.versions)

  def insert(self, clusters, version=0):
    self.ensure_version(version)
    if not isinstance(clusters, list):
      clusters = list(clusters)

    v = self.versions[version]
    clusters = [c for c in clusters if not v.contains(c)]
    self.insert_bufs[version].update(clusters)
    return len(clusters)

  def remove(self, clusters, version=0):
    self.ensure_version(version)
    if not isinstance(clusters, list):
      clusters = list(clusters)

    v = self.versions[version]
    rms = [v.remove(cluster) for cluster in clusters]
    return rms

  def neighbors(self, cluster, version=None):
    self.ensure_version(version)
    if version is not None:
      return self.versions[version].neighbors(cluster)

    ret = []
    for v in self.versions:
      ret.extend(v.neighbors(cluster))
    return ret


class AdjacencyVersion(object):

  def __init__(self, feature_mapper):
    #self.partitions_complete = partitions_complete
    self.cid = 0
    self.disc_idxs = {}
    self.feature_mapper = feature_mapper
    self.radius = .15
    self.metric = 'hamming'

    self._rtree = None  # internal datastructure
    self._ndim = None
    self.clusters = []
    self.id2c = dict()
    self.c2id = dict()

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
      for idx, col in enumerate(cols):
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
    bbox = self.bbox_rtree(cluster, enlarge=0.01)
    return self._rtree.intersection(bbox)
    res = [self.clusters[idx] for idx in self._rtree.intersection(bbox)]
    return filter(bool, res)

  def bulk_init(self, clusters):
    if not clusters: return

    self.setup_rtree(len(clusters[0].bbox[0]), clusters)
    self.clusters = clusters
    for cid, c in enumerate(clusters):
      self.id2c[cid] = c
      self.c2id[c] = cid
    
    for dim in self.feature_mapper.attrs:
      Xs = []
      for cidx, c in enumerate(clusters):
        Xs.append(self.feature_mapper(c, dim))
      idx = NearestNeighbors(
          radius=self.radius, 
          algorithm='ball_tree', 
          metric=self.metric
      )
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
      if name not in self.disc_idxs:
        return []
      vect = self.feature_mapper(cluster, name)
      index = self.disc_idxs[name]
      dists, idxs = index.radius_neighbors(vect, radius=self.radius)
      idxs = set(idxs[0].tolist())

      if ret is None:
        ret = idxs
      else:
        ret.intersection_update(idxs)
        #ret.update(idxs)
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




