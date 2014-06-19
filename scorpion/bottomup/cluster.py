import math
import pdb
import random
import numpy as np
import orange
import Orange
import sys
sys.path.extend(['.', '..'])

from itertools import chain, product, groupby
from collections import defaultdict
from operator import sub

from ..util.prob import *
from ..util.misc import valid_number
from ..util.rangeutil import *
from ..learners.cn2sd.rule import SDRule

from bounding_box import *

nan = float('nan')
inf = float('inf')


class Cluster(object):
  _id = 0
  
  def __init__(self, bbox, error, cols, parents=None, discretes=None, **kwargs):
      """
      @param cols the set of continuous columns
      """
      self.error = error
      self.bbox = bbox and (tuple(bbox[0]), tuple(bbox[1])) or ((),())
      self.discretes = defaultdict(set)
      self.cols = cols
      self.parents = parents or []
      self.npts = kwargs.get('npts', 1)
      self.kwargs = kwargs
      self.id = Cluster._id


      self.bad_inf = None
      self.good_inf = None
      self.mean_inf = None
      self.idxs = []

      self.inf_state = None

      # data structures for merger
      # for dominance in merger

      self.inf_range = None

      # lambda c: -> influence value
      self.inf_func = None

      # range of c values this cluster is potentially
      # optimal over
      self.c_range = None
      if self.c_range is None: 
        self.c_range = [-inf, inf]

      self.states = []
      self.cards = []
      self.bad_states = []
      self.bad_cards = []
      self.good_states = []
      self.good_cards = []

      self.rule = None

      Cluster._id += 1

      if discretes:
        for k,vs in discretes.iteritems():
            self.discretes[k].update(vs)

      self.hash = None
      self._bound_hash = None

  @staticmethod
  def from_dict(thedict):
      c = Cluster([], 0., [])
      c.__dict__.update(thedict)
      c.bbox = c.bbox and (tuple(c.bbox[0]), tuple(c.bbox[1])) or ((),())
      c.cols = map(str, c.cols)
      c.discretes = dict((str(k),set([str(v) if isinstance(v,basestring) else v for v in vs])) for k,vs in c.discretes)
      return c

  def __cmp__(self, o):
    if self.error < o.error:
      return -1
    if self.error > o.error:
      return 1
    return len(str(self)) - len(str(o))

  def to_dict(self):
      d = dict(self.__dict__)
      del d['rule']
      del d['inf_func']
      if 'parents' in d:
          del d['parents']
      if 'discretes' in d:
          d['discretes'] = [(k,list(v)) for k,v in self.discretes.iteritems()]
      return d

  @property
  def root(self):
    cur = self
    while cur.parents:
      cur = cur.parents[0]
    return cur

  def __volume__(self):
      return volume(self.bbox)
  volume = property(__volume__)

  def clone(self, copy_rule=False):
      c = Cluster(
        self.bbox, self.error, self.cols,
        parents=self.parents, discretes=self.discretes, 
        **self.kwargs
      )
      c.inf_func = self.inf_func
      c.inf_state = list(self.inf_state)
      c.c_range = list(self.c_range)
      if copy_rule:
        c.rule = self.rule
      return c


  def split_on(self, c):
    """
    Split current cluster via how it intersects with c
    """
    ex_clusters = [] # clusters that don't intersect with c
    in_clusters = [] # clusters that intersect with c
    
    if not volume(intersection_box(self.bbox, c.bbox)):
        return ()

    if self.discretes:
        # they have same keys, just need to diff them
        myd = self.discretes
        od = c.discretes

        intersect_discretes = dict()
        excluded_discretes = dict()
        for key in self.discretes:
            myvals = set(myd[key])
            ovals = set(od[key])
            intersect_discretes[key] = myvals.intersection(ovals)
            excluded_discretes[key] = myvals.difference(ovals)

        if not sum(map(len, intersect_discretes.values())):
            return ()

        if not sum(map(len, excluded_discretes.values())):
            return ([self.clone()], [])


        excluded_cluster = self.clone() 
        excluded_cluster.discretes = excluded_discretes
        ex_clusters.append(excluded_cluster)
    else:
        intersect_discretes = dict(self.discretes)

    if box_contained(self.bbox, c.bbox):
        intersect_cluster = self.clone()
        intersect_cluster.discretes = intersect_discretes
        return ([intersect_cluster], ex_clusters)

    # how the painful part.. splitting the contiuous domain
    inner_mins = map(max, zip(self.bbox[0], c.bbox[0]))
    inner_maxs = map(min, zip(self.bbox[1], c.bbox[1]))
    all_ranges = [[] for i in inner_mins]
    for idx, (bmin, bmax, imin, imax) in enumerate(zip(self.bbox[0], self.bbox[1], inner_mins, inner_maxs)):
        vals = sorted(set([bmin, bmax, imin, imax]))
        for i in xrange(len(vals)-1):
            all_ranges[idx].append((vals[i], vals[i+1]))

    all_boxes = product(*all_ranges)
    groups = defaultdict(list)
    for box in all_boxes:
        new_cluster = self.clone()
        new_cluster.bbox = tuple(zip(*box))
        new_cluster.discretes = dict(intersect_discretes)
        groups[box_contained(new_cluster.bbox, c.bbox)].append(new_cluster)

    ex_clusters.extend(groups.get(False, []))
    in_clusters = groups.get(True, [])
    for ec in ex_clusters:
        ec.parents = (self, c)
    return (list(in_clusters), list(ex_clusters))


  @staticmethod
  def merge(c1, c2, intersecting_clusters=None, min_volume=0):
    """
    Computes the cluster after merging c1 and c2

    If intersecting clusters is not empty, computes a
    weighed estimated error based on the intersecting clusters
    """
    if c1.cols != c2.cols:
      raise RuntimeError("columns do not match!  %s vs %s" % (str(c1.cols), str(c2.cols)))
    
    # Compute the new cluster
    newbbox = bounding_box(c1.bbox, c2.bbox)
    discretes = defaultdict(set)
    for k in chain(c1.discretes.keys(), c2.discretes.keys()):
        discretes[k].update(c1.discretes.get(k, ()))
        discretes[k].update(c2.discretes.get(k, ()))


    if not intersecting_clusters:
      newerror = float('-inf')
    else:
      # compute new error
      weights = []
      errors = []
      for intersection in intersecting_clusters:
          bbox = intersection_box(intersection.bbox, newbbox)
          weights.append(volume(bbox) or min_volume)
          errors.append(intersection.error)

      total_weight = sum(weights) or 1.
      if not total_weight:
          return None

      newerror = sum(e*w for e,w in zip(errors, weights)) / total_weight

    return Cluster(newbbox, newerror, c1.cols, parents=(c1, c2), discretes=discretes)


  @staticmethod
  def from_rule(rule, cols, error=None, parents=None):
    """
    @param cols list of attribute names
    """
    domain = rule.data.domain
    name_to_bounds = {}
    discretes = {}
    
    for c in rule.filter.conditions:
      attr = domain[c.position]
      name = attr.name
      if attr.varType == orange.VarTypes.Discrete:
        if name in cols:
          try:
            discretes[name] = [int(v) for v in c.values]
          except Exception as e:
            print e
            print c.values
      else:
        name_to_bounds[name] = (c.min, c.max)

    mins, maxs = [], []
    cont_cols = []
    for col in cols:
      if col in name_to_bounds:
        mins.append(name_to_bounds[col][0])
        maxs.append(name_to_bounds[col][1])
        cont_cols.append(col)

    bbox = (tuple(mins), tuple(maxs))
    error = error or rule.quality
    cluster = Cluster(bbox, error, cont_cols, parents=parents, discretes=discretes, npts=len(rule.examples))
    cluster.rule = rule
    cluster.c_range = rule.c_range
    cluster.inf_state = rule.inf_state
    return cluster

  def adjacent(self, o, thresh=0.7):
    """@return True if they overlap on one or more continuous attribute, 
        and their discrete attributes intersect"""
    d_intersects =  self.discretes_intersect(o)
    if not d_intersects:
        return False

    # needs to "sufficiently" intersect on N-1 dimensions and
    # at least touch on one dimension
    N = len(self.bbox[0])
    n_intersect = 0
    n_close = 0
    dists = []
    for myb, ob in  zip(zip(*self.bbox), zip(*o.bbox)):
        _min, _max = max(myb[0], ob[0]), min(myb[1], ob[1])
        myrange = myb[1] - myb[0]
        dist = _max - _min
        dists.append(dist)
        if dist > myrange * 0.1:
            n_intersect += 1
        elif dist > -myrange * 0.01:
            n_close += 1

    return n_intersect + n_close == N and n_close <= 1

  def inf_dominates(self, o, alpha=0, c_range=None):
    """
    Sample c values along both influence curves and ensure that
    this cluster's influence values are 
    """
    if c_range is None:
      c_range = r_union(self.c_range, o.c_range)
    xs = ((np.arange(20)/20.) * r_vol(c_range)) + c_range[0]
    mine = self.inf_func(xs)
    theirs = o.inf_func(xs) * (1.0 + alpha)
    return (mine >= theirs).all()

  def contains(self, o, epsilon=0.):
    if not box_contained(o.bbox, self.bbox, epsilon=epsilon):
      return False
    return self.discretes_contains(o)

  def same(self, o, epsilon=0.):
    for key in o.discretes.keys():
      if key in self.discretes:
        diff = set(o.discretes[key]).intersection(self.discretes[key])
        if len(diff) != len(o.discretes[key]) or len(diff) != len(self.discretes[key]):
          return False

    return box_same(o.bbox, self.bbox, epsilon=epsilon)

  def discrete_differences(self, o, epsilon=0.):
    """
    Returns None if o.bbox is different than self.bbox
            otherwise, dictionary of:
              attr -> set(values self doesn't have)
    """
    if not box_same(o.bbox, self.bbox, epsilon=epsilon):
      return None

    ret = {}
    for key, ovals in o.discretes.iteritems():
      vals = ovals.difference(self.discretes.get(key, ()))
      if vals:
        ret[key] = vals
    return ret


  def discretes_contains(self, o):
      for key in o.discretes.keys():
          if key in self.discretes:
              diff = set(o.discretes[key]).difference(self.discretes[key])
              if len(diff):
                  return False
      return True

  def discretes_same(self, o):
      mykeys = set(self.discretes.keys())
      okeys = set(o.discretes.keys())
      if mykeys != okeys:
          return False
      for key in mykeys:
          if set(self.discretes[key]) != set(o.discretes[key]):
              return False
      return True

  def discretes_overlap(self, o):
      myd = self.discretes
      od = o.discretes
      keys = set(myd.keys()).intersection(set(od.keys()))
      
      for key in keys:
          if len(od[key].intersection(myd[key])) == 0:
              return False
      return True

  def discretes_intersect(self, o):
      myd = self.discretes
      od = o.discretes
      keys = set(myd.keys()).intersection(set(od.keys()))
      
      for key in keys:
          if len(od[key].intersection(myd[key])) < min(len(od[key]), len(myd[key])):
              return False
      return True

  def discretes_distance(self, o):
      myd = self.discretes
      od = o.discretes
      keys = set(myd.keys()).union(set(od.keys()))
      
      diff = 0
      for key in keys:
          mvals = myd.get(key, set())
          ovals = od.get(key, set())
          diff += len(mvals) + len(ovals) - len(mvals.intersection(ovals))
      return diff



  def to_rule(self, table, cont_dists=None, disc_dists=None):
    """
    @param cols list of attribute names
    """
    if not self.rule:
      domain = table.domain
      attrnames = [attr.name for attr in domain]
      cont_dists = cont_dists or dict(zip(attrnames, Orange.statistics.basic.Domain(table)))
      disc_dists = disc_dists or dict(zip(attrnames, Orange.statistics.distribution.Domain(table)))
      conds = []

      for col, bound in zip(self.cols, zip(*self.bbox)):
        attr = domain[col]
        pos = domain.index(attr)
        table_bound = cont_dists[attr.name]
        minv, maxv = r_intersect(bound, [table_bound.min, table_bound.max])
        if maxv - minv > 0.99 * (table_bound.max-table_bound.min):
          continue
        
        conds.append(
          orange.ValueFilter_continuous(
              position=pos,
              max=bound[1],
              min=bound[0]
          )
        )

      for disc_name, vidxs in self.discretes.iteritems():
        attr = domain[disc_name]
        disc_pos = domain.index(attr)
        vals = [orange.Value(attr, attr.values[int(vidx)]) for vidx in vidxs if int(vidx) < len(attr.values)]

        if not vals or len(vals) == len(disc_dists[attr.name]):
          continue
        
        conds.append(
          orange.ValueFilter_discrete(
              position = disc_pos,
              values=vals
          )
        )

      rule = SDRule(table, None, conditions=conds)
      self.rule = rule

    rule = self.rule
    rule.quality = rule.score = self.error
    rule.inf_state = self.inf_state
    rule.c_range = self.c_range
    return rule

  def project(self, cols):
      """
      @param cols all of the columns (continuous and discrete) to keep
      """
      newbbox = [[], []]
      newcols = []
      for col, minv, maxv in zip(self.cols, self.bbox[0], self.bbox[1]):
          if col in cols:
              newbbox[0].append(minv)
              newbbox[1].append(maxv)
              newcols.append(col)

      newdiscretes = dict(filter(lambda (k,v): k in cols, self.discretes.iteritems()))
      return Cluster(newbbox, self.error, newcols, parents=self.parents, discretes=newdiscretes)



  def __ancestors__(self):
      if not self.parents:
          return (self,)
      ret = []
      for p in self.parents:
          ret.extend(p.ancestors)
      return ret
  ancestors = property(__ancestors__)
  
  def __str__(self):
      cr = "%.4f-%.4f" % tuple(self.c_range)
      return '%.4f\t%s\t%s' % (self.error, cr, str(self.rule))
      s = '\t'.join(['%s:(%.4f, %.4f)' % (col, bound[0], bound[1]) 
                  for col, bound in zip(self.cols, zip(*self.bbox))])
      d = '\t'.join(["%s:%s (%d els)" % (k, str(list(v)[:3]), len(v)) 
        for k,v in self.discretes.iteritems()])
      return '%.4f\t%s\t%s\t%s' % (self.error, cr, s, d)
      fmt = '\t'.join(['%.4f'] * len(self.bbox[0]))
      args = (self.error, fmt % self.bbox[0], fmt % self.bbox[1], str(self.discretes))
      return '%.6f\t%s\t%s\t%s' % args

  def __hash__(self):
    if self.hash is None:
      self.hash = self.bound_hash
    return self.hash


  @property
  def bound_hash(self):
    if self._bound_hash is None:
      discretes = []
      for dkey in sorted(self.discretes.keys()):
        discretes.append(dkey)
        discretes.append(":::")
        discretes.extend(map(hash, sorted(self.discretes[dkey]))) 
      self._bound_hash = hash((self.bbox, str(discretes)))
    return self._bound_hash

  def __eq__(self, o):
      return hash(self) == hash(o)


def filter_bad_clusters(clusters):
  f = lambda c: c and valid_number(c.error)
  return filter(f, clusters)

def compute_clusters_threshold(clusters, nstds=1.):
    if not clusters:
        return 0.
    errors = [c.error for c in filter_bad_clusters(clusters)]
    return np.percentile(errors, 70)
    npts = [c.npts for c in clusters]
    #npts = [1] * len(clusters)
    mean, std = wmean(errors, npts), wstd(errors, npts)        
    thresh = max(0, min(max(errors), mean + nstds * std))
    thresh = min(max(errors), mean + nstds * std)
    return thresh


def filter_top_clusters(clusters, nstds=1.):
    """
    compute mean and std of error of clusters.  return clusters
    nstds above mean
    """
    if len(clusters) <= 1:
        return clusters
    thresh = compute_clusters_threshold(clusters, nstds)
    f = lambda c: c.error >= thresh
    return filter(f, clusters)

    errors = [c.error for c in clusters]
    minv, maxv, mean, std = min(errors), max(errors), np.mean(errors), np.std(errors)
    thresh = min(maxv, mean + nstds * std)
    thresh = 0.6 * (maxv - minv) + minv
    f = lambda c: c.error >= thresh
    return filter(f, clusters)


def normalize_cluster_errors(clusters):
    if not clusters:
        return clusters
    errors = np.array([c.error for c in clusters])
    mean, std = np.mean(errors), np.std(errors)
    if std == 0: std = 1
    for c in clusters:
      c.error = (c.error - mean) / std
    return clusters
    mine, maxe = min(errors), max(errors)
    div = 1. if maxe == mine else (maxe-mine)

    for c in clusters:
        c.error = (c.error - mine) / div
    return clusters


def clusters_to_rules(clusters, table):
    import Orange
    attrnames = [attr.name for attr in table.domain]
    cont_dists = dict(zip(attrnames, Orange.statistics.basic.Domain(table)))
    disc_dists = dict(zip(attrnames, Orange.statistics.distribution.Domain(table)))
    args = {'cont_dists' : cont_dists,
            'disc_dists' : disc_dists}

    rules = [c.to_rule(table, **args) for c in clusters]

    return rules
