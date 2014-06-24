#
# numerous helper methods for comparing influence dominance
# between predicates across a range of c (lambda in job talk)
# values
#
# 

import json
import math
import pdb
import random
import numpy as np
import sys
import time
from collections import *
from scipy.optimize import fsolve
sys.path.extend(['.', '..'])

from scorpion.util import *

INF = float('inf')


class Frontier(object):
  _id = 0

  """
  Iteratively look for the skyline of a list of influence functions
  by identifying intersection points while scanning from left to right

  The module is actually passed lists of Cluster objects, which implement
  an influence function: 

      c.inf_func(c_value) -> influence value

  while heap:
    if init
      for the top cluster at c_range[0]
        find intersection points with all other clusters
        pick next intersection point, add to heap
    else
      swap at intersection point
      find intersection points with all other clusters
      pick next intersection point, add to heap

  NOTE: picks _at most one_ function on the frontier, not _all_
        functions on the frontier
  """
  def __init__(self, c_range, min_granularity=0):
    """
    Args
      min_granularity: minimum distance between adjacent intersections
    """
    self.c_range = c_range
    self.min_granularity = min_granularity
    self.heap = Intersection(c_range)
    self.stats = defaultdict(lambda: [0,0])
    self.id = Frontier._id 
    Frontier._id += 1

  def __call__(self, clusters):
    """
    Return:
      (frontier, removed)

    """
    if not clusters or len(clusters) <= 1: 
      return set(clusters), set()

    frontier = self.get_frontier(clusters)
    ret = self.frontier_to_clusters(frontier)
    rms = set(clusters)
    if frontier:
      rms.difference_update(zip(*frontier)[0])
    for rm in rms:
      rm.c_range = [rm.c_range[0], rm.c_range[0]]
    return ret, rms

  
  @instrument
  def frontier_to_clusters(self, frontier):
    """
    Return a list of clusters with proper bounds
    avoid cloning cluster if possible
    Args:
      list of (cluster, start, end)
    """
    ret = set()
    for cur, start, end in frontier:
      c = cur.clone(copy_rule=True)
      c.c_range = r_intersect([start, end], cur.c_range)
      ret.add(c)

    return ret

  
  def intersect_ranges(self, c1, c2):
    """
    @deprecated
    Given two clusters, split them up by the bounds where they are each 
    optimal
    """
    frontier = self.get_frontier([c1, c2])
    d = defaultdict(list)
    for c, s, e in frontier:
      d[c.id].append((s, e))
    
    c1s, c2s = [], []

    for (s, e) in d[c1.id]:
      c = c1.clone()
      c.c_range = r_intersect([s, e], c1.c_range)
      c1s.append(c)
    if not c1s:
      c1.c_range = [c1.c_range[0], c1.c_range[0]]

    
    for (s, e) in d[c2.id]:
      c = c2.clone()
      c.c_range = r_intersect([s, e], c2.c_range)
      c2s.append(c)
    if not c2s:
      c2.c_range = [c2.c_range[0], c2.c_range[0]]

    return c1s, c2s



  @instrument
  def get_frontier(self, clusters):
    """
    Groups clusters by non-overlapping c_ranges first
    """
    groups = self.group_by_crange(clusters)
    ret = []
    for group in groups:
      ret.extend(self._get_frontier(group))
    return ret
 
  def _get_frontier(self, clusters):
    """
    Core method that computes increasing intersections
    Rteurn:
      list of (cluster, start, end) 
    """
    # XXX: doesn't work for intersections of >2 models
    if not clusters: return []


    start = time.time()
    init_intersection = self.c_range[0]
    topcluster = max(clusters, key=lambda c: c.inf_func(init_intersection))
    heap = self.heap
    start = time.time()

    ignore = set()
    frontier = [(init_intersection, topcluster)]
    while frontier:
      cur_inter, c1 = frontier[-1]

      if cur_inter + self.min_granularity > self.c_range[1]:
        break

      start = time.time()
      # collect all intersections
      roots = []
      for c2 in clusters:
        if c1 == c2: continue
        #if c2 in ignore: continue

        root = heap(c1, c2, cur_inter + self.min_granularity)
        if root is not None:
          roots.append(root)
        else:
          # how could c2 be best if it doesn't intersect with the top?
          ignore.add(c2)

      cost = time.time() - start
      self.stats['root_scan'][0] += cost
      self.stats['root_scan'][1] += 1

      if not roots: break

      # pick cluster that will be top after the next intersection
      nextroot = min(roots)
      nextc = max(clusters, key=lambda c: c.inf_func(nextroot+1e-10))
      if nextc == c1:
        break

      frontier.append((nextroot, nextc))

    # extend last cluster to the end of the c_range
    frontier.append((self.c_range[1], frontier[-1][1]))

    ret = []
    for idx, (start, c) in enumerate(frontier[:-1]):
      ret.append((c, start, frontier[idx+1][0]))
    return ret


  def group_by_crange(self, clusters):
    """
    return disjoint groups of clusters wrt their c_ranges
    """
    clusters = sorted(clusters, key=lambda c: c.c_range[0])

    # list of: (group, union_c_range)
    groups, ranges = [], []
    for c in clusters:
      found = False
      for idx, (group, c_range) in enumerate(zip(groups, ranges)):
        if r_vol(r_intersect(c_range, c.c_range)):
          group.append(c)
          ranges[idx] = r_union(c_range, c.c_range)
          found = True
          break
      if not found:
        groups.append([c])
        ranges.append(list(c.c_range))
    return groups



class ContinuousFrontier(Frontier):
  """
  Maintains the frontier across a "stream" of cluster objects

  The update() method updates the internal set of clusters on the 
  frontier.
  """


  def __init__(self, c_range, min_granularity=0):
    """
    Args
      min_granularity: minimum distance between adjacent intersections
    """
    Frontier.__init__(self, c_range, min_granularity=min_granularity)

    # persistent frontier across all clusters this has seen
    self.frontier = set()
    self.seen = set()

  def __contains__(self, c):
    if c is None: return False
    if hasattr(c, 'bound_hash'):
      return c.bound_hash in set([c2.bound_hash for c2 in self.frontier])
      #self.seen
    return False

  @instrument
  def intersections_with_frontier(self, cluster):
    for c in self.frontier:
      if r_vol(r_intersect(c.c_range, cluster.c_range)):
        yield c


  @instrument
  def update(self, clusters):
    adds, rms, new_spans = self.probe(clusters)

    new_clusters = self.frontier_to_clusters(new_spans)
    self.frontier.difference_update(rms)
    self.frontier.update(new_clusters)
    self.seen.update([c.bound_hash for c in clusters])

    adds_hashes = set([c.bound_hash for c in clusters])
    adds = [c for c in self.frontier if c.bound_hash in adds_hashes]
    adds = set(adds)
    return adds, rms


  @instrument
  def probe(self, clusters):
    """
    merge clusters in argument with existing frontier
    """
    new_spans = []
    beaten_counts = defaultdict(lambda: 0)
    arg_frontier = set(self(clusters)[0])

    for c in arg_frontier:
      if c.bound_hash in self.seen:
        continue

      c_new_spans = []
      for cand_idx, cand in enumerate(self.frontier):
        if r_vol(r_intersect(c.c_range, cand.c_range)) == 0:
          continue
        
        valid_bound = r_intersect(c.c_range, cand.c_range)
        is_c_best = c.inf_func(valid_bound[0]) > cand.inf_func(valid_bound[0])
        # the following is indexed into using is_c_best
        pair = [cand, c]

        cur_inter = valid_bound[0]
        while cur_inter < valid_bound[1]:
          root = self.heap(c, cand, cur_inter)
          if root is None:
            root = valid_bound[1]

          root = min(valid_bound[1], root)
          if root <= cur_inter: break

          better_cluster = pair[is_c_best]
          c_new_spans.append((better_cluster, cur_inter, root))
          beaten_counts[pair[not is_c_best]] += 1

          cur_inter = root
          is_c_best = not is_c_best

      if not c_new_spans:
        new_spans.append((c, c.c_range[0], c.c_range[1]))
      else:
        new_spans.extend(c_new_spans)

    rms = set([c for c, n in beaten_counts.iteritems() if n > 0])
    new_spans = filter(lambda s: s[2] > s[1], new_spans)
    new_spans = filter(lambda s: s[0] in arg_frontier or s[0] in rms, new_spans)
    adds = set([c for c, minc, maxc in new_spans if c in arg_frontier])

    return adds, rms, new_spans




class CheapFrontier(Frontier):
  """
  Sampling based approximate frontier.  Computes influence values
  at a fixed number of c values for each cluster and uses those to
  pick the frontier.

  In practice, >40 c values gets pretty good results.
  Ends up being faster than the above root-finding-based approaches
  """
  buckets_cache = {}

  def __init__(self, c_range, min_granularity=0, K=3, nblocks=100):
    """
    Args
      min_granularity: minimum distance between adjacent intersections
      K: select top-k at each bucket
    """
    Frontier.__init__(self, c_range, min_granularity=min_granularity)


    self.seen = set()
    self.seen_clusters = set()

    self.nblocks = nblocks
    self.buckets = CheapFrontier.compute_normalized_buckets(nblocks, self.seen_clusters)
    self.buckets = self.buckets * r_vol(c_range) + c_range[0]
    self.nblocks = len(self.buckets)
    self.bests = defaultdict(list)   # bucket -> clusters
    self.K = K

    self.frontier = []
    self.frontier_hashes = []
    self.frontier_infs = []
    self.threshold = np.zeros(self.nblocks).astype(float)

  @staticmethod
  def compute_normalized_buckets(nblocks, clusters=[]):
    """
    return distribution of x buckets, normalized to be between 0 and 1
    """
    if nblocks in CheapFrontier.buckets_cache:
      return CheapFrontier.buckets_cache[nblocks]

    def mkf(pairs):
      return np.vectorize(lambda c: np.mean([t/pow(b,c) for t,b in pairs]) )

    inf_funcs = []
    for c in clusters:
      inf_funcs.append(c.inf_func)

    while len(inf_funcs) < 50:
      tops = np.random.rand(6) * 20
      bots = np.random.rand(6) * 20 + 1.
      pairs = zip(tops, bots)
      inf_funcs.append(mkf(pairs))

    nblocks = int(nblocks)
    xs = np.arange(nblocks * 2).astype(float) / (nblocks * 2.)
    all_infs = np.array([inf_func(xs) for inf_func in inf_funcs])
    medians = np.percentile(all_infs, 50, axis=0)
    block = (medians.max() - medians.min()) / nblocks
    optxs = []
    ys = []
    prev = None
    for idx, v in enumerate(medians):
      if prev == None or abs(v - prev) >= block:
        optxs.append(xs[idx])
        ys.append(v)
        prev = v
    optxs = np.array(optxs)
    optxs -= optxs.min()
    optxs /= optxs.max()

    if len(clusters) > 30:
      CheapFrontier.buckets_cache[nblocks] = optxs
    return optxs

  @instrument
  def cluster_infs(self, c):
    infs = c.inf_func(self.buckets)
    infs[np.abs(infs) == INF] = -1
    return infs

  @instrument
  def compute_thresholds(self, all_infs, thresholds=None, K=None):
    if thresholds is None:
      thresholds = np.zeros(self.nblocks).astype(float)
      thresholds[:] = -1e100
    if K is None:
      K = self.K

    ret = []
    for bidx, bucket in enumerate(self.buckets):
      bucket_infs = all_infs[:, bidx]
      if len(bucket_infs) <= K:
        thresh = bucket_infs.min()
      else:
        idx = np.argpartition(bucket_infs, -K)[-K]
        thresh = bucket_infs[idx]
      thresholds[bidx] = max(thresholds[bidx], thresh)

    return thresholds


  @instrument
  def _get_frontier(self, clusters, K=None):
    if len(clusters) == 0:
      return []
    if len(clusters) == 1:
      return [ (clusters[0], self.c_range[0], self.c_range[1]) ]

    clusters = list(clusters)
    all_infs = [self.cluster_infs(c) for c in clusters]
    try:
      all_infs = np.array(all_infs)
    except Exception as e:
      print e
      print all_infs
    thresholds = self.compute_thresholds(all_infs, K=K)


    # all_infs is a clusters x buckets matrix
    # with a 1 in a cell if a cluster is "best" for
    # that bucket
    return self.all_infs_to_spans(clusters, all_infs, thresholds)

  def all_infs_to_spans(self, clusters, all_infs, thresholds=None):
    ret = []
    for idx, c in enumerate(clusters):
      infs = all_infs[idx]
      ret.extend(self.infs_to_spans(c, infs, thresholds))
    return ret

  @instrument
  def infs_to_spans(self, cluster, infs, thresholds=None):
    if thresholds is None: 
      thresholds = self.thresholds

    ret = []
    passes = infs >= thresholds
    sidx = None
    for bidx in xrange(len(self.buckets)):
      if passes[bidx] == 1:
        if sidx is None:
          sidx = bidx
      else:
        if sidx is not None:
          start = self.buckets[sidx]
          end = self.buckets[bidx-1]
          if bidx < len(self.buckets):
            end = self.buckets[bidx]
          else:
            end = self.c_range[1]
          #print '%.2f-%.2f\t%s' % (start, end, cluster.rule)
          ret.append((cluster, start, end))
          sidx = None

    if sidx is not None:
      start = self.buckets[sidx]
      end = self.c_range[1]
      #print '%.2f-%.2f\t%s' % (start, end, cluster.rule)
      ret.append((cluster, self.buckets[sidx], self.c_range[1]))
    return ret

  def improvement(self, cluster):
    """
    cluster.influence - frontier for each bucket
    """
    infs = self.cluster_infs(cluster)
    if len(self.frontier_infs) == 0:
      return infs
    return infs - self.frontier_infs.max(axis=0)

  @instrument
  def update(self, clusters, K=None):
    clusters = self(clusters)[0]
    if len(clusters) == 0: 
      return set(), set()
    
    clusters = [c for c in clusters if c.bound_hash not in self.seen]

    arg_infs = [self.cluster_infs(c) for c in clusters]
    arg_infs.extend(self.frontier_infs)
    all_infs = np.array(arg_infs)
    all_clusters = list(clusters)
    all_clusters.extend(self.frontier)
    self.thresholds = self.compute_thresholds(all_infs, K=K)
    spans = self.all_infs_to_spans(all_clusters, all_infs, self.thresholds)

    span_hashes = set([tup[0].bound_hash for tup in spans])
    in_hash = lambda c: c.bound_hash in span_hashes

    rms = [c for c in self.frontier if not in_hash(c)]
    adds = [c for c in clusters if in_hash(c)]
    hash2infs = {c.bound_hash : infs for c, infs in zip(all_clusters, all_infs)}

    self.frontier = self.frontier_to_clusters(spans)
    self.frontier_hashes = [c.bound_hash for c in self.frontier]
    self.frontier_infs = []
    for h, c in izip(self.frontier_hashes, self.frontier):
      self.frontier_infs.append(hash2infs.get(h, self.cluster_infs(c)))
    self.frontier_infs = np.array(self.frontier_infs)
    self.frontier_hashes = set(self.frontier_hashes)

    for c in clusters:
      self.seen.add(c.bound_hash)
      if len(self.seen_clusters) < 50:
        self.seen_clusters.add(c)
    return adds, rms

  def __contains__(self, c):
    if c is None: return False
    if hasattr(c, 'bound_hash'):
      return c.bound_hash in self.frontier_hashes
    return False


class Intersection(object):
  """
  Computes and Caches intersection points between cluster curves
  """
  def __init__(self, bound):
    self.cache = defaultdict(list)
    self.bound = bound
    self.tokey = lambda c1, c2: (min(c1.bound_hash, c2.bound_hash), max(c1.bound_hash, c2.bound_hash))
    self.zero_thresh = 1e-7
    self.stats = defaultdict(lambda: [0, 0])
    self.rng = np.arange(5) / 5.

  @instrument
  def __call__(self, c1, c2, minv=None, maxv=None):
    """
    Return next intersection of the two models
           that is greater than minv
    """
    if c1 is None: return None
    if c2 is None: return None
    if abs(c1.error) == float('inf'): return None
    if abs(c2.error) == float('inf'): return None


    bound = list(self.bound)
    if minv is not None:
      bound[0] = minv
    if maxv is not None:
      bound[1] = maxv


    if bound[0] == bound[1]:
      return None

    # XXX: make a REALLY big assumption that inf functions
    #      are convex/concave, thus checking the values at a small
    #      number of points is enough to assess if they intersect
    #      at all
    xs = (self.rng * r_vol(bound)) + bound[0]
    start = time.time()
    c1s = c1.inf_func(xs)
    c2s = c2.inf_func(xs)
    try:
      nbools = (c1s < c2s).sum()
      nequals = (c1s == c2s).sum()
    except:
      return None
    cost = time.time() - start
    self.stats['aprox_check'][0] += cost
    self.stats['aprox_check'][1] += 1
    if nbools == len(xs) or nbools == 0:
      return None

    # the curves are identical
    if nequals > 2:
      return None # wrong to return none

    f = np.vectorize(lambda v: np.abs(c1.inf_func(v) - c2.inf_func(v)))

    key = self.tokey(c1, c2)
    if key in self.cache:
      roots = self.cache[key]
    else:
      start = time.time()
      roots = fsolve(f, bound[0], maxfev=50)
      roots.sort()
      cost = time.time() - start
      self.stats['fsolve_calls'][0] += cost
      self.stats['fsolve_calls'][1] += 1


    if key not in self.cache:
      self.cache[key] = roots

    roots = filter(lambda v: v > bound[0] and v <= bound[1], roots)
    roots = filter(lambda v: abs(f(v)) < self.zero_thresh, roots)
    if roots:
      return roots[0]
    return None



