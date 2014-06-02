#
# numerous helper methods for comparing influence dominance
# between predicates across a range of c (lambda in job talk)
# values
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


class Frontier(object):
  """
  Iteratively look for the skyline of a list of influence functions
  by identifying intersection points while scanning from left to right


  while heap:
    if init
      for the top cluster at c_range[0]
        find intersection points with all other clusters
        pick next intersection point, add to heap
    else
      swap at intersection point
      find intersection points with all other clusters
      pick next intersection point, add to heap
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

  def __call__(self, clusters):
    """
    Return:
      (frontier, removed)

    """
    if not clusters or len(clusters) <= 1: 
      return set(clusters), set()

    frontier = self.get_frontier(clusters)
    ret = self.frontier_to_clusters(frontier)
    rms = set(clusters).difference(zip(*frontier)[0])
    for rm in rms:
      rm.c_range = [rm.c_range[0], rm.c_range[0]]
    return ret, rms

  
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
        if c2 in ignore: continue

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




class Intersection(object):
  """
  Computes and Caches intersection points
  """
  def __init__(self, bound):
    self.cache = defaultdict(list)
    self.bound = bound
    self.tokey = lambda c1, c2: (min(c1.id, c2.id), max(c1.id, c2.id))
    self.zero_thresh = 1e-7
    self.stats = defaultdict(lambda: [0, 0])

  @instrument
  def __call__(self, c1, c2, minv=None, maxv=None):
    """
    Return next intersection of the two models
           that is greater than minv
    """
    if c1 is None: return None
    if c2 is None: return None


    bound = list(self.bound)
    if minv is not None:
      bound[0] = minv
    if maxv is not None:
      bound[1] = maxv

    # XXX: make a REALLY big assumption that inf functions
    #      are convex/concave 
    xs = ((np.arange(8)/8.) * r_vol(bound)) + bound[0]
    first = c1.inf_func(xs[0]) < c2.inf_func(xs[0])
    same = True
    for x in xs[1:]:
      if (c1.inf_func(x) < c2.inf_func(x)) != first:
        same = False
        break
    if same: return None


    f = lambda v: abs(c1.inf_func(v) - c2.inf_func(v))

    key = self.tokey(c1, c2)
    if key in self.cache:
      roots = self.cache[key]
    else:
      start = time.time()
      roots = fsolve(f, bound[0], maxfev=100)
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



