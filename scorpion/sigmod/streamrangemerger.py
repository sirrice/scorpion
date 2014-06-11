import json
import math
import pdb
import random
import numpy as np
import sys
import time
sys.path.extend(['.', '..'])

from itertools import chain, repeat, izip
from collections import defaultdict
from operator import mul, and_, or_
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

from ..util import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *

from frontier import *
from rangemerger import RangeMerger2

_logger = get_logger()

class StreamRangeMerger(RangeMerger2):
  def __init__(self, *args, **kwargs):
    super(StreamRangeMerger, self).__init__(*args, **kwargs)
    self.valid_cluster_f = kwargs.get('valid_cluster_f', lambda c: True)

    # idx -> clusters to expand  -- different than clusters on frontier!!
    self.tasks = defaultdict(list)

    # tracks the valid expansions for each frontier cluster
    self.valid_expansions = defaultdict(lambda: defaultdict(lambda: True))

    # stores the frontier after each iteration
    self.added = set()
    self.seen = set()
    self.frontiers = []
    self.adj_graph = None

  def get_frontier_obj(self, version):
    while version >= len(self.frontiers):
      self.frontiers.append(ContinuousFrontier(self.c_range, 0.05))
    return self.frontiers[version]

  def setup_stats(self, clusters):
    all_inf = lambda l: all([abs(v) == float('inf') for v in l])
    clusters = filter(lambda c: c.bound_hash not in self.added, clusters)
    clusters = filter(lambda c: not all_inf(c.inf_state[0]), clusters)
    clusters = filter(lambda c: not all_inf(c.inf_state[2]), clusters)
    self.added.update([c.bound_hash for c in clusters])

    super(StreamRangeMerger, self).setup_stats(clusters)

    start = time.time()
    if not self.adj_graph:
      self.adj_graph = self.make_adjacency([], True)
    self.adj_graph.insert(clusters)
    self.adj_graph.sync()
    self.stats['adj_sync'][0] += time.time() - start
    self.stats['adj_sync'][1] += 1
    return clusters

  def best_so_far(self):
    clusters = set()
    for frontier in self.frontiers:
      clusters.update(frontier.frontier)
    return self.get_frontier(clusters)[0]


  def add_clusters(self, clusters, idx=0):
    """
    Return list of new clusters that are on the frontier
    """
    if not clusters: return []

    if self.DEBUG:
      print "add_clusters"
      self.print_clusters(clusters)

    clusters = self.setup_stats(clusters)
    base_frontier = self.get_frontier_obj(idx)
    clusters, _ = base_frontier.update(clusters)

    if self.DEBUG:
      print "base_frontier"
      self.print_clusters(clusters)

    # clear out current tasks
    self.tasks[idx] = filter(base_frontier.__contains__, self.tasks[idx])
    self.tasks[idx].extend(clusters)

    if idx == 0:
      # remove non-frontier-based expansions from future expansion
      for tidx in self.tasks.keys():
        if tidx == 0: continue
        checker = lambda c: any(map(base_frontier.__contains__, c.ancestors))
        self.tasks[tidx] = filter(checker, self.tasks[tidx])

    return clusters

  @property
  def ntasks(self):
    if not self.tasks: return 0
    return sum(map(len, self.tasks.values()))

  def has_next_task(self):
    if not self.tasks: return False
    return self.ntasks > 0

  def next_tasks(self, n=1):
    ret = []
    for idx in reversed(self.tasks.keys()):
      tasks = self.tasks[idx]
      while len(ret) < n and tasks:
        ret.append((idx, tasks.pop()))
    return ret


  @instrument
  def __call__(self, clusters=[], n=2):
    """
    Return any successfully expanded clusters (improvements)
    """
    self.add_clusters(clusters)
    self.rejected_disc_vals = defaultdict(list)
    self.rejected_cont_vals = defaultdict(set)

    nmerged = self.nmerged
    start = time.time()
    tasks = self.next_tasks(n)
    improvements = set()
    for idx, cluster in tasks:
      if not (idx == 0 or self.valid_cluster_f(cluster)):
        print "merger\tbelow thresh skipping\t %s" % cluster
        continue

      expanded = self.greedy_expansion(cluster, self.seen, idx, None)
      expanded = filter(self.valid_cluster_f, expanded)

      expanded, _ = self.get_frontier_obj(idx).update(expanded)
      frontier, rms = self.get_frontier_obj(idx+1).update(expanded)
      improved_clusters = frontier.difference(set([cluster]))
      if not improved_clusters:
        continue

      self.adj_graph.insert(frontier, version=idx+1)
      self.add_clusters(improved_clusters, idx+1)
      improvements.update(improved_clusters)
    print "merger\ttook %.1f sec\t%d improved\t%d tried\t%d tasks left" % (time.time()-start, len(improvements), self.nmerged-nmerged,self.ntasks)
    return improvements

  def check_direction(self, valid_expansions, dim, direction):
    return valid_expansions[(dim, direction)] 
  
  def update_direction(self, valid_expansions, dim, direction, ok, vals):
    valid_expansions[(dim, direction)] &= ok
    # update rejection state
    for v in vals:
      if direction == 'disc':
        self.rejected_disc_vals[dim].append(set(v))
      if direction == 'inc':
        #v = c.bbox[1][dim]
        self.rejected_cont_vals[(dim, direction)].add(round(v, 1))
      if direction == 'dec':
        #v = c.bbox[0][dim]
        self.rejected_cont_vals[(dim, direction)].add(round(v, 1))
 
  @instrument
  def greedy_expansion(self, cluster, seen, version=None, frontier=None, valid_expansions=None):
    _logger.debug("merger\tgreedy_expand\t%s", cluster)
    if not valid_expansions:
      valid_expansions = defaultdict(lambda: True)
    if not frontier:
      frontier = ContinuousFrontier(self.c_range, 0.05)
      #frontier.frontier = set(self.get_frontier_obj(version).frontier)

    ret = set()
    for dim, direction, vals in self.dims_to_expand(cluster, seen, version=version):
      attrname = isinstance(dim, basestring) and dim or cluster.cols[dim]
      if not self.check_direction(valid_expansions, dim, direction):
        _logger.debug("merger\tnoexpand\t%s\t%s", attrname[:15], direction)
        continue

      tmp = set()
      realvals = self.pick_expansion_vals(cluster, dim, direction, vals)
      for v in realvals:
        if direction == 'inc':
          tmp.add(self.dim_merge(cluster, dim, None, v, seen))
        elif direction == 'dec':
          tmp.add(self.dim_merge(cluster, dim, v, None, seen))
        else:
          tmp.add(self.disc_merge(cluster, dim, v))
      tmp = filter(bool, tmp)

      if direction == 'disc':
        _logger.debug("merger\t%d cands \t%s\t%s", len(tmp), attrname[:15], direction)
      else:
        fmt = lambda v: '%.1f'%v
        _logger.debug("merger\t%d cands \t%s\t%s\t%s -> %s", 
            len(tmp), attrname[:15], direction, 
            str(map(fmt, vals)), str(map(fmt, realvals)))
      if not tmp:
        continue

      cluster.c_range = list(self.c_range)
      expanded, _ = frontier.update(tmp)
      seen.update([c.bound_hash for c in _])
      _logger.debug("merger\t%d improved", len(expanded))

      self.update_direction(valid_expansions, dim, direction, bool(expanded), realvals)
      ret.difference_update(_)
      ret.update(expanded)

      for c in expanded:
        copy = defaultdict(lambda: True)
        copy.update(valid_expansions)
        ret.update(self.greedy_expansion(c, seen, version, frontier, copy))
    return ret



