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

    # stores the frontier after each iteration
    self.added = set()
    self.seen = set()
    self.frontiers = []
    self.adj_graph = None

  def get_frontier_obj(self, version):
    while version >= len(self.frontiers):
      self.frontiers.append(ContinuousFrontier(self.c_range))
    return self.frontiers[version]

  def setup_stats(self, clusters):
    all_inf = lambda l: all([abs(v) == float('inf') for v in l])
    clusters = filter(lambda c: c.bound_hash not in self.added, clusters)
    clusters = filter(lambda c: not all_inf(c.inf_state[0]), clusters)
    clusters = filter(lambda c: not all_inf(c.inf_state[2]), clusters)
    self.added.update([c.bound_hash for c in clusters])

    super(StreamRangeMerger, self).setup_stats(clusters)

    if not self.adj_graph:
      self.adj_graph = self.make_adjacency([], True)
    self.adj_graph.insert(clusters)
    self.adj_graph.sync()
    return clusters

  def best_so_far(self):
    clusters = set()
    for frontier in self.frontiers:
      clusters.update(frontier.frontier)
    print "merger\tbest so far %d" % len(clusters)
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


  def __call__(self, clusters=[], n=2):
    """
    Return any successfully expanded clusters (improvements)
    """
    self.add_clusters(clusters)

    print "merger\t%d tasks left" % self.ntasks
    tasks = self.next_tasks(n)
    improvements = set()
    for idx, cluster in tasks:
      if not (idx == 0 or self.valid_cluster_f(cluster)):
        print "merger\tbelow thresh skipping\t %s" % cluster
        continue

      expanded = self.expand(cluster, self.seen, version=idx)

      expanded, _ = self.get_frontier_obj(idx).update(expanded)
      frontier, rms = self.get_frontier_obj(idx+1).update(expanded)
      improved_clusters = frontier.difference(set([cluster]))
      if not improved_clusters:
        continue
      improvements.update(improved_clusters)

      self.adj_graph.insert(frontier, version=idx+1)
      self.add_clusters(improved_clusters, idx+1)
    return improvements
 
