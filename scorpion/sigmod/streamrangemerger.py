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

    # stores the frontier after each iteration
    self.seen = set()
    self.frontiers = []
    self.adj_graph = None

  def get_frontier_obj(self, version):
    while version >= len(self.frontiers):
      self.frontiers.append(ContinuousFrontier(self.c_range))
    return self.frontiers[version]

  def setup_stats(self, clusters):
    clusters = filter(lambda c: c.bound_hash not in self.seen, clusters)
    self.seen.update([c.bound_hash for c in clusters])

    super(StreamRangeMerger, self).setup_stats(clusters)

    if not self.adj_graph:
      self.adj_graph = self.make_adjacency([], True)
    self.adj_graph.insert(clusters)
    self.adj_graph.sync()

  #frontiers = idx -> ContinuousFrontier

  #cluster = (idx, cluster)

  #queue = [ cluster* ]

  #logically keep a pool of bests, and used_to_be_bests

  #while True:
  #  idx, cluster = queue.pop()
  #  expanded = expandcluster()
  #  expanded = frontiers[idx+1](expanded)
  #  for c in expanded:
  #    queue.push((idx+1, c))

  def best_so_far(self):
    clusters = set()
    for frontier in self.frontiers:
      clusters.update(frontier.frontier)
    print "best so far pre-get_frontier %d" % len(clusters)
    return self.get_frontier(clusters)[0]



  def __call__(self, clusters):
    """
    Return the best clusters seen so far across _all_ calls
    """
    self.setup_stats(clusters)
    self.learner.update_status("merger running on %d rules" % len(clusters))

    idx = 0
    seen = set()
    ret = set()
    clusters, _ = self.get_frontier_obj(idx).update(clusters)

    while clusters:
      self.print_clusters(clusters)
      
      self.learner.update_status("expanding %d rules" % len(clusters))
      expanded, rms = self.expand_frontier(clusters, seen, version=idx)
      self.print_clusters(expanded)

      frontier, rms2 = self.get_frontier_obj(idx+1).update(expanded)
      self.learner.update_status("got expanded %d rules.  %d better than existing.  %d improved over clusters" % (
        len(expanded), len(frontier), len(frontier.difference(clusters))))
      print "forntier %d has %d vals" % (idx+1, len(self.get_frontier_obj(idx+1).frontier))

      if not frontier.difference(clusters):
        break
      
      #self.adj_graph.remove(rms, version=idx)
      self.adj_graph.insert(frontier, version=idx+1)
      clusters = frontier.difference(clusters)
      idx += 1

    self.learner.update_status("merger ran for %d iterations" % idx)
    ret = list(self.best_so_far())
    return ret

