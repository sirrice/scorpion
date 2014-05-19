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

from ..util import rm_attr_from_domain, get_logger, instrument
from ..util.table import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *

from rangemerger import *
from crange import *
from adjgraph import AdjacencyGraph

_logger = get_logger()


class StreamRangeMerger(RangeMerger):
  def __init__(self, *args, **kwargs):
    RangeMerger.__init__(self, *args, **kwargs)

    self.bad_tables, self.good_tables, self.full_table = load_data(foobar)
    self.frontier = set()

  def start(self):
    # spin up thread
    while True:
      clusters = self.fetch()
      if clusters is None:
        break
      frontier = self(clusters)
      self.respond(frontier)

  def fetch(self):
    return []
  
  def respond(frontier):
    return 

  @instrument
  def __call__(self, clusters, **kwargs):
    """
    Given a new set of clusters, update internal list of top-k
    Return:
      top-k computed from new clusters (may not return previous top-ks even if they
      are still in the top-k)
    """
    if not clusters:
      return list(self.frontier)

    if self.i == 0:
      self.set_params(**kwargs)
      self.setup_stats(clusters)
      self.adj_graph = self.make_adjacency(clusters, self.partitions_complete)
    self.i += 1

    frontier, removed_clusters = self.get_frontier(clusters + self.frontier)
    frontier.difference_update(self.frontier)

    clusters_set = set()
    start = time.time()
    while len(frontier) > 0:

      new_clusters, removed_clusters = self.expand_frontier(frontier)

      if (not new_clusters.difference(frontier)) or (time.time() - start) > 60:
        clusters_set = set(new_clusters)
        break

      map(self.adj_graph.remove, removed_clusters)
      map(self.adj_graph.insert, new_clusters)
      frontier = new_clusters

    self.learner.merge_stats(self.get_frontier.stats, 'frontier_')
    self.learner.merge_stats(self.get_frontier.heap.stats, 'inter_heap_')
    return list(clusters_set)
  



