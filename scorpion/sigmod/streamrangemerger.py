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

  def __call__(self, clusters):
    if not self.adj_graph:
      self.adj_graph = self.make_adjacency([], True)

    clusters = filter(lambda c: c.bound_hash not in self.seen, clusters)
    self.seen.update([c.bound_hash for c in clusters])

    self.setup_stats(clusters)
    self.adj_graph.insert(clusters)
    self.adj_graph.sync()
    idx = 0
    seen = set()
    ret = set()
    self.learner.update_status("merger running on %d rules" % len(clusters))
    while True:
      clusters, _ = self.get_frontier_obj(idx)(clusters)
      if not clusters:
        break
      self.print_clusters(clusters)
      
      frontier, rms = self.expand_frontier(clusters, seen, version=idx)

      self.print_clusters(frontier)
      self.learner.update_status("expand frontier got %d rules" % len(frontier))
      frontier, rms2 = self.get_frontier_obj(idx+1)(frontier)
      self.learner.update_status("compare w/prev frontier %d rules" % len(frontier))
      self.learner.update_status("%d rules improved" % len(frontier.difference(clusters)))

      if not frontier.difference(clusters):
        ret.update(frontier)
        ret.update(rms)
        ret.update(rms2)
        break
      
      #self.adj_graph.remove(rms, version=idx)
      self.adj_graph.insert(frontier, version=idx+1)
      idx += 1
    self.learner.update_status("merger ran for %d iterations" % idx)
    self.print_clusters(ret)
    return list(ret)

