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

    # all values for each dimension
    self.all_cont_vals = defaultdict(set) # idx -> values
    self.all_disc_vals = defaultdict(set) # name -> values

    # name -> { val -> # times failed }
    self.failed_disc_vals = defaultdict(lambda: defaultdict(lambda:0))

    # stores the frontier after each iteration
    self.added = set()
    self.seen = set()
    self.frontiers = []
    self.adj_graph = None


    if self.DEBUG:
      self.renderer = InfRenderer('/tmp/merger.pdf', c_range=self.c_range)

  def close(self):
    if self.DEBUG:
      self.renderer.close()

  def get_frontier_obj(self, version):
    while version >= len(self.frontiers):
      self.frontiers.append(ContinuousFrontier(self.c_range, 0.05))
    return self.frontiers[version]

  def setup_stats(self, clusters):
    all_inf = lambda l: all([abs(v) == float('inf') for v in l])
    clusters = filter(lambda c: c.bound_hash not in self.added, clusters)
    clusters = filter(lambda c: not all_inf(c.inf_state[0]), clusters)
    clusters = filter(lambda c: len(c.inf_state[2]) == 0 or not all_inf(c.inf_state[2]), clusters)
    self.added.update([c.bound_hash for c in clusters])

    super(StreamRangeMerger, self).setup_stats(clusters)

    start = time.time()
    if not self.adj_graph:
      self.adj_graph = self.make_adjacency([], True)
    self.adj_graph.insert(clusters)
    self.adj_graph.sync()
    self.stats['adj_sync'][0] += time.time() - start
    self.stats['adj_sync'][1] += 1

    for c in clusters:
      for idx in xrange(len(c.cols)):
        self.all_cont_vals[idx].add(c.bbox[0][idx])
        self.all_cont_vals[idx].add(c.bbox[1][idx])
      for disc, vals in c.discretes.iteritems():
        if len(vals) < 3:
          self.all_disc_vals[disc].update([(v,) for v in vals])
        else:
          self.all_disc_vals[disc].add(tuple(vals))

    return clusters

  def best_so_far(self, prune=False):
    clusters = set()
    for frontier in self.frontiers:
      clusters.update(frontier.frontier)
    if prune:
      clusters = self.get_frontier(clusters)[0]

    if self.DEBUG:
      self.renderer.new_page()
      self.renderer.set_title('best so far')
      self.renderer.plot_active_inf_curves(clusters)
    return clusters


  def add_clusters(self, clusters, idx=0):
    """
    Return list of new clusters that are on the frontier
    """
    if not clusters: return []

    if self.DEBUG:
      print "add_clusters"
      self.print_clusters(clusters)
      self.renderer.new_page()
      self.renderer.set_title("add_clusters %d clusters" % len(clusters))
      for f in self.frontiers:
        self.renderer.plot_inf_curves(f.frontier, color='grey') 
      self.renderer.plot_inf_curves(clusters, color='green')

    clusters = self.setup_stats(clusters)
    base_frontier = self.get_frontier_obj(idx)
    clusters, _ = base_frontier.update(clusters)

    if self.DEBUG:
      print "base_frontier"
      self.print_clusters(clusters)
      self.renderer.plot_active_inf_curves(clusters, color='red')

    # clear out current tasks
    self.tasks[idx] = filter(base_frontier.__contains__, self.tasks[idx])
    self.tasks[idx].extend(clusters)

    if idx == 0:
      # remove non-frontier-based expansions from future expansion
      for tidx in self.tasks.keys():
        if tidx == 0: continue
        checker = lambda c: not any(map(base_frontier.__contains__, c.ancestors))
        self.tasks[tidx] = filter(checker, self.tasks[tidx])

    if clusters:
      _logger.debug("merger:\tadded %d clusters\t%d tasks left", len(clusters), self.ntasks)
    return clusters

  @property
  def ntasks(self):
    if len(self.tasks) == 0: return 0
    return sum(map(len, self.tasks.values()))

  def has_next_task(self):
    if not self.tasks: return False
    return self.ntasks > 0

  def next_tasks(self, n=1):
    ret = []
    for idx in reversed(self.tasks.keys()):
      tasks = self.tasks[idx]
      while len(ret) < n and tasks:
        idx = random.randint(0, len(tasks)-1)
        ret.append((idx, tasks.pop(idx)))
    return ret


  @instrument
  def __call__(self, clusters=[], n=2):
    """
    Return any successfully expanded clusters (improvements)
    """
    self.add_clusters(clusters)

    to_hash = lambda cs: set([c.bound_hash for c in cs])

    nmerged = self.nmerged
    start = time.time()
    tasks = self.next_tasks(n)
    improvements = set()
    for idx, cluster in tasks:
      if not (idx == 0 or self.valid_cluster_f(cluster)):
        print "merger\tbelow thresh skipping\t %s" % cluster
        continue

      if self.DEBUG:
        self.renderer.new_page()
        self.renderer.set_title("expand %s" % str(cluster.rule))
        self.renderer.plot_inf_curves([cluster], color='grey') 

      expanded = self.greedy_expansion(cluster, self.seen, idx, None)
      expanded = [c for c in expanded if c.bound_hash != cluster.bound_hash]
      exp_bounds = to_hash(expanded)

      if self.DEBUG:
        self.renderer.plot_inf_curves(expanded, color='green')

      cur_expanded, rms = self.get_frontier_obj(idx).update(expanded)
      next_expanded, rms2 = self.get_frontier_obj(idx+1).update(cur_expanded)
      cur_bounds = to_hash(cur_expanded)
      next_bounds = to_hash(next_expanded)

      f = lambda c: c.bound_hash != cluster.bound_hash
      improved_clusters = set(filter(f, next_expanded))
      for c in chain(cur_expanded, rms):
        _logger.debug("merger\texpanded\tcur_idx(%s)\tnext_idx(%s)\t%.3f-%.3f\t%s", 
            (c.bound_hash in exp_bounds), 
            (c.bound_hash in next_bounds), 
            c.c_range[0], c.c_range[1],
            c.rule.simplify())
      if self.DEBUG:
        self.renderer.plot_active_inf_curves(self.get_frontier_obj(idx).frontier, color='blue')
        self.renderer.plot_active_inf_curves(self.get_frontier_obj(idx+1).frontier, color='red')

      if not improved_clusters:
        continue

      #self.adj_graph.insert(next_expanded, version=idx+1)
      debug = self.DEBUG
      self.DEBUG = False
      self.add_clusters(improved_clusters, idx+1)
      self.DEBUG = debug
      improvements.update(improved_clusters)
    print "merger\ttook %.1f sec\t%d improved\t%d tried\t%d tasks left" % (time.time()-start, len(improvements), (self.nmerged-nmerged), self.ntasks)
    return improvements



  @instrument
  def dims_to_expand(self, cluster, seen, version=None):
    for idx in xrange(len(cluster.cols)):
      vals = np.array(list(self.all_cont_vals[idx]))
      smaller = vals[(vals < cluster.bbox[0][idx])]
      bigger =  vals[(vals > cluster.bbox[1][idx])]
      yield idx, 'dec', smaller.tolist()
      yield idx, 'inc', bigger.tolist()

    for name, vals in cluster.discretes.iteritems():
      ret = []
      for disc_vals in self.all_disc_vals[name]:
        subset = set(disc_vals).difference(vals)
        subset.difference_update([v for v in subset if self.failed_disc_vals[name][str(v)] > 5])
        ret.append(subset)
      ret = filter(bool, ret)
      ret.sort(key=len)
      yield name, 'disc', ret

    

  def check_direction(self, cluster, dim, direction, vals):
    key = cluster.bound_hash
    if direction == 'disc':
      for subset in self.rejected_disc_vals[dim]:
        if subset.issubset(vals): return []
    if direction == 'inc':
      cont_vals = self.rejected_cont_vals[(dim, direction)]
      if cont_vals:
        vals = filter(lambda v: v > max(cont_vals), vals)
    if direction == 'dec':
      cont_vals = self.rejected_cont_vals[(dim, direction)]
      if cont_vals:
        vals = filter(lambda v: v < min(cont_vals), vals)
    return vals

    if not self.valid_expansions[key][(dim, direction)]: return False
    if cluster.parents:
      return self.check_direction(cluster.parents[0], dim, direction)
    return True
  
  def update_direction(self, cluster, dim, direction, ok, val):
    key = cluster.bound_hash
    self.valid_expansions[key][(dim, direction)] &= ok

    if direction == 'disc':
      for v in list(val):
        self.rejected_disc_vals[dim].append(set([v]))
        self.failed_disc_vals[dim][str(v)] += 1
    if direction == 'inc':
      self.rejected_cont_vals[(dim, direction)].add(round(val, 1))
    if direction == 'dec':
      self.rejected_cont_vals[(dim, direction)].add(round(val, 1))



  @instrument
  def greedy_expansion(self, cluster, seen, version=None, frontier=None, valid_expansions=None):
    _logger.debug("merger\tgreedy_expand\t%s", cluster.rule.simplify())
    if not valid_expansions:
      self.rejected_disc_vals = defaultdict(list)
      self.rejected_cont_vals = defaultdict(set)
      valid_expansions = defaultdict(lambda: True)
    if not frontier:
      frontier = ContinuousFrontier(self.c_range, 0.05)
      frontier.update([cluster])

    cols = cluster.cols
    for dim, direction, vals in self.dims_to_expand(cluster, seen, version=version):
      attrname = isinstance(dim, basestring) and dim or cols[dim]
      vals = self.check_direction(cluster, dim, direction, vals)
      realvals = self.pick_expansion_vals(cluster, dim, direction, vals)

      for v in realvals:
        tmp = None
        if direction == 'inc':
          tmp = self.dim_merge(cluster, dim, None, v, seen)
        elif direction == 'dec':
          tmp = self.dim_merge(cluster, dim, v, None, seen)
        else:
          tmp = self.disc_merge(cluster, dim, v)

        if not tmp: 
          _logger.debug("merger\tnoexpand\t%s\t%s\t%s options", attrname[:15], direction, len(vals))
          continue

        expanded, _ = frontier.update([tmp])
        _logger.debug("merger\tcand\t%s\t%s\t%s\t%s", attrname[:15], direction, bool(expanded), v)
 
        if not expanded:
          seen.add(tmp.bound_hash)
          self.update_direction(cluster, dim, direction, True, v)
          if direction != 'disc':
            break

        cluster = tmp

    for c in frontier.frontier:
      c.c_range = list(self.c_range)
    return frontier.frontier



