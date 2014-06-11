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
      return self.get_frontier(clusters)[0]
    return clusters


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

      cur_expanded, _ = self.get_frontier_obj(idx).update(expanded)
      next_expanded, rms = self.get_frontier_obj(idx+1).update(cur_expanded)
      improved_clusters = next_expanded.difference(set([cluster]))
      for c in expanded:
        _logger.debug("merger\texpanded\tcur_idx(%s)\tnext_idx(%s)\t%s", 
            (cluster in cur_expanded), (cluster in next_expanded), 
            c.rule.simplify())

      if not improved_clusters:
        continue

      #self.adj_graph.insert(next_expanded, version=idx+1)
      self.add_clusters(improved_clusters, idx+1)
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




  """
  def n_clauses(self, cluster):
    n = 0
    for idx, col in enumerate(cluster.cols):
      dist = self.learner.cont_dists[col]
      dbound = [dist.min, dist.max]
      cbound = [cluster.bbox[0][idx], cluster.bbox[1][idx]]
      if r_vol(r_intersect(cbound, dbound)) < r_vol(dbound):
        n += 1

    for col, vals in cluster.discretes.iteritems():
      dist = self.disc_dists[col]
      if len(vals) < len(dist.values):
        n += 1
    return n

  @instrument
  def greedy_expansion2(self, cluster, seen, version=None, frontier=None, valid_expansions=None):
    def check(n):
      return not(
        (seen and n.bound_hash in seen) or 
        (n==cluster) or 
        cluster.same(n, epsilon=0.01) or 
        sum(cluster.inf_state[1]) == 0 or
        any([abs(v) == float('inf') for v in cluster.inf_state[0]]) or
        cluster.contains(n) 
      )
    ret = []

    neighbors = self.adj_graph.neighbors(cluster)#, version=version)
    neighbors = filter(check, neighbors)
    xs = ((np.arange(20) / 20.) * r_vol(self.c_range)) + self.c_range[0]
    merged = [Cluster.merge(cluster, n, [], 0) for n in neighbors]
    merged.sort(key=lambda m: abs(self.n_clauses(m) - self.n_clauses(cluster)))
    print "%d neigh\t%s" % (len(neighbors), cluster.rule.simplify())
    for c in merged[:8]:
      c.to_rule(self.learner.full_table)
      c.error = self.influence(c)
      c.c_range = list(self.c_range)
      c.inf_func = self.learner.create_inf_func(c)
      ret.append(c)
      print '\t%s' % c.rule.simplify()
    return ret






    if neighbors > 8:
      xs = ((np.arange(20) / 20.) * r_vol(self.c_range)) + self.c_range[0]
      nconds = np.array([1+abs(self.n_clauses(n)-self.n_clauses(cluster)) for n in neighbors])
      nconds = nconds ** 2
      weights = np.array([np.percentile(n.inf_func(xs), 55) for n in neighbors])
      weights /= nconds
      weights -= weights.min()
      weights += weights.sum() * 0.05
      if weights.sum() == 0: 
        return []
      weights /= weights.sum()
      neighbors = np.random.choice(neighbors, 8, p=weights, replace=False)

    ret = []
    print "%d neigh\t%s" % (len(neighbors), cluster.rule.simplify())
    for n in neighbors:
      seen.add(n.bound_hash)
      self.nmerged += 1
      c = Cluster.merge(cluster, n, [], 0)
      if self.n_clauses(c) == 0:
        continue
      c.to_rule(self.learner.full_table)
      c.error = self.influence(c)
      c.c_range = list(self.c_range)
      c.inf_func = self.learner.create_inf_func(c)
      ret.append(c)
      print '\t%s' % n.rule.simplify()
      print '\t%s' % c.rule.simplify()
    return ret
  """


