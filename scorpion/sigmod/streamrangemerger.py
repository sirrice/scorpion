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

    # all values for each dimension
    self.all_cont_vals = defaultdict(set) # idx -> values
    # name -> { val -> [sum, count] }
    self.all_disc_vals = defaultdict(lambda: defaultdict(lambda: [0,0])) 

    # name -> { val -> # times failed }
    self.failed_disc_vals = defaultdict(lambda: defaultdict(lambda:0))

    # stores the frontier after each iteration
    self.added = set()
    self.seen = set()
    self.frontiers = []
    self.adj_graph = None


    self.K = 1
    self.nblocks = 50

    if len(self.learner.full_table) < 40000:
      self.K = 2
      self.nblocks = 60
    if len(self.learner.full_table) < 10000:
      self.nblocks = 100

    self.get_frontier = CheapFrontier(self.c_range, K=self.K, nblocks=self.nblocks)
    self.get_frontier.stats = self.stats


    if self.DEBUG:
      self.renderer = InfRenderer('/tmp/merger.pdf', c_range=self.c_range)

  def close(self):
    if self.DEBUG:
      self.renderer.close()

  def get_frontier_obj(self, version):
    while version >= len(self.frontiers):
      frontier = CheapFrontier(self.c_range, K=self.K, nblocks=self.blocks)
      frontier.stats = self.stats
      self.frontiers.append(frontier)
    return self.frontiers[version]

  @property
  def frontier_iter(self):
    return list(self.frontiers)


  @instrument
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
          vals = [(v,) for v in vals]
        else:
          vals = [tuple(vals)]
        for v in vals:
          self.all_disc_vals[disc][v][0] += c.inf_func(0.1)
          self.all_disc_vals[disc][v][1] += 1
        #self.all_disc_vals[disc].update(vals)

    return clusters

  @instrument
  def best_so_far(self, prune=False):
    clusters = set()
    for frontier in self.frontier_iter:
      clusters.update(frontier.frontier)

    if prune:
      for c in clusters:
        c.c_range = list(self.c_range)
      clusters = self.get_frontier(clusters)[0]
      clusters = filter(lambda c: r_vol(c.c_range), clusters)

    if self.DEBUG:
      self.renderer.new_page()
      self.renderer.set_title('best so far')
      self.renderer.plot_active_inf_curves(clusters)
    return clusters

  @instrument
  def best_at_c(self, c_val, K=6):
    clusters = set()
    for frontier in self.frontier_iter:
      clusters.update(frontier.seen_clusters)

    clusters = sorted(clusters, key=lambda c: c.inf_func(c_val), reverse=True)[:K]
    return clusters

  @instrument
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
      for f in self.frontier_iter:
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
    for tkey in reversed(self.tasks.keys()):
      tasks = self.tasks[tkey]
      while len(ret) < n and tasks:
        idx = random.randint(0, len(tasks)-1)
        ret.append((idx, tasks.pop(idx)))
    return ret



  @instrument
  def __call__(self, n=2):
    """
    Return any successfully expanded clusters (improvements)
    """
    nmerged = self.nmerged
    start = time.time()
    tasks = self.next_tasks(n)
    improvements = set()

    for idx, cluster in tasks:
      cur_frontier = self.get_frontier_obj(idx)
      next_frontier = self.get_frontier_obj(idx+1)
      new_clusters = self.run_task(idx, cluster, cur_frontier, next_frontier)

      debug = self.DEBUG
      self.DEBUG = False
      self.add_clusters(new_clusters, idx+1)
      self.DEBUG = debug
      improvements.update(new_clusters)

    _logger.debug("merger\ttook %.1f sec\t%d improved\t%d tried\t%d tasks left", 
        time.time()-start, len(improvements), (self.nmerged-nmerged), self.ntasks)
    return improvements


  def run_task(self, idx, cluster, cur_frontier, next_frontier):
    if not (idx == 0 or self.valid_cluster_f(cluster)):
      _logger.debug("merger\tbelow thresh skipping\t %s" % cluster)
      return []

    if self.DEBUG:
      self.renderer.new_page()
      self.renderer.set_title("expand %s" % str(cluster.rule))
      self.renderer.plot_inf_curves([cluster], color='grey') 

    self.rejected_disc_vals = defaultdict(list)
    self.rejected_cont_vals = defaultdict(set)
    expanded = self.greedy_expansion(cluster, self.seen, idx, cur_frontier)
    expanded = [c for c in expanded if c.bound_hash != cluster.bound_hash]

    if self.DEBUG:
      self.renderer.plot_inf_curves(expanded, color='green')

    cur_expanded, rms = cur_frontier.update(expanded)
    next_expanded, rms2 = next_frontier.update(cur_expanded)
    f = lambda c: c.bound_hash != cluster.bound_hash
    improved_clusters = set(filter(f, next_expanded))

    to_hash = lambda cs: set([c.bound_hash for c in cs])
    exp_bounds = to_hash(expanded)
    cur_bounds = to_hash(cur_expanded)
    next_bounds = to_hash(next_expanded)

    for c in chain(cur_expanded, rms):
      _logger.debug("merger\texpanded\tcur_idx(%s)\tnext_idx(%s)\t%.3f-%.3f\t%s", 
          (c.bound_hash in exp_bounds), 
          (c.bound_hash in next_bounds), 
          c.c_range[0], c.c_range[1],
          c.rule.simplify())
    if self.DEBUG:
      self.renderer.plot_active_inf_curves(cur_frontier.frontier, color='blue')
      self.renderer.plot_active_inf_curves(next_frontier.frontier, color='red')

    return improved_clusters


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
      maxval = (len(vals) > 1) and max(vals) or None
      vals2infs = self.all_disc_vals[name].items()
      vals2infs.sort(key=lambda p: p[1][0] / float(p[1][1]+1.), reverse=True)

      for disc_vals, score in vals2infs:
        subset = set(disc_vals).difference(vals)
        subset.difference_update([v for v in subset if self.failed_disc_vals[name][str(v)] > 1])
        if maxval:
          subset = set(filter(lambda v: v >= maxval, subset))
        ret.append(subset)
      ret = filter(bool, ret)

      if ret:
        yield name, 'disc', ret
        return
        p = np.arange(len(ret), 0, -1).astype(float)
        p /= p.sum()
        ret = np.random.choice(ret, min(len(ret), 10), p=p, replace=False)
        yield name, 'disc', ret

  @instrument
  def check_direction(self, cluster, dim, direction, vals):
    key = cluster.bound_hash
    if direction == 'disc':
      for subset in self.rejected_disc_vals[dim]:
        if subset.issubset(vals): 
          return []
    if direction == 'inc':
      cont_vals = self.rejected_cont_vals[(dim, direction)]
      if cont_vals:
        vals = filter(lambda v: v > max(cont_vals), vals)
    if direction == 'dec':
      cont_vals = self.rejected_cont_vals[(dim, direction)]
      if cont_vals:
        vals = filter(lambda v: v < min(cont_vals), vals)
    return vals

  @instrument
  def update_rejected_directions(self, cluster, dim, direction, val):
    if direction == 'disc':
      for v in list(val):
        self.rejected_disc_vals[dim].append(set([v]))
        self.failed_disc_vals[dim][str(v)] += 1
    if direction == 'inc':
      self.rejected_cont_vals[(dim, direction)].add(round(val, 1))
    if direction == 'dec':
      self.rejected_cont_vals[(dim, direction)].add(round(val, 1))

  @instrument
  def greedy_expansion(self, cluster, seen, version=None, frontier=None):
    _logger.debug("merger\tgreedy_expand\t%s", cluster.rule.simplify())
    if frontier is None:
      frontier = CheapFrontier(self.c_range, K=1, nblocks=15)
      frontier.stats = self.stats
      frontier.update([cluster])

    cols = cluster.cols
    for dim, direction, vals in self.dims_to_expand(cluster, seen, version=version):
      if len(vals) == 0: continue
      attrname = isinstance(dim, basestring) and dim or cols[dim]
      vals = self.check_direction(cluster, dim, direction, vals)
      realvals = self.pick_expansion_vals(cluster, dim, direction, vals)

      nfails = 0
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

        improvements = frontier.improvement(tmp)
        if improvements.max() > 0:
          print str(tmp)
          print "\t", [round(v,2) for v in improvements]
        frontier.update([tmp])
        isbetter = tmp in frontier
        _logger.debug("merger\tcand\t%s\t%s\t%s\t%s", attrname[:15], direction, isbetter, v)
 
        seen.add(tmp.bound_hash)
        if not isbetter:
          self.update_rejected_directions(cluster, dim, direction, v)
          if direction != 'disc':
            nfails += 1
            if nfails > 1:
              break
        if direction != 'disc':
          cluster = tmp

    return frontier.frontier


class PartitionedStreamRangeMerger(StreamRangeMerger):

  def __init__(self, *args, **kwargs):
    super(PartitionedStreamRangeMerger, self).__init__(*args, **kwargs)
    self.frontiers = defaultdict(list)
    self.tasks = defaultdict(list)

  def get_frontier_obj(self, version, partitionkey):
    frontiers = self.frontiers[partitionkey]
    while version >= len(frontiers):
      frontier = CheapFrontier(self.c_range, K=self.K, nblocks=self.nblocks)
      frontier.stats = self.stats
      frontiers.append(frontier)
    return frontiers[version]
  
  @property
  def frontier_iter(self):
    return chain(*self.frontiers.values())

  @instrument
  def add_clusters(self, 
      clusters, idx=0, partitionkey=None, skip_frontier=False):
    """
    Return list of new clusters that are on the frontier
    """
    if partitionkey is None:
      raise RuntimeError('addclusters partitionkey cannot be none')

    if not clusters: return []

    print "add %d clusters" % len(clusters)
    if self.DEBUG:
      self.renderer.new_page()
      self.renderer.set_title("add_clusters %d clusters" % len(clusters))
      for f in self.frontier_iter:
        self.renderer.plot_inf_curves(f.frontier, color='grey') 
      self.renderer.plot_inf_curves(clusters, color='green')

    nclusters = len(clusters)
    clusters = self.setup_stats(clusters)
    frontier = self.get_frontier_obj(idx, partitionkey)
    if not skip_frontier:
      clusters, _ = frontier.update(clusters)

    if self.DEBUG and not skip_frontier:
      print "base_frontier"
      self.print_clusters(clusters)

    if self.DEBUG:
      self.renderer.plot_active_inf_curves(clusters, color='red')

    # clear out current tasks
    tkey = (partitionkey, idx)
    self.tasks[tkey] = filter(frontier.__contains__, self.tasks[tkey])
    self.tasks[tkey].extend(clusters)

    if idx == 0:
      # remove non-frontier-based expansions from future expansion
      for (pkey, tidx) in self.tasks.keys():
        if pkey != partitionkey: continue
        if tidx == 0: continue
        checker = lambda c: not any(map(frontier.__contains__, c.ancestors))
        self.tasks[tkey] = filter(checker, self.tasks[tkey])

    _logger.debug("merger\t%s\tadding %d of %d clusters\t%d tasks left", partitionkey, len(clusters), nclusters, self.ntasks)
    return clusters


  def next_tasks(self, n=1):
    ret = []
    for tkey in reversed(self.tasks.keys()):
      if len(ret) >= n: break
      tasks = self.tasks[tkey]
      ntasks = len(tasks)
      if not ntasks: continue
      idxs = np.random.choice(ntasks, min(ntasks, n-len(ret)), replace=False).tolist()
      for idx in sorted(idxs, reverse=True):
        ret.append((tkey[0], tkey[1], tasks.pop(idx)))
    return ret


  def __call__(self, n=2):
    nmerged = self.nmerged
    start = time.time()
    tasks = self.next_tasks(n)
    improvements = set()

    for pkey, idx, cluster in tasks:
      cur_frontier = self.get_frontier_obj(idx, pkey)
      next_frontier = self.get_frontier_obj(idx+1, pkey)
      new_clusters = self.run_task(idx, cluster, cur_frontier, next_frontier)

      self.add_clusters(new_clusters, idx=idx+1, partitionkey=pkey, skip_frontier=True)
      improvements.update(new_clusters)

      _logger.debug("merger\t%s\ttook %.1f sec\t%d improved\t%d tried\t%d tasks left", 
          pkey, time.time()-start, len(improvements), (self.nmerged-nmerged), self.ntasks)
    return improvements




