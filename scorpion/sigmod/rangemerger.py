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
from merger import Merger

_logger = get_logger()



class BaseRangeMerger(Merger):
  """
  Merges clusters

  transparently scales
  - cluster bounding boxes by table bounds
  - errors by min/max error
  """


  def __init__(self, *args, **kwargs):
    Merger.__init__(self, *args, **kwargs)

    self.learner_hash = kwargs.get('learner_hash', '')
    self.c_range = kwargs.get('c_range', [0.01, 0.7])
    self.get_frontier = Frontier(self.c_range, 0.)
    self.CACHENAME = './dbwipes.rangemerger.cache'
    self.i = 0


    #
    # per execution state
    #

    # dim -> list of value subsets that were not on frontier
    # e.g., subregion -> [ (SR1, SR2), (SR3), ... ]
    self.rejected_disc_vals = defaultdict(list)

    # (dim, direction) -> range it has expanded along
    self.rejected_cont_vals = defaultdict(set)

  def __hash__(self):
    return hash((self.learner_hash, tuple(self.c_range)))

  def setup_stats(self, clusters):
    """
    computes error bounds and the minimum volume of a 0-volume cluster

    adds data structures to cluster object
    """
    Merger.setup_stats(self, clusters)

    for c in clusters:
      c.inf_func = self.learner.create_inf_func(c)
      c.c_range = list(self.c_range)
      c.inf_range = [c.inf_func(c.c_range[0]), c.inf_func(c.c_range[1])]

  def print_clusters(self, clusters):
    print "\n".join(map(str, clusters))

  def get_variety_frontier(self, clusters):
    seen = set()
    ret = []
    cluster_bcs = [list(c.inf_state[1]) for c in clusters]

    while True:
      for c in clusters:
        c.c_range = list(self.c_range)

      frontier,_ = self.get_frontier(clusters)
      frontier.difference_update(seen)
      print "variety"
      self.print_clusters(frontier)
      if not frontier: break
      ret.append(frontier)
      seen.update(frontier)

      for c in clusters:
        for f in frontier:
          for idx in xrange(len(c.inf_state[1])):
            c.inf_state[1][idx] = max(0, c.inf_state[1][idx] - 3*f.inf_state[1][idx])

    for idx, c in enumerate(clusters):
      c.inf_state = list(c.inf_state)
      c.inf_state[1] = cluster_bcs[idx]
    return ret, set(clusters).difference(seen)


  @instrument
  def __call__(self, clusters, **kwargs):
    if not clusters:
        return list(clusters)

    _logger.debug("merging %d clusters", len(clusters))
    _logger.debug("DEBUG = %s", self.DEBUG)

    self.set_params(**kwargs)
    self.setup_stats(clusters)
    self.rejected_disc_vals = defaultdict(list)
    self.rejected_cont_vals = defaultdict(set)


    self.learner.update_status("expanding frontier: indexing partitions")
    self.adj_graph = self.make_adjacency(clusters, self.partitions_complete)

    if self.DEBUG:
      self.renderer = InfRenderer('/tmp/infs.pdf', c_range=self.c_range)
      self.renderer.plot_inf_curves(clusters)
      self.renderer.set_title("inf 0")


    self.learner.update_status("expanding frontier, iter 0")
    frontiers, removed_clusters = self.get_variety_frontier(clusters)
    _logger.debug("%d clusters in frontier", sum(map(len, frontiers)))

    clusters.sort(key=lambda c: c.inf_func(0.1), reverse=True)
    self.print_clusters(chain(*frontiers))
    print
    self.print_clusters(clusters[:5])

    if self.DEBUG:
      self.boxrenderer = JitteredClusterRenderer('/tmp/boxes.pdf')
      self.boxrenderer.plot_clusters(removed_clusters, color="grey")
      for frontier in frontiers:
        self.boxrenderer.plot_clusters(frontier, color="red")
      self.renderer.new_page()
      self.renderer.plot_inf_curves(clusters)
      print self.get_frontier.stats.items()
      print self.get_frontier.heap.stats.items()


    start = time.time()
    iteridx = 1
    seen = set()
    clusters_set = set()
    for frontier in frontiers:
      for c in frontier:
        c.c_range = list(self.c_range)

      versionid = 0
      while len(frontier) > 0:
        self.learner.update_status("expanding frontier, iter %d" % iteridx)
        iteridx += 1

        if self.DEBUG:
          self.renderer.new_page()
          self.renderer.plot_inf_curves(frontier, color='grey', alpha=.2)
          self.renderer.set_title("frontier expansion %d" % iteridx)
          print "frontier"
          self.print_clusters(frontier)

        new_clusters, removed_clusters = self.expand_frontier(frontier, seen, None)

        if self.DEBUG:
          self.renderer.plot_inf_curves(removed_clusters, color="grey")
          self.renderer.plot_inf_curves(new_clusters)
          print "\nnew clusters"
          self.print_clusters(new_clusters)

        if (not new_clusters.difference(frontier)) or (time.time() - start) > 60:
          clusters_set.update(removed_clusters)
          clusters_set.update(new_clusters)
          break

        self.adj_graph.remove(removed_clusters, versionid)
        self.adj_graph.insert(new_clusters, versionid)
        self.adj_graph.sync()
        versionid += 1
        frontier = new_clusters

    print "returning %d clusters total!" % len(clusters_set)
    if self.DEBUG:
      self.renderer.new_page()
      self.renderer.plot_inf_curves(clusters_set)
      self.renderer.set_title("final frontier")
      self.renderer.close()
      self.boxrenderer.close()
      self.print_clusters(clusters_set)

    self.learner.merge_stats(self.get_frontier.stats, 'frontier_')
    self.learner.merge_stats(self.get_frontier.heap.stats, 'inter_heap_')
    return list(clusters_set)
  


  @instrument
  def expand_frontier(self, frontier, seen, version=None):
    """
    Return (newclusters, rmclusters)
    """
    newclusters = set(frontier)
    for cluster in frontier:
      merges = self.expand(cluster, seen, version=None)
      #for c in rms: c.c_range = list(self.c_range)
      #newclusters.update(rms)
      newclusters.update(merges)

    for cluster in newclusters:
      cluster.c_range = list(self.c_range)

    return self.get_frontier(newclusters)

 

  @instrument
  def dim_merge(self, cluster, dim, dec=None, inc=None, skip=None):
    if dec is not None:
      if round(dec, 1) in self.rejected_cont_vals[(dim, 'dec')]:
        return None
    if inc is not None:
      if round(inc, 1) in self.rejected_cont_vals[(dim, 'inc')]:
        return None
    merged = Merger.dim_merge(self, cluster, dim, dec, inc, skip)
    if merged:
      merged.c_range = list(self.c_range)
      merged.inf_func = self.learner.create_inf_func(merged)
    return merged


  @instrument
  def disc_merge(self, cluster, dim, vals, skip=None):
    # reject if union is a superset of anything in 
    # rejected_disc_vals
    vals = set(vals)
    vals.update(cluster.discretes.get(dim, ()))
    for subset in self.rejected_disc_vals[dim]:
      if vals.issuperset(subset):
        return None

    merged = Merger.disc_merge(self, cluster, dim, vals, skip)
    if merged:
      merged.c_range = list(self.c_range)
      merged.inf_func = self.learner.create_inf_func(merged)
    return merged


  @instrument
  def expand(self, c, seen, version=None):
    """
    Returns a frontier of clusters expanded from c that
    are possible optimals in _some_ c range

    XXX: optimization could be a minimum c range a cluster must be
          a candidate over
    """
    _logger.debug("expand\t%s", str(c.rule)[:100])
    start = time.time()
    self.rejected_disc_vals = defaultdict(list)
    self.rejected_cont_vals = defaultdict(set)
    cur_bests = set([c])
    ret = set()
    rms = set()
    all_merges = set()
    self.i = 0
    while cur_bests:
      if not self.DEBUG and (time.time() - start) > 30:
        ret.update(cur_bests)
        break

      cur = cur_bests.pop()
      if cur.bound_hash in seen: 
        _logger.debug("seen \t%s", str(cur.rule)[:100])
        continue
      seen.add(cur.bound_hash)
      _logger.debug("expand\tsub\t%.3f-%.3f\t%s", cur.c_range[0], cur.c_range[1],  str(cur.rule)[:100])


      cur_seen = set()
      dim_to_bests = self.greedy_expansion(cur, cur_seen, version=version)
      seen.update(cur_seen)

      merges = list()
      map(merges.extend, dim_to_bests.values())
      all_merges.update(map(hash, merges))
      merges = set(filter(lambda c: r_vol(c.c_range), merges))
      if cur in merges:
        _logger.debug("cur added to bests")
        ret.add(cur)
        merges.remove(cur)

      if len(merges) is 0:
        _logger.debug("# better expanded = 0")
        ret.add(cur)
        continue

      combined = set(cur_bests)
      combined.update(merges)
      frontier, losers = self.get_frontier(combined)


      if self.DEBUG:
        self.boxrenderer.new_page()
        self.boxrenderer.plot_clusters([l for l in losers if l not in merges], color="grey", alpha=0.2)
        self.boxrenderer.plot_clusters([m for m in merges if m in losers], color='blue', alpha=0.2)
        self.boxrenderer.plot_clusters([m for m in merges if m in frontier], color='red', alpha=0.3)
        self.boxrenderer.plot_clusters([cur], color="black", alpha=0.7)
        self.boxrenderer.set_title("expand iter %d\n%s" % (self.i, str(c.rule)))
        self.i += 1



      if cur_bests.issuperset(frontier):
        _logger.debug("frontier subset of curbests")
        ret.add(cur)
        continue
      
      for merged in frontier.difference(cur_bests):
        rms.update(merged.parents)
      self.adj_graph.insert(cur_bests)
      seen.update([c.bound_hash for c in rms])

      cur_bests = frontier
      cur_bests.difference_update([cur])
      _logger.debug("\t%d in frontier", len(cur_bests))




    return ret
    ret, more_rms = self.get_frontier(ret)
    rms.update(more_rms)
    return ret, rms

class RangeMerger(BaseRangeMerger):

  @instrument
  def greedy_expansion(self, cur, seen, version=None):
    expansions = self.expand_candidates(cur, seen, version=version)

    dim_to_bests = defaultdict(set)
    for dim, direction, g in expansions:
      dim_bests = self.expand_dim(cur, dim, direction, g, seen)
      #if cur in dim_bests: dim_bests.remove(cur)
      dim_to_bests[(dim, direction)] = dim_bests

    # cross product between inc and dec of each dimension

    return dim_to_bests
  
  @instrument
  def expand_dim(self, cur, dim, direction, g, seen):
    """
    Args
      dim: if direction == 'disc', dim is the attr name
           else dim is the index into cluster.cols
    """
    if True or direction == 'disc':
      cands = []
      bests = set([cur])
      prev_card = None
      cands = list(g)
      bests.update(cands)
      bests, _ = self.get_frontier(bests)
      for cand in cands:
        if cand not in bests:
          seen.add(cand.bound_hash)
    else:
      cands = []
      bests = set([cur])
      for cand in g:
        cands.append(cand)
        bests, _ = self.get_frontier(bests.union(set([cand])))
        if cand not in bests:
          seen.add(cand.bound_hash)
          break
      bests = [cur] + cands
    bests = list(bests)

    # update rejection state
    for c in cands:
      if c in bests: continue
      if direction == 'disc':
        self.rejected_disc_vals[dim].append(set(c.discretes[dim]))
      if direction == 'inc':
        v = c.bbox[1][dim]
        self.rejected_cont_vals[(dim, direction)].add(v)
      if direction == 'dec':
        v = c.bbox[0][dim]
        self.rejected_cont_vals[(dim, direction)].add(v)

    if self.DEBUG:
      if direction == 'disc':
        name = dim[:10]
        s = ','.join([str(len(c.discretes[dim])) for c in bests])
        ss = ','.join([str(c.discretes[dim]) for c in cands])
        if 'subregion' in str(cur):
          for c in cands:
            isbest = False
            if c in bests:
              isbest = True
            _logger.debug('\tbest? %s\tcand:\t%s', isbest, c)
      else:
        name = cur.cols[dim][:10]
        if bests:
          if direction == 'inc':
            s = ','.join(["%.4f" % c.bbox[1][dim] for c in bests])
            s = '%.4f - %s' % (bests[0].bbox[0][dim], s)
          else:
            s = ','.join(["%.4f" % c.bbox[0][dim] for c in bests])
            s = '%s - %.4f' % (s, bests[0].bbox[1][dim])
        else:
          s = '---'
        ss = ''
      _logger.debug("\t%s\t%s\t%d bests\t%d candidates", name, direction, len(bests), len(cands))
      _logger.debug("\tbests\t\t\t\t%s", s)
      _logger.debug("\tcands\t\t\t\t%s", ss)


    return bests


class RangeMerger2(BaseRangeMerger):

  def pick_expansion_vals(self, cluster, dim, direction, vals):
    if len(vals) == 0: return vals
    if direction == 'disc': 
      return vals
      return np.random.choice(vals, min(len(vals), 6), replace=False)

    vals = random.sample(vals, min(4, len(vals)))
    vals.sort(reverse=(direction == 'dec'))

    if direction == 'inc':
      baseval = cluster.bbox[1][dim]
    else:
      baseval = cluster.bbox[0][dim]

    rng = self.learner.cont_dists[cluster.cols[dim]]
    min_step = r_vol([rng.min, rng.max]) * 0.005
    vals = filter(lambda v: abs(v-baseval) > min_step, vals)

    if len(vals) <= 4:
      return vals
    vals = random.sample(vals, 4)
    return vals


  @instrument
  def greedy_expansion(self, cluster, seen, version=None):
    curset = set([cluster])
    for dim, direction, vals in self.dims_to_expand(cluster, seen, version=version):
      if self.DEBUG:
        self.boxrenderer.new_page()
        self.boxrenderer.plot_clusters(curset, color='blue', alpha=0.3)
        _logger.debug("%s\t%s\t%s", dim, direction, str(vals))


      tmp = set()
      realvals = self.pick_expansion_vals(cluster, dim, direction, vals)
      if direction != 'disc':
        _logger.debug("# to expand\t%s\t%s\t%d -> %d\t%s -> %s", 
            cluster.cols[dim][:10], direction, len(vals), 
            len(realvals), str(vals), str(realvals))
      if direction == 'inc':
        for inc in realvals:
          tmp.update([self.dim_merge(c, dim, None, inc, seen) for c in curset])
      elif direction == 'dec':
        for dec in realvals:
          tmp.update([self.dim_merge(c, dim, dec, None, seen) for c in curset])
      else:
        for disc in realvals:
          tmp.update([self.disc_merge(c, dim, disc) for c in curset])

      tmp = filter(bool, tmp)
      _logger.debug("# actual expansions\t%d using %d of %d vals\t%d in curset", len(tmp), len(realvals), len(vals), len(curset))
      if not tmp:
        continue
      for c in curset:
        c.c_range = list(self.c_range)
      tmp.extend(curset)
      curset,_ = self.get_frontier(tmp)
      seen.update([c.bound_hash for c in _])


      # update rejection state
      for v in vals:
        if direction == 'disc':
          self.rejected_disc_vals[dim].append(set(v))#c.discretes[dim]))
        if direction == 'inc':
          #v = c.bbox[1][dim]
          self.rejected_cont_vals[(dim, direction)].add(round(v, 1))
        if direction == 'dec':
          #v = c.bbox[0][dim]
          self.rejected_cont_vals[(dim, direction)].add(round(v, 1))



      if self.DEBUG:
        self.boxrenderer.plot_clusters(curset, color='red', alpha=0.3)
        self.boxrenderer.plot_clusters(_, color='grey', alpha=0.2)
        self.boxrenderer.set_title("greedy expansion\n%s" % cluster)


    ret = { 'dim': curset }
    return ret



