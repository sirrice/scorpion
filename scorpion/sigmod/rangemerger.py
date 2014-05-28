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

from crange import *
from merger import Merger

_logger = get_logger()


def frange(b):
  return (np.arange(100) / 100. * r_vol(b)) + b[0]

def reset_ax(ax, xbound, ybound):
  ax.cla()
  ax.set_xlim(*xbound)
  ax.set_ylim(*ybound)

def render(ax, c, xs, color='grey', alpha=0.3):
  ys = map(c.inf_func, xs)
  ax.plot(xs, ys, alpha=alpha, color=color)

def render_fulls(ax, cs, xs, color='grey', alpha=0.3):
  for c in cs:
    render(ax, c, xs, color, alpha)

def render_segs(ax, cs, color='grey', alpha=0.3):
  for c in cs:
    render(ax, c, frange(c.c_range), color, alpha)




class RangeMerger(Merger):
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
    self.yrange = None


    #
    # per execution state
    #

    # dim -> list of value subsets that were not on frontier
    # e.g., subregion -> [ (SR1, SR2), (SR3), ... ]
    self.rejected_disc_vals = defaultdict(list)

  def __hash__(self):
    return hash((self.learner_hash, tuple(self.c_range)))

  def setup_stats(self, clusters):
    """
    computes error bounds and the minimum volume of a 0-volume cluster

    adds data structures to cluster object
    """
    Merger.setup_stats(self, clusters)

    for c in clusters:
      c.inf_func = c.create_inf_func(self.learner.l)
      c.c_range = list(self.c_range)
      c.inf_range = [c.inf_func(c.c_range[0]), c.inf_func(c.c_range[1])]

    self.rejected_disc_vals = defaultdict(list)

  def print_clusters(self, clusters):
    rules = clusters_to_rules(clusters, self.learner.full_table)
    rules = ['%.4f-%.4f %s' % (r.c_range[0], r.c_range[1], r) for r in rules]
    print '\n'.join(rules)


  @instrument
  def __call__(self, clusters, **kwargs):
    if not clusters:
        return list(clusters)

    _logger.debug("merging %d clusters", len(clusters))
    _logger.debug("DEBUG = %s", self.DEBUG)

    self.set_params(**kwargs)
    self.setup_stats(clusters)

    self.adj_graph = self.make_adjacency(clusters, self.partitions_complete)
    #self.rtree = self.construct_rtree(clusters)

    if self.DEBUG:
      self.fig = fig = plt.figure(figsize=(4, 6))
      self.ax = ax = fig.add_subplot(111)
      self.xs = xs = frange(self.c_range)
      self.miny = miny = min([min(c.inf_func(xs[0]), c.inf_func(xs[1])) for c in clusters])
      self.maxy = maxy = max([max(c.inf_func(xs[0]), c.inf_func(xs[1])) for c in clusters])
      self.yrange = [miny, maxy]
      reset_ax(ax, self.c_range, (miny, maxy))
      render_fulls(ax, clusters, xs, 'grey', .2)
      fig.savefig('/tmp/infs-0.pdf')


    frontier, removed_clusters = self.get_frontier(clusters)
    _logger.debug("%d clusters in frontier", len(frontier))


    if self.DEBUG:
      reset_ax(ax, self.c_range, (miny, maxy))
      render_fulls(ax, removed_clusters, xs, 'grey', .2)
      render_segs(ax, frontier, 'red', .4)
      fig.savefig("/tmp/infs-01.pdf")
      print self.get_frontier.stats.items()
      print self.get_frontier.heap.stats.items()


    start = time.time()
    iteridx = 1
    seen = set()
    while len(frontier) > 0:#self.min_clusters:
      self.learner.update_status("expanding frontier, iter %d" % iteridx)
      iteridx += 1

      if self.DEBUG:
        reset_ax(ax, self.c_range, (miny, maxy))
        render_fulls(ax, frontier, xs, 'grey', .2)
        print "frontier"
        self.print_clusters(frontier)

      new_clusters, removed_clusters = self.expand_frontier(frontier, seen)

      if self.DEBUG:
        print "\nnew clusters"
        self.print_clusters(new_clusters)
        render_fulls(ax, removed_clusters, xs, 'grey', .2)
        render_segs(ax, new_clusters, 'red', .4)
        fig.savefig("/tmp/infs-%02d.pdf" % iteridx)

      if (not new_clusters.difference(frontier)) or (time.time() - start) > 60:
        clusters_set = set(removed_clusters)
        clusters_set.update(new_clusters)
        break

      map(self.adj_graph.remove, removed_clusters)
      map(self.adj_graph.insert, new_clusters)
      self.adj_graph.new_version()
      frontier = new_clusters

    print "returning %d clusters total!" % len(clusters_set)
    if self.DEBUG:
      self.print_clusters(clusters_set)

    self.learner.merge_stats(self.get_frontier.stats, 'frontier_')
    self.learner.merge_stats(self.get_frontier.heap.stats, 'inter_heap_')
    return list(clusters_set)
  


  @instrument
  def expand_frontier(self, frontier, seen):
    """
    Return (newclusters, rmclusters)
    """

    newclusters = set(frontier)
    for cluster in frontier:
      merges = self.expand(cluster, seen)
      #for c in rms: c.c_range = list(self.c_range)
      #newclusters.update(rms)
      newclusters.update(merges)

    for cluster in newclusters:
      cluster.c_range = list(self.c_range)

    return self.get_frontier(newclusters)

 

  @instrument
  def dim_merge(self, cluster, dim, dec=None, inc=None, skip=None):
    merged = Merger.dim_merge(self, cluster, dim, dec, inc, skip)
    if merged:
      merged.c_range = list(self.c_range)
      merged.inf_func = merged.create_inf_func(self.learner.l)
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
      merged.inf_func = merged.create_inf_func(self.learner.l)
    return merged

  @instrument
  def greedy_expansion(self, cur, seen):
    expansions = self.expand_candidates(cur, seen)

    dim_to_bests = defaultdict(set)
    for dim, direction, g in expansions:
      dim_bests = self.expand_dim(cur, dim, direction, g, seen)
      #if cur in dim_bests: dim_bests.remove(cur)
      dim_to_bests[(dim, direction)] = dim_bests

    return dim_to_bests
  
  @instrument
  def expand_dim(self, cur, dim, direction, g, seen):
    if direction == 'disc':
      cands = []
      bests = set([cur])
      prev_card = None
      for cand in g:
        cands.append(cand)
        bests.add(cand)
        bests, _ = self.get_frontier(bests)
        if cand not in bests:
          seen.add(hash(cand))
          #if prev_card is not None and len(cand.discretes[dim]) != prev_card:
          #  break
        prev_card = len(cand.discretes[dim])
      
    else:
      cands = []
      bests = set([cur])
      for cand in g:
        cands.append(cand)
        bests, _ = self.get_frontier(bests.union(set([cand])))
        if cand not in bests:
          seen.add(hash(cand))
          break
      bests = [cur] + cands

    # update rejection state
    for c in cands:
      if c in bests: continue
      self.rejected_disc_vals[dim].append(set(c.discretes[dim]))

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



  @instrument
  def expand(self, c, seen):
    """
    Returns a frontier of clusters expanded from c that
    are possible optimals in _some_ c range

    XXX: optimization could be a minimum c range a cluster must be
          a candidate over
    """
    _logger.debug("expand\t%s", str(c.rule)[:100])
    start = time.time()
    self.rejected_disc_vals = defaultdict(list)
    cur_bests = set([c])
    ret = set()
    rms = set()
    all_merges = set()
    while cur_bests:
      if not self.DEBUG and (time.time() - start) > 30:
        ret.update(cur_bests)
        break

      cur = cur_bests.pop()
      if hash(cur) in seen: 
        _logger.debug("seen \t%s", str(cur.rule)[:100])
        continue
      seen.add(hash(cur))
      _logger.debug("expand\tsub\t%.3f-%.3f\t%s", cur.c_range[0], cur.c_range[1],  str(cur.rule)[:100])


      dim_to_bests = self.greedy_expansion(cur, seen)#.union(all_merges))

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

      if cur_bests.issuperset(frontier):
        _logger.debug("frontier subset of curbests")
        ret.add(cur)
        continue
      
      for merged in frontier.difference(cur_bests):
        rms.update(merged.parents)
        self.adj_graph.insert(merged)
      seen.update(rms)

      cur_bests = frontier
      cur_bests.difference_update([cur])
      _logger.debug("\t%d in frontier", len(cur_bests))


    if self.DEBUG:
      reset_ax(self.ax, self.c_range, self.yrange)
      render_fulls(self.ax, ret, self.xs, alpha=.2)
      render_segs(self.ax, [c], 'red', .4)
      self.ax.set_title(str(c.rule)[:50])
      self.ax.title.set_fontsize(6)
      self.fig.savefig('/tmp/expand-%02d.pdf' % self.i)
      self.i += 1



    return ret
    ret, more_rms = self.get_frontier(ret)
    rms.update(more_rms)
    return ret, rms



