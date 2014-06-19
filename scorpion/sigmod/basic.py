import time
import pdb
import sys
import Orange
import orange
import heapq
sys.path.extend(['.', '..'])

import numpy as np
from itertools import chain
from collections import defaultdict
from sklearn.cluster import AffinityPropagation, KMeans
from scorpionsql.errfunc import ErrTypes, compute_bad_inf, compute_bad_score, compute_influence


from ..learners.cn2sd.rule import fill_in_rules
from ..learners.cn2sd.refiner import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *
from ..util import *


INF = inf = float('inf') # slowly replace inf with INF...



class Basic(object):


  def __init__(self, **kwargs):
    self.obj = kwargs.get('obj', None)
    self.aggerr = kwargs.get('aggerr', None)
    self.cols = list(self.aggerr.agg.cols)
    self.err_func = kwargs.get('err_func', self.aggerr.error_func.clone())
    self.merger = None
    self.params = {}
    self.costs = {}
    self.stats = defaultdict(lambda: [0, 0])  # used by @instrument

    self.bad_thresh = 0
    self.good_thresh = 0
    self.min_pts = kwargs.get('min_pts', 5)
    self.max_bests = 20
    self.max_complexity = kwargs.get('max_complexity', 4)
    self.granularity = kwargs.get('granularity', 100)

    self.l = kwargs.get('l', 0.5)
    self.c = kwargs.get('c', 0.3)
    self.c_range = kwargs.get('c_range', [0.05, 0.5])
    self.epsilon = kwargs.get('epsilon', 0.0001)
    self.tau = kwargs.get('tau', [0.1, 0.5])
    self.p = kwargs.get('p', 0.5)
    self.bincremental = kwargs.get('bincremental', True)
    self.use_cache = kwargs.get('use_cache', False)
    self.parallel = kwargs.get('parallel', False)

    self.DEBUG = kwargs.get('DEBUG', False)

    self.tablename = kwargs.get('tablename', None)


    self.scorer_cost = 0.
    self.merge_cost = 0.

    self.update_status = self.obj.update_status
    self.update_rules = self.obj.update_rules

    self.set_params(**kwargs)

  def __hash__(self):
    components = [
            self.__class__.__name__,
            str(self.aggerr.__class__.__name__),
            str(set(self.cols)),
            self.err_func.__class__.__name__,
            self.tablename,
            self.l,
            self.c
            ]
    components = map(str, components)
    return hash('\n'.join(components))

  def merge_stats(self, stats, prefix=''):
    for key, stat in stats.iteritems():
      mykey = '%s%s' % (prefix, key)
      if mykey in self.stats:
        self.stats[mykey][0] += stat[0]
        self.stats[mykey][1] += stat[1]
      else:
        self.stats[mykey] = list(stat)
              

  def set_params(self, **kwargs):
    self.cols = kwargs.get('cols', self.cols)
    self.use_cache = kwargs.get('use_cache', self.use_cache)
    self.params.update(kwargs)

  def setup_tables(self, full_table, bad_tables, good_tables, **kwargs):
    """
    create bad_err_funcs
    """
    self.full_table = full_table
    self.dummy_table = Orange.data.Table(full_table.domain)
    self.bad_tables = bad_tables
    self.good_tables = good_tables
    
    self.bad_err_funcs = self.aggerr.bad_error_funcs()
    self.good_err_funcs = self.aggerr.good_error_funcs(good_tables)

    for ef, t in zip(self.bad_err_funcs, bad_tables):
      ef.setup(t)

    for ef, t in zip(self.good_err_funcs, good_tables):
      ef.setup(t)

    domain = self.full_table.domain
    attrnames = [attr.name for attr in domain]
    self.cont_dists = dict(zip(attrnames, Orange.statistics.basic.Domain(self.full_table)))
    self.disc_dists = dict(zip(attrnames, Orange.statistics.distribution.Domain(self.full_table)))

  def __call__(self, full_table, bad_tables, good_tables, **kwargs):
    """
    table has been trimmed of extraneous columns.
    @return final_clusters
    """

    self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

    pass

  def rule_complexity(self, rule):
    ret = 0
    for cond in rule.filter.conditions:
      pos = cond.position
      attr = self.full_table.domain[pos]
      if attr.var_type == Orange.feature.Type.Discrete:
        fd = self.disc_dists[attr.name]
        if len(cond.values) != len(fd.values()):
          ret += 1
          ret += max(0, 0.01 * len(cond.values) - 2)
      else:
        fb = self.cont_dists[attr.name]
        if not r_contains([cond.min, cond.max], [fb.min, fb.max]):
          ret += 1
    return ret


  def create_inf_func(self, cluster):
    """
    Args:
      cluster-like object with attributes:
        .inf_state
        .rule

    """
    inf_state = cluster.inf_state
    if inf_state is None:
      raise Exception("inf_state is None, cant' create inf_func")

    l = self.l
    vs = [abs(gv) for gv, gc in zip(inf_state[2], inf_state[3]) if gc]
    #maxg = max(vs) if vs else 0
    maxg = np.percentile(vs, 85) if vs else 0

    bds, bcs = [], []
    for idx in xrange(len(inf_state[0])):
      bd, bc = inf_state[0][idx], inf_state[1][idx]
      if valid_number(bd) and valid_number(bc):
        bds.append(bd)
        bcs.append(bc)
      else:
        bds.append(0)
        bcs.append(0)

    rule = cluster.rule
    complexity = self.rule_complexity(rule)
    f = lambda c: compute_influence(l, compute_bad_score(bds, bcs, c), maxg, nclauses=complexity)
    #f = np.frompyfunc(f, 1, 1)
    return f



  @instrument
  def cluster_to_rule(self, cluster, table=None):
    if not table:
      table = self.dummy_table
    rule = cluster.to_rule(
        table, cont_dists=self.cont_dists, disc_dists=self.disc_dists
    )
    return rule


  @instrument
  def influence_state(self, rule):
    bdeltas, bcounts = self.compute_stat(rule, self.bad_err_funcs, self.bad_tables)
    gdeltas, gcounts = self.compute_stat(rule, self.good_err_funcs, self.good_tables)
    gdeltas = map(abs, gdeltas)
    return bdeltas, bcounts, gdeltas, gcounts

  @instrument
  def influence_from_state(self, bdeltas, bcounts, gdeltas, gcounts, c=None, nclauses=0):
    if c is None:
        c = self.c
    binf = compute_bad_score(bdeltas, bcounts, c)
    ginfs = [gdelta for gdelta,gcount in zip(gdeltas, gcounts) if gcount]
    ginf = ginfs and max(ginfs) or 0
    return compute_influence(self.l, binf, ginf, nclauses=nclauses)


  @instrument
  def influence_cluster(self, cluster, table=None):
    rule = self.cluster_to_rule(cluster, table)
    cluster.error = self.influence(rule)
    cluster.inf_state = rule.inf_state
    return cluster.error


  @instrument
  def influence(self, rule, c=None):
    inf_state = self.influence_state(rule)
    quality = self.influence_from_state(*inf_state, nclauses=len(rule.filter.conditions))
    rule.quality = quality
    rule.inf_state = inf_state
    return quality


  def compute_stat(self, rule, err_funcs, tables):
    datas = rule and map(rule.filter_table, tables) or tables
    infs = []
    lens = []
    for idx, (ef, data) in enumerate(zip(err_funcs, datas)):
      if len(data) == len(tables[idx]):
        influence = -INF
      else:
        arr = data.to_numpyMA('ac')[0]
        influence = ef(arr.data)
      infs.append(influence)
      lens.append(len(data))
    return infs, lens





  def all_unit_clauses(self, attr):
      # continuous: split 1000 ways, uniformly
      # discrete: every unique value
      attrobj = self.full_table.domain[attr]
      idx = self.full_table.domain.index(attrobj)
      if attrobj.var_type == Orange.feature.Type.Discrete:
          ddist = Orange.statistics.distribution.Domain(self.full_table)[idx]
          return ddist.keys()
      
      bdist = Orange.statistics.basic.Domain(self.full_table)[idx]
      minv, maxv = bdist.min, bdist.max
      if minv == maxv:
          return [[-INF, INF]]

      block = (maxv - minv) / self.granularity
      ranges = [[minv + i*block, minv + (i+1)*block] for i in xrange(self.granularity)]
      ranges[0][0] = -INF
      ranges[-1][1] = INF
      return ranges


  def get_all_clauses(self, attr, max_card):
      class Ret(object):
          def __init__(self, attr, max_card, par):
              self.attr = attr
              self.max_card = max_card
              self.par = par

          def __iter__(self):
              attrobj = self.par.full_table.domain[self.attr]
              if attrobj.var_type == Orange.feature.Type.Discrete:
                  return self.par.all_discrete_clauses(self.attr, self.max_card)
              else:
                  return self.par.all_cont_clauses(self.attr)
      return Ret(attr, max_card, self)

          
  def all_discrete_clauses(self, attr, max_card=None):
      all_vals = self.col_to_clauses[attr]
      attrobj = self.full_table.domain[attr]
      idx = self.full_table.domain.index(attrobj)
      
      if max_card:
          for card in xrange(1, max_card+1):
              for vals in combinations(all_vals, card):
                  vals = [orange.Value(attrobj, value) for value in vals]
                  yield orange.ValueFilter_discrete(
                          position = idx,
                          values = vals)
      else:
          for vals in powerset(all_vals):
              vals = [orange.Value(attrobj, value) for value in vals]
              yield orange.ValueFilter_discrete(
                      position = idx,
                      values = vals)


  def all_cont_clauses(self, attr):
      units = self.col_to_clauses[attr]
      idx = self.full_table.domain.index(self.full_table.domain[attr])
      for sidx in xrange(0, len(units)):
          for eidx in xrange(sidx, len(units)):
              minv = units[sidx][0]
              maxv = units[eidx][1]
              yield Orange.data.filter.ValueFilterContinuous(
                      position=idx,
                      oper=orange.ValueFilter.Between,
                      min=minv,
                      max=maxv)


