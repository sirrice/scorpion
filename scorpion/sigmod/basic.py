import time
import pdb
import sys
import Orange
import orange
import heapq
sys.path.extend(['.', '..'])

from itertools import chain
from collections import defaultdict
from sklearn.cluster import AffinityPropagation, KMeans


from ..learners.cn2sd.rule import fill_in_rules
from ..learners.cn2sd.refiner import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *
from ..util import *
from ..errfunc import ErrTypes, compute_bad_inf, compute_bad_score


inf = float('inf')



class Basic(object):


  def __init__(self, **kwargs):
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

    self.tablename = kwargs.get('tablename', None)


    

    self.scorer_cost = 0.
    self.merge_cost = 0.

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
    bdeltas, bcounts = Basic.bad_influences(self, rule)
    gdeltas, gcounts = Basic.good_influences(self, rule)
    gdeltas = map(abs, gdeltas)
    return bdeltas, bcounts, gdeltas, gcounts

  def influence_from_state(self, bdeltas, bcounts, gdeltas, gcounts, c=None):
    if c is None:
        c = self.c
    binf = compute_bad_score(bdeltas, bcounts, c)
    ginfs = [gdelta for gdelta,gcount in zip(gdeltas, gcounts) if gcount]
    ginf = ginfs and max(ginfs) or 0
    ret = self.l * binf - (1. - self.l) * ginf

    return ret


  @instrument
  def influences(self, rule, cs=[]):
    """
    compute influences for a list of c values
    """
    bdeltas, bcounts, gdeltas, gcounts = self.influence_state(rule)
    ginfs = [gdelta for gdelta,gcount in zip(gdeltas, gcounts) if gcount]
    
    ret = []
    for c in cs:
      binf = compute_bad_score(bdeltas, bcounts, c)
      ginf = ginfs and max(ginfs) or 0
      res = self.l * binf - (1. - self.l) * ginf

      ret.append(res)
    return ret

  @instrument
  def influence_cluster(self, cluster, table=None):
    rule = self.cluster_to_rule(cluster, table)
    inf_state = self.influence_state(rule)
    influence = self.influence_from_state(*inf_state)
    cluster.error = influence
    cluster.inf_state = inf_state
    return cluster.error


  def influence(self, rule, c=None):
    inf_state = self.influence_state(rule)
    quality = self.influence_from_state(*inf_state)
    rule.quality = quality
    rule.inf_state = inf_state
    return quality


  def bad_influences(self, rule):
    return self.compute_stat(rule, self.bad_err_funcs, self.bad_tables)

  def good_influences(self, rule):
    return self.compute_stat(rule, self.good_err_funcs, self.good_tables)


  def compute_stat(self, rule, err_funcs, tables):
    datas = rule and map(rule.filter_table, tables) or tables
    infs = []
    lens = []
    for idx, (ef, data) in enumerate(zip(err_funcs, datas)):
      if len(data) == len(tables[idx]):
        influence = -float('infinity')
      else:
        arr = data.to_numpyMA('ac')[0]
        influence = ef(arr.data)
      infs.append(influence)
      lens.append(len(data))
    return infs, lens


  def group_clauses(self, rules):
    """
    Cluster rules by clause
    """
    clause_to_id = defaultdict(lambda: 0)
    for rule in rules:
      for clause in rule.cond_strs:
        if clause not in clause_to_id:
          clause_to_id[clause] = len(clause_to_id)

    vectors = []
    for rule in rules:
      vector = np.zeros(len(clause_to_id))
      for clause in rule.cond_strs:
        vector[clause_to_id[clause]] = 1
      vectors.append(vector)


    try:
      clusterer = KMeans(3, n_init=4)
      clusterer.fit(vectors)
    except Exception as e:
      print "problem running kmeans"
      print e
      print '\n'.join(map(str, rules))
      return rules

    labels = clusterer.labels_
    rules = np.asarray(rules)
    ret = []
    for labelval in set(labels):
      idxs = labels == labelval
      labelrules = rules[idxs]
      strs = set()
      clauses = []
      for rule in labelrules:
        for cond, condstr in zip(rule.filter.conditions, rule.cond_strs):
          if condstr not in strs:
            strs.add(condstr)
            clauses.append(cond)
          
      newrule = SDRule(
          labelrules[0].data, 
          labelrules[0].targetClass, 
          [],
          labelrules[0].g)
      for clause in clauses:
        newrule = newrule.cloneAndAppendCondition(clause)
      ret.append(newrule)


    return ret
  
  def group_by_inf_state(self, rules):
    """
    super stupid check to group rules that have 
    identical influence states
    """
    state2rules = defaultdict(list)
    for r in rules:
      try:
        print r
        key = tuple(map(tuple, r.inf_state))
        state2rules[key].append(r)
      except:
        pdb.set_trace()

    ret = []
    for key, group in state2rules.iteritems():
      group.sort(key=lambda r: len(str(r)))
      rule = group[0]
      others = self.group_clauses(group[1:])
      rule.cluster_rules.update(others)
      ret.append(rule)
    return ret


  def group_rules(self, rules, nclusters=7):
      """
      Cluster rules by influence on good and bad results
      """
      if not rules: return rules
      if rules[0].c_range is not None:
        rules.sort(key=lambda r: r.c_range[1]-r.c_range[0], reverse=True)
        groups = []
        group = []
        for rule in rules:
          if not group:
            group.append(rule)
          elif group[0].c_range[0] == rule.c_range[0] and group[0].c_range[1] == rule.c_range[1]:
            group.append(rule)
          else:
            groups.extend(self.group_by_inf_state(group))
            group = [rule]
        if group:
          groups.extend(self.group_by_inf_state(group))
        return self.kmeans(groups, nclusters)
      else:
        rules.sort(key=lambda r: r.quality, reverse=True)
        return self.kmeans(rules, nclusters)

  def kmeans(self, rules, nclusters=7):
    if not rules: return rules
    #if len(rules) <= nclusters: return rules

    influences = []
    c = self.c
    def f((delta, count)):
      return compute_bad_inf(delta, count, c)

    nbadinfs = 0
    ngoodinfs = 0
    for rule in rules:
      bdeltas, bcounts, gdeltas, gcounts = rule.inf_state
      bad_infs = map(f, zip(bdeltas, bcounts))
      good_infs = []
      if self.good_tables:
        good_infs = gdeltas
      influences.append([rule.quality] + list(bad_infs) + list(good_infs))
      nbadinfs = len(bad_infs)
      ngoodinfs = len(good_infs)
    influences = np.asarray(influences)

    # normalize
    for colidx in xrange(influences.shape[1]):
      col = influences[:, colidx]
      mean, std = np.mean(col), np.std(col)
      # prefer bad influences
      if colidx == 0:
        std *= 1.5
      elif colidx <= nbadinfs:
        std *= 1.2
      col = (col - mean) 
      if std != 0: 
        col /= std
      influences[:,colidx] = col


    try:
      clusterer = KMeans(nclusters, n_init=4)
      clusterer.fit(influences)
    except Exception as e:
      print "problem running kmeans"
      print e
      print '\n'.join(map(str, rules))
      return rules

    words = ['gmmb', 'media', 'dc', 'washington']
    def get_features(r):
      ret = [word in str(r).lower() for word in words]
      q = r.quality - diff/100. * len(str(r))
      return q

    labels = clusterer.labels_
    rules = np.asarray(rules)
    diff = rules[0].quality - rules[min(len(rules)-1, 30)].quality
    ret = []
    for labelval in set(labels):
      idxs = labels == labelval
      labelrules = rules[idxs]

      # hack: try to find rules containing media buy or GMMC
      features = map(get_features, labelrules)
      #for idx, feature in enumerate(features):
        #feature.append(-1 * len(labelrules[idx]))
      #features = map(tuple, features)

      best_idx = max(range(len(labelrules)), key=lambda i: features[i])
      best = labelrules[best_idx]

      labelrules = filter(lambda r: r.id != best.id, labelrules)
      labelrules = self.group_clauses(labelrules)
      best.cluster_rules.update(labelrules)
      ret.append(best)
      
    return ret








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
          return [[-inf, inf]]

      block = (maxv - minv) / self.granularity
      ranges = [[minv + i*block, minv + (i+1)*block] for i in xrange(self.granularity)]
      ranges[0][0] = -inf
      ranges[-1][1] = inf
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


