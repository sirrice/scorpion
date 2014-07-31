import math
import pdb
import numpy as np
import Orange
from Orange.feature import Type as OType



class OverlapPenalty(object):
  def __init__(self, domain, cdists, ddists, granularity=100):
    """
    Args:
      domain: Orange.Domain object
    """
    self.domain = domain
    self.cdists = cdists
    self.ddists = ddists
    self.granularity = granularity
    self.funcs = {}
    self.allcounts = {}
    self.disc_nvals = {}
    self.setup()

  def setup(self):

    funcs = {}
    allcounts = {}
    for attr in self.domain:
      if attr.var_type == OType.Discrete:
        func, counts = self.setup_discrete_attribute(attr)
      else:
        func, counts = self.setup_continuous_attribute(attr)
      funcs[attr.name] = func
      allcounts[attr.name] = counts

    self.funcs, self.allcounts = funcs, allcounts

  def setup_continuous_attribute(self, attr):
    distribution = self.cdists[attr.name]
    minv, maxv = distribution.min, distribution.max
    if minv == maxv:
      func = lambda v: 0
      counts = np.zeros(1)
    else:
      def make_func(minv, block, gran):
        def f(v):
          return int(min(gran, max(0, math.ceil((v-minv)/block))))
        return f
      block = (maxv - minv) / float(self.granularity)
      counts = np.zeros(self.granularity+1)
      func = make_func(minv, block, self.granularity)

    return func, counts

  def setup_discrete_attribute(self, attr):
    vals = self.ddists[attr.name].keys()
    d = {val: idx for idx, val in enumerate(vals)}
    
    # values will be passed in as indices into vals
    def func(v):
      if v < 0 or v > len(vals) or v is None:
        return len(vals)
      return v
    counts = np.zeros(len(vals)+1)
    self.disc_nvals[attr.name] = len(vals)
    return func, counts

  def reset_counts(self):
    for counts in self.allcounts.values():
      counts[:] = 0

  def continuous_idxs(self, attr, minv, maxv):
    if isinstance(attr, basestring):
      name = attr
    else:
      name = attr.name

    dist = self.cdists[name]
    if minv <= dist.min and maxv >= dist.max:
      return []
    func = self.funcs[name]
    return np.arange(func(minv),func(maxv)+1)
  
  def discrete_idxs(self, attr, vals):
    if isinstance(attr, basestring):
      name = attr
    else:
      name = attr.name

    if len(vals) == self.disc_nvals[name]:
      return np.array([])
    func = self.funcs[name]
    return np.array(map(func, vals))

  def __call__(self, clusters, min_weight=0.7):
    """
    return weights to multiply to each cluster's influence
    """
    penalties = self.penalties(clusters)
    weights = 1. - penalties
    weights[weights <= min_weight] = min_weight
    return weights

  def penalties(self, clusters):
    """
    Compute a penalty for each cluster
    Return is normalzed to [0, 1]
    """
    self.reset_counts()
    penalties = np.array(map(self.penalty, clusters))
    if penalties.max() == 0:
      return penalties
    penalties /= penalties.max()
    return penalties

  def penalty(self, cluster):
    totals = {}
    for col, (minv, maxv) in zip(cluster.cols, zip(*cluster.bbox)):
      idxs = self.continuous_idxs(col, minv, maxv)
      if len(idxs):
        totals[col] = self.allcounts[col][idxs]
        self.allcounts[col][idxs] += .5

    for col, vals in cluster.discretes.iteritems():
      idxs = self.discrete_idxs(col, vals)
      if len(idxs):
        totals[col] = self.allcounts[col][idxs]
        self.allcounts[col][idxs] += 1

    smooth = lambda counts: max(0, (counts - 0.5).max())
    return sum(map(smooth, totals.values()))

def create_clusters(n):
  clusters = []
  for i in xrange(n):
    minv = random.random() * 70
    maxv = minv + 10
    bbox = ((minv,), (maxv,))
    a = ['a'+str(j) for j in range(3)]
    x = ['x'+str(j) for j in range(10)]
    discretes = {
      'a': nprand.choice(a, 2, replace=False),
      'x': nprand.choice(x, 3, replace=False)
    }
    cluster = Cluster(bbox, 0, ['c'], discretes=discretes)
    clusters.append(cluster)
  return clusters


if __name__ == '__main__':

  import pdb
  import random
  import timeit
  import numpy.random as nprand
  from scorpion.bottomup.cluster import Cluster
  from scorpion.arch import *
  rows = []
  cols = ['a', 'x', 'c']
  for i in xrange(100):
    a = 'a'+str(i%3)
    b = 'x'+str(i%10)
    c = 100 * random.random() 
    rows.append([a, b, c])

  table = create_orange_table(rows, cols)
  domain = table.domain
  cdists = dict(zip(cols, Orange.statistics.basic.Domain(table)))
  ddists = dict(zip(cols, Orange.statistics.distribution.Domain(table)))
  overlap = OverlapPenalty(domain, cdists, ddists)

  clusters = create_clusters(20)
  weights = overlap(clusters)
  print weights

  import profile, pstats, StringIO
  for n in [10, 20, 50]:
    clusters = create_clusters(n)
    f = lambda: overlap(clusters)
    print n, '\t', timeit.timeit(f, number=100)
    continue
    pr = profile.Profile()
    pr.run("print n, '\t', timeit.timeit(f, number=100)")
    s = StringIO.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('time')
    ps.print_stats()
    print s.getvalue()
  



