import pdb
import random
import numpy as np
from matplotlib import pyplot as plt
random.seed(.2)

from scorpion.sigmod.frontier import *
from scorpion.util import InfRenderer


class Cluster(object):
  _id = 0
  def __init__(self, tops, bots):
    self.id = Cluster._id
    Cluster._id += 1

    self.tops = tops
    self.bots = bots
    self.c_range = [0.0, 1]
    self.error = 0
    pairs = zip(self.tops, self.bots)
    f = lambda c: np.mean([t/pow(b,c) for t,b in pairs])
    self.inf_func = np.vectorize(f)

  def clone(self, *args, **kwargs):
    c = Cluster(self.tops, self.bots)
    c.c_range = list(self.c_range)
    return c
  
  @property
  def bound_hash(self):
    return hash(str([self.tops, self.bots]))

  def __hash__(self):
    return hash(str([self.tops, self.bots, self.c_range]))

  def __str__(self):
    return "%d\t%s\t%s\t%.4f, %.4f\t%.4f-%.4f\t%.4f - %.4f" % (
        self.id, self.tops, self.bots, self.c_range[0], self.c_range[1],
        self.c_range[0], self.c_range[1], self.inf_func(0), self.inf_func(1))

clusters = []
top, bot = 1, 1
for i in xrange(300):
  tops = [float(random.randint(0, 20)) for i in xrange(5)]
  bots = [float(random.randint(1, 20)) for i in xrange(5)]
  c = Cluster(tops, bots)
  clusters.append(c)



#xs = (np.arange(100) / 100.)
#all_infs = np.array([c.inf_func(xs) for c in clusters])
#print all_infs.shape
#medians = np.percentile(all_infs, 50, axis=0)
#print medians.shape
#print medians
#block = (medians.max() - medians.min()) / 50.
#print block
#opts = [0]
#ys = [medians[0]]
#prev = medians[0]
#for x, v in enumerate(medians):
#  if abs(v-prev) >= block:
#    opts.append(xs[x])
#    ys.append(v)
#    prev = v
#
#print len(opts)
#print opts
#print ys
#
#opts = np.array(opts)
#weights = opts[1:] - opts[:-1]
#weights = weights.astype(float) / weights.sum()
#tup = np.polyfit(opts[1:], weights, 2, full=True)
#print tup[0]
#print tup[1]
#def create_polynomial(coefficients):
#  """
#  Given a set of coefficients, return a function that takes a dataset size and computes the cost
#
#  Coefficients are for increasing orders of x.  E.g.,
#
#    [2,9,5] -> 2*x^0 + 9*x^2 + 5*x^3
#  """
#
#  def f(x):
#    return sum([a * (x**power) for power, a in enumerate(coefficients)])
#  f.coeffs = coefficients
#  return f
#
#f = create_polynomial(tup[0])
#
#
#
#renderer = InfRenderer('test.pdf')
#renderer.plot_inf_curves(clusters, color='grey', alpha=0.3)
#renderer.plot_points(xs, medians, color='red', alpha=1)
#renderer.plot_points(xs, np.average(all_infs, axis=0), color='blue', alpha=1)
#for x in opts:
#  renderer.plot_points([x, x], [0, 20], color='black')
#
#renderer.plot_points(xs, f(xs)*20, color='green', alpha=1)
#renderer.close()
#
#
#exit()


#clusters = []
#for i in xrange(5):
#  tops = [1,2,3,4,5]
#  bots = [1,3,5,2,1]
#  clusters.append(Cluster(tops, bots))


renderer = InfRenderer('test.pdf')

def print_stats(frontier):
  for key, val in frontier.stats.items():
    print key, '\t', val
  for key, val in frontier.heap.stats.items():
    print key, '\t', val


if True:
  get_frontier = Frontier([0,1])
  start = time.time()
  frontier, removed = get_frontier(clusters)
  print time.time() - start
  print_stats(get_frontier)

  renderer.plot_inf_curves(clusters, color='grey', alpha=0.3)
  renderer.plot_active_inf_curves(frontier)
  for c in frontier:
    print c
  if False:
    print 'removed'
    for c in removed:
      print c


if True:
  for c in clusters:
    c.c_range = [0,1]

  f = CheapFrontier([0,1], K=1, nblocks=25)
  start = time.time()
  frontier, removed = f(clusters)
  print time.time() - start
  print_stats(f)

  renderer.new_page()
  renderer.plot_inf_curves(clusters, color='grey', alpha=0.3)
  renderer.plot_active_inf_curves(frontier)
  for c in frontier:
    print c
  if False:
    print 'removed'
    for c in removed:
      print c

if True:
  for c in clusters:
    c.c_range = [0,1]

  f = CheapFrontier([0,1], K=1, nblocks=25)
  start = time.time()
  f.update(clusters[:50])
  f.update(clusters[50:])
  print time.time() - start
  print_stats(f)

  frontier = f.frontier
  renderer.new_page()
  #renderer.plot_inf_curves(clusters, color='grey', alpha=0.3)
  renderer.plot_active_inf_curves(frontier)
  for c in frontier:
    print c




if False:
  for c in clusters:
    c.c_range = [0,1]

  f2 = ContinuousFrontier([0,1])
  start = time.time()
  for c in clusters:
    map(len, f2.update([c]))

  print time.time() - start
  print_stats(f2)

  frontier = f2.frontier
  renderer.new_page()
  renderer.plot_inf_curves(clusters, color='grey', alpha=0.3)
  renderer.plot_active_inf_curves(frontier)

renderer.close()


