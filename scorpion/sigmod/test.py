import pdb
import random
import numpy as np
from frontier import *
from matplotlib import pyplot as plt
random.seed(.2)

from scorpion.util import InfRenderer


class Cluster(object):
  _id = 0
  def __init__(self, tops, bots):
    self.id = Cluster._id
    Cluster._id += 1

    self.tops = tops
    self.bots = bots
    self.c_range = [0.0, 1]
    pairs = zip(self.tops, self.bots)
    f = lambda c: np.mean([t/pow(b,c) for t,b in pairs])
    self.inf_func = np.vectorize(f)

  def clone(self, *args, **kwargs):
    c = Cluster(self.tops, self.bots)
    c.c_range = list(self.c_range)
    return c
  
  @property
  def bound_hash(self):
    return hash(str([self.tops, self.bots, self.c_range]))

  def __hash__(self):
    return hash(str([self.tops, self.bots, self.c_range]))

  def __str__(self):
    return "%d\t%s\t%s\t%.4f, %.4f\t%.4f-%.4f\t%.4f - %.4f" % (
        self.id, self.tops, self.bots, self.c_range[0], self.c_range[1],
        self.c_range[0], self.c_range[1], self.inf_func(0), self.inf_func(1))

clusters = []
top, bot = 1, 1
for i in xrange(1000):
  tops = [float(random.randint(0, 20)) for i in xrange(5)]
  bots = [float(random.randint(1, 20)) for i in xrange(5)]
  c = Cluster(tops, bots)
  clusters.append(c)





renderer = InfRenderer('test.pdf')


get_frontier = Frontier([0,1])
start = time.time()
frontier, removed = get_frontier(clusters)
print time.time() - start
for key, val in get_frontier.stats.items():
  print key, '\t', val
for key, val in get_frontier.heap.stats.items():
  print key, '\t', val

renderer.plot_inf_curves(removed)
renderer.plot_inf_curves(frontier)
renderer.plot_active_inf_curves(frontier)
renderer.new_page()
renderer.plot_inf_curves(removed)
renderer.plot_inf_curves(frontier)

for c in clusters:
  c.c_range = [0,1]

f2 = ContinuousFrontier([0,1])
start = time.time()
size = 100
f2.update(clusters[:500])

for c in clusters[500:]:
  map(len, f2.update([c]))

print time.time() - start
for key, val in f2.stats.items():
  print key, '\t', val
for key, val in f2.heap.stats.items():
  print key, '\t', val


frontier = f2.frontier
renderer.plot_active_inf_curves(frontier)
renderer.close()

for c in frontier:
  print c
if False:
  print 'removed'
  for c in removed:
    print c

