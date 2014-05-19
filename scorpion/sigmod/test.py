import pdb
import random
from crange import *
from matplotlib import pyplot as plt
random.seed(.2)


class Cluster(object):
  _id = 0
  def __init__(self, tops, bots):
    self.id = Cluster._id
    Cluster._id += 1

    self.tops = tops
    self.bots = bots
    self.c_range = [0.0, 1]
    pairs = zip(self.tops, self.bots)
    self.inf_func = lambda c: np.mean([t/pow(b,c) for t,b in pairs])

  def clone(self):
    c = Cluster(self.tops, self.bots)
    c.c_range = list(self.c_range)
    return c


  def __str__(self):
    return "%d\t%s\t%s\t%.4f, %.4f\t%.4f - %.4f" % (self.id, self.tops, self.bots, self.c_range[0], self.c_range[1],
        self.inf_func(0), self.inf_func(1))

clusters = []
top, bot = 1, 1
for i in xrange(1500):
  tops = [float(random.randint(0, 20)) for i in xrange(5)]
  bots = [float(random.randint(1, 20)) for i in xrange(5)]
  clusters.append(Cluster(tops, bots))



get_frontier = Frontier([0,1])
frontier, removed = get_frontier(clusters)
frontier = list(frontier)
removed = list(removed)


def frange(b):
  return (np.arange(50) / 50. * r_vol(b)) + b[0]

def render(ax, c, xs, color='grey', alpha=0.3):
  ys = map(c.inf_func, xs)
  ax.plot(xs, ys, alpha=alpha, color=color)

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
xs = frange([0, 1])
miny = min([min(c.inf_func(xs[0]), c.inf_func(xs[1])) for c in clusters])
maxy = max([max(c.inf_func(xs[0]), c.inf_func(xs[1])) for c in clusters])
ax.set_xlim(0, 1)
ax.set_ylim(miny, maxy)

for c in removed:
  render(ax, c, xs, color='grey', alpha=.2)
for c in frontier:
  render(ax, c, frange(c.c_range), color='red')
fig.savefig('test.pdf')



for c in frontier:
  print c
print 'removed'
for c in removed:
  print c

a = frontier[0]
if removed:
  b = removed[-1]
  a.c_range = [0., 1.]
  b.c_range = [0., 1.]
  get_frontier.intersect_ranges(a, b)

