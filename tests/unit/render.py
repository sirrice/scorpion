import random

from scorpion.bottomup.cluster import *
from scorpion.util.render import *


random.seed(0)

clusters = []
for i in xrange(50):
  x,y = random.randint(0, 90), random.randint(0, 90)
  bbox = [[x, y], [x+10, y+10]]
  c = Cluster(bbox, random.random(), ['x', 'y'])
  c.inf_state = [
      [random.randint(1, 50) for i in range(10)],
      [random.randint(1, 10) for i in range(10)],
      [],
      []
  ]
  c.c_range = [0, 1]
  c.inf_func = c.create_inf_func(0.5)
  clusters.append(c)

renderer = ClusterRenderer("/tmp/test.pdf")
renderer.plot_title("test title")
renderer.plot_clusters(clusters)
renderer.close()


renderer = InfRenderer('/tmp/infs.pdf')
renderer.plot_title("boooo")
renderer.plot_inf_curves(clusters)
renderer.close()
