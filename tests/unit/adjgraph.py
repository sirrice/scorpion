import pdb
import random
import numpy as np
from sklearn.neighbors import *
from scorpion.arch import create_orange_table
from scorpion.sigmod import AdjacencyGraph
from scorpion.bottomup import Cluster
random.seed(0)

vals = map(str, range(20))


rows = [ [v] for v in vals ]
table = create_orange_table(rows, ['moteid'])
domain = table.domain

clusters = []
for i in xrange(100):
  randvals = map(int, random.sample(vals, random.randint(1, 3)))
  c = Cluster([], 0, [], discretes={'moteid':randvals})
  clusters.append(c)
graph = AdjacencyGraph(clusters, domain)

metric = DistanceMetric.get_metric('hamming')
''.join(map(str,graph.feature_mapper(clusters[0], 'moteid')))
for n in graph.neighbors(clusters[0]):
  dist = metric.pairwise([
    graph.feature_mapper(clusters[0], 'moteid'),
    graph.feature_mapper(n, 'moteid')
  ])
  print '\t', dist[0], '\t', ''.join(map(str,graph.feature_mapper(n, 'moteid')))

