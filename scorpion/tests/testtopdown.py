import pdb
import sys
import random
import time
import sys
import matplotlib
import numpy as np
sys.path.extend( ['.', '..'])

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle as Rect
from matplotlib import cm 
import matplotlib.pyplot as plt
from collections import defaultdict
from sqlalchemy import create_engine

from scorpionsql.db import *
from scorpionsql.aggerror import *

from arch import *
from gentestdata import *
from util import reconcile_tables
from sigmod import *
from common import *

matplotlib.use("Agg")



def print_clusters(pp, clusters, tuples=[], title=''):
    fig = plt.figure(figsize=(8, 8))
    sub = fig.add_subplot(111)
    clusters.sort(key=lambda c: c.error)

    for cluster in clusters:
      x, y = tuple(map(list, zip(*cluster.bbox)))[:2]
      x[0] = max(0, x[0])
      x[1] = min(100, x[1])
      y[0] = max(0, y[0])
      y[1] = min(100, y[1])
      c = cm.jet(cluster.error)
      r = Rect((x[0], y[0]), x[1]-x[0], y[1]-y[0], alpha=max(0.1,cluster.error), ec=c, fill=False, lw=1.5)
      sub.add_patch(r)

    if tuples:
      cols = zip(*tuples)
      xs, ys, cs = cols[0], cols[1], cols[-1]
      sub.scatter(ys, xs, c=cs, alpha=0.5, lw=0)


    sub.set_ylim(-5, 105)
    sub.set_xlim(-5, 105)
    sub.set_title(title)
    plt.savefig(pp, format='pdf')



def run(pp, datasetidx, bounds, **params):
  test_datas = get_test_data(datasetnames[datasetidx])
  tablename = test_datas[-1]
  dbname = test_datas[0]
  db = create_engine('postgresql://localhost/%s' % dbname)

  costs, rules, all_ids, table_size, learner = run_experiment(datasetidx, **params)
  cost = costs['cost_total']


  try:
    truth = set(get_ids_in_bounds(db, tablename, bounds))
    all_stats = [compute_stats(ids, truth,  table_size) for ids in all_ids]
  except Exception as e:
    all_stats = map(learner.compute_stats, all_ids)

  #  print "\n".join(map(str, learner.costs.items()))
  for stats, rule, ids in zip(all_stats, rules, all_ids)[:1]:
    print "stats:%s,c(%.3f),cost(%.2f),%.6f,%.6f,%.6f,%.6f" % tuple([tablename,params['c'],cost]+list(stats))
    print 'stats:%s'% str(sdrule_to_clauses(rule.simplify())[0])

  try:
    clusters = normalize_cluster_errors([c.clone() for c in learner.final_clusters])
    best_clusters = sorted(clusters, key=lambda c: c.error, reverse=True)[:1]


    tuples = get_tuples_in_bounds(db, tablename, [(0,100), (0,100)], 'g = 7')
    cols = zip(*tuples)

    for idx in xrange(len(cols)-4):
      tuples = zip(cols[idx], cols[idx+1], np.array(cols[-2])/100.)
      print_clusters(pp, clusters,  tuples=tuples,title="merged clusters %.4f dim %d" % (params['c'], idx))
      print_clusters(pp, best_clusters[:10],  tuples=tuples, title="best clusters %.4f dim %d" % (params['c'], idx))
  except Exception as e:
    print e
    pdb.set_trace()
    pass




       

if __name__ == '__main__':
    

  np.seterr(all='raise')
  nbadresults = 10
  idxs = sys.argv[1:] or [0,1]
  bounds = [[42.2210925762524, 92.2210925762524], [37.89772014701512, 87.89772014701512]] 
  #bounds = [[52.73538209702353, 77.73538209702353], [44.3706389043392, 69.3706389043392]]
  bounds = [[24.73254341295866, 95.44322153161342], [22.19997047910137, 92.91064859775612], [12.31825640510083, 83.02893452375558], [7.583496039802494, 78.29417415845725]]
  cs = reversed([0., 0.05, 0.075, 0.09, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 0.75, 1.])
  cs = [0.25, 0.15, 0.125, 0.1, 0.05]
  cs = [0.25, 0.1]
  for datasetidx in idxs:
    pp = PdfPages('figs/topdown_all_%s.pdf' % str(datasetidx))
    for c in cs:
      run(pp, datasetidx, bounds,
        klass=BDT, 
        nbadresults = nbadresults,
        epsilon=0.0015,
        tau=[0.1, 0.5],
        p = 0.7,
        l=.5,
        min_pts = 10,
        min_improvement=.01,
        granularity=15,
        max_wait=1,#5*60,#None,
        naive=False,
        use_mtuples=False,
        use_cache=True,
        cs=cs,
        c=c
      )

    pp.close()

#
#  cluster = None
#  while True:
#      print "set cluster to a value"
#      pdb.set_trace()
#      if not cluster:
#          break
#      pp = PdfPages('figs/topdown_%s.pdf' % outname)
#      topdown.merger.adj_matrix.insert(cluster)
#      neighbors = topdown.merger.adj_matrix.neighbors(cluster)
#      for n in neighbors:
#          n.error = 0.5
#      cluster.error = 1
#      print_clusters(pp, list(neighbors) + [cluster], title='foo')
#
#      pp.close()
#
#


