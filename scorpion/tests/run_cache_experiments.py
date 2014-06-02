import os
from sqlalchemy import create_engine
from common import *

def print_clusters(sub, dim, clusters=[], tuples=[], title=''):
  xattr = 'a_%d' % dim
  yattr = 'a_%d' % (dim+1)

  for cluster in clusters:
    bbox = tuple(map(list, zip(*cluster.bbox)))
    # we want bounds for attrs a_dim, a_dim+1, but bbox cols
    # may not be ordered as we like
    if xattr in cluster.cols:
      xidx = cluster.cols.index(xattr)
      x = bbox[xidx]
      x[0] = max(0, x[0])
      x[1] = min(100, x[1])
    else:
      x = [0, 100]

    if yattr in cluster.cols:
      yidx = cluster.cols.index(yattr)
      y = bbox[yidx]
      y[0] = max(0, y[0])
      y[1] = min(100, y[1])
    else:
      y = [0, 100]

    c = cm.jet(cluster.error)
    r = Rect((x[0], y[0]), x[1]-x[0], y[1]-y[0], alpha=min(1., max(0.2,cluster.error)), ec=c, fill=False, lw=1.5)
    sub.add_patch(r)

  if tuples:
    cols = zip(*tuples)
    xs, ys, cs = cols[dim], cols[dim+1], cols[-2]
    cs = np.array(cs) / 100.
    sub.scatter(xs, ys, c=cs, alpha=0.5, lw=0)

  sub.set_ylim(-5, 105)
  sub.set_xlim(-5, 105)
  sub.set_title(title)



def print_all_clusters(pp, db, tablename, learner, c):
  try:
    all_clusters = [cluster.clone() for cluster in learner.all_clusters]
    all_clusters = normalize_cluster_errors(all_clusters)
    clusters = [cluster.clone() for cluster in learner.final_clusters]
    clusters = normalize_cluster_errors(clusters)
    best_clusters = sorted(clusters, key=lambda c: c.error, reverse=True)
    best_clusters = best_clusters[:2]
    best_clusters[0].error = 1
    tuples = get_tuples_in_bounds(db, tablename, [], 'g = 7')

    for cl in clusters:
      print str(cl), cl.c_range

    for dim in xrange(len(tuples[0])-4):
      suffix = "%.4f dim %d" % (c, dim)
      fig = plt.figure(figsize=(12, 4))
      print_clusters(fig.add_subplot(1, 3, 1), dim, all_clusters, tuples=tuples,title="merged %s" % suffix)
      print_clusters(fig.add_subplot(1, 3, 2), dim, clusters, tuples=tuples,title="merged %s" % suffix)
      print_clusters(fig.add_subplot(1, 3, 3), dim, best_clusters, tuples=tuples, title="best %s" % suffix)
      plt.savefig(pp, format='pdf')

  except Exception as e:
    import traceback
    traceback.print_exc()
    pdb.set_trace()
    pass



def run(pp, cutoff, **params):
  dataset = params['dataset']
  test_datas = get_test_data(datasetnames[dataset])
  tablename = test_datas[-1]
  dbname = test_datas[0]
  db = create_engine('postgresql://localhost/%s' % dbname)

  costs, rules, all_ids, table_size, learner = run_experiment(dataset, **params)
  cost = costs['cost_total']
  ft = learner.full_table
  print len(ft)
  truth = [int(row['id'].value) for row in ft if row['v'] >= cutoff]
  all_stats = [compute_stats(ids, truth,  table_size) for ids in all_ids]

  stats, rule, ids = tuple(zip(all_stats, rules, all_ids)[0])
  data = tuple([tablename,params['c'],cost]+list(stats))
  print "stats:%s,c(%.3f),cost(%.2f),%.6f,%.6f,%.6f,%.6f" % data
  print 'stats:%s'% str(sdrule_to_clauses(rule.simplify())[0])
  print_all_clusters(pp, db, tablename, learner, params['c'])
  return costs


def warmup(dim, cutoff, **kwargs):
  dataset = "data_%d_%d_1000_0d50_%duo" % (dim, dim, cutoff)
  params = {
    'klass':BDT, 
    'nbadresults' : 10,
    'epsilon':0.005,
    'tau':[0.1, 0.5],
    'p' : 0.7,
    'l':.5,
    'min_pts' : 10,
    'min_improvement':.01,
    'granularity':15,
    'max_wait':1,
    'naive':False,
    'use_mtuples':False,
    'use_cache': False
  }
  params.update(kwargs)

  ft, bts, gts, truth, aggerr, cols = get_parameters(dataset, **params)
  params.update({
    'aggerr' : aggerr,
    'cols' : cols,
    'tablename' : dataset,
    'dataset' : dataset
  })
  learner = BDT(**params)
  learner.setup_tables(ft, bts, gts, **params)
  learner.get_partitions(ft, bts, gts, **params)


def run_cache(dim, cutoff, cs, **kwargs):
  dataset = kwargs.get('dataset', "data_%d_%d_1000_0d50_%duo" % (dim, dim, cutoff))

  params = {
    'klass':BDT, 
    'nbadresults' : 10,
    'epsilon':0.005,
    'tau':[0.1, 0.5],
    'p' : 0.7,
    'l':.5,
    'min_pts' : 10,
    'min_improvement':.01,
    'granularity':15,
    'max_wait':20,
    'naive':False,
    'use_mtuples':False,
    'use_cache': False,
    dataset: dataset
  }
  params.update(kwargs)


  pp = PdfPages('figs/topdown_all_%s.pdf' % str(dataset))
  cost_dicts = []
  for c in cs:
    params['c'] = c
    cost_dict = run(pp, cutoff, **params)
    cost_dicts.append(cost_dict)

  pp.close()
  return cost_dicts


def reset_cache():
  try:
    os.system('rm dbwipes*.cache')
  except:
    pass


if __name__ == '__main__':
  np.seterr(all='raise')
  if len(sys.argv) < 4:
    print "python run_cache_experiments.py [dimensions] [30|80] [cache? 0|1] [list of cs values]"
    print "cs values defaults to [.5, .4, .3, .2, .1, .05, 0]"
    sys.exit()

  dim = int(sys.argv[1])
  uo = int(sys.argv[2])
  cache = bool(int(sys.argv[3]))
  cs = map(float, sys.argv[4:])
  if not cs:
    cs = [.5, .4, .3, .2, .1, 0.05, 0.0] 

  #reset_cache()
  #cachecost_dicts = run_cache(dim, uo, cs, l=0.95, tree_alg='rt', klass=NDT, use_cache=cache, tau= [0.1, 0.5])

  cachecost_dicts = run_cache(dim, uo, cs, l=0.85, tree_alg='rt', klass=BDT, 
      epsilon=0.001, use_cache=cache, tau= [0.02, 0.5],
      c_range=[0.01, 0.7],
      dataset='data2clust_2_2_2k_vol20_uo80')




  print "c,total,partbad,partgood,split,merge,cache"
  for c, cd in zip(cs, cachecost_dicts):
    print "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d" % (
      c, 
      cd.get('cost_total', -1),
      cd.get('cost_partition_bad', -1),
      cd.get('cost_partition_good', -1),
      cd.get('cost_split', -1),
      cd.get('cost_merge', -1), 
      cache
    )
