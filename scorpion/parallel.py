import random
import sys
import time
import pdb
import traceback
import errfunc
import numpy as np

from sklearn.cluster import KMeans
from itertools import chain
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process, Queue, Pool
from Queue import Empty

from db import *
from aggerror import *
from arch import *
from util import *
from sigmod import *



_logger = get_logger()


def parallel_debug(sharedobj, **kwargs):
    for aggerr in sharedobj.errors:
        parallel_runner(sharedobj, aggerr, **kwargs)
    create_clauses(sharedobj)
    

def parallel_runner(sharedobj, aggerr, **kwargs):
    random.seed(2)
    sql = sharedobj.sql
    badresults = aggerr.keys
    goodresults = sharedobj.goodkeys.get(aggerr.agg.shortname, None)
    label = aggerr.agg.shortname
    if not badresults:
        return

    bad_tuple_ids = set()
    for col in aggerr.agg.cols:
        bad_tuple_ids.update( sharedobj.get_bad_tuple_ids(col) )

    if False and bad_tuple_ids:
        
        table = get_provenance(sharedobj, aggerr.agg.cols, aggerr.keys)
        scores = []
        for row in table:
            row[ERROR_VAR] = '1'            
            if row['id'].value in bad_tuple_ids:
                scores.append( 1 )
            else:
                scores.append( 0 )
        _logger.debug("Classifying using user specified tuple ids")
        return table, []


    else:
        kwargs['ignore_attrs'] = kwargs.get('ignore_attrs', sharedobj.ignore_attrs)
        #cost, ncalls, table, rules = parallel_hybrid(sharedobj, aggerr, **kwargs)
        cost, ncalls, table, rules = serial_hybrid(sharedobj, aggerr, **kwargs)

    sharedobj.merged_tables[label] = table
    rules = zip(rules, reversed(range(len(rules))))
    sharedobj.rules[label] = rules
    return table, rules



def serial_hybrid(obj, aggerr, **kwargs):
    costs = {}
    db = connect(obj.dbname)
    obj.db = db
    start = time.time()
    all_keys = list(chain(aggerr.keys, obj.goodkeys[aggerr.agg.shortname]))
    all_tables = get_provenance_split(obj, aggerr.agg.cols, all_keys)
    bad_tables = all_tables[:len(aggerr.keys)]
    good_tables = all_tables[len(aggerr.keys):]
    costs['data_load'] = time.time() - start

    _logger.debug("bad table counts:  %s" % ', '.join(map(str, map(len, bad_tables))))
    _logger.debug("good table counts: %s" % ', '.join(map(str, map(len, good_tables))))
    print "agg error %s \t %s" % (aggerr.agg, aggerr.errtype)

    
    cost, ncalls = 0, 0
    rules = []
    try:
        full_start = time.time()
        start = time.time()
        cols = valid_table_cols(bad_tables[0], aggerr.agg.cols, kwargs)
        all_cols = cols + aggerr.agg.cols        
        torm = [attr.name for attr in bad_tables[0].domain if attr.name not in all_cols]
        _logger.debug("valid cols: %s" % cols)

        bad_tables = [rm_attr_from_domain(t, torm) for t in bad_tables]
        good_tables = [rm_attr_from_domain(t, torm) for t in good_tables]
        all_full_table = union_tables(bad_tables, good_tables)
        full_table = union_tables(bad_tables)

        costs['data_setup'] = time.time() - start


        # make sure aggerr keys and tables are consistent one last time
        if len(bad_tables) != len(aggerr.keys):
          pdb.set_trace()
          raise RuntimeError("#badtables (%d) != #aggerr keys (%d)" % (len(bad_tables), len(aggerr.keys)))


        params = {
          'aggerr':aggerr,
          'cols':cols,
          'c': obj.c,
          'aggerr':aggerr,
          'cols':cols,
          'c': obj.c,
          'c_range': [0.05, 1],
          'l' : 0.6,
          'msethreshold': 0.01,
          'max_wait':5,
          'DEBUG': False
        }

        params.update(dict(kwargs))

        if aggerr.agg.func.__class__ in (errfunc.SumErrFunc, errfunc.CountErrFunc):
          klass = MR 
          params.update({
            'use_cache': False,
            'use_mtuples': False,
            'granularity': 100
            })
          params['c'] = params.get('c', .15)

        else:
          klass = BDT
          params.update({
            'use_cache': True,
            'use_mtuples': False,#True,
            'epsilon': 0.0015,
            'min_improvement': 0.01,
            'tau': [0.08, 0.5],
            'p': 0.7
            })
          params['c'] = params.get('c', .3)

        #klass = SVM
        #params.update({})
        _logger.debug("c is set to: %.4f", params['c'])

        start = time.time()
        hybrid = klass(**params)
        clusters = hybrid(all_full_table, bad_tables, good_tables)
        rules = clusters_to_rules(clusters, full_table)
        print "nclusters: %d" % len(clusters)
        costs['rules_get'] = time.time() - start

        _logger.debug("clustering %d rules" % len(rules))
        for r in rules[:10]:
          _logger.debug("%.4f\t%.4f - %.4f\t%s" % (r.quality, r.c_range[0], r.c_range[1], str(r)))


        clustered_rules = hybrid.group_rules(rules, 5)
        rules = clustered_rules
        costs['rules_cluster'] = time.time() - start

        ncalls = 0
    except:
        traceback.print_exc()

    
    # return the best rules first in the list
    start = time.time()
    rules.sort(key=lambda r: r.c_range[0])
    rules = [r.simplify(all_full_table) for r in rules[:10]]
    costs['rules_simplify'] = time.time() - start

    cost = time.time() - full_start


    print "found rules"
    for rule in rules[:5]:
      print "%.5f\t%s" % (rule.quality, rule)

    print "=== Costs ==="
    for key, cost in costs.iteritems():
      print "%.5f\t%s" % (cost, key)
    
    return cost, ncalls, table, rules



   

def parallel_hybrid(obj, aggerr, **kwargs):

    db = connect(obj.dbname)
    obj.db = db

    def f(bad_tables, aggerr, klass, params, kwargs, queue):
      try:
        cols = valid_table_cols(bad_tables[0], aggerr.agg.cols, kwargs)
        all_cols = cols + aggerr.agg.cols        
        torm = [attr.name for attr in bad_tables[0].domain 
                if attr.name not in all_cols]

        bad_tables = [rm_attr_from_domain(t, torm) for t in bad_tables]
        good_tables = []
        _, full_table = reconcile_tables(bad_tables)
 
        start = time.time()
        hybrid = klass(**params)
        clusters = hybrid(full_table, bad_tables, good_tables)
        normalize_cluster_errors(clusters)
        rules = clusters_to_rules(clusters, full_table)
        cost = time.time() - start
        ncalls = 0
        
        queue.put( (rules, cost, ncalls) )
      except:
        traceback.print_exc()
        queue.put(None)

    nprocesses = kwargs.get('nprocesses', 4)
    parallelize = kwargs.get('parallelize', True)


    db = connect(obj.dbname)
    badresults = aggerr.keys
    queue = Queue()
    npending = 0    
    totalcost, totalncalls = 0., 0.

    bad_partition_tables = parallel_get_partitions(obj, aggerr, nprocesses)
    all_tables = []
    map(all_tables.extend, bad_partition_tables)
    _, mastertable = reconcile_tables(all_tables)
    cols = valid_table_cols(all_tables[0], aggerr.agg.cols, kwargs)

    params = {
        'aggerr':aggerr,
        'cols':cols,
        'c' : 0.2,
        'l' : 0.5,
        'msethreshold': 0.01
        }
    params.update(dict(kwargs))

    if aggerr.agg.func.__class__ in (errfunc.SumErrFunc, errfunc.CountErrFunc):
      klass = MR 
      params.update({
        'use_mtuples': False,
        'c': 0,
        })
    else:
      klass = BDT
      params.update({
        'use_cache': False,
        'use_mtuples': False,#True,
        'epsilon': 0.005,
        'min_improvement': 0.01,
        'tau': [0.1, 0.5],
        'c' : 0.3,
        'p': 0.7
        })

    
    for bad_tables in bad_partition_tables:
        args = (bad_tables, aggerr, klass, params, kwargs, queue)
        if parallelize:
            p = Process(target=f, args=args)
            p.start()
        else:
            f(*args)
        npending += 1

    results = []
    start = time.time()
    while npending > 0:
        try:
            result = queue.get(timeout=1)
            npending -= 1            
            if result:
                rules, cost, ncalls = result
                totalncalls += ncalls
                results.extend(rules)
            else:
                print "got error"

        except:
            pass

    results.sort(key=lambda r: r.quality, reverse=True)
    hybrid = klass(**params)
    hybrid.setup_tables(mastertable, all_tables, [])
    results = hybrid.group_rules(results)

    totalcost = time.time() - start
    db.close()

    return totalcost, totalncalls, mastertable, results

    partitions = map(merge_tables, bad_partition_tables)
    results, merge_cost = parallel_rank_rules(aggerr, partitions, results, **kwargs)

    print "found rules"
    print '\n'.join(map(str, results))
    
    return totalcost, totalncalls, mastertable, results



def normalize_rules(rules):
    qualities = [r.quality for r in rules if r.quality is not None]
    if not len(qualities):
        for r in rules:
            r.quality = 0
        return rules

    minv, maxv = min(qualities), max(qualities)
    div = 1. if maxv == minv else maxv-minv
    for r in rules:
        if r.quality is None:
            r.quality = 0
        else:
            r.quality = (r.quality - minv) / div
    return rules




def parallel_rank_rules(aggerr, tables, rules, **kwargs):
    """
    Parallel rank rules based on xval
    """
    def f(aggerr, table, rules, queue):
        try:
            err_func = aggerr.error_func
            err_func.setup(table)

            cloned_rules = []
            for rule in rules:
                rule = rule.cloneWithNewData(table)
                rule.set_data(table)

                if not len(rule.examples):
                    quality = None
                else:
                    quality = err_func(rule.examples.to_numpyMA('a')[0]) / len(rule.examples)
                rule.quality = quality
                cloned_rules.append(rule)


            normalize_rules(cloned_rules)
            scores = [rule.quality or 0. for rule in cloned_rules]
            queue.put(scores)
        except:
            traceback.print_exc()
            queue.put( None )


    parallelize = kwargs.get('parallelize', True)
    rules = list(rules)
    
    queue = Queue()
    npending = 0

    for table in tables:
        if parallelize:
            Process(target=f, args=(aggerr, table, rules, queue)).start()
        else:
            f(aggerr, table, rules, queue)
        npending += 1


    
    start = time.time()
    rule_scores = defaultdict(list)
    while npending > 0:
        try:
            result = queue.get(timeout=1)
            npending -= 1
            _logger.debug("parallel_process_results\t%d processes left", npending)
            if not result:
                continue

            for rule, score in zip(rules, result):
                rule_scores[rule].append(score)

        except Empty:
            continue
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()

    for rule, scores in rule_scores.iteritems():
        rule.quality = np.mean(scores)
        
    rules = rule_scores.keys()
    rules.sort(key=lambda r: r.quality, reverse=True)
    rules = remove_subsumed_rules(rules)

    cost = time.time() - start
    return rules, cost
        








class ProvenanceGetter(object):
  def __init__(self, obj, cols):
      self.obj = obj
      self.cols = cols

  def __call__(self, keysubrange):
    obj = self.obj.clone()
    try:
      db = connect(obj.dbname)
      obj.db = db
      return get_provenance_split(obj, self.cols, keysubrange)
      return get_provenance(obj, self.cols, keysubrange)
    except:
      traceback.print_exc()
    finally:
      try:
        db.close()
      except:
        pass


def parallel_get_partitions(obj, aggerr, nprocesses):
  """
  Returns a list of [table for each key subrange] lists
  """
  pool = Pool(8)
  queue = Queue()
  processes = []
  args = [ ]
  chunksize = max(1, len(aggerr.keys) / nprocesses)
  per_process_keys = block_iter(aggerr.keys, nprocesses)
  f = ProvenanceGetter(obj, aggerr.agg.cols)
  partition_tables = pool.map(f, per_process_keys)
  pool.close()
  partition_tables = filter(lambda x:x, partition_tables)
  print "got %s tables" % map(len, partition_tables)
  return partition_tables



def merge_tables(tables):
    discretes = defaultdict(list)
    conts = set()
    for table in tables:
        for attr in table.domain:
                conts.add(attr.name)

    features = []
    for attr in tables[0].domain:
        if attr.name in discretes:
            features.append(Orange.feature.Discrete(attr.name, values=discretes[attr.name]))
        else:
            features.append(Orange.feature.Continuous(attr.name))
    
    domain = Orange.data.Domain(features)
    domain.add_metas(tables[0].domain.get_metas())
    ret = Orange.data.Table(domain)
    for table in tables:
        ret.extend(table)
    return ret





if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

