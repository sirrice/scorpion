import sys
import random
import time
import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle as Rect
from matplotlib import cm 
from collections import defaultdict

from scorpionsql.db import *
from scorpionsql.aggerror import *
from scorpion.arch import *
from scorpion.tests.gentestdata import *
from scorpion.util import reconcile_tables
from scorpion.sigmod import *

matplotlib.use("Agg")

def get_row_ids(r, table):
    return set([row['id'].value for row in r(table)])

def get_tuples_in_bounds(db ,tablename, bounds, additional_where=''):
    with db.begin() as conn:
        where = ['%f <= a_%d and a_%d <= %f' % (minv, i, i, maxv) for i, (minv, maxv) in enumerate(map(tuple, bounds))]
        if additional_where:
            where.append(additional_where)
        where = ' and '.join(where)
        q = """ select * from %s where %s""" % (tablename, where)
        return [map(float, row) for row in conn.execute(q)]



def get_ids_in_bounds(db ,tablename, bounds):
  """
  Given a synthetic tablename and bounds of the form [ [min1, max1], ... ]
  Return the ids of the records that match
  """
  with db.begin() as conn:
      where = ['%f <= a_%d and a_%d <= %f' % (minv, i, i, maxv) for i, (minv, maxv) in enumerate(map(tuple, bounds))]
      where = ' and '.join(where)
      q = """ select id from %s where %s""" % (tablename, where)
      return [int(row[0]) for row in conn.execute(q)]




def compute_stats(found_ids, bad_tuple_ids, table_size):
    bad_tuple_ids = set(bad_tuple_ids)
    found_ids = set(found_ids)
    n = len(found_ids)        
    tp = len(bad_tuple_ids.intersection(found_ids))
    fn = len(bad_tuple_ids.difference(found_ids))
    fp = len(found_ids.difference(bad_tuple_ids))
    tn = table_size - tp - fn - fp

    accuracy = float(tp + tn) / table_size
    precision = float(tp) / (tp + fp) if n and (tp+fp) else 0.
    recall = float(tp) / (tp + fn) if n and (tp+fn) else 0.
    f1 = (precision*recall*2.) / (precision+recall) if precision+recall > 0 else 0.

    return accuracy, precision, recall, f1

def kname(klass):
    return klass.__name__[-7:]



def strip_columns(table, aggerr, cols=None):
    cols = cols or [attr.name for attr in table.domain]
    cols = [col for col in cols 
            if (col not in ['id', 'err', 'epochid', 'date', 'light'] and 
                col not in aggerr.agg.cols)]
    all_cols = cols + aggerr.agg.cols
    torm = [attr.name for attr in table.domain if attr.name not in all_cols]
    table = rm_attr_from_domain(table, torm)
    return table, cols



def get_parameters(datasetidx, **kwargs):
    name = datasetnames[datasetidx]
    outname = name

    test_data = get_test_data(name)
    dbname, sql, badresults, goodresults, errtype, get_ground_truth, tablename = test_data
    obj, table = create_sharedobj(*test_data[:-2])
    aggerr = obj.errors[0]

    # retrieve table for each good and bad key
    obj.db = connect(dbname)
    bad_tables = get_provenance_split(obj, aggerr.agg.cols, aggerr.keys) or []
    good_tables = get_provenance_split(obj, aggerr.agg.cols, obj.goodkeys[aggerr.agg.shortname]) or []

    (bad_tables, good_tables), full_table = reconcile_tables(bad_tables, good_tables)
    #_, full_table = reconcile_tables(bad_tables)

    # strip unnecessary columns
    user_cols = kwargs.get('cols', None)
    table, cols = strip_columns(table, aggerr, cols=user_cols)
    bad_tables = [strip_columns(t, aggerr, cols=user_cols)[0] for t in bad_tables]
    good_tables = [strip_columns(t, aggerr, cols=user_cols)[0] for t in good_tables]
    table = full_table

    truth = set(get_ground_truth(full_table))

    return full_table, bad_tables, good_tables, truth, aggerr, cols
 

def get_rules(full_table, bad_tables, good_tables, **kwargs):
    """
    Runs Scorpion on this dataset
    """
    klass = kwargs['klass']

    learner = klass(**kwargs)
    start = time.time()
    clusters = learner(full_table, bad_tables, good_tables, **kwargs)
    end = time.time()
    cost = end - start
    costs = learner.costs
    clusters = filter(bool, clusters)
    all_clusters = learner.all_clusters


    normalize = lambda arg: normalize_cluster_errors([c.clone() for c in arg])
    clusters, all_clusters = map(normalize, (clusters, all_clusters))
    best_clusters = filter_top_clusters(clusters, nstds=1)

    cols = kwargs['cols']
    
    thresh = compute_clusters_threshold(clusters)
    for c in clusters:
        c.isbest = (c.error >= thresh)
    merged = sorted(clusters_to_rules(clusters, full_table), key=lambda c: c.quality, reverse=True)
    #merged = [r.simplify() for r in merged]

    cost = cost - costs.get('merge_load_from_cache',(0,))[0] - costs.get('merge_cache_results', (0,))[0]
    costs['cost_merge'] = costs.get('cost_merge', 0) - (costs.get('merge_load_from_cache',(0,))[0] + costs.get('merge_cache_results', (0,))[0])
    costs['cost_total'] = cost
    return costs, merged, learner

def run_experiment(datasetidx, **kwargs):
    print kwargs
    ft,bts,gts, truth, aggerr, cols = get_parameters(datasetidx, **kwargs)

    params = {
        'epsilon':0.001,
        'tau':[0.01, 0.25],
        'lamb':0.5,
        'min_pts':3,
        'c' : 0.  
    }
    params.update(kwargs)
    params.update({
        'aggerr' : aggerr,
        'cols' : cols,
        'tablename' : datasetidx,
        'dataset' : datasetidx
        })

    costs, rules, learner = get_rules(ft, bts, gts, **params)
    ids = [get_row_ids(rule, ft) for rule in rules]

    learner.__dict__['compute_stats'] = lambda r: compute_stats(r, truth, len(ft))


    return costs, rules, ids, len(ft), learner


    f_stats = lambda r: compute_stats(r, truth, ft)

    bstats = map(f_stats, best)
    mstats = map(f_stats, merged)
    astats = map(f_stats, allr)
    return cost, costs, bstats, mstats, astats





