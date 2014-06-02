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
from settings import ID_VAR



_logger = get_logger()


def parallel_debug(sharedobj, **kwargs):
    for aggerr in sharedobj.errors:
      parallel_runner(sharedobj, aggerr, **kwargs)
    create_clauses(sharedobj)
    

def parallel_runner(sharedobj, aggerr, **kwargs):
    random.seed(2)
    sql = sharedobj.sql
    badresults = aggerr.keys
    label = aggerr.agg.shortname
    if not badresults:
        return

    cost, ncalls, table, rules = serial_hybrid(sharedobj, aggerr, **kwargs)

    sharedobj.merged_tables[label] = table
    rules = zip(rules, reversed(range(len(rules))))
    sharedobj.rules[label] = rules
    return table, rules



def serial_hybrid(obj, aggerr, **kwargs):
    costs = {}
    db = connect(obj.dbname)
    obj.db = db
    obj.update_status('loading inputs into memory')


    # get the input records!
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
        obj.update_status('preparing inputs')
        full_start = time.time()
        start = time.time()

        # "id" column is special so we need to deal with it
        # specially
        cols = valid_table_cols(bad_tables[0], aggerr.agg.cols, kwargs)
        all_cols = cols + aggerr.agg.cols        
        if 'id' not in all_cols:
          all_cols.append('id')
        torm = [attr.name for attr in bad_tables[0].domain if attr.name not in all_cols]
        _logger.debug("valid cols: %s" % cols)

        bad_tables = [rm_attr_from_domain(t, torm) for t in bad_tables]
        good_tables = [rm_attr_from_domain(t, torm) for t in good_tables]
        all_full_table = union_tables(bad_tables, good_tables)
        full_table = union_tables(bad_tables)
        print "total # rows:\t%d" % len(full_table)

        costs['data_setup'] = time.time() - start


        # make sure aggerr keys and tables are consistent one last time
        if len(bad_tables) != len(aggerr.keys):
          pdb.set_trace()
          raise RuntimeError("#badtables (%d) != #aggerr keys (%d)" % (len(bad_tables), len(aggerr.keys)))


        params = {
          'obj': obj,
          'aggerr':aggerr,
          'cols':cols,
          'c': obj.c,
          'c_range': [0.05, 1],
          'l' : 0.6,
          'msethreshold': 0.01,
          'max_wait':5,
          'granularity': 50,
          'use_mtuples': False,#True,
          'DEBUG': False
        }

        params.update(dict(kwargs))

        if aggerr.agg.func.__class__ in (errfunc.SumErrFunc, errfunc.CountErrFunc):
          klass = MR 
          params['c'] = params.get('c', .15)

        else:
          klass = BDT
          params.update({
            'epsilon': 0.0015,
            'min_improvement': 0.01,
            'tau': [0.08, 0.5],
            'p': 0.7
            })
          params['c'] = params.get('c', .3)

        if False:
          klass = SVM

        start = time.time()
        hybrid = klass(**params)
        clusters = hybrid(all_full_table, bad_tables, good_tables)
        costs['rules_get'] = time.time() - start

        obj.update_status('clustering results')
        print "nclusters: %d" % len(clusters)

        rules = group_clusters(clusters, hybrid)
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



def group_clusters(clusters, learner):
  # find hierarchy relationships
  # find equivalent rules via records matched
  # find equivalent rules via inf_state
  child2parent = get_hierarchies(clusters)

  non_children = []
  for c in clusters:
    if c in child2parent:
      pass
    non_children.append(c)

  non_children = filter_useless_clusters(non_children, learner)

  groups = []
  for key, group in group_by_inf_state(non_children).iteritems():
    subgroups = group_by_tuple_ids(group)
    subgroup = filter(bool, map(merge_clauses, subgroups.values()))
    groups.append(subgroup)

  return filter(bool, map(group_to_rule, groups))

def get_hierarchies(clusters):
  child2parent = {}
  for idx, c1 in enumerate(clusters):
    for c2 in clusters[idx+1:]:
      if c1.contains(c2):
        child2parent[c2] = c1
      elif c2.contains(c1):
        child2parent[c1] = c2
  return child2parent

def filter_useless_clusters(clusters, learner):
  THRESHOLD = 0.01
  mean_val = np.mean([ef.value for ef in learner.bad_err_funcs])
  threshold = THRESHOLD * mean_val

  f = lambda c: c.inf_state[0] and max(c.inf_state[0]) > threshold
  return filter(f, clusters)

def merge_clauses(clusters):
  if len(clusters) == 0: return None
  if len(clusters) == 1: return clusters[0]
  conds = {}
  for c in clusters:
    for cond in c.rule.filter.conditions:
      key = c.rule.condToString(cond)
      conds[key] = cond
  conds = conds.values()
  rule = SDRule(clusters[0].rule.data, None, conds, None)
  return Cluster.from_rule(rule, clusters[0].cols)


def group_to_rule(clusters):
  if len(clusters) == 0: return None
  rule = max(clusters, key=lambda c: r_vol(c.c_range)).rule
  for c in clusters[1:]:
    rule.cluster_rules = []
    rule.cluster_rules.append(c.rule)
  return rule


def group_by_inf_state(clusters):
  """
  super stupid check to group rules that have 
  identical influence states
  """
  f = lambda cluster: tuple(map(tuple, cluster.inf_state))
  return group_by_features(clusters, f)

def group_by_tuple_ids(clusters):
  def f(cluster):
    ids = [int(row['id'].value) for row in cluster.rule.examples]
    ids.sort()
    return hash(tuple(ids))
  return group_by_features(clusters, f)

def group_by_features(clusters, f):
  groups = defaultdict(list)
  for c in clusters:
    groups[f(c)].append(c)

  return groups
