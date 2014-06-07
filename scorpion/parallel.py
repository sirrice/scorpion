#
# This file used to be a parallel implementation of scorpion, 
# now it is the bare bones necessary code to run scorpion 
# single threaded
#

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
from multiprocessing import Process, Queue, Pool, Pipe
from Queue import Empty

from db import *
from aggerror import *
from arch import *
from util import *
from sigmod import *
from settings import ID_VAR
from sigmod.streamrangemerger import StreamRangeMerger


_logger = get_logger()

DEFAULT_PARAMS = {
  'c_range': [0.05, 1],
  'l' : 0.6,
  'msethreshold': 0.01,
  'max_wait':5,
  'granularity': 50,
  'use_mtuples': False,#True,
  'DEBUG': False,

  # the following are for BDT
  'epsilon': 0.0015,
  'min_improvement': 0.01,
  'tau': [0.08, 0.5],
  'p': 0.7
}

def parallel_debug(sharedobj, **kwargs):
  for aggerr in sharedobj.errors:
    runner(sharedobj, aggerr, **kwargs)
  

def runner(sharedobj, aggerr, **kwargs):
  random.seed(2)
  sql = sharedobj.sql
  badresults = aggerr.keys
  label = aggerr.agg.shortname
  if not badresults:
      return

  try:
    table, rules = serial_hybrid(sharedobj, aggerr, **kwargs)
  except:
    traceback.print_exc()
    rules = []
    table = None

  sharedobj.merged_tables[label] = table
  sharedobj.rules[label] = rules
  return table, rules



def load_tables(obj, aggerr, **kwargs):
  db = connect(obj.dbname)
  obj.db = db
  obj.update_status('loading inputs into memory')

  # get the input records!
  start = time.time()
  all_keys = list(chain(aggerr.keys, obj.goodkeys[aggerr.agg.shortname]))
  all_tables = get_provenance_split(obj, aggerr.agg.cols, all_keys)
  bad_tables = all_tables[:len(aggerr.keys)]
  good_tables = all_tables[len(aggerr.keys):]

  _logger.debug("bad table counts:  %s" % ', '.join(map(str, map(len, bad_tables))))
  _logger.debug("good table counts: %s" % ', '.join(map(str, map(len, good_tables))))
  print "agg error %s \t %s" % (aggerr.agg, aggerr.errtype)

  rules = []


  obj.update_status('preparing inputs')
  start = time.time()

  # "id" column is special so we need to deal with it specially
  cols = valid_table_cols(bad_tables[0], aggerr.agg.cols, kwargs)
  all_cols = cols + aggerr.agg.cols        
  if 'id' not in all_cols:
    all_cols.append('id')
  torm = [attr.name for attr in bad_tables[0].domain if attr.name not in all_cols]

  bad_tables = [rm_attr_from_domain(t, torm) for t in bad_tables]
  good_tables = [rm_attr_from_domain(t, torm) for t in good_tables]
  all_full_table = union_tables(bad_tables, good_tables)
  full_table = union_tables(bad_tables)

  print "total # rows:\t%d" % len(full_table)
  _logger.debug("valid cols: %s" % cols)


  # make sure aggerr keys and tables are consistent one last time
  if len(bad_tables) != len(aggerr.keys):
    pdb.set_trace()
    raise RuntimeError("#badtables (%d) != #aggerr keys (%d)" % (len(bad_tables), len(aggerr.keys)))

  return cols, bad_tables, good_tables, full_table, all_full_table




def serial_hybrid(obj, aggerr, **kwargs):
  costs = {}
  cols, bad_tables, good_tables, full_table, all_full_table = load_tables(obj, aggerr, **kwargs)

  params = dict(DEFAULT_PARAMS)
  params.update({
    'obj': obj,
    'aggerr':aggerr,
    'cols':cols,
    'c': obj.c
  })
  params.update(dict(kwargs))

  if aggerr.agg.func.__class__ in (errfunc.SumErrFunc, errfunc.CountErrFunc):
    klass = MR 
  else:
    klass = BDT
  if False:
    klass = SVM

  learner = klass(**params)
  learner.setup_tables(all_full_table, bad_tables, good_tables)


  if True:
    clusters = []
    par2chq = Queue()
    ch2parq = Queue()
    proc = Process(target=merger_process_f, args=(learner, params, (par2chq, ch2parq)))
    proc.start()

    # loop
    for batch in learner(all_full_table, bad_tables, good_tables):
      batch_dicts = [c.to_dict() for c in batch]
      print "send to child proc %d rules" % len(batch_dicts)
      par2chq.put(batch_dicts)
    print "send to child proc DONE"
    par2chq.put('done')
    par2chq.close()

    while True:
      try:
        batch_dicts = ch2parq.get()
        if batch_dicts == 'done':
          break
        print "got %d from merger proc" % len(batch_dicts)
        clusters.extend(map(Cluster.from_dict, batch_dicts))
      except IOError:
        print "got io error"
        continue
      except EOFError:
        break
    ch2parq.close()
    proc.join()

    for c in clusters:
      c.to_rule(full_table,learner.cont_dists, learner.disc_dists)
      c.inf_func = learner.create_inf_func(c)

    clusters, _ = Frontier(learner.c_range)(clusters)

  else:
    mparams = dict(params)
    mparams.update({
      'learner_hash': hash(learner),
      'learner' : learner,
      'partitions_complete': False
    })
    merger = StreamRangeMerger(**mparams)

    start = time.time()
    clusters = []
    for batch in learner(all_full_table, bad_tables, good_tables):
      clusters.extend(merger(batch))
    costs['rules_get'] = time.time() - start

  clusters = list(clusters)

  obj.update_status('clustering results')
  start = time.time()
  rules = group_clusters(clusters, learner)
  costs['rules_cluster'] = time.time() - start

  
  # return the best rules first in the list
  start = time.time()
  rules.sort(key=lambda r: r.c_range[0])
  rules = [r.simplify(all_full_table) for r in rules]
  costs['rules_simplify'] = time.time() - start



  print "found rules"
  for rule in rules:
    print "%.5f\t%s" % (rule.quality, rule)

  print "=== Costs ==="
  for key, cost in costs.iteritems():
    print "%.5f\t%s" % (cost, key)
  
  return full_table, rules


def merger_process_f(learner, params, (in_conn, out_conn)):
  # create and launch merger
  params = dict(params)
  params.update({
    'learner_hash': hash(learner),
    'learner' : learner,
    'partitions_complete': False
  })
  merger = StreamRangeMerger(**params)
  print "merger process done with setup!"

  # setup a merger
  done = False
  while not done:
    try:
      cluster_dicts = in_conn.get()
      if cluster_dicts == 'done': 
        done = True
        break

      # empty the queue
      while True:
        try:
          data = in_conn.get(False)
        except Empty:
          break
        if data == 'done':
          done = True
          break
        cluster_dicts.extend(data)

    except EOFError:
      break
    except IOError:
      continue

    try:
      clusters = []
      for d in cluster_dicts:
        c = Cluster.from_dict(d)
        c.to_rule(learner.full_table, learner.cont_dists, learner.disc_dists)
        learner.influence_cluster(c)
        c.c_range = list(merger.c_range)
        c.inf_func = learner.create_inf_func(c)
        clusters.append(c)

      merged = merger(clusters)
    except Exception as e:
      import traceback
      traceback.print_exc()
      merged = []

    merged_json = [c.to_dict() for c in merged]
    print "send merged json %d" % len(merged_json)
    out_conn.put(merged_json)

  print "child DONE"
  out_conn.put('done')
  in_conn.close()
  out_conn.close()



def group_clusters(clusters, learner):
  """
  filter subsumed rules (clusters) and group them by
  those with similar influence on the user selected results
  """
  # find hierarchy relationships
  child2parent = get_hierarchies(clusters)

  non_children = []
  for c in clusters:
    if c not in child2parent:
      non_children.append(c)

  non_children = filter_useless_clusters(non_children, learner)
  validf = lambda c: valid_number(c.c_range[0]) and valid_number(c.c_range[1])
  non_children = filter(validf, non_children)

  groups = []
  for key, group in group_by_inf_state(non_children).iteritems():
    subgroups = group_by_tuple_ids(group)
    subgroup = filter(bool, map(merge_clauses, subgroups.values()))
    groups.append(subgroup)

  return filter(bool, map(group_to_rule, groups))

def get_hierarchies(clusters):
  """
  Return a child -> parent relationship mapping
  """
  child2parent = {}
  for idx, c1 in enumerate(clusters):
    for c2 in clusters[idx+1:]:
      if c1.contains(c2) and c1.inf_dominates(c2, 0):
        child2parent[c2] = c1
      elif c2.contains(c1) and c2.inf_dominates(c1):
        child2parent[c1] = c2
  return child2parent

def filter_useless_clusters(clusters, learner):
  """
  if cluster doesn't affect outliers enough (in absolute terms),
  then throw it away
  """
  THRESHOLD = 0.01
  mean_val = np.mean([ef.value for ef in learner.bad_err_funcs])
  threshold = THRESHOLD * mean_val

  f = lambda c: c.inf_state[0] and max(c.inf_state[0]) > threshold
  return filter(f, clusters)

def merge_clauses(clusters):
  """
  assuming the clusters match the same input records, combine their clauses
  into a single cluster
  """
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
  """
  pick a representative cluster from the arguments and 
  return a single one
  """
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
  """
  group the clusters by the set of input tuples they match
  """
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
