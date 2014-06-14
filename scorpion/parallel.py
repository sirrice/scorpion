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


def rules_to_clusters(rules, learner):
  clusters = []
  fill_in_rules(
    rules, 
    learner.full_table, 
    cols=learner.cols, 
    cont_dists=learner.cont_dists
  )

  for r in rules:
    c = Cluster.from_rule(r, learner.cols)

    if not c.inf_state:
      learner.influence_cluster(c)
    if not r.c_range or abs(r.c_range[0]) == float('inf'):
      c.c_range = list(learner.c_range)
    c.inf_func = learner.create_inf_func(c)
    clusters.append(c)
  return clusters




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
  parallel = params.get('parallel', False)
  print "executing in parallel?", parallel

  start = time.time()

  if parallel:
    clusters = []
    par2chq = Queue()
    ch2parq = Queue()
    args = (learner, aggerr, params, _logger, (par2chq, ch2parq))
    proc = Process(target=merger_process_f, args=args)
    proc.start()

    # loop
    for rules in learner(all_full_table, bad_tables, good_tables):
      if not rules: continue
      jsons = [r.to_json() for r in rules]
      _logger.debug("main\tsend to child proc %d rules" % len(jsons))
      par2chq.put(jsons)
    _logger.debug("main\tsend to child proc DONE")
    par2chq.put('done')
    par2chq.close()
    learner.update_status("waiting for merging step to finish")

    # wait for results from merger to come back
    jsons = ch2parq.get()
    rules = [SDRule.from_json(d, learner.full_table) for d in jsons]
    ch2parq.close()
    proc.join()

    clusters = rules_to_clusters(rules, learner)

  else:
    mparams = dict(params)
    mparams.update({
      'learner_hash': hash(learner),
      'learner' : learner,
      'partitions_complete': False
    })
    merger = StreamRangeMerger(**mparams)

    allrules = []
    for rules in learner(all_full_table, bad_tables, good_tables):
      allrules.extend(rules)
    clusters = rules_to_clusters(allrules, learner)
    pdb.set_trace()
    merger.add_clusters(clusters)
    learner.update_rules(
      aggerr.agg.shortname, 
      group_clusters(merger.best_so_far(), learner)
    )
    learner.update_status("waiting for merging step to finish")

    while merger.has_next_task():
      merger()
      learner.update_rules(
        aggerr.agg.shortname, 
        group_clusters(merger.best_so_far(), learner)
      )
    clusters = merger.best_so_far(True)

  costs['rules_get'] = time.time() - start

  clusters = list(clusters)

  print "found clusters"
  for c in clusters:
    print "\t", str(c)



  obj.update_status('clustering results')
  start = time.time()
  rules = group_clusters(clusters, learner)
  costs['rules_cluster'] = time.time() - start
  learner.update_rules(aggerr.agg.shortname, rules)

  
  # return the best rules first in the list
  start = time.time()
  rules.sort(key=lambda r: r.c_range[0])
  rules = [r.simplify(cdists=learner.cont_dists, ddists=learner.disc_dists) for r in rules]
  costs['rules_simplify'] = time.time() - start



  print "grouped rules"
  for rule in rules: 
    print '\t', str(rule)

  print "=== Costs ==="
  for key, cost in costs.iteritems():
    print "%.5f\t%s" % (cost, key)
  for key, (cost, count) in learner.stats.iteritems():
    print "%.5f\t%s\t%s" % (cost,key, count)
  
  return full_table, rules


def merger_process_f(learner, aggerr, params, _logger, (in_conn, out_conn)):
  # create and launch merger
  THRESHOLD = 0.01
  threshold = THRESHOLD * np.median([ef.value for ef in learner.bad_err_funcs])
  valid_cluster_f = lambda c: c.inf_state[0] and max(c.inf_state[0]) > threshold
  valid_cluster_f = lambda c: True

  params = dict(params)
  params.update({
    'learner_hash': hash(learner),
    'learner' : learner,
    'partitions_complete': False,
    'valid_cluster_f': valid_cluster_f 
  })
  merger = StreamRangeMerger(**params)

  # setup a merger
  merged = []
  jsons = []
  done = False
  while not done or merger.has_next_task():
    if not done:
      try:
        jsons = in_conn.get(False)
      except Empty:
        jsons = []
        pass

    if jsons == 'done': 
      done = True
      jsons = []

    try:
      rules = [SDRule.from_json(d, learner.full_table) for d in jsons]
      clusters = rules_to_clusters(rules, learner)

      if clusters:
        _logger.debug("merger\tadd_clusters %d" % len(clusters))
        added = merger.add_clusters(clusters)

        if added:
          start = time.time()
          merged = list(merger.best_so_far())
          rules = group_clusters(merged, learner)
          rules = [r.simplify(cdists=learner.cont_dists, ddists=learner.disc_dists) for r in rules]
          learner.update_rules(aggerr.agg.shortname, rules)
          _logger.debug("merger\tadded %d rules\t%.4f sec" % (len(rules), time.time()-start))

      if merger.has_next_task():
        _logger.debug("merger\tprocess tasks\t%d tasks left" % merger.ntasks)
        if merger():
          start = time.time()
          merged = list(merger.best_so_far())
          rules = group_clusters(merged, learner)
          rules = [r.simplify(cdists=learner.cont_dists, ddists=learner.disc_dists) for r in rules]
          learner.update_rules(aggerr.agg.shortname, rules)
          _logger.debug("merger\tupdated %d rules\t%.4f sec" % (len(rules), time.time()-start))
        else:
          _logger.debug("merger\tno improvements")

    except Exception as e:
      print "problem in merger process"
      print e
      import traceback
      traceback.print_exc()
      merged = []

  _logger.debug("merger\texited loop")


  merged = merger.best_so_far(True)
  _logger.debug("merger\tsending %d results back", len(merged))
  #for c in merged:
  #  c.rule.c_range = c.c_range
  #  c.rule.inf_state = c.inf_state
  #  _logger.debug("merger\tsending result\t%s", c)
  out_conn.put([c.rule.to_json() for c in merged])
  in_conn.close()
  out_conn.close()

  for key, (cost, count) in merger.stats.iteritems():
    print "%.5f\t%s\t%s" % (cost, key, count)
 
  print "child DONE"





def group_clusters(clusters, learner):
  """
  filter subsumed rules (clusters) and group them by
  those with similar influence on the user selected results

  Return
    list of rule objects sorted by c_range[0]
  """
  # find hierarchy relationships
  child2parent = get_hierarchies(clusters)

  non_children = []
  for c in clusters:
    if c not in child2parent:
      non_children.append(c)

  #non_children = filter_useless_clusters(non_children, learner)
  validf = lambda c: valid_number(c.c_range[0]) and valid_number(c.c_range[1])
  non_children = filter(validf, non_children)

  groups = []
  for key, group in group_by_inf_state(non_children, learner).iteritems():
    subgroups = group_by_tuple_ids(group)
    subgroup = filter(bool, map(merge_clauses, subgroups.values()))
    groups.append(subgroup)

  rules = filter(bool, map(group_to_rule, groups))
  rules = sort_rules(rules, learner)
  return rules

def get_hierarchies(clusters):
  """
  Return a child -> parent relationship mapping
  """
  clusters = list(clusters)
  child2parent = {}
  for idx, c1 in enumerate(clusters):
    for c2 in clusters[idx+1:]:
      if c1.contains(c2) and c1.inf_dominates(c2, 0):
        child2parent[c2] = c1
      elif c2.contains(c1) and c2.inf_dominates(c1):
        child2parent[c1] = c2
  return child2parent

def sort_rules(clusters, learner):
  """
  if cluster doesn't affect outliers enough (in absolute terms),
  then throw it away
  """
  THRESHOLD = 0.01
  mean_val = np.median([ef.value for ef in learner.bad_err_funcs])
  threshold = THRESHOLD * mean_val

  def f(c):
    infs = filter(lambda v: abs(v) != float('inf'), c.inf_state[0])
    if not infs:
      return -1e10
    return max(infs) - threshold
  h = lambda c: r_vol(c.c_range)

  rules = sorted(rules, key=f, reverse=True)
  return rules


  return filter(h, filter(f, clusters))

def merge_clauses(clusters):
  """
  assuming the clusters match the same input records, combine their clauses
  into a single cluster
  """
  if len(clusters) == 0: return None
  if len(clusters) == 1: 
    clusters[0].rule.c_range = clusters[0].c_range
    return clusters[0]

  conds = {}
  for c in clusters:
    for cond in c.rule.filter.conditions:
      key = c.rule.condToString(cond)
      conds[key] = cond

  conds = conds.values()
  mainc = clusters[0]
  rule = SDRule(mainc.rule.data, None, conds, None)
  rule.c_range = list(mainc.c_range)
  rule.quality = mainc.error
  c = Cluster.from_rule(rule, mainc.cols)
  c.c_range = list(mainc.c_range)
  c.inf_state = map(tuple, mainc.inf_state)
  c.inf_func = mainc.inf_func
  c.error = mainc.error
  return c



def group_to_rule(clusters):
  """
  pick a representative cluster from the arguments and 
  return a single one
  """
  if len(clusters) == 0: return None
  clusters = sorted(clusters, key=lambda c: r_vol(c.c_range), reverse=True)
  rule = clusters[0].rule
  rule.c_range = list(clusters[0].c_range)
  for c in clusters[1:]:
    rule.cluster_rules = []
    rule.cluster_rules.append(c.rule)
  return rule


def group_by_inf_state(clusters, learner):
  """
  super stupid check to group rules that have 
  identical influence states
  """
  mean_val = np.median([ef.value for ef in learner.bad_err_funcs])
  block = mean_val / 10.
  def trunc(v):
    if abs(v) == float('inf'):
      return 'inf'
    return int(float(v) / block)

  def f(c):
    bad_infs =  tuple(map(trunc, c.inf_state[0]))
    good_infs = tuple(map(trunc, c.inf_state[2]))
    return (bad_infs, good_infs)
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
