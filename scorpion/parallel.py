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
import numpy as np

from sklearn.cluster import KMeans
from itertools import chain, groupby
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process, Queue, Pool, Pipe
from Queue import Empty

from scorpionsql.db import *
from scorpionsql.aggerror import *
import scorpionsql.errfunc as errfunc

from arch import *
from util import *
from sigmod import *
from settings import ID_VAR
from sigmod.streamrangemerger import *


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

def pick(l, key):
  return [v[key] for v in l]

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
    table, rules, top_k_rules = serial_hybrid(sharedobj, aggerr, **kwargs)
  except:
    traceback.print_exc()
    rules = []
    table = None

  sharedobj.merged_tables[label] = table
  sharedobj.rules[label] = rules
  sharedobj.top_k_rules[label] = top_k_rules
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
    all_json_pairs = []
    for pairs in learner(all_full_table, bad_tables, good_tables):
      if not pairs: continue

      json_pairs = [(r.to_json(), idxkey) for r, idxkey in pairs]
      _logger.debug("main\tsend to child proc %d rules" % len(json_pairs))
      try:
				par2chq.put(json_pairs, False)
				_logger.debug("main\tsent")
      except Queue.Full:
				all_json_pairs.extend(json_pairs)
				_logger.debug("main\itq full")

    _logger.debug("main\tsend last batch to merger %d rules", len(all_json_pairs))
    par2chq.put(all_json_pairs)
    _logger.debug("main\tsend to child proc DONE")
    par2chq.put('done')
    par2chq.close()
    learner.update_status("waiting for merging step to finish")

    # wait for results from merger to come back
    json_rules = ch2parq.get()
    rules = [SDRule.from_json(d, learner.full_table) for d in json_rules]

    top_k_json_rules = ch2parq.get()
    top_k_json_rules = { 
      key: pick(g, 1) 
      for key, g in groupby(top_k_json_rules, key=lambda p: p[0])
    }

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
    merger = PartitionedStreamRangeMerger(**mparams)

    allpairs = []
    for pairs in learner(all_full_table, bad_tables, good_tables):
      allpairs.extend(pairs)

    rules, keyidxs = zip(*allpairs)
    clusters = rules_to_clusters(rules, learner)

    pairs = zip(clusters, keyidxs)
    for key, g in groupby(pairs, key=lambda p: p[1]):
      clusters = pick(g, 0)
      merger.add_clusters(clusters, idx=0, partitionkey=key)


    while merger.has_next_task():
			if merger():
				learner.update_rules(
					aggerr.agg.shortname, 
					group_clusters(merger.best_so_far(), learner)
				)
    clusters = merger.best_so_far(True)

    c_vals = CheapFrontier.compute_normalized_buckets(25, clusters)
    c_vals = c_vals * r_vol(learner.c_range) + learner.c_range[0]
    top_k_json_rules = defaultdict(list)
    for c_val in c_vals:
      for c in merger.best_at_c(c_val):
        rule = c.rule.clone()
        rule.quality = c.inf_func(c_val)
        top_k_json_rules[c_val].append(rule.to_json())
    
    merger.close()

  costs['rules_get'] = time.time() - start


  top_k_rules = defaultdict(list)
  for c_val, json_rules in top_k_json_rules.iteritems():
    for d in json_rules:
      rule = SDRule.from_json(d, data=full_table)
      top_k_rules[c_val].append(rule)

  clusters = list(clusters)

  print "found clusters"
  for c in clusters:
    print "\t", str(c)


  obj.update_status('clustering results')
  start = time.time()
  rules = group_clusters(clusters, learner)
  costs['rules_cluster'] = time.time() - start
  learner.update_rules(aggerr.agg.shortname, rules)

  rules.sort(key=lambda r: r.c_range[0])
  rules = [r.simplify(cdists=learner.cont_dists, ddists=learner.disc_dists) for r in rules]

  print "grouped rules"
  for rule in rules: 
    print '\t', str(rule)

  print "=== Costs ==="
  for key, cost in costs.iteritems():
    print "%.5f\t%s" % (cost, key)
  for key, (cost, count) in learner.stats.iteritems():
    print "%.5f\t%s\t%s" % (cost,key, count)

  return full_table, rules, top_k_rules



def merger_process_f(learner, aggerr, params, _logger, (in_conn, out_conn)):
  valid_cluster_f = lambda c: True
  status = Status(learner.obj.status.reqid)

  def update_status(msg):
    start = time.time()
    merged = list(merger.best_so_far(True))
    rules = group_clusters(merged, learner)
    simplify = lambda r: r.simplify(cdists=learner.cont_dists, ddists=learner.disc_dists) 
    rules = map(simplify, rules)
    status.update_rules(aggerr.agg.shortname, rules)
    _logger.debug("merger\t%s %d rules\t%.4f sec", msg, len(rules), time.time()-start)

  params = dict(params)
  params.update({
    'learner_hash': hash(learner),
    'learner' : learner,
    'partitions_complete': False,
    'valid_cluster_f': valid_cluster_f 
  })
  merger = PartitionedStreamRangeMerger(**params)

  # setup a merger
  merged = []
  json_pairs = []
  done = False
  while not done or merger.has_next_task():
    if not done:
      try:
        json_pairs = in_conn.get(False)
      except Empty:
        json_pairs = []
        pass

    if json_pairs == 'done': 
      done = True
      json_pairs = []

    try:
      if json_pairs:
        pairs = [(SDRule.from_json(d, learner.full_table), keyidx) for d, keyidx in json_pairs]
        rules, idxkeys = tuple(zip(*pairs))
        clusters = rules_to_clusters(rules, learner)
        pairs = zip(clusters, idxkeys)

        for key, g in groupby(pairs, key=lambda p: p[1]):
          added = merger.add_clusters(pick(g, 0), idx=0, partitionkey=key)

        if added:
          update_status("added")

      if merger.has_next_task():
        _logger.debug("merger\tprocess tasks\t%d tasks left" % merger.ntasks)
        if merger():
          update_status("updated")
        else:
          _logger.debug("merger\tno improvements\t%d tasks left", merger.ntasks)

    except Exception as e:
      print "problem in merger process"
      print e
      import traceback
      traceback.print_exc()
      merged = []

  _logger.debug("merger\texited loop")


  best = merger.best_so_far(True)
  best_json_rules = [c.rule.to_json() for c in best]

  c_vals = CheapFrontier.compute_normalized_buckets(25, best)
  c_vals = c_vals * r_vol(learner.c_range) + learner.c_range[0]
  top_k_json_rules = []
  for c_val in c_vals:
    for c in merger.best_at_c(c_val):
      rule = c.rule.clone()
      rule.quality = c.inf_func(c_val)
      top_k_json_rules.append((c_val, rule.to_json()))

  merger.close()
  _logger.debug("merger\tsending %d results back", len(merged))
  out_conn.put(best_json_rules)
  out_conn.put(top_k_json_rules)
  in_conn.close()
  out_conn.close()
  status.close()

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
  validf = lambda c: valid_number(c.c_range[0]) and valid_number(c.c_range[1])

  non_children = []
  for c in clusters:
    if c in child2parent:
      _logger.debug("groupclust\trm child cluster\t%s", c)
    elif not validf(c):
      _logger.debug("groupclust\tc_range invalid \t%s", c)
    else:
      non_children.append(c)

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
      if c1.contains(c2) and c1.inf_dominates(c2, .01, c_range=c2.c_range):
        child2parent[c2] = c1
      elif c2.contains(c1) and c2.inf_dominates(c1, .01, c_range=c1.c_range):
        child2parent[c1] = c2
  return child2parent

def sort_rules(rules, learner):
  """
  if cluster doesn't affect outliers enough (in absolute terms),
  then throw it away
  """
  THRESHOLD = 0.01
  mean_val = np.median([ef.value for ef in learner.bad_err_funcs])
  threshold = THRESHOLD * mean_val

  def f(c):
    infs = None
    if c.inf_state:
      infs = filter(lambda v: abs(v) != float('inf'), c.inf_state[0])
    if not infs:
      return -1e10
    return max(infs) - threshold
  h = lambda c: r_vol(c.c_range)

  rules = sorted(rules, key=f, reverse=True)
  return rules

def merge_clauses(clusters):
  """
  assuming the clusters match the same input records, combine their clauses
  into a single cluster
  """
  if len(clusters) == 0: 
    _logger.debug("groupclust\tmerge_clauses\t%d clusters", len(clusters))
    return None
  if len(clusters) == 1: 
    clusters[0].rule.c_range = clusters[0].c_range
    return clusters[0]


  mainc = clusters[0]
  domain = mainc.rule.data.domain
  conds = {}
  for c in clusters:
    for d in c.rule.cond_dicts:
      col = d['col']
      if col not in conds:
        conds[col] = d
        continue

      if d['type'] == 'num':
        conds[col]['vals'] = r_intersect(conds[col]['vals'], d['vals'])
      else:
        conds[col]['vals'] = set(conds[col]['vals']).intersection(d['vals'])

  j = mainc.rule.to_json()
  j['clauses'] = conds.values()
  rule = SDRule.from_json(j, data=mainc.rule.data)
  c = Cluster.from_rule(rule, mainc.cols)
  c.inf_func = mainc.inf_func
  c.error = mainc.error
  return c


def group_to_rule(clusters):
  """
  pick a representative cluster from the arguments and 
  return a single one
  """
  _logger.debug("groupclust\tgroup->rule\t%d clusters", len(clusters))
  if len(clusters) == 0: return None
  for c in clusters:
    _logger.debug("groupclust\t\t%s", c)

  clusters = sorted(clusters, key=lambda c: r_vol(c.c_range), reverse=True)
  rule = clusters[0].rule
  rule.c_range = list(clusters[0].c_range)
  rule.cluster_rules = []
  for c in clusters[1:]:
    rule.cluster_rules.append(c.rule)

  return rule

def group_by_inf_state(clusters, learner):
  """
  super stupid check to group rules that have 
  identical influence states
  """
  errtype = learner.bad_err_funcs[0].errtype.errtype
  get_errval = lambda ef: ef.value
  if errtype == ErrTypes.EQUALTO:
    get_errval = lambda ef: ef.errtype(ef.value, ef.errtype.erreq)
  efs = np.array(map(get_errval, learner.bad_err_funcs))
  blocks = efs / 30.
  def trunc((idx,v)):
    if abs(v) == float('inf'):
      return 'inf'
    return int(float(v) / blocks[idx])

  def f(c):
    bad_infs =  tuple(map(trunc, enumerate(c.inf_state[0])))
    good_infs = tuple(c.inf_state[2])
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
