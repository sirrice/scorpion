import sys
import random
import time
import sys
import matplotlib
import numpy as np

from itertools import *
from collections import defaultdict

from scorpionsql.db import *
from scorpionsql.aggerror import *
from scorpion.arch import *
from scorpion.tests.gentestdata import *
from scorpion.util import reconcile_tables
from scorpion.sigmod import *
from scorpion.parallel import load_tables, serial_hybrid

matplotlib.use("Agg")


class ParamsCreator(object):
  def __init__(self, defaults):
    # default dictionary
    self.defaults = defaults

    # key -> list of options
    self.options = {}

  def add_options(self, name, opts):
    self.options[name] = opts

  @property
  def params(self):
    optnames = self.options.keys()
    if not optnames:
      yield dict(self.defaults)
      return

    for optvals in product(*self.options.values()):
      d = dict(self.defaults)
      d.update(dict(zip(optnames, optvals)))
      yield d



def get_row_ids(r, table):
  """
  args:
    r: rule
    table: orange table
  """
  if r is None: r = lambda t: t
  return set([row['id'].value for row in r(table)])

def get_tuples_in_bounds(db ,tablename, bounds, additional_where=''):
  """
  Get tuples for synthetic dataset
  """
  with db.begin() as conn:
    where = ['%f <= a_%d and a_%d <= %f' % (minv, i, i, maxv) for i, (minv, maxv) in enumerate(map(tuple, bounds))]
    if additional_where:
        where.append(additional_where)
    where = ' and '.join(where)
    q = """ select * from %s where %s""" % (tablename, where)
    return [map(float, row) for row in conn.execute(q)]


def get_ids_in_bounds(db ,tablename, bounds, additional_where=''):
  """
  Given a synthetic tablename and bounds of the form [ [min1, max1], ... ]
  Return the ids of the records that match
  """
  with db.begin() as conn:
    where = ['%f <= a_%d and a_%d <= %f' % (minv, i, i, maxv) for i, (minv, maxv) in enumerate(map(tuple, bounds))]
    if additional_where:
      where.append(additional_where)
    where = ' and '.join(where)
    q = """ select id from %s where %s""" % (tablename, where)
    return [int(row[0]) for row in conn.execute(q)]


def get_ground_truth(db, datasetidx, c):
  """
  The naive experiments are supposed to also log the ground truth.
  Retrieve them from the experiment stats table
  """
  with db.begin() as conn:
    q = """select ids from stats 
    where klass = 'Naive' and dataset = %s and c = %s 
    order by expid desc"""
    for row in conn.execute(q, datasetidx, c).fetchall():
      idstr = row[0]
      return map(int, map(float, idstr.split(',')))
    return None




def compute_stats(found_ids, bad_tuple_ids, table_size):
  """
  given sets of IDs, compute f1/prec/rec values
  """
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


def get_parameters(datasetidx, **kwargs):
    name = datasetnames[datasetidx]
    outname = name

    test_data = get_test_data(name)
    dbname, sql, badresults, goodresults, errtype, get_ground_truth, tablename = test_data
    obj, table = create_sharedobj(*test_data[:-2])
    aggerr = obj.errors[0]
    obj.db = connect(dbname)
    return obj, aggerr, get_ground_truth

def run_experiment(datasetidx, **kwargs):
    print kwargs
    obj, aggerr, get_ground_truth = get_parameters(datasetidx, **kwargs)

    params = {
      'epsilon':0.001,
      'tau':[0.01, 0.25],
      'lamb':0.5,
      'min_pts':3,
      'c' : 0.  
    }
    params.update(kwargs)
    params.update({
      #'aggerr' : aggerr,
      'tablename' : datasetidx,
      'dataset' : datasetidx
    })
    cs = params['cs']

    #costs['cost_merge'] = costs.get('cost_merge', 0) - (costs.get('merge_load_from_cache',(0,))[0] + costs.get('merge_cache_results', (0,))[0])
    #costs['cost_total'] = cost
    learner, table, rules, c2rules = serial_hybrid(obj, aggerr, **params)
    truth = set(get_ground_truth(table))
    ft = table
    costs = defaultdict(lambda: -1)
    costs.update(learner.stats)

    ids = [get_row_ids(rule, ft) for rule in rules]

    learner.__dict__['compute_stats'] = lambda r: compute_stats(r, truth, len(ft))


    return costs, rules, c2rules, ids, len(ft), learner



def run(db, datasetidx, pp=None, params={}, **kwargs):
  """
  args:
    datasetidx: either idx or the tablename
  """

  params = dict(params)
  params.update(kwargs)
  expid = nextexpid(db)
  try:
    if 'cs' in params:
      run_params_cs(db, datasetidx, expid, pp=pp, **params)
    else:
      run_params(db, datasetidx, expid, pp=pp, **params)

    complete(db, expid)
  except:
    import traceback
    traceback.print_exc()


def rule2vals(expid, datasetidx, learner, rule, truth, c, params):
  pkeys = ['klassname', 'cols', 'epsilon', 'c', 'lambda']
  table = learner.full_table
  table_size = len(table)
  ids = get_row_ids(rule, table)
  ids_str = ','.join(map(str, ids))

  if hasattr(rule, 'inf_func'):
    rule.quality = rule.inf_func(c)

  vals = [expid, str(datasetidx), params.get('notes', '')]
  vals.extend([params.get(key, None) for key in pkeys])
  vals[-2] = c
  vals.extend((
    learner.stats['cost_total'][0],
    rule.quality, 
    rule.isbest, 
    str(rule), 
    ids_str
  ))
  return vals


def run_params_cs(db, datasetidx, expid, pp=None, **params):
  """
  Runs scorpion and stores results.
  doesn't bother computing statistics
  """
  cs = params['cs']

  params['klassname'] = kname(params['klass'])
  costs, rules, c2rules, all_ids, table_size, learner = run_experiment(datasetidx, **params)

  # c2rules: mapping from c value to list of rules
  # all_ids: the ids that match each rules in rules

  for c in cs:
    if params['klassname'] == 'Naive':
      rules = rules[:1]
    truth = []

    for rule in [r for r in rules if r_contains(r.c_range, [c,c])]:
      vals = rule2vals(expid, datasetidx, learner, rule, truth, c, params)
      row_id = save_result(db, vals, costs)
      print mkfmt(vals[:-1]) % tuple(vals[:-1])

    if params['klassname'] == 'Naive':
      for timein, rule in learner.checkpoints_per_c[c]:
        vals = rule2vals(expid, datasetidx, learner, rule, truth, c, params)
        vals[8] = timein
        vals[-3] = False
        row_id = save_checkpoint(db, vals)
        print mkfmt(vals[:-1]) % tuple(vals[:-1])



def run_params(db, datasetidx, expid, pp=None, **params):
  c = params['c']

  params['klassname'] = kname(params['klass'])
  costs, rules, c2rules, all_ids, table_size, learner = run_experiment(datasetidx, **params)
  if params['klassname'] == 'Naive':
    rules = rules[:1]
  truth = []

  for rule in rules:
    vals = rule2vals(expid, datasetidx, learner, rule, truth, c, params)
    row_id = save_result(db, vals, costs)
    print mkfmt(vals[:-1]) % tuple(vals[:-1])

  if params['klassname'] == 'Naive':
    for timein, rule in learner.checkpoints_per_c[c]:
      vals = rule2vals(expid, datasetidx, learner, rule, truth, c, params)
      vals[8] = timein
      vals[-3] = False
      row_id = save_checkpoint(db, vals)
      print mkfmt(vals[:-1]) % tuple(vals[:-1])



def save_result(db, stat, costs):
  with db.begin() as conn:
    q = """insert into results(
      expid, dataset, notes, klass, cols, epsilon, c,     
      lambda, cost, score, 
      isbest, rule, ids) values(%s) returning id
    """ % (','.join(['%s']*len(stat)))
    sid = conn.execute(q, *stat).fetchone()[0]

    q = """insert into costs(sid, name, cost) values(%s,%s,%s)"""
    for name, cost in costs.items():
        if isinstance(cost, list) or isinstance(cost, tuple):
            cost = cost[0]
        conn.execute(q, sid, name, cost)
    return sid

def save_checkpoint(db, stat):
  with db.begin() as conn:
    q = """insert into results(
      expid, dataset, notes, klass, cols, epsilon, c,     
      lambda, cost, score, 
      isbest, rule, ischeckpoint, ids) values(%s) returning id""" % (','.join(['%s']*len(stat)))
    sid = conn.execute(q, *stat).fetchone()[0]
    return sid


def complete(db, expid):
  with db.begin() as conn:
    q = """update results set completed = TRUE where expid = %s"""
    conn.execute(q, expid)



def nextexpid(db):
  with db.begin() as conn:
    q = """select max(expid)+1 from results"""
    row = conn.execute(q).fetchone()
    expid = row[0]
    if expid == None:
        return 0
    return expid





