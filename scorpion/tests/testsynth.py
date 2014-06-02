#
# For sigmod SYNTH Tests
# 1) compute synth for 2/3/4-d with medium and high outlier boxes
# 2) compute the two c values and store it
# 3) compute and store the boxes
# 4) execute suite of tests for each dataset
#
import pdb
import sys
import math
import json
import random
import numpy as np

from sqlalchemy import create_engine
from scorpion.util import get_logger
from scorpion.misc.gensinglecluster import gen_points, in_box


from common import *
from setup_experiments import init_db

random.seed(0)

format_list = lambda fmt, l: [fmt % (isinstance(p, list) and tuple(p) or p) for p in l]


def generate_datasets(db, ndim, uo=30, npts=2000):
    random.seed(0)
    """
    generates the 2/3/4-d datasets and stores them in db: sigmod
    """
    mid_box, high_box, schema, generator = gen_points(npts, ndim, ndim, 0.25, 10, 10, uo, 10)
    pts = [pt for pt in generator]
    c_mid, c_high = compute_c_values(pts, mid_box, high_box)

    # store boundaries
    tablename = 'data_%d_%d' % (ndim, int(uo))
    id = add_config(db, tablename, 2000, ndim, ndim, 0.25, 10, 10, uo, 10, mid_box, high_box, c_high)

    # save the damn thing
    save_pts(db, tablename, pts)
    

def setup(db):
    try:
        with db.begin() as conn:
            conn.execute("""
            create table synth (
                id serial,
                tablename varchar(128),
                npts int,
                ndim int,
                kdim int,
                volperc float,
                uh float,
                sh float,
                uo float,
                so float,
                mid text,  -- json encoded boundary for mid level
                high text,  -- json encoded
                c float -- threshold for picking  high over mid
            )
            """)
    except:
        import traceback
        traceback.print_exc()
        pass

def save_pts(db, tablename, pts):
    ndims = len(pts[0]) - 2
    with db.begin() as conn:
        q = """
        create table %s (
        id serial,
        %s,
        g float,
        v float)""" % (tablename, ','.join(format_list('a%d float', xrange(ndims))))
        conn.execute(q)

        pts_str = []
        for pt in pts:
            s = '(%s)' % ','.join(['%f']*(ndims+2)) % tuple(pt)
            pts_str.append(s)


        bigq = """insert into %s (%s) values
           %s""" % (tablename, ','.join(format_list('a%d', xrange(ndims))+['g','v']), ','.join(pts_str))
        conn.execute(bigq)


def add_config(db, *args):
    args = list(args)
    with db.begin() as conn:
        q = """
        insert into synth (tablename, npts, ndim, kdim, volperc, uh, sh, uo, so, mid, high, c)
        values (%s) returning id""" % ((',%s' * 12)[1:])
        args[-3] = json.dumps(args[-3])
        args[-2] = json.dumps(args[-2])
        id = conn.execute(q, *args ).fetchone()[0]
        return id

def all_configs(db):
    with db.begin() as conn:
        q = """ select * from synth"""
        for config in conn.execute(q).fetchall():
            config = list(config)
            config[-3] = json.loads(config[-3])
            config[-2] = json.loads(config[-2])
            yield config



def get_config(db, tablename):
    with db.begin() as conn:
        q = """ select * from synth where tablename = %s"""
        config = conn.execute(q, tablename).fetchone()
        config = list(config)
        config[-3] = json.loads(config[-3])
        config[-2] = json.loads(config[-2])
        return config

def get_pts(db, tablename):
    with db.begin() as conn:
        q = """ select * from %s""" % tablename
        return [pt[1:] for pt in conn.execute(q).fetchall()]



def compute_c_values(pts, mid_bounds, high_bounds, f=np.mean):
  """
  @deprecated don't use
  """
  return 0, 1

  orig_f = f
  f = lambda pts: orig_f([pt[-1] for pt in pts])
  pts = set(map(tuple,pts))
  all_vs, mid_vs, high_vs = set(), set(), set()
  n_mid, n_high = 0, 0
  for pt in pts:
      if pt[-2] != 8:
          continue
      all_vs.add(pt)
      if in_box(pt[:-2], mid_bounds):
          mid_vs.add(pt)
      if in_box(pt[:-2], high_bounds):
          high_vs.add(pt)


  orig = f(all_vs)
  nomid = f(all_vs.difference(mid_vs))
  nohigh = f(all_vs.difference(high_vs))

  dm = orig - nomid
  dh = orig - nohigh

  cm = len(mid_vs)
  ch = len(high_vs)
  
  highc = math.log(dh / dm) / math.log(1.*ch/cm)
  
  stats = []
  for c in xrange(0, 50, 5):
      c = c / 100.
      minfs, hinfs = [], []
      for n in xrange(5000):
          mid_samp = random.sample(mid_vs.difference(high_vs), random.randint(1, 100))
          high_samp = random.sample(high_vs, random.randint(1, 100))
          mix_samp = mid_samp + high_samp
          dm = orig - f(all_vs.difference(mix_samp)) 
          dh = orig - f(all_vs.difference(high_samp))
          ch, cm = float(len(high_samp)), float(len(mix_samp))
          if dh <= 0 or dm <= 0 or dh == dm or ch == cm:
              continue

          minfs.append(dm / cm**c)
          hinfs.append(dh / ch**c)

      mmean, mstd = np.mean(minfs), np.std(minfs)
      hmean, hstd = np.mean(hinfs), np.std(hinfs)
      stats.append((c, mmean, mstd, hmean, hstd))
      print ('\t%.4f'*5) % stats[-1]
  return highc - .25*np.std(highcs), highc + .25*np.std(highcs)

  return highc

def save_result(db, total_cost, costs, stats, rule, ids, dataset, notes, kwargs):
  acc, p, r, f1 = stats
  ids_str = ','.join(map(str, ids))
  isbest = rule.isbest
  vals = [0, dataset, notes]
  pkeys = ['klassname', 'cols', 'epsilon', 'c', 'lambda', 'boundtype']
  vals.extend([kwargs.get(key, None) for key in pkeys])
  vals.extend((total_cost, acc, p, r, f1, rule.quality, isbest, str(rule), ids_str))
  stat = vals


  with db.begin() as conn:
      q = """insert into stats(expid, dataset, notes, klass, cols, epsilon, c,     
                                lambda, boundtype, cost, acc, prec, recall, f1, score, 
                                isbest, rule, ids) values(%s) returning id""" % (','.join(['%s']*len(stat)))
      sid = conn.execute(q, *stat).fetchone()[0]

      q = """insert into costs(sid, name, cost) values(%s,%s,%s)"""
      for name, cost in costs.items():
          if isinstance(cost, list):
              cost = cost[0]
          conn.execute(q, sid, name, cost)
      return sid



def compute_stats_from_bounds(sigmoddb, learner, dataset, bounds, table_size, ids):
    try:
        truth = set(get_ids_in_bounds(sigmoddb, dataset, bounds))
        return compute_stats(ids, truth,  table_size)
    except:
        print "defaulting"
        return learner.compute_stats(ids)

def compute_all_stats_from_bounds(sigmoddb, learner, dataset, bounds, table_size, all_ids):
    try:
        truth = set(get_ids_in_bounds(sigmoddb, dataset, bounds))
        return [compute_stats(ids, truth,  table_size) for ids in all_ids]
    except:
        print "defaulting"
        return map(learner.compute_stats,all_ids)




def run(sigmoddb, statsdb, tablename, all_bounds, **kwargs):

    try:
        os.system('rm *.cache')
    except:
        pass


    params = {
            'epsilon' : 0.0005,
            'tau' : [0.1, 0.5],
            'p' : 0.7,
            'l' : 0.5,
            'min_improvement' : 0.01,
            'c' : 0.0,
            'max_wait' : 20 * 60,
            'klass' : BDT,
            'granularity' : 20
            }

    params.update(kwargs)
    klassname = params['klass'].__name__
    params['klassname'] = klassname
    params['dataset'] = tablename
    params['use_mtuples'] = params.get('use_mtuples', True) and klassname == 'BDT'
    cs = params['cs']




    if klassname == 'Naive':
        print "running experiment"
        costs, rules, all_ids, table_size, learner = run_experiment(tablename, **params)
        print "computing stats and saving"
        for bounds, boundtype in all_bounds:
            params['boundtype'] = boundtype


            for c, rules in learner.bests_per_c.items():
                if not rules: continue
                params['c'] = c
                rule = max(rules, key=lambda r: r.quality)
                ids = get_row_ids(rule, learner.full_table)
                stats = compute_stats_from_bounds(sigmoddb, learner, tablename, bounds, table_size, ids)
                save_result(statsdb, costs['cost_total'], costs, stats, rule, ids, tablename, str(bounds), params)

            for c, checkpoints in learner.checkpoints_per_c.items():
                params['c'] = c
                for timein, rule in checkpoints:
                    chk_ids = get_row_ids(rule, learner.full_table)
                    stats = compute_stats_from_bounds(sigmoddb, learner,  tablename, bounds, table_size, chk_ids)
                    save_result(statsdb, timein, {}, stats, rule, chk_ids, tablename, str(bounds), params)
        print "done"
    else:
        for c in cs:
            params['c'] = c
            print "running experiment"
            costs, rules, all_ids, table_size, learner = run_experiment(tablename, **params)
            print "computing stats and saving"

            for bounds, boundtype in all_bounds:
                params['boundtype'] = boundtype
#                truth = set(get_ids_in_bounds(sigmoddb, tablename, bounds))
#                all_stats = [compute_stats(ids, truth,  table_size) for ids in all_ids]
                all_stats = compute_all_stats_from_bounds(sigmoddb, learner, tablename, bounds, table_size, all_ids)
                for stats, rule, ids in zip(all_stats, rules, all_ids):
                    save_result(statsdb, costs['cost_total'], costs, stats, rule, ids, tablename, str(bounds), params)
            print "done"


def run_tests(sigmoddb, statsdb, **params):
    """
    These recreate the synthetic tests that use 2k pts per group, and 
    vary dimensionality and easy vs hard.  These are all graphs
    section 8.3 except figures 15 and 16

    defaults: 
     epsilon: 0.001
     tau: [0.1, 0.5]
     p: 0.7
     lambda: 0.5
     min_improvement: 0.01
    """

    for ndim in [2,3,4]:
        for uo in [30, 80]:

            tablename = "data_%d_%d" % (ndim, uo)
            config = get_config(sigmoddb, tablename)
            mid_bounds = config[-3]
            high_bounds = config[-2]
            all_bounds = [(mid_bounds, 'mid'), (high_bounds, 'high')]
            pts = get_pts(sigmoddb, tablename)
            cs = reversed([0., 0.05, 0.075, 0.09, 0.1, 0.15, 0.2, 0.5])
            #cs = [0.5, 0.2]

            for klass in [Naive, BDT, MR]:
                run(sigmoddb, statsdb, tablename, all_bounds, cs=cs, klass=klass, **params)

statsdb = create_engine('postgresql://localhost/dbwipes')
sigmoddb = create_engine('postgresql://localhost/sigmod')
init_db(statsdb)



for dataset in ['0', '11']:
    for klass in [MR,BDT,NDT]:
        run(sigmoddb, statsdb, dataset, [(None, 'none')], 
            cs=[0,0.05,0.075,0.09,0.1,0.15,0.2,0.5], 
            klass=klass, 
            granularity=15, 
            use_mtuples=True, 
            use_cache=False
        )
exit()

#tablename = 'data_2_30'
#config = get_config(sigmoddb, tablename)
#pts = get_pts(sigmoddb, tablename)
#run_tests(sigmoddb, statsdb, max_wait=12*60, granularity=15, use_mtuples=True, use_cache=False)

if False:
    setup(sigmoddb)
    for uo in [30, 80]:#40, 50, 60, 70, 80, 90, 100]:
        generate_datasets(sigmoddb, 2, uo=uo)
        generate_datasets(sigmoddb, 3, uo=uo)
        generate_datasets(sigmoddb, 4, uo=uo)
