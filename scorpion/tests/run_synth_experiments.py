import click
import json

from test import *
from common import *
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle as Rect
from matplotlib import cm 

from scorpion.sigmod.svm import SVM
from scorpion.sigmod.clique import MR
from scorpion.sigmod.bdt import BDT

def run_synth(db, pp, ndim, uo, volperc):
  defaults = {
    'ndim': ndim,
    'kdim': ndim,
    'npts': 2000,
    'uo': uo,
  }
  creator = ParamsCreator(defaults)

  # options to pass into runner
  rundefaults = {
    'cs': (np.arange(20) * 5. / 100.).tolist(),
    'c_range': [0.0, 1.]
  }
  runoptions = {
    'klass': [Naive, MR],
    'max_wait': [45*60],
    'max_complexity': [4],
    'min_improvement': [.01],
    'l': [.5],
    'p': [0.6],
    'parallel': [True],
    'granularity': [20],
  }

  vol = int(100. * volperc)
  for tableargs in creator.params:
    targs = pluck(tableargs, ['ndim', 'kdim', 'npts', 'uo'])
    targs.insert(-1, vol)
    tablename = 'data_%d_%d_%d_0d%d_%duo' % tuple(targs)

    params = {
      'cs': (np.arange(50) * 2. / 100.).tolist(),
      'c_range': [0.01, 1.],
      'klass': Naive,
      'max_wait': 45*60,
      'max_complexity': 4,
      'epsilon': 0.1,
      'tau': [0.1, .55],
      'min_improvement': .01,
      'l': .5,
      'p': 0.6,
      'parallel': True,
      'granularity': 20,
    }
    #run(db, tablename, pp, params=params)
    #bdt:
    #  epsilon: .001, .01, .05
    # l: 0.5, 1

    #params['klass'] = BDT
    #run(db, tablename, pp, params=params)


    params['klass'] = MR 
    run(db, tablename, pp, params=params)
    break


    pc = ParamsCreator(rundefaults)
    for name, opts in runoptions.iteritems():
      pc.add_options(name, opts)

    for params in pc.params:
      expid = nextexpid(db)
      run(db, tablename, pp, params=params)

def run_synth_single_c(db, pp, ndim, volperc):
  params = []
  kdim = ndim
  npts = 1000
  for uo in (80,): # (30, 80)
    args = (ndim, kdim, npts, int(100.*volperc), uo)
    tablename = 'data_%d_%d_%d_0d%d_%duo' % args
    params.append((tablename, (npts, ndim, kdim, volperc, 10, 10, uo, 10)))

  for tablename, args in params:
    if ndim == 2 and kdim == 1:
        continue

    expid = nextexpid(db)
    #run_params(db, tablename, expid, pp, klassname='Naive', klass=Naive, max_wait=45*60, cs=cs, granularity=10, naive=True, notes=tablename, p=.6, cols=None, l=.5, max_complexity=4)

    #klasses=[Naive], max_wait=45*60, c=cs, cs=cs, granularity=15, naive=True, notes=tablename)
    #klasses=[NDT],  c=cs, granularity=20, naive=True, notes=tablename)
    #klasses= [BDT], l=[0.5], c=cs, epsilon=[0.0001, 0.001], tau=[0.1, 0.55], p=0.7, notes=tablename)
    #klasses= [BDT], l=[0.5], c=[1.], epsilon=[0.01, 0.1], tau=[0.1, 0.55], p=0.7, notes=tablename)
    run(db, tablename, pp, klasses=[MR], l=[.5], cs=cs, c=cs, granularity=20, notes=tablename, parallel=True)
    break


def get_synth_ground_truth(db, table):
  q = """select boxes from dataset_metadata where tablename = %s"""
  for row in db.execute(q, (table,)).fetchall():
    boxes = json.loads(row[0])
    for key, bounds in boxes.items():
      ids = set()
      for bound in bounds:
        ids.update(get_ids_in_bounds(db, table, bound))
      yield key, ids
    return
    


def compute_statistics(db):
  q = """select expid, id, dataset, ids
  from results as r
  where r.completed = true and (
    select count(*) from stats as s
    where s.expid = r.expid and s.resultid = r.id
  ) = 0 """
  for expid, resid, dataset, ids_str in db.execute(q).fetchall():
    if ids_str:
      ids = set(map(int, map(float, ids_str.split(','))))
    else:
      ids = set()
    for boxtype, truth in get_synth_ground_truth(db, dataset):
      _, prec, recall, f1 = compute_stats(ids, truth, 1)
      q = """insert into stats
        values(%s, %s, default, default, %s, %s, %s, %s)"""
      db.execute(q, (expid, resid, prec, recall, f1, boxtype))

@click.command()
@click.option('--stats', is_flag=True)
@click.argument('ndim', type=int)
@click.argument('uo', type=int)
@click.argument('volperc', type=float)
def main(stats, ndim, uo, volperc):
  from sqlalchemy import create_engine
  db = create_engine('postgresql://localhost/sigmod')
  if stats:
    compute_statistics(db)
    return

  print "did you run setup_experiments.py to generate the testing env?"
  pp = PdfPages('figs/test.pdf')
  try:
    run_synth(db, pp, ndim, uo, volperc)
    compute_statistics(db)
  except Exception as e:
    print e
  pp.close()

if __name__ == '__main__':
  main()
