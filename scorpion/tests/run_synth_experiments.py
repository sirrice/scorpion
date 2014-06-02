from test import run, nextexpid, run_params
from common import *
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle as Rect
from matplotlib import cm 


def run_synth(db, pp, ndim, volperc):
  params = []
  kdim = ndim
  npts = 2000
  for uo in (80,): # (30, 80)
    args = (ndim, kdim, npts, int(100.*volperc), uo)
    tablename = 'data_%d_%d_%d_0d%d_%duo' % args
    params.append((tablename, (npts, ndim, kdim, volperc, 10, 10, uo, 10)))

  for tablename, args in params:
    if ndim == 2 and kdim == 1:
        continue

    cs = [.61, .62, .63, .64, .65, .66, .67, .68, .69]
    expid = nextexpid(db)
    #run_params(db, tablename, expid, pp, klassname='Naive', klass=Naive, max_wait=45*60, cs=cs, granularity=10, naive=True, notes=tablename, p=.6, cols=None, l=.5, max_complexity=4)

    #run(db, tablename, pp, klasses=[Naive], max_wait=45*60, c=cs, cs=cs, granularity=15, naive=True, notes=tablename)
    #run(db, tablename, pp, klasses=[NDT],  c=cs, granularity=20, naive=True, notes=tablename)
    #run(db, tablename, pp, klasses= [BDT], l=[0.5], c=cs, epsilon=[0.0001, 0.001], tau=[0.1, 0.55], p=0.7, notes=tablename)
    #run(db, tablename, pp, klasses= [BDT], l=[0.5], c=[1.], epsilon=[0.01, 0.1], tau=[0.1, 0.55], p=0.7, notes=tablename)
    run(db, tablename, pp, klasses=[MR], l=[.5], c=cs, granularity=20, notes=tablename)

if __name__ == '__main__':
  from sqlalchemy import create_engine
  db = create_engine('postgresql://localhost/sigmod')
  pp = PdfPages('figs/test.pdf')

  print "did you run setup_experiments.py to generate the testing env?"

  run_synth(db, pp, 2, 0.5)
  #run_synth(db, pp, 3, 0.5)
  #run_synth(db, pp, 4, 0.5)

  pp.close()


