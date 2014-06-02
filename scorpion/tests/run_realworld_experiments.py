from common import *
from test import run

if __name__ == '__main__':
  from sqlalchemy import create_engine
  db = create_engine('postgresql://localhost/sigmod')
  pp = PdfPages('figs/test.pdf')

  print "did you run setup_experiments.py to generate the testing env?"
  run(db, 19, pp, klasses=[MR], max_wait=10*60, c=[0., .25, .5, .75, 1.], granularity=20)
  exit()

  # run and gather "ground truth" for everything
  run(db, 0, pp, klasses=[Naive], max_wait=10*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)
  run(db, 11, pp, klasses=[Naive], max_wait=10*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)
  run(db, 5, pp, klasses=[Naive], max_wait=10*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)
  run(db, 15, pp, klasses=[Naive], max_wait=5*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)


  # run all others on intel_noon
  run(db, 0, pp, klasses=[NDT], c=[0., 0.25, 0.5, 0.75, 1.])
  run(db, 0, pp, klasses= [BDT], l=[0.5], c=[0., 0.25, 0.5, 0.75, 1.], epsilon=[0.0001, 0.001], tau=[0.1, 0.45])
  run(db, 0, pp, klasses= [BDT], l=[0.5], c=[1.], epsilon=[0.01, 0.1], tau=[0.1, 0.45])
  run(db, 0, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=20)

  # run obama
  run(db, 11, pp, klasses=[NDT], c=[0., 0.25, 0.5, 0.75, 1.])
  run(db, 11, pp, klasses=[BDT], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], epsilon=[0.0001, 0.001], tau=[0.1, 0.45])
  run(db, 11, pp, klasses=[BDT], l=[.5], c=[1.], epsilon=[0.01, 0.1], tau=[0.1, 0.45])
  run(db, 11, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=20)

  # run harddata 1
  run(db, 5, pp, klasses=[NDT], c=[0., 0.25, 0.5, 0.75, 1.])
  run(db, 5, pp, klasses=[BDT], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], epsilon=[0.0001, 0.001], tau=[0.1, 0.5])
  run(db, 5, pp, klasses=[BDT], l=[.5], c=[1.], epsilon=[0.01, 0.1])
  run(db, 5, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=10)
  run(db, 5, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=20)

  # run harddata 15  -- high dim
  run(db, 15, pp, klasses=[NDT], c=[0., 0.25, 0.5, 0.75, 1.])
  run(db, 15, pp, klasses=[BDT], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], epsilon=[0.0001, 0.001])
  run(db, 15, pp, klasses=[BDT], l=[.5], c=[1.], epsilon=[0.0001, 0.001, 0.01, 0.1])
  run(db, 15, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=10)
  run(db, 15, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=20)


  run(db, 0, pp, klasses=[NaiveMR], max_wait=20*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)
  run(db, 11, pp, klasses=[NaiveMR], max_wait=20*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)
  run(db, 5, pp, klasses=[NaiveMR], max_wait=20*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)
  run(db, 15, pp, klasses=[NaiveMR], max_wait=20*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)



  # run naive on intel_noon and increase the available columns
  # will take forever...
  run(db, 0, pp, klasses=[Naive], max_wait=30*60, cols=['voltage'], granularity=20, naive=True)
  run(db, 0, pp, klasses=[Naive], max_wait=30*60, cols=['voltage', 'humidity'], granularity=20, naive=True)
  run(db, 0, pp, klasses=[Naive], max_wait=30*60, cols=['voltage', 'humidity', 'light'], granularity=20, naive=True)
  run(db, 0, pp, klasses=[Naive], max_wait=30*60, cols=['moteid'], granularity=20, naive=True)
  run(db, 0, pp, klasses=[Naive], max_wait=30*60, granularity=20, naive=True)


  pp.close()
