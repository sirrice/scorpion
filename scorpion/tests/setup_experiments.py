import sys
import pdb
import random
sys.path.extend(('.', '..'))


from common import *
from misc.gensinglecluster import gen_points, gen_multi_outliers
from util import get_logger

_logger = get_logger()
random.seed(0)


def init_db(db):
  try:
    with db.begin() as conn:
        conn.execute("""create table stats (
    expid int,
    id serial,
    tstamp timestamp default current_timestamp,
    dataset text,
    klass varchar(128) null,
    cols text null,
    epsilon float null,
    c float null,
    lambda float null,
    cost float,
    acc float,
    prec float,
    recall float,
    f1 float,
    score float,
    isbest bool,
    ischeckpoint bool default FALSE,
    checkpoint float null,
    completed bool,
    rule text null,
    ids text null,
    notes text null,
    boundtype text null
            )""")
        conn.execute("""create table costs (
        id serial,
        sid int,
        name varchar(128),
        cost float)""")
  except:
      pass


def setup_synth_data(tablename, schema, pts):
  with file('/tmp/clusters.txt', 'w') as f:
      print >>f, '\t'.join(schema)
      for pt in pts:
          print >>f, '\t'.join(['%.4f']*len(pt)) % tuple(pt)
  os.system('importmydata.py %s sigmod /tmp/clusters.txt 2> /tmp/dbtruck.err > /tmp/dbtruck.log' % tablename)

def setup_sigmod(ndim, volperc):
  params = []
  #for kdim in xrange(1, ndim+1):
  kdim = ndim
  for uo in (30, 80):
    tablename = 'data_%d_%d_1000_0d%d_%duo' % (ndim, kdim, int(100.*volperc), uo)
    params.append((tablename, (2000, ndim, kdim, volperc, 10, 10, uo, 10)))


  for tablename, args in params:
    obox, sbox, schema, pts = gen_points(*args)
    setup_synth_data(tablename, schema, pts)

def setup_multicluster(ndim, volperc):
  print >>sys.stderr, "loading multi-cluster outliers"

  params = []
  for uo in (30, 80):
    tablename = 'data2clust_%d_%d_2k_vol%d_uo%d' % (ndim, ndim, int(100*volperc), uo)
    params.append((tablename, (2000, ndim, ndim, volperc, 10, 10, uo, 10)))
 
  for tablename, args in params:
    oboxes, sboxes, schema, pts = gen_multi_outliers(*args)
    setup_synth_data(tablename, schema, pts)

if __name__ == '__main__':
  from sqlalchemy import *
  db = create_engine('postgresql://localhost/sigmod')
  init_db(db) 


  #setup_sigmod(2, 0.5)
  #setup_sigmod(3, 0.5)
  #setup_sigmod(4, 0.5)
  #setup_sigmod(5, 0.5)
  #setup_multicluster(2, .15)
  #setup_multicluster(3, .15)
  #setup_multicluster(2, .2)
  #setup_multicluster(3, .2)
  setup_multicluster(4, .15)
  setup_multicluster(4, .2)


