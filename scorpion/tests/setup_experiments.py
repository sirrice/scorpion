import sys
import pdb
import random
sys.path.extend(('.', '..'))

from scorpion.misc.gensinglecluster import gen_points, gen_multi_outliers
from scorpion.util import get_logger

from common import *

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

  try:
    with db.begin() as conn:
      conn.execute(""" CREATE TABLE dataset_metadata (
        id serial,
        tablename text,
        npts int,
        ndim int,
        kdim int,
        volperc float,
        uh float,
        sh float,
        uo float,
        so float,
        boxes text) 
      """)
  except Exception as e:
    print e


def setup_synth_metadata(tablename, params, oboxes, sboxes):
  from scorpion.db import connect
  eng = connect('sigmod')
  db = eng.connect()
  val = json.dumps({
    'oboxes': oboxes,
    'sboxes': sboxes
  })

  q = "insert into dataset_metadata values(DEFAULT, %s)" % ', '.join(["%s"]*10)
  args = [tablename]
  args.extend(params)
  args.append(val)
  db.execute(q, args)
  db.close()
  eng.dispose()

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
  npts = 2000
  for uo in (30, 80):
    tablename = 'data_%d_%d_%d_0d%d_%duo'
    args = (ndim, kdim, npts, int(100.*volperc), uo)
    tablename = tablename % args
    params.append((tablename, (npts, ndim, kdim, volperc, 10, 10, uo, 10)))


  for tablename, args in params:
    obox, sbox, schema, pts = gen_points(*args)
    setup_synth_data(tablename, schema, pts)
    setup_synth_metadata(tablename, args, [obox], [sbox])

def setup_multicluster(ndim, volperc):
  print >>sys.stderr, "loading multi-cluster outliers"

  npts = 2000
  params = []
  for uo in (30, 80):
    tablename = 'data2clust_%d_%d_%d_vol%d_uo%d' 
    args = (ndim, ndim, npts, int(100*volperc), uo)
    tablename = tablename % args
    params.append((tablename, (2000, ndim, ndim, volperc, 10, 10, uo, 10)))
 
  for tablename, args in params:
    oboxes, sboxes, schema, pts = gen_multi_outliers(*args)
    setup_synth_data(tablename, schema, pts)
    setup_synth_metadata(tablename, args, oboxes, sboxes)

if __name__ == '__main__':
  from sqlalchemy import create_engine
  db = create_engine('postgresql://localhost/sigmod')
  init_db(db) 




  setup_sigmod(2, 0.5)
  setup_sigmod(3, 0.5)
  setup_sigmod(4, 0.5)
  setup_sigmod(5, 0.5)
  setup_multicluster(2, .15)
  setup_multicluster(3, .15)
  setup_multicluster(2, .2)
  setup_multicluster(3, .2)
  setup_multicluster(4, .15)
  setup_multicluster(4, .2)
  setup_sigmod(2, 0.01)
  setup_sigmod(3, 0.01)
  setup_sigmod(4, 0.01)
  setup_sigmod(5, 0.01)
  setup_sigmod(2, 0.1)
  setup_sigmod(3, 0.1)
  setup_sigmod(4, 0.1)
  setup_sigmod(5, 0.1)



  db.dispose()

