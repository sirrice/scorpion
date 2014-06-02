import sys
import pdb

from sqlalchemy import create_engine
import psycopg2

try:
  from monetdb import sql as msql
except:
  msql = None

def connect(dbname, engine='pg'):
    try:
      if msql and engine == 'monet' and dbname in ('intel', 'med'):
        db = msql.connect(user='monetdb', password='monetdb', hostname='localhost', database=dbname)
      else:
        conn = "postgresql://localhost/%s" % dbname
        db = create_engine(conn)
        #connection = "dbname='%s' user='sirrice' host='localhost' port='5432'" % (dbname)
        #db = psycopg2.connect(connection)
    except:
        sys.stderr.write( "couldn't connect\n")
        sys.exit()
    return db


def query(db, queryline, params=None):
    if db == None:
        return

    if 'monet' in str(db.__class__):
      res = db.cursor()
      if params:
        res.execute(queryline, params)
      else:
        res.execute(queryline)
    else:
      if params:
        res = db.execute(queryline, params)
      else:
        res = db.execute(queryline)
    try:
      for row in res.fetchall():
        yield row
    except:
      print queryline, params
      raise
    finally:
      res.close()

def db_type(db, table, col):
  q = """SELECT pg_type.typname FROM pg_attribute, pg_class, pg_type where 
    relname = %%s and pg_class.oid = pg_attribute.attrelid and attname = '%s' and
    pg_type.oid = atttypid"""
  try:
    row = db.execute(q % col, table).fetchone()
    return row[0]
  except Exception as e:
    print e
    return None



def close(db):
    if db != None:
        db.close()

