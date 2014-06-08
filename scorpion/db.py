import sys
import pdb
import psycopg2

from sqlalchemy import create_engine
from datetime import datetime, date, timedelta
from datetime import time as dttime
from collections import deque, defaultdict
from dateutil.parser import parse as dateparse


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


def db_schema(db, table):
  """
  only works for postgres

  @return dictionary of column name -> data type
  """
  typedict = [('int', int), ('double', float),
              ('timestamp', datetime),
              ('date', date), ('time', dttime),
              ('text', str), ('char', str)]
  ret = {}
  q = '''select column_name, data_type
          from information_schema.columns
          where table_name = %s;'''
  #q = """select c.name, c.type from sys.columns as c, sys.tables as t where table_id = t.id and t.name = %s;"""
  # and data_type != 'date' and position('time' in data_type) =
  # 0 and column_name != 'humidity'
  for row in query(db, q, (table,)):
      name, dtype = tuple( row[:2] )
      name = str(name)
      for tn, tt in typedict:
          if tn in dtype:
              ret[name] = tt
              break
      if name not in ret:
          msg = "can't find type of %s\t%s"
          raise RuntimeError(msg % (name, dtype))
  return ret




def close(db):
    if db != None:
        db.close()

