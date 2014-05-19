#
# Validates that table doesn't contain any nulls
# and optionally replaces null values with sane defaults
#
from sqlalchemy import *
from datetime import date



def get_cols(db, tablename):
  q = "select attname from pg_class, pg_attribute where relname = %s and attrelid = pg_class.oid and attnum > 0;"
  ret = []
  for (attr,) in db.execute(q, tablename):
    ret.append(attr)
  return ret


def fix_typ(db, tablename, col, count, typ, bfix):
  vals = []
  if 'varchar' in typ:
    t = str
    vals = ['', 'NULL', '_NULL_']
  elif 'int' in typ:
    t = int
    vals = [0, -1]
  elif 'float' in typ:
    t = float
    vals = [0, -1]
  elif 'date' in typ:
    vals = [date(1000, 1, 1)]
    t = date
  else:
    vals = []
    print "\t%s\t%s\tno defaults" % (typ, col)

  if count == 0: return

  print "fixing %s as %s" % (col, typ)
  for val in vals:
    q = "SELECT count(*) from %s where %s = %%s" % (tablename, col)
    nconflicts = db.execute(q, val).fetchone()[0]
    print "\tconflicts: %s\t%s" % (nconflicts, val)
    
    if nconflicts is 0 and bfix:
      print "\tupdating values to %s" % val
      q = "UPDATE %s SET %s = %%s WHERE %s is null " % (tablename, col, col)
      db.execute(q, val)
      try:
        db.commit()
      except Exception as e:
        print "\t%s" % e
        pass
      print "\tsuccess"
      break




def check_and_fix(db, tablename, col, bfix):
  q = "SELECT count(distinct %s) from %s where %s is not null" % (col, tablename, col)
  q = "SELECT count(*) from %s where %s is null" % (tablename, col)
  row = db.execute(q).fetchone()
  count = row[0]

  q = """SELECT pg_type.typname FROM pg_attribute, pg_class, pg_type where 
    relname = %%s and pg_class.oid = pg_attribute.attrelid and attname = '%s' and
    pg_type.oid = atttypid"""
  row = db.execute(q % col, tablename).fetchone()
  typ = row[0]
  print "%s\t%s\t%s" % (count, typ, col)

  fix_typ(db, tablename, col, count, typ, bfix)

import sys
if len(sys.argv) <= 2:
  print "python fixnulls.py <dbname> <tablename> <fix>"
  exit()
else:
  dbname = sys.argv[1]
  tablename = sys.argv[2]
  bfix = False
  if len(sys.argv) >= 4:
    bfix = bool(sys.argv[3])

if bfix:
  v = raw_input("going to modify database!  enter 'x' to abort: ")
  if 'x' in v:
    print "aborting"
    exit()

db = create_engine("postgresql://localhost/%s" % dbname)

cols = get_cols(db, tablename)
for col in cols:
  check_and_fix(db, tablename, col, bfix)

#from matplotlib import pyplot as plt
#
#plt.bar(range(len(vals)), vals)
#plt.show()
