from scorpion.sharedobj import *
from scorpion.sql import *
from scorpion.sqlparser import *
from scorpion.db import *



parsed = parse_sql("select hr, avg(temp) from readings group by hr")
parsed.where.append("sensor in %s")
params = [('1', '2', '18')]

db = connect("intel")

obj = SharedObj(db, parsed=parsed, params=params)
for t in obj.get_filter_rows(['2004-03-05 12:00:00']):
  print t
