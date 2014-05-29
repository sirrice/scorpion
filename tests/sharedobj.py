from scorpion.db import *
from scorpion.sqlparser import *
from scorpion.sharedobj import *

eng = connect('intel')
sql = parse_sql(None, 'select sensor, count(temp) from readings group by sensor')
so = SharedObj(eng, '', parsed=sql, dbname='intel')
print so.get_tuples([None])
print len(so.get_tuples(['18']))
