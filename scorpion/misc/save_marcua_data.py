import sys
sys.path.append('/Users/sirrice/Dropbox/code/qurkexp/qurkexp/sensitive/')
sys.path.append('..')

from datasets import data_iterator
import matplotlib.pyplot as plt
from collections import *

from scorpionsql.db import *


def load_marcua_dataset():
    for expid, (prop, answers, truth) in enumerate(data_iterator()):
        cur_wid = 0
        wids = {}
        rowid = 0
        for ans in answers:
            # normalize worker ids
            if ans[1] not in wids:
                wids[ans[1]] = cur_wid
                cur_wid += 1
            row = list(ans)
            row[1] = wids[ans[1]]
            row.append(rowid)
            rowid += 1
            row.insert(0, prop)
            row.insert(0, expid)
            row.append(truth)

            yield row


db = connect('intel')
try:
    prepare(db, 'drop table marcua')
except:
    pass

try:
    q = """create table marcua (expid int, prop varchar(128),
    est float, wid int, atime float, secs float, height int, width int, id int,
    truth float);"""
    prepare(db, q)
except:
    raise


attrs = ['expid', 'prop', 'est', 'wid', 'atime', 'secs', 'h', 'w', 'id', 'truth']
for row in load_marcua_dataset():
    try:
        q = "insert into marcua values (%s)" % (','.join(['%s'] * len(attrs)))
        prepare(db, q, tuple(row))
    except:
        import traceback
        traceback.print_exc()

db.close()
