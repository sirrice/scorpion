from sqlalchemy import create_engine
from common import *
db = create_engine('postgresql://localhost/dbwipes')
sigmoddb = create_engine('postgresql://localhost/sigmod')

def wrap(f):
    def _f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            import traceback
            traceback.print_exc()
            import pdb
            pdb.set_trace()
    return _f
            



def get_ids_from_bounds(dataset, bounds):
    """ bounds: [ (dimsuffix, lower, upper)* ] """

    wheres = []
    for d,l,u in bounds:
        if not math.isinf(l):
            wheres.append("%f <= a%d" % (l, d))
        if not math.isinf(u):
            wheres.append("a%d <= %f" % (d, u))
    where = ' and '.join(wheres) if wheres else ' 1 = 1 '
    with sigmoddb.begin() as sconn:
        q = "select id from %s where %s" % (dataset, where)
        ret = []
        for (id,) in sconn.execute(q):
            ret.append(int(id))

    return set(ret)

@wrap
def get_truths():

    db = create_engine('postgresql://localhost/dbwipes')
    q = """
    select distinct dataset, boundtype, notes from stats  where substring(dataset from 1 for 1) = 'd'   """
#    q = """
#    select s1.dataset, s1.expid, s1.c, s1.cost, s1.ids
#    from stats as s1, (select expid, c, max(cost) as cost
#                       from stats as s2
#                       where s2.klass = 'Naive'
#                       group by expid, c) as maxes
#    where klass = 'Naive' and 
#          s1.expid = maxes.expid and
#          s1.c = maxes.c and 
#          s1.cost = maxes.cost
#    """

    with db.begin() as conn:
        for dataset, boundtype, notes in conn.execute(q):
            pts = map(float,notes.replace(' ', '').replace('[','').replace(']', '').split(','))
            bounds = [(dim, pts[dim*2], pts[dim*2+1]) for dim in xrange(len(pts) / 2)]

            yield dataset, boundtype, get_ids_from_bounds(dataset, bounds)
            

cache = {}

@wrap
def get_alternative_results(db, did, boundtype):
    q = """
    select klass, id, c, cost, rule
    from stats
    where dataset = %s and boundtype = %s and klass = 'MR'
    order by klass, c, cost asc
    """
    with db.begin() as conn:
        for klass, id, c, cost, rule in conn.execute(q, did, boundtype).fetchall():
            key = hash((klass, c, did, rule))
            if key in cache:
                yield id, c, cost, cache[key]

            if rule:
                dimstrs = rule.split(' and ')
                dimstrs = [ds.split(' ') for ds in dimstrs]
                bounds = [(int(arr[2][-1]), float(arr[0]), float(arr[-1])) for arr in dimstrs]
            else:
                bounds = []

            cache[key] = get_ids_from_bounds(did, bounds)

            yield id, c, cost, cache[key] 

def save_scores(db, rowid, acc, p, r, f1):
    q = """
    update stats set acc = %s, prec = %s, recall = %s, f1 = %s
    where id = %s
    """
    with db.begin() as conn:
        conn.execute(q, acc, p, r, f1, rowid)

def table_size(dbname, table):
    db = create_engine('postgresql://localhost/%s' % dbname)
    with db.begin() as conn:
        return conn.execute("select count(*) from %s" % table).fetchone()[0]

def dataset_to_info(did):
    if did == 0:
        return ('intel', 'readings')
    if did == 5:
        return ('harddata', 'harddata_1')
    if did == 11:
        return ('fec12', 'expenses')
    if did == 15:
        return ('harddata','multdim') 
    return ('sigmod', did)

for dataset, boundtype, true_ids in get_truths():
    dbname, tname = dataset_to_info(dataset)
    print dbname, tname, boundtype
    tsize = table_size(dbname, tname)
    #for rowid, cost, ids in  get_checkpoints(db, eid, c, max_cost):
    for rowid, c, cost, ids in get_alternative_results(db, dataset, boundtype):
        acc, p, r, f1 = compute_stats(ids, true_ids, tsize)
        #print '\t%.4f\t%.4f\t%.4f\t%.4f' % ( cost, p, r, f1)
        save_scores(db, rowid, acc, p, r, f1)



