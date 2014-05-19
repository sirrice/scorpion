import pdb
import datetime
import operator
import sqlparse
import json
import logging
import orange
import Orange
import orngTree
import orngStat
import orngTest
import orngDisc
import Orange.feature as orf
import numpy as np


from datetime import datetime, date, timedelta
from datetime import time as dttime
from collections import deque, defaultdict
from dateutil.parser import parse as dateparse


from db import *
from sql import *
from util import *
from score import *
from classify import *
from aggerror import *
from settings import *
from sqlparser import *



_logger = get_logger()


# shared object
#
# get_provenance(results)
# for each result:
#   get provenance of result
#   rank provenance
#   SD2 cluster
# construct input dataset
# clean input dataset
# expand input dataset
# score(inputs)

def get_distribution(obj, aggerr, goodresults):
    goodtable = get_provenance(obj, aggerr.agg.cols, goodresults)
    err_func = aggerr.error_func
    err_func.setup(goodtable)
    good_dist = err_func.distribution(goodtable)
    return good_dist


def get_provenance(sharedobj, cols, keys):
    schema = sharedobj.rules_schema
    schema.update(cols)
    #_logger.debug( "get_provenance schema\t%s", schema)
    rows = sharedobj.get_tuples(keys, attrs=schema)
    if not len(rows):
        return None
    table = create_orange_table(rows, schema)
    return table


def get_provenance_split(sharedobj, cols, keys):
    """
    NOTE: #RNGE errors are when source table contains nulls.
    """
    gb = sharedobj.parsed.select.nonaggs[0]
    schema = sharedobj.rules_schema
    schema.update(cols)
    schema = list(schema)
    schema.append(str(gb))
    #_logger.debug( "get_provenance schema\t%s", schema)

    
    start = time.time()
    rows = sharedobj.get_tuples(keys, attrs=schema)
    print "dbquery cost %.4f" % (time.time() - start)
    if not len(rows): return None


    # create a column for the partitioning key so we can efficiently
    # apply as a mask
    cols = map(np.array, zip(*rows))

    # NOTE: don't have nulls anymore
    #badmask = bad_row_mask(schema, cols[:-1])
    #cols = [col[badmask] for col in cols]
    keycol = np.array(cols.pop())
    schema.pop()

    start = time.time()
    # this domain still allows nulls in discrete attrs
    domain, funcs = orange_schema_funcs(cols, schema)
    print "schema funcs %.4f" % (time.time() - start)

    start = time.time()
    table = construct_orange_table(domain, schema, funcs, cols)
    # removed nulls from domain NOTE: no more nulls
    #clean_domain = domain_rm_nones(domain)
    clean_domain = domain
    print "create complete table %.4f" % (time.time() - start)

    start = time.time()
    tables = []
    data = table.to_numpyMA('ac')[0].data
    # find the rows that contain None values in discrete columns
    print "got numpy array %.4f" % (time.time() - start)

    start = time.time()
    for idx, key in enumerate(keys):
      # mask out rows that are not in this partition or have Nulls
      partition_data = data[(keycol == key),:]
      partition = Orange.data.Table(clean_domain, partition_data)

      #badrows = [row for row in partition if '#RNGE' in map(str, row)]
      #if badrows:
      #  import pdb
      #  pdb.set_trace()


      tables.append(partition)

    print "create orange time %.4f " % (time.time() - start)
    return tables






def extract_agg_vals(vals, col_type):
    fmts = [
      '%Y-%m-%dT%H:%M:%S.%fZ',
      '%Y-%m-%dT%H:%M:%S.%f',
      '%Y-%m-%dT%H:%M:%S',
      '%Y-%m-%dT%H:%M',
      '%Y-%m-%dT%H'
    ]
    for fmt in fmts:
      try:
        ret = [datetime.strptime(val, fmt) for val in vals]
        print vals
        if col_type == 'date':
          ret = [d.date() for d in ret]
        elif 'Z' in fmt:
          #ret = [d - timedelta(hours=5) for d in ret] # compensate for 'Z' +4 timezone
          pass
        return ret
      except Exception as e:
        pass

    try:
      ret = [datetime.strptime(val, '%Y-%m-%d').date() for val in vals]
      return ret
    except Exception as ee:
      print ee
      return vals



def parse_debug_args(db, form, dbname=None):
    data = json.loads(form.get('data', '{}'))
    goodkeys = json.loads(form.get('goodkeys', '{}'))
    errids = json.loads(form.get('bad_tuple_ids', '{}'))
    sql = form['query']
    attrs = json.loads(form.get('attrs', '[]'))

    errids = dict([(key.strip('()'), ids) for key, ids in errids.items()])

    try:
      c = float(form.get('c', 0.3))
    except:
      c = 0.3

    obj = SharedObj(db, sql, bad_tuple_ids=errids)
    obj.c = c

    ignore_attrs = set(obj.attrnames).difference(attrs)
    ignore_attrs.add('finan_icu_days')
    obj.ignore_attrs = ignore_attrs
    qobj = obj.parsed    


    erreq = errtype = None
    if 'errtype' in form:
      errtype = int(form['errtype'])
      try:
		    erreqs = json.loads(form.get('erreq', '{}')) # only if error type == EQUALTO
      except:
        erreqs = {}

    nonagg = qobj.select.nonaggs[0]
    col_type = db_type(db, qobj.fr, nonagg.cols[0])
    
    errors = []
    for agg in qobj.select.aggregates:
        label = agg.shortname
        if label not in data:
            continue

        if errtype == ErrTypes.EQUALTO:
          erreq = erreqs[label]
          if len(erreq) != len(data[label]):
            raise RuntimeError("errtype was EQUAL but number of erreq values (%d) != number of aggs (%d) for agg %s" % (len(erreq), len(data[label]), label))
          print "erreq for %s: %s" % (label, ', '.join(map(str,erreq)))

        bad_keys = extract_agg_vals(data[label], col_type)
        err = AggErr(agg, bad_keys, 20, errtype, {'erreq' : erreq})
        errors.append(err)
        obj.goodkeys[label] = extract_agg_vals(goodkeys.get(label, []), col_type)

    obj.errors = errors
    obj.dbname = dbname

    return obj

def create_sharedobj(dbname, sql, badresults, goodresults, errtype, bad_tuple_ids={}):
    db = connect(dbname)
    obj = SharedObj(db, sql, dbname=dbname)


    qobj = obj.parsed
    errors = []
    for agg in qobj.select.aggregates:
        label = agg.shortname
        aggerr = AggErr(agg, extract_agg_vals(badresults), 20, errtype, {'erreq' : None})
        errors.append(aggerr)
        obj.goodkeys[label] = extract_agg_vals(goodresults)
    obj.errors = errors

    table = get_provenance(obj, obj.errors[0].agg.cols, obj.errors[0].keys)
    return obj, table


class SharedObj(object):

    def __init__(self, db, sql,
                 errors=[],
                 bad_tuple_ids=None,
                 goodkeys={},
                 ignore_attrs=[],
                 schema=[],
                 dbname=None,
                 **kwargs):
        if not db and not dbname:
            raise "SharedObj requires a database connection!"
        self.db = db or connect(dbname)
        dbname = dbname or str(self.db.url).split("/")[-1]
        self.monetdb = connect(dbname, engine='monet')
        self.dbname = dbname
        if 'parsed' in kwargs:
          self.parsed = kwargs['parsed']
        else:
          self.parsed = parse_sql(db, sql)
        if len(self.parsed.fr) > 1:
            # XXX: only support single table queries
            raise "Don't support joins yet!"
        self.errors = errors
        self.goodkeys = goodkeys or {}
        self.schema = schema or SharedObj.get_schema(db, self.parsed.tables[0])
        # aggregate type -> {groupby key -> ids of "bad" tuples}
        self._bad_tuple_ids = bad_tuple_ids or defaultdict(set)
        self.ignore_attrs = ignore_attrs
        self.merged_tables = {}
        self.rules = {}
        self.clauses = {}
        self.c = kwargs.get('c', 0.3)
        
        
    def get_tuples(self, keys, attrs=None):
        try:
            if keys is None or not len(list(keys)):
                return []
        except:
            pass
        attrs = attrs or self.rules_schema
            
        return [list(row) for row in self.get_filter_rows(keys=keys, attrs=attrs)]
        
    
    def get_bad_tuple_ids(self, label=None):
        if label:
            return self._bad_tuple_ids.get(label, [])
        return self._bad_tuple_ids

    def clone(self):
        return SharedObj(self.db, self.sql,
                         dbname=self.dbname,
                         errors=self.errors,
                         bad_tuple_ids=self._bad_tuple_ids,
                         goodkeys=self.goodkeys,
                         ignore_attrs=self.ignore_attrs,
                         schema=self.schema)

    
    def add_where(self, where):
        self.parsed.where.append(where)

    def get_agg_rows(self, where=None, params=()):
        qobj = self.parsed.clone()
        if where:
            qobj.where.append(where)
        return query(self.monetdb, str(qobj), params)

    def get_agg_dicts(self, *args, **kwargs):
        selects = map(lambda s: s.shortname, self.parsed.select)
        for row in self.get_agg_rows(*args, **kwargs):
            yield dict(zip(selects, row))

    def get_filter_rows(self, keys=None, attrs=None, where=None, params=()):
        """
        Need to deal with keys and such outside of function
        """
        qobj = self.parsed.get_filter_qobj(keys=keys)
        if attrs:
            qobj.select = Select(attrs)
        if where:
            qobj.where.append(where)
        return query(self.db, str(qobj), params)

    def get_filter_dicts(self, *args, **kwargs):
        attrnames = kwargs.get('attrnames', None)
        if not attrnames:
            attrs = kwargs.get('attrs', None)
            if not attrs:
                attrs = self.filter.select
            attrnames = map(lambda s: isinstance(s, str) and s or s.shortname, attrs)
        if len(attrnames) != len(attrs):
            raise RuntimeError("attrnames and attrs should be same length\n\t%s\n\t%s" % 
                               (attrnames, attrs))
            
        for row in self.get_filter_rows(*args, **kwargs):
            yield dict(zip(attrnames, row))

    attrnames = property(lambda self: self.schema.keys())

    def attrs_without(self, colss=None):
        if colss is None:
            colss = map(lambda sel: sel.cols, self.parsed.select)
        if isinstance(colss, str):
            colss = [colss]
            
        names = set(self.attrnames)
        for cols in cols:
            if isinstance(cols, str):
                cols = [cols]
            names.difference_update(cols)
        return names

    @staticmethod
    def get_schema(db, table):
        """
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


    def get_rules_schema(self):
        """
        """
        invalid_types = [date, datetime, dttime]
        used_attrs = set()
        for selexpr in self.parsed.select:
            used_attrs.update(selexpr.cols)
        
        schema = dict(filter(lambda p: p[1] not in invalid_types, self.schema.iteritems()))
        ret = set(schema.keys()).difference(used_attrs)
        ret.add('id')
        return ret

    rules_schema = property(get_rules_schema)
    bad_tuple_ids = property(get_bad_tuple_ids)
    sql = property(lambda self: str(self.parsed))
    prettify_sql = property(lambda self: self.parsed.prettify())
    filter = property(lambda self: self.parsed.get_filter_qobj())
















def is_discrete(attr, col):
    if attr in [
      'epochid', 'voltage', 'xloc', 'yloc', 
      'est', 'height', 'width', 'atime', 'v',
      'light', 'humidity', 'age', 
      'finan_icu_days', 'dxage']:
        return False
    if attr in ['recipient_zip', 'sensor', 'moteid' 'file_num']:
        return True

    # continuous or discrete?
    # uniquecol = set(col)
    # nonnulls = filter(lambda x:x, col)
    # strtypes = map(lambda c: isinstance(c, str), nonnulls[:20])
    # istypestr = reduce(operator.or_, strtypes) if strtypes else True

    # if its strings
    nonnulls = filter(lambda x:x, col)
    strtypes = map(lambda c: isinstance(c, basestring), nonnulls[:20])
    istypestr = reduce(operator.or_, strtypes) if strtypes else True
    if istypestr:
        return True

    # if its floats
    try:
        isints = map(lambda v: int(v) == float(v), nonnulls)
        istypeint = reduce(operator.and_, isints) if isints else True
        if not istypeint:
            return False
    except:
        pass

    # or if there are too many unique values
    uniquecols = set(nonnulls)
    if len(uniquecols) > 0.05 * len(col):
        return False

    return True

def detect_discrete_cols(rows, attrs):
    attrs = list(attrs)
    cols = map(list, zip(*rows))
    dcols = []
    for idx, (attr, col) in enumerate(zip(attrs, cols)):
      if is_discrete(attr, col):
        dcols.append(attr)
    return dcols

def bad_row_mask(schema, cols):
  """
  returns a mask that is a 1D boolean array with length arr.shape[0]
  where the value is
    1 if row should be preserved
    0 if row should be removed
  """
  bad_attrs = ['moteid', 'sensor']
  mask = np.zeros(len(cols[0])).astype(bool)
  for attr, col in zip(schema, cols):
    if attr not in bad_attrs: continue
    colmask = np.equal(col, None) | (col == 'None')
    mask |= colmask

  try:
    return np.invert(mask)
  except:
    pdb.set_trace()



def orange_schema_funcs(cols, attrs):
  """
  @return (Domain, [funcs]) funcs is list of functions to apply to 
          the corresponding column value
  """
  # orngDisc.orngDisc.entropyDiscretization('')
  features = []
  funcs = []
  for idx, (attr, col) in enumerate(zip(attrs, cols)):
    start = time.time()
    bdiscrete = is_discrete(attr, col)

    if bdiscrete:
      feature = Orange.feature.Discrete(attr, values=map(str, set(col)))
      func = str
    else:
      feature = Orange.feature.Continuous(attr)
      func = lambda v: v is not None and float(v) or 0.

    print "\tschema func for %s\t%s\t%.4f" % (attr, func, time.time() - start)

    features.append(feature)
    funcs.append(func)

  domain = Orange.data.Domain(features)
  return domain, funcs

def domain_rm_nones(domain):
  features = []
  for attr in domain:
    if isinstance(attr, Orange.feature.Discrete):
      vals = attr.values
      vals = filter(lambda v: v not in [None, 'None'], vals)
      attr = Orange.feature.Discrete(attr.name, values=vals)
    features.append(attr)
  return Orange.data.Domain(features)


def construct_orange_table(domain, attrs, funcs, cols):
  start = time.time()
  for idx in xrange(len(cols)):
    if funcs[idx] == str:
      cols[idx] = cols[idx].astype(str)
    else:
      try:
        f = np.frompyfunc(funcs[idx], 1, 1)
        cols[idx] = funcs[idx](cols[idx])
      except Exception as e:
        _logger.debug(str(e))
        f = np.vectorize(funcs[idx])
        try:
          cols[idx] = f(cols[idx])
        except Exception as e:
          _logger.debug(str(e))
          cols[idx] = map(funcs[idx], cols[idx])

  print "func(cols) %.4f" % (time.time() - start)

  start = time.time()
  rows = map(list, zip(*cols))
  print "transpose %.4f" % (time.time() - start)
  start = time.time()
  table = Orange.data.Table(domain)
  table.extend(rows)
  print "created table %.4f" % (time.time() - start)
  return table


def create_orange_table(rows, attrs):
  cols = zip(*rows)
  domain, funcs = orange_schema_funcs(cols, attrs)
  attrs = list(attrs)
  return construct_orange_table(domain, attrs, funcs, cols)


def merge_tables(tables):
    """
    remove duplicates
    @return single orange table
    """
    domain = tables[0].domain
    ret = Orange.data.Table(domain)
    map(ret.extend, tables)
    ret.remove_duplicates()
    return ret





def create_clauses(sharedobj):
    """
    Convert clauses into SQL predicate strings
    """
    def filter_clause(clause):
      if not clause:
          return False
      if len(clause) > 1000:
          _logger.warn( "clause too long\t%d", len(clause))
          return False
      return True

    for label, rules in sharedobj.rules.iteritems():
      rules = map(lambda p: p[0], rules)
      sharedobj.clauses[label] = []
      for rule in rules:
        clauses = rule_to_clauses(rule)
        clause_str = ' or '.join(clauses)
        sharedobj.clauses[label].append(clause_str)
      #clauses = map(lambda rule:
                    #' or '.join(rule_to_clauses(rule)),
                    #rules)
      
      #sharedobj.clauses[label] = clauses




def rule_to_clauses(rule):
    try:
      return sdrule_to_clauses(rule)
    except:
      try:
        return c45_to_clauses(rule.tree)
      except:
        try:
          return tree_to_clauses(rule.tree)
        except:
          pass
    return []

def sdrule_to_clauses(rule):
    from learners.cn2sd.rule import infinity
    ret = []
    for i, c in enumerate(rule.filter.conditions):
      attr = rule.data.domain[c.position]
      name = attr.name
      # _logger.debug( "stringifying\t%s\t%s\t%s\t%s", c, type(c),
      #                isinstance(c,  Orange.core.ValueFilter_continuous), attr.varType)
      if isinstance(c, Orange.core.ValueFilter_continuous):
        # XXX: rounding to the 3rd decimal place as a hack            
        clause = []#'%s is not null' % name]
        if c.min == c.max and c.min != -infinity:
          v = round(c.min, 5)
          vint = int(v)
          vfloat = v - vint
          v = vint + float(str(vfloat).rstrip('0.') or '0')
          clause.append( 'abs(%s - %s) < 0.001' % (v, name) )
        else:
          if c.min != -infinity:
            clause.append( '%.7f <= %s' % (round(c.min, 5), name) )
          if c.max != infinity:
            clause.append( '%s <= %.7f ' % (name, round(c.max, 5)))
        if clause:
          ret.append( ' and '.join(clause) )
      elif attr.varType == orange.VarTypes.Discrete:
        if len(c.values) == len(attr.values):
          continue
        elif len(c.values) == 1:
          val = attr.values[int(c.values[0])]
        else:
          val = [attr.values[int(v)] for v in c.values]
          val = filter(lambda v: v != None, val)
        ret.append( create_clause(name, val) )


    return [ ' and '.join(ret) ]
        

def c45_to_clauses(node, clauses=None):
    clauses = clauses or []
    if not node:
        return []

    var = node.tested
    attr = var.name
    ret = []

    if node.node_type == 0: # Leaf
        if int(node.leaf) == 1:
            ret = ['(%s)' % ' and '.join(clauses)]

    elif node.node_type == 1: # Branch
        for branch, val in zip(node.branch, attr.values):
            clause = create_clause(attr,  val)
            clauses.append( clause )
            ret.extend( c45_to_clauses(branch, clauses) )
            clauses.pop()

    elif node.node_type == 2: # Cut
        for branch, comp in zip(node.branch, ['<=', '>']):
            clause = create_clause(attr,  node.cut, comp)
            clauses.append( clause )
            ret.extend( c45_to_clauses(branch, clauses) )
            clauses.pop()

    elif node.node_type == 3: # Subset
        for i, branch in enumerate(node.branch):
            inset = filter(lambda a:a[1]==i, enumerate(node.mapping))
            inset = [var.values[j[0]] for j in inset]
            if len(inset) == 1:
                clause = create_clause(attr, inset[0])
            else:
                clause = create_clause(attr, inset)
            clause.append( clause )
            ret.extend( c45_to_clauses(branch, clauses) )
            clauses.pop()

    ret = filter(lambda c: c, ret)
    return ret

            

def tree_to_clauses(node, clauses=None):
    clauses = clauses or []
    if not node:
        return []

    ret = []
    if node.branch_selector:
        varname = node.branch_selector.class_var.name
        for branch, bdesc in zip(node.branches,
                                 node.branch_descriptions):
            if ( bdesc.startswith('>') or 
                 bdesc.startswith('<') or 
                 bdesc.startswith('=') ):
                clauses.append( '%s %s'% (varname, bdesc) )
            else:
                clauses.append( create_clause(varname, bdesc) )
            ret.extend( tree_to_clauses(branch, clauses) )
            clauses.pop()
    else:
        major_class = node.node_classifier.default_value
        if major_class == '1' and clauses:
            ret.append( '(%s)' % ' and '.join(clauses) )

    ret = filter(lambda c: c, ret)
    return ret


def create_clause(attr, val, cmp='='):
    cmps = ['<', '<=', '>', '>=', '=']
    if isinstance(val, (list, tuple)):
        strings = filter(lambda v: isinstance(v, str), val)
        if len(strings):
            val = map(quote_sql_str, val)
        cmp = 'in'
        val = '(%s)' % ','.join(map(str, val))
    else:
        if isinstance(val, str):
            # it may be a discretized continuous condition (e.g., "<= 5")
            isnumerical = False
            for c in cmps:
                try:
                    if val.startswith(c):
                        val = float(val.split(c)[1])
                        cmp = c
                        isnumerical = True
                        break
                except:
                    pass
            if not isnumerical:
                try:
                    val = float(val)
                except:
                    val = quote_sql_str(val)

    return '%s %s %s' %  (attr, cmp, val)
    #return '%s is not null and %s %s %s' %  (attr, attr, cmp, val)

        

if __name__ == '__main__':
    from learners.cn2sd.rule import SDRule

    d = Orange.data.Table('iris')
    rule = SDRule(d, d.domain.classVar.values[0])
    rule = rule.cloneAndAddContCondition(d.domain[0], 4, 5)
    print sdrule_to_clauses(rule)
    exit()

    
    import sys
    import random
    import time
    import matplotlib
    import matplotlib.pyplot as plt    
    sys.path.append('.')

    from matplotlib.backends.backend_pdf import PdfPages
    from collections import Counter

    matplotlib.use("Agg")    
    pp = PdfPages('figs/arch.pdf')
    

    def plot_points(scores, true_scores, ylabel, xlabel, table):
        from collections import Counter, defaultdict
        
        add_jitter = lambda vect: [s + (random.random()-0.5)*(0.01*max(vect)) for s in vect]
        
        errs15 = []
        errsnorm = []
        for est, true, row in zip(scores, true_scores, table):
            if true > 0.8:
                errs15.append( (true, (est - true)) )
            else:
                errsnorm.append( (true, (est - true)) )

        fig = plt.figure()
        sub = fig.add_subplot(111)
        sub.scatter(add_jitter(true_scores),
                    add_jitter(scores),
                    s=2, alpha=0.2, lw=0, c='blue')
        sub.set_xlabel(xlabel)
        sub.set_ylabel(ylabel)
        
        sub2 = sub.twinx()
        xs, ys = zip(*errsnorm)
        sub2.set_ylim(min(ys), max(ys))
        sub2.scatter(add_jitter(xs),
                     add_jitter(ys),
                     s=4, alpha=0.2, lw=0, c='red')
        if errs15:
            xs, ys = zip(*errs15)
            sub2.set_ylim(min(ys), max(ys))
            sub2.scatter(add_jitter(xs),
                         add_jitter(ys),
                         s=4, alpha=0.2, lw=0, c='green')
        sub2.set_ylabel("est - true")
        plt.savefig(pp, format='pdf')


    def compare_scores(scores, true_scores):
        mse, wmse = 0., 0.
        n = min(len(scores), len(true_scores))
        for est_score, true_score in zip(scores, true_scores):
            mse += (est_score - true_score) ** 2
            wmse += ((est_score - true_score) * true_score) ** 2
        mse /= n
        wmse /= n
        return mse, wmse

    def run_combined(obj, badkeys, goodkeys):
        raise
        table = get_provenance(obj, ['temp'], badkeys)
        goodtable = get_provenance(obj, ['temp'], goodkeys)
        good_err_func = FastAvgErrFunc(['temp'])
        good_err_func.setup(goodtable)
        good_dist = good_err_func.distribution(goodtable)
        print good_dist
        table, rules = classify_error_tuples_modified(table,
                                                      good_dist,
                                                      AvgErrFunc(aggerr.agg.cols[:1]))
        for rule in rules:
            print rule.ruleToString()


    def run_server(obj):
        run_stuff(obj)
        for key, rules in obj.rules.items():
            for r, score in rules:
                print r.ruleToString()

    
    sql = '''select stddev(temp), avg(temp),
    ((extract(epoch from date+time - '2004-3-1'::timestamp)) / (30*60))::int as dist
    from newreadings
    where date+time > '2004-3-1'::timestamp and date+time < '2004-3-7'::timestamp
    group by dist order by dist;'''
    keyids = [126,127,65,66,67,68,74,75,76,122,121,124,123,125,128,129,130,131]
    goodids = [40, 41]
    aggerr = AggErr(SelectAgg('avg', 'avg', ['temp'], 'temp', None),
                    keyids, 0, ErrTypes.TOOHIGH)
    
    db = connect('intel')
    obj = SharedObj(db, sql, errors=[aggerr], goodkeys={'avg':goodids})
    table = get_provenance(obj, ['temp'], keyids)
        
    bad_tuple_ids = set([row['id'].value for row in table if row['temp'].value > 100])
    print 'bad tuples', len(bad_tuple_ids)

    klasses = [QuadScoreSample1, QuadScoreSample3, QuadScoreSample4]#, Scorer]
    scoress = []
    ncallss = []
    costs = []

    def scoreit(klass):
        table = get_provenance(obj, ['temp'], keyids)
        bad_tuple_ids = set([row['id'].value for row in table if row['temp'].value > 100])

        print "getting scores using", klass.__name__
        start = time.time()
        aggerr = AggErr(SelectAgg('avg', 'avg', ['temp'], 'temp', None),
                        keyids, 0, ErrTypes.TOOHIGH)
        
        scores, ncalls = score_inputs(table, aggerr, klass=klass)
        cost = time.time() - start
        scoress.append( scores )
        ncallss.append( ncalls )
        costs.append( cost )

        bad_scores = [score for row, score in zip(table, scores) if row['temp'].value > 100]
        good_scores = [score for row, score in zip(table, scores) if row['temp'].value <= 100]

        ignore_attrs = [table.domain[col].name for col in aggerr.agg.cols]
        ignore_attrs += ['voltage', 'humidity', 'light', 'epochid']

        start = time.time()
        table, rules, itercost = classify_error_tuples(table, scores, width=5, ignore_attrs=ignore_attrs, bdiscretize=False)
        classify_cost = time.time() - start
        print 'avg bad tuple scores\t', np.mean(bad_scores), np.std(bad_scores)
        print 'avg good tuple scores\t', np.mean(good_scores), np.std(good_scores)
        print "classify cost\t%.4f\t%.4f" % (classify_cost, itercost)
        for r in rules:
            print rule_to_clauses(r)
        

    import cProfile
    cProfile.run('scoreit(klasses[0])', 'tests/out/archprofile')    
    #for klass in klasses:
    #    scoreit(klass)

    exit()


    for klass, scores in zip(klasses, scoress):
        print "Rules using %s" % klass.__name__

        

    # compute pairwise errors
    # plot pairwise comparisons
    for k1, s1 in zip(klasses, scoress):
        for k2, s2 in zip(klasses, scoress):
            if k1 == k2: continue
            mse, wmse = compare_scores(s1, s2)
            print "Errors\t%s\t%s\tMSE: %f\tWMSE: %f" % (k1.__name__, k2.__name__, mse, wmse)
            plot_points(s1, s2, k1.__name__, k2.__name__, table)
    pp.close()

    print "Nrows\t       \t%d" % len(table)
    for k, n in zip(klasses, ncallss):
        print "Ncalls\t%s\t%d" % (k.__name__,n)

    for k, c in zip(klasses, costs):
        print "Costs\t%s\t%d" % (k.__name__, c)



    db.close()
