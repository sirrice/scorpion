import pdb
import datetime
import operator
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


from sharedobj import *
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

    # NOTE: we assume there are _no null values_
    #       enforce this by running misc/fixnulls.py
    keycol = np.array(cols.pop())
    schema.pop()

    start = time.time()
    # this domain still allows nulls in discrete attrs
    domain, funcs = orange_schema_funcs(cols, schema)
    print "schema funcs %.4f" % (time.time() - start)

    start = time.time()
    table = construct_orange_table(domain, schema, funcs, cols)
    domain = table.domain

    # removed nulls from domain NOTE: no more nulls
    print "create complete table %.4f" % (time.time() - start)

    start = time.time()
    tables = []
    data = table.to_numpyMA('ac')[0].data
    # find the rows that contain None values in discrete columns
    print "got numpy array %.4f" % (time.time() - start)

    start = time.time()
    for idx, key in enumerate(keys):
      # mask out rows that are not in this partition or have Nulls
      idxs = (keycol == key)
      partition_data = data[idxs,:]
      partition = Orange.data.Table(domain, partition_data)
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












def is_discrete(attr, col):
    if attr in [
      'epochid', 'voltage', 'xloc', 'yloc', 
      'est', 'height', 'width', 'atime', 'v',
      'light', 'humidity', 'age', 
      'finan_icu_days', 'dxage', 'job_count']:
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
  """
  Args:
    cols: list of numpy arrays
  """
  start = time.time()
  for idx in xrange(len(cols)):
    if funcs[idx] == str:
      print cols[idx]
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
  cols = map(np.array, zip(*rows))
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

        

