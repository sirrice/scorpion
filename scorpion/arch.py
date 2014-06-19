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

from scorpionsql.db import *
from scorpionsql.sql import *
from scorpionsql.aggerror import *
from scorpionsql.sqlparser import *

from sharedobj import *
from util import *
from settings import *



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




def is_discrete(attr, col):
  """
  Determines if a column is discrete or continuous.

  A huge number of special cases so we don't need to solve this 
  problem for the paper
  """
  if attr in [
    'epochid', 'voltage', 'xloc', 'yloc', 
    'est', 'height', 'width', 'atime', 'v',
    'light', 'humidity', 
    'finan_icu_days', 'dxage', 'job_count', 'b']:
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


def orange_schema_funcs(cols, attrs):
  """
  Args
    cols:  values of table in columnar format
           list of columns. each column is a list of values
    attrs: list of attribute names for each column
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


