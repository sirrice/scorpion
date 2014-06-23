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
from scorpionsql.sqlparser import *
from scorpionsql.aggerror import *

from util import *
from score import *
from classify import *
from settings import *




class SharedObj(object):

  def __init__(
      self, db,
      errors=[],
      goodkeys={},
      ignore_attrs=[],
      schema=[],
      dbname=None,
      parsed=None,
      params=[],
      **kwargs):
    if not db and not dbname:
        raise "SharedObj requires a database connection!"
    self.db = db or connect(dbname)
    dbname = dbname or str(self.db.url).split("/")[-1]
    self.monetdb = connect(dbname, engine='monet')
    self.dbname = dbname
    self.parsed = parsed
    self.params = params   # parameters for parsed SQL object
    self.errors = errors
    self.goodkeys = goodkeys or {}
    self.schema = schema or db_schema(db, self.parsed.tables[0])
    # aggregate type -> {groupby key -> ids of "bad" tuples}
    self.ignore_attrs = ignore_attrs
    self.merged_tables = {}
    self.rules = {}
    self.top_k_rules = {}
    self.clauses = {}
    self.c = kwargs.get('c', 0.3)

    # created by server to track status of scorpion
    # processing
    # should be set when creating SharedObj
    self.status = None   
    
    if not self.parsed:
      raise Error("expected a parsed SQL object!")

    if len(self.parsed.fr) > 1:
        # XXX: only support single table queries
        raise "Don't support joins yet!"

      
  def clone(self):
    return SharedObj(
      self.db, 
      parsed=self.parsed,
      dbname=self.dbname,
      errors=self.errors,
      goodkeys=self.goodkeys,
      ignore_attrs=self.ignore_attrs,
      schema=self.schema,
      params=self.params
    )

  def get_tuples(self, keys, attrs=None):
    try:
        if keys is None or not len(list(keys)):
            return []
    except:
        pass
    attrs = attrs or self.rules_schema
    return [list(row) for row in self.get_filter_rows(keys=keys, attrs=attrs)]
    

  
  def get_filter_rows(self, keys=None, attrs=None, where=None):
    """
    Need to deal with keys and such outside of function
    """
    qobj = self.parsed.get_filter_qobj(keys=keys)
    if attrs:
        qobj.select = Select(attrs)
    if where:
        qobj.where.append(where)

    params = list(self.params)
    if keys:
      params.append(tuple(list(keys)))

    return query(self.db, str(qobj), [params])


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

  def update_status(self, s):
    if self.status:
      self.status.update_status(s)

  def update_rules(self, label, rules):
    if self.status:
      self.status.update_rules(label, rules)

  attrnames = property(lambda self: self.schema.keys())
  rules_schema = property(get_rules_schema)
  sql = property(lambda self: str(self.parsed))
  prettify_sql = property(lambda self: self.parsed.prettify())
  filter = property(lambda self: self.parsed.get_filter_qobj())






def create_sharedobj(dbname, sql, badresults, goodresults, errtype):
  from arch import get_provenance
  db = connect(dbname)
  parsed = parse_sql(sql)
  obj = SharedObj(db, parsed=parsed, dbname=dbname)


  qobj = obj.parsed
  nonagg = qobj.select.nonaggs[0]
  xcol = nonagg.cols[0]
  col_type = db_type(db, qobj.fr, xcol)

  # assumes every aggregate has the same bad keys
  badresults = extract_agg_vals(badresults, col_type)
  goodresults = extract_agg_vals(goodresults)

  errors = []
  for agg in qobj.select.aggregates:
    aggerr = AggErr(agg, badresults, 20, errtype, {'erreq' : None})
    errors.append(aggerr)

    label = agg.shortname
    obj.goodkeys[label] = goodresults
  obj.errors = errors

  table = get_provenance(obj, obj.errors[0].agg.cols, obj.errors[0].keys)
  return obj, table



