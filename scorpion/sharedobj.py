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


from db import *
from sql import *
from util import *
from score import *
from classify import *
from aggerror import *
from settings import *
from sqlparser import *




def create_sharedobj(dbname, sql, badresults, goodresults, errtype, bad_tuple_ids={}):
  from arch import get_provenance
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

        # created by server to track status of scorpion
        # processing
        # should be set when creating SharedObj
        self.status = None   
        
        
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

    def update_status(self, s):
      if self.status:
        self.status.update_status(s)

    rules_schema = property(get_rules_schema)
    bad_tuple_ids = property(get_bad_tuple_ids)
    sql = property(lambda self: str(self.parsed))
    prettify_sql = property(lambda self: self.parsed.prettify())
    filter = property(lambda self: self.parsed.get_filter_qobj())






