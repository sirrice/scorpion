import os
import time
import json
import logging
import sys
import random
import math
import numpy as np

from collections import defaultdict
from itertools import *
from Orange.data.filter import *

"""
Super generic functions that can be used across modules
Global values that are used in every module
"""
LINESTYLES = ['-', '--o']#, '--', '-.', ',', 'o', 'v', '^', '>', '1', '*', ':']
COLORS = ['#d62728', '#1f77b4', '#9467bd', '#ff7f0e', '#2ca02c',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#000000', 'red', 'orange', 'blue', 'purple', 'grey',
          'green', 'yellow', 'purple']

NAN = float('nan')
INF = float('inf')





class GlobalLogger(object):
    def __init__(self):
        self._logger = None
    
    def __call__(self, fname='/tmp/scorpion.log', flevel=logging.DEBUG, plevel=logging.WARNING):
        if self._logger:
            # set levels
            return self._logger

        self._logger = logging.getLogger()
        self._logger.setLevel(logging.DEBUG)

        #formatter = logging.Formatter('%(asctime)s - %(lineno)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(lineno)s - %(levelname)s - %(message)s')

        fname = os.path.abspath(fname)
        fh = logging.FileHandler(fname)
        fh.setLevel(flevel)
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

        ph = logging.StreamHandler(sys.stdout)
        ph.setLevel(plevel)
        ph.setFormatter(formatter)
        self._logger.addHandler(ph)
        return self._logger

try:
  get_logger = GlobalLogger()
except:
  try:
    get_logger = GlobalLogger('./scorpion.log')
  except:
    get_logger = logging.getLogger()
    get_logger.setLevel(logging.DEBUG)




def instrument(fn):
  func_name = fn.__name__
  nargs = fn.func_code.co_argcount
  def w(self, *args, **kwargs):
    start = time.time()
    if len(args) == nargs:
      ret = fn(*args, **kwargs)
    else:
      ret = fn(self, *args, **kwargs)

    if hasattr(self, 'stats'):
      if func_name not in self.stats:
          self.stats[func_name] = [0, 0]
      self.stats[func_name][0] += (time.time() - start)
      self.stats[func_name][1] += 1
    return ret
  return w


# JSON Encoder
class ScorpionEncoder(json.JSONEncoder):
  def default(self, o):
    if isinstance(o, float):
      if o == INF:
        return 1e100
      elif o == -INF:
        return -1e100

    if hasattr(o, 'isoformat'):
      s =  o.isoformat()
      if not s.endswith("Z"):
        s += 'Z'
      return s
    return super(ScorpionEncoder, self).default(o)

def mkfmt(arr, sep='\t', mapping=None):
  """
  Given a list of object types, returns a tab separated formatting str
  """
  if mapping is None:
    mapping = [(float, '%.4f'), (int, '%d'),  (object, '%s')]
  fmt = []
  for v in arr:
    for t,f in mapping:
      if isinstance(v,t):
        fmt.append(f)
        break

  return sep.join(fmt)

def pluck(o, keys=[]):
  return [o.get(key, None) for key in keys]

def pick(l, key):
  return [v[key] for v in l]


def block_iter(l, nblocks=2):
  """
  partitions l into nblocks blocks and returns generator over each block
  @param l list
  @param nblocks number of blocks to partition l into
  """
  blocksize = int(math.ceil(len(l) / float(nblocks)))
  i = 0
  while i < len(l):
    yield l[i:i+blocksize]
    i += blocksize      


def rm_dups(seq, key=lambda e:e):
  """
  returns a list of items with duplicates removed
  @note from http://www.peterbe.com/plog/uniqifiers-benchmark
  """
  seen = set()
  seen_add = seen.add
  return [ x for x in seq if key(x) not in seen and not seen_add(key(x))]


def groupby(iterable, key=lambda: 1):
  ret = defaultdict(list)
  for i in iterable:
    ret[key(i)].append(i)
  return ret.iteritems()

def powerset(iterable):
  "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
  s = list(iterable)
  return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def valid_number(v):
  return v is not None and v != -INF and not(math.isnan(v)) and v != INF



