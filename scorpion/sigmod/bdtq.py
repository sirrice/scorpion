import pdb
import sys
import heapq
import bisect
import numpy as np
sys.path.extend(['.', '..'])

from collections import deque
from itertools import chain

from ..util import *

inf = float('inf')
_logger = get_logger()


class CandidateQueue():
  def __init__(self, exp=3, highexp=None):
    self.q = []
    self.exp = exp
    self.highexp = highexp
    if not self.highexp:
      self.highexp = self.exp * 2

  def __len__(self):
    return len(self.q)

  def add(self, tup):
    """
    ARG: 
        tup: (score, estinf, (node, datas, srs, infs))
    """
    _logger.debug("%.4f\t%.4f\t%s" % (tup[0], tup[1], tup[2][0].rule))
    bisect.insort_left(self.q, tup)
    #self.q.append(tup)

  def next(self):
    if not self.q:
      return None
    
    first = self.q[0]
    _logger.debug("first: %.4f\t%.4f\t%s", 
      first[0], first[1], first[2][0].rule)

    if len(self.q) == 1:
      self.total_score = 0
      return self.q.pop()[2:]

    exp = self.exp
    if random.random() < .5:
      exp = self.highexp
    idx = int((1-np.random.power(exp)) * len(self.q))
    return self.q.pop(idx)[2:]
