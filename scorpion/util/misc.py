import os
import time
import json
import logging
import sys
import random
import math
import matplotlib
import numpy as np

from itertools import *
from Orange.data.filter import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

nan = float('nan')
inf = float('inf')

matplotlib.use("Agg")

"""
Super generic functions that can be used across modules
Global values that are used in every module
"""
LINESTYLES = ['-', '--o']#, '--', '-.', ',', 'o', 'v', '^', '>', '1', '*', ':']
COLORS = ['#d62728', '#1f77b4', '#9467bd', '#ff7f0e', '#2ca02c',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#000000', 'red', 'orange', 'blue', 'purple', 'grey',
          'green', 'yellow', 'purple']
#COLORS = ['%s%s' % (color, m) for color in __colors__ for m in __markers__]
NAN = float('nan')

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

get_logger = GlobalLogger()


def instrument(fn):
    func_name = fn.__name__
    def w(self, *args, **kwargs):
        start = time.time()
        ret = fn(self, *args, **kwargs)

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
      if o == float('inf'):
        return 1e100
      elif o == float('-inf'):
        return -1e100

    if hasattr(o, 'isoformat'):
      s =  o.isoformat()
      if not s.endswith("Z"):
        s += 'Z'
      return s
    return super(ScorpionEncoder, self).default(o)



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




def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def valid_number(v):
  return v is not None and v != -inf and not(math.isnan(v)) and v != inf


class LinearFit(object):

    def __init__(self, data=None):
        self.x2 = 0.
        self.y2 = 0.
        self.xm = 0.
        self.ym = 0.
        self.xy = 0.
        self.n = 0.
        self.ssxx = 0.
        self.ssyy = 0.
        self.ssxy = 0.
        self.b = 0.
        self.a = 0.
        self.r2 = 0.
        self.p = 0.
        if data is not None:
            self(data)

    def __call__(self, xs, ys):
        #pts = data
        #xs, ys = pts[:,0], pts[:,1]
        n = len(xs)
        self.x2 = np.dot(xs, xs)
        self.y2 = np.dot(ys, ys)
        self.xm = np.mean(xs)
        self.ym = np.mean(ys)
        self.xy = np.dot(xs, ys)
        self.n  = n

        self.ssxx = self.x2 - self.n*(self.xm**2)
        self.ssyy = self.y2 - self.n*self.ym*self.ym
        self.ssxy = self.xy - self.n*self.xm*self.ym

        self.b = self.ssxy / self.ssxx # slope
        self.a = self.ym - self.b*self.xm  # intercept
        self.r2 = self.ssxy**2 / (self.ssxx*self.ssyy)   # correlation coefficient
        self.p = self.ssxy / math.sqrt(self.ssxx*self.ssyy)  # pearson's = sqrt(r2)
        return self.a, self.b, self.r2, self.p


    def delta(self, add=None, rm=None):
        if add is None and rm is None:
            return self.a, self.b, self.r2, self.p
        
        x2, y2 = self.x2, self.y2
        xm, ym, xy = self.xm, self.ym, self.xy
        n = self.n

        if add is not None:
            #xs, ys = add[:,0], add[:,1]
            xs, ys = tuple(add)
            if n == len(xs):
                return 0,0,0,0
            
            x2 += np.dot(xs, xs)
            y2 += np.dot(ys, ys)
            xm = (xm * n + xs.sum()) / (n + len(xs))
            ym = (ym * n + ys.sum()) / (n + len(ys))
            xy += np.dot(xs, ys)
            n += len(xs)
        if rm is not None:
            #xs, ys = rm[:,0], rm[:,1]
            xs, ys = tuple(rm)
            if n == len(xs):
                return 0,0,0,0
            
            x2 -= np.dot(xs, xs)
            y2 -= np.dot(ys, ys)
            xm = (xm * n - xs.sum()) / (n - len(xs))
            ym = (ym * n - ys.sum()) / (n - len(ys))
            xy -= np.dot(xs, ys)
            n -= len(xs)

       
        ssxx = x2 - (n*xm*xm)
        ssyy = y2 - (n*ym*ym)
        ssxy = xy - (n*xm*ym)

        if n == 0 or ssxx <= 0 or ssyy <= 0:
            return NAN, NAN, NAN, NAN
        
        b = ssxy / ssxx # slope
        a = ym - b*xm   # intercept
        r2 = ssxy**2 / (ssxx*ssyy)   # correlation coefficient
        p = ssxy / math.sqrt(ssxx*ssyy)  # pearson's = sqrt(r2)
        return a, b, r2, p


    def linear(self, add=None, rm=None):
        a,b,r2,p = self.delta(add=add, rm=rm)
        return a, b, r2

    def corr(self, add=None, rm=None):
        a,b,r2,p = self.delta(add=add, rm=rm)
        return p

    def pearson(self, **kwargs):
        return self.corr(**kwargs)

if __name__ == '__main__':
    


    
    import random
    random.seed(0)
    pts = []
    for i in xrange(1000):
        pts.append( (i, i + random.random()) )
    pts = np.array(pts)

    
    lf = LinearFit()
    print lf.corr(pts)
    print lf.corr(pts)    

