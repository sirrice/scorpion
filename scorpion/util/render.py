import pdb
import sys
import random
import time
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle as Rect
from matplotlib import cm 


from scorpion.util.rangeutil import *

matplotlib.use("Agg")

class BaseRenderer(object):
  def __init__(self, fname, *args, **kwargs):
    self.pp = PdfPages(fname)
    self.clusters = []
    self.fig = plt.figure(figsize=(8, 8))
    self.sub = self.fig.add_subplot(111)
    self.xbound = None
    self.ybound = None

  def new_page(self):
    if (self.fig):
      self.fig.savefig(self.pp, format='pdf')
    self.sub.cla()

  def close(self):
    if self.fig:
      plt.savefig(self.pp, format='pdf')
    self.pp.close()
    self.pp = None
    self.fig = self.sub = None

  def set_lims(self, xbound=None, ybound=None):
    xbound = xbound or self.xbound
    ybound = ybound or self.ybound
    if xbound:
      xbound = r_expand(xbound)
      ybound = r_expand(ybound)
      self.sub.set_xlim(*xbound)
      self.sub.set_ylim(*ybound)


  def plot_title(self, title):
    self.sub.set_title(title)
    self.sub.title.set_fontsize(6)

  def set_title(self, title):
    self.sub.set_title(title)
    self.sub.title.set_fontsize(6)


class ClusterRenderer(BaseRenderer):
  def __init__(self, *args, **kwargs):
    BaseRenderer.__init__(self, *args, **kwargs)
    self.xbound = self.ybound = None

  def new_page(self):
    BaseRenderer.new_page(self)
  
  def transform_box(self, x, y):
    return x, y

  def plot_clusters(self, clusters, cols=None, color=None, alpha=None):
    self.clusters = clusters
    errors = [c.error for c in clusters]
    errors = np.array(errors)
    mean, std = np.mean(errors), np.std(errors)
    if std == 0: std = 1
    errors = (errors - mean) / std

    if not cols:
      cols = [0, 1]
    if isinstance(cols[0], basestring):
      cols = map(clusters.cols.index, cols)

    for idx, cluster in enumerate(clusters):
      tup = tuple(map(list, zip(*cluster.bbox)))
      x, y = tup[cols[0]], tup[cols[1]]
      x, y = self.transform_box(x, y)
      if not self.xbound:
        self.xbound = [x[0], x[1]]
        self.ybound = [y[0], y[1]]
      else:
        self.xbound = r_union(self.xbound, x)
        self.ybound = r_union(self.ybound, y)

      a = alpha or min(1, max(0.1, errors[idx]))
      c = color or cm.jet(errors[idx])
      r = Rect((x[0], y[0]), x[1]-x[0], y[1]-y[0], alpha=a, ec=c, fill=False, lw=1.5)
      self.sub.add_patch(r)

    self.set_lims()

  def plot_tuples(self, tuples):
    self.tuples = tuples
    cols = zip(*tuples)
    xs, ys, cs = cols[0], cols[1], cols[-1]
    self.sub.scatter(ys, xs, c=cs, alpha=0.5, lw=0)

    if not self.xbound:
      self.xbound = [min(xs), max(xs)]
      self.ybound = [min(ys), max(ys)]
    self.xbound = r_union(self.xbound, [min(xs), max(xs)])
    self.ybound = r_union(self.ybound, [min(ys), max(ys)])
    self.set_lims()


class JitteredClusterRenderer(ClusterRenderer):
  def transform_box(self, x, y):
    f = lambda v: (random.random()-0.5)*2+v
    x = [f(x[0]), f(x[1])]
    y = [f(y[0]), f(y[1])]
    return x, y


class InfRenderer(BaseRenderer):
  def __init__(self, *args, **kwargs):
    BaseRenderer.__init__(self, *args, **kwargs)
    self.c_range = kwargs.get('c_range', [0, 1])
    self.xbound = None
    self.ybound = None


  def new_page(self):
    BaseRenderer.new_page(self)

  def plot_points(self, xs, ys, color='grey', alpha=0.3):
    self.sub.plot(xs, ys, color=color, alpha=alpha)

    ys = filter(lambda v: abs(v) != float('inf'), ys)
    if len(ys): 
      if not self.ybound:
        self.xbound = [min(xs), max(xs)]
        self.ybound = [ min(ys), max(ys) ]
      else:
        inf_range = [min(ys), max(ys)]
        self.xbound = [min(self.xbound[0], min(xs)), max(self.xbound[1], max(xs))]
        self.ybound = [min(self.ybound[0], min(ys)), max(self.ybound[1], max(ys))]

    self.set_lims()



  def plot_active_inf_curves(self, clusters, color='red', alpha=0.3):
    alpha = alpha or 0.3
    color = color or 'red'
    for c in clusters:
      xs = (np.arange(100) / 100. * r_vol(c.c_range)) + c.c_range[0]
      ys = map(c.inf_func, xs)
      self.sub.plot(xs, ys, color=color, alpha=alpha)

      ys = filter(lambda v: abs(v) != float('inf'), ys)
      if not ys: continue
      if not self.ybound:
        self.xbound = c.c_range
        self.ybound = [ min(ys), max(ys) ]
      else:
        inf_range = [min(ys), max(ys)]
        self.xbound = [min(zip(self.xbound, c.c_range)[0]),  max(zip(self.xbound, c.c_range)[1])]
        self.ybound = [min(zip(self.ybound, inf_range)[0]),  max(zip(self.ybound, inf_range)[1])]

    self.set_lims()


  def plot_inf_curves(self, clusters, c_range=None, color=None, alpha=None):
    """
    By default, renders each cluster's inf curve and colors the valid 
    cluster.c_range segment as red.

    Optionally, can render all clusters the same color and alpha based on
    passed-in args
    """
    if not c_range: c_range = self.c_range
    xs = (np.arange(100) / 100. * r_vol(c_range)) + c_range[0]

    for c in clusters:
      alpha = alpha or 0.3
      color = color or 'grey'
      ys = map(c.inf_func, xs)
      self.sub.plot(xs, ys, color=color, alpha=alpha)

      ys = filter(lambda v: abs(v) != float('inf'), ys)
      if not ys: continue
      if not self.ybound:
        self.xbound = c_range
        self.ybound = [ min(ys), max(ys) ]
      else:
        inf_range = [min(ys), max(ys)]
        self.xbound = [min(zip(self.xbound, c_range)[0]),  max(zip(self.xbound, c_range)[1])]
        self.ybound = [min(zip(self.ybound, inf_range)[0]),  max(zip(self.ybound, inf_range)[1])]



    self.set_lims()
