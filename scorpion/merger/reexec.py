import math
import pdb
import random
import numpy as np
import sys
import time
sys.path.extend(['.', '..'])

from itertools import chain
from collections import defaultdict
from scipy.spatial import KDTree
from rtree.index import Index as RTree
from rtree.index import Property as RProp
from operator import mul, and_

from util import rm_attr_from_domain, get_logger
from bottomup.bounding_box import *
from bottomup.cluster import *
from merger import Merger

_logger = get_logger()



class ReexecMerger(Merger):
    """
    Computes new cluster error by rerunning error function on new cluster
    """

    def __init__(self, **kwargs):
        self.full_table = None
        self.bad_tables = []
        self.good_tables = []
        self.bad_err_funcs = []
        self.good_err_funcs = []
        self.err_func = None
        self.cols = None

        Merger.__init__(self, **kwargs)


    def set_params(self, **kwargs):
        print kwargs.keys()
        self.cols = kwargs.get('cols', self.cols)
        self.full_table = kwargs.get('full_table', self.full_table)
        self.bad_tables  = kwargs.get('bad_tables', self.bad_tables)
        self.good_tables = kwargs.get('good_tables', self.good_tables)
        self.bad_err_funcs = kwargs.get('bad_err_funcs', self.bad_err_funcs)
        self.good_err_funcs = kwargs.get('good_err_funcs', self.good_err_funcs)
        assert self.bad_tables is not None, "table not set"
        assert self.bad_err_funcs is not None, "error func not set"

        self.table = self.full_table

        domain = self.full_table.domain
        attrnames = [attr.name for attr in domain]
        self.cont_dists = dict(zip(attrnames, Orange.statistics.basic.Domain(self.full_table)))
        self.disc_dists = dict(zip(attrnames, Orange.statistics.distribution.Domain(self.full_table)))

        Merger.set_params(self, **kwargs)

    def compute_stat(self, rule, err_funcs, tables):
        datas = map(rule.filter_table, tables)
        infs = []
        for ef, data in zip(err_funcs, datas):
            arr = data.to_numpyMA('ac')[0]
            inf = ef(arr.data)
            infs.append(inf)
        return infs, map(len, datas)

    def compute_error(self, rule):
        bad_stats, bad_counts = self.compute_stat(rule, self.bad_err_funcs, self.bad_tables)
        good_stats, good_counts = self.compute_stat(rule, self.good_err_funcs, self.good_tables)
        good_stats = map(abs, good_stats)
        good_skip = False

        # compute diff
        f = lambda counts: math.sqrt(np.mean(counts))
        bad_stat = np.mean(bad_stats)
        good_stat = good_stats and max(good_stats) or 0
        npts = max(bad_counts + good_counts)
        mean_pts = np.mean(bad_counts + good_counts)
        bad_inf = bad_stat / (1+f(bad_counts))
        good_inf = good_stat / (1+f(good_counts))
        inf =  bad_inf - good_inf
        return inf
 


    def merge(self, c1, c2, intersecting):
        newcluster = Cluster.merge(c1, c2, intersecting, self.point_volume)
        if not newcluster:
            pdb.set_trace()
        rule = newcluster.to_rule(self.full_table, cont_dists=self.cont_dists, disc_dists=self.disc_dists)
        
        if len(rule.examples) == 0:
            return None
        else:
            error = self.compute_error(rule)            
#            error = self.err_func(rule.examples.to_numpyMA('ac')[0]) / len(rule.examples)

        newcluster.error = error
        newcluster.npts = len(rule.examples)
        #if '9572.00 <= epochid < 9601.00 and 0.00 <= light < 0.00 and 24.29 <= humidity < 27.43 and moteid = 15' in str(rule):
        #    pdb.set_trace()
        print len(rule.examples), '\t', c1.error, '\t', c2.error, '\t', error, '\t', str(rule)
        return newcluster

        
