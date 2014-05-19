import time
import pdb
import sys
import Orange
import orange
sys.path.extend(['.', '..'])


from learners.cn2sd.rule import fill_in_rules
from score import QuadScoreSample7
from bottomup.bounding_box import *
from bottomup.cluster import *
from merger.merger import Merger
from merger.reexec import ReexecMerger
from util import *
from sampler import SampleDecisionTree







class TopK(object):
    def __init__(self, **kwargs):
        self.aggerr = kwargs.get('aggerr', None)
        self.cols = list(self.aggerr.agg.cols)
        self.err_func = kwargs.get('err_func', self.aggerr.error_func.clone())
        self.merger = None
        self.params = {}
        

        self.scorer_cost = 0.
        self.merge_cost = 0.

        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        self.cols = kwargs.get('cols', self.cols)
        self.params.update(kwargs)


        

    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        Topdown will use all columns to construct rules
        """
        self.set_params(**kwargs)

        self.bad_tables = bad_tables
        self.good_tables = good_tables

        # allocate and setup error functions
        self.bad_err_funcs = [self.err_func.clone() for t in bad_tables]
        self.good_err_funcs = [self.err_func.clone() for t in good_tables]

        for ef, t in zip(chain(self.bad_err_funcs,self.good_err_funcs),
                         chain(bad_tables, good_tables)):
            ef.setup(t)



        
            


        self.err_func = self.bad_err_funcs[0]
        fill_in_rules(rules, full_table, cols=self.cols)
        self.all_clusters = [Cluster.from_rule(r, self.cols) for r in rules]
