import numpy
import json
import time
import pdb
import sys
import Orange
import orange
import heapq
sys.path.extend(['.', '..'])

from collections import deque
from itertools import chain
from rtree.index import Index as RTree
from rtree.index import Property as RProp
from Orange.classification import svm
from sklearn import svm as sksvm
from scorpionsql.errfunc import ErrTypes

from ..learners.cn2sd.rule import fill_in_rules
from ..learners.cn2sd.refiner import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *
from ..util import *
from ..settings import *

from basic import Basic
from sampler import Sampler
from merger import Merger
from bdtpartitioner import *

inf = 1e10000000
_logger = get_logger()





class SVM(Basic):

    def __init__(self, **kwargs):
        Basic.__init__(self, **kwargs)
        self.all_clusters = []
        self.cost_split = 0.
        self.cost_partition_bad = 0.
        self.cost_partition_good = 0.
        self.cache = None
        self.use_mtuples = kwargs.get('use_mtuples', False)




    def setup_tables(self, full_table, bad_tables, good_tables, **kwargs):
        Basic.setup_tables(self, full_table, bad_tables, good_tables, **kwargs)


        self.SCORE_ID = add_meta_column(
          chain([full_table], bad_tables, good_tables),
          SCORE_VAR
        )

        domain = self.full_table.domain
        attrnames = [attr.name for attr in domain]
        self.cont_dists = dict(zip(attrnames, Orange.statistics.basic.Domain(self.full_table)))
        self.disc_dists = dict(zip(attrnames, Orange.statistics.distribution.Domain(self.full_table)))

        self.bad_states = [ef.state(t) for ef, t in zip(self.bad_err_funcs, self.bad_tables)]
        self.good_states = [ef.state(t) for ef, t in zip(self.good_err_funcs, self.good_tables)]

        self.good_table = Orange.data.Table(domain, good_tables[0])
        for gt in good_tables[1:]:
          self.good_table.extend(gt)

        self.bad_table = Orange.data.Table(domain, bad_tables[0])
        for bt in bad_tables[1:]:
          self.bad_table.extend(bt)



    def learn(self, data, nu=0.01, kernel=svm.kernels.RBF, svmtype=svm.SVMLearner.OneClass):
        learner = svm.SVMLearner(svm_type=svmtype, kernel_type=kernel, nu=nu)
        classifier = learner(data)
        return classifier


    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)


        gooddata = numpy.array( [
          [float(v.value != 'None' and v.value or -1) for v in row] 
          for row in self.good_table
        ])

        allbaddata = numpy.array( [
          [float(v.value != 'None' and v.value or -1) for v in row] 
          for row in self.bad_table
        ])

        learner = sksvm.OneClassSVM(kernel='rbf', nu=0.01)
        sample = random.sample(gooddata, 8000)
        clf = learner.fit(sample)
        labels = clf.predict(allbaddata)
        bad = allbaddata[labels == -1]
        import pdb
        pdb.set_trace()

 
        classifier = self.learn(self.good_table)
        labels = map(int, map(classifier, self.bad_table))
        badrows = [self.bad_table[idx] for idx, label in enumerate(labels) if label == -1]
        sv = classifier.support_vectors
        datas = [ ['%4f' % float(v.value != 'None' and v.value or 0) for v in row] for row in sv]
        import pdb
        pdb.set_trace()

        # create a cluster
        return None



        clusters = self.get_partitions(full_table, bad_tables, good_tables, **kwargs)
        self.all_clusters = clusters

        start = time.time()
        _logger.debug('merging')
        self.final_clusters = self.merge(clusters)        
        filt = lambda c: not math.isinf(c.error) and not math.isnan(c.error)
        self.final_clusters = filter(filt, self.final_clusters)
        self.cost_merge = time.time() - start


        self.costs.update( {'cost_partition_bad' : self.cost_partition_bad,
                'cost_partition_good' : self.cost_partition_good,
                'cost_split' : self.cost_split,
                'cost_merge' : self.cost_merge})
        
        return self.final_clusters

