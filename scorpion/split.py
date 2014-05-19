import numpy as np
import Orange, orange
import math

from settings import *
from util import *

class Split(object):
    def __init__(self, table, attr):
        self.table = table
        self.attr = attr
        self.filters = []

        self.set_filters()

    def set_filters(self):
        pass

    def __call__(self, table=None):
        if table is None:
            table = self.table
        return map(table.filter_ref, self.filters)

    
class DiscreteSplit(Split):
    def __init__(self, table, attr, pos, neg):
        self.posvals = pos
        self.negvals = neg

        super(DiscreteSplit, self).__init__(table, attr)
        

    def set_filters(self):
        pos = self.table.domain.index(self.attr)
        f1 = Orange.data.filter.ValueFilterDiscrete(position=pos,
                                                    values=self.posvals)
        f2 = Orange.data.filter.ValueFilterDiscrete(position=pos,
                                                    values=self.negvals)
        c1 = Orange.data.filter.Values(domain=self.table.domain)
        c1.conditions.append(f1)
        c2 = Orange.data.filter.Values(domain=self.table.domain)
        c2.conditions.append(f2)

        self.filters = [c1, c2]


    def __str__(self):
        vals = map(lambda v: v.value, self.posvals)
        return ' '.join(map(str, ['discretesplit', len(self.table), self.attr.name, str(vals)]))
        
class ContSplit(Split):
    def __init__(self, table, attr, midv, minv, maxv):
        self.midv = midv
        self.minv = minv
        self.maxv = maxv

        super(ContSplit, self).__init__(table, attr)        

    def set_filters(self):
        pos = self.table.domain.index(self.attr)
        f1 = Orange.data.filter.ValueFilterContinuous(position=pos,
                                          oper=orange.ValueFilter.LessEqual,
                                          ref=self.midv)
        f2 = Orange.data.filter.ValueFilterContinuous(position=pos,
                                          oper=orange.ValueFilter.Greater,
                                          ref=self.midv)
        c1 = Orange.data.filter.Values(domain=self.table.domain)
        c1.conditions.append(f1)
        c2 = Orange.data.filter.Values(domain=self.table.domain)
        c2.conditions.append(f2)

        self.filters = [c1, c2]

    def __str__(self):
        return ' '.join(map(str, ['continuoussplit', len(self.table), self.attr.name, self.midv]))

