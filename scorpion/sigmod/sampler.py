import sys
import numpy as np
import Orange
import orange
import math
import random
import pdb
sys.path.extend(['.', '..'])

from collections import Counter, defaultdict
from itertools import chain
from scorpionsql.aggerror import *

from ..settings import *
from ..util import *
from ..split import *
from ..learners.cn2sd.refiner import *

_logger = get_logger()
inf = float('inf')


class Sampler(object):
    def __init__(self, SCORE_ID):
        self.SCORE_ID = SCORE_ID

    def get_sample_size(self, pop, *args, **kwargs):
        return best_sample_size(pop, *args, **kwargs)


    """
    Reuses previously computed samples, if possible
    """
    def __call__(self, table, samp_rate):
        f = Orange.data.filter.ValueFilterContinuous(position=self.SCORE_ID,
                                                     oper=orange.ValueFilter.NotEqual,
                                                     ref=-inf)
        c = Orange.data.filter.Values(domain=table.domain,
                                      conditions=[f],
                                      negate=True)
        scored = table.filter_ref(c)

        pop = len(table)
        samp_size = min(pop, samp_rate * pop + 1)
        if not samp_size:
            return table

        if len(scored) >= samp_size:
            # only use 0.5 from previously computed samples
            p1 = (1. * samp_size) / float(len(scored))
            indices2 = Orange.data.sample.SubsetIndices2(p0=p1)
            idxs = indices2(scored) if len(scored) > 1 else [0]
            scored = scored.select_ref(idxs, negate=True)

        samp_size -= len(scored)
        c.negate = False        
        unscored = table.filter_ref(c)        
        try:
            p0 = float(samp_size) / len(unscored)
            indices2 = Orange.data.sample.SubsetIndices2(p0=p0)
            idxs = indices2(unscored) if len(unscored) > 1 else [0]
            samples = unscored.select_ref(idxs, negate=True)
        except:
            pdb.set_trace()

        samples.extend(scored)
        return samples



