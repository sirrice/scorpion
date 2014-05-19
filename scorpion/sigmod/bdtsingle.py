import time
import pdb
import sys
import Orange
import orange
import heapq
sys.path.extend(['.', '..'])

from itertools import chain


from ..learners.cn2sd.rule import fill_in_rules
from ..learners.cn2sd.refiner import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *
from ..util import *
from ..settings import *

from merger.merger import Merger
from merger.reexec import ReexecMerger
from basic import Basic
from sampler import Sampler

inf = 1e10000000





class Splits(object):
    def __init__(self, influence):
        self.splits = []
        self.influence = influence

    def __stop__(self):
        return len(self.splits) == 0
    stop = property(__stop__)

    def add_split(self, s):
        self.splits.append(s)

class Split(object):
    def __init__(self, attr, *args):
        self.attr = attr
        self.stop = attr is None
        if self.stop:
            self.influence = args[0]
        else:
            self.rules = args[0]
            self.score = args[1]



class BDTPartitioner(Basic):
    def set_params(self, **kwargs):
        Basic.set_params(self, **kwargs)

        self.p = kwargs.get('p', 0.6)
        self.tau = kwargs.get('tau', [0.001, 0.05])
        self.epsilon = kwargs.get('epsilon', 0.005)
        self.min_pts = 5
        self.samp_rate = 1.
        self.SCORE_ID = kwargs['SCORE_ID']
        self.inf_bounds = [inf, -inf]

        self.sampler = Sampler(self.SCORE_ID)


    def setup_table(self, table):
        self.table = table
        self.err_func.setup(table)

   
    def __call__(self, table, **kwargs):
        self.setup_table(table)

        base_rule = SDRule(table, None)
        node = self.grow(base_rule, self.samp_rate)

        return node.leaves


    def grow(self, rule, samp_rate):
        data = rule.examples
        sample = self.sample(data, samp_rate)
        node = Node(rule)

        if self.should_stop(sample):
            node.set_score(self.estimate_inf(sample))
            return node

        scores = []
        for attr, new_rules in self.child_rules(rule):
            score = self.get_score(new_rules, sample)
            scores.append(new_rules, score)
        
        if not scores:
            node.set_score(self.estimate_inf(sample))
            return node

        new_rules, score = min(scores, key=lambda pair: p[1])
        if len(new_rules) == 1:
            return self.grow(new_rules, samp_rate)


        new_sample_rates = self.update_sample_rates(new_rules, samp_rate)
        for new_rule, new_samp_rate in zip(new_rules, new_sample_rates):
            child = self.grow(new_rule, new_samp_rate)
            if child.n:
                node.add_child(child)

        return node

    def estimate_inf(self, sample):
        return np.mean(map(self.influence, sample))
        
    def get_score(self, new_rules, sample):
        new_samples = map(lambda r: r.filter_table(sample), new_rules)
        score = sum(map(self.compute_score, new_samples))
        return score

    def should_stop(self, data):
        if len(data) < self.min_pts:
            return True

        infmax = max(map(self.influence, data))
        thresh = self.compute_threshold(infmax, self.inf_bounds[0], self.inf_bounds[1])
        score = self.compute_score(data)
        return score < thresh

    def compute_score(self, data):
        return np.std(map(self.influence, data))


    def child_rules(self, rule, attrs=None):
        attrs = attrs or self.cols
        next_rules = defaultdict(list)
        refiner = BeamRefiner(attrs=attrs, fanout=2)
        for attr, new_rule in refiner(rule):
            next_rules[attr].append(new_rule)
        return next_rules.items()


    def sample(self, data, samp_rate):
        return self.sampler(data, samp_rate)


    def update_sample_rates(self, rules, samp_rate):
        influences, counts = [], []
        for rule in rules:
            influence = sum(map(self.influence, rule.examples))
            influences.append(influence)
            counts.append(float(len(rule.examples)))

        total_inf = sum(influences)
        total_count = sum(counts)
        samp_rates = []
        for influence, count in zip(influences, counts):
            nsr = samp_rate * (influence/total_inf) * (count/total_count)
            samp_rates.append(nsr)

        return samp_rates



    def compute_threshold(self, infmax, infl, infu):
        tau, p = self.tau, self.p
        s = (tau[0] - tau[1]) / ((1-p)*infu - p * infl)
        w = tau[0] + s*(infu - infmax)
        w = min(tau[1], w)
        return w * (infu - infl)       

    def influence(self, row):
        if row[self.SCORE_ID].value == -inf:
            influence = self.err_func((row,))
            row[self.SCORE_ID] = influence
            self.inf_bounds[0] = min(influence, self.inf_bounds[0])
            self.inf_bounds[1] = max(influence, self.inf_bounds[1])
        return row[self.SCORE_ID].value


