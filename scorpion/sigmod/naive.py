import time
import pdb
import sys
import Orange
import orange
import heapq
sys.path.extend(['.', '..'])

from itertools import chain
from collections import deque


from ..learners.cn2sd.rule import fill_in_rules
from ..learners.cn2sd.refiner import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *
from ..util import *

from basic import Basic

inf = float('inf')
_logger = get_logger()

class Naive(Basic):
    def __init__(self, *args, **kwargs):
        Basic.__init__(self, *args, **kwargs)
        self.start = time.time()
        self.max_wait = kwargs.get('max_wait', 60*60*2) # 2 hours default
        self.n_rules_checked = 0
        self.stop = False
        self.cs = kwargs.get('cs', None)

        self.checkpoints_per_c = defaultdict(list)


    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)
        if not self.cs:
            self.cs = [self.c]
        
        self.bests_per_c = defaultdict(list)
        for c in self.cs:
            self.bests_per_c[c] = list()
            self.checkpoints_per_c[c] = list()

        self.max_complexity = kwargs.get('max_complexity', self.max_complexity)
        self.granularity = kwargs.get('granularity', self.granularity)
        discretes = [attr for attr in full_table.domain 
                     if attr.name in self.cols and attr.var_type == Orange.feature.Type.Discrete]


        self.all_clauses = map(self.all_unit_clauses, self.cols)
        self.col_to_clauses = dict(zip(self.cols, self.all_clauses))
        base_rule = SDRule(self.full_table, None)            


        start = time.time()
        if discretes:
            max_card = max([len(attr.values) for attr in discretes])
            for m in xrange(1, max_card+1):
                if self.stop:
                    break
                self.foo(base_rule, max_card=m)
        else:
            self.foo(base_rule)
        self.cost = time.time() - start


        # given a rule = c1,..,cn
        # options for each clause c_i
        # 1) extend c_i to the left, to the right
        # 2) append a new clause
        rules = self.bests_per_c[self.cs[0]]
        rules.sort(key=lambda r: r.quality, reverse=True)
        _logger.debug("best\n%s", "\n".join(map(lambda r: '%.4f\t%s' % (r.quality, str(r)), rules)))

        fill_in_rules(rules, full_table, cols=self.cols)
        clusters = [Cluster.from_rule(rule, self.cols, rule.quality) for rule in rules]
        self.all_clusters = clusters

        self.costs = {'cost' : self.cost}
        return clusters

    def max_card_in_conds(self, conds):
        lens = []
        for c in conds:
            attr = self.full_table.domain[c.position]
            if attr.var_type == Orange.feature.Type.Discrete:
                lens.append(len(attr.values))
        return lens and max(lens) or 0

    def foo(self, rule, max_card=None):
        for cols in powerset(self.cols):
            if not cols:
                continue
            if len(cols) > self.max_complexity:
                continue

            _logger.debug(str( cols))
            all_clauses = [self.get_all_clauses(col, max_card) for col in cols]
            for clauses in self.dfs(*all_clauses):
                new_rule = SDRule(rule.data, None, clauses, None)
                
                self.n_rules_checked -= len(clauses)
                if self.n_rules_checked <= 0:
                    diff = time.time() - self.start
                    for c in self.cs:
                        if (not self.checkpoints_per_c or 
                            not self.checkpoints_per_c[c] or 
                            diff - self.checkpoints_per_c[c][-1][0] > 10):
                            bests = self.bests_per_c[c]
                            if bests:
                                best_rule = max(bests, key=lambda r: r.quality)
                                clone = best_rule.clone()
                                clone.quality = best_rule.quality
                                self.checkpoints_per_c[c].append((diff, clone))
                    self.stop = diff > self.max_wait
                    self.n_rules_checked = 100

                    _logger.debug( "%.4f\t%d", time.time() - self.start, self.n_rules_checked)
                    _logger.debug(str( new_rule))

                if self.stop:
                    return

                if max_card is not None and self.max_card_in_conds(clauses) < max_card:
                    continue

                influences = [self.influence(new_rule, c) for c in self.cs]
                for c, influence in zip(self.cs, influences):
                    clone = new_rule.clone()
                    clone.quality = influence
                    clone.__examples__ = None

                    if not valid_number(clone.quality):
                        continue

                    if len(self.bests_per_c[c]) < self.max_bests:
                        heapq.heappush(self.bests_per_c[c], clone)
                    else:
                        heapq.heapreplace(self.bests_per_c[c], clone)
                new_rule.__examples__ = None

    def dfs(self, *iterables, **kwargs):
        q = kwargs.get('q', deque())
        if not iterables:
            yield q
            return

        for o in iterables[0]:
            q.append(o)
            for clauses in self.dfs(*iterables[1:], q=q):
                yield clauses
            q.pop()


