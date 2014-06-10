import orange, Orange
import sys, math, heapq
import pdb
import numpy as np
import time

from collections import Counter

from ...util import ids_filter, block_iter

from rule import *

 
        
class BeamRefiner(object):
    def __init__(self, attrs=None, **kwargs):
        self.cost = 0.
        self.fanout = kwargs.get('fanout', 2)
        self.bdists = None
        self.ddists = None
        self.attrs = attrs or []
        self.bad_rules = set()

    def set_attributes(self, attrs=[]):
        self.attrs = attrs or []

    def add_bad_rule(self, rule):
        self.bad_rules.add( rule )

    def skip_attribute(self, attr):
        if self.attrs:
            if (attr not in self.attrs and attr.name not in self.attrs):
                return True
        return False

    def next_attributes(self, rule):
        """
        returns a generator distributions of candidate attributes to add to rule
        only returns attributes with position >= maximum attribute in current rule
        (by position in domain)
        """
        ddists = Orange.statistics.distribution.Domain(rule.examples)
        self.ddists = ddists
        bdists = self.bdists or Orange.statistics.basic.Domain(rule.examples)
        self.bdists = bdists
        attrs = rule.examples.domain
        useds = [False] * len(attrs)
        for vf in rule.filter.conditions:
            useds[vf.position] = True

        # get rule's maximum attribute (in terms of position)        
        ignore_attrs = []

        for pos, (d,b,a,u) in enumerate(zip(ddists, bdists, attrs, useds)):
            if a in ignore_attrs:
                continue            
            b = Orange.statistics.distribution.Distribution(
                a, rule.examples)
            b = Orange.statistics.distribution.Continuous(b)
            yield pos,d,b,a,u

    def construct_new_rule(self, rule, idx, ddist, bdist, attr, used, negate):
        if attr.var_type == Orange.feature.Type.Discrete:
            matches = filter(lambda c: idx == c.position,
                             rule.filter.conditions)
            if matches:
                keys = [attr.values[int(v)] for v in matches[0].values]
            else:
                keys = [k for k,v in ddist.items() if v > 0]

            if len(keys) <= 1:
                return
            

            fanout = len(keys) / 5 if len(keys) / self.fanout > 5 else self.fanout
            fanout = self.fanout
            #fanout = 60
            fanout = min(fanout, len(keys))#fanout)
            #fanout = len(keys)
            
            for keyblock in block_iter(keys, fanout):
                new_rule = rule.cloneAndAddCondition(attr, keyblock, used=used, negate=negate)
                new_rule.parent_rule = rule
                if new_rule not in self.bad_rules:
                    yield new_rule


        else:
            minv = bdist.percentile(0)
            maxv = bdist.percentile(100)
            avgv = bdist.percentile(50)
            #avgv = Orange.data.Value(bdist.variable, bdist.avg)
            #minv = Orange.data.Value(bdist.variable, bdist.min)# - 0.5
            #maxv = Orange.data.Value(bdist.variable, bdist.max)# + 0.5
            if minv == maxv:
                return

            if used:
              # This shouldn't be an issues because rule.examples
              # should filetr the distribution correctly
              minv, maxv = None, None
              for cond in rule.filter.conditions:
                if cond.position == idx:
                  minv = max(cond.min, minv) if minv else cond.min
                  maxv = min(cond.max, maxv) if maxv else cond.max

            #block = (maxv - minv) / self.fanout
            #ranges = [(minv + i*block, minv + (i+1)*block) for i in xrange(self.fanout)]
            ranges = [[minv, avgv], [avgv, maxv]]

            for minv, maxv in ranges:
                if minv == maxv: # edge case
                  continue
                new_rule = rule.cloneAndAddContCondition(
                  attr,
                  minv,
                  maxv,
                  orange.ValueFilter.Between,
                  used=used,
                  negate=negate
                )
                new_rule.parent_rule = rule
                if new_rule in self.bad_rules:
                  continue
                
                yield new_rule

        

    def __call__(self, rule, negate=False, **kwargs):
        if rule.fixed:
            return
        if not len(rule.examples):
            return

        start = time.time()        
            
        for idx, ddist, bdist, attr, used in self.next_attributes(rule):
            if self.skip_attribute(attr):
                continue

            for new_rule in self.construct_new_rule(rule, idx, ddist, bdist, attr, used, negate):
                yield attr, new_rule

        self.cost += time.time() - start



class GraphBeamRefiner(BeamRefiner):
    def __init__(self, attrs=None, graph=None, **kwargs):
        BeamRefiner.__init__(self, attrs=attrs)
        if not graph:
            raise
        self.graph = graph



    def __call__(self, rule, negate=False, extend=False, **kwargs):
        #import pdb
        #pdb.set_trace()
        if rule.fixed:
            return
        if not len(rule.examples):
            return
        

        start = time.time()        
        ddists = Orange.statistics.distribution.Domain(rule.examples)
        self.ddists = ddists
        bdists = self.bdists or Orange.statistics.basic.Domain(rule.examples)
        self.bdists = bdists
        attrs = rule.examples.domain.attributes
        useds = [False] * len(attrs)
        for vf in rule.filter.conditions:
            useds[vf.position] = True

        rule_positions = [cond.position for cond in rule.filter.conditions]
        rule_attrs = [rule.examples.domain[pos] for pos in rule_positions]

        if not len(rule_attrs):
            rule_attrs = [None]
            
        next_attrs = set(rule_attrs)        
        if extend:
            for rule_attr in rule_attrs:
                if rule_attr:
                    next_attrs.update(self.graph[rule_attr.name])
                else:
                    next_attrs.update(self.graph[rule_attr])

        for idx, (ddist, bdist, attr, used) in enumerate(zip(ddists, bdists, attrs, useds)):

            if (self.attrs and
                (attr not in self.attrs and attr.name not in self.attrs)):
                continue
            if attr not in next_attrs and attr.name not in next_attrs:
                continue

            for new_rule in self.construct_new_rule(rule, idx, ddist, bdist, attr, used, negate):
                yield attr, new_rule

# graph is a partial ordering
# graph = defaultdict(list)
# graph.update({'A': ['B', 'C'],
#               'B': ['C', 'D']})
