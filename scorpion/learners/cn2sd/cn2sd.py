import orange, Orange
import sys, math, heapq
import numpy as np
import pdb

from collections import Counter

from ...util import *

from rule import *
from evaluator import *
from refiner import *
from beam import *

_logger = get_logger()



class CN2_SD(object):
    def __init__(self,
                 k,
                 rank_id = None,
                 counter_id=None,
                 num_of_rules=0,                 
                 bdiscretize=True,
                 **kwargs):
        self.k = k
        self.counter = counter_id or orange.newmetaid()
        self.rank_id = rank_id or orange.newmetaid()
        self.weightID = orange.newmetaid()
        self.attrs = kwargs.get('attrs', [])
        self.max_rules = num_of_rules
        self.rbf = kwargs.get('rbf', kwargs['beamfinder'](self.rank_id, **kwargs))
        
        # self.rbf = BeamFinder(self.rank_id, width=width)
        # self.rbf.evaluator = RuleEvaluator_WRAccAdd()
        self.bdiscretize = bdiscretize
        
    def discretize(self, data):

        data_discretized = False
        # If any of the attributes are continuous, discretize them
        if data.domain.hasContinuousAttributes():
            original_data = data
            data_discretized = True
            new_domain = []
            discretize = orange.EntropyDiscretization(forceAttribute=True)
            for attribute in data.domain.attributes:
                if self.attrs:
                    if attribute not in self.attrs and attribute.name not in self.attrs:
                        continue
                
                if attribute.varType == orange.VarTypes.Continuous:
                    d_attribute = discretize(attribute, data)
                    # An attribute is irrelevant, if it is discretized into a single interval
                    #if len(d_attribute.getValueFrom.transformer.points) > 0:

                    # remove the D_ prefix because it's screwing things up!
                    d_attribute.name = d_attribute.name[2:]  
                    new_domain.append(d_attribute)
                else:
                    new_domain.append(attribute)
            new_domain.append(original_data.domain.class_var)
            new_domain = Orange.data.Domain(new_domain)
            new_domain.add_metas(original_data.domain.get_metas())
            data = original_data.select(new_domain)
        return data_discretized, data

    def setup_weights(self):
        # weighted covering
        if not self.data.domain.has_meta(self.weightID):
            self.data.addMetaAttribute(self.weightID)  # set weights of all examples to 1
        if not self.data.domain.has_meta(self.rank_id):
            self.data.addMetaAttribute(self.rank_id)  # set weights of all examples to 1
        if not self.data.domain.has_meta(self.counter):
            self.data.addMetaAttribute(self.counter)   # set counters of all examples to 0


    def compute_rules(self, data, targetClass):
        self.data = data
        rules = []

        self.setup_weights()        
        tc = orange.Value(data.domain.classVar, targetClass)

        while self.max_rules==0 or len(rules) < self.max_rules:
            tmpRules =  self.rbf(data, self.weightID, targetClass, None)

            found = True
            for rule in tmpRules:
                if rule not in rules:
                    found = False
                    rules.append(rule)

                rule.set_data(self.data)
                self.decreaseExampleWeights(rule)

                
                # remove points in rule from dataset
                # update evaluator cache
                negrule = rule.cloneAndNegate()
                data = negrule.filter(data)
                self.rbf.evaluator.clear_cache()
                
            if found:
                break
                
        return rules
            
    def __call__(self, data, targetClass):
        '''Returns CN2-SD rules by performing weighted covering algorithm.'''

        if self.bdiscretize:
            original_data = data
            data_discretized, data = self.discretize(data)
        else:
            data_discretized = False

        rules = self.compute_rules(data, targetClass)
            
        if data_discretized:
            # change beam so the rules apply to original data
            targetClassRule = SDRule(original_data, targetClass, conditions=[], g=1)
            rules = [rule.getUndiscretized(original_data) for rule in rules]
        else:
            targetClassRule = SDRule(data, targetClass, conditions=[], g =1)

        return SDRules(rules, targetClassRule, "CN2-SD")

    
    def decreaseExampleWeights(self, rule):
        for d in rule.filter_table(self.data):
            #if d.getclass()==rule.targetClass:
            tmp = d.getweight(self.counter)+1.0
            if tmp>self.k:
                d.setweight(self.weightID, 0)
            else:
                d.setweight(self.weightID, 1.0/(tmp+1))
            d.setweight(self.counter, tmp)
        


class Attribute_CN2_SD(CN2_SD):
    """
    Try each attribute individually, one at a time
    """
    def compute_rules(self, data, targetClass):
        self.data = data
        rules = []

        tc = orange.Value(data.domain.classVar, targetClass)

        rules = []
        for attr in data.domain:
            for row in self.data:
                row.setweight(self.weightID, 1)
                row.setweight(self.rank_id, 1)
                row.setweight(self.counter, 0)

            if self.attrs:
                if attr not in self.attrs and attr.name not in self.attrs:
                    continue
            self.setup_weights()
            self.rbf.refiner.set_attributes([attr])

            targetClassRule = SDRule(data, targetClass, conditions=[], g=1)
            rules.extend( self.rbf(data, self.weightID, targetClass, None) )


        rules.sort(reverse=True)
        return rules


class Level_CN2_SD(CN2_SD):
    def compute_rules(self, data, targetClass):
        self.data = data
        self.setup_weights()
        tc = orange.Value(data.domain.classVar, targetClass)

        self.setup_weights()

        targetClassRule = SDRule(data, targetClass, conditions=[], g=1)
        level_rules = self.rbf(data, self.weightID, targetClass,None)
        return level_rules[-1]

