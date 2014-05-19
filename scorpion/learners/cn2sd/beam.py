#
# Beam Search algorithm
# 
#


import numpy as np
import pdb
import orange, Orange
import sys
import math
import heapq

from collections import Counter

from ...util import *

from rule import *
from evaluator import *
from refiner import *
from normalize import *


_logger = get_logger()


class StoppingValidator(object):
    def __call__(self, new_rule, par_examples):
        """
        @par_examples parent rule's examples
        @par_dist parent rule's distribution
        @return True if new_rule is not good enough, False if it is good enough
        """
        if not new_rule.quality or new_rule.quality <= 0:
            return False
        if new_rule.complexity > 5:
            return False
        return True


class EntropyEval(object):
    def get_entropy(self, vals):
        n, tot = 0. ,0.
        noDif0 = 0
        for v in vals:
            if v > 0:
                tot += v*math.log(v)
                n += v
                noDif0 += 1
        return noDif0 > 1 and (math.log(n)-tot/n)/math.log(2.0) or 0
                        
    def __call__(self, rule, table, weight_id, target_class, apriori):
        raise
        obs_dist = rule.classDistribution
        if not obs_dist.cases:
            return None
        if target_class == -1:
            return -self.get_entropy(obs_dist)

        p = target_class < len(obs_dist) and obs_dist[target_class] or 0.0
        P = (target_class < len(apriori) and apriori[target_class] > 0) and apriori[target_class] or 1e-5
        n = obs_dist.abs - p
        N = apriori.abs - P
        n = n <= 0.0 and 1e-6 or n        
        N = N <= 0.0 and 1e-6 or N
        p = p <= 0.0 and 1e-6 or p

        return (p*math.log(p) + n*math.log(n) - obs_dist.abs * math.log(obs_dist.abs)) / obs_dist.abs
    


class BeamFinder(object):
    def __init__(self, rank_id, width=5, **kwargs):
        self.width = width
        self.rank_id = rank_id
        self.past_best_rules = set()
        self.evaluator = kwargs.get('evaluator', None)
        self.stopping_validator = kwargs.get('stopping_validator', None)
        self.eval_entropy = kwargs.get('eval_entropy', None)
        self.normalizer = kwargs.get('normalizer', ScoreNormalizer)()
        self.refiner = None

        refiner = kwargs.get('refiner', BeamRefiner)
        if type(refiner) == type:
            refiner = refiner(**kwargs)
        self.refiner = refiner


    def initializer(self,data, weight_id, target_class, base_rules, apriori, best_rule):
        if base_rules:
            raise "don't support base rules"
        
        rule = SDRule(data, target_class)
        self.evaluator(rule, data, self.rank_id, weight_id, target_class, apriori)        
        return [rule]

    def candidate_selector(self, rule_list, data, weight_id):
        ret = []
        for rule in rule_list:
            if rule.stats_std == 0:
                _logger.debug("cand_sel\tskipping\t%s", rule)
                continue
            ret.append( rule )
        return ret

    def rule_filter(self, rule_list, data, weight):
        if len(rule_list) <= self.width:
            return rule_list
        def _cmp_(r1, r2):
            if r1.quality > r2.quality:
                return 1
            if r1.quality == r2.quality and r1.complexity < r2.complexity:
                return 1
            return -1
        rule_list.sort(cmp=_cmp_, reverse=True)
        return rule_list[:self.width]


    def should_skip(self, rule, data, weight_id, target_class, apriori, cand_rule, rule_list, terminal_rules):
        if rule.quality - cand_rule.quality <= 0:
            return True
        if cand_rule.stats_mean is not None:
            if (rule.stats_mean <= cand_rule.stats_mean and
                rule.stats_nmean > cand_rule.stats_nmean):
                return True

        if (self.eval_entropy and
            self.eval_entropy(rule, data, weight_id, target_class, apriori) > -0.01):
            print "entropy too high"
            return True
        if (self.stopping_validator and
            self.stopping_validator(rule, cand_rule.examples, weight_id)):
            return True
        if rule in rule_list:
            return True
        if rule in terminal_rules:
            return True
        if len(rule_list) >= self.width and rule <= rule_list[0]:
            return True
        return False



    def is_subsumed(self, r, prev_best):
        for pb in prev_best:
            if (len(r.examples) == len(pb.examples) and
                r.isSubsumed(pb)):
                return True
        return False

    def remove_subsumed(self, new_rules, prev_best):
        ret = []
        for r in new_rules:
            if self.is_subsumed(r, prev_best):
                _logger.info("subsumed\t%s", r)
            else:
                ret.append(r)
        return ret
    
    def __call__(self, data, weight_id, target_class, base_rules):

        if not self.evaluator:
            raise "Looks like you didn't set the evaluator!"

        apriori = Orange.statistics.distribution.Distribution(data.domain.class_var, data)
        rule_list = self.initializer(data, weight_id, target_class, base_rules, apriori, None)


        previous_bests = []
        while len(rule_list):
            candidates = self.candidate_selector(rule_list, data, weight_id)[:self.width]
            added_new_rule = False
            new_rules = []

            for cand_rule in candidates:
                
                for attr, new_rule in self.refiner(cand_rule):
                    if self.is_subsumed(new_rule, previous_bests):
                       _logger.info("subsumed\t%s", new_rule)
                       continue
                    
                    if new_rule.parent_rule:
                        if len(new_rule.examples) == len(new_rule.parent_rule.examples):
                            continue
                    
                    self.evaluator(new_rule, data, self.rank_id, weight_id, target_class, apriori)

                    if new_rule.score == None:
                        self.refiner.add_bad_rule(new_rule)
                        continue


                    new_rules.append(new_rule)


                        

            # compute quality based on order of measures across all new rules
            new_rules = self.remove_subsumed(new_rules, previous_bests)
            all_rules_so_far = new_rules + previous_bests + [r.parent_rule for r in new_rules] + candidates
            if not self.normalizer(all_rules_so_far):
                _logger.debug("not assigned qualities.  breaking")
                break
            new_rules.sort(reverse=True)



            rule_list = []
            for rule in new_rules:
                if rule.quality < min([r.quality for r in candidates]):# and len(rule_list) > self.width:
                    break
                if rule not in previous_bests:
                    rule_list.append(rule)
                    previous_bests.append(rule)

            s = [ '\tquality: %d\t%.4f\t%.4f\t%s' % (r.id, r.quality, r.improvement, r.ruleToString())
                  for r in rule_list if r.parent_rule]
            _logger.debug( "in rule list:\n%s", '\n'.join(s))


        allrules = previous_bests
        self.normalizer(allrules)
        allrules.sort(reverse=True)

        s = [ '\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%s' % (r.id, r.quality, r.score, r.stats_mean, r.stats_nmean, r.ruleToString())
              for r in allrules]
        _logger.info("all top rules:\n%s", '\n'.join(s))

        idx = 1
        while idx < len(allrules) and allrules[idx].eq(allrules[idx-1]):
            idx += 1
        _logger.info( "beam: returning: %s", ' and '.join(map(str, allrules[:idx])))


        return allrules[:idx]


class RankBeamFinder(BeamFinder):
    def __init__(self, *args, **kwargs):
        kwargs['normalizer'] = RankNormalizer
        BeamFinder.__init__(self, *args, **kwargs)

class BoundedBeamFinder(BeamFinder):
    def __init__(self, *args, **kwargs):
        kwargs['normalizer'] = BoundedNormalizer
        BeamFinder.__init__(self, *args, **kwargs)
            
        
    

class AttributeBeamFinder(BeamFinder):
    """
    Beam Finder for single attribute at a time search
    """
    
    def __call__(self, data, weight_id, target_class, base_rules):
        if not self.evaluator:
            raise "looks like you didn't set the evaluator!"


        apriori = Orange.statistics.distribution.Distribution(data.domain.class_var, data)
        rule_list = self.initializer(data, weight_id, target_class, base_rules, apriori, None)
        heapq.heapify( rule_list )

        while len(rule_list):
            s = [ '\t%d\t%.4f\t%s' % (r.id, r.quality, r.ruleToString()) for r in rule_list]
            _logger.debug( "in rule list:\n%s", '\n'.join(s))

            
            candidates = self.candidate_selector(rule_list, data, weight_id)
            added_new_rule = False
            for cand_rule in candidates:
                new_rules = self.refiner(cand_rule)

                for attr, new_rule in new_rules:
                    self.evaluator(new_rule, data, self.rank_id, weight_id, target_class, apriori)

                    if new_rule.quality <= 0:
                        self.refiner.add_bad_rule(new_rule)


                    if (self.eval_entropy and
                        self.eval_entropy(new_rule, data, weight_id, target_class, apriori) > -0.01):
                        print "entropy too high"
                        continue
                    if (self.stopping_validator and
                        self.stopping_validator(new_rule, cand_rule.examples, weight_id)):
                        continue
                    if new_rule in rule_list:
                        continue

                    print "rule\t%.4f\t%.4f\t%.4f\t%.4f\t%s" % (new_rule.quality,
                                                                 new_rule.quality - cand_rule.quality,
                                                                 new_rule.stats_mean or -1,
                                                                 new_rule.stats_std or -1,
                                                                 new_rule.ruleToString())

                    
                    if len(rule_list) >= self.width and new_rule <= rule_list[0]:
                        continue

                    # need a list that contains high mean rules with std = 0

                        
                    if len(rule_list) >= self.width:
                        heapq.heappushpop(rule_list, new_rule)
                    else:
                        heapq.heappush(rule_list, new_rule)
                    added_new_rule = True
                    # print "added\t%.4f\t%.4f\t%.4f\t%s" %(new_rule.quality,
                    #                                       new_rule.stats_mean,
                    #                                       new_rule.stats_std,
                    #                                       new_rule.ruleToString())

            rule_list = self.rule_filter(rule_list, data, weight_id)
            if not added_new_rule:
                break

        return rule_list




class LevelBeamFinder(BeamFinder):
    """
    Tries all rules up to a fixed number of levels (clauses)
    """
    def __init__(self, rank_id, width=5, **kwargs):
        super(LevelBeamFinder, self).__init__(rank_id, width=width, **kwargs)
        self.levels = kwargs.get('max_levels', 3)

    
    def __call__(self, data, weight_id, target_class, base_rules):
        if not self.evaluator:
            raise "looks like you didn't set the evaluator!"

        apriori = Orange.statistics.distribution.Distribution(data.domain.class_var, data)
        rule_list = self.initializer(data, weight_id, target_class, base_rules, apriori, None)
        best_rule = rule_list[0]

        level_rules = [rule_list]
        next_rules = set()
        for level in xrange(self.levels):
            for cand_rule in level_rules[-1]:
                if cand_rule.stats_std == 0:
                    next_rules.add(cand_rule)
                    continue
                
                for attr, new_rule in self.refiner(cand_rule):
                    self.evaluator(new_rule, data, self.rank_id, weight_id, target_class, apriori)
                    #if new_rule.quality < 0:
                    #    continue
                    next_rules.add(new_rule)

            level_rules.append(next_rules)
            next_rules = set()
            print "level ", level

        return level_rules




class GraphBeamFinder(BeamFinder):
    """
    Run beam finder based on graph provided by user
    """
    def __init__(self, *args, **kwargs):
        BeamFinder.__init__(self, *args, **kwargs)
    
    def __call__(self, data, weight_id, target_class, base_rules):

        if not self.evaluator:
            raise "Looks like you didn't set the evaluator!"

        apriori = Orange.statistics.distribution.Distribution(data.domain.class_var, data)
        rule_list = self.initializer(data, weight_id, target_class, base_rules, apriori, None)
        heapq.heapify( rule_list )
        best_rule = rule_list[0]
        terminal_rules = set()

        added_new_rule = False
        while len(rule_list):
            s = [ '\t%d\t%.4f\t%s' % (r.id, r.quality, r.ruleToString()) for r in rule_list]
            _logger.debug( "in rule list:\n%s", '\n'.join(s))

            extend = not added_new_rule
            added_new_rule = False
            refined = False
            
            candidates = self.candidate_selector(rule_list, data, weight_id)
            for cand_rule in candidates:

                new_rules = self.refiner(cand_rule, extend=extend)
                for attr, new_rule in new_rules:
                    self.evaluator(new_rule, data, self.rank_id, weight_id, target_class, apriori)

                    if new_rule.quality == 0:
                        self.refiner.add_bad_rule(new_rule)
                        continue
                    if self.should_skip(new_rule, data, weight_id, target_class, apriori,
                                        cand_rule, rule_list, terminal_rules):
                        continue
                    refined = True
                    
                    if new_rule > best_rule:
                        best_rule = new_rule

                    if new_rule.stats_std == 0:
                        terminal_rules.add(new_rule)
                        continue

                    # need a list that contains high mean rules with std = 0                        
                    if len(rule_list) >= self.width:
                        heapq.heappushpop(rule_list, new_rule)
                    else:
                        heapq.heappush(rule_list, new_rule)
                    added_new_rule = True

            rule_list = self.rule_filter(rule_list, data, weight_id)
            _logger.info( "beam: best iter: %s", best_rule.printRule() )
            if not added_new_rule and not refined:
                break

        allrules = set()
        allrules.update(rule_list)
        allrules.update(terminal_rules)
        allrules = sorted(allrules, reverse=True)

        idx = 1
        while idx < len(allrules) and allrules[idx].eq(allrules[idx-1]):
            idx += 1
        _logger.info( "beam: returning: %s", ' and '.join(map(str, allrules[:idx])))
        return allrules[:idx]
            
        
        
        self.past_best_rules.add(hash(best_rule))

        return best_rule
    
