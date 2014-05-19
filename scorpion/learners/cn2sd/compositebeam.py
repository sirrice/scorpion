import orange, Orange
import sys, math
from rule import *






class CompositeBeamFinder(BeamFinder):
    def __init__(self, rank_id, err_func, width=5):
        super(CompositeBeamFinder, self).__init__(rank_id, width=width)
        self.err_func = err_func

    def evaluator(self, new_rule, examples, rank_id, weight_id, target_class, prior):
        idxs = [row['id'] for row in new_rule.filter(examples)]
        self.err_func(idxs)
            
    def candidate_selector(self, rule_list):
        return rule_list[:]

    def rule_filter(self, rule_list):
        pass
    
    def refiner(self):
        pass

    def evaluator(self, rule, data):

        cond_data = rule(data)
        score = self.err_func( cond_data )
        

    def __call__(self):

        rule_list = initialize()

        while len(rule_list):
            for cand_rule in candidates():

                for new_rule in expand_rules():

                    if better(new_rule, best_rule):
                        best_rule = new_rule

                    if not should_stop(new_rule):
                        rule_list.append(new_rule)

            rule_list = rule_filter(rule_list)
                    
