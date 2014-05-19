import orange, Orange, orngTree, orngStat, orngTest, orngDisc
import Orange.feature as orf

from learners.cn2sd.beam import *
from learners.cn2sd.cn2sd import *
from learners.cn2sd.evaluator import *
from settings import *
from aggerror import *

def learn_good_keys(obj, aggerr):
    table = get_provenance(obj,
                           aggerr.agg.cols,
                           obj.goodkeys.get(aggerr.agg.shortname, []))
    learner = Orange.classification.svm.SVMLearner()
    leaner.tune_parameters()
    svm = learner(table)
    

def append_scores(table, scores):
    # normalize the scores
    max_score = max(scores) if max(scores) > 0 else 1.0
    scores = map(lambda v: v / max_score, scores)


    # add an initial weight_id for rank
    # XXX: only works for MultWeights
    #      AddWeights requires setting "Coverage" meta variable correctly
    weight_id = Orange.feature.Descriptor.new_meta_id()
    counter_id = Orange.feature.Descriptor.new_meta_id()
    table.addMetaAttribute(weight_id)
    table.addMetaAttribute(counter_id)
    table.domain.addmeta(weight_id, Orange.feature.Continuous(RANK_VAR))
    table.domain.addmeta(counter_id, Orange.feature.Continuous(COUNTER_VAR))
    for instance, score in zip(table, scores):
        instance[weight_id] = score
        instance[counter_id] = score and (1. / score) - 1 or 0.0
    return weight_id, counter_id
    


def classify_error_tuples(table, scores, width=3, ignore_attrs=[], bdiscretize=True, **kwargs):    
    table.domain.class_var = table.domain[ERROR_VAR]
    target_class = list(table.domain.class_var.values).index('1')
    rank_id, counter_id = append_scores(table, scores)
    ignore_attrs += ['id']    
    attrs = [attr.name for attr in table.domain if attr.name not in ignore_attrs and attr not in ignore_attrs]

    evaluator = RuleEvaluator_WRAccAdd()
    learner = CN2_SD(5,
                     rank_id=rank_id,
                     counter_id=counter_id,
                     width=width,
                     num_of_rules=4,
                     bdiscretize=bdiscretize,
                     attrs=attrs,
                     beamfinder=BeamFinder,
                     evaluator=evaluator)
    classifier = learner(table, target_class)
    return table, classifier.rules, learner.rbf.evaluator.cost





def classify_error_tuples_modified(table,  good_dist,  err_func, **kwargs):
    kwargs['cn2sd'] = CN2_SD
    return classify_error_tuples_combined(table, good_dist, err_func, **kwargs)

def classify_error_tuples_attributes(table, good_dist, err_func, **kwargs):
    kwargs['cn2sd'] = Attribute_CN2_SD
    return classify_error_tuples_combined(table, good_dist, err_func, **kwargs)

def classify_error_tuples_combined(table,
                                   good_dist,
                                   err_func,
                                   cn2sd = CN2_SD,
                                   evaluator=RuleEvaluator_RunErr_Next,
                                   beamfinder = BeamFinder,
                                   ignore_attrs=[],
                                   **kwargs):
    table.domain.class_var = table.domain[ERROR_VAR]
    target_class = list(table.domain.class_var.values).index('1')
    ignore_attrs += ['id']
    attrs = [attr.name for attr in table.domain if attr.name not in ignore_attrs and attr not in ignore_attrs]

    err_func.setup(table)
    
    evaluator = evaluator(good_dist, err_func, attrs=attrs, **kwargs)
    learner = cn2sd(3,
                    num_of_rules=4,
                    evaluator=evaluator,
                    beamfinder=beamfinder,
                    attrs=attrs,
                    **kwargs)


    # learner.data = table
    # learner.setup_weights()
    # return rbf(table, learner.weightID, target_class, [])

    classifier = learner(table, target_class)
    #print "N calls\t", err_func.ncalls
    
    return table, classifier.rules, learner.rbf.evaluator.n_sample_calls

