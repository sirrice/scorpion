import orange, Orange
import sys, math, heapq
from rule import *
from refiner import *
from collections import Counter
from ...util import ids_filter, get_logger, max_prob
import numpy as np
import pdb

_logger = get_logger()


class RuleEvaluator_WRAccAdd(orange.RuleEvaluator):
    def __init__(self, *args, **kwargs):
        self.cost = 0.

    def clear_cache(self):
        pass

    def __call__(self, newRule, examples, rank_id, weightID, targetClass, prior):
        """compute: prob(class,condition) - p(cond)*p(class)
           or:      p(cond) * ( p(class|cond) - p(class) )
        """
        ncond = N = nclasscond = nclass = 0.0
        np_rank_all = examples.to_numpyMA('w', rank_id)[0].reshape(len(examples))
        np_weight_all = examples.to_numpyMA('w', weightID)[0].reshape(len(examples))
        np_rank_new = newRule.examples.to_numpyMA('w', rank_id)[0].reshape(len(newRule.examples))
        np_weight_new = newRule.examples.to_numpyMA('w', weightID)[0].reshape(len(newRule.examples))

        start = time.time()
        N = np_weight_all.sum()
        nclass = np.dot(np_rank_all, np_weight_all)
        ncond = np_weight_new.sum()
        nclasscond = np.dot(np_rank_new, np_weight_new)
        wracc = nclasscond / N - (ncond * nclass) / (N * N)
        self.cost += time.time()-start
        
        if wracc > 0:
            _logger.debug( 'wracc\t%.5f\t%s', wracc, newRule.ruleToString())
        if N == 0:
            wracc = -1
        if math.isnan(wracc):
            wracc = -1


        newRule.quality = wracc
        newRule.score = wracc
        newRule.stats_mean = 0.
        newRule.stats_nmean = 0.
        return wracc


class RuleEvaluator_WRAccMult(orange.RuleEvaluator):
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, newRule, examples, weightID, targetClass, prior):
        raise


#####################################################
#
# The following evaluators are all for combined rule learning
#
#
#####################################################

class ErrorRunner(object):
    def __init__(self, err_func):
        self.err_func = err_func
        self.cache = {}

    def __call__(self, rule):
        rulehash = hash(rule)
        if rulehash in self.cache:
            return self.cache[rulehash]
        if not len(rule.examples):
            return 0.

        if True:
            data = rule.examples
        else:
            rule.filter(t, negate=negate)
        score = self.err_func( data.to_numpyMA('a')[0] ) 
        score /= (1 + len(rule.examples))
            
        if math.isinf(score):
            pdb.set_trace()
            self.err_func(data.to_numpyMA('a')[0])

        if math.isnan(score):
            return 0.

        
        self.cache[rulehash] = score
        return score


class ErrorRunnerNegated(object):
    def __init__(self, err_func):
        self.err_func = err_func
        self.cache = {}

    def __call__(self, rule):
        rulehash = hash(rule)
        if rulehash in self.cache:
            return self.cache[rulehash]
        if not len(rule.examples):
            return 0.
        
        rows = rule.filter(self.err_func.table)
        if not len(rows):
            score = 0.
        else:
            score = np.mean([ row['temp'].value for row in rows ]) - self.err_func.mean
        
        self.cache[rulehash] = score
        return score
        

class ConfidenceRefiner(object):
    def __init__(self, nsamples=10, get_error=None, refiner=None, good_dist=None, **kwargs):
        self.good_dist = good_dist
        self.nsamples = 10
        self.refiner = refiner or BeamRefiner()
        self.get_error = get_error
        self.cache = {}
        self.ncalls = 0
        
    def clear_cache(self):
        self.cache = {}

    def __call__(self, rule, negate=False):
        rulehash = hash('%s\t[%s]' % (rule, negate))
        if rulehash in self.cache:
            return self.cache[rulehash]

        if negate:
            res = self.run_negated(rule)
        else:
            res = self.run(rule)

        self.cache[rulehash] = res
        return res

    def run_negated(self, rule):
        base_data = rule.data
        err_func = self.get_error.err_func
        examples = rule.examples

        # sample values in base_data that are not in examples
        sampsize = max(1, int(0.05 * len(base_data)))        
        sampler = Orange.data.sample.SubsetIndices2(p0=sampsize)
        sampler.random_generator = Orange.misc.Random(len(examples))
        

        scores = []
        for new_rule in self.refiner(rule, negate=True):
            sample = base_data.select_ref(sampler(base_data), negate=True) # select 0.05%
            new_rule.filter.negate=not new_rule.filter.negate
            sample = sample.filter_ref(rule.filter)
            n = len(sample)
            if n == 0:
                continue
            
            score = err_func(sample.to_numpyMA('a')[0])
            self.ncalls += 1            

            if not math.isnan(score):
                score /= n
                scores.append( score )
            

        if not len(scores):
            return 0., 0.,
        
        mean, std = np.mean(scores), np.std(scores)
        std = math.sqrt( sum( (mean - score)**2 for score in scores ) / (len(scores) - 1.5) )

        if self.good_dist:
            mean -= self.good_dist[0]
        return mean, std

            
            

    def run(self, rule):
        base_data = rule.data
        examples = rule.examples
        err_func = self.get_error.err_func

        scores = []
        for new_rule in self.refiner(rule):
            if not len(new_rule.examples):
                continue

            #return new_rule.filter(t, negate=negate)
            data = new_rule.examples
            score = self.err_func(data.to_numpyMA('a')[0]) / len(new_rule.examples)
            self.ncalls += 1            

            if not math.isnan(score):
                scores.append(score)
        if not len(scores):
            return 0., 0.

        mean, std = np.mean(scores), np.std(scores)
        # unbiased std estimator
        std = math.sqrt( sum( (mean - score)**2 for score in scores ) / (len(scores) - 1.5) )

        if self.good_dist:
            mean -= self.good_dist[0]
        return mean, std



class ConfidenceSample(object):
    def __init__(self, nsamples=10, get_error=None, good_dist=None, **kwargs):
        self.good_dist = good_dist
        self.nsamples = 50
        self.get_error = get_error
        self.cache = {}
        self.ncalls = 0
        
    def clear_cache(self):
        self.cache = {}
    
    def __call__(self, rule, negate=False):
        """
        run error function on samples of the rule to compute a confidence score
        """
        rulehash = hash('%s\t[%s]' % (rule,negate))
        if rulehash in self.cache:
            return self.cache[rulehash]

        if negate:
            res = self.run_negated(rule)
        else:
            res = self.run(rule)

        self.cache[rulehash] = res
        return res

    def run_negated(self, rule):

        base_data = rule.data
        err_func = self.get_error.err_func
        examples = rule.examples

        if len(examples) == len(base_data):
            return self.good_dist[0], self.good_dist[1], 0., 0.

        # sample values in base_data that are not in examples
        sampsize = max(1, int(0.1 * len(base_data)))        
        sampler = Orange.data.sample.SubsetIndices2(p0=sampsize)
        sampler.random_generator = Orange.misc.Random(len(examples))

        scores = []
        tries = 0
        while len(scores) < min(self.nsamples, len(base_data)-len(examples)):
            # XXX: doesn't work for the slow error functions
            sample = base_data.select_ref(sampler(base_data), negate=True)
            rule.filter.negate=not rule.filter.negate
            sample = sample.filter_ref(rule.filter)
            rule.filter.negate=not rule.filter.negate
            n = len(sample)
            tries += 1
            if n == 0:
                continue
            data = sample.to_numpyMA('a')[0]
            score = err_func(data)
            self.ncalls += 1            

            if not math.isnan(score):
                score /= n
                scores.append( score )
            else:
                pdb.set_trace()
                score = err_func(data)

        if not len(scores):
            return 0., 0., 0., 0.

        mean, std, minv, maxv = np.mean(scores), np.std(scores), min(scores), max(scores)
        std = math.sqrt( sum( (mean - score)**2 for score in scores ) / (len(scores) - 1.5) )

        if self.good_dist:
            mean -= self.good_dist[0]

        return mean, std, minv, maxv

        
        

    def run(self, rule):
        err_func = self.get_error.err_func
        examples = rule.examples

        sampsize = max(2, int(0.1 * len(examples)))
        if len(examples) < 2:
            sampler = lambda table: [0]*len(table)
        else:
            sampler = Orange.data.sample.SubsetIndices2(p0=sampsize)
            sampler.random_generator = Orange.misc.Random(len(examples)) 

        scores = []
        for i in xrange(self.nsamples):
            idxs = sampler(examples)
            n = (len(idxs) - sum(idxs))
            if n == 0:
                continue

            if True:
                data = examples.select_ref(idxs, negate=True)
            else:
                data = examples.select_ref(idxs, negate=False)
            score = err_func(data.to_numpyMA('a')[0]) / n
            self.ncalls += 1
            
            if not math.isnan(score):
                scores.append( score )


        if not len(scores):
            return 0., 0., 0., 0.


        mean, std, minv, maxv = np.mean(scores), np.std(scores), min(scores), max(scores)
        std = math.sqrt( sum( (mean - score)**2 for score in scores ) / (len(scores) - 1.5) )

        if self.good_dist:
            mean -= self.good_dist[0]            
        
        return mean, std, minv, maxv




            
class RuleEvaluator_RunErr(orange.RuleEvaluator):
    def __init__(self, good_dist, get_error, confidence, **kwargs):
        self.good_dist = good_dist
        self.get_error = get_error
        self.confidence = confidence
        self.cost = 0.
        self.n_sample_calls = 0
        self.beta = kwargs.get('beta', .1)  # beta.  weight of precision vs recall

    def clear_cache(self):
        self.confidence.clear_cache()

    def get_weights(self, newRule, examples, weightID):
        N = len(examples)
        ncond = len(newRule.examples)
        if weightID is None:
            return 1., 1.,
        weight_all = examples.to_numpyMA('w', weightID)[0].reshape(N)
        weight_cond = newRule.examples.to_numpyMA('w', weightID)[0].reshape(ncond)

        allweights = np.mean(weight_all)
        condweights = np.mean(weight_cond) if ncond else 0.
        return condweights, allweights

    def __call__(self, newRule, examples, rank_id, weightID, targetClass, prior):
        condweights, allweights = self.get_weights(newRule, examples, weightID)
        weight = condweights / allweights

        score = self.get_error( newRule )

        if (not allweights or
            not len(newRule.examples)):
            newRule.score = None
            return 0.

        mean,std,minv, maxv = self.confidence(newRule)
        negmean, negstd,nminv,nmaxv = self.confidence(newRule, negate=True)

        ret = 0

        stats = (weight, score, minv * 100, maxv * 100, std, negmean, len(newRule.examples))
        stats = ('\t%.4f' * len(stats)) % stats
        _logger.debug( 'wracc samp:%s\t%s' % (stats, newRule) )

        newRule.score = score
        newRule.weight = condweights / allweights
        newRule.stats_mean, newRule.stats_std = mean, std
        newRule.stats_minv, newRule.stats_maxv = minv, maxv
        newRule.stats_nmean, newRule.stats_nstd = negmean, negstd
        newRule.stats_nminv, newRule.stats_nmaxv = nminv, nmaxv

        self.n_sample_calls += self.confidence.nsamples
        return ret



class RuleEvaluator_RunErr_Next(RuleEvaluator_RunErr):
    """
    Uses leave-predicate-out based scoring
    """
    def __init__(self, good_dist, err_func, **kwargs):
        get_error = ErrorRunner(err_func)
        confidence = kwargs.get('confidence',
                                ConfidenceRefiner(get_error=get_error,
                                                  good_dist=good_dist,
                                                  **kwargs))
        RuleEvaluator_RunErr.__init__(self, good_dist, get_error, confidence, **kwargs)


class RuleEvaluator_RunErr_Sample(RuleEvaluator_RunErr):
    def __init__(self, good_dist, err_func, **kwargs):
        get_error = ErrorRunner(err_func)
        confidence = kwargs.get('confidence', ConfidenceSample(get_error=get_error,
                                                               good_dist=good_dist,
                                                               **kwargs))
        RuleEvaluator_RunErr.__init__(self, good_dist, get_error, confidence, **kwargs)
        

class RuleEvaluator_RunErr_Negated(RuleEvaluator_RunErr):
    """
    Uses error(predicate) scoring
    """
    def __init__(self, good_dist, err_func, **kwargs):
        get_error = ErrorRunnerNegated(err_func)
        confidence = kwargs.get('confidence', ConfidenceSample(get_error=get_error))
        RuleEvaluator_RunErr.__init__(self, good_dist, get_error, confidence, **kwargs)

    def __call__(self, newRule, examples, rank_id, weightID, targetClass, prior):
        if not len(newRule.examples):
            return 0.
        
        condweights, allweights = self.get_weights(newRule, examples, weightID)
        score = self.get_error( newRule )
        if score > 0:
            _logger.debug( 'wracc samp:\t%.4f\t%d\t%.4f\t%s',
                       score, len(newRule.examples), condweights, newRule.ruleToString())

        return score / len(newRule.examples)
