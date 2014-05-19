import numpy as np
import Orange
import orange
import math
import random
import pdb

from collections import Counter, defaultdict
from itertools import chain

from settings import *
from util import *
from split import *
from aggerror import *
from functions import StdFunc, AvgFunc
from learners.cn2sd.refiner import *

_logger = get_logger()
inf = 1e10000000



class ScorerBlah(object):
    def __init__(self, f, bad_tables, good_tables, err_funcs):
        self.bad_tables = bad_tables
        self.good_tables = good_tables
        self.err_funcs = err_funcs

    def eval(self, predicate, table):
        return 0

    def __call__(self, predicate):
        bad_scores = []
        good_scores = []
        bad_counts = []
        good_counts = []
        
        for bad_table in bad_tables:
            matches = predicate.filter_table(bad_table)
            score = self.err_funcs[bad_table](matches)
            bad_scores.append(score)
            bad_counts.append(len(matches))

        for good_table in good_tables:
            matches = predicate.filter_table(good_table)
            score = self.err_funcs[good_table](matches)
            good_scores.append(score)
            good_counts.append(len(matches))

        return self.f(bad_scores, good_scores, bad_counts, good_counts, predicate.complexity)


class Scorer(object):

    def __call__(self, bad_stats, good_stats):
        pass





class PartitionStats(object):
    def __init__(self, est, **kwargs):
        self.est = est
        self.__dict__.update(kwargs)

    def __str__(self):
        d = dict(self.__dict__)
        if 'vals' in d:
            del d['vals']
        return '%f\t%s' % (self.est, str(d))


class Sampler(object):
    def __init__(self, errprob, SCORE_ID):
        self.SCORE_ID = SCORE_ID
        self.errprob = errprob

    
    def get_sample_size(self, pop, *args, **kwargs):
        return best_sample_size(pop, *args, **kwargs)


    """
    Reuses previously computed samples, if possible
    """
    def __call__(self, tables):
        return map(self.sample, tables)

    def sample(self, table):
        f = Orange.data.filter.ValueFilterContinuous(position=self.SCORE_ID,
                                                     oper=orange.ValueFilter.NotEqual,
                                                     ref=-inf)
        c = Orange.data.filter.Values(domain=table.domain,
                                      conditions=[f],
                                      negate=True)
        scored = table.filter_ref(c)



        pop = len(table)
        samp_size = min(pop, self.get_sample_size(pop, self.errprob[-1]) + 1)
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


class Evaluator(object):


    def __init__(self, SCORE_ID, errprob, err_funcs, aggcols, epsilon, **kwargs):
        self.global_std = StdFunc()
        self.global_mean = AvgFunc()
        self.global_bounds = [1e10000, -1e10000]

        self.SCORE_ID = SCORE_ID
        self.err_funcs = err_funcs
        self.aggcols = aggcols
        self.epsilon = epsilon
        self.errprob = errprob
        self.min_points = kwargs.get('min_points', 2)

        self.sampler = Sampler(self.errprob, self.SCORE_ID)        

        
    def kn(self, n):
        """
        return Kn, where UMVU estimator of std is Kn*S
        """
        try:
            return math.sqrt(2./(n-1)) * (math.gamma(n/2.) / (math.gamma((n-1.)/2.)))
        except:
            return 1.


    def evaluate(self, tables, sample=True):
        if isinstance(tables, list):
            self.samples = self.sampler(tables) if sample else tables

            if not self.samples:
                return None
            
            ests, stds, vals = [], [], []
            for table, err_func in zip(self.samples, self.err_funcs):
                est, std, vs = self.evaluate_table(table, err_func)
                ests.append(est)
                stds.append(std)
                vals.extend(vs)

            est = np.mean(ests)
            std = np.mean(stds)
            
            if len(self.err_funcs) and 'Sum' in str(self.err_funcs[0].klass):
                est = est / sum(map(len, self.samples)) * sum(map(len, tables))


            if len(vals) != sum(map(len, self.samples)):
                raise RuntimeError("# vals != # samples")
            
                if sample and len(vals) != sum(map(len, tables)):
                    raise RuntimeError("# vals != # pts")

            return PartitionStats(est, std=std, vals=vals)
        else:
            return self.evaluate([tables], sample=sample)


    def evaluate_table(self, table, err_func):
        if not len(table):
            return (err_func([]),
                    0.,
                    [])

        
        vals = []
        newvals = []
        for row in table:
            if row[self.SCORE_ID].value == -inf:
                est = err_func([row])
                row[self.SCORE_ID] = est
                newvals.append(est)
            vals.append(row[self.SCORE_ID].value)
        samp_size = len(vals)


        newvals = np.array(newvals)
        self.global_std.delta(add=[newvals], update=True)
        self.global_mean.delta(add=[newvals], update=True)
        self.global_bounds[0] = min(self.global_bounds[0], min(vals))
        self.global_bounds[1] = max(self.global_bounds[1], max(vals))

        if samp_size == 1:
            est, std = vals[0], 0.
        else:
            # slightly biased std estimator
            try:
                est = np.mean(vals)
            except:
                pdb.set_trace()
            S2 = 1. / (samp_size - 1) * sum([(v-est)**2 for v in vals])
            S = math.sqrt(S2)
            std = self.kn(samp_size) * S

        if samp_size > 2:
            _logger.debug('\tsampsize(%d)\t%.4f+-%.4f\t%.4f - %.4f',
                          samp_size,
                          est,
                          std,
                          self.global_bounds[0],
                          self.global_bounds[1]
                           )

        return est, std, vals


    def weight(self, val):
        u = self.global_mean.value()
        std = self.global_std.value()
        if std == 0:
            return 1.

        max_std = 2.58
        #max_std = 1.6

        # weight increases quadratically.
        nstds = (val - u) / std
        nstds = min(max(0, nstds + 2), max_std)
        y = (nstds / max_std) ** 2

        return y

        # linear scale, hits maximum at 2.58-0.5 i think
        r = 2.58 + 2.58 + 0.5 # why is a 0.5 here?
        v = min(r, max(0., (val - u) / std - 0.5))
        return 0.0001 + (v / r) * (1 - 0.0001)

        # using ERF
        w = .5 * (1 + math.erf( (val-u) / math.sqrt(2*std**2) ))
        # rescale to be between 0.2 - 1
        return 0.001 + w * (1 - 0.001)


    def should_stop(self, tables, bad_stats, good_stats):
        if max(map(len,tables)) <= self.min_points:
            return True


        # val, allsame = None, True
        # for i, row in enumerate(table):
        #     if i == 0:
        #         val = tuple([row[aggcol].value for aggcol in self.aggcols])
        #     else:
        #         if val != tuple([row[aggcol].value for aggcol in self.aggcols]):
        #             allsame = False
        #             break

        # if allsame or std == 0:
        #     return True
        if bad_stats.std == 0:
            return True


        weight = self.weight(max(bad_stats.vals))
        if weight == 0:
            return True
        threshold = (self.global_bounds[1] - self.global_bounds[0]) * self.epsilon / weight
        bounds = max(bad_stats.vals) - min(bad_stats.vals)
        bounds = max(bounds, bad_stats.std * 2.58 * 2)
        return bounds < threshold
        #w = self.weight(est + 2.58 * std)        
        wmse = np.mean([self.weight(v) * (abs(v - bad_stats.est))**2 for v in bad_stats.vals])
        return wmse < self.epsilon * (self.global_bounds[1] * 0.8)



class SampleDecisionTree(object):
    """
    Same, but using the refiner instead of custom splitting!
    """

    def __init__(self, full_table, bad_tables, good_tables, bad_err_funcs, good_err_funcs, aggcols, **kwargs):
        """
        Add new meta attribute to table to recore error score
        """
        self.attrs = []  # attrs, types (descrete, cont), and vals (values, range)
        self.aggcols = aggcols
        self.SCORE_ID = Orange.feature.Descriptor.new_meta_id()
        self.errprob = [kwargs.get('errprob', 0.001)]
        self.min_table_size = kwargs.get('min_table_size', 50)
        self.complexity_multiplier = kwargs.get('complexity_multiplier', 1.2)
        self.max_levels = kwargs.get('max_levels', 70)

        


        self.full_table = full_table        
        self.bad_tables = list(bad_tables)
        self.good_tables = list(good_tables)
        self.cur_tables = list(self.bad_tables)
        self.bad_err_funcs = bad_err_funcs
        self.good_err_funcs = good_err_funcs


        self.base_rule = SDRule(self.full_table, None)

        
        #self.base_table = table    # this is never modified
        #self.err_func = err_func

        for table in chain(self.bad_tables, self.good_tables):
            table.domain.addmeta(self.SCORE_ID, Orange.feature.Continuous(SCORE_VAR))
            table.add_meta_attribute(self.SCORE_ID, -inf)


        for ef, table in zip(chain(bad_err_funcs, good_err_funcs),
                             chain(bad_tables, good_tables)):
            ef.setup(table)


        self.bad_evaluator = Evaluator(self.SCORE_ID,
                                       self.errprob,
                                       self.bad_err_funcs,
                                       aggcols,
                                       kwargs.get('epsilon', 0.1))
        self.good_evaluator = Evaluator(self.SCORE_ID,
                                        self.errprob,
                                       self.good_err_funcs,
                                       aggcols,
                                       kwargs.get('epsilon', 0.1))


        self.stats = []
        self.refiners = [('ref2', BeamRefiner(attrs=aggcols))
                         #('ref3', BeamRefiner(attrs=aggcols, fanout=3))
                         #('ref5', BeamRefiner(attrs=aggcols, fanout=5))
                         ]


        self.score_mode = 'neg'
        
        self.rules = set()
        
        _logger.debug("%s with aggcols: %s", self.__class__.__name__, aggcols)

    def __call__(self):
        self.score_mode = 'neg'
        self.quadtree_score(self.base_rule.clone())

        # print "refining!!\n"
        
        # self.score_mode = 'pos'
        # for rule in list(self.rules):
        #     self.quadtree_score(rule.clone())
        # somehow simplify resulting rules!


    def stop(self, prev_rule, cur_stats, good_stats, tables):
        if len(prev_rule.examples) == 0:
            prev_rule.quality = -inf
            return

        for table in tables:
            for row in table:
                if row[self.SCORE_ID].value == -inf:
                    row[self.SCORE_ID] = cur_stats.est

        score = self.get_score(cur_stats, good_stats, prev_rule.complexity, 'pos')
        if not math.isnan(score):
            prev_rule.quality = prev_rule.score = score
            self.rules.add(prev_rule)
        print prev_rule.quality, '\t', np.mean(map(len,tables)), '\t', prev_rule



    def get_key_func(self, bounds=None, nbuckets=100):
        if bounds:
            bmax, bmin = bounds[1], bounds[0]
        else:
            bmax = max(#self.good_evaluator.global_bounds[1],
                       [self.bad_evaluator.global_bounds[1]])
            bmin = min(#self.good_evaluator.global_bounds[0],
                       [self.bad_evaluator.global_bounds[0]])
        if bmax == bmin:
            return None

        splits = [bmin]
        for i in xrange(30):
            splits.append(splits[-1] + (bmax - splits[-1]) / 2.5)

        def f(v):
            for i, split in enumerate(splits):
                if v <= split:
                    break
            return i
        return f
            
        bucketsize = (bmax - bmin) / float(nbuckets)
        # map evaluator score -> bucket
        keyf = lambda v: math.floor((v - bmin) / bucketsize)
        #keyf = lambda v: v > (bmax - bmin) * 0.9 + bmin and 1 or 0
        return keyf

    def evaluate_split(self, split, mode='neg'):
        (refname, attr), (rules, stats_list) = split

        
        
        bad_stats, good_stats = zip(*stats_list)
        bad_stats = filter(lambda x:x, bad_stats)
        good_stats = filter(lambda x:x, good_stats)
        if not bad_stats:
            return -inf

        bad_vals = [s.vals for s in bad_stats]
        good_vals = [s.vals for s in good_stats]

        keyf = self.get_key_func()#bounds=[min(map(min, bad_vals)), max(map(max, bad_vals))])
        if not keyf:
            return -inf


        bad_igr = self.get_igr(bad_stats, keyf) if bad_stats else -inf
        good_igr = self.get_igr(good_stats, keyf) if good_stats else 0

        
        comp_mult = 1.
        complexity = bad_stats[0].complexity
        if complexity:
            comp_mult = complexity ** self.complexity_multiplier
            params = bad_igr, good_igr, comp_mult, str([len(bs.vals) for bs in bad_stats]), attr.name
            print '%.4f\t%.4f\t%d\t%s\t%s' % params

        return (bad_igr - good_igr) / comp_mult
        
        
    def get_igr(self, stats, keyf=None):
        keyf = keyf or self.get_key_func()
        stats = filter(lambda x:x, stats)
        H = self.entropy(chain(*[stat.vals for stat in stats]), keyf)
        N = float(sum(map(len, (stat.vals for stat in stats))))

        ig, iv = 0., 0.
        for stat in stats:
            c = len(stat.vals)
            ig += c / N * self.entropy(stat.vals, keyf)
            iv += c / N * math.log(c / N, 2)
        
        ig = H - ig
        iv *= -1
        # if iv == 0:
        #     return 0.
        return ig# / iv
        return ig / iv

    def entropy(self, vals, keyf=None):
        keyf = keyf or self.get_key_func()
        c = Counter(map(keyf, vals))
        n = float(sum(c.values()))

        if n == 0:
            return 0
        return -sum(count / n * math.log(count / n ,2) for count in c.values() if count > 0)
                
    def get_score(self, bad_stat, good_stat, complexity, mode='neg', nstd=1.):
        comp_mult = 1.
        if complexity:
            comp_mult = self.complexity_multiplier * complexity


        good_prob = 0
        if mode == 'neg':
            bad_prob = -bad_stat.est
            if good_stat:
                good_prob = -good_stat.est
        else:
            bad_prob = bad_stat.est
            if good_stat:
                good_prob = good_stat.est

        good_prob = max(0, good_prob)
        
        return (bad_prob - good_prob) / comp_mult
        
        # Intrinsic Value
        # ratio_a = |pts.attr=a| / |pts|
        # - SUM_attr=a  ratio * log(ratio)
        

        good_prob = 0
        if mode == 'neg':
            bad_prob = -(bad_stat.est + 2.58 * bad_stat.std)
            if good_stat:
                good_prob = -(good_stat.est + 2.58 * good_stat.std)

        else:
            bad_prob = (bad_stat.est + nstd * bad_stat.std)
            if good_stat:
                good_prob = (good_stat.est + nstd * good_stat.std)

        good_prob = abs(good_prob)
        
        return (bad_prob - good_prob) / comp_mult


    def get_best_split(self, prev_rule, cur_score, bad_tables, good_tables):
        def compute_stats(rule, evaluate, tables):
            stats = None
            samples = map(rule.filter_table, tables)
            if sum(map(len, samples)):
                stats = evaluate(samples, sample=False)
            
            return stats

        N = sum(map(len, bad_tables))

        # apply rules to sample table
        splits = defaultdict(lambda: (list(), list()))

        for refname, refiner in  self.refiners:

            # sanity check that sum of table filtered by attr rules == original table
            attr_counts = defaultdict(list)

            for attr, rule in refiner(prev_rule):
                
                key = (refname, attr)
                
                bad_stats = compute_stats(rule, self.bad_evaluator.evaluate, bad_tables)
                good_stats = compute_stats(rule, self.good_evaluator.evaluate, good_tables)

                if bad_stats:
                    bad_stats.__dict__['complexity'] = rule.complexity
                    attr_counts[attr].append(bad_stats)

                splits[key][0].append(rule)
                splits[key][1].append((bad_stats, good_stats))


            # for attr, sl in attr_counts.iteritems():
            #     NN = sum(len(s.vals) for s in sl)
            #     assert NN == N, "fuck this shit %s\t%d\t%d" % (attr.name, NN, N)

        # if there are any discrete attributes that strictly simplify the rule,
        # use it
        nexts = []
        for (_, attr), (rules, stats_list) in splits.items():
            if (len(filter(lambda x:x, zip(*stats_list)[0])) == 1 and 
                attr.var_type ==  Orange.feature.Type.Discrete and
                rules[0].complexity == prev_rule.complexity):
                print "short cutting", attr, prev_rule
                nexts.append((rules, attr, stats_list))
        if nexts:
            return nexts


        key_func = lambda split: self.evaluate_split(split, self.score_mode)
        splits = sorted(splits.items(), key=key_func, reverse=True)
        evals = map(key_func, splits)
        rules, stats_list = zip(*zip(*splits)[1])
        all_bad_stats = [zip(*sl)[0] for sl in stats_list]
        all_good_stats = [zip(*sl)[1] for sl in stats_list]        

        print len(prev_rule.examples), prev_rule
        print 'split\t', [xx.name for xx in zip(*zip(*splits)[0])[1]]
        print '     \t', [np.std([bs.est for bs in bss if bs]) for bss in all_bad_stats]
        print '     \t', evals
        print '     \t', [self.get_igr(bss) for bss in all_bad_stats]
        print '     \t', [self.get_igr(bss) for bss in all_good_stats]



        if max(evals) <= 0:
            return []


        prev_score = None
        nexts = []
        
        for ((refname, attr), (rules, stats_list)), score in zip(splits, evals):
            counts = map(lambda r: len(r.examples), rules)

            if len(filter(lambda n: n, counts)) < 1:
                continue
            if not len(zip(rules, stats_list)):
                continue
            if prev_score is not None and score != prev_score:
                break
            nexts.append(( rules, attr, stats_list ))
            prev_score = score

        return nexts


    def quadtree_score(self, prev_rule):
        if prev_rule.complexity > self.max_levels: raise



        bad_tables = map(prev_rule.filter_table, self.bad_tables)
        good_tables = map(prev_rule.filter_table, self.good_tables)



        # evaluate current partition using sample
        cur_stats = self.bad_evaluator.evaluate(bad_tables, True)
        good_stats = self.good_evaluator.evaluate(good_tables, True)

        bad_samples = self.bad_evaluator.samples
        good_samples = self.good_evaluator.samples


        # _logger.info("Stats: %s\tpop(%d)\tsamp(%d)\t%f-%f\t%f-%f",
        #              should_stop,
        #              len(table),
        #              len(bad_samples),
        #              cur_stats.est-2.58*cur_stats.std,
        #              cur_stats.est+2.58*cur_stats.std,
        #              min(cur_stats.vals),
        #              max(cur_stats.vals))


        should_stop = self.bad_evaluator.should_stop(bad_tables, cur_stats, good_stats)
        if should_stop:
            self.stop(prev_rule, cur_stats, good_stats, bad_tables)
            return

        cur_score = self.get_score(cur_stats, good_stats, prev_rule.complexity, 'pos', 0.05)
        if not math.isnan(cur_score):
            prev_rule.quality = prev_rule.score = cur_score
            self.rules.add(prev_rule)

        
        bests = self.get_best_split(prev_rule, cur_score, bad_tables, good_tables)#bad_samples, good_samples)
        if not bests:
            self.stop(prev_rule, cur_stats, good_stats, bad_tables)
            _logger.info("couldn't find a split.  giving up")
            print '%d\t%.5f\t%.5f\t' % (np.mean(map(len,bad_tables)), cur_stats.est, self.errprob[-1]), prev_rule
            return


        
        for best_rules, best_attr, best_stats_list in bests:

            # check whether or not the best split is worth pursuing
            best_scores = [self.get_score(bs, gs, rule.complexity, 'pos', 0.05)
                           for rule, (bs, gs) in zip(best_rules, best_stats_list) if bs]
            best_score = max(best_scores)

            if best_score * 1.25 < cur_score and prev_rule.complexity > 0:
                print "not good enough, done", best_attr
                print "\t", prev_rule
                print "\t", map(str, best_rules)
                print "\t", best_score, cur_score
                self.stop(prev_rule, cur_stats, good_stats, bad_tables)
                continue



            _logger.info("Splitting on %s", best_attr.name)
            for next_rule, stats in zip(best_rules, best_stats_list):
                partition = next_rule.examples
                if not len(partition):
                    continue

                if len(self.stats) > 0:
                    ratio = (self.stats[-1].std / cur_stats.std)
                    newerrprob = self.errprob[-1] * max(ratio , 1)
                    #newerrprob = self.errprob[-1] * len(table) / (len(partition) * 1.3)
                else:
                    newerrprob = self.errprob[-1]


                self.stats.append(cur_stats)
                self.errprob.append(min(1.0, newerrprob))
                self.quadtree_score(next_rule)
                self.stats.pop()
                self.errprob.pop()



            self.sanity(best_rules)

    def sanity(self, rules):
        return
        # sanity check, sum(partitions) == table
        part_sizes = map(len, [r.examples for r in rules])
        total_part_size = sum( part_sizes )
        msg = "Partition sizes wrong: %d!=%d\t%s\t%s"
        msg = msg % (total_part_size,
                     np.mean(map(len,tables)),
                     str(part_sizes),
                     best_attr.name)

        try:
            assert total_part_size == len(table), msg
        except:
            pass
