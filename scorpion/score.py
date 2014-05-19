import numpy as np
import Orange
import orange
import math
import random
import pdb

from itertools import chain

from settings import *
from util import *
from split import *
from aggerror import *
from functions import StdFunc, AvgFunc
from learners.cn2sd.refiner import *

_logger = get_logger()
inf = 1e10000000


class Scorer(object):
    def __init__(self, table, aggcols, err_func, **kwargs):
        """
        Add new meta attribute to table to recore error score
        """
        self.attrs = []  # attrs, types (descrete, cont), and vals (values, range)
        self.base_table = table    # this is never modified
        self.err_func = err_func
        self.aggcols = aggcols
        self.SCORE_ID = Orange.feature.Descriptor.new_meta_id()
        
        table.domain.addmeta(self.SCORE_ID, Orange.feature.Continuous(SCORE_VAR))
        table.add_meta_attribute(self.SCORE_ID, -inf)

        self.err_func.setup(table)
    
    def __call__(self):
        for row in self.base_table:
            if row[self.SCORE_ID].value == -inf:
                row[self.SCORE_ID] = self.err_func([row])



class PartitionStats(object):
    def __init__(self, est, **kwargs):
        self.est = est
        self.__dict__.update(kwargs)

class QuadScore(Scorer):
    def __init__(self, table, aggcols, err_func, **kwargs):
        super(QuadScore, self).__init__(table, aggcols, err_func, **kwargs)

        self.cur_tables = [table]  # this is modified

        # setup quad tree related stuff
        self.max_levels = 70#math.ceil( math.log(len(table), 2) )
        self.min_points = kwargs.get('min_points', 2)
        self.cutoff = kwargs.get('cutoff', [0.4, 0.9])
        self.errprob = [kwargs.get('errprob', 0.001)]

        
    
    def possible_splits(self, cur_table):
        raise NotImplementedError
    def should_stop(self, table, stats):
        raise NotImplementedError
    def evaluate(self, table):
        raise NotImplementedError
    def quadtree_score(self, table, level=0):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


        


class QuadScoreSample1(QuadScore):
    """
    Samples each child partition separately.
    """
    def __call__(self):
        for table in self.cur_tables:
            self.quadtree_score(table)
        

    def possible_splits(self, cur_table):
        for attr in cur_table.domain:
            # if attr.name == 'id':
            #     continue
            if attr.name in self.aggcols:
                continue
            if attr.var_type == orange.VarTypes.Discrete:
                # extract the unique values actually in the table (sample)
                # then add the rest of the values in the attribute randomly

                col = cur_table.select([attr])
                col.remove_duplicates()
                idxs = Orange.data.sample.SubsetIndices2(p0=0.5)(col)
                f = lambda row: row[0]
                left = map(f, col.select_ref(idxs))
                right = map(f, col.select_ref(idxs,  negate=True))
                both = map(f, col)

                rest = filter(lambda v: v not in both, attr.values)
                rest = map(lambda v: Orange.data.Value(attr, v), rest)
                for val in rest:
                    if random.random() < 0.5:
                        left.append( val )
                    else:
                        right.append( val )

                # naiively split values in domain in half
                # this works very poorly
                #col = attr.values
                #idxs = [random.random() < 0.5 and 1 or 0 for i in col]
                #left = [Orange.data.Value(attr, pos) for pos, idx in enumerate(idxs) if idx]
                #right = [Orange.data.Value(attr, pos) for pos, idx in enumerate(idxs) if not idx]
                
                if len(left) == 0 or len(right) == 0:
                    continue
                yield DiscreteSplit(cur_table, attr, left, right)
            else:
                # compute min/max
                # split in half
                maxv = max(cur_table, key=lambda r: r[attr].value)[attr]
                minv = min(cur_table, key=lambda r: r[attr].value)[attr]
                midv = (maxv + minv) / 2.
                yield ContSplit(cur_table, attr, midv, minv, maxv)

    def should_stop(self, table, stats):
        if len(table) <= self.min_points:
            return True

        std, est = stats.std, stats.est
        val, allsame = None, True
        for i, row in enumerate(table):
            if i == 0:
                val = tuple([row[aggcol].value for aggcol in self.aggcols])
            else:
                if val != tuple([row[aggcol].value for aggcol in self.aggcols]):
                    allsame = False
                    break
        if allsame:
            return True

        if stats.vals:
            vals = stats.vals
            
            # XXX: Needs to be fixed to
            # 1) see if values are in top X% of known range
            # 2) values are within bounds of current estimates
            # The following just *happens* to work well for anomalous values
            withinbounds = (max(stats.vals) <= est + 2.58 * std and
                            min(stats.vals) >= est - 2.56 * std)
            withinbounds = (max(stats.vals) <= est + 2.56 * std)
            withinbounds = (max(stats.vals) <= est + 1.25 * std)
        else:
            withinbounds = True


        # threshold goes from 0.2 to 0.01 for est from 0 to 1
        threshold = est * (0.001 - 0.2) + 0.2
        below_threshold = std < threshold

        if withinbounds and below_threshold:
            return True
        return False

    def kn(self, n):
        """
        return Kn, where UMVU estimator of std is Kn*S
        """
        try:
            return math.sqrt(2./(n-1)) * (math.gamma(n/2.) / (math.gamma((n-1.)/2.)))
        except:
            return 1.

    def evaluate(self, table):
        if len(table) == 1:
            row = table[0]
            if row[self.SCORE_ID].value == -inf:
                #pred = ids_filter(table, [row['id']], negate=True)
                row[self.SCORE_ID] = self.err_func([row])
            est = row[self.SCORE_ID]
            return PartitionStats(est,
                                  std=0.,
                                  vals=[est])

        pop = len(table)
        samp_size = sample_size(pop=pop) or pop
        p0 = float( samp_size ) / pop
        indices2 =  Orange.data.sample.SubsetIndices2(p0=p0)
        idxs = indices2(table) if samp_size > 1 else [0]
        samples = table.select_ref( idxs, negate=True )

        vals = []
        for row in samples:
            if row[self.SCORE_ID].value == -inf:
                row[self.SCORE_ID] = self.err_func([row])
            vals.append( row[self.SCORE_ID].value )
        samp_size = len(vals)

        est = np.mean(vals)

        S2 = 1. / (samp_size - 1) * sum([(v-est)**2 for v in vals])
        S = math.sqrt(S2)
        std = self.kn(samp_size) * S
        
        _logger.debug( '\tsampsize(%d)\t%f\t%f-%f\t%f-%f\t%f',
                       samp_size,
                       est,
                       min(vals),
                       max(vals),
                       est - 2.58*std,
                       est + 2.58*std,
                       std)
        return PartitionStats(est, std=std, vals=vals)

    def evaluate_split(self, stats_list):
        # want: identify higher error points
        #       partitions that cannot have error points
        # no:   partitions that have error and non-error points
        # what is the cut off for error?
        # goal: minimize overlap within cutoff range
        aggest = 0.0
        for stats in stats_list:
            std, est = stats.std, stats.est
            
            estmin, estmax = est-2.58*std, est+2.58*std
            if estmax < self.cutoff[0] or self.cutoff[1] < estmin:
                overlap = 0.
            else:
                omin = max(estmin, self.cutoff[0])
                omax = min(estmax, self.cutoff[1])
                overlap = omax - omin
            aggest += std
        return aggest



    def quadtree_score(self, table, level=0):
        # at this step, need to compute sample size
        # so that with 95% confidence, the margin
        # of error is within 10% of the true value.
        if level > self.max_levels: raise

        best_split, best_est, best_parts = None, None, []
        for split in self.possible_splits(table):
            _logger.debug("Checking Split\t%s\t%s", split.attr.name,
                          ','.join(map(str,map(len, split()))))
            partinfo = []
            for partition in split():
                if len(partition) == 0:
                    continue

                stats = self.evaluate(partition)
                partinfo.append( stats )

            split_est = self.evaluate_split(partinfo)

            if len(partinfo) > 1 and (not best_split or split_est < best_est):
                best_split = split
                best_est = split_est
                best_parts = partinfo

        if not best_split:
            raise
        
        _logger.info( "quadtree\t%d\t%d\t%s\t%f",
                      len(table),
                      level, 
                      best_split.attr.name,
                      best_est)

        for partition, stats in zip(best_split(table), best_parts):
            if self.should_stop(partition, stats ):
                _logger.info( "stopped on partition\t%d\test(%f)\tstd(%f)",
                              len(partition),
                              stats.est,
                              stats.std)
                
                for row in partition:
                    if row[self.SCORE_ID].value == -inf:
                        row[self.SCORE_ID] = stats.est
            else:
                self.errprob.append( min(1.0, self.errprob[-1] * len(table) / len(partition)) )
                self.quadtree_score(partition, level+1)
                self.errprob.pop()



class QuadScoreSample2(QuadScoreSample1):
    """
    Constructs a single sample and uses it to evaluate children
    """
    def __call__(self):
        for table in self.cur_tables:
            self.quadtree_score(table)

    def get_sample_size(self, pop, *args, **kwargs):
        return sample_size(pop)

    def evaluate(self, table):
        if len(table) == 1:
            row = table[0]
            if row[self.SCORE_ID].value == -inf:
                row[self.SCORE_ID] = self.err_func([row])
            est = row[self.SCORE_ID]
            return PartitionStats(est,
                                  std=0.,
                                  vals=[est])
        

        vals = []
        for row in table:
            if row[self.SCORE_ID].value == -inf:
                est = self.err_func([row])
                row[self.SCORE_ID] = est
            vals.append( row[self.SCORE_ID].value )
        samp_size = len(vals)

        est = np.mean(vals)

        S2 = 1. / (samp_size - 1) * sum([(v-est)**2 for v in vals])
        S = math.sqrt(S2)
        std = self.kn(samp_size) * S

        _logger.debug( '\tsampsize(%d)\t%f\t%f-%f\t%f-%f\t%f',
                       samp_size,
                       est,
                       min(vals),
                       max(vals),
                       est - 2.58*std,
                       est + 2.58*std,
                       std)

        return PartitionStats(est, std=std, vals=vals)

    def get_samples(self, table):
        pop = len(table)
        samp_size = self.get_sample_size(pop, self.errprob[-1]) or pop
        p0 = float(samp_size) / pop
        indices2 = Orange.data.sample.SubsetIndices2(p0=p0)
        idxs = indices2(table) if pop > 1 else [0]
        samples = table.select_ref(idxs, negate=True)
        return samples

    
    def quadtree_score(self, table, prev_splits=None):
        prev_splits = prev_splits or []
        if len(prev_splits) > self.max_levels: raise

        samples = self.get_samples(table)

        # evaluate current partition using sample
        cur_stats = self.evaluate(samples)
        should_stop = self.should_stop(samples, cur_stats)
        _logger.info("Stats: %s\tpop(%d)\tsamp(%d)\t%f-%f\t%f-%f",
                     should_stop,
                     len(table),
                     len(samples),
                     cur_stats.est-2.58*cur_stats.std,
                     cur_stats.est+2.58*cur_stats.std,
                     min(cur_stats.vals),
                     max(cur_stats.vals))
        if should_stop:
            for row in table:
                if row[self.SCORE_ID].value == -inf:
                    row[self.SCORE_ID] = cur_stats.est
            return

        # ok find the best split
        best_split, best_est, best_stats = None, None, []
        for split in self.possible_splits(samples):
            _logger.debug("Checking Split\t%s", str(split))
            
            stats_list = []
            for partition in split(samples):
                if len(partition) == 0:
                    continue

                stats = self.evaluate(partition)
                stats_list.append( stats )

            split_est = self.evaluate_split(stats_list)
            
            if len(stats_list) > 1 and (not best_split or  split_est < best_est):
                best_split = split
                best_est = split_est
                best_stats = stats_list

        if not best_split:
            raise

        _logger.info("Splitting on %s\t%s", best_split.attr.name,
                      ','.join(map(str,map(len, best_split(table)))))
        

        for partition in best_split(table):
            prev_splits.append(best_split)
            self.errprob.append( min(1.0, self.errprob[-1] * len(table) / len(partition)) )
            self.quadtree_score(partition, prev_splits=prev_splits)
            prev_splits.pop()
            self.errprob.pop()

        # sanity check, sum(partitions) == table
        part_sizes = map(len, best_split(table))
        total_part_size = sum( part_sizes )
        msg = "Partition sizes wrong: %d!=%d\t%s\t%s"
        msg = msg % (total_part_size,
                     len(table),
                     str(part_sizes),
                     str(best_split))
        assert total_part_size == len(table), msg


    
    
    
class QuadScoreSample3(QuadScoreSample2):
    """
    Uses best_sample_size instead o sample_size
    """
    def get_sample_size(self, pop, *args, **kwargs):
        return best_sample_size(pop, *args, **kwargs)
    


class QuadScoreSample4(QuadScoreSample3):
    """
    Only evaluates attributes used in the aggregate function
    """
    def possible_splits(self, cur_table):
        for attr in cur_table.domain:
            if attr.name not in self.aggcols:
                continue

            if attr.var_type == orange.VarTypes.Discrete:
                # extract the unique values actually in the table (sample)
                # then add the rest of the values in the attribute randomly

                col = cur_table.select([attr])
                col.remove_duplicates()
                idxs = Orange.data.sample.SubsetIndices2(p0=0.5)(col)
                f = lambda row: row[0]
                left = map(f, col.select_ref(idxs))
                right = map(f, col.select_ref(idxs,  negate=True))
                both = map(f, col)

                rest = filter(lambda v: v not in both, attr.values)
                rest = map(lambda v: Orange.data.Value(attr, v), rest)
                for val in rest:
                    if random.random() < 0.5:
                        left.append( val )
                    else:
                        right.append( val )
                
                if len(left) == 0 or len(right) == 0:
                    continue
                yield DiscreteSplit(cur_table, attr, left, right)
            else:
                # compute min/max
                # split in half
                maxv = max(cur_table, key=lambda r: r[attr].value)[attr]
                minv = min(cur_table, key=lambda r: r[attr].value)[attr]
                midv = (maxv + minv) / 2.
                yield ContSplit(cur_table, attr, midv, minv, maxv)

class QuadScoreSample5(QuadScoreSample4):
    """
    uses global std and mean
    """
    def __init__(self, *args, **kwargs):
        QuadScoreSample4.__init__(self, *args, **kwargs)
        self.global_std = StdFunc()
        self.global_mean = AvgFunc()
        self.global_bounds = [1e10000, -1e10000]
        self.epsilon = kwargs.get('epsilon', 0.005)
    
    def evaluate_split(self, stats_list):
        probs = []
        for stat in stats_list:
            if not stat:
                continue
            est, std = stat.est, stat.std
            if not len(stat.vals):
                prob = 0.
            elif std == 0:
                prob = 1.
            else:
                weight = self.weight(max(stat.vals))
                if weight == 0:
                    prob = 1.
                else:
                    bound = max(stat.vals) - min(stat.vals)
                    prob = (std * (2.58 + 2.58)) / weight
                    prob = 1 - prob  / (self.global_bounds[1] - self.global_bounds[0])
            #prob = est + 2.58 * std
            # if std == 0:
            #     prob = 1.
            # else:
            #     # Prob( (X-mean)^2 < epsilon ) >= 0.95
            #     w = self.weight(est + 2.58 * std)
            #     alpha = self.epsilon * abs(est) / w
            #     prob = math.erf(alpha / (std * math.sqrt(2.)))
            probs.append(prob)
        return np.mean(probs) if probs else 0.
            

    def evaluate(self, table):
        if not len(table):
            return PartitionStats(self.global_mean.value(),
                                  std=self.global_std.value(),
                                  vals=[])
        
        vals = []
        newvals = []
        for row in table:
            if row[self.SCORE_ID].value == -inf:
                est = self.err_func([row])
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
            est = np.mean(vals)
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
        return PartitionStats(est, std=std, vals=vals)

    def weight(self, val):
        u = self.global_mean.value()
        std = self.global_std.value()
        if std == 0:
            return 1.

        max_std = 2.58
        #max_std = 1.6

        # weight increases quadratically.
        nstds = (val - u) / std
        nstds = min(max(0, nstds), max_std)
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

    def should_stop(self, table, stats):
        if len(table) <= self.min_points:
            return True

        std, est = stats.std, stats.est
        val, allsame = None, True
        for i, row in enumerate(table):
            if i == 0:
                val = tuple([row[aggcol].value for aggcol in self.aggcols])
            else:
                if val != tuple([row[aggcol].value for aggcol in self.aggcols]):
                    allsame = False
                    break
        if allsame:
            return True

        # Prob( (X-mean)^2 < epsilon ) >= 0.95
        w = self.weight(est + 2.58 * std)
        if w == 0 or std == 0:
            prob = 1.
        else:
            alpha = math.sqrt( self.epsilon * abs(est) / w )
            #alpha = self.epsilon * 2.58 * self.global_std.value() / w
            #alpha = math.sqrt( self.epsilon * 2 * 2.58 * self.global_std.value() / w )
            prob = math.erf(alpha / (std * math.sqrt(2.)))

        return prob >= 0.95


class QuadScoreSample6(QuadScoreSample5):
    """
    Reuses previously computed samples, if possible
    """
    def get_samples(self, table):
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




class QuadScoreSample7(QuadScoreSample6):
    """
    Same, but using the refiner instead of custom splitting!
    """

    def __init__(self, table, aggcols, err_func, **kwargs):
        # attrs_to_rm = [attr.name for attr in table.domain
        #                if attr.name not in aggcols]
        # table = rm_attr_from_domain(table, attrs_to_rm)
        super(QuadScoreSample7, self).__init__(table, aggcols, err_func, **kwargs)
        self.refiners = [('ref2', BeamRefiner(attrs=aggcols))
                         #('ref3', BeamRefiner(attrs=aggcols, fanout=3))
                         #('ref5', BeamRefiner(attrs=aggcols, fanout=5))
                         ]
        self.stats = []
        self.min_table_size = kwargs.get('min_table_size', 50)
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.complexity_multiplier = kwargs.get('complexity_multiplier', 1.2)
        
        self.rules = set()
        
        _logger.debug("%s with aggcols: %s", self.__class__.__name__, aggcols)

    def __call__(self):
        for table in self.cur_tables:
            base_rule = SDRule(table, None)
            self.quadtree_score(base_rule)

    def evaluate_split(self, stats_list):
        probs = []
        for stat in stats_list:
            if not stat:
                continue
            
            est, std = stat.est, stat.std
            # complexity multiplier
            comp_mult = self.complexity_multiplier if not stat.complexity else 1.
            
            probs.append(-(est + 2.58 * std) * comp_mult)
            #probs.append(comp_mult * (est + 2.58*std))
            continue
            if not len(stat.vals):
                prob = 0.
            elif std == 0:
                prob = 1.
            else:
                weight = self.weight(max(stat.vals))
                if weight == 0:
                    prob = 1.
                else:
                    bounds = max(stat.vals) - min(stat.vals)
                    threshold = (self.global_bounds[1] - self.global_bounds[0]) / weight
                    prob = bounds / threshold


            # complexity multiplier
            comp_mult = 1.2 if stat.complexity else 1.
            probs.append(prob * comp_mult)

        
        return max(probs) if probs else 0.


    def should_stop(self, table, stats):
        if len(table) <= self.min_points:
            return True

        std, est = stats.std, stats.est
        val, allsame = None, True
        for i, row in enumerate(table):
            if i == 0:
                val = tuple([row[aggcol].value for aggcol in self.aggcols])
            else:
                if val != tuple([row[aggcol].value for aggcol in self.aggcols]):
                    allsame = False
                    break

        if allsame or std == 0:
            return True

        weight = self.weight(max(stats.vals))
        if weight == 0:
            return True
        threshold = (self.global_bounds[1] - self.global_bounds[0]) * self.epsilon / weight
        bounds = max(stats.vals) - min(stats.vals)
        bounds = max(bounds, std * 2.58 * 2)
        return bounds < threshold
        #w = self.weight(est + 2.58 * std)        
        wmse = np.mean([self.weight(v) * (abs(v - est))**2 for v in stats.vals])
        return wmse < self.epsilon * (self.global_bounds[1] * 0.8)
        



    def quadtree_score(self, prev_rule):
        if prev_rule.complexity > self.max_levels: raise


        table = prev_rule.examples
        
        samples = self.get_samples(table)

        # evaluate current partition using sample
        cur_stats = self.evaluate(samples)
        should_stop = self.should_stop(samples, cur_stats) or len(samples) == len(table)
        _logger.info("Stats: %s\tpop(%d)\tsamp(%d)\t%f-%f\t%f-%f",
                     should_stop,
                     len(table),
                     len(samples),
                     cur_stats.est-2.58*cur_stats.std,
                     cur_stats.est+2.58*cur_stats.std,
                     min(cur_stats.vals),
                     max(cur_stats.vals))


        if should_stop:
            for row in table:
                if row[self.SCORE_ID].value == -inf:
                    row[self.SCORE_ID] = cur_stats.est

            prev_rule.quality = prev_rule.score = cur_stats.est
            self.rules.add(prev_rule)
            print prev_rule.quality, '\t', len(table), '\t', prev_rule
            return

        # apply rules to sample table
        splits = defaultdict(lambda: (list(), list()))
        for refname, refiner in  self.refiners:
            for attr, new_rule in refiner(prev_rule):
                key = (refname, attr)
                partition = new_rule.filter_table(samples)
                stats = self.evaluate(partition) if len(partition) else None

                if stats:
                    stats.__dict__['complexity'] = new_rule.complexity - prev_rule.complexity
                    
                splits[key][0].append(new_rule)
                splits[key][1].append(stats)
                

        # if 'recipient_nm = RATHBUN JESSICA, GMMB INC., AMLIN JEFFREY, SHUMAKER PDT (2+ ..)' in str(prev_rule):
        #     scores = [(k[1].name, self.evaluate_split(s)) for (k,(r,s)) in splits.iteritems()]
        #     scores.sort(key=lambda p: p[1], reverse=True)
        #     complexities = [(k[1].name, np.mean([s.complexity for s in sl])) for (k,(r,sl)) in splits.iteritems()]
        #     pdb.set_trace()
            
        splits = sorted(splits.items(),
                        key=lambda (key, (rules, statslist)): self.evaluate_split(statslist),
                        reverse=True)


        best_split = None
        for (refname, attr), (rules, stats_list) in splits:
            counts = map(lambda r: len(r.examples), rules)
            if len(filter(lambda n: n, counts)) <= 1:
                continue

            best_split = ((refname, attr), (rules, stats_list))
            break

        if not best_split:
            best_split = splits[random.randint(0, len(splits)-1)]
            #print '\n'.join(map(str,best_split[1][0]))
            #print
            #pdb.set_trace()
            # _logger.info("couldn't find a split.  giving up")
            # self.evaluate(table)
            # print '%d\t%.5f\t%.5f\t' % (len(table), cur_stats.est, self.errprob[-1]), prev_rule
            # return


        
        (refname, attr), (rules, stats_list) = best_split
        _logger.info("Splitting on %s", attr.name)
        for next_rule, stats in zip(rules, stats_list):
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


        # sanity check, sum(partitions) == table
        part_sizes = map(len, [r.examples for r in rules])
        total_part_size = sum( part_sizes )
        msg = "Partition sizes wrong: %d!=%d\t%s\t%s"
        msg = msg % (total_part_size,
                     len(table),
                     str(part_sizes),
                     attr.name)

        #assert total_part_size == len(table), msg



        



def score_inputs(table, aggerr, klass=Scorer, **kwargs):
    """
    @param klass the class of the scorer to use.  defaults to
    exhaustive 
    """
    agg = aggerr.agg
    err_func = aggerr.error_func
    cols = list(agg.cols)
    torm = [attr.name for attr in table.domain if attr.name not in cols and attr.name != 'v']


    table = rm_attr_from_domain(table, ['err'])
    qscore = klass(table, cols, err_func, **kwargs)
    qscore()

    _logger.info( "score_inputs: %s\tCalled error function %d times",
                  klass.__name__,
                  err_func.ncalls)
    scores = [ row[qscore.SCORE_ID].value for row in table ]
    # center and normalize scores
    mins, maxs = min(scores), max(scores)
    rs = maxs - mins
    _logger.info( "score_inputs: score range: [%f, %f]",
                  mins, maxs)
    scores = map(lambda s: (s - mins) / rs, scores )    
    
    return scores, err_func.ncalls
    


