import pdb
import math

from ...util import *


_logger = get_logger()

class ScoreNormalizer(object):

    def __call__(self, all_rules):
        for r in all_rules:
            r.quality = r.score
        return True



class RankNormalizer(object):

    def __call__(self, all_rules):
        """
        Rank the rules and assign the partial ranking 
        Use position in partial ranking as normalized 0-1 score
        compute F1 score
        """
        rules_score = sorted(all_rules, key=lambda r: r.score)
        rules_mean = sorted(all_rules, key=lambda r: r.stats_mean)
        rules_nmean = sorted(all_rules, key=lambda r: r.stats_nmean, reverse=True)

        _logger.info("measure bounds: score: %s\tmean: %s\tnmean: %s",
                     '%s\t%s' % (min(rules_score), max(rules_score)),
                     '%s\t%s' % (min(rules_mean), max(rules_mean)),
                     '%s\t%s' % (min(rules_nmean), max(rules_nmean)))
        

        mean_idx, nmean_idx, score_idx = 1., 1., 1.
        for idx, (rule, nrule, srule) in enumerate(zip(rules_mean, rules_nmean, rules_score)):
            if idx > 0:
                if rule.stats_mean != rules_mean[idx-1].stats_mean:
                    mean_idx += 1
                if nrule.stats_nmean != rules_nmean[idx-1].stats_nmean:
                    nmean_idx += 1
                if srule.score != rules_score[idx-1].score:
                    score_idx += 1
            rule.stats_meannorm = mean_idx
            nrule.stats_nmeannorm = nmean_idx
            nrule.score_norm = score_idx

        for rule in all_rules:
            s = rule.score_norm / score_idx  # normalized score
            p = rule.stats_meannorm / mean_idx  # precision
            r = rule.stats_nmeannorm / nmean_idx # recall
            f1 = 2. * (p * r) / (p + r)
            rule.quality = 2. * (s * f1) / (s + f1)
            rule.quality = rule.quality * rule.weight
        return True





class BoundedNormalizer(object):
    def __init__(self, *args, **kwargs):
        self.score_bounds = None
        self.mean_bounds = None
        self.nmean_bounds = None

    def update_bounds(self, f, all_rules, bounds):
        maxv, minv = None, None
        for rule in all_rules:
            v = f(rule)
            if v is not None:
                maxv = v if maxv is None else max(maxv, v)
                minv = v if minv is None else min(minv, v)
        if not maxv:
            return None

        if bounds:
            ret = [min(bounds[0], minv), max(bounds[1], maxv)]
        else:
            ret = [minv, maxv]

        return ret

    def normalize(self, v, bounds):
        if not bounds or  bounds[1] - bounds[0] == 0:
            return 1.
        return (v - bounds[0]) / float(bounds[1] - bounds[0])
        

    def __call__(self, all_rules, b1=1., b2=1.):
        # update bounds
        f_score = lambda r: r.score
        f_mean = lambda r: r.stats_mean + 2.698 * r.stats_std
        f_nmean = lambda r: r.stats_nmean - 2.698 * r.stats_nstd

        filtered_rules = []
        for rule in all_rules:
            if rule.score is None or rule.stats_mean is None or rule.stats_nmean is None:
                rule.quality = None
                continue
            filtered_rules.append(rule)
        all_rules = filtered_rules
            
        self.score_bounds = self.update_bounds(f_score, all_rules, self.score_bounds)
        self.mean_bounds = self.update_bounds(f_mean, all_rules, self.mean_bounds)
        self.nmean_bounds = self.update_bounds(f_nmean, all_rules, self.nmean_bounds)
        
        if not (self.score_bounds and self.mean_bounds and self.nmean_bounds):
            return False

        _logger.info("measure bounds: score: %s\tmean: %s\tnmean: %s",
                     '%s\t%s' % tuple(self.score_bounds),
                     '%s\t%s' % tuple(self.mean_bounds),
                     '%s\t%s' % tuple(self.nmean_bounds))
        

        # normalize row stats based on these bounds
        for rule in all_rules:
            if rule.score is None:
                rule.quality = None
                continue

            s = self.normalize(f_score(rule), self.score_bounds)
            p = self.normalize(f_mean(rule), self.mean_bounds)
            r = 1. - self.normalize(f_nmean(rule), self.nmean_bounds)
            
            f1 = (1+b1) * (p * r) / (b1*b1*p + r) if p+r != 0 else 0.
            rule.quality = (1+b2) * (s * f1) / (b2*b2*s + f1) if s+f1 != 0 else 0.0
            rule.quality = rule.quality * rule.weight
            rule.quality = 0. if math.isnan(rule.quality) else rule.quality
        return True
