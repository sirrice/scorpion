import time
import pdb
import sys
import Orange
import orange
import heapq
sys.path.extend(['.', '..'])

from collections import deque
from itertools import chain
from multiprocessing import Process, Queue, Pool, Pipe
from Queue import Empty

from ..learners.cn2sd.rule import *
from ..learners.cn2sd.refiner import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *
from ..util import *
from ..settings import *

from basic import Basic
from sampler import Sampler
from merger import Merger
from node import Node

inf = 1e10000000
_logger = get_logger()


def partition_f(name, params, tables, full_table, (inq, outq)):
  try:
    partitioner = BDTTablesPartitioner(**params)
    partitioner.setup_tables(tables, full_table)
    gen = partitioner()
    start = time.time()
    while not partitioner.is_done:
      try: 
        bound = inq.get(False)
        if bound:
          partitioner.update_inf_bound(bound)
      except Empty:
        pass

      pairs = []
      for node, isleaf in gen:
        rule = node.rule
        rule.quality = node.influence
        rule = rule.simplify(full_table)
        pairs.append((rule, isleaf))
        if len(pairs) >= 10: break

      if not pairs: 
        if not partitioner.is_done:
          _logger.debug("%s\tno nodes from generator but partitioner not done..."%name)
        continue

      bound = partitioner.get_inf_bound()

      dicts = [(r.to_json(), isleaf) for r, isleaf in pairs]
      _logger.debug("%s\tsend %d rules\t%s" % (name, len(pairs), map(hash, map(str, dicts))))
      outq.put((dicts, bound))
      _logger.debug("%s\tsent %s!" % (name, len(dicts)))
  except Exception as e:
    print e
    import traceback
    traceback.print_exc()

  _logger.debug("%s\tpartitioner DONE" % name)
  outq.put('done')
  outq.close()
  inq.close()



      








class BDTTablesPartitioner(Basic):

    def set_params(self, **kwargs):
        Basic.set_params(self, **kwargs)

        self.p = kwargs.get('p', 0.6)
        self.tau = kwargs.get('tau', [0.001, 0.09])
        self.epsilon = kwargs.get('epsilon', 0.005)
        self.partition_min_pts = 2
        self.min_pts = 5
        self.SCORE_ID = kwargs['SCORE_ID']
        self.inf_bounds = None 
        self.min_improvement = kwargs.get('min_improvement', .01)
        self.err_funcs = kwargs.get('err_funcs', None)


        self.max_wait = kwargs.get('max_wait', 2*60*60)
        self.start_time = None

        self.sampler = Sampler(self.SCORE_ID)
        self.seen = set()
        self.is_done = False


        if self.err_funcs is None:
          raise RuntimeError("errfuncs is none")


    @instrument
    def setup_tables(self, tables, merged):
        self.merged_table = merged
        self.tables = tables

        for ef, t in zip(self.err_funcs, self.tables):
          ef.setup(t)

        self.sampler = Sampler(self.SCORE_ID)
        self.samp_rates = [best_sample_size(len(t), self.epsilon)/(float(len(t))+1) for t in self.tables]

        if self.inf_bounds is None:
          self.inf_bounds = [[inf, -inf] for table in tables]

        # attributes to partition upon
        self.cont_attrs = [attr.name for attr in merged.domain if attr.name in self.cols and attr.var_type != Orange.feature.Type.Discrete]
        self.dist_attrs = [attr.name for attr in merged.domain if attr.name in self.cols and attr.var_type == Orange.feature.Type.Discrete]

        # remove undesirable columns 
        self.cont_attrs = filter(lambda c: c in self.cols, self.cont_attrs)
        self.dist_attrs = filter(lambda c: c in self.cols, self.dist_attrs)

    def get_inf_bound(self):
      bound = [inf, -inf]
      for inf_bound in self.inf_bounds:
        bound = r_union(bound, inf_bound)
      return bound

    def update_inf_bound(self, bound):
      for idx, inf_bound in enumerate(self.inf_bounds):
        self.inf_bounds[idx] = r_union(bound, inf_bound)

    def __call__(self, tables=None, full_table=None, root=None, **kwargs):
      if tables and full_table:
        self.setup_tables(tables, full_table)

      self.seen = set()
      if not root:
        root = Node(SDRule(self.merged_table, None))
      self.root = root

      samples = [self.sample(t, sr) for t, sr in zip(self.tables, self.samp_rates)]
      f = lambda (idx, samps): self.compute_infs(idx, samps)
      #all_infs = map(f, enumerate(self.tables))
      all_infs = map(f, enumerate(samples))

      for leaf in root.leaves:
        parent = leaf.parent
        leaf.parent = None
        #self.grow(leaf, self.tables, self.samp_rates, all_infs)
        for pair in self.grow(leaf, samples, self.samp_rates, all_infs):
          n, isleaf = pair
          _logger.debug("yield:\t%s", str(n))
          yield pair
        leaf.parent = parent

      self.is_done = True
      #return self.root.nodes

    @instrument
    def sample(self, data, samp_rate):
        return self.sampler(data, samp_rate)

    def should_idx_stop(self, args):
      idx, infs = args
      if len(infs) <= self.partition_min_pts:
          return True

      infmax = max(infs)
      thresh = self.compute_threshold(infmax, idx)
      std = np.std(infs)
      maxv, minv = max(infs), min(infs)
      return std < thresh #and maxv - minv < thresh

    def print_status(self, rule, datas, sample_infs):
      bools = map(self.should_idx_stop, enumerate(sample_infs))
      perc_passed = np.mean(map(float, bools))

      maxstd, maxrange, minthresh = -inf, -inf, inf
      maxbound = -inf
      perc = -inf
      maxidx = -1
      for idx, infs in enumerate(sample_infs):
        if not infs: continue
        std = np.std(infs)
        if std > maxstd:
          maxstd = std
          maxidx = idx
          maxbound = self.inf_bounds[idx]
          if maxbound:
            perc = (max(infs) - self.inf_bounds[idx][0]) / (maxbound[1]-maxbound[0])
          else:
            perc = -inf
        maxrange = max(maxrange, max(infs)-min(infs))
        minthresh = min(minthresh, self.compute_threshold(max(infs), idx))
      npts = sum(map(len, datas))

      fmt = "status:\t%.3f\tid(%d)\tn(%d)\tstd(%.4f)\tperc(%.4f)\tbnd(%.3f, %.3f)\tthrsh(%.4f)\t%s"
      try:
        _logger.debug(fmt,
            perc_passed,
            maxidx,
            npts,
            maxstd,
            perc,
            maxbound[0],
            maxbound[1],
            minthresh,
            str(rule))
      except:
        pdb.set_trace()


    def should_stop(self, sample_infs): 
      bools = map(self.should_idx_stop, enumerate(sample_infs))
      if sum(map(len, sample_infs)) < self.min_pts: 
        return True
      return reduce(and_, bools)


    def influence(self, row, idx):
       if row[self.SCORE_ID].value == -inf:
            influence = self.err_funcs[idx]((row,))
            row[self.SCORE_ID] = influence
            self.inf_bounds[idx][0] = min(influence, self.inf_bounds[idx][0])
            self.inf_bounds[idx][1] = max(influence, self.inf_bounds[idx][1])
       return row[self.SCORE_ID].value

    def compute_threshold(self, infmax, idx):
        infl, infu = tuple(self.inf_bounds[idx])
        tau, p = self.tau, self.p
        inflection = p*(infu-infl) + infl
        s = (tau[0] - tau[1]) / (infu - inflection)
        w = tau[0] + s*(infmax - infu)
        w = min(tau[1], w)
        ret = w * (infu - infl)       
        if ret == -inf:
            raise RuntimeError()
        if ret < 0:
          pdb.set_trace()
        return ret


    def compute_infs(self, idx, samples):
      return [self.influence(r, idx) for r in samples]

    def estimate_inf(self, sample_infs):
      means = map(np.mean, filter(bool, sample_infs))
      if means:
        return np.mean(means)
      return -inf


    @instrument
    def databyrule2infs(self, rules, datas):
      """
      compute list of influence values for each combination of rule x data
      @param rules list of rule
      @param datas list of data
      @return 
      """
      data2infs = defaultdict(list)
      rule2infs = defaultdict(list)
      rule2datas = defaultdict(list)
      for idx, data in enumerate(datas):
        for rule in rules:
          filtered = rule.filter(data)
          infs = self.compute_infs(idx, filtered)
          rule2infs[rule].append(infs)
          data2infs[idx].append(infs)
          rule2datas[rule].append(filtered)
      return data2infs, rule2infs, rule2datas


    @instrument
    def get_scores(self, rules, samples):
      sample2infs, rule2infs, _ = self.databyrule2infs(rules, samples)
      scores = []

      for idx in xrange(len(samples)):
        allinfs = sample2infs[idx]
        score = self.get_score_for_infs([idx]*len(allinfs), allinfs)
        scores.append(score)

      for r in rules:
        allinfs = rule2infs[r]
        r.quality = self.estimate_inf(allinfs)

      scores = filter(lambda s: s!=-inf, scores)
      return scores

    @instrument
    def get_score_for_infs(self, idxs, samp_infs):
        scores, counts = [], []
        for idx, infs in zip(idxs, samp_infs):
          if not len(infs): continue
          thresh = self.compute_threshold(max(infs), idx)
          bounds = self.inf_bounds[idx]
          inf_range = bounds[1] - bounds[0]
          if not inf_range:
            scores.append(0)
            counts.append(len(infs))
          else:
            std = np.std(infs)
            #scores.append(((thresh - bounds[0]) / inf_range) * (std - thresh))
            scores.append((1. - (thresh / inf_range)) * (std - thresh))
            counts.append(len(infs))
        if scores:
          return np.mean(scores)
        return -inf

    def merge_scores(self, scores):
        if scores:
          return np.percentile(scores, 75) #10)#75) #XXX: hack
        return -inf

    def adjust_score(self, score, node, attr, rules):
      # penalize excessive splitting along a single dimension if it is not helping
      if attr.var_type == Orange.feature.Type.Discrete:
        if attr == node.prev_attr:
          score = score - (0.15) * abs(score)
      else:
        if attr == node.prev_attr:
          #score = score + (0.01) * abs(score)
          if False and self.skinny_penalty(rules):
            score = score + (0.6) * abs(score)
      return score


    def get_states(self, tables):#node):
        #tables = map(node.rule,self.tables)

        # find tuples in each table that is closest to the average influence
        all_infs = []
        for idx, table in enumerate(tables):
            infs = [self.influence(row, idx) for row in table]
            all_infs.append(infs)
        states = []

        for idx, t, infs in zip(xrange(len(tables)), tables, all_infs):
            if infs:
                avg = np.mean(infs)
                min_tup = min(t, key=lambda row: self.influence(row, idx))
                state = self.err_funcs[idx].state((min_tup,))
                states.append(state)
            else:
                states.append(None)

        return states
    

    @instrument
    def skinny_penalty(self, rules):
      for rule in rules:
        edges = []
        for c in rule.filter.conditions:
            attr = self.merged_table.domain[c.position]
            if attr.var_type == Orange.feature.Type.Discrete:
                continue
            edges.append(c.max - c.min)
        if len(edges) > 1:
            volume = reduce(mul, edges)
            mean_edge = sum(edges) / float(len(edges))
            max_vol = mean_edge ** len(edges)
            perc = (volume / max_vol) ** (1./len(edges))
            if perc < 0.05:
                return True
            return (1. - perc) * 1.5
        return False

        
    def time_exceeded(self):
      return (
          self.start_time is not None and
          self.max_wait is not None and
          (time.time() - self.start_time) >= self.max_wait
          )
        


    def grow(self, node, tables, samp_rates, sample_infs=None):
      if self.time_exceeded():
        _logger.debug("time exceeded %.2f > %d", (time.time()-self.start_time), self.max_wait)
        yield (node, False)
        return

      if node.rule in self.seen:
        _logger.debug("rule seen %d\t%s", hash(node.rule), node.rule)
        yield (node, False)
        return
      self.seen.add(node.rule)

      if self.start_time is None and node.depth >= 1:
        self.start_time = time.time()

      rule = node.rule
      datas = tables
      if not sample_infs:
        datas = map(rule.filter_table, tables)
      node.cards = map(len, datas)
      node.n = sum(node.cards)

      if node.n == 0:
        yield (node, False)
        return

      #
      # Precompute influences and scores
      #

      samples = datas
      if sample_infs is None:
        f = lambda (idx, samps): self.compute_infs(idx, samps)
        samples = [self.sample(*pair) for pair in zip(datas, samp_rates)]
        sample_infs = map(f, enumerate(samples))

      curscore = self.get_score_for_infs(range(len(sample_infs)), sample_infs)
      est_inf = self.estimate_inf(sample_infs)
      node.set_score(est_inf)

      if node.parent:
        self.print_status(rule, datas, sample_infs)
        if self.should_stop(sample_infs):
          node.states = self.get_states(datas)
          yield (node, True)
          return


      if self.time_exceeded():
        _logger.debug("time exceeded %.2f > %d", (time.time()-self.start_time), self.max_wait)
        yield (node, False)
        return 


      #
      # compute scores for each attribute to split on
      #
      attr_scores = []
      for attr, new_rules in self.child_rules(rule):
        if not new_rules: continue
        scores = self.get_scores(new_rules, samples)
        score = self.merge_scores(scores)
        score = self.adjust_score(score, node, attr, new_rules)
        _logger.debug("score:\t%.4f\t%s\t%s", score, attr.name[:6], new_rules[0])
        if score == -inf: continue
        attr_scores.append((attr, new_rules, score, scores))


      if not attr_scores:
        node.states = self.get_states(datas)
        yield (node, True)
        return

      attr_scores.sort(key=lambda p: p[-2])

      attr, new_rules, score, scores = attr_scores[0]
      node.score = min(scores) 
      minscore = curscore - abs(curscore) * self.min_improvement
      if node.score >= minscore and minscore != -inf:
        _logger.debug("bdt:  \tscore didn't improve\t%.7f >= %.7f", min(scores), minscore)
        yield (node, True)
        return

      if node.score <= curscore - abs(curscore) * 0.05:
        _logger.debug("bdt:  \tbig improvement\t%s", str(new_rules[0]))
        yield (node, False)

      ncands = max(1, 2 - node.depth)
      for attr, new_rules, score, scores in attr_scores[:ncands]:
        data2infs, rule2infs, rule2datas = self.databyrule2infs(new_rules, datas)
        #new_srses = self.update_sample_rates(new_rules, data2infs, samp_rates)
        new_srses = [samp_rates] * len(new_rules)
        new_pairs = zip(new_rules, new_srses)
        new_pairs.sort(key=lambda (new_r, new_s): new_r.quality, reverse=True)

        for new_rule, new_srs in new_pairs:
          child = Node(new_rule)
          child.prev_attr = attr
          child.parent = node
          node.add_child(child)

          args = (child, rule2datas[new_rule], new_srs, rule2infs[new_rule])
          #self.candidates.add((score, new_rule.quality, args))
          for pair in self.grow(*args):
            yield pair

      yield (node, False)


    @instrument
    def child_rules(self, rule):
        next_rules = defaultdict(list)

        if self.cont_attrs:
            refiner = BeamRefiner(attrs=self.cont_attrs, fanout=2)
            for attr, new_rule in refiner(rule):
                next_rules[attr].append(new_rule)
        if self.dist_attrs:
            refiner = BeamRefiner(attrs=self.dist_attrs, fanout=5)
            for attr, new_rule in refiner(rule):
                next_rules[attr].append(new_rule)
        return next_rules.items()



    @instrument
    def update_sample_rates(self, rules, data2infs, srs):
      """return list where each element are the sample rates for a single rule across the tables"""
      srs_by_table = [[0]*len(srs) for i in data2infs]
      for idx in data2infs:
        sr = srs[idx]
        all_infs = data2infs[idx]

        if not sr: continue
        if not sum(map(len, all_infs)): continue

        new_srs = self.update_sample_rates_helper(all_infs, sr, idx)
        srs_by_table[idx] = new_srs
      return zip(*srs_by_table)

    def update_sample_rates_helper(self, all_infs, samp_rate, idx):
        influences, counts = [], []
        for infs in all_infs:
          influence = sum(np.array(infs)-self.inf_bounds[idx][0])
          influences.append(influence)
          counts.append(len(infs) + 1.)

        total_inf = sum(influences)
        total_count = sum(counts)
        if not total_inf:
          return [0]*len(all_infs)
        samp_rates = []
        nsamples = total_count * samp_rate
        for influence, count in zip(influences, counts):
          infr = influence / total_inf
          sub_samples = infr * nsamples
          nsr = sub_samples / count
          nsr = max(0, min(1., nsr))
          samp_rates.append(nsr)

        return samp_rates




