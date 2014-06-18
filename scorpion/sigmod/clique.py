import json
import time
import pdb
import sys
import Orange
import orange
import heapq
sys.path.extend(['.', '..'])

from itertools import chain, ifilter


from ..learners.cn2sd.rule import fill_in_rules
from ..learners.cn2sd.refiner import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *
from ..util import *

from basic import Basic
from streamrangemerger import StreamRangeMerger
from rangemerger import RangeMerger, RangeMerger2
from merger import Merger
from grouper import Grouper, Blah

_logger = get_logger()
 


class MR(Basic):

  def __init__(self, *args, **kwargs):
    Basic.__init__(self, *args, **kwargs)
    self.best = []
    self.max_wait = kwargs.get('max_wait', 2 * 60 * 60) # 2 hours
    self.start = None
    self.stop = False
    self.n_rules_checked = 0
    self.naive = kwargs.get('naive', False)
    self.max_bests = 50
    self.max_complexity = kwargs.get('max_complexity', 3)

    self.checkpoints = []

    self.cost_clique = 0


  def __hash__(self):
    components = [
      self.__class__.__name__,
      str(self.aggerr.__class__.__name__),
      str(set(self.cols)),
      self.epsilon,
      self.tau,
      self.p,
      self.err_func.__class__.__name__,
      self.tablename,
      self.aggerr.keys,
      self.max_wait,
      self.c_range
    ]
    components = map(str, components)
    return hash('\n'.join(components))


  def setup_tables(self, full_table, bad_tables, good_tables, **kwargs):
    Basic.setup_tables(self, full_table, bad_tables, good_tables, **kwargs)
    self.grouper = Grouper(full_table, self) 

    self.SCORE_ID = add_meta_column(
            chain([full_table], bad_tables, good_tables),
            'SCOREVAR' 
    )



  def set_params(self, **kwargs):
    self.cols = kwargs.get('cols', self.cols)
    self.params.update(kwargs)
    self.good_thresh = 0.0001
    self.granularity = kwargs.get('granularity', self.granularity)

  def make_rules(self, cur_groups):
    if cur_groups == None:
      new_groups = self.grouper.initial_groups()
    else:
      new_groups = self.grouper.merge_groups(cur_groups)

    rules = {}

    for attrs, groups in new_groups:
      start = time.time()
      for ro in self.grouper(attrs, groups):
        if self.max_wait:
          self.n_rules_checked -= len(ro.rule.filter.conditions)
          if self.n_rules_checked <= 0:
            diff = time.time() - self.start
            if not self.checkpoints or diff - self.checkpoints[-1][0] > 10:
              if self.best:
                best_rule = max(self.best).rule
                self.checkpoints.append((diff, best_rule))
            self.stop = diff > self.max_wait
            self.n_rules_checked = 1000
          if self.stop:
            _logger.debug("wait %d > %d exceeded." % (diff, self.max_wait))
            return


        yield attrs, ro
#        print "group by\t%s\t%.4f" % (str([attr.name for attr in attrs]), time.time()-start)



  def __call__(self, full_table, bad_tables, good_tables, **kwargs):
    self.setup_tables(full_table, bad_tables, good_tables, **kwargs)
    self.update_status("running bottom up algorithm")
    for pairs in self.find_cliques():
      rules = [(b.rule, iteridx) for b, iteridx in pairs]
      yield rules
    self.update_status("bottom up algorithm done")


  def find_cliques(self):
    """
    table has been trimmed of extraneous columns.
    """
    #clusters = self.load_from_cache()
    #if clusters is not None:
      #yield clusters
      #return 

    rules = None
    self.best = []
    self.start = time.time()

    added = []
    nseen = 0
    niters = 0 
    while (niters < self.max_complexity and 
           not self.stop and 
           (rules is None or rules)):
      niters += 1
      self.update_status("running bottomup iter %d" % niters)
      _logger.debug("=========iter %d=========", niters)

      nadded = 0
      seen = set()
      nnewgroups = 0
      new_rules = defaultdict(list)
      
      # for each combination of attributes
      # prune the groups that are less influential than the parent group's 
      #  

      for attr, ro in self.make_rules(rules):
        nseen += 1
        if nseen % 50 == 0 and nseen > 0:
          self.update_status("bottomup processed %d rules" % nseen)

        if self.stop:
            break

        if self.top_k(ro):
          nadded += 1

        if self.naive:
            new_rules[attr] = [None]
            nnewgroups += 1
        elif self.prune_rule(ro):
            new_rules[attr].append(ro.group)
            nnewgroups += 1
        ro.rule.__examples__ = None


        if nadded % 25 == 0 and nadded > 0:
          newbests = filter(lambda c: c not in seen, self.best)
          seen.update(self.best)
          yield zip(newbests, [niters]*len(newbests))


      newbests = filter(lambda c: c not in seen, self.best)
      seen.update(self.best)
      yield zip(newbests, [niters]*len(newbests))
      if not nadded: 
        break 

      rules = new_rules
      if niters == 1:
        best = self.best
      else:
        best = set(self.best)
        if prev_best and prev_best in self.best:
          self.best.remove(prev_best)
        best = list(best)

      self.best = [max(self.best)] if self.best else []
      prev_best = max(self.best) if self.best else None

    _logger.debug("finished, merging now")
    self.cost_clique = time.time() - self.start

    #self.cache_results(clusters)

  def prune_rule(self, ro):
    if ro.npts < self.min_pts:
        _logger.debug("prune? %s\t%s", 'FALSE', str(ro))
        return False
    
    if (math.isnan(ro.bad_inf) or
        math.isnan(ro.good_inf) or
        math.isnan(ro.inf)):
        _logger.debug("prune? %s\t%s", 'FALSE', str(ro))
        return False
    

    # assuming the best case (the good_stat was zero)
    # would the influence beat the best so far across
    # the full c_range?
    if self.best:
      if ro.dominated_by(max(self.best)):
        _logger.debug("prune? %s\t%s", 'FALSE', str(ro))
        return False

    #if self.best and ro.best_inf <= max(self.best).inf:
    #    # if best tuple influence < rule influence:
    #    if ro.best_tuple_inf <= max(self.best).inf:
    #        _logger.debug("%s\t%s", 'FALSE', str(ro))
    #        return False

    # check max good influence
    if False and ro.good_inf < self.good_thresh:
        # TODO: can skip computing good_stats
        ro.good_skip = True


    #_logger.debug("%s\t%.4f\t%s", 'T', self.best and max(self.best).inf or 0, str(ro))
    return True


  def top_k(self, ro):
    n = 0
    best = self.best and max(self.best, key=lambda ro: ro.inf) or None
    if len(self.best) >= self.max_bests:
      bound = best.inf - self.best[0].inf
      thresh = self.best[0].inf + bound * 0.02
      if ro.inf <= thresh:
        return False

    if ro in self.best:
      return False
    if math.isnan(ro.inf):
      return False

    
    if len(self.best) < self.max_bests:
      n += 1            
      _logger.debug(str(ro))
      heapq.heappush(self.best, ro)
    else:
      n += 1            
      _logger.debug(str(ro))
      heapq.heapreplace(self.best, ro)
    
    best = best and max(best, ro) or ro

    return True

  @instrument
  def load_from_cache(self):
    import bsddb as bsddb3
    self.cache = bsddb3.hashopen('./dbwipes.mr.cache')
    try:
      myhash = str(hash(self))
      if myhash in self.cache and self.use_cache:
        self.update_status("loading partitions from cache")
        dicts, errors = json.loads(self.cache[myhash])
        clusters = map(Cluster.from_dict, dicts)
        for c in clusters:
          self.influence_cluster(c, self.full_table)
        return clusters
    except Exception as e:
      print e
      pdb.set_trace()
      pass
    finally:
      self.cache.close()
    return None


  @instrument
  def cache_results(self, clusters):
    import bsddb as bsddb3
    # save the clusters in a dictionary
    if self.use_cache:
      myhash = str(hash(self))
      self.cache = bsddb3.hashopen('./dbwipes.mr.cache')
      try:
        dicts = [c.to_dict() for c in clusters]
        errors = [c.error for c in clusters]
        self.cache[myhash] = json.dumps((dicts, errors))
      except Exception as e:
        print e
        pdb.set_trace()
        pass
      finally:
        self.cache.close()


