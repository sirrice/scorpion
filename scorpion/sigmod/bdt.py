import json
import time
import pdb
import sys
import Orange
import orange
import heapq
sys.path.extend(['.', '..'])

from multiprocessing import Process, Queue, Pool, Pipe
from Queue import Empty
from collections import deque
from itertools import chain
from rtree.index import Index as RTree
from rtree.index import Property as RProp

from ..learners.cn2sd.rule import fill_in_rules
from ..learners.cn2sd.refiner import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *
from ..errfunc import ErrTypes
from ..util import *
from ..settings import *

from basic import Basic
from sampler import Sampler
from merger import Merger
from rangemerger import RangeMerger, RangeMerger2
from streamrangemerger import *
from frontier import Frontier
from bdtpartitioner import *

inf = float('inf')
_logger = get_logger()








class BDT(Basic):

    def __init__(self, **kwargs):
        Basic.__init__(self, **kwargs)
        self.all_clusters = []
        self.cost_split = 0.
        self.cost_partition_bad = 0.
        self.cost_partition_good = 0.
        self.cache = None
        self.use_mtuples = kwargs.get('use_mtuples', False)
        self.max_wait = kwargs.get('max_wait', None)


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
                self.max_wait
                ]
        components = map(str, components)
        return hash('\n'.join(components))
 

    def setup_tables(self, full_table, bad_tables, good_tables, **kwargs):
        Basic.setup_tables(self, full_table, bad_tables, good_tables, **kwargs)


        # this is to cache a row's influence 
        self.SCORE_ID = add_meta_column(
          chain([full_table], bad_tables, good_tables),
          SCORE_VAR
        )

        domain = self.full_table.domain
        attrnames = [attr.name for attr in domain]
        self.cont_dists = dict(zip(attrnames, Orange.statistics.basic.Domain(self.full_table)))
        self.disc_dists = dict(zip(attrnames, Orange.statistics.distribution.Domain(self.full_table)))

        self.bad_states = [ef.state(t) for ef, t in zip(self.bad_err_funcs, self.bad_tables)]
        self.good_states = [ef.state(t) for ef, t in zip(self.good_err_funcs, self.good_tables)]


    def nodes_to_clusters(self, nodes, table):
      clusters = []
      rules = []
      for node in nodes:
        node.rule.quality = node.influence
        rule = node.rule.simplify(table, self.cont_dists, self.disc_dists)
        rules.append(rule)

      fill_in_rules(rules, table, cols=self.cols)
      for rule in rules:
        cluster = Cluster.from_rule(rule, self.cols)
        cluster.states = node.states
        cluster.cards = node.cards
        clusters.append(cluster)
      return clusters

    def nodes_to_popular_clusters(self, nodes, table):
      """
      Look for clauses found in more than X% of the nodes
      and turn them into clusters
      """
      if not nodes: return []
      from collections import Counter
      counter = Counter()
      str_to_rule = {}

      for node in nodes:
        r = node.rule
        if len(r.filter.conditions) > 1:
          for cond in r.filter.conditions:
            newr = SDRule(r.data, r.targetClass, [cond], r.g)
            newr.quality = node.influence
            counter[newr] += 1
            str_to_rule[newr] = newr

      if len(counter) == 0:
        return []

      thresh = np.percentile(counter.values(), 80)
      rules = []
      for strrule, count in counter.iteritems():
        if count >= thresh:  #0.25 * len(nodes):
          r = str_to_rule[strrule]
          rules.append(r)

      fill_in_rules(rules, table, cols=self.cols)
      clusters = [Cluster.from_rule(r, self.cols) for r in rules]
      return clusters


    @instrument
    def merge(self, clusters, nonleaves):
      if len(clusters) <= 1:
          return clusters

      self.update_status("starting merge phase")
      start = time.time()
      if [attr for attr in self.full_table.domain if attr.varType == orange.VarTypes.Discrete]:
        use_mtuples = False
      else:
        use_mtuples = self.use_mtuples
      params = dict(self.params)
      params.update({
        'learner_hash': hash(self),
        'learner' : self,
        'cols' : self.cols,
        'c_range': self.c_range,
        'use_mtuples' : use_mtuples
      })
      #self.merger = StreamRangeMerger(**params)
      self.merger = RangeMerger2(**params)
      #self.merger = Merger(**params)


      merged_clusters = self.merger(to_add)
      #merged_clusters = self.merger(clusters)
      self.merge_cost = time.time() - start

      self.merge_stats(self.merger.stats, 'merge_')
      return merged_clusters

    def pick_clusters(self, clusters, nonleaves):
      """
      """
      _logger.debug("compute initial cluster errors. %d clusters", len(clusters))
      start = time.time()
      for c in clusters:
        c.error = self.influence_cluster(c)
        c.c_range = list(self.c_range)
        c.inf_func = self.create_inf_func(c)
      self.stats['init_cluster_errors'] = [time.time()-start, 1]

      self.update_status("computing frontier")
      _logger.debug("compute initial frontier")
      frontier,_ = Frontier(self.c_range, 0.001)(clusters)

      ret = list(frontier)
      _logger.debug("get nonleaves containing frontier")
      for nonleaf in nonleaves:
        for c in frontier:
          if nonleaf.contains(c):
            nonleaf.error = self.influence_cluster(nonleaf)
            ret.append(nonleaf)
            break

      self.update_status("expanding frontier (%d rules)" % len(ret))
      _logger.debug("second merger pass")
      return ret


    def create_rtree(self, clusters):
        if not len(clusters[0].bbox[0]):
            class k(object):
                def intersection(self, foo):
                    return xrange(len(clusters))
            return k()

        ndim = len(clusters[0].bbox[0]) + 1
        p = RProp()
        p.dimension = ndim
        p.dat_extension = 'data'
        p.idx_extension = 'index'

        rtree = RTree(properties=p)
        for idx, c in enumerate(clusters):
            rtree.insert(idx, c.bbox[0] + (0,) + c.bbox[1] + (1,))
        return rtree

    
    @instrument
    def intersect(self, bclusters, hclusters):
        errors = [c.error for c in bclusters]
        u, std = np.mean(errors), np.std(errors)
        u = min(max(errors), u + std)
        bqueue = deque(bclusters)
        low_influence = []
#        low_influence = [c for c in bclusters if c.error < u]
        bqueue = deque([c for c in bclusters if c.error >= u])

        hclusters = [c for c in hclusters if c.error >= u]
        if not hclusters:
            for c in bclusters:
                c.bad_states = c.states
                c.bad_cards = c.cards
            return bclusters
        hindex = self.create_rtree(hclusters)
        ret = []


        while bqueue:
            c = bqueue.popleft()

            idxs = hindex.intersection(c.bbox[0] + (0,) + c.bbox[1] + (1,))
            hcs = [hclusters[idx] for idx in idxs]
            hcs = filter(c.discretes_intersect, hcs)
            hcs = filter(c.discretes_contains, hcs)

            if not hcs:
                c.bad_inf = c.error
                c.good_inf = None
                c.bad_states = c.states
                c.bad_cards = c.cards
                ret.append(c)
                continue

            # first duplicate c into clusters that have the same discrete values as hcs
            # and those that are not
            # put the diff clusters back
            matched = False
            for hc in hcs:
                # are they exactly the same? then just skip
                split_clusters = c.split_on(hc)

                if not split_clusters: 
                    # hc was false positive, skip
                    continue

                matched = True

                intersects, excludes = split_clusters
                if len(intersects) == 1 and not excludes:
                    c.good_inf = hc.error
                    c.bad_inf = c.error
                    c.good_states = hc.states
                    c.bad_states = c.states
                    c.bad_cards = c.cards
                    c.good_cards = [math.ceil(n * c.volume / hc.volume) for n in c.cards]
                    ret.append(c)
                    continue
                else:
                    for cluster in chain(intersects, excludes):
                        cluster.good_inf, cluster.bad_inf, cluster.error = hc.error, c.error, c.error
                        cluster.states = c.states
                        new_vol = cluster.volume
                        cluster.cards = [math.ceil(n * new_vol / c.volume) for n in c.cards]

                    bqueue.extendleft(intersects)
                    bqueue.extendleft(excludes)
                    break

            if not matched:
                c.bad_inf = c.error
                c.good_inf = -inf
                c.bad_states = c.states
                c.bad_cards = c.cards
                ret.append(c)

        _logger.info( "intersection %d untouched, %d split" , len(low_influence), len(ret))
        ret.extend(low_influence)
        return ret


    @instrument
    def get_partitions(self, full_table, bad_tables, good_tables, **kwargs):
      clusters, nonleaf_clusters = self.load_from_cache()
      if clusters:
        yield clusters
        return
        #return clusters, nonleaf_clusters

      max_wait = self.params.get('max_wait', kwargs.get('max_wait', None))
      bad_max_wait = good_max_wait = None
      if max_wait:
        bad_max_wait = max_wait * 2. / 3.
        good_max_wait = max_wait / 3.


      bad_params = dict(self.params)
      bad_params.update(kwargs)
      bad_params.update({
        'SCORE_ID': self.SCORE_ID,
        'err_funcs': self.bad_err_funcs,
        'max_wait': bad_max_wait
      })
      good_params = dict(self.params)
      good_params.update(kwargs)
      good_params.update({
        'SCORE_ID': self.SCORE_ID,
        'err_funcs': self.good_err_funcs,
        'max_wait': good_max_wait
      })

      if not self.parallel:
        self.update_status("running partitioners in serial")
        partitioner = BDTTablesPartitioner(**bad_params)
        partitioner.setup_tables(bad_tables, full_table)
        gen = partitioner()
        nodes = [n for n in gen]
        clusters = self.nodes_to_clusters(nodes, full_table)
        map(self.influence_cluster, clusters)
        yield clusters
        bound = partitioner.get_inf_bound()

        partitioner = BDTTablesPartitioner(**good_params)
        partitioner.setup_tables(good_tables, full_table)
        partitioner.update_inf_bound(bound)
        gen = partitioner()
        nodes = [n for n in gen]
        clusters = self.nodes_to_clusters(nodes, full_table)
        map(self.influence_cluster, clusters)
        yield clusters
        return


      self.update_status("running partitioners in parallel")
      bad_p2cq = Queue()
      bad_c2pq = Queue()
      good_p2cq = Queue()
      good_c2pq = Queue()
      bad_args = ('bad', bad_params, bad_tables, full_table, (bad_p2cq, bad_c2pq))
      good_args = ('good', good_params, good_tables, full_table, (good_p2cq, good_c2pq))

      bad_proc = Process(target=partition_f, args=bad_args)
      good_proc = Process(target=partition_f, args=good_args)
      bad_proc.start()
      good_proc.start()

      bdone = gdone = False
      while not bdone or not gdone:
        bnodes = []
        gnodes = []
        bound = None
        if not bdone:
          try:
            data = bad_c2pq.get(False)
            if data == 'done':
              self.update_status("done partitioning outlier datasets")
              bdone = True
            else:
              (bnodes, bound) = data
          except Empty:
            bnodes = []

        if not gdone:
          try:
            data = good_c2pq.get(False)
            if data == 'done':
              self.update_status("done partitioning normal datasets")
              gdone = True
            else:
              gnodes = data[0]
          except Empty:
            gnodes = []

        
        if bound and not gdone:
          good_p2cq.put(bound)

        dicts = []
        if bnodes:
          dicts.extend(bnodes)
        if gnodes:
          dicts.extend(gnodes)

        if dicts:
          print "bdt\tgot %s\t%s" % (len(dicts), map(hash, map(str, dicts)))
          rules = [SDRule.from_json(d, full_table) for d in dicts]
          fill_in_rules(rules, full_table, cols=self.cols)
          clusters = [Cluster.from_rule(rule, self.cols) for rule in rules]
          map(self.influence_cluster, clusters)
          yield clusters

      self.update_status("done partitioning")
      bad_p2cq.close()
      good_c2pq.close()
      bad_c2pq.close()
      good_c2pq.close()
      bad_proc.join()
      good_proc.join()

      return






      ## 
      ## Setup and run partitioner for bad outputs
      ##
      #self.update_status("partitioning bad examples")
      #params = dict(self.params)
      #params.update(kwargs)
      #params['SCORE_ID'] = self.SCORE_ID
      #params['err_funcs'] = self.bad_err_funcs
      #max_wait = params.get('max_wait', None)
      #bad_max_wait = good_max_wait = None
      #if max_wait:
      #  bad_max_wait = max_wait * 2. / 3.
      #  good_max_wait = max_wait / 3.


      #start = time.time()
      #params['max_wait'] = bad_max_wait
      #bpartitioner = BDTTablesPartitioner(**params)
      #bpartitioner(bad_tables, full_table)
      #self.cost_partition_bad = time.time() - start
      #tree = bpartitioner.root.clone()

      #clusters = self.nodes_to_clusters(tree.leaves, full_table)
      #for c in clusters:
      #  self.influence_cluster(c)
      #_logger.debug( "==== Best Leaf Nodes (%d total) ====" , len(clusters))
      #_logger.debug( '\n'.join(map(str, sorted(clusters, reverse=True)[:10])))

      #return clusters, []


      #
      #for hnode in tree.nodes:
      #  hnode.frombad = True
      #_logger.debug('\npartitioning bad tables done\n')

      #start = time.time()

      ## 
      ## Setup and run partitioner for good outputs
      ##
      #self.update_status("partitioning good examples")
      #inf_bound = [inf, -inf]
      #for ib in bpartitioner.inf_bounds:
      #  inf_bound = r_union(ib, inf_bound)
      #inf_bounds = [list(inf_bound) for t in good_tables]
      #print 'partitioner inf_bound: %.4f - %.4f' % tuple(inf_bound)

      #params['err_funcs'] = self.good_err_funcs
      #params['max_wait'] = good_max_wait
      #hpartitioner = BDTTablesPartitioner(**params)
      #hpartitioner.inf_bounds = inf_bounds
      #hpartitioner(good_tables, full_table, root=tree)

      #self.stats['partition_good'] = [time.time()-start, 1]
      #self.cost_partition_good = time.time() - start


      #leaves = list(tree.leaves)
      #nonleaves = list(tree.nonleaves)
      #_logger.debug( "==== Best Leaf Nodes (%d total) ====" , len(leaves))
      #_logger.debug( '\n'.join(map(str, sorted(leaves, key=lambda n: n.influence)[:10])))


      #self.update_status("reconciling %d partitions" % (len(nonleaves) + len(leaves)))
      #start = time.time()
      ##popular_clusters = self.nodes_to_popular_clusters(nonleaves, full_table)
      #self.stats['intersect_partitions'] = [time.time()-start, 1]
      #self.cost_split = time.time() - start



      ## NOTE: if good partitioner starts with tree from bad partitioner then
      ##       no need to intersect their results
      ##clusters = self.intersect(bclusters, hclusters)
      #clusters = self.nodes_to_clusters(tree.leaves, full_table)
      #nonleaf_clusters = self.nodes_to_clusters(nonleaves, full_table)
      ##clusters.extend(popular_clusters)


      #if False:
      #  start = time.time()
      #  for c in chain(clusters, nonleaf_clusters):
      #    if not c.inf_state:
      #      c.error = self.influence_cluster(c)
      #  self.stats['init_cluster_errors'] = [time.time()-start, 1]

      #if self.DEBUG:
      #  renderer = ClusterRenderer('/tmp/bdt.pdf')
      #  renderer.plot_clusters(clusters)
      #  renderer.new_page()
      #  renderer.plot_clusters(nonleaf_clusters)

      #  tuples = [map(float,[row['a_0'], row['a_1'], row['v']]) for row in self.bad_tables[0]]
      #  renderer.plot_tuples(tuples)
      #  renderer.close()



      #self.cache_results(clusters, nonleaf_clusters)

      #self.merge_stats(bpartitioner.stats, 'bdtp_bad_')
      #self.merge_stats(hpartitioner.stats, 'bdtp_good_')


      #return clusters, nonleaf_clusters




    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

        for clusters in self.get_partitions(full_table, bad_tables, good_tables, **kwargs):
          yield clusters
        return

        #clusters, nomerge_clusters = self.get_partitions(full_table, bad_tables, good_tables, **kwargs)
        #self.all_clusters = self.pick_clusters(clusters, nomerge_clusters)
        #return [self.all_clusters]


        #_logger.debug('merging')
        #final_clusters = self.merge(clusters, nomerge_clusters)        
        ##final_clusters.extend(nomerge_clusters)

        #self.final_clusters = final_clusters


        #self.costs.update({
        #  'cost_partition_bad' : self.cost_partition_bad,
        #  'cost_partition_good' : self.cost_partition_good,
        #  'cost_split' : self.cost_split
        #})
        #
        #_logger.debug("=== Costs ===")
        #for key, stat in sorted(self.stats.items(), key=lambda p: p[1][0]):
        #  _logger.debug("%.4f\t%d\t%s", stat[0], stat[1], key)

        #return self.final_clusters




    @instrument
    def load_from_cache(self):
      import bsddb as bsddb3
      self.cache = bsddb3.hashopen('./dbwipes.cache')
      try:
        myhash = str(hash(self))
        if myhash in self.cache and self.use_cache:
          self.update_status("loading partitions from cache")
          dicts, nonleaf_dicts, errors = json.loads(self.cache[myhash])
          clusters = map(Cluster.from_dict, dicts)
          nonleaf_clusters = map(Cluster.from_dict, nonleaf_dicts)
          for err, c in zip(errors, chain(clusters, nonleaf_clusters)):
            c.error = err
          return clusters, nonleaf_clusters
      except Exception as e:
        print e
        pass
      finally:
        self.cache.close()
      return None, None

    @instrument
    def cache_results(self, clusters, nonleaf_clusters):
      import bsddb as bsddb3
      # save the clusters in a dictionary
      if self.use_cache:
        myhash = str(hash(self))
        self.cache = bsddb3.hashopen('./dbwipes.cache')
        try:
          dicts = [c.to_dict() for c in clusters]
          nonleaf_dicts = [c.to_dict() for c in nonleaf_clusters]
          errors = [c.error for c in chain(clusters, nonleaf_clusters)]
          self.cache[myhash] = json.dumps((dicts, nonleaf_dicts, errors))
        except:
          pass
        finally:
          self.cache.close()



