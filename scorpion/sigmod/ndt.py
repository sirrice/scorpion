#
# This file implements a naive decision tree algorithm that 
# 1) labels points in the bad tables by using a cutoff 
#    2 std from the mean tuple influence value
# 2) learns a decision tree using Orange
# 3) sorts the leaves by influence and returns the top k
#

import time
import pdb
import sys
import orngTree
import Orange
import orange
import heapq
import numpy as np
sys.path.extend(['.', '..'])

from itertools import chain
from sklearn import tree as sktree


from ..learners.cn2sd.rule import fill_in_rules
from ..learners.cn2sd.refiner import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *
from ..util import *
from ..settings import *

from basic import Basic
from merger import Merger

inf = 1e10000000
_logger = get_logger()

class NDT(Basic):

    def __init__(self, **kwargs):
        Basic.__init__(self, **kwargs)
        # values: c45, or, dt, rt
        self.tree_alg = kwargs.get('tree_alg', 'rt')

    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

        self.SCORE_ID = add_meta_column(
                chain(self.bad_tables, self.good_tables),
                SCORE_VAR)
        self.CLASS_ID = add_meta_column(
                chain(self.bad_tables, self.good_tables),
                "INFCLASS",
                vals=['0', '1'])

        start = time.time()
        self.compute_perrow_influences(self.bad_tables, self.bad_err_funcs)
        self.compute_perrow_influences(self.good_tables, self.good_err_funcs)
        self.cost_compute_inf = time.time() - start



        start = time.time()
        if self.tree_alg == 'c45':
          table, rules = self.c45_rules()
        elif self.tree_alg == 'or':
          table, rules = self.orange_dt_rules()
        elif self.tree_alg == 'dt':
          table, rules = self.sk_dt_rules(max_depth=12)
        elif self.tree_alg == 'rt':
          table, rules = self.sk_rt_rules(max_depth=12)
        else:
          _logger.warn("unknown NDT algorithm %s.  Defaulting to regression tree", self.tree_alg)
          table, rules = self.sk_rt_rules(max_depth=12)
        self.cost_learn = time.time() - start


        #
        # ok now convert rules to clusters
        #

        _logger.debug( "got %d rules", len(rules))
        fill_in_rules(rules, table, cols=self.cols)

        self.cost_learn = time.time() - start

        clusters = [Cluster.from_rule(rule, self.cols) for rule in rules]
        for cluster in clusters:
          cluster.error = self.influence_cluster(cluster)
        clusters = filter_bad_clusters(clusters)
        clusters.sort(key=lambda c: c.error, reverse=True)
        print '\n'.join(map(str, clusters[:5]))

        self.all_clusters = self.final_clusters = clusters
        return self.final_clusters

        #
        # merge the clusters
        #
        thresh = compute_clusters_threshold(clusters, nstds=1.5)
        is_mergable = lambda c: c.error >= thresh

        params = dict(kwargs)
        params.update({
          'cols' : self.cols,
          'err_func' : self.err_func,
          'influence' : lambda c: self.influence_cluster(c),
          'influence_components': lambda c: self.influence_cluster_components(c),
          'is_mergable' : is_mergable,
          'use_mtuples' : False,
          'learner' : self})
        self.merger = Merger(**params)
        merged_clusters = self.merger(clusters)
        merged_clusters.sort(key=lambda c: c.error, reverse=True)


        clusters.extend(merged_clusters)
        normalize_cluster_errors(clusters)
        clusters = list(set(clusters))
        self.all_clusters = clusters
        self.final_clusters = merged_clusters

        self.costs = {
            'cost_learn' : self.cost_learn
        }
        return self.final_clusters

    def orange_dt_rules(self):

      start = time.time()
      bad_cutoff = self.influence_cutoff(self.bad_tables)
      good_cutoff = self.influence_cutoff(self.good_tables)
      _logger.debug( "cutoffs\t%f\t%f" , bad_cutoff, good_cutoff)
      self.cost_cutoff = time.time() - start

      _logger.debug( "creating training data")
      training = self.create_training(bad_cutoff, good_cutoff)


      #_logger.debug( "training on %d points" , len(training))
      tree = orngTree.TreeLearner(training)
      rules = tree_to_clauses(training, tree.tree)
      #_logger.debug('\n'.join(map(lambda r: '\t%s' % r, rules)))

      # tree = Orange.classification.tree.C45Learner(training, cf=0.001)
      # rules = c45_to_clauses(training, tree.tree)
      return training, rules



    def c45_rules(self):
      start = time.time()
      bad_cutoff = self.influence_cutoff(self.bad_tables)
      good_cutoff = self.influence_cutoff(self.good_tables)
      _logger.debug( "cutoffs\t%f\t%f" , bad_cutoff, good_cutoff)
      self.cost_cutoff = time.time() - start

      _logger.debug( "creating training data")
      training = self.create_training(bad_cutoff, good_cutoff)

      tree = Orange.classification.tree.C45Learner(training, cf=0.001)
      rules = c45_to_clauses(training, tree.tree)
      return training, rules



    def sk_dt_rules(self, **kwargs):
      start = time.time()
      _logger.debug( "computing cutoffs" )
      bad_cutoff = self.influence_cutoff(self.bad_tables)
      good_cutoff = self.influence_cutoff(self.good_tables)
      _logger.debug( "cutoffs\t%f\t%f" , bad_cutoff, good_cutoff)
      self.cost_cutoff = time.time() - start


      _logger.debug( "creating training data")
      training = self.create_training(bad_cutoff, good_cutoff)

      class_var = training.domain[self.CLASS_ID]
      skdata = rm_attr_from_domain(training, [class_var])
      Xs = np.array([ [v.value for v in row] for row in skdata ])
      Ys = np.array([int(row[class_var].value) for row in training])

      clf = sktree.DecisionTreeClassifier(criterion='entropy')
      clf.fit(Xs, Ys)
      rules = self.sktree_to_rules(skdata, clf.tree_, **kwargs)
      return skdata, rules


    def sk_rt_rules(self, **kwargs):
      orTable, Xs, Ys = self.create_regression_training()
      clf = sktree.DecisionTreeRegressor(
          criterion='mse',
          min_samples_split=10,
          min_samples_leaf=5
      )
      clf.fit(Xs, Ys)
      rules = self.sktree_to_rules(orTable, clf.tree_, **kwargs)
      return orTable, rules



    # assumes everything is continuous!!!
    def sktree_to_rules(self, table, tree, node_id=0, parent=None, clauses=None, max_depth=9, bdists=None):
      def clauses_to_rule(table, bdists, clauses):
        attr_to_ranges = {}
        for attridx, minval, maxval in clauses:
          attr = table.domain[attridx]
          dist = bdists[attr]
          cur = attr_to_ranges.get(attridx, [dist.min, dist.max])
          cur = (max(cur[0], minval), min(cur[1], maxval))
          attr_to_ranges[attridx] = cur

        conds = []
        for pos, (minv, maxv) in attr_to_ranges.iteritems():
          conds.append(Orange.data.filter.ValueFilterContinuous(
              position=pos,
              oper=orange.ValueFilter.Between,
              min=minv,
              max=maxv))
        
        return SDRule(table, None, conds)



      clauses = clauses or []
      ret = []
      left = tree.children_left[node_id]
      right = tree.children_right[node_id]

      if bdists is None:
        bdists = Orange.statistics.basic.Domain(table)
      

      if left < 0 or len(clauses) >= max_depth:
        val = tree.value[node_id]
        #if val[0][1] > val[0][0]:
        return [clauses_to_rule(table, bdists, clauses)]
      else:
        attridx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        leftclause = (attridx, -inf, threshold)
        rightclause = (attridx, threshold, inf)

        clauses.append(leftclause)
        ret.extend(self.sktree_to_rules(table, tree, left, node_id, clauses, max_depth, bdists))
        clauses.pop()

        clauses.append(rightclause)
        ret.extend(self.sktree_to_rules(table, tree, right, node_id, clauses, max_depth, bdists))
        clauses.pop()
      return ret


      
 
    def compute_perrow_influences(self, tables, err_funcs):
        infs = []
        for table, err_func in zip(tables, err_funcs):
          for row in table:
            influence = err_func([row])
            row[self.SCORE_ID] = inluencef
            infs.append(influence)
        return infs

    def influence_cutoff(self, tables, percentile=80):
        infs = []
        for table in tables:
          for row in table:
            infs.append(float(row[self.SCORE_ID]))

        #u = np.mean(infs)
        std = np.std(infs)
        #return u + std * 2
        return np.percentile(infs, percentile)

    def label_bad_tuples(self, cutoff):
        pass


    def create_regression_training(self):
        domain = Orange.data.Domain(self.bad_tables[0].domain)
        score_var = domain[self.SCORE_ID]
        newtable = Orange.data.Table(domain)

        Ys = []
        for t in chain(self.bad_tables, self.good_tables):
          newtable.extend(t)
          for r in t:
            Ys.append(float(r[self.SCORE_ID].value))

        Xs = np.array([ [v.value for v in row] for row in newtable])
        Ys = np.array(Ys)
        return newtable, Xs, Ys


    def create_training(self, bad_cutoff, good_cutoff):
        import orngTree
        extend_bad = lambda rule, t: rule.cloneAndAddContCondition(
                t.domain[self.SCORE_ID],
                bad_cutoff,
                1e100000)
        extend_good = lambda rule, t: rule.cloneAndAddContCondition(
                t.domain[self.SCORE_ID],
                -1e100000,
                good_cutoff)

        domain = self.bad_tables[0].domain
        score_var = domain[self.SCORE_ID]
        class_var = domain[self.CLASS_ID]
        domain = list(self.bad_tables[0].domain)
        domain = [a for a in domain if a.name in self.cols]
        domain = Orange.data.Domain(domain, class_var)
        domain.add_meta(self.SCORE_ID, score_var)
        self.CLASS_ID = 'INFCLASS'

        train_table = Orange.data.Table(domain)

        for table in self.bad_tables:
            rule = SDRule(table, None)
            bad_rule = extend_bad(rule, table)
            pos_matches = Orange.data.Table(domain,bad_rule.filter_table(table))
            neg_matches =  Orange.data.Table(domain,bad_rule.cloneAndNegate().filter_table(table))

            for row in pos_matches:
                row[class_var] ='1'
            for row in neg_matches:
                row[class_var] ='0'

            train_table.extend(pos_matches)
            train_table.extend(neg_matches)
        return train_table

        for table in self.good_tables:
            rule = SDRule(table, None)
            good_rule = extend_good(rule, table)
            matches = Orange.data.Table(domain, good_rule.filter_table(table))
            for row in matches:
                row[class_var] = '0'
            train_table.extend(matches)

        return train_table



def c45_to_clauses(table, node, bdists=None, clauses=None):
    clauses = clauses or []
    if not node:
      return []
    if bdists is None:
      bdists = Orange.statistics.basic.Domain(table)

    
    if node.node_type == 0: # Leaf
        quality = node.class_dist[1] 
        if int(node.leaf) == 1 and node.items > 0 and clauses is not None:
            ret = [rule_from_clauses(table, clauses)]
            for rule in ret:
                rule.quality = quality
            return ret 
        return []

    var = node.tested
    ret = []


    if node.node_type == 1: # Branch
        for branch, val in zip(node.branch, var.values):
            clause = create_clause(table, var,  val, bdists)
            clauses.append( clause )
            ret.extend( c45_to_clauses(table, branch, bdists, clauses) )
            clauses.pop()

    elif node.node_type == 2: # Cut
        for branch, comp in zip(node.branch, ['<=', '>', '<', '>=']):
            clause = create_clause(table, var,  node.cut, bdists, comp)
            clauses.append( clause )
            ret.extend( c45_to_clauses(table, branch, bdists, clauses) )
            clauses.pop()

    elif node.node_type == 3: # Subset
        for i, branch in enumerate(node.branch):
            inset = filter(lambda a:a[1]==i, enumerate(node.mapping))
            inset = [var.values[j[0]] for j in inset]
            if len(inset) == 1:
                clause = create_clause(table, var, inset[0], bdists)
            else:
                clause = create_clause(table, var, inset, bdists)
            clause.append( clause )
            ret.extend( c45_to_clauses(table, branch, bdists, clauses) )
            clauses.pop()

    ret = filter(lambda c: c, ret)
    return ret



def create_clause(table, attr, val, bdists, cmp='='):
    cmps = ['<', '<=', '>', '>=', '=']
    if attr.varType == Orange.feature.Type.Discrete:
        if not isinstance(val, (list, tuple)):
            val = [val]
        vals = [orange.Value(attr, v) for v in val]
        filt = orange.ValueFilter_discrete(
            position = table.domain.index(attr),
            values = vals)
        return filt
    else:
        # it may be a discretized continuous condition (e.g., "<= 5")
        isnumerical = False
        for c in cmps:
            try:
                if val.startswith(c):
                    val = float(val.split(c)[1])
                    cmp = c
                    isnumerical = True
                    break
            except:
                pass

        if not isnumerical:
            val = float(val)

        bdist = bdists[attr]

        minv, maxv = bdist.min, bdist.max
        op = None
        if cmp == '>=':
            minv = val
        elif cmp == '>':
            minv = val
        elif cmp == '<=':
            maxv = val
        elif cmp == '<':
            maxv = val
        elif cmp == '=':
            maxv = minv = val
        else:
            raise

        return Orange.data.filter.ValueFilterContinuous(
            position=table.domain.index(attr),
            oper=orange.ValueFilter.Between,
            min=minv,
            max=maxv)


def rule_from_clauses(table, clauses):
    domain = table.domain
    pos_to_cont_cond = {}
    pos_to_disc_cond = {}
    for c in clauses:
        pos = c.position
        attr = domain[pos]
        if attr.varType == Orange.feature.Type.Discrete:
            if pos in pos_to_disc_cond:
                vals = pos_to_disc_cond[pos]
                vals = vals.intersection(set(map(int, c.values)))
            else:
                vals = set(map(int, c.values))
            pos_to_disc_cond[pos] = vals
        else:
            if pos in pos_to_cont_cond:
                minv, maxv = pos_to_cont_cond[pos]
                minv = max(minv, c.min)
                maxv = min(maxv, c.max)
            else:
                minv, maxv = c.min, c.max
            pos_to_cont_cond[pos] = (minv, maxv)

    conds = []
    for pos, (minv, maxv) in pos_to_cont_cond.iteritems():
        conds.append(Orange.data.filter.ValueFilterContinuous(
            position=pos,
            oper=orange.ValueFilter.Between,
            min=minv,
            max=maxv))
    for pos, vals in pos_to_disc_cond.iteritems():
        conds.append(Orange.data.filter.ValueFilterDiscrete(
            position=pos,
            values=[Orange.data.Value(domain[pos], v) for v in vals]))
    
    return SDRule(table, None, conds)


def tree_to_clauses(table, node, bdists=None, clauses=None, strclauses=None):
    clauses = clauses or []
    strclauses = strclauses or []
    if not node:
        return []
    if bdists is None:
      bdists = Orange.statistics.basic.Domain(table)

    ret = []
    if node.branch_selector:
        varname = node.branch_selector.class_var.name
        var = table.domain[varname]
        for branch, bdesc in zip(node.branches,
                                 node.branch_descriptions):
            if ( bdesc.startswith('>') or 
                 bdesc.startswith('<') or 
                 bdesc.startswith('=') ):
                clauses.append( create_clause(table, var, bdesc, bdists))
            else:
                clauses.append( create_clause(table, var, bdesc, bdists) )
            strclauses.append((varname, bdesc))
            ret.extend( tree_to_clauses(table, branch, bdists, clauses, strclauses) )
            clauses.pop()
    else:
        major_class = node.node_classifier.default_value
        #if major_class == '1' and clauses:
        ret.append(rule_from_clauses(table, clauses))

    ret = filter(bool, ret)
    return ret


