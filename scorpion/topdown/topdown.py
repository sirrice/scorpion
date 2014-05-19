import time
import pdb
import sys
import Orange
import orange
sys.path.extend(['.', '..'])


from learners.cn2sd.rule import fill_in_rules
from score import QuadScoreSample7
from bottomup.bounding_box import *
from bottomup.cluster import *
from merger.merger import Merger
from merger.reexec import ReexecMerger
from util import *
from sampler import SampleDecisionTree

class TopDown(object):
    def __init__(self, **kwargs):
        self.aggerr = kwargs.get('aggerr', None)
        self.cols = list(self.aggerr.agg.cols)
        self.err_func = kwargs.get('err_func', self.aggerr.error_func.clone())
        self.klass = SampleDecisionTree#QuadScoreSample7
        self.merger = None
        self.params = {}
        

        self.scorer_cost = 0.
        self.merge_cost = 0.

        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        self.cols = kwargs.get('cols', self.cols)
        self.klass = kwargs.get('klass', self.klass)
        self.params.update(kwargs)


        

    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        Topdown will use all columns to construct rules
        """
        self.set_params(**kwargs)

        self.bad_tables = bad_tables
        self.good_tables = good_tables

        # allocate and setup error functions
        self.bad_err_funcs = [self.err_func.clone() for t in bad_tables]
        self.good_err_funcs = [self.err_func.clone() for t in good_tables]

        for ef, t in zip(chain(self.bad_err_funcs,self.good_err_funcs),
                         chain(bad_tables, good_tables)):
            ef.setup(t)

        # self.err_func.setup(table)
        # self.table = table

        #rules = self.get_scorer_rules(table, self.cols, self.err_func)
        rules = self.get_scorer_rules(full_table,
                                      bad_tables,
                                      good_tables,
                                      self.bad_err_funcs,
                                      self.good_err_funcs,
                                      self.cols)

        self.err_func = self.bad_err_funcs[0]
        fill_in_rules(rules, full_table, cols=self.cols)
        self.all_clusters = [Cluster.from_rule(r, self.cols) for r in rules]

        

        thresh = compute_clusters_threshold(self.all_clusters)
        is_mergable = lambda c: c.error >= thresh
        print "threshold", thresh

        
        start = time.time()
        params = dict(self.params)
        params.update({'cols' : self.cols,
                       'table' : full_table,
                       'err_func' : self.err_func})
        self.merger = ReexecMerger(**params)
        self.final_clusters = self.merger(self.all_clusters, is_mergable=is_mergable)
        self.final_clusters.sort(key=lambda c: c.error, reverse=True)
        self.merge_cost = time.time() - start

        return self.final_clusters

    def get_scorer_rules(self,
                         full_table,
                         bad_tables,
                         good_tables,
                         bad_err_funcs,
                         good_err_funcs,
                         cols):
        start = time.time()
        params = dict(self.params)
        if 'err_func' in params:
            del params['err_func']
        self.scorer = self.klass(full_table,
                                 bad_tables,
                                 good_tables,
                                 bad_err_funcs,
                                 good_err_funcs,
                                 cols,
                                 **params)
        self.scorer()
        rules = self.scorer.rules
        self.scorer_cost = time.time() - start
        return rules
        
        

    def fill_in_rules(self, rules, table, cols=None):
        # compute bounds for columns in self.cols
        if cols is None:
            cols = [attr.name for attr in table.domain]
            
        nparr = table.to_numpyMA('ac')[0]
        ref_bounds = {}
        for col in cols:
            attr = table.domain[col]
            if attr.var_type == Orange.feature.Type.Discrete:
                continue
            pos = table.domain.index(attr)
            arr = nparr[:,pos]
            ref_bounds[col] = (arr.min(), arr.max())
        
        
        for rule in rules:
            self.fill_in_rule(rule, table, ref_bounds)

    def fill_in_rule(self, rule, table, ref_bounds):
        domain = table.domain

        # if there are any cols not in the rule, fill them in with table bounds
        conds = {}
        for c in rule.filter.conditions:
            attr = domain[c.position]
            name = attr.name
            conds[name] = True

        for col, (minv, maxv) in ref_bounds.iteritems():
            if col in conds:
                continue
            
            pos = domain.index(domain[col])
            cond = orange.ValueFilter_continuous(
                position=pos,
                oper = orange.ValueFilter.Between,
                min = minv,
                max = maxv
                )
            rule.filter.conditions.append(cond)



class DecisionTopDown(TopDown):

    def tree_to_rule(self, table, node, conds=None):
        conds = conds or []
        if not node:
            return []

        ret = []
        if node.branch_selector:
            attr = node.branch_selector.class_var
            for branch, bdesc in zip(node.branches,
                                     node.branch_descriptions):

                minv, maxv = -1e10000, 1e10000
                op = None
                if bdesc.startswith('>='):
                    op = Orange.data.filter.ValueFilter.GreaterEqual
                    minv = float(bdesc[2:])
                elif bdesc.startswith('>'):
                    op = Orange.data.filter.ValueFilter.Greater
                    minv = float(bdesc[1:])
                elif bdesc.startswith('<='):
                    op = Orange.data.filter.ValueFilter.LessEqual
                    maxv = float(bdesc[2:])
                elif bdesc.startswith('<'):
                    op = Orange.data.filter.ValueFilter.Less
                    maxv = float(bdesc[1:])
                elif bdesc.startswith('='):
                    op = Orange.data.filter.ValueFilter.Equal
                    maxv = minv = float(bdesc[1:])

                if op:
                    op = orange.ValueFilter.Between                    
                    conds.append(Orange.data.filter.ValueFilterContinuous(
                        position=node.branch_selector.position,
                        oper=orange.ValueFilter.Between,
                        min=minv,
                        max=maxv))
                else:
                    vals = [orange.Value(attr, str(bdesc))]
                    conds.append(Orange.data.filter.ValueFilterDiscrete(
                        position=node.branch_selector.position,
                        values=vals
                        ))
                    #conds.append( self.create_cond(table, node.branch_selector.position, bdesc) )
                ret.extend( self.tree_to_rule(table, branch, conds) )
                conds.pop()                    


                
        else:
            major_class = node.node_classifier.default_value
            if major_class == '1' and conds:
                ret.append( SDRule(table, None, conds) )

        ret = filter(lambda c: c, ret)
        return ret


    def create_cond(self, table, pos, val, cmp='='):
        cmps = ['<', '<=', '>', '>=', '=']

        if isinstance(val, (list, tuple)):
            strings = filter(lambda v: isinstance(v, str), val)
            pdb.set_trace()
            
            if len(strings):
                val = map(quote_sql_str, val)
            cmp = 'in'
            val = '(%s)' % ','.join(map(str, val))
        else:
            pdb.set_trace()
            if isinstance(val, str):
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
                    try:
                        val = float(val)
                    except:
                        val = quote_sql_str(val)

        return None

    


    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        cols = self.cols
        self.cols = self.aggerr.agg.cols
        TopDown.__call__(self, full_table, bad_tables, good_tables, **kwargs)
        self.cols = cols
        table = full_table

        domain = Orange.data.Domain(table.domain, Orange.feature.Discrete('err', values=['0', '1']))
        rows = []
        for row in table:
            row = row.native()
            row.append('0')
            rows.append(row)
        table = Orange.data.Table(domain, rows)
        rules = clusters_to_rules(filter_top_clusters(self.final_clusters), table)

        total = 0
        for rule in rules:
            print 'setting for rule %s' % rule
            for row in rule.filter_table(table):
                row.set_class('1')
                total += 1
        print "setting %d error tuples of %d" % (total, len(table))


        table = rm_attr_from_domain(table, self.aggerr.agg.cols)

            

        import orngTree

        tree = Orange.classification.tree.C45Learner(table, cf=0.001)
        rules = c45_to_clauses(table, tree.tree)
        #print tree
        
        error = max(self.final_clusters, key=lambda c:c.error).error + 1

        self.all_clusters = self.final_clusters = [Cluster.from_rule(rule, self.cols, error=error) for rule in rules]
        return self.final_clusters


        
class SmartDecisionTopDown(DecisionTopDown):
    


    pass

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

def c45_to_clauses(table, node, clauses=None):
    clauses = clauses or []
    if not node:
        return []
    
    if node.node_type == 0: # Leaf
        if int(node.leaf) == 1 and node.items > 0 and clauses is not None:
            return [rule_from_clauses(table, clauses)]
        return []

    var = node.tested
    ret = []


    if node.node_type == 1: # Branch
        for branch, val in zip(node.branch, var.values):
            clause = create_clause(table, var,  val)
            clauses.append( clause )
            ret.extend( c45_to_clauses(table, branch, clauses) )
            clauses.pop()

    elif node.node_type == 2: # Cut
        for branch, comp in zip(node.branch, ['<=', '>', '<', '>=']):
            clause = create_clause(table, var,  node.cut, comp)
            clauses.append( clause )
            ret.extend( c45_to_clauses(table, branch, clauses) )
            clauses.pop()

    elif node.node_type == 3: # Subset
        for i, branch in enumerate(node.branch):
            inset = filter(lambda a:a[1]==i, enumerate(node.mapping))
            inset = [var.values[j[0]] for j in inset]
            if len(inset) == 1:
                clause = create_clause(table, var, inset[0])
            else:
                clause = create_clause(table, var, inset)
            clause.append( clause )
            ret.extend( c45_to_clauses(table, branch, clauses) )
            clauses.pop()

    ret = filter(lambda c: c, ret)
    return ret



def create_clause(table, attr, val, cmp='='):
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


        minv, maxv = -1e10000, 1e10000
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
