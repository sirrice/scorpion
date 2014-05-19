import sys
import math
import Orange
import orange
import cStringIO

from collections import defaultdict


inf = infinity = 1e10000

#class SDRule (orange.Rule):
class SDRule(object) :
    __id__ = 0

    
    def __init__(self, data, targetClass, conditions=[] ,g =1, negate=False, c_range = None):
        self.g = g
        self.data = data
        self.targetClass = targetClass
        self.filter = orange.Filter_values(domain = data.domain,
                                           conditions=conditions,
                                           conjunction=1,
                                           negate=negate)
        self.__examples__ = None
        self.filterAndStore()   # set examples, classifier, distribution; TP, calculate quality, complexity, support
        self.id = SDRule.__id__
        self.fixed = False # fixed rules cannot be extended

        self.weight = 1.
        self.quality = -inf
        self.score = 0.
        self.score_norm = None
        self.inf_state = None
        self.isbest = False

        self.c_range = c_range
        if not self.c_range: self.c_range = [infinity, -infinity]

        self.stats_mean = None
        self.stats_std = None
        self.stats_meannorm = None
        self.stats_nmean = None
        self.stats_nstd = None
        self.stats_nmeannorm = None
        self.stats_max = None
        self.parent_rule = None

        # rules within the cluster (for Basic.group_rules)
        self.cluster_rules = set()
        SDRule.__id__ += 1

    def __improvement__(self):
        if self.parent_rule:
            return self.quality - self.parent_rule.quality
        return self.quality
    improvement = property(__improvement__)

    def set_data(self, data):
        # need to convert discrete conditions to the new dataset -- ugh
        newconds = []
        for cond in self.filter.conditions:
            attr = self.data.domain[cond.position]
            newattr = data.domain[attr.name]
            if newattr.var_type == Orange.feature.Type.Discrete:
                if type(cond) == Orange.data.filter.ValueFilterContinuous:
                    # find discrete values within filtered range
                    # construct new discrete filter
                    subset_table = self.data.select_ref(Orange.data.filter.Values(
                        domain=self.data.domain,
                        conditions=[cond]))
                    newvals = set()
                    for row in subset_table:
                        if row[attr].is_special():
                            continue
                        oldval = row[attr].value
                        try:
                            v = orange.Value(newattr, str(oldval))
                            newvals.add(str(oldval))
                        except:
                            try:
                                v = orange.Value(newattr, str(int(oldval)))
                                newvals.add(str(int(oldval)))
                            except:
                                raise

                    newvals = map(lambda v: orange.Value(newattr, v), newvals)
                else:
                    newvals = []
                    for val in cond.values:
                        try:
                            newvals.append( orange.Value(newattr, attr.values[int(val)])  )
                        except:
                            pass

                if not newvals:
                    continue
                
                newcond = orange.ValueFilter_discrete(
                    position = data.domain.index(newattr),
                    values = newvals)
                newconds.append(newcond)
            else:
                if type(cond) == Orange.data.filter.ValueFilterDiscrete:
                    # construct new continuous condition
                    newcond = Orange.data.filter.ValueFilterContinuous(
                        position = data.domain.index(newattr),
                        oper = orange.ValueFilter.Between,
                        min = min(cond.values),
                        max = max(cond.values))
                else:
                    newcond = cond
                newconds.append(newcond)
        
        self.filter = orange.Filter_values(domain = data.domain,
                                           conditions=newconds,
                                           conjunction=1,
                                           negate=self.filter.negate)
                    
        
        self.data = data 
        self.filterAndStore()

    def __get_examples__(self):
        if not self.__examples__:
            self.__examples__ = self.filter_table(self.data)
        return self.__examples__
    examples = property(__get_examples__)

    def filterAndStore(self):
        self.complexity = len(self.filter.conditions)
        return
        c = 0
        for cond in self.filter.conditions:
            c += 1
            if type(cond) == Orange.data.filter.ValueFilterDiscrete:
                c += max(0., 0.2 - 1. / len(cond.values))
        
        self.complexity = c

    def covers(self, example):
        return len(self.filter([example])) > 0

    def filter_table(self, table, **kwargs):
        try:
            return table.filter_ref(self.filter, **kwargs)
        except:
            return self.filter(table, **kwargs)

    def __call__(self, table):
        return self.filter_table(table)


    def fill_in_rule(self, table, ref_bounds):
        domain = table.domain

        # if there are any cols not in the rule, fill them in with table bounds
        conds = {}
        for c in self.filter.conditions:
            attr = domain[c.position]
            name = attr.name
            conds[name] = True

        filtered = self.filter_table(table)
        filteredarr = filtered.to_numpyMA('ac')[0]

        for col, bounds in ref_bounds.iteritems():
            if col in conds:
                continue

            if bounds is None:
                attr = domain[col]
                pos = domain.index(attr)
#                vals = map(int, set(filteredarr[:,pos]))
                vals = range(len(attr.values))
                vals = [orange.Value(attr, attr.values[v]) for v in vals]
                cond = orange.ValueFilter_discrete(position=pos, values=vals)
            else:
                (minv, maxv) = bounds
                
                pos = domain.index(domain[col])
                cond = orange.ValueFilter_continuous(
                    position=pos,
                    oper = orange.ValueFilter.Between,
                    min = minv,
                    max = maxv
                    )
            self.filter.conditions.append(cond)


    def cloneAndAppendRule(self, rule):
        conds = list(self.filter.conditions)
        conds.extend(rule.filter.conditions)

        return SDRule(self.data, self.targetClass, conds, self.g)

    def cloneAndAppendCondition(self, cond):
      attr = self.data.domain[cond.position]
      if type(cond) == Orange.data.filter.ValueFilterContinuous:
        return self.cloneAndAddContCondition(attr, cond.min, cond.max)
      else:
        values = cond.values
        vals = [attr.values[int(v)] for v in values]
        return self.cloneAndAddCondition(attr, vals)



    def cloneAndAddCondition(self, attribute, values, used=False, negate=False):
        '''Returns a copy of this rule which condition part is extended by attribute = value'''
        conds = list(self.filter.conditions)

        if not(values):
            return self
        if not isinstance(values, list):
            values = [values]

        pos = self.data.domain.index(attribute)                    
        conds = filter(lambda cond: cond.position != pos, conds)
        values = [orange.Value(attribute, value) for value in values]
        conds.append(
            orange.ValueFilter_discrete(
                position = self.data.domain.index(attribute),
                values = values
                )
            )
        conds.sort(key=lambda c: c.position)

        return SDRule (self.data, self.targetClass, conds, self.g)

    def cloneAndAddContCondition(self, attribute, minv, maxv, op=orange.ValueFilter.Between, used=False, negate=False):
        conds = list(self.filter.conditions)
        pos = self.data.domain.index(attribute)

        if used:
            for cond in conds:
                if cond.position == pos:
                    assert op == orange.ValueFilter.Between
                    minv = max(cond.min, minv)
                    maxv = min(cond.max, maxv)
            conds = filter(lambda cond: cond.position != pos, conds)
        vfc = Orange.data.filter.ValueFilterContinuous(
            position=pos,
            oper=op,
            min=minv,
            max=maxv)

        conds.append( vfc )
        conds.sort(key=lambda c: c.position)
        return SDRule (self.data, self.targetClass, conds, self.g)

    def cloneAndNegate(self, negate=True):
        conds = self.filter.conditions[:]
        return SDRule(self.data, self.targetClass, conds, self.g, negate=negate)

    def clone(self, data=None):
        if not data:
            ret = self.cloneAndNegate(self.filter.negate)
            ret.cluster_rules = set(self.cluster_rules)
            return ret
        ret = SDRule(data, self.targetClass, self.filter.conditions[:], self.g, negate=self.filter.negate)
        ret.cluster_rules = set(self.cluster_rules)
        ret.c_range = list(self.c_range)
        return ret

    def cloneWithNewData(self, newdata):
        conds = self.filter.conditions[:]
        rule = SDRule(newdata, self.targetClass, conds, self.g, negate=self.filter.negate)
        rule.quality = self.quality
        rule.fix = self.fixed
        rule.stats_mean = self.stats_mean
        rule.stats_std = self.stats_std 
        rule.stats_nmean = self.stats_nmean
        rule.stats_nstd = self.stats_nstd 
        rule.stats_max = self.stats_max 
        return rule


    def isSubsumed(self, rule):
        """
        Returns True if self is subsumed by rule
        subsumed if self's conditions are tighter than rule, but
        has the same number examples
        """
        for c in rule.filter.conditions:
            found = False
            for myc in self.filter.conditions:
                # does c contain myc?
                if c.position != myc.position:
                    continue
                if rule.data.domain[c.position].varType != self.data.domain[myc.position].varType:
                    continue
                if rule.data.domain[c.position].varType == orange.VarTypes.Discrete:
                    domain = rule.data.domain[c.position]
                    cvals = [domain.values[int(vidx)] for vidx in c.values]
                    
                    domain = self.data.domain[myc.position]
                    mycvals = [domain.values[int(vidx)] for vidx in myc.values]
                    
                    if not set(cvals).issuperset(set(mycvals)):
                        continue
                else:
                    if not(myc.min >= c.min and myc.max <= c.max):
                        continue                    
                found = True
                break
            if not found:
                return False
        return True
        
    
    def isIrrelevant(self, rule):
        '''Returns True if self is irrelevant compared to rule.'''
        def isSubset(subset, set):
            if len(subset) > len(set):
                return False
            else:
                index = 0;
                for e in subset:
                    while index<len(set) and e != set[index]:
                        index+=1
                    if index >= len(set):
                        return False
                return True

        if isSubset(self.TP, rule.TP) and isSubset(rule.FP, self.FP):
            return True
        else:
            return False

    def __lt__(self, o):
        if self.quality < o.quality:
            return True
        elif self.quality == o.quality:
            if self.complexity > o.complexity:
                return True
            elif self.complexity == o.complexity:
                if len(self.examples) < len(o.examples):
                    return True
        return False

    def __gt__(self, o):
        return o.__lt__(self)

    def eq(self, o):
        return (self.quality == o.quality and
                self.complexity == o.complexity and
                len(self.examples) == len(o.examples))

    def __ne__(self, o):
        return not self.eq(o)
        
    def __cmp__(self, other):
        if self.quality < other.quality:
            return -1
        if self.quality > other.quality:
            return 1
        if self.complexity > other.complexity:
            return -1
        if self.complexity < other.complexity:
            return 1
        if len(self.examples) < len(other.examples):
            return -1
        if len(self.examples) > len(other.examples):
            return 1
        return 0

    def __hash__(self):
        condStrs = self.toCondStrs()
        condStrs.append('rule')
        condStrs.sort()
        return hash(tuple(condStrs))


    def __eq__(self, o):
        return hash(self) == hash(o)

    def __str__(self):
        return self.ruleToString()

    def __attributes__(self):
        domain = self.data.domain
        attrs = list(set([domain[c.position].name for c in self.filter.conditions]))
        attrs.sort()
        return tuple(attrs)
    attributes = property(__attributes__)


    def simplify(self, data=None, ddists=None, bdists=None):
        #subset = data and self(data) or self.data
        data = data or self.data #examples
        ret = self.clone()

        positions = [cond.position for cond in self.filter.conditions]
        full_ddists = ddists or Orange.statistics.distribution.Domain(data)
        full_bdists = bdists or Orange.statistics.basic.Domain(data)

        conds = []
        for old_cond, idx in zip(self.filter.conditions, positions):
          attr = data.domain[idx]
          fd = full_ddists[idx]
          fb = full_bdists[idx]

          if attr.var_type == Orange.feature.Type.Discrete:
            cvals = set([str(attr.values[int(v)]) for v in old_cond.values])
            fvals = set([k for k,v in fd.items() if v])
            #svals = [k for k,v in sd.items() if v]
            vals = set(cvals).intersection(fvals)
            if len(vals) == len(fvals): continue
            cond = orange.ValueFilter_discrete(
              position = idx,
              values = [orange.Value(attr, val) for val in vals]
            )
          else:
            if old_cond.min <= fb.min and old_cond.max >= fb.max:
              continue
            cond = old_cond
          conds.append(cond)
          continue


          ret.filter.conditions = [old_cond]
          subset = ret.filter(data)
          sub_ddists = ddists or Orange.statistics.distribution.Domain(subset)
          sub_bdists = bdists or Orange.statistics.basic.Domain(subset)

          sd = sub_ddists[idx]
          sb = sub_bdists[idx]

          if attr.var_type == Orange.feature.Type.Discrete:
            svals = [k for k,v in sd.items() if v]
            fvals = [k for k,v in fd.items() if v]
            if set(svals) == set(fvals): continue

            cond = orange.ValueFilter_discrete(
                    position = idx,
                    values = [orange.Value(attr, val) for val in svals]
                    )
            conds.append(cond)
          else:
            if sb.min <= fb.min and sb.max >= fb.max:
                continue
            conds.append(old_cond)
            continue

            cond = Orange.data.filter.ValueFilterContinuous(
                    position=idx,
                    oper = orange.ValueFilter.Between,
                    min=fb.min,
                    max=fb.max)
        
        ret.quality = self.quality
        ret.filter.conditions = conds
        ret.c_range = list(self.c_range)
        return ret



    def condToString(self, c):
        domain = self.data.domain
        name = domain[c.position].name        

        if domain[c.position].varType == orange.VarTypes.Discrete:
            if len(c.values) == 0:
                return
            if len(c.values) == 1:
                vidx = int(c.values[0])
                v = domain[c.position].values[vidx]
            elif len(c.values) == len(domain[c.position].values):
                return
            else:
                vs = [domain[c.position].values[int(vidx)] for vidx in c.values]
                vs = map(str, vs)
                strlen = len(vs[0])
                idx = 0
                while idx < len(vs)-1 and strlen + len(vs[idx+1]) < 60:
                    strlen += len(vs[idx+1])
                    idx += 1
                v = ', '.join(vs[:idx+1])
                if idx < len(vs)-1:
                    v = '%s (%d+ ..)' % (v, len(vs) - 1 - idx)
            return '%s = %s' % (name, v)

        elif domain[c.position].varType == orange.VarTypes.Continuous:
            return '%.7f <= %s < %.7f' % (c.min, name, c.max) 

        return None

    def condToDict(self, c):
      ret = {
        'col': None,
        'type': None,
        'vals': None
      }

      domain = self.data.domain
      ret['col'] = name = domain[c.position].name        

      if domain[c.position].varType == orange.VarTypes.Discrete:
        if len(c.values) == 0 or len(c.values) == len(domain[c.position].values):
          return None

        vals = [domain[c.position].values[int(vidx)] for vidx in c.values]
        ret['type'] = 'str'
        ret['vals'] = vals
      else:
        ret['type'] = 'num'
        if any([t in str(type(c.min)) for t in ['date', 'time']]):
          ret['type'] = 'timestamp'
        ret['vals'] = [c.min, c.max]
      return ret




    def ruleToString(self):
        domain = self.data.domain
        ret = []
        for i,c in enumerate(self.filter.conditions):
            s = self.condToString(c)
            if s:
                ret.append(s)

        ret.sort()
        rule = ' and '.join(ret)
        if self.filter.negate:
            rule = '%s (Neg)' % rule
        return '%.4f  %s' % ((self.quality or -inf), rule)

    def toCondStrs(self):
        domain = self.data.domain
        ret = []
        if self.filter.negate:
          ret.append('neg')

        for i,c in enumerate(self.filter.conditions):
            s = self.condToString(c)
            if s:
                ret.append(s)

        return ret
    cond_strs = property(toCondStrs)


    def toCondDicts(self):
        domain = self.data.domain
        if self.filter.negate:
          # DON'T SUPPORT NEGATION YET
          pass

        dicts = map(self.condToDict, self.filter.conditions)
        dicts = filter(bool, dicts)
        return dicts
    cond_dicts = property(toCondDicts)


    
    def printRule(self):
        fmt = "quality= %2.2f complexity=%2d covered=%2d %s"
        return  fmt % (self.quality,
                       self.complexity,
                       len(self.examples),
                       self.ruleToString())



    def getUndiscretized(self, original_data):
        cond = []
        for c in self.filter.conditions:
            d_attribute = self.data.domain[c.position]
            if d_attribute in original_data.domain:
                c.position = original_data.domain.index(d_attribute)
                cond.append(c)
            else:
                position = original_data.domain.index(original_data.domain[d_attribute.name]) #[2:]])

                points = d_attribute.getValueFrom.transformer.points
                value_idx = int(c.values[0])

                if value_idx == 0: # '<='
                    cond.append(
                            orange.ValueFilter_continuous(
                                    position = position,
                                    max = points[0],
                                    min = float(-infinity),
                                    outside = False)
                                )
                elif 0 < value_idx < len(points): # (x,y]
                    cond.append(
                            orange.ValueFilter_continuous(
                                    position = position,
                                    max = points[value_idx],
                                    min = points[value_idx-1],     # zaprti interval '[' namesto odprti '('
                                    outside = False)
                                )
                elif value_idx == len(points): # '>'
                    cond.append(
                            orange.ValueFilter_continuous(
                                    position = position,
                                    max = float(infinity),
                                    min = points[-1],
                                    outside = True)
                                )


        rule = SDRule(original_data, self.targetClass, cond, self.g)
        rule.quality = self.quality
        rule.fix = self.fixed
        rule.score = self.score
        rule.stats_mean = self.stats_mean
        rule.stats_std = self.stats_std 
        rule.stats_nmean = self.stats_nmean
        rule.stats_nstd = self.stats_nstd 
        rule.stats_max = self.stats_max
        return rule
                










class SDRules(object):
    def __init__(self, listOfRules, targetClassRule, algorithmName="SD"):
        self.rules = listOfRules
        self.targetClassRule = targetClassRule
        self.algorithmName = algorithmName

    def makeSelection(self, indexes):
        """Returns the rules that are at specified indexes."""
        rulesSubset = []
        for i in indexes:
            rulesSubset.append(self.rules[i])
        return SDRules(rulesSubset, self.targetClassRule)

    def sortByConf(self):
        self.rules.sort(lambda x, y: -cmp(x.confidence, y.confidence))

    def printRules(self):
        for rule in self.rules:
            rs = rule.printRule()
            print rs


#
# Utility functions for list of rules
#

class RuleBound(object):

    def __init__(self, ref_bounds, rule=None, bounds=None, roots=None):
        self.ref_bounds = ref_bounds
        self.rule = rule
        self.bounds = bounds
        self.roots = roots

        if self.rule and not self.bounds:
            self.bounds = self.rule_to_bounds()

        self.id = hash(str(self))
        
        if not self.roots:
            self.roots = [self.id]

        self.area = self.get_area()

        
    def __hash__(self):
        return self.id

    def __eq__(self, o):
        return hash(o) == hash(self)

    def __str__(self):
        b = self.bounds
        ret = []
        for key in sorted(b.keys()):
            if key.startswith('_'): continue
            ret.append(key)
            try:
                ret.append(('%.4f ' * len(b[key][1])) % tuple(b[key][1]))
            except:
                ret.append(str(b[key][1]))
        return ':'.join(ret)

    def normalize_bounds(self):
        bounds = self.bounds
        if bounds:
            for key, (isd, vals) in bounds.iteritems():
                if isd:
                    bounds[key] = (isd, set(vals))
                else:
                    ref_isd, refb = self.ref_bounds[key]
                    vals = ((vals[0] - refb[0]) / (refb[1]-refb[0]),
                            (vals[1] - refb[0]) / (refb[1]-refb[0]))
                    bounds[key] = (isd, vals)
        self.area = self.get_area()                    
        return bounds

    def unnormalize_bounds(self):
        bounds = self.bounds
        if bounds:
            for key, (isd, vals) in bounds.iteritems():
                if isd:
                    bounds[key] = (isd, set(vals))
                else:
                    ref_isd, refb = self.ref_bounds[key]
                    vals = ((vals[0] * (refb[1]-refb[0])) + refb[0],
                            (vals[1] * (refb[1]-refb[0])) + refb[0])
                    bounds[key] = (isd, vals)
        self.area = self.get_area()                    
        return bounds
        

    def to_rule(self, data):
        if self.rule:
            ret = self.rule.clone()
            ret.set_data(data)
            return ret

        domain = data.domain
        conds = []
        for key in sorted(self.bounds.keys()):
            if key.startswith('_'):
                continue
            ref_isd, refb = self.ref_bounds[key]
            isd, vals = self.bounds[key]
            attr = domain[key]
            pos = domain.index(attr)
            
            if ref_isd:
                conds.append(orange.ValueFilter_discrete(
                    position = pos,
                    values = [orange.Value(attr, v) for v in vals]
                    )
                )
            else:
                
                conds.append(orange.ValueFilter_continuous(
                    position = pos,
                    oper = orange.ValueFilter.Between,
                    min = vals[0],
                    max = vals[1]
                    )
                )
                
        return SDRule(data, None, conds)
            

    def rule_to_bounds(self):
        rule = self.rule
        domain = rule.data.domain
        bounds = {}

        for c in rule.filter.conditions:
            attr = domain[c.position]
            name = attr.name
            pos = domain.index(attr)
            is_discrete = attr.varType == orange.VarTypes.Discrete

            # transform each attribute to the same type as the
            # reference
            if self.ref_bounds[name][0]:
                if is_discrete:
                    vals = set([attr.values[int(vidx)]
                                for vidx in c.values])
                else:
                    vals = set([v for v in self.ref_bounds[name]
                                if v >= c.min and v < c.max])
            else:
                if is_discrete:
                    vals = (min(vals), max(vals))
                else:
                    vals = [c.min, c.max]                    
                    

            is_discrete = self.ref_bounds[name][0]
            bounds[attr.name] = (is_discrete, vals)

        return bounds


    def union_range(self, attrname, r1, r2):
        """
        @param r1 a tuple of (is_discrete, values).  if null, then we
        use the range defined in self.bounds
        @param r2 see r1
        @return a tuple of (is_discrete, values) thas represents the
        union of the two ranges.  
        """
        if not r1 or not r2:
            return self.bounds[attrname]

        isd1, isd2 = r1[0], r2[0]
        vals1, vals2 = r1[1], r2[1]
        isd = isd1 and isd2

        if isd:
            vals = set(vals1).union(set(vals2))
        else:
            if isd1:
                vals1 = [vals1[0], vals1[1]]
            if isd2:
                vals2 = [vals2[0], vals2[1]]
            vals = min(vals1[0], vals2[0]), max(vals1[1], vals2[1])

        return isd, vals


    def intersect_range(self, attrname, r1, r2):
        if not r1:
            return r2
        if not r2:
            return r1

        isd1, isd2 = r1[0], r2[0]
        vals1, vals2 = r1[1], r2[1]
        isd = isd1 or isd2

        if isd:
            if not isd1:
                vals1 = [v for v in self.bounds[attrname]
                         if v >= vals1[0] and v < vals1[1]]
            if not isd2:
                vals2 = [v for v in self.bounds[attrname]
                         if v >= vals2[0] and v < vals2[1]]
            vals = set(vals1).intersection(set(vals2))
        else:
            vals = max(vals1[0], vals2[0]), min(vals1[1], vals2[1])

        return isd, vals


    def intersect(self, rb):
        keys = set(self.bounds.keys()).union(rb.bounds.keys())
        ret = {}
        for key in keys:
            if key.startswith('_'): continue
            r1 = self.bounds.get(key, None)
            r2 = rb.bounds.get(key, None)
            ret[key] = self.intersect_range(key, r1, r2)
        
        return RuleBound(self.ref_bounds, bounds=ret, roots=self.roots+rb.roots)

    def union(self, rb):
        keys = set(self.bounds.keys()).intersection(rb.bounds.keys())
        ret = {}
        for key in keys:
            if key.startswith('_'): continue
            r1 = self.bounds.get(key, None)
            r2 = rb.bounds.get(key, None)
            ret[key] = self.union_range(key, r1, r2)

        return RuleBound(self.ref_bounds, bounds=ret, roots=self.roots+rb.roots)

    def too_different(self, rb):
        """
        if any of the dimensions are > 10x different
        """
        if len(self.bounds) != len(rb.bounds):
            return True
        keys = set(self.bounds.keys()).intersection(rb.bounds.keys())

        for key in keys:
            if key not in self.bounds or key not in rb.bounds:
                return True
            isd1, vals1 = self.bounds[key]
            isd2, vals2 = rb.bounds[key]

            if not isd1:
                v1 = (vals1[1] - vals1[0])
                v2 = (vals2[1] - vals2[0])
                if v1 > 100 * v2 or v2 > 100 * v1:
                    return True
        return False
                

    def get_area(self):
        ret = 1.
        for key in self.bounds:
            if key.startswith('_'): continue
            (isd, vals) = self.bounds[key]
            if isd:
                ret *= len(vals) / float(len(self.ref_bounds[key][1]))
            else:
                rf = self.ref_bounds[key][1]
                ret *= max(0, (vals[1] - vals[0])) / (rf[1] - rf[0])

        return max(0,ret)



def rules_to_bounds(rules, ref_bounds):
    bounds = []
    for rule in rules:
        if isinstance(rule, SDRule):
            bound = RuleBound(ref_bounds, rule=rule)
        elif isinstance(rule, RuleBound):
            bound = rule
        else:
            raise

        bounds.append(bound)
    return bounds

def get_ref_bounds(data):
    bdists = Orange.statistics.basic.Domain(data)
    contattrs = []
    ret = {}
    for idx, attr in enumerate(data.domain):
        if attr.var_type == Orange.feature.Type.Discrete:
            isd = True
            vals = list(attr.values)
            ret[attr.name] = (isd, vals)
        else:
            contattrs.append(attr)
            ret[attr.name] = (False, [1e10000, -1e10000])

    for row in data:
        for attr in contattrs:
            if row[attr].value and row[attr].value != '?':
                ret[attr.name][1][0] = min(float(row[attr].value), ret[attr.name][1][0])
                ret[attr.name][1][1] = max(float(row[attr].value), ret[attr.name][1][1])
    return ret


def merge_rules(data, rules, threshold=0.001):
    ref_bounds = get_ref_bounds(data)
    kept_boxes = rules_to_bounds(rules, ref_bounds)
    kept_boxes = set(kept_boxes)

    while kept_boxes:
        #print len(kept_boxes)
        boxes = list(kept_boxes)
        added = False
        
        for idx, b1 in enumerate(boxes):
            for b2 in boxes[idx+1:]:
                #if b1.too_different(b2):
                #    continue

                u = b1.union(b2)
                i = b1.intersect(b2)
                a = u.area
                d = a - b1.area - b2.area + i.area
                
                #print '\t'.join(['%.7f']*6) % (d/a, a, d, b1.area, b2.area, i.area), b1, b2

                if i.area > (b1.area + b2.area):
                    import pdb
                    pdb.set_trace()
                    i.get_area()
                if a < 1e-10:
                    continue
                if d / a > (threshold - 1e-10):
                    continue

                kept_boxes.difference_update([b1, b2])
                added = added or u not in kept_boxes
                kept_boxes.add(u)


        if not added:
            break

    ret = []
    for bound in kept_boxes:
        ret.append(bound.to_rule(data))
        
    return ret



def rule_overlaps(rule, rules, threshold):
    """
    Checks if rule overlaps with any of the rules
    Overlap is defined by same set of discrete values and interval
    overlap of >= 95%
    """
    domain = rule.data.domain
    d = dict([(rule.data.domain[c.position].name, c)
              for c in rule.filter.conditions])
    
    for r2 in rules:
        boverlap = True
        domain2 = r2.data.domain
        for c2 in r2.filter.conditions:
            attr2 = domain2[c2.position]
            c = d.get(attr2.name, None)

            if (not c or
                domain[c.position].varType !=
                domain2[c2.position].varType):
                boverlap = False
                break
            
            attr = domain[c.position]
            
            if attr.varType == orange.VarTypes.Discrete:
                vals1 = [attr.values[int(vidx)] for vidx in c.values]
                vals2 = [attr2.values[int(vidx)] for vidx in c2.values]
                if set(vals1) != set(vals2):
                    boverlap = False
                    break
            else:
                if (math.isinf(c.min) != math.isinf(c2.min) or
                    math.isinf(c.max) != math.isinf(c2.max)):
                    boverlap = False
                    break

                if math.isinf(c.min) or math.isinf(c.max):
                    boverlap = False
                    break

                inner = abs(max(c.min, c2.min) - min(c.max, c2.max))
                outer = abs(max(c.max, c2.max) - min(c.min, c2.min))
                if float(inner) / outer < threshold:
                    boverlap = False
                    break

        if boverlap:
            return True
    return False
            

def remove_duplicate_rules(rules, threshold=0.95):
    # cluster by attributes    
    clusters = defaultdict(list)
    for r in rules:
        attrs = [r.data.domain[c.position].name
                 for c in r.filter.conditions]
        attrs.sort()
        clusters[tuple(attrs)].append(r)

    # merge rules that overlap significantly in their ranges
    # discrete rules are merged if they contain exact same set of
    # values
    ret = []
    for attrs, cluster in clusters.iteritems():
        uniquerules = []
        for rule in cluster:
            if not rule_overlaps(rule, uniquerules, threshold):
                uniquerules.append(rule)
        ret.extend(uniquerules)

    return filter(lambda r: r in ret, rules)

def remove_subsumed_rules(rules):
    """
    goes through rules in order of quality and removes rules that are
    subsumed by higher quality rules
    @return filtered list of rules
    """
    def subsumed(rule, better_rules):
        for br in better_rules:
            if rule.isSubsumed(br):
                return True
        return False

    rules = sorted(rules,
                   key=lambda r:(r.quality, r.complexity),
                   reverse=True) 

    ret = []
    for idx, rule in enumerate(rules):
        # does an earlier rule subsume this?
        if not subsumed(rule, rules[:idx]):
            ret.append(rule)
    return ret


def fill_in_rules(rules, table, cols=None):
    # compute bounds for columns in self.cols
    if cols is None:
        cols = [attr.name for attr in table.domain]

    nparr = table.to_numpyMA('ac')[0]
    ref_bounds = {}
    for col in cols:
        attr = table.domain[col]
        if attr.var_type == Orange.feature.Type.Discrete:
            ref_bounds[col] = None
            continue
        pos = table.domain.index(attr)
        arr = nparr[:,pos]
        ref_bounds[col] = (arr.min(), arr.max())


    for rule in rules:
        rule.fill_in_rule(table, ref_bounds)



# model rules as trees.  look for opportunities to prune --> how?
