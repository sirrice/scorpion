import sys
import math
import Orange
import orange
import cStringIO

from collections import defaultdict
from scorpion.util.rangeutil import *


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

    for col, bounds in ref_bounds.iteritems():
      if col in conds:
          continue

      attr = domain[col]
      pos = domain.index(attr)

      if bounds is None:
        vals = range(len(attr.values))
        vals = [orange.Value(attr, attr.values[v]) for v in vals]
        cond = orange.ValueFilter_discrete(position=pos, values=vals)
      else:
        (minv, maxv) = bounds
        
        cond = orange.ValueFilter_continuous(
          position=pos,
          oper = orange.ValueFilter.Between,
          min = minv-1,
          max = maxv+1
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
      if not data or data == self.data:
        ret = self.cloneAndNegate(self.filter.negate)
        ret.__examples__ = self.__examples__
      else:
        ret = SDRule(data, self.targetClass, self.filter.conditions[:], self.g, negate=self.filter.negate)

      ret.cluster_rules = set(self.cluster_rules)
      if self.c_range is not None:
        ret.c_range = list(self.c_range)
      if self.inf_state is not None:
        ret.inf_state = list(self.inf_state)
      ret.quality = self.quality
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


  def simplify(self, data=None, cdists=None, ddists=None):
    """
    Args:
      data:   non-filtered! data
      cdists: non-filtered Continuous distribution
      ddists: non-filtered discrete distribution
    """
    subset = data and self(data) or self.examples
    data = data or self.data
    ret = self.clone()

    positions = [cond.position for cond in self.filter.conditions]
    cdists = cdists or Orange.statistics.basic.Domain(data)
    ddists = ddists or Orange.statistics.distribution.Domain(data)
    #scdists = Orange.statistics.basic.Domain(subset)
    #sddists = Orange.statistics.distribution.Domain(subset)

    conds = []
    for old_cond, idx in zip(self.filter.conditions, positions):
      attr = data.domain[idx]

      # if rule values == full dataset values, then remove rule
      # filter down to the values that intersect the subset of data
      if attr.var_type == Orange.feature.Type.Discrete:
        full_d = ddists[attr.name]
        #sub_d = sddists[attr.name]
        fvals = [k for k,v in full_d.items() if v]
        cvals = set([str(attr.values[int(v)]) for v in old_cond.values])
        if len(cvals) == len(fvals):
          continue

        #dvals = [k for k,v in sub_d.items() if v]
        #vals = set(cvals).intersection(dvals)
        vals = cvals
        cond = orange.ValueFilter_discrete(
          position = idx,
          values = [orange.Value(attr, val) for val in vals]
        )
      else:
        fb = cdists[attr.name]
        #sb = scdists[attr.name]
        old_bound = [fb.min, fb.max]
        cond_bound = [old_cond.min, old_cond.max]

        bound = r_intersect(old_bound, cond_bound)
        if r_vol(bound) >= r_vol(old_bound): continue
        #bound = r_intersect(bound, [sb.min, sb.max])
        cond = old_cond
        cond.min, cond.max = bound[0], bound[1]
      conds.append(cond)
      continue



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
    """
    Translate condition into a dictionary necoding the column, data type and values
    Useful for JSONifying the rule
    """
    ret = {
      'pos': None,
      'col': None,
      'type': None,
      'vals': None
    }

    domain = self.data.domain
    ret['col'] = name = domain[c.position].name        
    ret['pos'] = c.position

    if domain[c.position].varType == orange.VarTypes.Discrete:
      if len(c.values) == 0 or len(c.values) == len(domain[c.position].values):
        return None

      def reencode(v):
        if v == 'None' or v == 'NULL':
          return None
        return v
      vals = [domain[c.position].values[int(vidx)] for vidx in c.values]
      vals = map(reencode, vals)
      ret['type'] = 'str'
      ret['vals'] = vals
    else:
      ret['type'] = 'num'

      if any([t in str(type(c.min)) for t in ['date', 'time']]):
        ret['type'] = 'timestamp'
      
      ret['vals'] = [max(-1e100, c.min), min(1e100, c.max)]

    return ret
  
  @staticmethod
  def dictToCond(d, data):
    if d['type'] == 'num':
      return orange.ValueFilter_continuous(
          position = d['pos'],
          oper = orange.ValueFilter.Between,
          min = d['vals'][0],
          max = d['vals'][1]
      )

    # XXX: NULL hack
    attr = data.domain[d['col']]
    vals = []
    for v in d['vals']:
      if v is None:
        if 'NULL' in attr.values:
          v = 'NULL'
        elif 'None' in attr.values:
          v = 'None'
      vals.append(orange.Value(attr, v))
    return orange.ValueFilter_discrete(position=d['pos'], values=vals)


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
    return '%.2f %d  %s' % ((self.quality or -inf), len(self.examples), rule)

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

  def to_json(self):
    return {
      #'conds': self.toCondDicts(),
      'clauses': self.cond_dicts,
      'count': len(self.examples),
      'c_range': self.c_range,
      'quality': self.quality,
      'score': self.quality,
      'inf_state': self.inf_state,
      'alt_rules': [r.cond_dicts for r in self.cluster_rules]
    }

  @staticmethod
  def from_json(j, data=None):
    conds = [SDRule.dictToCond(d, data) for d in j['clauses']]
    rule = SDRule(data, None, conds, None)
    rule.c_range = j['c_range']
    rule.quality = j['quality']
    rule.inf_state = j['inf_state']
    return rule


  
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

def rule_to_json(rule, **kwargs):
  """
  Args
    kwargs: other key-val pairs to add in the json object
            e.g., yalias: "xxx"
  Returns
    JSON object
  """
  def rnd(v):
    if v == float('-inf'): return -1e100
    if v == float('inf'): return 1e100
    return round(v, 3)

  ret = rule.to_json()
  ret['c_range'] = map(rnd, ret['c_range'])
  ret['score'] = rnd(ret['score'])
  ret['quality'] = rnd(ret['quality'])
  return ret


def fill_in_rules(rules, table, cols=None, cont_dists=None):
    # compute bounds for columns in self.cols
    if cols is None:
        cols = [attr.name for attr in table.domain]

    nparr = None
    ref_bounds = {}
    for col in cols:
        attr = table.domain[col]
        if attr.var_type == Orange.feature.Type.Discrete:
            ref_bounds[col] = None
            continue

        if cont_dists:
          bound = cont_dists[attr.name].min, cont_dists[attr.name].max
        else:
          if nparr is None:
            nparr = table.to_numpyMA('ac')[0]
          pos = table.domain.index(attr)
          arr = nparr[:,pos]
          bound = (arr.min(), arr.max())

        ref_bounds[col] = bound


    for rule in rules:
        rule.fill_in_rule(table, ref_bounds)



# model rules as trees.  look for opportunities to prune --> how?
