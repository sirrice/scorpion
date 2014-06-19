from itertools import *

from ..learners.cn2sd.rule import SDRule
from ..util import powerset

INF = float('inf')

class Node(object):
    def __init__(self, rule):
        self.rule = rule
        self.children = []
        self.parent = None
        self.n = 0
        self.influence = -INF
        self.prev_attr = None
        self.score = -INF


        # boolean set if it was cloned from results of
        # partitioning outlier outputs.  See bdt.get_partitions()
        self.frombad = False


        self.cards = None # caches the cardinality of in each input group
        self.states = None # caches M-tuples

    def clone(self):
      """
      clones the structure of the tree, but loses score/influence info
      """
      node = Node(self.rule.clone())
      for child in self.children:
        child = child.clone()
        child.parent = node
        node.add_child(child)
      return node

    def set_score(self, score):
      if self.influence == -INF or self.influence > score:
        self.influence = score
        if self.parent:
          self.parent.set_score(score)

    def add_child(self, child):
        self.children.append(child)
    
    def __depth__(self):
        if not self.parent:
          return 0
        return 1 + self.parent.depth
    depth = property(__depth__)

    def __path__(self, q=None):
      q = q or list()
      if self.parent:
        self.parent.__path__(q)
      q.append(self)
      return q
    path = property(__path__)
        

    def __leaves__(self):
      if self.n is None:
        return []
      if not self.children:
        return [self]
      return chain(*[child.leaves for child in self.children])
    leaves = property(__leaves__)

    def __nonleaves__(self):
      """
      @deprecated
      Return ancestors of the leaf nodes found in this tree
      """
      if self.n is None:
        return []
      if not self.children:
        return []
      nodes = [child.nonleaves for child in self.children] + [[self]]
      return chain(*nodes)

    #def __nonleaves__(self):
    #  """
    #  Return all possible parents of the leaf nodes
    #  e.g., if leaf node has conditions (a,b,c)
    #        returns (a), (b), (c), (a,b), (a,c), (b,c), (a,b,c)
    #  """
    #  print "get nonleaves"
    #  ret = set()
    #  for leaf in self.leaves:
    #    conds = leaf.rule.filter.conditions
    #    for subset in powerset(conds):
    #      if len(subset) == len(conds): continue
    #      newrule = SDRule(leaf.rule.data, None, subset, leaf.rule.g)
    #      ret.add(newrule)
    #  print "done"
    #  return map(Node, ret)
    nonleaves = property(__nonleaves__)



    def __nodes__(self):
      if self.n is None:
        return []
      allnodes = [[self]] + [child.nodes for child in self.children]
      return chain(*allnodes)
    nodes = property(__nodes__)

    def __str__(self):
        return '%.4f\t%d\t%s' % (self.influence, self.n, self.rule)


