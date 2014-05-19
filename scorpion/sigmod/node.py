from itertools import chain

inf = float('inf')


class Node(object):
    def __init__(self, rule):
        self.rule = rule
        self.children = []
        self.parent = None
        self.n = 0
        self.influence = -inf
        self.prev_attr = None
        self.score = -inf


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
      if self.influence == -inf or self.influence > score:
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
      if self.n is None:
        return []
      if not self.children:
        return []
      nodes = [child.nonleaves for child in self.children] + [[self]]
      return chain(*nodes)
    nonleaves = property(__nonleaves__)



    def __nodes__(self):
      if self.n is None:
        return []
      allnodes = [[self]] + [child.nodes for child in self.children]
      return chain(*allnodes)
    nodes = property(__nodes__)

    def __str__(self):
        return '%.4f\t%d\t%s' % (self.influence, self.n, self.rule)


