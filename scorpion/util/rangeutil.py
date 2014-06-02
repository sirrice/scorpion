#
# helper methods for manipulating bounds
#

def r_vol(bound):
  return max(0, bound[1] - bound[0])

def r_empty(bound):
  return bound[1] <= bound[0]

def r_equal(bound1, bound2):
  return bound1[0] == bound2[0] and bound1[1] == bound2[1]

def r_lt(bound1, bound2):
  "bound1 values < bound2 values"
  return bound1[0] < bound2[0] and bound1[1] < bound2[1]

def r_lte(bound1, bound2):
  "bound1 values <= bound2 values"
  return bound1[0] <= bound2[0] and bound1[1] <= bound2[1]

def r_expand(bound, perc=0.05):
  v = bound[1] - bound[0]
  v *= perc
  return [bound[0]-v, bound[1]+v]

def r_scontains(bound1, bound2):
  "bound1 strictly contains bound2"
  return bound1[0] < bound2[0] and bound2[1] < bound1[1]

def r_contains(bound1, bound2):
  "bound1  contains bound2"
  return bound1[0] <= bound2[0] and bound2[1] <= bound1[1]


def r_intersect(bound1, bound2):
  return [max(bound1[0], bound2[0]), min(bound1[1], bound2[1])]

def r_union(bound1, bound2):
  return [min(bound1[0], bound2[0]), max(bound1[1], bound2[1])]

def r_subtract(bound1, bound2):
  """
  remove bound2 from bound1. 
  Return list of bound
  """
  if r_contains(bound2, bound1):
    return [ [bound1[0], bound1[0]] ]
  if r_scontains(bound1, bound2):
    return [ [bound1[0], bound2[0]], [bound2[1], bound1[1]] ]
  inter = r_intersect(bound1, bound2)
  if r_lte(inter, bound1):
    return [ [inter[1], bound1[1]] ]
  return [ [bound1[0], inter[0]] ]


