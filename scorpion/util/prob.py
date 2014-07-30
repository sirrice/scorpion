import logging
import sys
import random
import math
import matplotlib
import numpy as np


def wmean(vals, weights=None):
  np.average(vals, weights=weights)

def wstd(vals, weights=None):
  """
  compute a weighted standard deviation
  """
  mean = wmean(vals, weights)
  vals = np.array(vals).astype(float)
  weights = np.array(weights).astype(float)    
  weights /= weights.sum()

  top = (weights * ((vals - mean)**2)).sum()
  bot = weights.sum()
  return math.sqrt(top / bot)
  

def prob_no_error(pop, errprob, npts):
  """
  computers the probability from a population with error probability `errprob` that
  a random sample of `npts` points will not contain any error points

  @param pop population size
  @param errprob probability of an error point (e.g., user specified lower bound)
  @param npts sample size
  """
  def choose(n, k):
    n,k = int(n), int(k)
    v = 0.
    if k > n/2:
      v += sum(map(math.log, xrange(k,n+1)))
      v -= sum(map(math.log, xrange(1,n-k+1)))
    else:
      v += sum(map(math.log, xrange(n-k,n+1)))
      v -= sum(map(math.log, xrange(1,k+1))) 
    return v
  
  # c(pop*(1-errprob), npts) / c(pop, npts)
  c1 =  choose(pop*(1-errprob), npts)
  c2 = choose(pop, npts)
  return math.exp(c1 - c2)

   
def best_sample_size(pop, errprob, confidence=0.95):
  """
  given a population and an error probability, computes the the minimum sample size `s` such
  that with 95% confidence, `s` will contain at least one error point
  """
  sample_size, best_prob = None, None
  threshold = max(0, 1. - confidence)
  mins, maxs = 1, pop

  while maxs - mins > 20:
    size = max(1, int((maxs + mins) / 2.))
    #print size, '\t', prob_no_error(pop, errprob, size)
    good = prob_no_error(pop, errprob, size) < threshold
    if good:
      # either this is the best, or we should update the ranges and
      # look again
      if prob_no_error(pop, errprob, size-1) < threshold:
        maxs = size
        continue
      else:
        return size
    else:
      mins = size+1

  for size in xrange(mins, maxs+1, 1):
    if prob_no_error(pop, errprob, size) < threshold:
      return size
  return pop


def sample_size(moe=0.1, pop=10, zval=2.58):
  """
  sample size based on estimator closed form solutions
  @param moe margin of error
  @param pop population size (size of partition)
  @param zval confidence interval (default: 99%)  95% is 1.96
  """
  ss = ((zval**2) * 0.25) / (moe ** 2)
  if pop:
      ss = ss / (1 + (ss-1)/pop)
  return min(ss, pop)

