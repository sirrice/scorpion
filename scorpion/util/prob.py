import logging
import sys
import random
import math
import matplotlib
import numpy as np


def wmean(vals, weights=None):
    if weights is None: return np.mean(vals)
    vals = np.array(vals)
    weights = np.array(weights).astype(float)
    return (vals * weights).sum() / weights.sum()


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
    


def max_prob(u, s):
    vs = []
    for i in xrange(1000):
        a = random.gauss(u,s)
        if a >= u:
            vs.append(a)
    return np.mean(vs)

def order(u1, s1, u2, s2):
    """
    -1 if u1 < u2
    1  if u1 > u2 
    """
    vs = []
    for i in xrange(1000):
        a,b = random.gauss(u1,s1), random.gauss(u2,s2)
        if a<u1 or b<u2:
            continue
        vs.append( a - b )
    return np.mean(vs)

def gauss(u, s):
    """
    returns a gaussian function defined by mean and standard deviation
    @param u mean
    @param s standard deviation
    """
    a,b,c = 1./ (s * math.sqrt(2*math.pi)), u, s
    f = lambda x: (a * math.exp(- (x-b)**2 / (2 * c**2) )) 
    return f
    

def gaussorder(u1, s1, u2, s2, nsteps=20.):
    f1, f2 = gauss(u1, s1), gauss(u2, s2)

    minv, maxv = max(u1, u2), max(u1 + s1*5, u2 + s2*5)
    v, inc = minv, (maxv-minv) / float(nsteps)
    vs = []
    while v <= maxv:
        a = f1(v) * v if v > u1 else 0.
        b = f2(v) * v if v > u2 else 0.
        vs.append( a - b )
        v += inc
    return np.mean(vs)
    


def welchs_ttest(n1, mean1, sem1, n2, mean2, sem2, alpha):
    svm1 = sem1**2 * n1
    svm2 = sem2**2 * n2
    if svm1 == 0 and svm2 == 0:
        return 1.

    t_s_prime = (mean1 - mean2)/math.sqrt(svm1/n1+svm2/n2)
    return t_s_prime
    t_alpha_df1 = scipy.stats.t.ppf(1-alpha/2, n1 - 1)
    t_alpha_df2 = scipy.stats.t.ppf(1-alpha/2, n2 - 1)
    t_alpha_prime = (t_alpha_df1 * sem1**2 + t_alpha_df2 * sem2**2) / \
    (sem1**2 + sem2**2)
    return abs(t_s_prime) > t_alpha_prime, t_s_prime, t_alpha_prime



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

    
    #import scipy as sp
    #c = sp.misc.comb
    #nerr = pop * errprob
    e = int(math.ceil(errprob * pop))  # number of error points that is in the sample
    # initial equation.  Has numerical overflow issues
    # prob = c(npts, e) * (errprob ** e) * ((1.-errprob) ** (npts - e))
    

    # npts choose e = npts! / (e! (npts-e)!)
    bce = 0
    for i in xrange(1, npts):
        bce += math.log(i)
    for i in xrange(1, e):
        bce -= math.log(i)
    for i in xrange(1, npts - e):
        bce -= math.log(i)


    prob = 0.
    # errprob ** e    
    prob += e * math.log(errprob)
    # (1 - errprob) ** (npts - e)
    prob += (npts - e) * math.log(1. - errprob)

    prob = bce + prob
    prob = math.pow(math.e, prob)
    return prob

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


if __name__ == '__main__':
    r = np.arange(10)
    print wmean(r, r)
    print wstd(r, r)
