#
# Generates random data for SYNTH dataset in paper
# N dimensions, 1 group-by dim (5 outlier, 5 normal groups), 1 value attr
# Normal points draw value from N(u_h, s_h) -- gaussian, u_h mean, s_h std
# Outlier points draw value from N(u_o, s_o)
# Outlier points fall in a 10% (variable) volume box in k<=N dimensions
#
import pdb
import sys
import random
from operator import and_, mul
random.seed(0)

def rand_box(ndim, kdim, vol, bounds=None):
    """
    outlier attributes are a_1,...a_kdim
    """
    if not bounds:
        bounds = [(0, 100)] * ndim
    bedges = [b[1]-b[0] for b in bounds]

    totalvol = reduce(mul, bedges)
    usedvol = reduce(mul, bedges[kdim:], 1)
    availvol = max(0, vol * totalvol / usedvol)
    edge = availvol ** (1. / kdim)

    ret = []
    for attr, bound in zip(xrange(ndim), bounds):
        if attr < kdim:
            lower = random.random() * (bound[1]-bound[0] - edge) + bound[0]
            upper = lower + edge
        else:
            lower, upper = bound[0], bound[1]
        ret.append((lower, upper))
    return ret 


def in_box(pt, box):
    return reduce(and_, [l <= v and v <= u for v, (l, u) in zip(pt, box)])

def rand_point(ndim):
    return [random.random() * 100 for i in xrange(ndim)]


def gen_multi_outliers(npts, ndim, kdim, vol, uh=10, sh=5, uo=90, so=5):
  # completely reproducable
  random.seed(0)

  nclusters = 2
  norm_gen, mid_gen, outlier_gen = make_gen(uh, sh), make_gen((uo-uh)/2+uh, so),  make_gen(uo, so)
  med_boxes = [rand_box(ndim, kdim, vol/2.) for i in xrange(nclusters)]
  high_boxes = [rand_box(ndim, kdim, vol, bounds=med_box) for med_box in med_boxes]

  for med_box in med_boxes:
    print >>sys.stderr, map(lambda arr: map(int, arr), med_box)
  for high_box in high_boxes:
    print >>sys.stderr, map(lambda arr: map(int, arr), high_box)


  schema = ['a_%d' % i for i in xrange(ndim)] + ['g', 'v']

  def generate():
    for gid in xrange(10):
      for i in xrange(npts):
        pt = rand_point(ndim)

        # add group and value
        pt.append(gid)

        if gid >= 5 and any([in_box(pt, mb) for mb in med_boxes]):
          if any([in_box(pt, hb) for hb in high_boxes]):
            pt.append(outlier_gen())
          else:
            pt.append(mid_gen())
        else:
          pt.append(norm_gen())

        yield pt

  return med_boxes, high_boxes, schema, generate()



def gen_points(npts, ndim, kdim, vol, uh=10, sh=5, uo=90, so=5):
  # completely reproducable
  random.seed(0)

  norm_gen, mid_gen, outlier_gen = make_gen(uh, sh), make_gen((uo-uh)/2+uh, so),  make_gen(uo, so)
  outlier_box = rand_box(ndim, kdim, vol)
  super_box = rand_box(ndim, kdim, vol, bounds=outlier_box)
  print >>sys.stderr, map(lambda arr: map(int, arr), outlier_box)
  print >>sys.stderr, map(lambda arr: map(int, arr), super_box)


  schema = ['a_%d' % i for i in xrange(ndim)] + ['g', 'v']

  def generate():
    for gid in xrange(10):
      for i in xrange(npts):
        pt = rand_point(ndim)

        # add group and value
        pt.append(gid)

        if gid >= 5 and in_box(pt, outlier_box):
          if in_box(pt, super_box):
              pt.append(outlier_gen())
          else:
              pt.append(mid_gen())
        else:
          pt.append(norm_gen())

        yield pt

  return outlier_box, super_box, schema, generate()


def make_gen(u, s):
    return lambda: random.gauss(u, s)



if __name__ == '__main__':
    import sys
    if len(sys.argv) < 5:
        print "Generates SYNTH dataset for paper. (single k dim sub-cluster of outlier points) "
        print "tuple format: [a_0,..., a_ndim, a_group, a_val]"
        print "Usage:\tpython gensinglecluster.py npts ndim kdim volperc [uh, sh, uo, so]"
        print "\tnpts: number of points per group (10 groups, 5 outlier, 5 normal)"
        print "\tkdim: dimensions of outlier cluster"
        print "\tvolperc: percentage volume of outlier cluster (default: 10%)"
        print "\tuh/uo: mean of normal and outlier a_val value (defaults: 10, 90)"
        print "\tsh/so: std of a_val values (defaults: 5, 5)"
        exit()


    npts, ndim, kdim = map(int, sys.argv[1:4])
    volperc = float(sys.argv[4])
    if len(sys.argv) > 5:
        uh, sh, uo, so = map(float, sys.argv[5:])
    else:
        uh, sh, uo, so = 10, 5, 90, 5

    # print schema
    outlier_box, super_box, schema,  pts = gen_points(npts, ndim, kdim, volperc, uh, sh, uo, so)
    print "\t".join(schema)
    for pt in pts:
        print '\t'.join(['%.4f'] * len(pt)) % tuple(pt)



