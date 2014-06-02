from operator import mul, and_


def points_bounding_box(points):
    if not len(points):
        return ((0,), (0,))
    return (tuple(points.min(0)), tuple(points.max(0)))


def bounding_box(bbox1, bbox2):
    mins = tuple([min(min1, min2) for min1, min2 in zip(bbox1[0], bbox2[0])])
    maxs = tuple([max(max1, max2) for max1, max2 in zip(bbox1[1], bbox2[1])])
    return (mins, maxs)

def intersection_box(bbox1, bbox2):
    mins = map(max, zip(bbox1[0], bbox2[0]))
    maxs = map(min, zip(bbox1[1], bbox2[1]))
    return (mins, maxs)

def box_contained(box, bound, epsilon=0.):
    if not len(zip(*box)) and not len(zip(*bound)):
      return True
    if len(zip(*box)) != len(zip(*bound)):
      print >>sys.stderr, "box_contained: bounding boxes have different dimensions: %d vs %d" % (len(box[0]), len(bound[0]))
      return False
    inter = intersection_box(box, bound)
    idiffs = [p[1]-p[0] for p in zip(*inter)]
    bdiffs = [p[1]-p[0] for p in zip(*box)]
    return reduce(and_, map(lambda (i,b): i >= (1.-epsilon) * b, zip(idiffs, bdiffs)))

    return (reduce(and_, (min1 >= min2 for min1, min2 in zip(box[0], bound[0]))) and
            reduce(and_, (max1 <= max2 for max1, max2 in zip(box[1], bound[1]))) )

def box_completely_contained(box, bound, epsilon=0.):
    if not len(zip(*box)) or not len(zip(*bound)):
        return False
    for inner, outer in zip(zip(*box), zip(*bound)):
        if inner[0] <= outer[0] or inner[1] >= outer[1]:
            return False
        if (outer[1]-outer[0]) - (inner[1]-inner[0]) <= epsilon:
            return False
    return True


def box_same(box1, box2, epsilon=0.):
    ivol = volume(intersection_box(box1, box2))
    f = lambda box: ivol >= (1. - epsilon) * volume(box)
    return reduce(and_, map(f, (box1, box2)))

def volume(bbox):
    if not len(bbox[0]): return 0.
    diffs = [maxv - minv for minv, maxv in zip(*bbox)]
    if min(diffs) < 0: return 0
    return reduce(mul, diffs) 

