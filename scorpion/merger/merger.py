import math
import pdb
import random
import numpy as np
import sys
import time
sys.path.extend(['.', '..'])

from itertools import chain
from collections import defaultdict
from scipy.spatial import KDTree
from rtree.index import Index as RTree
from rtree.index import Property as RProp
from operator import mul, and_

from util import rm_attr_from_domain, get_logger
from util.table import *
from bottomup.bounding_box import *
from bottomup.cluster import *
from zero import Zero

_logger = get_logger()


class ShouldMerge(object):
    def __init__(self, **kwargs):
        self.msethreshold = kwargs.get('msethreshold', 0.001)
        self.areathreshold = kwargs.get('areathreshold', 0.05)
        self.absthreshold = kwargs.get('absthreshold', 2.3)

    def discretes_intersect(self, c1, c2):
        d1, d2 = c1.discretes, c2.discretes
        keys = set(chain(c1.discretes.keys(), c2.discretes.keys()))
        for key in keys:
            if key in d1 and key in d2:
                if not d1[key].intersection(d2[key]):
                    return False
        return True
        

    def __call__(self, newcluster, intersecting, min_volume=0.00001, err_scale=1., err_thresh=1e1000):
        """
        """
        if not newcluster:
            return False

        if reduce(and_, [newcluster.error > c.error for c in intersecting]):
            return True



        # Weighted MSE
        errors = []
        squared_errors = []
        volumes = []
        for inter in intersecting:
            if not self.discretes_intersect(newcluster, inter):
                continue
            v = volume(intersection_box(inter.bbox, newcluster.bbox))
            if not v:
                v = min_volume
            if newcluster.error > inter.error:
                continue
            serr = v * math.pow(inter.error - newcluster.error, 2)
            errors.append(inter.error)
            squared_errors.append(serr)
            volumes.append(v)
        wmse = sum(squared_errors) / sum(volumes)
        we = sum(errors) / sum(volumes)



        threshold = self.msethreshold * (err_scale**2)
        print "merge?", newcluster, wmse, threshold

        if wmse >= threshold:
            return False

        p0, p1 = newcluster.parents[0], newcluster.parents[1]
        different = False
        for k in chain(p0.discretes.keys(), p1.discretes.keys()):
            if p0.discretes.get(k, ()) != p1.discretes.get(k, ()):
                different = True
                break

        if different:
            prev_vol = p0.volume + p1.volume - volume(intersection_box(p0.bbox, p1.bbox))
            if newcluster.volume == 0:
                return False
            if (newcluster.volume - prev_vol) / newcluster.volume > self.areathreshold:
                return False

        return True
        
        



class Merger(object):
    """
    Merges clusters

    transparently scales
    - cluster bounding boxes by table bounds
    - errors by min/max error
    """

    def __init__(self, **kwargs):
        self.min_clusters = 1
        self.should_merge = ShouldMerge(**kwargs)
        self.is_mergable = lambda c: c.error > 0
        
        self.densitythreshold = 0.05
        self.absthreshold = 0.8
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        self.min_clusters = kwargs.get('min_clusters', self.min_clusters)
        self.should_merge = kwargs.get('should_merge', self.should_merge)
        self.is_mergable = kwargs.get('is_mergable', self.is_mergable)
        

    def setup_stats(self, clusters):
        """
        computes error bounds and the minimum volume of a 0-volume cluster

        """
        vols = np.array([c.volume for c in clusters if c.volume > 0])
        if len(vols) == 0:
            self.point_volume = 0.00001
        else:
            self.point_volume = vols.min() / 2.

        self.setup_errors(clusters)

        # setup table bounds to that calls to kdtree and rtree are zeroed
        # this should be transparent from the caller
        self.cont_cols = continuous_columns(self.table, self.cols)
        self.cont_pos = column_positions(self.table, self.cont_cols)
        self.search_data = self.table.to_numpyMA('ac')[0].data[:, self.cont_pos]
        self.search_bbox = points_bounding_box(self.search_data)

        self.zero = Zero(range(len(self.cont_pos)), bounds=self.search_bbox)

    def setup_errors(self, clusters):
        errors = np.array([c.error for c in clusters])
        self.mean_error = np.mean(errors)
        self.std_error = np.std(errors)
        self.min_error = errors.min()
        self.max_error = errors.max()
        self.diff_error = (self.max_error - self.min_error) or 1.
        

    def scale_error(self, error):
        return (error - self.min_error) / self.diff_error

    def scale_point(self, point):
        return self.zero.zero(np.array(point)).tolist()

    def scale_box(self, box):
        return self.zero.zero(np.array(box)).tolist()

    def normalize_cluster_errors(self, clusters):
        if not self.diff_error:
            return

        for c in clusters:
            c.error = (c.error - self.min_error) / self.diff_error
        return clusters

    def unnormalize_cluster_errors(self, clusters):
        if not self.diff_error:
            return

        for c in clusters:
            c.error = c.error * self.diff_error + self.min_error

    def filter_discrete(self, c1, c2, intersecting_clusters):
        return filter(c1.discretes_intersect,
                      filter(c2.discretes_intersect, intersecting_clusters))

    def construct_rtree(self, clusters):
        ndim = max(2, len(clusters[0].centroid))
        p = RProp()
        p.dimension = ndim
        p.dat_extension = 'data'
        p.idx_extension = 'index'

        rtree = RTree(properties=p)
        for idx, c in enumerate(clusters):
            if len(c.centroid) == 0:
                rtree.insert(idx, [0, 0, 1, 1])
            else:
                centroid = self.scale_point(c.centroid)
                box = self.scale_box(c.bbox)
                if ndim == 1:
                    #rtree.insert(idx, box[0] + [0] + box[1] + [1])
                    rtree.insert(idx, centroid + [0] + centroid[1])
                else:
                    #rtree.insert(idx, box[0] + box[1])
                    rtree.insert(idx, centroid + centroid)
        return rtree

    def get_intersection(self, rtree, bbox):
        if len(bbox[0]) == 0:
            return rtree.intersection([0, 0, 1, 1])

        bbox = self.scale_box(bbox)
        if len(bbox[0]) == 1:
            return rtree.intersection(bbox[0] + [0] + bbox[1] + [1])
        return rtree.intersection(bbox[0] + bbox[1])        




    def construct_kdtree(self, clusters):
        if len(clusters[0].centroid) == 0:
            #kdtree = KDTree([(self.scale_error(c.error), 0, 0) for c in clusters])
            kdtree = KDTree([( 0, 0) for c in clusters])
        else:
            #kdtree = KDTree([[self.scale_error(c.error)] + self.scale_point(c.centroid) for c in clusters])
            kdtree = KDTree([self.scale_point(c.centroid) for c in clusters])
        return kdtree


    def get_nn(self, cluster, kdtree, clusters):
        if len(kdtree.data) == 1:
            return None

        centroid = [0,0] if not len(cluster.centroid) else cluster.centroid
        #centroid = [self.scale_error(cluster.error)] + self.scale_point(centroid)
        centroid = self.scale_point(centroid)
        

        k, prev_k = min(3, len(kdtree.data)), 0
        ret = None
        while k <= len(kdtree.data):
            results = kdtree.query(centroid, k=k)
            bestidx = None
            for dist, idx in zip(*results)[prev_k:]:
                if idx == len(clusters):
                    return None
                if clusters[idx] == cluster:
                    continue

                ret = clusters[idx]
                if cluster.bbox == ret.bbox and str(cluster.discretes) == str(ret.discretes):
                    return ret
                    continue
                return ret
            prev_k = k
            k += 10

            _logger.debug("increasing by 10 to %d", k)

        return ret

    def merge(self, cluster, neighbor, intersecting_clusters):
        merged = Cluster.merge(cluster, neighbor, intersecting_clusters, self.point_volume)
        return merged

        
    def __call__(self, clusters, **kwargs):
        if not clusters:
            return list(clusters)


        self.set_params(**kwargs)
        self.setup_stats(clusters)
        clusters_set = set(clusters)


        while len(clusters_set) > self.min_clusters:
            self.setup_errors(clusters_set)

            clusters = sorted(clusters_set, key=lambda c: c.error, reverse=True)
            mergable_clusters = filter(self.is_mergable, clusters)
            if not mergable_clusters:
                break


            kdtree = self.construct_kdtree(mergable_clusters)
            rtree = self.construct_rtree(clusters)

            merged_clusters, new_clusters = set(), set()
            seen = set()

            for cluster in mergable_clusters:
                if cluster in merged_clusters or cluster in new_clusters or cluster in seen:
                    continue
                
                neighbor = self.get_nn(cluster, kdtree, mergable_clusters)
                if not neighbor or neighbor in seen:
                    continue

                seen.update((cluster, neighbor))
                
                newbbox = bounding_box(cluster.bbox, neighbor.bbox)
                cidxs = self.get_intersection(rtree, newbbox)
                intersecting_clusters = [clusters[cidx] for cidx in cidxs]
                intersecting_clusters = [cluster, neighbor]
                merged = self.merge(cluster, neighbor, intersecting_clusters)
                if not merged or merged in clusters or merged in new_clusters:
                    continue
                
                if self.should_merge(merged, intersecting_clusters, self.point_volume, self.diff_error, self.mean_error + self.std_error):
                    new_clusters.add(merged)                    
                    merged_clusters.update((cluster, neighbor))
                    for c in intersecting_clusters:
                        if box_contained(c.bbox, merged.bbox):
                            merged_clusters.add(c)

            _logger.debug("merged %d\t%d new clusters\tout of %d\t%s",
                          len(merged_clusters),
                          len(new_clusters),
                          len(mergable_clusters),
                          str(new_clusters and list(new_clusters)[0] or None))

            
            if not new_clusters:
                break

            clusters_set.difference_update(merged_clusters)
            clusters_set.update(new_clusters)

        #self.unnormalize_cluster_errors(clusters_set)
        return list(clusters_set)
            
    
