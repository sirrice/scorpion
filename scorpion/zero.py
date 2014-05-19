import numpy as np

from bottomup.cluster import Cluster

class Zero(object):
    """
    Given data bounds, normalizes and zeros data to range between [0, 1]
    """
    def __init__(self, cols, bounds=None):
        self.normalize_bounds = None
        self.normalize_ranges = None
        self.cols = cols

        assert self.cols is not None, "column cannot be set to None"

        self.set_bounds(bounds)

    def set_bounds(self, bounds):
        if bounds is not None:
            self.normalize_bounds = np.array(bounds)
            self.normalize_ranges = self.normalize_bounds[1] - self.normalize_bounds[0]
            nonzeroidxs = np.nonzero(self.normalize_ranges==0)[0]
            self.normalize_ranges[nonzeroidxs] = 1.


    @staticmethod
    def compute_bounds(data):
        maxs = np.amax(data, axis=0)
        mins = np.amin(data, axis=0)
        return (mins, maxs)
        

    def zero(self, data):
        assert self.cols is not None, "columns not defined"
        assert self.normalize_bounds is not None, "bounds not set"
        
        if not len(self.cols):
            self.normalize_bounds = (0, 0)
            return data

        data = data.copy()

        if len(data.shape) == 1:
            data[self.cols] = (data[self.cols] - self.normalize_bounds[0]) / self.normalize_ranges
        else:
            subset = data[:, self.cols]
            data[:,self.cols] = (subset - self.normalize_bounds[0]) / self.normalize_ranges

        return data


    def unzero(self, data):
        assert self.cols is not None, "columns not defined"
        assert self.normalize_bounds is not None, "bounds not set"

        if not len(self.cols):
            return data
        
        data = data.copy()
        mins, maxs = self.normalize_bounds

        if len(data.shape) == 1:
            data[self.cols] = (data[self.cols] * self.normalize_ranges) + mins
        else:
            data[:, self.cols] = (data[:, self.cols] *  self.normalize_ranges) + mins

        return data

    def unzero_cluster(self, cluster):
        assert self.cols is not None, "columns not defined"
        assert self.normalize_bounds is not None, "bounds not set"

        if not len(self.cols):
            return cluster
        
        bbox = cluster.bbox

        mins, maxs = self.normalize_bounds
        ranges = maxs - mins
        bbox = (tuple((np.array(bbox[0]) * self.normalize_ranges) + mins),
                tuple((np.array(bbox[1]) * self.normalize_ranges) + mins))
        return Cluster(bbox,
                       cluster.error,
                       cluster.cols,
                       parents=cluster.parents,
                       discretes=cluster.discretes,
                       **cluster.kwargs)
