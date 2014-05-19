import time
import pdb
import sys
import Orange
import orange
import heapq
sys.path.extend(['.', '..'])

from itertools import chain


from ..learners.cn2sd.rule import fill_in_rules
from ..learners.cn2sd.refiner import *
from ..bottomup.bounding_box import *
from ..bottomup.cluster import *
from ..util import *

from basic import Basic
from merger import Merger
from grouper import Grouper, Blah

_logger = get_logger()


class NaiveGrouper(Grouper):

    def __call__(self, attrs, valid_groups):
        """ every group is valid!!!"""

        start = time.time()
        bad_table_rows = []
        good_table_rows = []
        for table in self.mr.bad_tables:
            bad_table_rows.append(self.table_influence(attrs, table))
        for table in self.mr.good_tables:
            good_table_rows.append(self.table_influence(attrs, table))
        #print "scan time\t", (time.time() - start)

        def get_infs(all_table_rows, err_funcs, g, bmaxinf):
            ret = []
            counts = []
            maxinf = None
            for idx, ef, table_rows in zip(range(len(err_funcs)), err_funcs, all_table_rows):
                rows = table_rows.get(g, [])
                if rows:
                    if bmaxinf:
                        cur_maxinf = self.influence_tuple(max(rows, key=lambda row: self.influence_tuple(row, ef)), ef)
                        if not maxinf or cur_maxinf > maxinf:
                            maxinf = cur_maxinf

                    ret.append(ef(rows))
                    counts.append(len(rows))
            return ret, counts, maxinf


        start = time.time()
        for g in self.keys_iter(attrs):
            bds, bcs, maxinf = get_infs(bad_table_rows, self.mr.bad_err_funcs, g, True)
            gds, gcs, _ = get_infs(good_table_rows, self.mr.good_err_funcs, g, False)
            if not bcs:
                continue
            yield Blah(attrs, g, bds, bcs, gds, gcs, maxinf, self.mr, self)
        #print "comp infs\t", (time.time() - start)


    def influence_tuple(self, row, ef):
        if row[self.mr.SCORE_ID].value == -1e10000000000:
            influence = ef((row,))
            row[self.mr.SCORE_ID] = influence
        return row[self.mr.SCORE_ID].value


    def keys_iter(self, attrs):
        seen = set()
        for row in self.data:
            group = tuple([self.gbfuncs[attr](row[attr]) for attr in attrs])
            if group in seen:
                continue
            seen.add(group)
            yield group 


    def table_influence(self, attrs, table):
        """scan table once and group tuples by their respective groups"""
        groups = defaultdict(list)
        for row in table:
            group = tuple([self.gbfuncs[attr](row[attr]) for attr in attrs])
            groups[group].append(row)
        return groups


    def merge_groups(self, prev_groups):
        """
        prev_groups: attributes -> groups
        attributes are sorted
        group: attr -> idx
        """
        start = time.time()
        attrs_list = prev_groups.keys()
        for a_idx, attrs1 in enumerate(attrs_list):
            sattrs1 = set(attrs1)
            for attrs2 in attrs_list[a_idx+1:]:
                pgroup1, pgroup2 = prev_groups[attrs1], prev_groups[attrs2]

                sattrs2 = set(attrs2)
                merged_attrs = tuple(sorted(sattrs1.union(sattrs2)))
                if len(merged_attrs) != len(sattrs1)+1:
                    continue
                yield (merged_attrs, [])
            

