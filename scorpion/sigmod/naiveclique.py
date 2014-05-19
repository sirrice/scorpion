from ..util import *

from clique import MR
from naivegrouper import NaiveGrouper
from basic import Basic

class NaiveMR(MR):
    def setup_tables(self, full_table, bad_tables, good_tables, **kwargs):
        Basic.setup_tables(self, full_table, bad_tables, good_tables, **kwargs)
        self.grouper = NaiveGrouper(full_table, self) 

        self.SCORE_ID = add_meta_column(
                chain([full_table], bad_tables, good_tables),
                'SCOREVAR' 
        )



    def __call__(self, *args, **kwargs):
        self.naive = True

        return MR.__call__(self, *args, **kwargs)


