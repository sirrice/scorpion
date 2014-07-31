from scorpion.util import *

from scorpion.sigmod.clique import MR
from scorpion.sigmod.naivegrouper import NaiveGrouper
from scorpion.sigmod.basic import Basic

class NaiveMR(MR):
  """
  Version of MR that uses the NaiveGrouper
  """
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


