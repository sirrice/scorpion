import orange
import Orange
import math
from itertools import chain


def discrete_columns(table, cols=None):
    if cols is None:
        cols = [attr.name for attr in table.domain]

    ret = []
    for col in cols:
        attr = table.domain[col]
        if attr.varType == orange.VarTypes.Discrete:
            ret.append(col)
    return ret


def continuous_columns(table, cols=None):
    if cols is None:
        cols = [attr.name for attr in table.domain]

    ret = []
    for col in cols:
        attr = table.domain[col]
        if attr.varType != orange.VarTypes.Discrete:
            ret.append(col)
    return ret

        
def column_positions(table, cols):
    return [table.domain.index(table.domain[col]) for col in cols]


def valid_table_cols(table, cols, kwargs={}):
  bad_attrs = [
    'id', 'err', 'pickup_id', 'pickup_address', 'epoch', 'userid', 
    'mid', 'imdb', 'tstamp', "unknown", "action", "adventure", 
    "animation", "children", "comedy", "crime", "documentary", 
    "drama", "fantasy", "noir", "horro", "musical", "mystery", 
    "romance", "scifi", "thriller", "war", "western", 'lin_ima', 'com_nam',
    'total_charges', 'total_cost',  "totalcosts_ia", "totalcharges_ia",
    "estimatednetrevenue_ia", "estimated_net_revenue", "lengthofstay"
  ]
  ignore_attrs = kwargs.get('ignore_attrs', [])
  print "ignoring attrs: %s" % str(ignore_attrs)
  attrs = table.domain
  ret = []
  for attr in attrs:
    if attr.name in bad_attrs:
      continue
    if attr.name in cols:
      continue
    if attr.name in ignore_attrs:
      continue
    if attr.name.endswith('id') and attr.name != 'moteid':
      continue

    nunique = len(set([row[attr].value for row in table]))
    print "nunique %s\t%s" % (nunique, attr.name)

    if attr.varType != orange.VarTypes.Continuous:
      if nunique > 100 and nunique > 0.7 * len(table) or nunique > 7000:
        print "%s skipped" % attr.name
        continue
    ret.append(attr.name)
  return ret
 

def rm_attr_from_domain(table, attrs):
    """
    Remove attribute(s)/cols(s) from table
    Creates a copy of the original table :(
    """
    domain = table.domain
    try:
        erridxs = [domain.index(domain[attr]) for attr in attrs]
    except:
        erridxs = [domain.index(domain[attrs])]
    erridxs.sort(reverse=True)

    attrs = list(table.domain)
    map(attrs.pop, erridxs)
    
    newd = Orange.data.Domain(attrs)
    newd.add_metas(table.domain.get_metas())
    return Orange.data.Table(newd, table)
    # alternative way to create new table
    # return table.select_ref(newd)



def ids_filter(table, ids, attrname='id', negate=False):
    
    attr = table.domain[attrname]
    id_pos = table.domain.index(attr)
    vals = ids
    if not isinstance(ids[0], Orange.data.Value):
        vals = [Orange.data.Value(attr, id) for id in ids]
    if attr.var_type == Orange.feature.Type.Discrete:
        f = [ValueFilterDiscrete(position=id_pos, values=vals)]
    else:
        f = [ValueFilterContinuous(position=id_pos, ref=val, oper=ValueFilter.Equal) for val in vals]

    c = Orange.data.filter.Values(domain=table.domain,
                                  negate=negate,
                                  conjunction=0,
                                  conditions=f)
    def tmp(t, negate=negate):
        c.negate = negate
        return c(t)
    return tmp




def reconcile_tables(*all_tables):
  """
  Ensure bad and good tables use the same domain
  @return list of reconciled tables, union of all tables
  """
  domain = None
  full_table = None

  for table in chain(*all_tables):
    if domain is None:
      domain = list(table.domain.clone())
    else:
      for idx, (newattr, oldattr) in enumerate(zip(domain, table.domain)):
        if newattr.varType == orange.VarTypes.Discrete:
          if oldattr.varType == orange.VarTypes.Discrete:
            map(newattr.add_value, oldattr.values)
          else:
            domain[idx] = table.domain.clone()[idx]

  domain = Orange.data.Domain(domain)
  translated_tables = []

  for tables in all_tables:
    group = []
    translated_tables.append(group)
    
    for table in tables:
      #if not len(table):
      #continue
      
      if len(table):
        table = Orange.data.Table(domain)
      else:
        rows = [[v.value for v in row] for row in table]
        table = Orange.data.Table(domain, rows)

      group.append(table)
  
      if full_table is None:
        full_table = Orange.data.Table(domain, table)
      else:
        full_table.extend(table)
        

  if not full_table:
    return (), None

  return tuple(translated_tables), full_table

  translate = lambda t: Orange.data.Table(domain, t.to_numpyMA('ac')[0].data)
  translated_tables = []
  for tables in all_tables:#shared_domain_tables:
      translated_tables.append(map(translate, tables))

  return tuple(translated_tables), full_table


def union_tables(*tables):
  ret = None
  for t in chain(*tables):
    if ret is None:
      domain = t.domain.clone()
      ret = Orange.data.Table(domain)
    ret.extend(t)
  return ret

def add_meta_column(tables, name, vals=None, default=None):
    """
    @param name name of new attribute
    @param vals None if meta should be continuous, otherwise vals is a list of string values
           set to values of discrete attribute
    @param default default value.  
           if None, set to -inf for continuous attribute, or first element of vals for discrete
    """
    meta_id = Orange.feature.Descriptor.new_meta_id()
    for table in tables:
        if not vals:
            table.domain.addmeta(meta_id, Orange.feature.Continuous(name))
            table.add_meta_attribute(meta_id, default or -1e1000000)
        else:
            table.domain.addmeta(meta_id, Orange.feature.Discrete(name, values=vals))
            table.add_meta_attribute(meta_id, default or vals[0])
    return meta_id


