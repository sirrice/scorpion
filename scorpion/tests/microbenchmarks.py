import sys
sys.path.extend( ['.', '..'])
from db import *
from score import *
from aggerror import *
from arch import *
from arch import get_provenance
from learners.cn2sd.refiner import *
import random
import time
import sys
import matplotlib
import timeit


COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def discretize(data, ignore_attrs=[]):

    data_discretized = False
    # If any of the attributes are continuous, discretize them
    if data.domain.hasContinuousAttributes():
        original_data = data
        data_discretized = True
        new_domain = []
        discretize = orange.EntropyDiscretization(forceAttribute=True)
        for attribute in data.domain.attributes:
            if ( attribute.varType == orange.VarTypes.Continuous and
                 attribute.name not in ignore_attrs ):
                d_attribute = discretize(attribute, data)
                # An attribute is irrelevant, if it is discretized into a single interval
                #if len(d_attribute.getValueFrom.transformer.points) > 0:

                # remove the D_ prefix because it's screwing things up!
                d_attribute.name = d_attribute.name[2:]  
                new_domain.append(d_attribute)
            else:
                new_domain.append(attribute)
        new_domain.append(original_data.domain.class_var)
        new_domain = Orange.data.Domain(new_domain)
        new_domain.add_metas(original_data.domain.get_metas())
        data = original_data.select(new_domain)
    return data_discretized, data



if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('figs/microbenchmarks.pdf')
    import matplotlib.pyplot as plt

    db = connect('intel')
    sql = '''select stddev(temp), avg(temp),
    ((extract(epoch from date+time - '2004-3-1'::timestamp)) / (30*60))::int as dist
    from readings
    where date > '2004-3-1'::timestamp and date < '2004-3-7'::timestamp
    group by dist order by dist;'''
    # drop table tmp;
    # create table tmp as select *
    # from readings
    # where date > '2004-3-1' and date < '2004-3-7' and
    # ((extract(epoch from date+time - '2004-3-1'::timestamp)) / (30*60))::int in (126,127,65, 66,67,68,74,75,76,122,121,124,123,125,128,129,130,131);


    allkeyids = [126,127,65, 66,67,68,74,75,76,122,121,124,123,125,128,129,130,131]

    def get_cost(table):
        _, table = discretize(table)

        refiner = BeamRefiner()
        base_rule = SDRule(table, '1')

        costs = []
        biggest_rule, biggest_n = None, None
        for new_rule in refiner(base_rule):
            # f = lambda: [x for x in query(db,'select count(*) from tmp where %s' % ' and '.join( rule_to_clauses(new_rule) ))][0]
            # n = f()
            # t = timeit.Timer(f)
            # t.timeit(10)
            # costs.append( (n, t.timeit(100) / 100.) )
            # print costs[-1]            
            # continue
            f = lambda: table.filter_ref(new_rule.filter)#new_rule.filter(table)
            n = len(f())            
            t = timeit.Timer(f)
            t.timeit(10)
            costs.append( (n, t.timeit(100) / 100.) )
            if not biggest_n or n > biggest_n:
                biggest_rule, biggest_n = new_rule, n
            print costs[-1]
        exit()

        for new_rule in refiner(biggest_rule):
            f = lambda: table.filter_ref(new_rule.filter)#new_rule.filter(table)
            n = len(f())            
            t = timeit.Timer(f)
            t.timeit(10)
            costs.append( (n, t.timeit(100) / 100.) )
            print costs[-1]            
            
        import numpy as np
        return costs, np.mean(costs), np.std(costs)



    fig = plt.figure()
    sub = fig.add_subplot(111)
        
    xs, means, stds = [],[],[]
    for cidx, i in enumerate(xrange(1, len(allkeyids)+1, 2)):
        aggerr = AggErr(SelectAgg('avg', 'avg', ['temp'], 'temp', None),
                        allkeyids[:i], 0, ErrTypes.TOOHIGH)
        obj = SharedObj(db, sql, errors=[aggerr])
        table = get_provenance(obj, aggerr)
        costs, mean, std = get_cost(table)
        
        xs.append(len(table))
        means.append(mean)
        stds.append(std)
        
        sub.scatter(*zip(*costs), label='%d points' % len(table), s=4, alpha=0.6, lw=0, c=COLORS[cidx])
    plt.savefig(pp, format='pdf')
        

    print xs
    print means
    print stds
    
    print "plotting"
    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.errorbar(xs, means, yerr=stds)
    sub.set_title('overall trend')
    plt.savefig(pp, format='pdf')

    pp.close()
    print "done"

