import sys
import random
import matplotlib
sys.path.extend( ['.', '..'])

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pymongo import *

from util import COLORS, LINESTYLES

matplotlib.use("Agg")





def plot_query(mdb, query, pp):
    points = list( mdb.find(query) )
    resultss = [ pt['results'] for pt in points ]
    configs = [ pt['config'] for pt in points]

    if not resultss:
        return

    ynames = ['nresults', 'dist_cost', 'total_cost', 'score_cost',
              'classify_cost', 'ncalls', 'ninputs', 'n_samp_calls']
    stats = ['accuracy', 'precision', 'recall', 'score']
    xname = 'nresults'
    xs = [ result[xname] for result in resultss[0] ]


    for yname in ynames:
        fig = plt.figure(figsize=(20, 10))
        sub = fig.add_subplot(111)

        for idx, (results, config) in enumerate(zip(resultss, configs)):
            # if 'separate' not in config['mode'] or '5' in config['klass'] or '4' in config['klass']:
            #     continue
            
            ys = [result[yname] for result in results]
            sub.plot(xs,
                     ys,
                     label='%s:%s:%s' % (config['mode'], config['width'], config['klass']),
                     c=COLORS[idx % len(COLORS)], ls=LINESTYLES[idx / len(COLORS)])

        sub.set_xlabel(xname)
        sub.set_ylabel(yname)
        sub.legend(loc='upper center', ncol=2)
        plt.setp(sub.get_legend().get_texts(), fontsize='9')
        plt.savefig(pp, format='pdf')

    for stat in stats:
        
        fig = plt.figure(figsize=(20, 10))
        sub = fig.add_subplot(111)

        for idx, (results, config) in enumerate(zip(resultss, configs)):
            # if 'separate' not in config['mode'] or '5' in config['klass'] or '4' in config['klass']:
            #     continue
            pts = []
            for xidx, (x, result) in enumerate(zip(xs, results)):
                for yidx, y in enumerate(result[stat]):
                    pts.append( (x + random.random() * 1.5, y + random.random() * 0.01) )
            _xs, _ys = zip(*pts)
            print len(pts), config
            sub.scatter(_xs,
                        _ys,
                        alpha=0.4,
                        s=60,
                        lw=0,
                        label='%s:%s:%s' % (config['mode'], config['width'], config['klass']),
                        c=COLORS[idx])
        sub.set_ylim(0, 1.5)
        sub.set_xlabel(xname)
        sub.set_ylabel(stat)
        sub.legend(loc='upper center', ncol=2)
        plt.setp(sub.get_legend().get_texts(), fontsize='9')
        plt.savefig(pp, format='pdf')



if __name__ == '__main__':
    import sys
    exp_id = int(sys.argv[1])
    
    pp = PdfPages('figs/endtoend.pdf')
    mconn = Connection()
    mdb = mconn.dbwipes.endtoend
    plot_query(mdb, {'type' : 'data', 'experiment_id' : exp_id}, pp)    
    pp.close()
