from testtopdown import *


def print_clusters(sub, clusters, tuples=[], title=''):
    clusters.sort(key=lambda c: c.error)

    for cluster in clusters:
        x, y = tuple(map(list, zip(*cluster.bbox)))
        x[0] = max(0, x[0])
        x[1] = min(100, x[1])
        y[0] = max(0, y[0])
        y[1] = min(100, y[1])
        c = cm.jet(cluster.error)
        c = 'black'#cm.get_cmap('Greys_r')(cluster.error)
        r = Rect((x[0], y[0]), x[1]-x[0], y[1]-y[0], alpha=max(0.1,cluster.error), ec=c, fill=False, lw=1.5)
        sub.add_patch(r)

    if tuples:
        xs, ys, cs = zip(*tuples)
        cs = map(lambda c: cm.get_cmap('Greys')(c+0.05), cs)
        sub.scatter(ys, xs, c=cs, alpha=0.5, lw=0)

    [i.set_linewidth(0) for name, i in sub.spines.iteritems() if name in ('top', 'right')]
    font = {'family' : 'Times New Roman',
#                'weight' : 'bold',
                        'size'   : 25}
    matplotlib.rc('font', **font)
    



if __name__ == '__main__':
    sigmoddb = create_engine('postgresql://localhost/sigmod')
    tuples = get_tuples_in_bounds(sigmoddb, 'data_2_80', [(0,100),(0,100)], 'g=7')
    cols = zip(*tuples)
    tuples = zip(cols[2], cols[1], [v / 100. for v in cols[-1]])

    bounds = [(0 , ((66.67, 73.34), (66.66, 73.33))),
        (0.05 , ((66.67, 73.34),(60.00, 66.66))),
        (0.1 , ((46.68, 53.34),(79.99, 86.66))),
        (0.2 , ( (80.00, 86.67),(60.00, 66.66) )),
        (0.5 , ((86.67, 93.33),(60.00, 66.66) ))]
    bounds = json.loads('[[0.0, "high", [[13.35, 6.67], [93.33, 93.33]]], [0.5, "high", [[66.67, 46.66], [73.34, 53.33]]], [0.2, "high", [[60.01, 46.66], [80.0, 60.0]]], [0.1, "high", [[53.34, 40.0], [80.0, 73.33]]], [0.05, "high", [[46.68, 40.0], [93.33, 86.66]]], [0.0, "mid", [[13.35, 6.67], [93.33, 93.33]]], [0.5, "mid", [[66.67, 46.66], [73.34, 53.33]]], [0.2, "mid", [[60.01, 46.66], [80.0, 60.0]]], [0.1, "mid", [[53.34, 40.0], [80.0, 73.33]]], [0.05, "mid", [[46.68, 40.0], [93.33, 86.66]]]]')
    bounds = [b for b in bounds if b[1] == 'mid']
    bounds.sort()

    pp = PdfPages('figs/topdown_naive_results.pdf')
#    pp = PdfPages('/tmp/foo.pdf')
    fig = plt.figure(figsize=(25, 4))
    for idx, (c, boundtype, bound) in enumerate(bounds):
        print c, bound
        sub = fig.add_subplot('1%d%d' % (len(bounds), idx+1))
        print_clusters(sub, [Cluster(bound, 1., [])], tuples=tuples)
        sub.set_ylim(-5, 105)
        sub.set_xlim(-5, 105)
        sub.set_title("C = %.2f" % (c))
        sub.set_xlabel('Attribute 1')
        if idx == 0:
            sub.set_ylabel('Attribute 2')
    plt.savefig(pp, format='pdf', bbox_inches='tight')

    pp.close()
    exit()


