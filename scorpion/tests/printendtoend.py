from pymongo import *
import sys
sys.path.extend( ['.', '..'])
from db import *
from score import *
from aggerror import *
from arch import *
from datetime import datetime
import pickle
import time
from learners.cn2sd.evaluator import *

mconn = Connection()
mdb = mconn.dbwipes.endtoend

try:
    exp_id = int(sys.argv[1])
except:
    print "need experiment id as parameter"
    print "max experiment id:", mdb.find_one({'type' : 'global_id'})['global_id']
    for data in mdb.find( {'type' : 'data'}, {'start_time':1, 'notes' : 1, 'experiment_id' : 1} ):
        print data
    exit()


header = '\t'.join(['',
                 'nres',
                 'discre',
                 'totcost',
                 'ncalls ',
                 'Precis',
                 'Recall',
                 'Accur '])

for row in mdb.find({'type' : 'data', 'experiment_id' : exp_id}):
    config = row['config']
    print '%s\t%s\t%s' % ( config['mode'], config['width'], config['klass'])
    print header
    for result in row['results']:
        for rule, meta in result['rules']:
            params = (result['nresults'],
                      result['config'].get('bdiscretize', None),
                      result['total_cost'],
                      result['ncalls'],
                      meta['precision'],
                      meta['recall'],
                      meta['accuracy'])
            params = '%d\t%s\t%.3f\t%d\t%.3f\t%.3f\t%.3f' % params
            print "\t%s\t%s" % (params, ' or '.join(rule))
        
