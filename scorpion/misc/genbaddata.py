"""
Generate dataset that is hard to greedy hill climb

If the search algorithm splits into 1/5s each time and then
the range x:1-20 y:1-20 should not affect avg, even though (2)
should eventually be detected.

Algorithm should also be confused because (1) is an obvious bad data
range

Data ranges
    x: 1 to 100
    y: 1 to 100
    z: 1 to 20
    val: -5 - 5

Bad data ranges
    1) x: 50 - 55, y: 50 - 55, val: 50
    2) x: 1 - 20, y: 1  - 10, val: 100
    3) x: 1 - 20, y: 10 - 20, val: -100


"""
import subprocess
import random
import sys
import os


if len(sys.argv) < 2:
    print "need argument of 1,2,3"
    exit()

harddata = int(sys.argv[1])
tablename = 'harddata_%d' % harddata


with file('/tmp/%s' % tablename, 'w') as f:

    ndim_cont = 10
    ndim_disc = 5
    # ndim_cont continuous dimensions.
    # ndim_disc discrete dimensions
    

    
    
    for z in xrange(20):
        for i in xrange(10000):
            x = random.random() * 100 
            y = random.random() * 100
            val = random.random() * 10 - 5

            if harddata == 1:
                if x >= 30 and x < 70 and y >= 30 and y <= 70:
                    if x >= 40 and x <= 60 and y >= 40 and y <= 60:
                        if z < 10:
                            val = 100 + random.random() * 10 - 5
                    else:
                        val = -100 + random.random() * 10 - 5

            if harddata == 2:        
                if (int(x / 20) + int(y / 20)) % 2:
                    val = 100 + random.random() * 10 - 5
                else:
                    val = -100 + random.random() * 10 - 5

            if harddata == 3:
                if z < 10:
                     if x >= 1 and x < 20:
                         if y >= 1 and y < 10:
                             val = 100 + random.random() * 10 - 5
                         elif y >= 7 and y < 20:
                             val = -(100. + random.random() * 10. - 5)
                     elif x >= 50 and x < 55 and y >= 50 and y < 55:
                         val = 50 + random.random() * 10 - 5

            
            print >> f, '%f,%f,%f,%f' % (x,y,z,val)


dbname = 'harddata'

try:
  os.system("psql -h localhost %s -c 'drop table %s'" % (dbname, tablename))
except Exception as e:
  print e

try:
  os.system("psql -h localhost %s -c 'create table %s (id serial, x float, y float, z float, v float)'" % (dbname, tablename))
except Exception as e:
  print e

os.system("psql -h localhost %s -c \"copy %s (x,y,z,v) from '/tmp/%s' with csv\"" % (dbname, tablename, tablename))
