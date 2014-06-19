import sys
sys.path.extend(['.', '..'])

from scorpion.arch import *
from scorpionsql.aggerror import ErrTypes


class DatasetNames(object):
    def __init__(self):
         self.datasetnames = ['intel_noon',
                'intel_corr',
                'intel_mote18',
                'intel_first_spike',
                'intel_mass_failures',
                'intel_low_voltage',
                'harddata1_avg',
                'harddata1_std',
                'harddata1_sum',
                'harddata1_skewed',
                'harddata2_avg',
                'harddata3_avg',                
                'fec12_obama',
                'intel_foo',
                'fec12_donation',
                'fec12_donation2',
                'harddata4_sum',
                'data_2_2_1000_0d25',
                'data_2_2_1000_0d1',
                'lqm' # 19
                ]

    def __getitem__(self, key):
        try:
            return self.datasetnames[int(key)]
        except:
            return key
datasetnames = DatasetNames()

# sigmod_<ndim>_<kdim>_<npts/group>_<volume>[_uh_sh_uo_so]




def get_test_data(name, **kwargs):

    if name.startswith('data_') or name.startswith('data2c'):
        testdata = get_sigmod_data(name)
    else:
        cmd = 'testdata = get_%s()' % name
        exec(cmd)

    if len(testdata) != 6:
      raise RuntimeException("expected 6 results")

    sql, badresults, goodresults, get_ground_truth, errtype, tablename = tuple(testdata)

    if 'intel' in name:
        dbname = 'intel'
    elif 'harddata' in name:
        dbname = 'harddata'
    elif 'fec12' in name:
        dbname = 'fec12'
    elif 'lqm' in name:
        dbname = 'med'
    elif 'fec' in name:
        dbname = 'fec'
    elif name.startswith('data_') or name.startswith('data2c'):
        dbname = 'sigmod'
    else:
        raise RuntimeException("i don't understand this test data: %s", name)

    nbadresults = kwargs.get('nbadresults', len(badresults))
    badresults = badresults[:nbadresults]

    return dbname, sql, badresults, goodresults, errtype, get_ground_truth, tablename



def get_intel_noon():
    """
    Before motes start to fail.
    """
    sql = """SELECT stddev(temp),
    ((extract(epoch from date+time - '2004-3-1'::timestamp)) / (30*60)) :: int as dist
    FROM readings
    WHERE date+time > '2004-3-1'::timestamp and date+time < '2004-3-10'::timestamp
    GROUP BY dist ORDER BY dist"""
    badresults = [64, 72, 126, 127, 65, 66, 67, 68, 74, 75, 76, 122, 121, 124, 123, 125, 128, 129, 130, 131]
    goodresults = [0, 1, 9, 10, 2, 3, 4, 5, 6, 7, 8, 11, 12]

    def get_ground_truth(table):
        return [row['id'].value for row in table if row['temp'].value > 50]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'readings'

def get_intel_corr():
    sql = """SELECT corr(temp, voltage), ((extract(epoch from date+time - '2004-3-1'::timestamp)) / (30*60)) :: int as dist FROM readings WHERE date+time > '2004-3-1'::timestamp and date+time < '2004-3-7'::timestamp and moteid < 100 GROUP BY dist ORDER BY dist"""

    badresults = [63, 64, 65, 66, 67, 74, 75, 121, 123, 126, 127, 128, 129, 130, 131, 137, 138, 209]
    goodresults = [0, 1, 9, 10, 2, 3, 4, 5, 6, 7, 8, 11, 12]

    def get_ground_truth(table):
        return [row['id'].value for row in table if int(row['moteid']) > 100 or row['temp'] >= 60]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOLOW, 'readings'

    
def get_intel_mote18():
    sql = """SELECT stddev(temp),
    ((extract(epoch from date+time - '2004-3-1'::timestamp)) / (30*60)) :: int as dist
    FROM readings
    WHERE date+time > '2004-3-8'::timestamp and date+time < '2004-3-15'::timestamp
    GROUP BY dist ORDER BY dist"""
    badresults = [378, 379, 387, 388, 397, 396, 406, 405, 414, 415, 423,
                  424, 432, 433, 441, 442, 450, 380, 381, 382, 383, 384,
                  385, 386, 389, 390, 391, 392, 393, 394, 395, 399, 398,
                  400, 401, 402, 403, 404, 407, 408, 409, 410, 411, 412,
                  413, 416, 417, 419, 418, 421, 420, 422, 425, 426, 427,
                  428, 430, 429, 431, 434, 435, 436, 437, 438, 439, 440,
                  444, 443, 445, 446, 448, 447, 449] 
    goodresults = [333, 334, 342, 343, 351, 352, 335, 336, 337, 338, 339,
                   340, 341, 344, 345, 346, 347, 348, 349, 350]
    def get_ground_truth(table):
        return [row['id'].value for row in table if row['temp'].value > 50]#str(row['moteid'].value) == '18']
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'readings'

def get_intel_first_spike():
    sql = """SELECT stddev(temp),
    ((extract(epoch from date+time - '2004-3-1'::timestamp)) / (30*60)) :: int as dist
    FROM readings
    WHERE date+time > '2004-3-8'::timestamp and date+time < '2004-3-20'::timestamp
    GROUP BY dist ORDER BY dist"""

    badresults = [505, 507, 504, 506, 522, 523, 524, 525, 542, 543,
                  540, 541, 558, 559, 560, 561, 576, 577, 578, 579, 594, 597, 595,
                  596, 612, 613, 614, 615, 633, 630, 631, 632, 648, 650, 649, 651,
                  666, 667, 668, 669, 684, 685, 686, 687, 702, 703, 704, 499, 500,
                  501, 503, 502, 510, 508, 509, 511, 513, 512, 514, 515, 516, 517,
                  518, 519, 520, 521, 526, 529, 527, 528, 530, 531, 532, 534, 533,
                  535, 536, 538, 537, 539, 546, 544, 545, 547, 550, 548, 549, 551,
                  552, 553, 554, 555, 556, 557, 564, 565, 562, 563, 566, 567, 568,
                  569, 570, 571, 572, 573, 574, 575, 580, 581, 583, 582, 584, 585,
                  586, 588, 589, 590, 587, 593, 591, 592, 599, 600, 598, 601, 602,
                  603, 604, 605, 606, 607, 608, 609, 610, 611, 616, 618, 619, 617,
                  620, 621, 622, 624, 625, 626, 623, 629, 628, 627, 634, 636, 637,
                  635, 638, 639, 640, 642, 641, 643, 644, 645, 647, 646, 653, 652,
                  654, 655, 658, 656, 657, 659, 661, 662, 660, 663, 664, 665, 670,
                  671, 672, 673, 674, 675, 676, 677, 679, 680, 678, 681, 682, 683,
                  688, 689, 690, 691, 693, 694, 692, 695, 698, 696, 697, 701, 699,
                  700]
    goodresults = [333, 334, 342, 343, 351, 352, 335, 336, 337, 338, 339,
                   340, 341, 344, 345, 346, 347, 348, 349, 350]

    def get_ground_truth(table):
        return [row['id'].value for row in table if row['temp'].value > 50]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'readings'
    
def get_intel_mass_failures():
    sql = """SELECT stddev(temp), ((extract(epoch from date+time - '2004-3-1'::timestamp)) / (30*60)) :: int as dist
    FROM readings
    WHERE date+time > '2004-3-15'::timestamp and date+time < '2004-4-1'::timestamp
    GROUP BY dist ORDER BY dist"""
    badresults = [842, 845, 843, 844, 846, 848, 847, 849, 850, 851,
                  853, 852, 854, 855, 856, 857, 858, 859, 860, 861, 865, 866, 862,
                  863, 864, 867, 870, 868, 869, 871, 872, 873, 874, 875, 876, 877,
                  882, 880, 879, 881, 878, 885, 883, 884, 887, 886, 889, 888, 890,
                  892, 891, 893, 894, 895, 896, 897, 901, 902, 898, 899, 900, 905,
                  903, 904, 906, 907, 910, 909, 908, 911, 912, 916, 913, 914, 915,
                  917, 919, 920, 921, 923, 918, 922, 926, 927, 924, 925, 928, 930,
                  933, 929, 931, 932, 937, 938, 934, 935, 936, 939, 940, 941, 942,
                  943, 944, 945, 946, 948, 947, 949, 952, 950, 951, 953, 954, 955,
                  956, 958, 957, 959, 960, 961, 962, 963, 968, 966, 967, 965, 964,
                  969, 970, 971, 972, 974, 973, 975, 976, 977, 978, 979]
    goodresults = [684, 685, 686, 687, 688, 693, 689, 690, 691, 692,
                   694, 695, 698, 696, 697, 701, 702, 699, 700, 703, 704]


    sql = """SELECT stddev(temp), date_trunc('hour', date+time) as dt
    FROM readings
    WHERE date + time > '2004-3-8' and date + time < '2004-3-15'
    GROUP BY dt ORDER BY dt"""

    badresults =  [datetime(2004, 3, 9, 6, 0), datetime(2004, 3, 9, 5, 0), datetime(2004, 3, 9, 7, 0), datetime(2004, 3, 9, 8, 0), datetime(2004, 3, 9, 9, 0), datetime(2004, 3, 9, 10, 0), datetime(2004, 3, 9, 11, 0), datetime(2004, 3, 9, 12, 0), datetime(2004, 3, 9, 13, 0), datetime(2004, 3, 9, 14, 0), datetime(2004, 3, 9, 15, 0), datetime(2004, 3, 9, 16, 0), datetime(2004, 3, 9, 17, 0), datetime(2004, 3, 9, 18, 0), datetime(2004, 3, 9, 19, 0), datetime(2004, 3, 9, 21, 0), datetime(2004, 3, 9, 20, 0), datetime(2004, 3, 9, 22, 0), datetime(2004, 3, 9, 23, 0), datetime(2004, 3, 10, 0, 0), datetime(2004, 3, 10, 1, 0), datetime(2004, 3, 10, 2, 0), datetime(2004, 3, 10, 3, 0), datetime(2004, 3, 11, 13, 0), datetime(2004, 3, 11, 14, 0), datetime(2004, 3, 11, 15, 0), datetime(2004, 3, 11, 16, 0), datetime(2004, 3, 11, 17, 0), datetime(2004, 3, 11, 18, 0), datetime(2004, 3, 11, 19, 0), datetime(2004, 3, 11, 20, 0), datetime(2004, 3, 11, 22, 0), datetime(2004, 3, 11, 21, 0), datetime(2004, 3, 11, 23, 0), datetime(2004, 3, 12, 0, 0), datetime(2004, 3, 12, 2, 0), datetime(2004, 3, 12, 1, 0), datetime(2004, 3, 12, 3, 0), datetime(2004, 3, 12, 4, 0), datetime(2004, 3, 12, 6, 0), datetime(2004, 3, 12, 5, 0), datetime(2004, 3, 12, 7, 0), datetime(2004, 3, 12, 8, 0), datetime(2004, 3, 12, 9, 0), datetime(2004, 3, 12, 11, 0), datetime(2004, 3, 12, 10, 0)]
   
    goodresults = []
    def get_ground_truth(table):
        return [row['id'].value for row in table if row['temp'].value > 50] #< 2.4]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'readings'

def get_intel_low_voltage():
    sql = """SELECT avg(temp), stddev(temp), date_trunc('hour',(date) + (time)) as dt
    FROM readings
    WHERE (((date) + (time)) > '2004-3-10') and (((date) + (time)) < '2004-3-25')
    GROUP BY dt
    ORDER BY dt ASC"""
    badresults = map(parse, [u'2004-03-23T03:00:00.000Z', u'2004-03-23T04:00:00.000Z', u'2004-03-23T05:00:00.000Z', u'2004-03-23T06:00:00.000Z', u'2004-03-23T07:00:00.000Z', u'2004-03-23T08:00:00.000Z', u'2004-03-23T09:00:00.000Z', u'2004-03-23T10:00:00.000Z', u'2004-03-23T11:00:00.000Z', u'2004-03-23T12:00:00.000Z', u'2004-03-23T13:00:00.000Z', u'2004-03-23T14:00:00.000Z', u'2004-03-23T15:00:00.000Z', u'2004-03-23T16:00:00.000Z', u'2004-03-23T17:00:00.000Z', u'2004-03-23T18:00:00.000Z', u'2004-03-23T19:00:00.000Z', u'2004-03-23T20:00:00.000Z', u'2004-03-23T21:00:00.000Z', u'2004-03-23T22:00:00.000Z', u'2004-03-23T23:00:00.000Z', u'2004-03-24T00:00:00.000Z', u'2004-03-24T01:00:00.000Z', u'2004-03-24T02:00:00.000Z', u'2004-03-24T03:00:00.000Z', u'2004-03-24T04:00:00.000Z', u'2004-03-24T05:00:00.000Z', u'2004-03-24T06:00:00.000Z', u'2004-03-24T07:00:00.000Z', u'2004-03-24T08:00:00.000Z', u'2004-03-24T09:00:00.000Z', u'2004-03-24T10:00:00.000Z'])

    good_results = map(parse, [u'2004-03-14T21:00:00.000Z', u'2004-03-14T22:00:00.000Z', u'2004-03-14T23:00:00.000Z', u'2004-03-15T00:00:00.000Z', u'2004-03-15T01:00:00.000Z', u'2004-03-15T02:00:00.000Z', u'2004-03-15T03:00:00.000Z', u'2004-03-15T04:00:00.000Z', u'2004-03-15T05:00:00.000Z', u'2004-03-15T06:00:00.000Z', u'2004-03-15T07:00:00.000Z', u'2004-03-15T08:00:00.000Z', u'2004-03-15T09:00:00.000Z', u'2004-03-15T10:00:00.000Z', u'2004-03-15T11:00:00.000Z', u'2004-03-15T12:00:00.000Z', u'2004-03-15T13:00:00.000Z', u'2004-03-15T14:00:00.000Z', u'2004-03-15T15:00:00.000Z', u'2004-03-15T16:00:00.000Z'])

    def get_ground_truth(table):
        return [row['id'].value for row in table if row['temp'].value > 50] #< 2.4]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'readings'


def get_fec12_obama():
    sql = """SELECT sum(disb_amt),
                   ((extract( epoch from (disb_dt) - (('2007-1-2')::timestamp) )) / (((24) * (60)) * (60)))::int as day
    FROM expenses
    WHERE (cand_id) = 'P80003338' GROUP BY day"""
    badresults = [1959,1952,1968,1991,1980,1963,1998]
    goodresults = [1839,1838,1841,1840,1837,1815,1814,1816,1813,1812,1820,1821,1817,1819,1828,1830,1831,1829,1827,1835,1833,1834,1836,1826,1822,1823,1824]


    def get_ground_truth(table):
        return [row['id'].value for row in table if row['disb_amt'].value > 1500000]
#    if 'GMMB' in row.domain['recipient_nm'].values[int(row['recipient_nm'])]] # 
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'expenses'



def get_harddata1_avg():
    sql = """select avg(v), z from harddata_1 group by z"""
    badresults = range(10)
    goodresults = range(10, 20)

    def get_ground_truth(table):
        return [row['id'].value for row in table if row['v'] > 50]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'harddata_1'

def get_harddata1_skewed():
    sql = """select (avg(x) - avg(y)), z from harddata_1 group by z"""
    badresults = range(10)
    goodresults = range(10, 20)

    def get_ground_truth(table):
        return [row['id'].value for row in table if row['x'] - row['y'] > 40]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'harddata_1'

def get_harddata1_sum():
    sql = """select sum(v + 105), z from harddata_1 group by z"""
    badresults = range(10)
    goodresults = range(10, 20)

    def get_ground_truth(table):
        return [row['id'].value for row in table if row['x'] - row['y'] > 40]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'harddata_1'


def get_harddata1_std():
    sql = """select stddev(v), z from harddata_1 group by z"""
    badresults = range(10)
    goodresults = range(10, 20)

    def get_ground_truth(table):
        return [row['id'].value for row in table if abs(row['v']) > 10]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'harddata_1'


def get_harddata2_avg():
    sql = """select avg(v), z from harddata_2 group by z"""
    badresults = range(10)
    goodresults = range(10, 20)

    def get_ground_truth(table):
        return [row['id'].value for row in table if row['v'] > 50]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'harddata_2'


def get_harddata3_avg():
    sql = """select sum(v), z from harddata_3 group by z"""
    badresults = range(10)
    goodresults = range(10, 20)

    def get_ground_truth(table):
        return [row['id'].value for row in table if row['v'] > 50]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'harddata_3'

def get_harddata4_sum():
    sql = """select sum(val), iter from multdim group by iter"""
    badresults = range(5)
    goodresults = range(5, 10)

    def get_ground_truth(table):
        return [row['id'].value for row in table if row['val'] >= 40]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'multdim'



def get_intel_foo():
    sql = """SELECT avg(temp), ((extract( epoch from ((date) + (time)) - (('2004-3-1')::timestamp) )) / ((30) * (60)))::int as dist
    FROM readings
    WHERE (((date) + (time)) > (('2004-3-10')::timestamp)) and (((date) + (time)) < (('2004-4-5')::timestamp))
    GROUP BY dist ORDER BY dist ASC"""
    badresults = [1146, 1147, 1148, 1149, 1150, 1157, 1151, 1152, 1153, 1154, 1158, 1159, 1155, 1156, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191]

    def get_ground_truth(table):
        return [row['id'].value for row in table if row['temp'] > 50]
    return sql, badresults, [], get_ground_truth, ErrTypes.TOOHIGH, 'readings'

def get_fec12_donation():
    sql = """SELECT avg(contb_receipt_amt), contb_receipt_dt as contb_receipt_dt FROM donations GROUP BY contb_receipt_dt"""
    badresults = ["2011-03-15T04:00:00.000Z","2011-01-01T05:00:00.000Z","2011-01-04T05:00:00.000Z","2011-01-03T05:00:00.000Z","2011-02-15T05:00:00.000Z","2011-02-07T05:00:00.000Z","2011-02-14T05:00:00.000Z","2011-01-24T05:00:00.000Z","2011-01-18T05:00:00.000Z","2011-01-31T05:00:00.000Z"]
    goodresults = ["2011-10-13T04:00:00.000Z","2011-10-16T04:00:00.000Z","2011-10-09T04:00:00.000Z","2011-10-07T04:00:00.000Z","2011-10-14T04:00:00.000Z","2011-10-02T04:00:00.000Z","2011-10-18T04:00:00.000Z","2011-10-11T04:00:00.000Z","2011-10-08T04:00:00.000Z","2011-10-17T04:00:00.000Z","2011-10-04T04:00:00.000Z","2011-10-15T04:00:00.000Z","2011-10-19T04:00:00.000Z","2011-10-23T04:00:00.000Z","2011-10-27T04:00:00.000Z","2011-10-28T04:00:00.000Z","2011-11-01T04:00:00.000Z","2011-10-20T04:00:00.000Z","2011-11-03T04:00:00.000Z","2011-11-07T05:00:00.000Z","2011-10-29T04:00:00.000Z","2011-10-30T04:00:00.000Z","2011-11-05T04:00:00.000Z","2011-11-06T04:00:00.000Z","2011-10-22T04:00:00.000Z","2011-10-21T04:00:00.000Z"]
 
    def get_ground_truth(table):
        return [row['id'].value for row in table if row['contb_receipt_amt'].value > 1500]
    return sql, badresults, goodresults, get_ground_truth, 'donations'


def get_fec12_donation2():
    sql = """SELECT sum(contb_receipt_amt) as None, contb_receipt_dt as contb_receipt_dt FROM donations WHERE (cand_nm) = 'Obama, Barack' GROUP BY contb_receipt_dt"""
    badresults = ["2012-03-31T04:00:00.000Z","2012-06-28T04:00:00.000Z","2012-06-30T04:00:00.000Z","2012-06-29T04:00:00.000Z","2012-04-30T04:00:00.000Z","2012-06-26T04:00:00.000Z","2012-05-31T04:00:00.000Z"]
    goodresults = ["2011-10-23T04:00:00.000Z","2011-10-22T04:00:00.000Z","2011-10-26T04:00:00.000Z","2011-10-25T04:00:00.000Z","2011-10-24T04:00:00.000Z"]

    def get_ground_truth(table):
        return [row['id'].value for row in table if row['contb_receipt_amt'].value > 1500]
    return sql, badresults[:2], goodresults[:2], get_ground_truth, ErrTypes.TOOHIGH,  'donations'

def get_fec12_donation2():
    sql = """SELECT sum(contb_receipt_amt) as None, contb_receipt_dt as contb_receipt_dt FROM donations WHERE (cand_nm) = 'Obama, Barack' GROUP BY contb_receipt_dt"""
    badresults = ["2012-03-31T04:00:00.000Z","2012-06-28T04:00:00.000Z","2012-06-30T04:00:00.000Z","2012-06-29T04:00:00.000Z","2012-04-30T04:00:00.000Z","2012-06-26T04:00:00.000Z","2012-05-31T04:00:00.000Z"]
    goodresults = ["2011-10-23T04:00:00.000Z","2011-10-22T04:00:00.000Z","2011-10-26T04:00:00.000Z","2011-10-25T04:00:00.000Z","2011-10-24T04:00:00.000Z"]

    def get_ground_truth(table):
        return [row['id'].value for row in table if row['contb_receipt_amt'].value > 1500]
    return sql, badresults[:2], goodresults[:2], get_ground_truth, ErrTypes.TOOHIGH,  'donations'

def get_lqm():
    sql = """SELECT sum(total_cost), ccm_payor FROM lqm group by ccm_payor"""
    badresults = ['MCARE']
    goodresults = ["BCOAH" ,"TCALL" ,"HOSPI",
        "HPMHC" ,"TCLPR" ,"MOTRV" ,"AUCHK" , 
        "MULTI" ,"MDMHC" ,"OTMHC" ,"NHMHC"]

    def get_ground_truth(table):
      return [row['id'].value for row in table if row['attending_physician'] in [18243 , 14122 , 33831 , 11587]]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH, 'lqm'
 


def get_sigmod_data(fname):
    """
    fname: data_<ndim>_<kdim>_<npts/group>_<volume>[_uh_sh_uo_so]
    """
    sql = """SELECT avg(v), g FROM %s GROUP BY g""" % fname
    badresults = range(5,10)
    goodresults = range(5)

    def get_ground_truth(table):
        return [row['id'].value for row in table if row['v'].value > 20]
    return sql, badresults, goodresults, get_ground_truth, ErrTypes.TOOHIGH,  fname


if __name__ == '__main__':
    print '\n'.join(map(str,enumerate(datasetnames)))
