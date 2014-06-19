from scorpion.db import connect
import random

db = connect('fec12')

for i in xrange(500):
  amt = 100000 + random.randint(-5000, 5000)
  nm = 'Obama, Barack'
  candid = 'P80003338'
  disb_dt = '2012-12-10'
  recipient_nm = "eugene"
  recipient_city = "Berkeley"
  recipient_st = "CA"
  recipient_zip = random.randint(94520, 94530)
  memo_text = "eugene"
  disb_desc = "eugene"
  cmte_id = "myid%d" % (i + 10000000)

  q = """INSERT INTO expenses_extradata(
     cmte_id, cand_id, cand_nm, recipient_nm,
     recipient_city, recipient_st, recipient_zip,
     memo_text, disb_desc, disb_dt, disb_amt) 
     values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
  db.execute(q, (
    cmte_id,
    candid,
    nm,
    recipient_nm,
    recipient_city,
    recipient_st,
    recipient_zip,
    memo_text,
    disb_desc,
    disb_dt,
    amt
  ))



