

# Info

## Demo Queries

### Intel Dataset

Normal time series

    select stddev(temp), avg(temp),
    ((extract(epoch from date+time - '2004-3-1'::timestamp)) / (30*60))::int as dist
    from readings
    where date+time > '2004-3-1'::timestamp and date+time < '2004-3-7'::timestamp
    group by dist order by dist;

Expected Result

    and not ((epochid >28681.500 and moteid = 18))
    and not ((voltage <=2.362)) and not ((humidity <=8.216))
    and not ((moteid = 24 and epochid <=46677.500))

Highlight a slice of rightmost and select the 120 deg tuples

    SELECT avg(temp), stddev(temp),
    (extract(epoch from time) / (60)) :: int as e
    FROM readings
    WHERE date+time > '2004-3-20'::timestamp and date+time < '2004-4-5'::timestamp
    GROUP BY e ORDER BY e

Expected Result

    and not ((epochid >28681.500 and moteid = 18)) and 
    not ((voltage <=2.362)) and 
    not ((humidity <=8.216)) and 
    not ((moteid = 24 and epochid <=46677.500)) 


### FEC Queries

John McCain 08 donations can't find anything DONT USE

    SELECT sum(contb_receipt_amt), ((extract(epoch from contb_receipt_dt)) / (24*60*60)) :: int as day
    FROM contrib08 WHERE cand_nm='McCain John S'
    GROUP BY day ORDER BY day

    
Mitt romney negative donations

* highlight negative results
* highlight ~900 negative tuples

    SELECT min(contb_receipt_amt), ((extract(epoch from contb_receipt_dt)) / (24*60*60)) :: int as day
    FROM contrib
    WHERE cand_nm='Romney Mitt'
    GROUP BY day ORDER BY day

Obama: find where he spent his money

    SELECT sum(disb_amt), (extract( epoch from disb_dt-'2007-1-2'::timestamp) / (24 * 60 * 60)) :: int as day 
    FROM expend WHERE cand_id = 'P80003338' GROUP BY day ;
