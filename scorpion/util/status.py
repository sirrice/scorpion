from sqlalchemy import create_engine

class Status(object):
  def __init__(self, reqid=None):
    self.engine = create_engine("postgresql://localhost/status")
    self.db = self.engine.connect()
    self.create_table()
    self.reqid = reqid
    self.reqid = self.get_req_id()

  def close(self):
    try:
      self.db.close()
    except:
      pass
    try:
      self.engine.dispose()
    except:
      pass

  def create_table(self):
    try:
      self.db.execute("create table requests(id serial)")
      self.db.execute("create table status(id serial, reqid int, status varchar)")
    except:
      pass

  def cleanup(self):
    self.db.execute("delete from status where reqid = %s", self.reqid)
    self.db.execute("delete from requests where id = %s", self.reqid)
    self.reqid = None

  def get_req_id(self):
    if self.reqid is not None: 
      q = "select count(*) from requests where id = %s"
      count = self.db.execute(q, self.reqid).fetchone()[0] 
      print 'status count', count
      if count > 0: 
        return self.reqid
    q = "insert into requests values (default) returning id"
    return self.db.execute(q).fetchone()[0]

  def update_status(self, s):
    print "update status", self.reqid, s
    q = "insert into status values (default, %s, %s)"
    self.db.execute(q, self.reqid, s)

  def latest_status(self):
    q = "select status from status where reqid = %s order by id desc limit 1"
    rows = self.db.execute(q, self.reqid).fetchall()
    print q
    print rows
    if len(rows):
      return rows[0][0]
    return 'no status yet'

