import json
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

class Status(object):
  def __init__(self, reqid=None):
    self.engine = create_engine("postgresql://localhost/status", poolclass=NullPool)
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
    stmts = [
      "create table requests(id serial)",
      "create table status(id serial, reqid int, status varchar)",
      "create table rules(reqid int, label text, val text)"
    ]
    for stmt in stmts:
      try:
        self.db.execute(stmt)
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
    if len(rows):
      return rows[0][0]
    return 'no status yet'

  def update_rules(self, label, rules):
    q = """update rules set val=%s where reqid=%s and label = %s;
    insert into rules
      select %s, %s, %s 
      where not exists (
        select 1 from rules where reqid=%s and label = %s
      );
    """
    from scorpion.learners.cn2sd.rule import rule_to_json
    jsons = []
    for rule in rules:
      if isinstance(rule, dict):
        jsons.append(rule)
      else:
        jsons.append(rule_to_json(rule, yalias=label))
    val = json.dumps(jsons)
    self.db.execute(q, 
        val, self.reqid, label, 
        self.reqid, label, val, 
        self.reqid, label)

  def get_rules(self):
    q = "select label, val from rules where reqid = %s"
    rows = self.db.execute(q, self.reqid).fetchall()
    if len(rows):
      ret = [ (str(row[0]), json.loads(row[1])) for row in rows ]
      return ret
    return []
    
    

