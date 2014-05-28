from scorpion.db import *
from scorpion.util import Status

db = connect("status")

status = Status(db)
assert "initialized" == status.latest_status()
for i in xrange(10):
  status.update_status('testing %d' % i)
  assert ("testing %d" % i) == status.latest_status()
status.cleanup()
assert "no status yet" == status.latest_status()

