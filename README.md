scorpion
========


Python Setup

    pip install 
      dateutils
      flask
      flask-cache
      flask-compress
      nltk
      matplotlib
      numpy
      orange
      psycopg2
      python-dateutil
      rtree
      scipy
      sqlalchemy


Database Setup

    createdb status

Run for each database/table you want to run scorpion over

    python fixnulls.py <dbname> <tablename>

At this point, the command line programs are not-so-good, 
so switch over to the summary package, which is a webserver that
provides a front-end interface.
