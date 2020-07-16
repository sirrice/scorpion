scorpion
========


Install

    pip install scorpion

If you wish to run the Scorpion web-based demo, you also should load the intel dataset into postgresql.  To do so, download the following ddl file, create a database called intel, and load the ddl into the database:

    # download the ddl
    wget "https://www.dropbox.com/s/glutiyu2uju4ijq/intel.ddl?dl=0"

    # create the database
    createdb intel   

    # Load the database
    psql -f intel.ddl\?dl=0 intel

To run Scorpion on custom tables, we need to replace NULL values with surrogates and clean up
nested quotes.  Run the following for each database/table you want to run scorpion over.

    python fixnulls.py <dbname> <tablename>


The command line programs included with this package are immature.
The `dbwipes` package integrates with scorpion and proves a visual
front-end interface.

    pip install dbwipes

Read the [dbwipes README](https://github.com/sirrice/dbwipes) for further instructions
