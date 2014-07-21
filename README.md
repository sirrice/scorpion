scorpion
========


Install

    pip install scorpion

Run the following for each database/table you want to run scorpion over.
The script replaces NULL values with surrogates and removes nested quotes.

    python fixnulls.py <dbname> <tablename>


The command line programs included with this package are immature.
The `dbwipes` package integrates with scorpion and proves a visual
front-end interface.


    pip install dbwipes

Read the dbwipes README for further instructions
