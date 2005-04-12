"""
Python package for connecting to the BIC scheduler database.
The simplest way to obtain a connection is:

>>> import scheduler
>>> db = scheduler.connect()
"""

def connect():
    import scheduler.connection
    return scheduler.connection.SchedulerConnection()
