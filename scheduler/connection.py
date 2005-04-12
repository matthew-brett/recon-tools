import MySQLdb.connections
import MySQLdb.cursors

##############################################################################
class SchedulerConnection (MySQLdb.connections.Connection):

    #-------------------------------------------------------------------------
    def __init__( self ):
        MySQLdb.connections.Connection.__init__( self,
            host="neuroscience.berkeley.edu", user="bic_user",
            passwd="bic_user_pass", db="bic" )
        self.__cursor = self.cursor( cursorclass=MySQLdb.cursors.DictCursor )


    #-------------------------------------------------------------------------
    def execute(self, query, args=None ):
        return self.__cursor.execute( query, args=args ) 


    #-------------------------------------------------------------------------
    def executemany(self, query, args):
        return self.__cursor.executemany( query, args ) 


    #-------------------------------------------------------------------------
    def query( self, query, args=None ):
        self.execute( query, args=args )
        return self.fetchall()
    

    #-------------------------------------------------------------------------
    def fetchall( self ):
        return self.__cursor.fetchall()


    #-------------------------------------------------------------------------
    def fetchmany( self, size=None ):
        return self.__cursor.fetchmany( size=size )


    #-------------------------------------------------------------------------
    def fetchone( self ):
        return self.__cursor.fetchone()


    #-------------------------------------------------------------------------
    def __del__( self ):
        self.__cursor.close()
        del self.__cursor
