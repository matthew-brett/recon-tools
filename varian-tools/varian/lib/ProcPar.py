"Interface to the procpar file in Varian scan output"
from itertools import imap, ifilter
from tokenize import generate_tokens
import token as tokids


class StaticObjectDictMixin (object):
    """
    When mixed-in with a class which supports the dict interface,
    this mixin will make it read-only and also add the ability to
    access its values as attributes.
    """
    def __setitem__( self, key, val ):
        raise AttributeError, 'object is read-only'

    def __setattr__( self, attname, val ):
        raise AttributeError, 'object is read-only'

    def __getattr__( self, attname ):
        try:
            return self[attname]
        except KeyError:
            raise AttributeError, attname



def advanceTo( tokens, stopid ):
    collection = []
    for tokid, tok in tokens:
        collection.append( (tokid, tok) )
        if tokid == stopid: break
    return collection


def advanceBy( tokens, count ):
    collection = []
    for tokid, tok in tokens:
        if count < 1: break
        if tokid == tokids.NEWLINE: continue
        collection.append( (tokid, tok) )
        count -= 1
    return collection


def cast( toktup ):
    tokid, tok = toktup
    if   tokid == tokids.STRING: return tok[1:-1]
    elif tokid == tokids.NUMBER:
        characteristic = tok[0]=="-" and tok[1:] or tok
        if characteristic.isdigit(): return int(tok)
        else: return float(tok)
    else: 
        print tokids.tok_name[tokid], tok
        return tokids.tok_name[tokid]


class Parser (StaticObjectDictMixin, dict):
    """
    A read-only representation of the named records found in a Varian procpar file.
    Record values can be accessed by record name in either dictionary style or object
    attribute style.  A record value is either a single element or a tuple of elements.
    Each element is either a string, int, or float.
    """

    def __init__( self, filename ):
        class foo: pass # just need a mutable object to hang the negative flag on
        state = foo()
        state.isneg = False # flag indicating when we've parse a negative sign

        # stream of each element's first two items (tokenid and token) from the raw
        # token stream returned by the tokenize module's generate_tokens function
        tokens = imap( lambda t: t[:2], generate_tokens( file( filename ).readline ) )

        # filter out negative ops (but remember them in case they come before a number)
        def negfilter( toktup ):
            tokid, tok = toktup
            if tok == "-":
                state.isneg = True
                return False
            return True
        tokens = ifilter( negfilter, tokens )

        # add back the negative sign for negative numbers
        def negnums( toktup ):
            tokid, tok = toktup
            extra = ""
            if state.isneg:
                state.isneg = False
                if tokid == tokids.NUMBER: extra = "-"
            return tokid, "%s%s"%(extra,tok)
        tokens = imap( negnums, tokens )

        class KeyValPairs:
            def __iter__( self ): return self
            def next( self ):
                toks = advanceTo( tokens, tokids.NAME )
                if not toks or not toks[-1][0] == tokids.NAME: raise StopIteration
                else: name = toks[-1][1]
                rest = advanceTo( tokens, tokids.NEWLINE )
                numvalues = int( tokens.next()[1] )
                values = tuple( map( cast, advanceBy( tokens, numvalues ) ) )
                return name, values
        
        self.update( dict( [pair for pair in KeyValPairs()] ) )


if __name__ == "__main__":
    import pprint
    p = Parser( "procpar" )
    pprint.pprint( p )
