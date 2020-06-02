from itertools import chain, combinations

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

def transformKey(variableName):
    name = variableName
    args = name.split('_')
    if (args[0] == 'I'):
         return ('I', int(args[1]), int(args[2]) )
    elif (args[0] == 'z'):
        return ('z', int(args[1]), int(args[2]), int(args[3]) )
    elif (args[0] == 'q'):
        return ('q', int(args[1]), int(args[2]), int(args[3]) )
    elif (args[0] == 'y'):
        return ('y', int(args[1]), int(args[2]), int(args[3]), int(args[4]) )
    else:
        print('Format key error!!')
        return 0

def IRPCStransformKey(variableName):
    name = variableName
    args = name.split('_')
    if (args[0] == 'I'):
         return ('I', int(args[1]), int(args[2]) )
    elif (args[0] == 'S'):
         return ('S', int(args[1]), int(args[2]) )
    elif (args[0] == 'z'):
        return ('z', int(args[1]), int(args[2]) )
    elif (args[0] == 'q'):
        return ('q', int(args[1]), int(args[2]), int(args[3]) )
    elif (args[0] == 'p'):
        return ('p', int(args[1]), int(args[2]), int(args[3]) )
    elif (args[0] == 'y'):
        return ('y', int(args[1]), int(args[2]), int(args[3]) )
    elif (args[0] == 'v'):
        return ('v', int(args[1]), int(args[2]), int(args[3]), int(args[4]) )
    elif (args[0] == 'x'):
        return ('x', int(args[1]), int(args[2]), int(args[3]), int(args[4]) )
    else:
        print('Format key error!!')
        return 0

def keyOpVRP(key):
    elements = key.split('_')
    if len(elements) == 3:
        return ('z', int(elements[1]), int(elements[2]))
    elif len(elements) == 4:
        return ('y', int(elements[1]), int(elements[2]), int(elements[3]) )
    else:
        print('Format Variable Name Errors!!')
        return 0

def keyOpTSP(key):
    return (key.split('_')[0], int(key.split('_')[1]), int(key.split('_')[2]))


if __name__ == '__main__': pass