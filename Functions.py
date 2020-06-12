from itertools import chain, combinations
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import *

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

def keyOpMVRPD(key):
    elements = key.split('_')
    if len(elements) == 4:
        return (elements[0], int(elements[1]), int(elements[2]), int(elements[3]) )
    else:
        print('Format Variable Name Errors!!')
        return 0

def keyOpTSP(key):
    return (key.split('_')[0], int(key.split('_')[1]), int(key.split('_')[2]))

def get_expr_coos(expr, var_indices):
    for i in range(expr.size()):
        dvar = expr.getVar(i)
        yield expr.getCoeff(i), var_indices[dvar]

def get_matrix_coos(m):
    dvars = m.getVars()
    constrs = m.getConstrs()
    var_indices = {v: i for i, v in enumerate(dvars)}
    for row_idx, constr in enumerate(constrs):
        for coeff, col_idx in get_expr_coos(m.getRow(constr), var_indices):
            yield row_idx, col_idx, coeff

def GenNbsGraph(path = "MIPLIB//js1n25_h_3.mps"):
    m = read(path)
    nzs = pd.DataFrame(get_matrix_coos(m), columns=['row_idx', 'col_idx', 'coeff'])

    edges = {}
    actRow = 0
    varsInRow = []
    for index, row in nzs.iterrows():
        
        if row['row_idx'] == actRow:
            varsInRow.append(row['col_idx'])
            
        else:
            for x1 in varsInRow:
                for x2 in varsInRow:
                    
                    if int(x1) < int(x2):
                        if (int(x1), int(x2)) in edges.keys():
                            edges[( int(x1) , int(x2) )] += 1
                        else:
                            edges[( int(x1) , int(x2) )] = 1

            actRow = row.row_idx
            varsInRow = []
            varsInRow.append(row.col_idx)

    #print(edges)
    plt.scatter(nzs.col_idx, nzs.row_idx, 
            marker='.', lw=0)
    plt.show()

if __name__ == '__main__':
    GenNbsGraph()