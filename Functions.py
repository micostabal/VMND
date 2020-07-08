from itertools import chain, combinations
import pandas as pd
import time
import matplotlib.pyplot as plt
from gurobipy import *
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans,SpectralClustering
from scipy.sparse import csr_matrix
import os
import sys

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

def get_expr_coos_new(expr, var_indices):
    for i in range(expr.size()):
        dvar = expr.getVar(i)
        yield var_indices[dvar], dvar.VarName

def get_matrix_coos_new(m):
    dvars = m.getVars()
    constrs = m.getConstrs()
    var_indices = {v: i for i, v in enumerate(dvars)}
    for row_idx, constr in enumerate(constrs):
        for col_idx, name in get_expr_coos_new(m.getRow(constr), var_indices):
            yield row_idx, col_idx, name

def VisualizeNonZeros(path = os.path.join("MIPLIB", "SomeInstanceIRPCS25_7_3.mps" ) ):
    m = read(path)
    nzs = pd.DataFrame(get_matrix_coos(m), columns=['row_idx', 'col_idx', 'coeff'])

    actRow = 0
    varsInRow = []

    for index, row in nzs.iterrows():
        if row['row_idx'] == actRow:
            varsInRow.append(row['col_idx'])

            actRow = row.row_idx
            varsInRow = []
            varsInRow.append(row.col_idx)



    plt.scatter(nzs.col_idx, nzs.row_idx, 
            marker='.', lw=0)
    plt.show()

def genAffinityMatrix(
        path = os.path.join("MIPLIB", "SomeInstanceIRPCS15_8_3.mps" ),
        varFilter = lambda x : x[0] == 'x',
        verbose = True ):
    starting_time = time.time()
    m = read(path)
    nzs = pd.DataFrame(get_matrix_coos_new(m), columns=['row_idx', 'col_idx', 'name'])

    # An undirected affinity graph is created.
    graph = nx.Graph()

    actRow = 0
    varsInRow = []

    if varFilter == None:
        varFilter = lambda x : True

    for index, row in nzs.iterrows():

        if int(row.row_idx) == actRow and varFilter(row['name']):
            varsInRow.append(row['name'])
        elif int(row.row_idx) != actRow and varFilter(row['name']):

            # We update the edge dictionary:
            if len(varsInRow) > 1:
                for n1 in varsInRow:
                    for n2 in varsInRow:
                        if n1 < n2:
                            if not graph.has_edge(n1, n2):
                                graph.add_edge(n1, n2, weight = 1)
                            else:
                                graph[n1][n2]['weight'] += 1
            
            # The row and varaibles list parameters are set back to 0 and empty respectively.
            actRow = int(row.row_idx)
            varsInRow = []
            if verbose:
                print('Completed {} of total rows.'.format( round( index / len(nzs) , 3) ) )

            # If is not filtered, we add the variable name to the list.
            if varFilter(row['name']):
                varsInRow.append(row['name'])
    
    if verbose:
        print('------ Affinity Matrix successfully stored. Elapsed : {} ------'.format( round(time.time() - starting_time , 3) ) )
    return graph

def genClusterNeighborhoods(
        path = os.path.join("MIPLIB", "SomeInstanceIRPCS20_6_3.mps" ),
        nClusters = 18,
        verbose = True,
        fNbhs = False,
        varFilter = lambda y : y[0] == 'x'):

    graph = genAffinityMatrix(path, varFilter, verbose = verbose)

    adj_matrix = nx.to_numpy_matrix(graph)

    clusters = SpectralClustering(
        affinity = 'precomputed',
        assign_labels = "discretize",
        random_state = 0,
        n_clusters = nClusters).fit_predict(adj_matrix)

    node_list = list(graph.nodes)

    dLabels = { node_list[i] : clusters[i] for i in range(len(clusters))}
    m = read(path)

    if varFilter == None:
        keyVars = list(  map( lambda var: var.VarName, list(m.getVars()) ) )
    else:
        keyVars = list( filter ( varFilter , map( lambda var: var.VarName, list(m.getVars() )  ) ) )

    
    incomplete = False
    for ind in range(len(keyVars)):
        if keyVars[ind] not in node_list:
            incomplete = True
            dLabels[keyVars[ind]] = ind % nClusters

    if verbose:
        print('------ The key variable choice was incomplete ------')    

    if verbose:
        print('------ Cluster labels computed ------')

    if not fNbhs:
        outer = {}
        for i in range(nClusters):
            outer[i + 1] = {
                0 : tuple(filter( lambda x : dLabels[x] != i , node_list ))
            }
            #print("------ A {}% of neighborhoods stored ------".format( round(100 * i / nClusters, 3) ) )
        
        return outer

    else:
        return dLabels

def mps_reader(file_name = os.path.join( 'MIPLIB' , 'abs1n5_1.mps' )):
    for row in open(file_name, "r"):
        yield row


if __name__ == '__main__':
    #genAffinityMatrix()
    #gc1 = genClusterNeighborhoods( path = os.path.join( 'MIPLIB' , 'abs1n5_1.mps' ), fNbhs = False)

    started = False
    finished = False
    varFilter = lambda x: x[0] == 'x'
    rvar = {}
    
    for i in mps_reader(file_name = os.path.join( 'MIPLIB' , 'SomeInstanceIRPCS60_12_3.mps' )):
        row = i.strip('\n')
        if row == 'COLUMNS':
            print('column detected')
            started = True
            continue
        elif row == 'RHS':
            print('end of rows')
            finished = True
        
        if started and not finished and "'MARKER'" not in row:
            tupleRow = tuple( filter(lambda x : x != '', i.strip('\n').strip(' ').split(' ') ) )
            varName = tupleRow[0]
            constrName = tupleRow[1]
            if varFilter(varName):
                if constrName in rvar.keys():
                    rvar[constrName].append(varName)
                else:
                    rvar[constrName] = [varName]
    
    print(sys.getsizeof(rvar))
    keys = tuple(rvar.keys())
    edges = {}
    for constr in keys:
        for v1 in rvar[constr]:
            for v2 in rvar[constr]:
                if v1 < v2:
                    if (v1, v2) in edges.keys():
                        edges[(v1, v2)] += 1
                    else:
                        edges[(v1, v2)] = 1
        del rvar[constr]

    print(sys.getsizeof(edges))

