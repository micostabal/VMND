##### The imports depend on which folder is the file located #####
import sys
import os
sys.path.append(os.path.pardir)
sys.path.append(os.path.join(os.path.pardir, 'Instances'))
import TSP as TSP
from Neighborhood import Neighborhoods, varClusterFromMPS
from VMNDproc import solver
from Cuts import Cut
from ConComp import getSubsets

##### The loading of the model and creation of the mps file #####

# The instance is loaded from the MPS File
nodes = 60

tspInst = TSP.TSP(nodes)

# The model is created and written as a mps file in the folder MIPLIB
tspInst.exportMPS()

"""
The variables are set as x_i_j with i < j, for this is an
undirected formulation of TSP.
"""

# We save the path of the mps file
pathTSPInst = tspInst.pathMPS

# We shall see ..\MIPLIB\randomTSP60nodes.mps
#print(pathTSP)


##### The creation of the Neighborhoods #####

# Neighborhoods are created from varaible cluster
nbhs = varClusterFromMPS(pathTSPInst, numClu = 5, varFilter = None)

# Example of subtour Elimination function
def sepFunctionTSP(solValues):

    # List of cuts is initialized as empty
    cuts = []

    # A new dictionary can be created, formating the keys as tuples (string 'x', int i, int j) instead of strings
    # The values are the same of course as those of solValues and contain the binary variables x_i_j with i < j.
    solValues = { (key.split('_')[0], int(key.split('_')[1]), int(key.split('_')[2]) ) : solValues[key] for key in solValues.keys()}
    
    # A list stores with tuples of integers the active edges.
    edges = []
    for i in range(nodes):
        for j in range(nodes):
            if i < j:
                # Numerical precaution, this cn be set to >= 1 as well.
                if solValues[('x', i, j)] >= 0.99:
                    edges.append((i, j))

    #This function returns through a list of lists of integers
    # all connected components in case there is more than one.
    subsets = getSubsets(edges, nodes)


    if len(edges) > 0:
        for subset in subsets:

            # A new Cut object is created
            newCut = Cut()
            nonzeros = {}

            # The coefficients are always 1, we add all edges associated to the elements of the subset.
            nonzeros.update({ 'x_{}_{}'.format(i, j) : 1 for i in subset for j in subset if i < j })
            newCut.nonzero = nonzeros

            # The sense is added as a string '<='
            newCut.sense = '<='

            # The cardinality of subset S can be computed as the length of the subset list
            newCut.rhs = len(subset) - 1

            # We append the new cut to the list
            cuts.append(newCut)
    return cuts


##### The Heuristic is executed #####

# The heuristic is executed as usual, but this time
# we set addLazy as True and we set sepFunctionTSP to the funlazy input 

solver(
        pathTSPInst,
        verbose = True,
        addlazy= True,
        funlazy = sepFunctionTSP,
        importNeighborhoods=True,
        importedNeighborhoods= nbhs,
        funTest= None,
        callback = 'vmnd',
        alpha = 1,
        minBCTime= 4,
        timeLimitSeconds= None
)

# Of course this method is already implemented it suffices to execute run method with the desired parameters like this:

tspInst.run()

# Finally a nice matplotlib visualization can be performed with the 
tspInst.visualizeRes()


if __name__ == '__main__': pass