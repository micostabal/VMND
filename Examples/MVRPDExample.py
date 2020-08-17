##### The imports depend on which folder is the file located #####
import sys
import os
sys.path.append(os.path.pardir)
sys.path.append( os.path.join(os.path.pardir, 'Instances'))
import MVRPD as MVRPD
from sklearn.neighbors import NearestNeighbors
from Neighborhood import Neighborhoods
from VMNDproc import solver

##### The loading of the model and creation of the mps file #####
 
# The instance is loaded from the MPS File
inst1 = MVRPD.MVRPD(path = os.path.join( os.path.pardir, 'Instances', 'MVRPDInstances' , 'ajs1n25_h_3.dat' ) )
#'x_i_j_k_t'


# The model is created and written as a mps file in the folder MIPLIB
inst1.exportMPS()

# We save the path of the mps file.
path = inst1.pathMPS

##### The creation of the Neighborhoods #####

## The k-nearest neighbors are computed ##

X = inst1.positions

#print(X)

# In larrain et al 2019 the neighbors parameter is set to 20.
nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(X)
indices = nbrs.kneighbors(X)[1]

#print(indices)


# The function that decides whether a variable is fixed in a certain neighborhood/parameterization is set.
def fNbhs(varName, depth, param):

    # The different elements of the name of the variable are separated.
    elements = varName.split('_')

    # If the name has three elements won't be fixed.
    if len(elements) < 4:
        return False
    else:
        tl = int(elements[3])
        il = int(elements[1])
        jl = int(elements[2])

        if depth == 1:
            return tl != param
        elif depth == 2:
            return il not in indices[param] and jl not in indices[param]
        else:
            print('Error 23 Nbhds Function!! ')
            return 0
    return True


# outerNeighborhoods parameter will be this dictionary.
# The keys are 1 and 2 as the neighborhoods.
# The parameterizations are declared as a tuple for each parameterization.
outer = {
    1 : tuple([ tf for tf in range(1, inst1.H + 1) ]),
    2 : tuple([ i for i in range(1, inst1.V + 1) ])
}

# We declare in a list all the different names of the variables that could be fixed.
klist = ['x_{}_{}_{}'.format( i, j, t )
    for t in range(1, inst1.H  + 1) for i in range(inst1.V + 1) for j in range(inst1.V + 1) if i != j]


functionNeighborhoods =  Neighborhoods(
    lowest = 1,
    highest = 2,
    keysList= klist,
    randomSet=False,
    outerNeighborhoods=outer,
    funNeighborhoods = fNbhs,
    useFunction=True
)


##### The Heuristic is executed #####

# Alpha is set to 1, no minimum time in B&C and time limit of 300 seconds. 
modelOutput = solver(
    path = path, # Of the mps file.
    addlazy = False,
    funlazy= None,
    importNeighborhoods= True,
    importedNeighborhoods= functionNeighborhoods,
    funTest= inst1.genTestFunction(),
    alpha = 1,
    callback = 'vmnd',
    verbose = True,
    minBCTime = 5,
    timeLimitSeconds= 300
)


##### Other Features #####

# Of course this functionalities and other neighborhoods can be directly implemented.

# We cun run directly the heuristic with the same function neighborhoods with this command:
"""inst1.run(
    outImportedNeighborhoods='function',
    writeResult=False,
    outVerbose=True,
    outCallback = 'vmnd'
)

# Results can be visualized for every period.!
inst1.visualizeRes()"""

if __name__ == '__main__': pass