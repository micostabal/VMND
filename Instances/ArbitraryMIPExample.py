import sys
import os
sys.path.append(os.path.pardir)
from VMNDproc import solver
from Neighborhood import Neighborhoods, varClusterFromMPS
from gurobipy import *

# MPS file obtained from:
# https://miplib.zib.de/instance_details_binkar10_1.html


# We declare the path of the mps file:
path = os.path.join( os.path.pardir, 'MIPLIB', 'binkar10_1.mps' )

# MPS file obtained from:
# https://miplib.zib.de/instance_details_binkar10_1.html



#########################################################
##### Neighborhoods generated from VARIABLE CLUSTER #####
#########################################################

#nbhs = varClusterFromMPS(path, numClu = 5, varFilter = None)

############################################################
##### Neighborhoods given by the user: FUNCTIONAL Case #####
############################################################

# To get the model varaibles, we hace to read it from the mps file.
m = read(path)

# A list containing the variable's names. They're 'C0001' to 'C2298'
klist = list( map( lambda var: var.VarName, m.getVars() ) )

# For the sake of comprehension we will define only one parameterization (tuple of length one)
outerNbhs = {
    i : (0, ) for i in range(1, 6)
}

def funNbhs(varName, depth, param):
    num = int(varName.lstrip('C'))
    if depth == 1:
        # Free if lower or equal than 500
        return num > 500
    elif depth == 2:
        # Free from 501 to 1000
        return depth < 501 or depth > 1000
    elif depth == 3:
        # Free from 1001 to 1500
        return  depth < 1001 or depth > 1500
    elif depth == 4:
        # Free from 1501 to 2000
        return  depth < 1501 or depth > 2000
    elif depth == 5:
        # Free from 2001 to 2298
        return depth < 2001
    else:
        # An error is raised if varaible is not in the correct format.
        print('Error, this name does not exist!')
        return 0


"""nbhs = Neighborhoods(
    lowest = 1,
    highest = 5,
    keysList= klist,
    randomSet= False, 
    outerNeighborhoods= outerNbhs,
    useFunction=True,
    funNeighborhoods= funNbhs
)"""

##########################################################
##### Neighborhoods given by the user: PHYSICAL Case #####
##########################################################

# This can be slower for larger instances, so it is recomended to use them
# only if the names do not (or could not be modified) obey any prefix/index/number rule.


# In this case, the input outerNeighborhoods is a dictionary of dictionaries.
# We can see that all names are extensively included in each list
outerNbhs = {
    1 : {
        0 : [element for element in klist if int(element.lstrip('C')) > 500 ]
    },
    2 : {
        0 : [element for element in klist if int(element.lstrip('C')) <= 500 or int(element.lstrip('C')) >= 1001 ]
    },
    3 : {
        0 : [element for element in klist if int(element.lstrip('C')) <= 1000 or int(element.lstrip('C')) >= 1501 ]
    },
    4 : {
        0 : [element for element in klist if int(element.lstrip('C')) <= 1500 or int(element.lstrip('C')) >= 2001 ]
    },
    5 : {
        0 : [element for element in klist if int(element.lstrip('C')) <= 2000 ]
    }
}
# We use klist as well because we need the list of varaible's names


nbhs = Neighborhoods(
    lowest = 1,
    highest = 5,
    keysList= None,
    randomSet= False, 
    outerNeighborhoods= outerNbhs,
    useFunction=False,
    funNeighborhoods= None
)


# The heuristic si executed.
# Alpha is set to 1, with no minimum time in B&C and 300 seconds of time limit.
solver(
        path,
        verbose = True,
        addlazy= False,
        funlazy = None,
        importNeighborhoods=True,
        importedNeighborhoods= nbhs,
        funTest= None,
        callback = 'vmnd',
        alpha = 2,
        minBCTime= 10,
        timeLimitSeconds= 300
)

if __name__ == '__main__': pass