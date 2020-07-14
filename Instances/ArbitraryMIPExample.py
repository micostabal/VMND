import sys
import os
sys.path.append(os.path.pardir)
from VMNDproc import solver
from Neighborhood import varClusterFromMPS

# MPS file obtained from:
# https://miplib.zib.de/instance_details_binkar10_1.html


# We declare the path of the mps file:
path = os.path.join( os.path.pardir, 'MIPLIB', 'binkar10_1.mps' )

# MPS file obtained from:
# https://miplib.zib.de/instance_details_binkar10_1.html


# Neighborhoods are created from the Cluster Algorithm:
nbhs = varClusterFromMPS(path, numClu = 5, varFilter = None)


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
        minBCTime= 5,
        timeLimitSeconds= 300
    )

if __name__ == '__main__': pass