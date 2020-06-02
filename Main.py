from gurobipy import *
import numpy as np
from ConComp import getSubsets
from Neighborhood import Neighborhoods, genIRPneighborhoods, genIRPneigh
from Others import loadMPS
from Test import getCheckSubTour
from Cuts import genSubtourLazy, Cut
import time
from VMNDproc import solver
import os

n, H, K = 15, 3, 2

nsAct = Neighborhoods(lowest = 2, highest = 5, randomSet = False, outerNeighborhoods = genIRPneigh(n, H, K))

path = os.path.join('MIPLIB', 'abs1n15_3.mps')
solver(
    path,
    addlazy = True,
    funlazy = genSubtourLazy(n, H, K),
    importNeighborhoods = True,
    importedNeighborhoods = nsAct,
    funTest = getCheckSubTour(n, H, K),
    alpha = 3,
    callback = 'pure',
    verbose = True
    )



if __name__ == '__main__': pass