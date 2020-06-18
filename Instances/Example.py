import sys
import os
sys.path.append(os.path.pardir)
from MVRPD import MVRPD
from VMNDproc import solver
from Neighborhood import Neighborhoods, importNeighborhoods
from Functions import genClusterNeighborhoods
from Cuts import Cut

##### TUTORIAL #####

##### Execute from .mps with different Neighborhoods #####

pathMPS = os.path.join( os.path.pardir, 'MIPLIB', 'binkar10_1.mps' )

##### Neighborhoods based on cluster
Ncl = 18
outerNbhs = genClusterNeighborhoods(pathMPS, Ncl, verbose = True)
nbhs = Neighborhoods(lowest = 1, highest = Ncl, keysList= None, randomSet=None, outerNeighborhoods = outerNbhs )


##### Neighborhoods created from scratch
"""outerNbhs = {
    1 : {
        1 : ['C{}'.format(i) for i in range(1000, 2290) if i >= 1050 ],
        2 : ['C{}'.format(i) for i in range(1000, 2290) if i >= 1100 or i <= 1050  ]
    },
    2 : {
        1 : ['C{}'.format(i) for i in range(1000, 2290) if i >= 2050 ],
        2 : ['C{}'.format(i) for i in range(1000, 2290) if i >= 2100 or i <= 2050 ]
    }
}
nbhs = Neighborhoods(lowest = 1, highest = 2, keysList= None, randomSet= False, outerNeighborhoods= outerNbhs)"""

##### Neighborhoods randomly generated, by far the worst option.

## we need a list of model variables (not necessarily all of them)
#klist = [ 'C{}'.format(i) for i in range(1000, 2299) ]

## We can add lowest and highest according to the model size, in this case 1 and 5.
#nbhs = Neighborhoods(lowest = 1, highest = 5, keysList = klist, randomSet = True, outerNeighborhoods = None)

##### Neighborhoods imported from a .txt file, based on previously created Nbhs. 
#filename = 'binkar10_1Neighborhoods'
#nbhs.exportNeighborhood(filename)
#nbhsImported = importNeighborhoods(filename + '.txt')

def sepFun(solValues):
    added = []

    if solValues['C2221'] > 0 and solValues['C2224'] > 0:

        # New Cut is created
        newCut = Cut()

        # Nonzeros are declared: C1980 + C1981 <= 1
        newCut.nonzero = {
            'C2221' : 1,
            'C2224' : 1
        }

        # Sense of constraint declared.
        newCut.sense = '<='

        # Right hand side of constraint declared.
        newCut.rhs = 1

        ## We add the new cut into the added cuts.
        added.append(newCut)

    if solValues['C2226'] > 0 and solValues['C2228'] > 0:
        ## The same with the new variables

        newCut = Cut()
        newCut.nonzero = {
            'C2226' : 1,
            'C2228' : 1
        }
        newCut.sense = '<='
        newCut.rhs = 1

        added.append(newCut)
    
    return added

modelExample = solver(
    pathMPS,
    verbose = True,
    addlazy = False,
    funlazy = None,
    importNeighborhoods = True,
    importedNeighborhoods = nbhs,
    funTest = None,
    callback= 'vmnd',
    alpha = 1,
    minBCTime = 0,
    timeTimitSeconds = 200
)

## Variables can be printed with gurobi method.
#modelExample.printAttr('X')


##### Execute the heuristic of an instance #####

"""# The path of the data (.dat or .vrp or .txt) file.
pathMVRPDInstance = os.path.join('MVRPDInstances' , 'ajs3n75_l_6.dat')

# The instance is created.
instanceMVRPD = MVRPD(pathMVRPDInstance)

# We can print some attributes:

# The number of retailers or customers:
print('V : ', instanceMVRPD.V)

# The number of Periods:
print('H : ', instanceMVRPD.H)

# The number of vehicles
print('m : ', instanceMVRPD.m)

# The capacity of the vehicles:
print('Q : ', instanceMVRPD.Q)

# The initial Inventory
print('q : ', instanceMVRPD.q)

# We run the instance
instanceMVRPD.run()

# Solution (results) can be visualized
instanceMVRPD.visualizeRes()"""


if __name__ == '__main__': pass