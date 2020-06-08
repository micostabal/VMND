from gurobipy import *
from Others import loadMPS
import numpy as np
from ConComp import getSubsets
from Functions import transformKey
import os


class Cut:

    def __init__(self):
        self.nonzero = {}
        self.sense = '<='
        self.rhs = 0

    def __str__(self):
        return str(self.nonzero) + ' ' + self.sense + ' ' + str(self.rhs)


# Cut function
def genSubtourLazy(n, H, K, mainStVarName = 'y', secondStVarName = 'z', keyOperator = transformKey):
    def f1(solValues):
        cuts = []
        solValues = { keyOperator(key) : solValues[key] for key in solValues.keys()}

        for k in range(1, K + 1):
            for t in range(1, H + 1):

                edges = []
                for i in range(n + 1):
                    for j in range(n + 1):
                        if i < j:
                            if solValues[(mainStVarName, i, j, k, t)] > 0.5:
                                edges.append((i, j))
                
                subsets = getSubsets(edges, n)


                #if len(edges) > 0:
                #    visualize(edges)


                if len(edges) > 0:
                    for subset in subsets:
                        for element in subset:
                            newCut = Cut()
                            nonzeros = {}
                            nonzeros.update({ '{}_{}_{}_{}_{}'.format(mainStVarName, i, j, k, t) : 1 for i in subset for j in subset if i < j })

                            nonzeros.update({'{}_{}_{}_{}'.format(secondStVarName, i, k, t) : -1 for i in subset if i != element })
                            newCut.nonzero = nonzeros
                            newCut.sense = '<='
                            newCut.rhs = 0
                            cuts.append(newCut)
        return cuts
    return f1

def SubtourElim(model, where):

    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._vars)

        # Necessary (subtour) cuts need to be added.

        n = 10
        H = 3
        K = 2
        newCuts = genSubtourLazy(n, H, K)(vals)

        ## This will be int future method "addCuts" 
        
        senseDict = {
            '<=' : GRB.LESS_EQUAL,
            '==' : GRB.EQUAL,
            '>=' : GRB.GREATER_EQUAL
        }

        for cut in newCuts:
            #print(cut)
            model.cbLazy( quicksum( model._vars[key] * cut.nonzero[key] for key in cut.nonzero.keys() ) , senseDict[cut.sense], cut.rhs )

def getCheckSubTour(n, H, K, stVarName='y', keyOperator = transformKey):
    def checkSubTour(vals):
        vals = { keyOperator(var) : vals[var] for var in vals.keys() if var[0] == stVarName and vals[var] >= 0.001 }

        errorcnt = 0
        for k in range(1, K + 1):
            for t in range(1, H + 1):
                    
                edges = [(key[1], key[2]) for key in vals.keys() if (key[3], key[4]) == (k, t) and key[0] == stVarName]
                if len(edges) > 0:
                    #visualize(edges)
                    subsets = getSubsets(edges, n)
                    if len(subsets) > 0:
                        print(k, t, subsets)
                        print('---------- ERROR! ----------')
                        errorcnt += 1
        
        if errorcnt == 0:
            print('[TEST] SUBTOUR CORRECT MODEL')
            return True
        else:
            print('[TEST] SUBTOUR ERRORS')
            return False
    return checkSubTour


if __name__ == '__main__': pass