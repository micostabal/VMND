import os
import sys
sys.path.append(os.path.pardir)
from gurobipy import Model, quicksum, GRB
from Instance import Instance
import numpy as np
from Neighborhood import genIRPneighborhoods, Neighborhoods
from Cuts import Cut
from VMNDproc import solver

def loadCFL(path):
    instDict = {}
    lines = list(map(lambda x: list(filter(lambda y: y != '', x.replace(' ', '\t').strip('\n').split('\t'))), open(path, 'r').readlines()))
    #lines = list(filter( lambda x: x != [] and x != [''], lines))
    instDict['J'] = int(lines[0][0])
    instDict['I'] = int(lines[0][1])

    restOfRows = []
    for l in lines[1:]:
        restOfRows += l
    
    startD = 0
    startS = startD + instDict['I']
    startF = startS + instDict['J']
    startC = startF + instDict['J']
    end = startC + instDict['I'] * instDict['J']

    d = np.zeros(shape=(instDict['I']))
    s = np.zeros(shape=(instDict['J']))
    f = np.zeros(shape=(instDict['J']))
    c = np.zeros(shape=(instDict['J'], instDict['I']))

    for cnt, i in enumerate(range(startD, startS)):
        d[cnt] = restOfRows[i]
    for cnt, i in enumerate(range(startS, startF)):
        s[cnt] = restOfRows[i]
    for cnt, i in enumerate(range(startF, startC)):
        f[cnt] = restOfRows[i]
    for cnt, i in enumerate(range(startC, end)):
        c[ cnt % instDict['J'] ][ cnt % instDict['I'] ] = restOfRows[i]

    instDict['d'] = d
    instDict['s'] = s
    instDict['f'] = f
    instDict['c'] = c
    return instDict


class CFL(Instance):
 
    def __init__(self, path, instName = ''):
        super().__init__()
        instDict = loadCFL(path)
        self.name = str(os.path.basename(path)).rstrip('.plc')
        self.I = instDict['I']
        self.J = instDict['J']
        self.d = instDict['d']
        self.s = instDict['s']
        self.f = instDict['f']
        self.c = instDict['c']

    def createInstance(self):
        
        model = Model()
        modelVars = {}

        for i in range(self.I):
            for j in range(self.J):
                Vname = 'x_{}_{}'.format(i, j)
                modelVars[Vname] = model.addVar(lb = 0 ,ub = GRB.INFINITY , vtype = GRB.CONTINUOUS, name = Vname)

        for j in range(self.J):
            Vname = 'y_{}'.format(j)
            modelVars[Vname] = model.addVar(vtype = GRB.BINARY, name = Vname)

        # Term 1: Objective Function.

        obj = quicksum( self.c[j][i] * modelVars['x_{}_{}'.format(i, j)] for i in range(self.I) for j in range(self.J) ) +\
         quicksum( self.f[j] * modelVars['y_{}'.format(j)] for j in range(self.J) )

        # Term 2: Capacity Constraint.

        model.addConstrs( quicksum( modelVars['x_{}_{}'.format(i, j) ] for i in range(self.I) )
         <= self.s[j] * modelVars['y_{}'.format(j)] for j in range(self.J) )

        # Term 3: Demand Satisfaction Constraint.
        model.addConstrs( quicksum( modelVars['x_{}_{}'.format(i, j)] for j in range(self.J) ) == self.d[i] for i in range(self.I)  )

        # Term 4: Redundant, tightens formulation.
        model.addConstrs( modelVars['x_{}_{}'.format(i, j)] <= self.d[i] for i in range(self.I) for j in range(self.J))

        # Objective Function is set.
        model.setObjective(obj, GRB.MINIMIZE)       

        model.optimize()
        #return model

    def exportMPS(self, writePath = os.path.join(os.path.pardir, 'MIPLIB')):
        model = self.createInstance()
        self.pathMPS = os.path.join(writePath, self.name + '.mps' )
        model.write(self.pathMPS)

    def genNeighborhoods(self, intervals = 10):
        """intervalsJ = np.array_split(range(self.J), self.J / intervals )

        outer = {
            i + 1 : {
                0 : ['y_{}'.format(intervalsJ[partition][j]) for j in partition for partition in range(intervals) if partition != i]
            }
            for i in range(intervals)
        }

        return Neighborhoods(lowest = 1, highest = 2, keysList=None, randomSet=False, outerNeighborhoods= outer)"""
        return None

    def genLazy(self): pass

    def genTestFunction(self): pass

    def run(self, thisAlpha = 2):
        self.exportMPS()
        modelOut = solver(
            path = self.pathMPS,
            addlazy = False,
            funlazy= None,
            importNeighborhoods= True,
            importedNeighborhoods= None,
            funTest= None,
            alpha = thisAlpha,
            minBCTime = 0,
            callback = 'pure',
            verbose = True
        )
        self.resultVars = {transformKey(var.varName) : var.x for var in modelOut.getVars() if var.x > 0 }
        return modelOut

    def analyzeRes(self): pass

    def visualizeRes(self): pass


if __name__ == '__main__':
    cfl1 = CFL(os.path.join('CFLInstances' , 'p800-4400-61.plc' ))
    cfl1.createInstance()
    #cfl1.run()