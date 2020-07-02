import os
import sys
sys.path.append(os.path.pardir)
import numpy as np
from Instance import Instance
from gurobipy import Model, GRB, quicksum
from Neighborhood import genIRPneighborhoods, Neighborhoods
from Functions import genClusterNeighborhoods
from VMNDproc import solver


class Event:

    def __init__(self, eid, start, time):
        self.id = eid
        self.time = time
        self.start = start # true or false

    def __repr__(self):
        return 'Id: {}, Time: {}, Start: {}'.format(self.id, self.time, self.start)

    def __lt__(self, other):
        if self.time == other.time:
            return self.start == False and other.start
        return self.time < other.time

    def __eq__(self, other):
        return self.time == other.time and self.start == other.start


def procedure(eventList):
    # 1 : Initialization

    maximalSubsets = {}
    eventList = sorted(eventList)
    
    inc = 0
    S = set()
    k = 0

    # 2 : Iteration

    for h in range(len(eventList)):
        e_h = eventList[h]

        if e_h.start:
            S.add(e_h.id)
            inc = 0
        else:
            if inc == 0:
                k += 1
                maximalSubsets[k] = tuple(S)
                inc = 1
            S.remove(e_h.id)

    # 3 : Evaluation
    return maximalSubsets

def loadOISRC(path):
    outdict = {}
    lines = list(map(lambda x: list(filter(lambda y: y != '', x.replace(' ', '\t').strip('\n').split('\t'))), open(path, 'r').readlines()))
    outdict['m'] = int(lines[0][0])
    outdict['R'] = int(lines[1][0])
    outdict['n'] = int(lines[2][0])
    outdict['st'] = np.zeros(shape = (outdict['n']))
    outdict['ft'] = np.zeros(shape = (outdict['n']))
    outdict['r'] = np.zeros(shape = (outdict['n']))
    outdict['v'] = np.zeros(shape = (outdict['n']))
    for line in lines[3:]:
        id = int(line[0])
        outdict['st'][id] = int(line[1])
        outdict['ft'][id] = int(line[2])
        outdict['r'][id] = int(line[3])
        outdict['v'][id] = int(line[4])

    eventList = []
    for i in range(outdict['n']):
        eventList.append( Event( i, True, outdict['st'][i] ) )
        eventList.append( Event( i, False, outdict['ft'][i] ) )
    outdict['J'] = procedure(eventList)
    outdict['k'] = len( outdict['J'].keys() )


    return outdict


class OISRC(Instance):

    def __init__(self, path):
        super().__init__()
        instDict = loadOISRC(path)
        self.name = str(os.path.basename(path)).rstrip('.oisrc')
        self.pathMPS = None
        self.outputvars = None
        self.n = instDict['n']
        self.R = instDict['R']
        self.m = instDict['m']
        self.r = instDict['r']
        self.st = instDict['st']
        self.et = instDict['ft']
        self.v = instDict['v']
        self.J = instDict['J']
        self.k = instDict['k']
        
    def createInstance(self):

        model = Model()
        modelVars = {}

        # x variables are assigned.
        for i in range(self.m):
            for j in range(self.n):
                Vname = 'x_{}_{}'.format(i, j)
                modelVars[Vname] = model.addVar( vtype = GRB.BINARY, name = Vname)
        
        # Term 2: Objective Function
        obj = quicksum( self.v[j] * modelVars['x_{}_{}'.format(i, j)] for i in range(self.m) for j in range(self.n) )

        # Term 3: Each job is assigned to at most 1 machine.
        model.addConstrs( quicksum(modelVars['x_{}_{}'.format(i, j)] for i in range(self.m)) <= 1 for j in range(self.n) )

        # Term 4: Allows the assignment of at most R resources to a machine.
        model.addConstrs( quicksum( modelVars['x_{}_{}'.format(i, j)] * self.r[j] for j in self.J[h] ) <= self.R
         for i in range(self.m) for h in range(1, self.k + 1) )

        # Term 5: Implicit in varaible definition.

        # Objective Function is set.
        model._vars = modelVars
        model.setObjective(obj, GRB.MAXIMIZE)

        #print(model.getAttr('ModelSense') == GRB.MINIMIZE)

        return model

    def exportMPS(self, writePath = os.path.join(os.path.pardir, 'MIPLIB')):
        model = self.createInstance()
        self.pathMPS = os.path.join(writePath, self.name + '.mps' )
        model.write(self.pathMPS)

    def genNeighborhoods(self, verbose = False):
        
        numClu = min(int(self.m * self.n / 10), 20)
        print(numClu)

        labelsDict = genClusterNeighborhoods( self.pathMPS, numClu, fNbhs = True, varFilter=lambda x: x[0] == 'x')
        def fClusterNbhs(varName, depth, param):
            return labelsDict[varName] != depth - 1          

        
        outerNbhs = { i : (0,) for i in range(1, numClu + 1) }

        klist = ['x_{}_{}'.format(i, j) for i in range(self.m) for j in range(self.n) ]

        return Neighborhoods(
            lowest = 1,
            highest = numClu,
            keysList= klist,
            randomSet=False,
            outerNeighborhoods=outerNbhs,
            useFunction=True,
            funNeighborhoods=fClusterNbhs
            )

    def genLazy(self): pass

    def genTestFunction(self): pass

    def run(self):
        self.exportMPS()

        nbhs = self.genNeighborhoods()

        exModel = solver(
            self.pathMPS,
            verbose = True,
            addlazy= False, 
            funlazy= None,
            importNeighborhoods= True, 
            importedNeighborhoods= nbhs,
            funTest= None, 
            callback = 'vmnd',
            alpha = 1,
            minBCTime= 0
        )
        self.outputvars = {var.varName : var.x for var in exModel.getVars() if var.x > 0 }
        return exModel

    def analyzeRes(self): pass

    def visualizeRes(self): pass


if __name__ == '__main__':
    si1 = OISRC(os.path.join('OISRCInstances', 'instance_15_2_170_1.oisrc'))

    si1.run()