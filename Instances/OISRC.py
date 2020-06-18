import os
import numpy as np
from Instance import Instance
from gurobipy import Model, GRB, quicksum

def loadOISRC(path):
    outdict = {}
    lines = list(map(lambda x: list(filter(lambda y: y != '', x.replace(' ', '\t').strip('\n').split('\t'))), open(path, 'r').readlines()))
    outdict['m'] = int(lines[0][0])
    outdict['R'] = int(lines[1][0])
    outdict['n'] = int(lines[2][0])
    outdict['st'] = np.zeros(shape = (outdict['n']))
    outdict['et'] = np.zeros(shape = (outdict['n']))
    outdict['r'] = np.zeros(shape = (outdict['n']))
    outdict['v'] = np.zeros(shape = (outdict['n']))
    for line in lines[3:]:
        id = int(line[0])
        outdict['st'][id] = int(line[1])
        outdict['et'][id] = int(line[2])
        outdict['r'][id] = int(line[3])
        outdict['v'][id] = int(line[4])
    return outdict


class OISRC(Instance):

    def __init__(self, path):
        super().__init__()
        instDict = loadOISRC(path)
        self.n = instDict['n']
        self.R = instDict['R']
        self.m = instDict['m']
        self.r = instDict['r']
        self.st = instDict['st']
        self.et = instDict['et']
        self.v = instDict['v']
        
    def createInstance(self):

        model = Model()
        modelVars = {}

        # x variables are assigned.
        for i in range(self.m):
            for j in range(self.n):
                Vname = 'x_{}_{}'.format(i, j)
                modelVars[Vname] = model.addVar( vtype = GRB.BINARY, name = Vname)
        
        # Term 2: Objective Function
        obj = quicksum( self.v[j] * modelVars['x_{}_{}'.format(i, j)] for i in range(self.n) for j in range(self.m) )

        # Term 3: Each job is assigned to at most 1 machine.
        model.addConstrs( quicksum(modelVars['x_{}_{}'.format(i, j)] for i in range(self.m)) <= 1 for j in range(self.n) )

        # Term 4: Allows the assignment of at most R resources to a machine.
        model.addConstrs( for i in range(self.m) for h in range() )

        # Term 5: Implicit in varaible definition.

        # Objective Function is set.
        model._vars = modelVars
        model.setObjective(obj, GRB.MINIMIZE)
        return model

        return model

    def exportMPS(self, writePath = os.path.join(os.path.pardir, 'MIPLIB')): pass

    def genNeighborhoods(self): pass

    def genLazy(self): pass

    def genTestFunction(self): pass

    def run(self): pass

    def analyzeRes(self): pass

    def visualizeRes(self): pass


if __name__ == '__main__':
    loadOISRC(os.path.join('OISRCInstances', 'instance_4_2_100_1.oisrc'))