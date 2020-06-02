import os
import sys
import pathlib
sys.path.append(os.path.pardir)
import numpy as np
from gurobipy import *
import networkx as nx
import matplotlib.pyplot as plt
from Cuts import Cut
from ConComp import getSubsets 
from VMNDproc import solver
from Neighborhood import Neighborhoods
from Functions import keyOpVRP
from Instance import Instance

def loadVRP(fileLines):
    outinstance = {}
    nodes, mintrucks = list(map( lambda x : int(x.strip('n').strip('k')), fileLines[0].split(' ')[-1].split('-')[1:] ) )
    outinstance['nodes'] = nodes - 1 # We substract 1 since the parameters is the number of "retailers"
    outinstance['trucks'] = mintrucks
    outinstance['capacity'] = int(fileLines[5].split(':')[1].strip(' '))
    outinstance['positions'] = np.zeros(shape=(nodes, 2))
    outinstance['demands'] = np.zeros(shape=(nodes))
    outinstance['optimum'] = float(fileLines[1].split(':')[-1].strip(')').strip(' '))
    outinstance['name'] = fileLines[0].split(':')[-1].strip(' ')

    for linenumber in range(7, 7 + nodes):
        ind = linenumber - 7
        outinstance['positions'][ind][0] = int(fileLines[linenumber].split(' ')[2])
        outinstance['positions'][ind][1] = int(fileLines[linenumber].split(' ')[3])
    for linenumber in range(8 + nodes, 8 + 2 * nodes):
        ind = linenumber - 8 - nodes
        outinstance['demands'][ind] = int(fileLines[linenumber].split(' ')[1])
    return outinstance


class VRP(Instance):

    def __init__(self, path):
        super().__init__()
        fileLines = list(map(lambda x: x.strip('\n'), open( path , 'r').readlines()))
        instDict = loadVRP(fileLines)
        self.path = path
        self.pathMPS = None
        self.name = instDict['name']
        self.retailers = instDict['nodes'] - 1
        self.totalNodes = instDict['nodes']
        self.capacity = instDict['capacity']
        self.trucks = instDict['trucks']
        self.positions = instDict['positions']
        self.demands = instDict['demands']
        self.cost = np.zeros(shape = (self.totalNodes, self.totalNodes))
        for j in range(1, self.totalNodes):
            for i in range(j):
                self.cost[i][j] = np.linalg.norm(np.array([ self.positions[i][0] , self.positions[i][1] ])
                 - np.array([ self.positions[j][0] , self.positions[j][1] ]))

    def createInstance(self):
        model = Model()

        # We create variables
        modelVars = {}

        for k in range(1, self.trucks + 1):
            for j in range(self.totalNodes):
                for i in range(j):
                    modelVars['y_{}_{}_{}'.format(i, j, k)] = model.addVar(vtype = GRB.BINARY, name = 'y_{}_{}_{}'.format(i, j, k))

            for i in range(self.totalNodes):
                modelVars['z_{}_{}'.format(i, k)] = model.addVar(vtype = GRB.BINARY, name = 'z_{}_{}'.format(i, k))

        model._vars = modelVars

        # Declaration of Objective Function.
        obj = quicksum( model._vars['y_{}_{}_{}'.format(i, j, k)]
         for k in range(1, self.trucks + 1) for i in range(self.totalNodes) for j in range(self.totalNodes) if i < j )

        # Term 1: Truck capacity constraint.
        model.addConstrs( quicksum( model._vars['z_{}_{}'.format(i, k)] * self.demands[i]
         for i in range(1, self.totalNodes) ) <= self.capacity for k in range(1, self.trucks + 1) )
        
        # Term 2: Every location must be visited.
        model.addConstrs( quicksum( model._vars['z_{}_{}'.format(i, k)]
         for k in range(1, self.trucks + 1)) == 1 for i in range(1, self.totalNodes) )

        # Term 4: Degree Conservation
        model.addConstrs( quicksum( model._vars['y_{}_{}_{}'.format(i, j, k)] for j in range(self.totalNodes) if i < j) + 
         quicksum( model._vars['y_{}_{}_{}'.format(j, i, k)] for j in range(self.totalNodes) if j < i)
         == 2 *model._vars['z_{}_{}'.format(i, k)]
         for k in range(1, self.trucks + 1) for i in range(1, self.totalNodes) )

        # Term 5 (Ommited): Subtour Elimination

        # Symmetry Breaking Constrints

        model.addConstrs( modelVars['y_{}_{}_{}'.format(i, j, k)] <= modelVars['z_{}_{}'.format(i, k)]
         for i in range(self.totalNodes) for j in range(self.totalNodes) for k in range(1, self.trucks + 1) if i < j )

        # Term 29: Symmetry Breaking Constraint

        model.addConstrs( modelVars['y_{}_{}_{}'.format(i, j, k)] <= modelVars['z_{}_{}'.format(j, k)]
         for i in range(self.totalNodes) for j in range(self.totalNodes) for k in range(1, self.trucks + 1) if i < j )

        # Term 30: Symmetry Breaking Constraint

        model.addConstrs( modelVars['y_{}_{}_{}'.format(0, i, k)] <= 2 * modelVars['z_{}_{}'.format(i, k)]
         for i in range(1, self.totalNodes) for k in range(1, self.trucks + 1) )

        # Term 31: Symmetry Breaking Constraint

        model.addConstrs( modelVars['z_{}_{}'.format(i, k)] <= modelVars['z_{}_{}'.format(0, k)]
         for i in range(1, self.totalNodes) for k in range(1, self.trucks + 1) )

        # Term 32: Symmetry Breaking Constraint

        model.addConstrs( modelVars['z_{}_{}'.format(0, k)] <= modelVars['z_{}_{}'.format(0, k - 1)]
         for k in range(2, self.trucks + 1) )

        # Term 33: Symmetry Breaking Constraint

        model.addConstrs( modelVars['z_{}_{}'.format(i, k)] <= quicksum( modelVars['z_{}_{}'.format(j, k - 1)]
         for j in range(self.totalNodes) if j < i )
         for i in range(1, self.totalNodes) for k in range(2, self.trucks + 1) )

        # Objective Function is set
        model.setObjective(obj, GRB.MINIMIZE)

        return model

    def exportMPS(self, writePath = os.path.join(os.path.pardir, 'MIPLIB')):
        m = self.createInstance()
        self.pathMPS = os.path.join( writePath , self.name + '.mps' )
        m.write(self.pathMPS)

    def genNeighborhoods(self):
        outerN = {
            1: {
                truck : [ 'y_{}_{}_{}'.format(i, j, k) for k in range(1, self.trucks + 1)
                 for i in range(self.totalNodes) for j in range(self.totalNodes)
                 if i < j and k != truck ] for truck in range(1, self.trucks + 1)
            },
            2: {
                (tr1, tr2) : [ 'y_{}_{}_{}'.format(i, j, k) for k in range(1, self.trucks + 1)
                 for i in range(self.totalNodes) for j in range(self.totalNodes)
                 if i < j and k != tr1 and k != tr2 ] for tr1 in range(1, self.trucks + 1) for tr2 in range(1, self.trucks + 1) if tr1 < tr2
            }
        }
        return Neighborhoods(lowest = 1, highest = 2, keysList = None, randomSet = False, outerNeighborhoods = outerN)

    def genLazy(self):
        mainStVarName = 'y'
        secondStVarName = 'z'
        def f1(solValues):
            cuts = []
            solValues = { keyOpVRP(key) : solValues[key] for key in solValues.keys()}

            for k in range(1, self.trucks + 1):
                edges = []
                for i in range(self.totalNodes):
                    for j in range(self.totalNodes):
                        if i < j:
                            if solValues[(mainStVarName, i, j, k)] > 0.5:
                                edges.append((i, j))

                subsets = getSubsets(edges, self.totalNodes)

                if len(edges) > 0:
                    for subset in subsets:
                        for element in subset:
                            newCut = Cut()
                            nonzeros = {}
                            nonzeros.update({ '{}_{}_{}_{}'.format(mainStVarName, i, j, k) : 1 for i in subset for j in subset if i < j })

                            nonzeros.update({'{}_{}_{}'.format(secondStVarName, i, k) : -1 for i in subset if i != element })
                            newCut.nonzero = nonzeros
                            newCut.sense = '<='
                            newCut.rhs = 0
                            cuts.append(newCut)
            return cuts
        return f1

    def genTestFunction(self):
        def checkSubTour(vals):
            vals = { keyOpVRP(var) : vals[var] for var in vals.keys() if var[0] == 'y' and vals[var] > 0 }

            errorcnt = 0
            for k in range(1, self.trucks + 1):
                        
                edges = [(key[1], key[2]) for key in vals.keys() if key[3] == k and key[0] == 'y']
                if len(edges) > 0:
                    #visualize(edges)
                    subsets = getSubsets(edges, self.totalNodes)
                    if len(subsets) > 0:
                        print(k, subsets)
                        print('---------- ERROR! ----------')
                        errorcnt += 1
            
            if errorcnt == 0:
                print('[TEST] SUBTOUR CORRECT MODEL')
                return True
            else:
                print('[TEST] SUBTOUR ERRORS')
            return False
        return checkSubTour

    def run(self):
        self.exportMPS()
        VRPmod = solver(
            self.pathMPS,
            addlazy= True,
            funlazy= self.genLazy(),
            importNeighborhoods= True,
            importedNeighborhoods= self.genNeighborhoods(),
            funTest= None,
            alpha = 2,
            callback = 'vmnd',
            verbose = True
        )
        return VRPmod

    def analyzeRes(self): pass

    def visualizeRes(self): pass


if __name__ == '__main__':
    vrp1 = VRP(os.path.join('VRP Instances', 'A-n33-k5.vrp'))
    vrp1.run()