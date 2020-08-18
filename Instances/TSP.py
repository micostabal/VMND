from gurobipy import *
import sys
import os
sys.path.append(os.path.pardir)
import random as rd
import numpy as np
from ConComp import getSubsets
from Neighborhood import Neighborhoods
from sklearn.cluster import KMeans
from VMND import solver
from Cuts import Cut
import networkx as nx
import matplotlib.pyplot as plt
from Instance import Instance
from Functions import keyOpTSP, genClusterNeighborhoods

rd.seed(2**10 + 1)


class TSP(Instance):

    def __init__(self, n, nbsRandom = True):
        super().__init__()
        self.n = n
        self.positions = np.zeros(shape=(n, 2))
        for i in range(n):
            self.positions[i][0] = rd.random() * 200
            self.positions[i][1] = rd.random() * 200

        self.E = { (i, j) : np.linalg.norm( np.array(self.positions[i][0], self.positions[i][1]) -
         np.array(self.positions[j][0], self.positions[j][1]) ) for i in range(n) for j in range(n) if i <  j}
        self.name = 'randomTSP{}nodes.mps'.format(self.n)
        self.pathMPS = None
        self.resultVars = None

    def createInstance(self):
        model = Model()
        modelVars = {}
        for key in self.E:
            modelVars['x_{}_{}'.format(key[0], key[1]) ] = model.addVar(vtype = GRB.BINARY, name = 'x_{}_{}'.format(key[0], key[1]))
        model._vars = modelVars

        # Setting objective Function
        obj = quicksum( model._vars['x_{}_{}'.format(i, j)] * self.E[(i, j)] for i in range(self.n) for j in range(self.n) if  i < j)

        # Degree conservation
        model.addConstrs( quicksum( model._vars['x_{}_{}'.format(j, i)] for j in range(self.n) if j < i ) +
        quicksum( model._vars['x_{}_{}'.format(i, j)] for j in range(self.n) if i < j ) == 2 for i in range(self.n) )
        
        model._vars = modelVars
        model.setObjective(obj, GRB.MINIMIZE)
        return model

    def exportMPS(self, writePath = os.path.join(os.path.pardir, 'MIPLIB')):
        model = self.createInstance()
        self.pathMPS = os.path.join( os.path.join( writePath,  self.name ))
        model.write(self.pathMPS)

    def genNeighborhoods(self, setRandom = False, varCluster = False):
        if varCluster:
            numClu = int(self.n / 20) + 1

            outerNbhs = { i : (0,) for i in range(1, numClu + 1) }

            labelsDict = genClusterNeighborhoods( self.pathMPS, numClu, fNbhs = True, varFilter=lambda x: x[0] == 'x')
            def fClusterNbhs(varName, depth, param):
                return labelsDict[varName] != depth - 1

            klist = ['x_{}_{}'.format(i, j) for i in range(self.n) for j in range(self.n) if i < j ]

            return Neighborhoods(
                lowest = 1,
                highest = numClu,
                keysList= klist,
                randomSet=False,
                outerNeighborhoods=outerNbhs,
                useFunction=True,
                funNeighborhoods=fClusterNbhs
                )


        if not setRandom:
            outer = {}
            groups = [5]
            for k in range(1, 2):
                kmeans = KMeans(n_clusters=groups[k - 1], random_state=0).fit(self.positions)
                labels = kmeans.labels_

                params = {}
                for i in range(groups[k - 1]):
                    indexes = [index for index in range(self.n) if labels[index] == i]
                    params[i] = ['x_{}_{}'.format(i, j) for i in indexes for j in indexes if i < j]
                    
                outer[k] = params
            nNbh = Neighborhoods(1, 2, keysList=None, randomSet=False, outerNeighborhoods= outer)
            return nNbh
        else:
            nNbh = Neighborhoods(1, 4, keysList=['x_{}_{}'.format(i, j) for i in range(self.n) for j in range(self.n) if i < j],
             randomSet=True, outerNeighborhoods= None)
            return nNbh

    def genLazy(self):
        def f1(solValues):
            cuts = []
            solValues = { (key.split('_')[0], int(key.split('_')[1]), int(key.split('_')[2]) ) : solValues[key] for key in solValues.keys()}
            
            
            edges = []
            for i in range(self.n):
                for j in range(self.n):
                    if i < j:
                        if solValues[('x', i, j)] > 0.5:
                            edges.append((i, j))

            subsets = getSubsets(edges, self.n)

            if len(edges) > 0:
                for subset in subsets:
                    newCut = Cut()
                    nonzeros = {}
                    nonzeros.update({ 'x_{}_{}'.format(i, j) : 1 for i in subset for j in subset if i < j })
                    newCut.nonzero = nonzeros
                    newCut.sense = '<='
                    newCut.rhs = len(subset) - 1
                    cuts.append(newCut)
            return cuts
        return f1

    def genTestFunction(self):
        def checkSubTour(vals):
            vals = { keyOpTSP(var) : vals[var] for var in vals.keys() if var[0] == 'x' and vals[var] > 0 }

            errorcnt = 0
            
            edges = [(key[1], key[2]) for key in vals.keys() if key[0] == 'x']
            if len(edges) > 0:
                #visualize(edges)
                subsets = getSubsets(edges, self.n)
                if len(subsets) > 0:
                    print(subsets)
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
        mout = solver (
                        self.pathMPS,
                        addlazy= True,
                        funlazy= self.genLazy(),
                        importNeighborhoods= True,
                        importedNeighborhoods= self.genNeighborhoods( varCluster=True ),
                        funTest = self.genTestFunction(),
                        alpha = 1,
                        callback = 'vmnd',
                        verbose = True,
                        plotGapsTime = False,
                        writeTestLog = True
                )

        self.resultVars = {keyOpTSP(var.varName) : var.x for var in mout.getVars() if var.x > 0 }
        return mout

    def analyzeRes(self): pass

    def visualizeRes(self):
        if self.resultVars is None:
            print('The model must be run first before visualizing results! Execute first the run method')
            return 0
        outRoutes = {key : self.resultVars[key] for key in self.resultVars.keys() if self.resultVars[key] >= 1}

        edges = [(key[1], key[2]) for key in outRoutes.keys()]

        if len(edges) == 0:
            return 0
        pos =  {i : (self.positions[i][0], self.positions[i][1]) for i in range(self.n) }

        G_1 = nx.Graph()
        G_1.add_edges_from(edges)

        nx.draw(G_1, pos, edge_labels = True, with_labels=True, font_weight='bold')
        plt.show()


if __name__ == '__main__':
    tspExp = TSP(76)
    tspExp.run()
    tspExp.visualizeRes()

