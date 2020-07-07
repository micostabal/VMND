from gurobipy import *
import os
import sys
sys.path.append(os.path.pardir)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ConComp import getSubsets
from Neighborhood import genIRPneighborhoods, Neighborhoods
from Functions import transformKey, genClusterNeighborhoods
from Cuts import Cut
from VMNDproc import solver
from Instance import Instance
import time

def loadIRP(path='abs1n20_2.dat', sep='\t'):
    output = {
    'n': 0,
    'H': 0,
    'C' : 0,
    }

    # The file is opened and parsed line by line
    file = open(os.path.join('IRPInstances', path), 'r')
    listLines = list(map(lambda x: x.strip('\n'), file.readlines()))

    # First line contains n, h and C
    args0 = list(filter(lambda x: x != '', listLines[0].split(sep)))
    output['n'] = int(args0[0]) - 1
    output['H'] = int(args0[1])
    output['C'] = int(args0[2])

    # Second line contains Supplier information
    args1 = list(filter(lambda x: x != '', listLines[1].split(sep)))
    positions = np.zeros(shape=(output['n'] + 1, 2))
    positions[0, 0] = args1[1]
    positions[0, 1] = args1[2]


    output['r'] = [0 for i in range(1, output['n'] + 2)]
    output['h'] = [0 for i in range(1, output['n'] + 2)]
    output['U'] = [0 for i in range(1, output['n'] + 2)]
    output['I0'] = [0 for i in range(1, output['n'] + 2)]

    output['I0'][0] = float(args1[3])

    output['r'][0] = float(args1[4])
    output['h'][0] = float(args1[5])

    # The retailers are read and parsed accordingly
    for i, line in enumerate(listLines[2:]):
        ind = i + 1
        args = list(filter(lambda x: x != '', line.split(sep)))
        positions[ind][0] = args[1]
        positions[ind][1] = args[2]
        output['I0'][ind] = float(args[3])
        output['U'][ind] = float(args[4])
        output['r'][ind] = float(args[6])
        output['h'][ind] = float(args[7])

    dist = np.zeros(shape=(output['n'] + 1, output['n'] + 1))
    for i in range(output['n'] + 1):
        for j in range(output['n'] + 1):
            if (i < j):
                dist[i, j] = np.linalg.norm( np.array( positions[i, 0] , positions[i, 1] ) -
                 np.array( positions[j, 0] , positions[j, 1] ) )
                dist[j, i] = dist[i, j]

    output['dist'] = dist
    output['positions'] = positions
    return output


class IRP(Instance):

    def __init__(self, path = 'abs1n20_2.dat'):
        super().__init__()
        self.name = path
        self.pathMPS = None
        dictinstance = loadIRP(path)
        self.n = dictinstance['n']
        self.H = dictinstance['H']
        self.r = dictinstance['r']
        self.I0 = dictinstance['I0']
        self.U =  dictinstance['U']
        self.h = dictinstance['h']
        self.C = dictinstance['C']
        self.positions = dictinstance['positions']
        self.K = 8
        self.dist = dictinstance['dist']
        self.resultVars = None

    def createInstance(self):
        model = Model()
        modelVars = {}

        # We create y and z varaibles
        for k in range(1, self.K + 1):
            for t in range(1, self.H + 1):

                for i in range(self.n + 1):
                    for j in range(self.n + 1):
                        if i < j:
                            if (i != j and i > 0):
                                modelVars['y_{}_{}_{}_{}'.format(i, j, k, t)] = model.addVar(vtype=GRB.BINARY,
                                name='y_{}_{}_{}_{}'.format(i, j, k, t))
                            elif i != j and i == 0:
                                modelVars['y_{}_{}_{}_{}'.format(i, j, k, t)] = model.addVar(lb = 0, ub = 2, vtype=GRB.INTEGER,
                                name='y_{}_{}_{}_{}'.format(i, j, k, t))


                for i in range(self.n + 1):
                    modelVars['z_{}_{}_{}'.format(i, k, t)] = model.addVar(vtype=GRB.BINARY,
                    name='z_{}_{}_{}'.format(i, k, t))

        # Variables I and q are created.
        for t in range(1, self.H + 1):
            for i in range(self.n + 1):
                modelVars[ 'I_{}_{}'.format(i, t) ] = model.addVar(lb = 0, vtype = GRB.CONTINUOUS, name='I_{}_{}'.format(i, t))

                for k in range(1, self.K + 1):
                    modelVars['q_{}_{}_{}'.format(i, k, t)] = model.addVar(lb = 0, vtype = GRB.CONTINUOUS, name='q_{}_{}_{}'.format(i, k, t))

        # Inventory at t = 0.
        modelVars['I_0_0'] = model.addVar(lb = self.I0[0], ub = self.I0[0], vtype = GRB.CONTINUOUS, name='I_0_0')
        for i in range(1, self.n + 1):
            modelVars[ 'I_{}_{}'.format(i, 0) ] = model.addVar(lb = self.I0[i], ub = self.I0[i], vtype = GRB.CONTINUOUS, name='I_{}_{}'.format(i, 0))

        # Term a) Objective Function
        obj = quicksum(self.h[0] * modelVars['I_{}_{}'.format(0, t)] for t in range(1, self.H + 1)) +\
            quicksum( self.h[i] * modelVars['I_{}_{}'.format(i, t)]  for t in range(1, self.H + 1) for i in range(1, self.n + 1)) +\
            quicksum( modelVars['y_{}_{}_{}_{}'.format(i, j, k, t)] * self.dist[i, j] * 0 for k in range(1, self.K + 1) for i in range(self.n + 1) for j in range(self.n + 1)\
            for t in range(1, self.H + 1) for i in range(self.n + 1) if i < j )

        # Term b) Inventory at depot.
        model.addConstrs( modelVars['I_{}_{}'.format(0, t)] == modelVars['I_{}_{}'.format(0, t - 1)] + self.r[0]
        - quicksum( modelVars['q_{}_{}_{}'.format(i, k, t)] for i in range(1, self.n + 1) for k in range(1, self.K + 1))
        for t in range(1, self.H + 1))


        # Term c) Inventory at N' location.
        model.addConstrs( modelVars['I_{}_{}'.format(i, t)] == modelVars['I_{}_{}'.format(i, t - 1)] - self.r[i]
        + quicksum( modelVars['q_{}_{}_{}'.format(i, k, t)] for k in range(1, self.K + 1))
        for t in range(1, self.H + 1) for i in range(1, self.n + 1))

        # Term d) Force absence of stockouts
        model.addConstrs( modelVars['I_{}_{}'.format(i, t)] >= 0 for i in range(self.n + 1) for t in range(1, self.H + 1))

        # Term e) Maximum inventory is not exceeded.
        model.addConstrs( quicksum(modelVars['q_{}_{}_{}'.format(i, k, t)] for k in range(1, self.K + 1)) <= self.U[i] - modelVars['I_{}_{}'.format(i, t - 1)]
        for i in range(1, self.n + 1) for t in range(1, self.H + 1))

        # Term f) Qty cannot be transfred if location is not visited.
        model.addConstrs( modelVars['q_{}_{}_{}'.format(i, k, t)] <= self.U[i] * modelVars['z_{}_{}_{}'.format(i, k, t)]
        for i in range(1, self.n + 1) for t in range(1, self.H + 1) for k in range(1, self.K + 1) )

        # Term g) Vehicle capacity constraint.
        model.addConstrs( quicksum(modelVars['q_{}_{}_{}'.format(i, k, t)] for i in range(1, self.n + 1)) <= self.C * modelVars['z_{}_{}_{}'.format(0, k, t)]
        for t in range(1, self.H + 1) for k in range(1, self.K + 1) )


        # Term h) Location cannot be visited by more than one vehicle per period.
        model.addConstrs( quicksum(modelVars['z_{}_{}_{}'.format(i, k, t)] for k in range(1, self.K + 1)) <= 1
        for t in range(1, self.H + 1) for i in range(1, self.n + 1) )

        # Term i) Node degree constraint.
        model.addConstrs( quicksum(modelVars['y_{}_{}_{}_{}'.format(i, j, k, t)] for j in range(self.n + 1) if i < j)
        + quicksum(modelVars['y_{}_{}_{}_{}'.format(j, i, k, t)] for j in range(self.n + 1) if j < i ) == 2 * modelVars['z_{}_{}_{}'.format(i, k, t)]
        for t in range(1, self.H + 1) for i in range(self.n + 1) for k in range(1, self.K + 1) )

        # Term j) Subtour elimination (Commented because of its LazyConstraint nature).

        # Term k) Symmetry Breaking.
        model.addConstrs( modelVars['z_{}_{}_{}'.format(0, k, t)] >= modelVars['z_{}_{}_{}'.format(0, k + 1, t)]
         for t in range(1, self.H + 1) for k in range(1, self.K))

        # Term l) Symmetry Breaking.
        model.addConstrs( quicksum( 2**(j - i) * modelVars['z_{}_{}_{}'.format(i, k, t)] for i in range(1, j + 1))
        >= quicksum( 2**(j - i) * modelVars['z_{}_{}_{}'.format(i, k + 1, t)] for i in range(1, j + 1))
        for j in range(1, self.n + 1) for t in range(1, self.H + 1) for k in range(1, self.K))

        # terms m), n), o) and p) are already incluided in the var. definition.
        
        model._vars = modelVars

        model.setObjective(obj, GRB.MINIMIZE)
        return model

    def exportMPS(self, writePath = os.path.join(os.path.pardir, 'MIPLIB'), writeName = ''):
        if len(writeName) == 0:
            writeName = self.name.strip('.dat')
        model = self.createInstance()
        self.pathMPS = os.path.join(writePath, writeName + '.mps' )
        model.write(self.pathMPS)
        
    def genNeighborhoods(self, varCluster = False, nCltrs = 30, funNbhs = False):
        if funNbhs:
            def fNbhs(varName, depth, param):
                elements = varName.split('_')
                if len(elements) < 5:
                    return False
                else:
                    kl = int(elements[3])
                    tl = int(elements[4])

                    if depth == 2:
                        return (kl, tl) != param
                    elif depth == 3:
                        return tl != param
                    elif depth == 4:
                        return kl != param
                    elif depth == 5:
                        return tl != param[0] and tl != param[1]
                    else:
                        print('Error 23 Nbhds Function!! ')
                        return 0
                return False

            outer = {
                2 : tuple([ (kf, tf) for kf in range(1, self.K + 1) for tf in range(1, self.H + 1) ]),
                3 : tuple([ tf for tf in range(1, self.H + 1) ]),
                4 : tuple([ kf for kf in range(1, self.K + 1) ]), 
                5 : tuple([ (tf1, tf2) for tf1 in range(1, self.H  + 1) for tf2 in range(1, self.H  + 1) if tf1 < tf2 ])
            }

            klist = ['y_{}_{}_{}_{}'.format( i, j, k, t )
             for k in range(1, self.K + 1) for t in range(1, self.H  + 1) for i in range(self.n + 1) for j in range(self.n + 1) if i < j]
            return Neighborhoods(
                lowest = 2,
                highest = 5,
                keysList= klist,
                randomSet=False,
                outerNeighborhoods=outer,
                funNeighborhoods= fNbhs,
                useFunction=True)

        if varCluster:

            amountClusters = self.H * self.K

            outerNbhs = { i : (0,) for i in range(1, amountClusters + 1) }

            labelsDict = genClusterNeighborhoods( self.pathMPS, amountClusters, fNbhs = True, varFilter=lambda x: x[0] == 'y')
            def fClusterNbhs(varName, depth, param):
                return labelsDict[varName] == depth

            klist = ['y_{}_{}_{}_{}'.format( i, j, k, t )
             for k in range(1, self.K + 1) for t in range(1, self.H  + 1) for i in range(self.n + 1) for j in range(self.n + 1) if i < j]

            return Neighborhoods(
                lowest = 1,
                highest = amountClusters,
                keysList= klist,
                randomSet = False,
                outerNeighborhoods = outerNbhs,
                useFunction= True,
                funNeighborhoods= fClusterNbhs
            )

        outerNhs =  {
        2 : {(kf, tf) : tuple([ '{}_{}_{}_{}_{}'.format('y', i, j, k, t)
            for k in range(1, self.K + 1) for t in range(1, self.H  + 1) for i in range(self.n + 1) for j in range(self.n + 1)
            if i < j and (k, t) != (kf, tf) ])
                for kf in range(1, self.K + 1) for tf in range(1, self.H + 1)
        },
        3: {
            tf : tuple([ '{}_{}_{}_{}_{}'.format('y', i, j, k, t)
            for k in range(1, self.K + 1) for t in range(1, self.H  + 1) for i in range(self.n + 1) for j in range(self.n + 1)
            if i < j and t != tf ])
            for tf in range(1, self.H + 1)
        },
        
        4: {
            kf : tuple([ '{}_{}_{}_{}_{}'.format('y', i, j, k, t)
            for k in range(1, self.K + 1) for t in range(1, self.H  + 1) for i in range(self.n + 1) for j in range(self.n + 1)
            if i < j and k != kf ])
            for kf in range(1, self.K + 1)
        },
        5: {
            (tf1, tf2) : tuple([ '{}_{}_{}_{}_{}'.format('y', i, j, k, t)
            for k in range(1, self.K + 1) for t in range(1, self.H  + 1) for i in range(self.n + 1) for j in range(self.n + 1)
            if i < j and tf1 != t and tf2 != t ])
            for tf1 in range(1, self.H  + 1) for tf2 in range(1, self.H  + 1) if tf1 < tf2
        }
        }
        return Neighborhoods(lowest = 2, highest = 5, keysList = None, randomSet = False, outerNeighborhoods = outerNhs, useFunction=False)

    def genLazy(self):
        def f1(solValues):
            cuts = []
            solValues = { transformKey(key) : solValues[key] for key in solValues.keys()}

            mainStVarName = 'y'
            secondStVarName = 'z'

            for k in range(1, self.K + 1):
                for t in range(1, self.H + 1):

                    edges = []
                    for i in range(self.n + 1):
                        for j in range(self.n + 1):
                            if i < j:
                                if solValues[(mainStVarName, i, j, k, t)] > 0.5:
                                    edges.append((i, j))
                    
                    subsets = getSubsets(edges, self.n)

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

    def genTestFunction(self):
        def checkSubTour(vals):
            vals = { transformKey(var) : vals[var] for var in vals.keys() if var[0] == 'y' and vals[var] >= 0.99 }

            errorcnt = 0
            for k in range(1, self.K + 1):
                for t in range(1, self.H + 1):
                        
                    edges = [(key[1], key[2]) for key in vals.keys() if (key[3], key[4]) == (k, t) and key[0] == 'y']
                    if len(edges) > 0:
                        #visualize(edges)
                        subsets = getSubsets(edges, self.n)
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

    def run(
        self,
        outImportNeighborhoods = True,
        outImportedNeighborhoods = 'function',
        outFunTest = None,
        outAlpha = 1,
        outCallback = 'vmnd',
        outVerbose = True,
        outMinBCTime = 0,
        outTimeLimitSeconds = 7200,
        writeResult = True):
        self.exportMPS()

        if outImportedNeighborhoods is not 'cluster':
            modelOut = solver(
                path = self.pathMPS,
                addlazy = True,
                funlazy= self.genLazy(),
                importNeighborhoods= True,
                importedNeighborhoods= self.genNeighborhoods(funNbhs = True, varCluster=False),
                funTest= self.genTestFunction(),
                alpha = outAlpha,
                minBCTime = outMinBCTime,
                callback = outCallback,
                verbose = outVerbose
            )
        elif outImportedNeighborhoods == 'cluster':
            modelOut = solver(
                path = self.pathMPS,
                addlazy = True,
                funlazy= self.genLazy(),
                importNeighborhoods= True,
                importedNeighborhoods= self.genNeighborhoods(varCluster = True),
                funTest= self.genTestFunction(),
                alpha = outAlpha,
                minBCTime = 0,
                callback = outCallback,
                verbose = True
            )
        else:
            print('Error')
            return 0

        if writeResult:
            file = open(os.path.join( os.path.pardir, 'Results' , 'results.txt'), 'a')
            line = self.name
            if modelOut.status == GRB.OPTIMAL or modelOut.status == GRB.TIME_LIMIT :
                if outCallback == 'vmnd':
                    line += modelOut._line + '-{}-'.format(outImportedNeighborhoods) + '--MIPGAP: {}--'.format(round(modelOut.MIPGap, 3)) + '\n'
                else:
                    line += modelOut._line + '-{}-'.format('-pureB&C-') + '--MIPGAP: {}--'.format(round(modelOut.MIPGap, 3)) + '\n'
             
            else:
                line += ' Feasable solution was not found. ' + '\n'
            file.write(line)
            file.close()



        self.resultVars = {transformKey(var.varName) : var.x for var in modelOut.getVars() if var.x > 0 }
        return modelOut

    def analyzeRes(self): pass

    def visualizeRes(self):
        if self.resultVars is None:
            print('The model must be run first before visualizing results! Execute first the run method')
            return 0
        outRoutes = {key : self.resultVars[key] for key in self.resultVars.keys() if self.resultVars[key] >= 0.99
         and key[0] == 'y'}
        for k in range(1, self.K + 1):
            for t in range(1, self.H + 1):
                edges = [(key[1], key[2]) for key in outRoutes.keys() if key[3] == k and key[4] == t]
                print(edges)
                if len(edges) == 0:
                    continue
                pos =  {i : (self.positions[i][0], self.positions[i][1]) for i in range(self.n + 1) }

                G_1 = nx.Graph()
                G_1.add_edges_from(edges)

                nx.draw(G_1, pos, edge_labels = True, with_labels=True, font_weight='bold')
                plt.show()


def runSeveralIRP(instNames, nbhs = ('function', 'cluster'), timeLimit = 100):
    
    for inst in instNames:
        instAct = IRP(inst)

        for nbhType in nbhs:
            
            instAct.run(
                outImportNeighborhoods=True,
                outImportedNeighborhoods=nbhType,
                outVerbose=False,
                outTimeLimitSeconds=timeLimit,
                writeResult=True
            )
        instAct = IRP(inst)
        instAct.run(
            outImportNeighborhoods=True,
            outImportedNeighborhoods='function',
            outVerbose=False,
            outTimeLimitSeconds=timeLimit,
            outCallback='pure',
            writeResult=True
        )



if __name__ == '__main__':
    runSeveralIRP(
        [ 'abs1n5_3.dat', 'abs1n15_4.dat' ],
        timeLimit=100,
        nbhs= ('normal', 'cluster') )

    print('----------------- Program reached End of Execution Succesfully -----------------')
