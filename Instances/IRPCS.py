import os
import sys
sys.path.append(os.path.pardir)
from math import sqrt
from itertools import chain, combinations
import numpy as np
from gurobipy import *
import networkx as nx
import matplotlib.pyplot as plt
from Cuts import Cut, getSubsets, genSubtourLazy, getCheckSubTour
from Neighborhood import Neighborhoods, genIRPneigh
from VMNDproc import solver
from Functions import IRPCStransformKey, genClusterNeighborhoods

def loadIRPCS(path):
    outdict = {}
    lines = list(map(lambda x: list(filter(lambda y: y != '', x.replace(' ', '\t').strip('\n').split('\t'))), open(path, 'r').readlines()))

    outdict['V'] = int(lines[0][0])
    outdict['H'] = int(lines[0][1])
    outdict['K'] = 3
    outdict['W'] = 2
    outdict['gamma'] = {1: 40, 2: 65}
    outdict['U'] = {i : 65 for i in range(1, outdict['V'] + 1)}
    outdict['M'] = outdict['K'] * max(outdict['U'].values())
    outdict['h'] = {i : 0.5 for i in range(1, outdict['V'] + 1)}
    outdict['s'] = {i : 30 for i in range(1, outdict['V'] + 1)}
    outdict['kappa'] = 300
    outdict['zeta'] = 4 / 60 # Times will be stated as hours
    outdict['theta'] = 40
    outdict['positions'] = np.zeros(shape = (outdict['V'] + 1, 2))
    outdict['I0'] = {}
    outdict['I0'][0] = 100
    outdict['r'] = {}
    outdict['a'] = {}
    outdict['a'][0] = 0
    outdict['delta'] = {}
    
    for ind, i in enumerate(lines[3:]):
        outdict['positions'][ind][0] = float(i[1])
        outdict['positions'][ind][1] = float(i[2])
        
        if ind > 0:
            outdict['I0'][ind] = min(float(i[3]), 64)
            outdict['a'][ind] = int(i[9])
            deltaAct = [int(i[6]), int(i[7]), int(i[8])]
            for t in range(1, outdict['H'] + 1):
                outdict['delta'][(ind, t)] = deltaAct[(t - 1) % 3]
                outdict['r'][(ind, t)] = float(i[9 + t])
            outdict['r'][(ind, 0)] = float(i[outdict['H'] + 9])

        else:
            for t in range(outdict['H'] + 2):
                outdict['r'][(0, t)] = 60

    outdict['tau'] = np.zeros(shape = (outdict['V'] + 1, outdict['V'] + 1))
    outdict['cost'] = np.zeros(shape = (outdict['V'] + 1, outdict['V'] + 1))

    for i in range(0, outdict['V']):
        for j in range(i + 1, outdict['V']):
            outdict['cost'][i][j] = np.linalg.norm( np.array([outdict['positions'][i][0], outdict['positions'][i][1]]) - 
                                            np.array([outdict['positions'][j][0], outdict['positions'][j][1]]) )
            outdict['tau'][i][j] = (outdict['cost'][i][j] / 20) + (1 / 6)
    return outdict


class IRPCS:

    def __init__(self, path, Vtrunc = 15, Htrunc = 8, Ktrunc = 3):
        inDict = loadIRPCS(path)
        self.pathMPS = None
        self.name = 'IRPCS_{}_{}_{}'.format(Vtrunc, Htrunc, Ktrunc)
        self.V = min(inDict['V'], Vtrunc)
        self.H = min(inDict['H'], Htrunc)
        self.r = inDict['r']
        self.U = inDict['U']
        self.K = min(inDict['K'], Ktrunc)
        self.kappa = inDict['kappa']
        self.positions = inDict['positions']
        self.tau = inDict['tau']
        self.cost = inDict['cost']
        self.zeta = inDict['zeta']
        self.W = inDict['W']
        self.theta = inDict['theta']
        self.gamma = inDict['gamma']
        self.h = inDict['h']
        self.s = inDict['s']
        self.delta = inDict['delta']
        self.I0 =  inDict['I0']
        self.a = inDict['a']
        self.M = inDict['M']
        self.resultVars = None

    def createInstance(self):
        model = Model()

        modelVars = {}

        # We add model variables
        for i in range(self.V + 1):
            for j in range(self.V + 1):
                for t in range(1, self.H + 1):
                    for k in range(1, self.K + 1):
                        if i < j :
                            if i == 0:
                                modelVars['x_0_{}_{}_{}'.format(j, k, t)] = \
                                model.addVar(0, 2, vtype = GRB.INTEGER, name='x_{}_{}_{}_{}'.format(0, j, k, t))
                            else:
                                modelVars['x_{}_{}_{}_{}'.format(i, j, k, t)] = \
                                model.addVar(vtype = GRB.BINARY, name='x_{}_{}_{}_{}'.format(i, j, k, t))

            for t in range(1, self.H + 1):
                if i > 0:
                    modelVars['z_{}_{}'.format(i, t)] =\
                    model.addVar(vtype = GRB.BINARY, name='z_{}_{}'.format(i, t))

                    if t == 1:
                        modelVars['z_{}_{}'.format(i, 0)] =\
                        model.addVar(vtype = GRB.BINARY, name='z_{}_{}'.format(i, 0))
                

                for k in range(1, self.K + 1):
                    modelVars['y_{}_{}_{}'.format(i, k, t)] =\
                    model.addVar(vtype = GRB.BINARY, name='y_{}_{}_{}'.format(i, k, t))

                    if t == 1 and i != 0:
                        modelVars['p_{}_{}_{}'.format(i, k, 0)] =\
                        model.addVar(0, 0, vtype = GRB.CONTINUOUS, name='p_{}_{}_{}'.format(i, k, 0))
                        modelVars['q_{}_{}_{}'.format(i, k, 0)] =\
                        model.addVar(0, 0, vtype = GRB.CONTINUOUS, name='q_{}_{}_{}'.format(i, k, 0))

                    if i != 0:
                        modelVars['p_{}_{}_{}'.format(i, k, t)] =\
                            model.addVar(0, GRB.INFINITY, vtype = GRB.CONTINUOUS, name='p_{}_{}_{}'.format(i, k, t))
                        modelVars['q_{}_{}_{}'.format(i, k, t)] =\
                            model.addVar(0, GRB.INFINITY, vtype = GRB.CONTINUOUS, name='q_{}_{}_{}'.format(i, k, t))

                if t == 1:
                    inv_0 = abs(self.I0[i])
                    sto_0 = max(-1 * self.I0[i], 0)

                    modelVars['I_{}_{}'.format(i, 0)] =\
                    model.addVar(inv_0, inv_0, vtype = GRB.CONTINUOUS, name='I_{}_{}'.format(i, 0))
                    modelVars['S_{}_{}'.format(i, 0)] = \
                    model.addVar(sto_0, sto_0, vtype = GRB.CONTINUOUS, name='S_{}_{}'.format(i, 0))

                modelVars['I_{}_{}'.format(i, t)] = \
                model.addVar(0, GRB.INFINITY, vtype = GRB.CONTINUOUS, name='I_{}_{}'.format(i, t))
                modelVars['S_{}_{}'.format(i, t)] = \
                model.addVar(0, GRB.INFINITY, vtype = GRB.CONTINUOUS, name='S_{}_{}'.format(i, t))

                if t == self.H:
                    modelVars['I_{}_{}'.format(i, t + 1)] = \
                    model.addVar(0, GRB.INFINITY, vtype = GRB.CONTINUOUS, name='I_{}_{}'.format(i, t + 1))
                    modelVars['S_{}_{}'.format(i, t + 1)] = \
                    model.addVar(0, GRB.INFINITY, vtype = GRB.CONTINUOUS, name='S_{}_{}'.format(i, t + 1))

        for i in range(self.V + 1):
            for k in range(1, self.K + 1):
                for t in range(1, self.H + 1):
                    for w in range(1, self.W + 1):
                        modelVars['v_{}_{}_{}_{}'.format(i, k, t, w)] = \
                        model.addVar(vtype = GRB.BINARY, name = 'v_{}_{}_{}_{}'.format(i, k, t, w))
        
        #  Term 1: Objective function set.
        
        obj = quicksum( self.a[i] * self.h[i] * modelVars['I_{}_{}'.format(i, t)] + self.a[i] * self.s[i] * modelVars['S_{}_{}'.format(i, t)]
        for i in range(1, self.V + 1) for t in range(1, self.H + 2) ) +\
        quicksum( modelVars['x_{}_{}_{}_{}'.format(i, j, k, t)] * self.cost[i][j]
        for i in range(self.V + 1) for j in range(self.V + 1) for k in range(1, self.K + 1) for t in range(1, self.H + 1) if i < j )
        
        # Term 2: Conservation of inventory flows in Depot.
        model.addConstrs( modelVars['I_{}_{}'.format(0, t)] == modelVars['I_{}_{}'.format(0, t - 1)] + self.r[(0, t - 1)] -
        quicksum( self.a[i] * modelVars['q_{}_{}_{}'.format(i, k, t - 1)] for i in range(1, self.V + 1) for k in range(1, self.K + 1) ) +
        quicksum( self.a[i] * modelVars['p_{}_{}_{}'.format(i, k, t - 1)] for i in range(1, self.V + 1) for k in range(1, self.K + 1) )
        for t in range(1, self.H + 2) )

        # Term 3: Force z = 1 if stockout.
        model.addConstrs( modelVars['I_{}_{}'.format(i, t)] - self.r[(i, t)] +
        quicksum( modelVars['q_{}_{}_{}'.format(i, k, t)] for k in range(1, self.K + 1)) -
        quicksum( modelVars['p_{}_{}_{}'.format(i, k, t)] for k in range(1, self.K + 1)) <=
        self.U[i] * ( 1 - modelVars['z_{}_{}'.format(i, t)] )
        for i in range(1, self.V + 1) for t in range(1, self.H + 1) )

        # Term 4: Force z = 1 if stockout, the other side.
        model.addConstrs( -1 * modelVars['I_{}_{}'.format(i, t)] + self.r[(i, t)] -
        quicksum( modelVars['q_{}_{}_{}'.format(i, k, t)] for k in range(1, self.K + 1)) +
        quicksum( modelVars['p_{}_{}_{}'.format(i, k, t)] for k in range(1, self.K + 1)) <=
        self.U[i] *  modelVars['z_{}_{}'.format(i, t)]
        for i in range(1, self.V + 1) for t in range(1, self.H + 1) )

        # Term 5: Inventory Conservation (Linearized).
        model.addConstrs( modelVars['I_{}_{}'.format(i, t - 1)] - self.r[(i, t - 1)] +
        quicksum( modelVars['q_{}_{}_{}'.format(i, k, t - 1)] for k in range(1, self.K + 1)) -
        quicksum( modelVars['p_{}_{}_{}'.format(i, k, t - 1)] for k in range(1, self.K + 1)) <=
        modelVars['I_{}_{}'.format(i, t)]
        for i in range(1, self.V + 1) for t in range(1, self.H + 2) )

        # Term 6: Inventory Conservation (Linearized).
        M = 100000
        model.addConstrs( modelVars['I_{}_{}'.format(i, t)] <= modelVars['I_{}_{}'.format(i, t - 1)] - self.r[(i, t - 1)] +
        quicksum( modelVars['q_{}_{}_{}'.format(i, k, t - 1)] for k in range(1, self.K + 1)) -
        quicksum( modelVars['p_{}_{}_{}'.format(i, k, t - 1)] for k in range(1, self.K + 1)) +
        M * modelVars['z_{}_{}'.format(i, t - 1)]
        for i in range(1, self.V + 1) for t in range(1, self.H + 2) )

        # Term 7: Inventory Conservation (Linearized).
        model.addConstrs( modelVars['I_{}_{}'.format(i, t + 1)] <= self.U[i] * ( 1 - modelVars['z_{}_{}'.format(i, t)] )
        for i in range(1, self.V + 1) for t in range(0, self.H + 1) )
        
        # Term 8: Stockout Conservation (Linearized).
        model.addConstrs( modelVars['S_{}_{}'.format(i, t)] >= -1 * modelVars['I_{}_{}'.format(i, t - 1)] + self.r[(i, t - 1)] -
        quicksum( modelVars['q_{}_{}_{}'.format(i, k, t - 1)] for k in range(1, self.K + 1)) +
        quicksum( modelVars['p_{}_{}_{}'.format(i, k, t - 1)] for k in range(1, self.K + 1) )
        for i in range(1, self.V + 1) for t in range(1, self.H + 2) )

        # Term 9: Capacity of each ATM is not exceeded in each period.
        model.addConstrs( modelVars['I_{}_{}'.format(i, t)] + quicksum( modelVars['q_{}_{}_{}'.format(i, k, t)] for k in range(1, self.K + 1)) -
        quicksum( modelVars['p_{}_{}_{}'.format(i, k, t)] for k in range(1, self.K + 1)) <= self.U[i]
        for i in range(1, self.V + 1) for t in range(1, self.H + 1) )

        # Term 10: old cassette is picked up with all its contents.
        # Term 10.1: First inequality (left).
        model.addConstrs( modelVars['I_{}_{}'.format(i, t)] - self.U[i] * ( 1 - modelVars['y_{}_{}_{}'.format(i, k, t)]) <= modelVars['p_{}_{}_{}'.format(i, k, t)]
        for i in range(1, self.V + 1) for t in range(1, self.H + 1) for k in range(1, self.K + 1))

        # Term 10.2: Second inequality (right).
        model.addConstrs( modelVars['p_{}_{}_{}'.format(i, k, t)] <= self.U[i] * modelVars['y_{}_{}_{}'.format(i, k, t)]
        for i in range(1, self.V + 1) for t in range(1, self.H + 1) for k in range(1, self.K + 1) )
        
        # Term 11: Old cassette is picked up with all its contents.
        model.addConstrs( modelVars['p_{}_{}_{}'.format(i, k, t)] <= modelVars['I_{}_{}'.format(i, t)]
        for i in range(1, self.V + 1) for t in range(1, self.H + 1) for k in range(1, self.K + 1))

        
        # Term 12: The quantity to deliver to each ATM matches one of the cassette sizes.
        model.addConstrs( modelVars['q_{}_{}_{}'.format(i, k, t)] == quicksum( self.gamma[w] * modelVars['v_{}_{}_{}_{}'.format(i, k, t, w)] for w in range(1, self.W + 1) )
        for i in range(1, self.V + 1) for k in range(1, self.K + 1) for t in range(1, self.H + 1) )

        
        # Term 13: Only one cassette is used per delivery.
        model.addConstrs( modelVars['y_{}_{}_{}'.format(i, k, t)] == quicksum( modelVars['v_{}_{}_{}_{}'.format(i, k, t, w)] for w in range(1, self.W + 1) )
        for i in range(self.V + 1) for k in range(1, self.K + 1) for t in range(1, self.H + 1) )

        # Term 14: Imposes that each ATM can be visited by at most one vehicle per period.
        model.addConstrs( quicksum( modelVars['y_{}_{}_{}'.format(i, k, t)] for k in range(1, self.K + 1) ) <= 1
        for i in range(1, self.V + 1) for t in range(1, self.H + 1) )

        # Term 15: Links the quantities delivered to the routing variables.
        model.addConstrs( modelVars['q_{}_{}_{}'.format(i, k, t)] <= self.U[i] * modelVars['y_{}_{}_{}'.format(i, k, t)]
        for i in range(1, self.V + 1) for t in range(1, self.H + 1) for k in range(1, self.K + 1) )
        
        # Term 16: Imposes the maximum capacity of the vehicles.
        model.addConstrs( quicksum( self.a[i] * modelVars['q_{}_{}_{}'.format(i, k, t)] for i in range(1, self.V + 1)) <=
        self.kappa * modelVars['y_{}_{}_{}'.format(0, k, t)]
        for t in range(1, self.H + 1) for k in range(1, self.K + 1) )

        # Term 17: Locations are only visited in the allowed periods.
        model.addConstrs( modelVars['y_{}_{}_{}'.format(i, k, t)] <= self.delta[(i, t)]
        for i in range(1, self.V + 1) for t in range(1, self.H + 1) for k in range(1, self.K + 1))

        # Term 18: Duration of a route cannot exceed the maximum route length.
        model.addConstrs( quicksum( self.tau[i][j] * modelVars['x_{}_{}_{}_{}'.format(i, j, k, t)] for i in range(self.V + 1) for j in range(self.V + 1) if i < j) +
        self.zeta * quicksum( modelVars['y_{}_{}_{}'.format(i, k, t)] * self.a[i] for i in range(self.V + 1) )
        <= self.theta
        for t in range(1, self.H + 1) for k in range(1, self.K + 1) )

        # Term 19: Degree Constraint
        model.addConstrs(
        quicksum( modelVars['x_{}_{}_{}_{}'.format(i, j, k, t)] for j in range(self.V + 1) if i < j) +
        quicksum( modelVars['x_{}_{}_{}_{}'.format(j, i, k, t)] for j in range(self.V + 1) if j < i) \
        == 2 * modelVars['y_{}_{}_{}'.format(i, k, t)]
        for i in range(self.V + 1) for t in range(1, self.H + 1) for k in range(1, self.K + 1))

        # Term 20: Sub-Tour Elimination: Ommited and alledgely added via callback.

        # Terms 21 - 27: Are already included in variable's definition

        # Terms 28 - 33: Symmetry Breaking Constraints

        # Term 28: Symmetry Breaking Constraint
        
        model.addConstrs( modelVars['x_{}_{}_{}_{}'.format(i, j, k, t)] <= modelVars['y_{}_{}_{}'.format(i, k, t)]
        for i in range(self.V + 1) for j in range(self.V + 1) for k in range(1, self.K + 1) for t in range(1, self.H + 1) if i < j )

        # Term 29: Symmetry Breaking Constraint

        model.addConstrs( modelVars['x_{}_{}_{}_{}'.format(i, j, k, t)] <= modelVars['y_{}_{}_{}'.format(j, k, t)]
        for i in range(self.V + 1) for j in range(self.V + 1) for k in range(1, self.K + 1) for t in range(1, self.H + 1) if i < j )

        # Term 30: Symmetry Breaking Constraint

        model.addConstrs( modelVars['x_{}_{}_{}_{}'.format(0, i, k, t)] <= 2 * modelVars['y_{}_{}_{}'.format(i, k, t)]
        for i in range(1, self.V + 1) for k in range(1, self.K + 1) for t in range(1, self.H + 1) )

        # Term 31: Symmetry Breaking Constraint

        model.addConstrs( modelVars['y_{}_{}_{}'.format(i, k, t)] <= modelVars['y_{}_{}_{}'.format(0, k, t)]
        for i in range(1, self.V + 1) for k in range(1, self.K + 1) for t in range(1, self.H + 1) )

        # Term 32: Symmetry Breaking Constraint

        model.addConstrs( modelVars['y_{}_{}_{}'.format(0, k, t)] <= modelVars['y_{}_{}_{}'.format(0, k - 1, t)]
        for k in range(2, self.K + 1) for t in range(1, self.H + 1) )

        # Term 33: Symmetry Breaking Constraint

        model.addConstrs( modelVars['y_{}_{}_{}'.format(i, k, t)] <= quicksum( modelVars['y_{}_{}_{}'.format(j, k - 1, t)]
        for j in range(1, self.V + 1) if j < i )
        for i in range(1, self.V + 1) for k in range(2, self.K + 1) for t in range(1, self.H + 1) )

        # Objective Function is set.
        model._vars = modelVars
        model.setObjective(obj, GRB.MINIMIZE)
        return model

    def exportMPS(self, writePath = os.path.join(os.path.pardir, 'MIPLIB')):
        model = self.createInstance()
        self.pathMPS = os.path.join(writePath , self.name + '.mps')
        model.write( self.pathMPS )

    def genNeighborhoods(self, varCluster = False, k = 20, funNbhs = False):
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

            klist = ['x_{}_{}_{}_{}'.format( i, j, k, t )
             for k in range(1, self.K + 1) for t in range(1, self.H  + 1) for i in range(self.V + 1) for j in range(self.V + 1) if i < j]
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


            labelsDict = genClusterNeighborhoods( self.pathMPS, amountClusters, fNbhs = True, varFilter=lambda x: x[0] == 'x')
            def fClusterNbhs(varName, depth, param):
                return labelsDict[varName] == depth

            klist = ['x_{}_{}_{}_{}'.format( i, j, k, t )
             for k in range(1, self.K + 1) for t in range(1, self.H  + 1) for i in range(self.V + 1) for j in range(self.V + 1) if i < j]

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
        2 : {(kf, tf) : [ '{}_{}_{}_{}_{}'.format('x', i, j, k, t)
            for k in range(1, self.K + 1) for t in range(1, self.H  + 1) for i in range(self.V + 1) for j in range(self.V + 1)
            if i < j and (k, t) != (kf, tf) ]
                for kf in range(1, self.K + 1) for tf in range(1, self.H + 1)
        },
        3: {
            tf : [ '{}_{}_{}_{}_{}'.format('x', i, j, k, t)
            for k in range(1, self.K + 1) for t in range(1, self.H  + 1) for i in range(self.V + 1) for j in range(self.V + 1)
            if i < j and t != tf]
            for tf in range(1, self.H + 1)
        },
        
        4: {
            kf : [ '{}_{}_{}_{}_{}'.format('x', i, j, k, t)
            for k in range(1, self.K + 1) for t in range(1, self.H  + 1) for i in range(self.V + 1) for j in range(self.V + 1)
            if i < j and k != kf ]
            for kf in range(1, self.K + 1)
        },
        5: {
            (tf1, tf2) : [ '{}_{}_{}_{}_{}'.format('x', i, j, k, t)
            for k in range(1, self.K + 1) for t in range(1, self.H  + 1) for i in range(self.V + 1) for j in range(self.V + 1)
            if i < j and tf1 != t and tf2 != t]
            for tf1 in range(1, self.H  + 1) for tf2 in range(1, self.H  + 1) if tf1 < tf2
        }
        }
        return Neighborhoods(lowest = 2, highest = 5, keysList = None, randomSet = False, outerNeighborhoods = outerNhs, useFunction=False)

    def genLazy(self):
        def f1(solValues):
            cuts = []
            solValues = { IRPCStransformKey(key) : solValues[key] for key in solValues.keys()}

            for k in range(1, self.K + 1):
                for t in range(1, self.H + 1):

                    edges = []
                    for i in range(self.V + 1):
                        for j in range(self.V + 1):
                            if i < j:
                                if solValues[('x', i, j, k, t)] > 0.5:
                                    edges.append((i, j))
                    
                    subsets = getSubsets(edges, self.V)

                    if len(edges) > 0:
                        for subset in subsets:
                            for element in subset:
                                newCut = Cut()
                                nonzeros = {}
                                nonzeros.update({ '{}_{}_{}_{}_{}'.format('x', i, j, k, t) : 1 for i in subset for j in subset if i < j })

                                nonzeros.update({'{}_{}_{}_{}'.format('y', i, k, t) : -1 for i in subset if i != element })
                                newCut.nonzero = nonzeros
                                newCut.sense = '<='
                                newCut.rhs = 0
                                cuts.append(newCut)
            return cuts
        return f1

    def genTestFunction(self):
        def SubtourCheck(vals):
            vals = { IRPCStransformKey(var) : vals[var] for var in vals.keys() if var[0] == 'x' and vals[var] >= 0.99 }

            errorcnt = 0
            for k in range(1, self.K + 1):
                for t in range(1, self.H + 1):
                        
                    edges = [(key[1], key[2]) for key in vals.keys() if (key[3], key[4]) == (k, t) and key[0] == 'x']
                    if len(edges) > 0:
                        subsets = getSubsets(edges, self.V)
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
        return SubtourCheck

    def run(
        self,
        outImportNeighborhoods = True,
        outImportedNeighborhoods = 'function',
        outFunTest = None,
        outAlpha = 1,
        outCallback = 'vmnd',
        outVerbose = False,
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
                importedNeighborhoods= self.genNeighborhoods(funNbhs=True),
                funTest= self.genTestFunction(),
                alpha = outAlpha,
                callback = outCallback,
                verbose = outVerbose,
                timeLimitSeconds= outTimeLimitSeconds
            )
        else:
            modelOut = solver(
                path = self.pathMPS,
                addlazy = True,
                funlazy= self.genLazy(),
                importNeighborhoods= True,
                importedNeighborhoods= self.genNeighborhoods(varCluster=True),
                funTest= self.genTestFunction(),
                alpha = outAlpha,
                callback = outCallback,
                verbose = outVerbose,
                timeLimitSeconds= outTimeLimitSeconds
            )

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
        
        self.resultVars = {IRPCStransformKey(var.varName) : var.x for var in modelOut.getVars() if var.x > 0 }

        return modelOut

    def analyzeRes(self): pass

    def visualizeRes(self):
        if self.resultVars is None:
            print('The model must be run first before visualizing results! Execute first the run method')
            return 0
        outRoutes = {key : self.resultVars[key] for key in self.resultVars.keys() if self.resultVars[key] >= 0.99
         and key[0] == 'x'}
        for k in range(1, self.K + 1):
            for t in range(1, self.H + 1):
                edges = [(key[1], key[2]) for key in outRoutes.keys() if key[3] == k and key[4] == t]
                print(edges)
                if len(edges) == 0:
                    continue
                pos =  {i : (self.positions[i][0], self.positions[i][1]) for i in range(self.V + 1) }

                G_1 = nx.Graph()
                G_1.add_edges_from(edges)

                nx.draw(G_1, pos, edge_labels = True, with_labels=True, font_weight='bold')
                plt.show()


def runSeveralIRPCS(instNames, nbhs = ('function', 'cluster'), timeLimit = 100, outVtrunc = 20, outHtrunc = 3, outKtrunc = 3):

    for inst in instNames:
        instAct = IRPCS(inst, Vtrunc = outVtrunc, Htrunc = outHtrunc, Ktrunc = outKtrunc)

        for nbhType in nbhs:
            
            instAct.run(
                outImportNeighborhoods=True,
                outImportedNeighborhoods=nbhType,
                outVerbose=False,
                outTimeLimitSeconds=timeLimit,
                writeResult=True
            )
        instAct = IRPCS(inst)
        instAct.run(
            outImportNeighborhoods=True,
            outImportedNeighborhoods='function',
            outVerbose=False,
            outTimeLimitSeconds=timeLimit,
            outCallback='pure',
            writeResult=True
        )

if __name__ == '__main__':
    runSeveralIRPCS( [ os.path.join( 'IRPCSInstances', 'inst1.txt') ] )