from gurobipy import *
import sys
import os
sys.path.append(os.path.pardir)
from Instance import Instance
from math import ceil
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from Neighborhood import Neighborhoods
from Functions import keyOpMVRPD, genClusterNeighborhoods
from VMNDproc import solver
from ConComp import getSubsets

def loadMVRPD(path):
    outdict = {}
    lines = list(map(lambda x: list(filter(lambda y: y != '', x.replace(' ', '\t').strip('\n').split('\t'))), open(path, 'r').readlines()))

    outdict['V'] = int(lines[0][0]) - 1
    outdict['H'] = int(lines[0][1])
    outdict['Q'] = int(lines[0][2])
    outdict['m'] = 1
    outdict['h'] = {i : 1 for i in range(1, outdict['V'] + 1)}
    outdict['p'] = {i : 1 for i in range(1, outdict['V'] + 1)}
    outdict['demand'] = {i : 1 for i in range(1, outdict['V'] + 1)}
    outdict['release'] = {i : 1 for i in range(1, outdict['V'] + 1)}
    outdict['duedates'] = {i : 1 for i in range(1, outdict['V'] + 1)}
    outdict['positions'] = np.zeros(shape = (outdict['V'] + 1, 2))
    outdict['C'] = []
    
    for ind, i in enumerate(lines[1:]):
        outdict['positions'][ind][0] = float(i[1])
        outdict['positions'][ind][1] = float(i[2])
        
        if ind > 0:
            outdict['demand'][ind] = float(i[3])
            outdict['h'][ind] = float(i[4])
            outdict['p'][ind] = round(10 * outdict['h'][ind], 1)
            
            #outdict['duedates'][ind] = int(i[5])
            #outdict['release'][ind] = int(max(1 , outdict['duedates'][ind] - 2))

            outdict['release'][ind] = int(i[5])
            outdict['duedates'][ind] = int(min(outdict['H'] , outdict['release'][ind] + 2))
            if outdict['release'][ind] + 2 > outdict['H']:
                outdict['C'].append(ind)

    outdict['cost'] = np.zeros(shape = (outdict['V'] + 1, outdict['V'] + 1))

    for i in range(outdict['V'] + 1):
        for j in range(outdict['V'] + 1):
            outdict['cost'][i][j] = np.linalg.norm( np.array([outdict['positions'][i][0], outdict['positions'][i][1] ]) - 
                                            np.array([outdict['positions'][j][0], outdict['positions'][j][1]]) )
    return outdict


class MVRPD(Instance):

    def __init__(self, path = ''):
        super().__init__()
        self.name = str(os.path.basename(path)).rstrip('.dat')
        self.pathMPS = None
        dictInst = loadMVRPD(path)
        self.V = dictInst['V'] 
        self.h = dictInst['h']
        self.p = dictInst['p']
        self.H = dictInst['H']
        self.Q = dictInst['Q']
        self.m = dictInst['m']
        self.q = dictInst['demand']
        self.q[0] = 0
        self.positions = dictInst['positions']
        self.release = dictInst['release']
        self.C = dictInst['C']
        self.dueDates = dictInst['duedates']
        self.cost = dictInst['cost']
        self.positions = dictInst['positions']
        self.resultVars = None
        self.vals = None

    def createInstance(self):
        model = Model()
        modelVars = {}

        # We add model variables
        for i in range(self.V + 1):
            for j in range(self.V + 1):
                for t in range(1, self.H + 1):
                    if i != j:
                        
                        modelVars['x_{}_{}_{}'.format(i, j, t)] = \
                        model.addVar(vtype = GRB.BINARY, name='x_{}_{}_{}'.format(i, j, t))

                        modelVars['l_{}_{}_{}'.format(i, j, t)] = \
                        model.addVar(0, GRB.INFINITY, vtype = GRB.CONTINUOUS, name='l_{}_{}_{}'.format(i, j, t))

        #Term 1.1 : Objective Function: Routing Costs.
        obj = quicksum( self.cost[i][j] * modelVars['x_{}_{}_{}'.format(i, j, t)] for t in range( self.release[i] , self.H + 1 ) 
         for i in range(self.V + 1) for j in range(self.V + 1) if i != j )

        #Term 1.2 : Objective Function: Inventory holding cost for clients visited during the planning horizon.
        obj += quicksum( self.q[i] * self.h[i] * (t - self.release[i]) * modelVars['x_{}_{}_{}'.format(i, j, t)]
         for j in range(self.V + 1) for i in range(1, self.V + 1) for t in range( self.release[i] , self.H + 1 )  if i != j )

        #Term 1.3 : Objective Function: Inventory holding cost and penalty for postponed customers.
        obj += quicksum( ( self.q[i] * self.h[i] * (self.H - self.release[i] ) + self.q[i] * self.p[i]) *
         (1 - quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)] for j in range(self.V + 1)
         for t in range( self.release[i] , self.H + 1 ) if j != i ) ) for i in self.C )
        
        #Term 2: Every mandatory customer is visited exactly once:
        model.addConstrs( quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)]
         for j in range(self.V + 1) if j != i for t in range( self.release[i] , self.dueDates[i] + 1 ) ) == 1
         for i in range(1, self.V + 1) if i not in self.C )

        #Term 3: Optional customers are visited only once.
        model.addConstrs( quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)]
         for j in range(self.V + 1) for t in range( self.release[i] , self.dueDates[i] + 1 ) if j != i ) <= 1
          for i in self.C )

        #Term 4: Flow Conservation.
        model.addConstrs( quicksum( modelVars['x_{}_{}_{}'.format(j, i, t)] for j in range(self.V + 1) if j != i ) == 
         quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)] for j in range(self.V + 1) if i != j )
         for i in range(self.V + 1) for t in range(1, self.H + 1) )

        #Term 5: Number of Vehicles is m at most.
        model.addConstrs( quicksum( modelVars['x_{}_{}_{}'.format(0, j, t)] for j in range(self.V + 1) if j != 0 ) <= self.m
         for t in range(1, self.H + 1) ) 

        #Term 6: Load conserveation through x variables:
        model.addConstrs( quicksum( 
            modelVars['l_{}_{}_{}'.format(j, i, t)] for j in range(self.V + 1) if j != i ) -
            quicksum( modelVars['l_{}_{}_{}'.format(i, j, t)] for j in range(self.V + 1) if j != i ) ==
            self.q[i] * quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)] for j in range(self.V + 1) if j != i )
         for i in range(1, self.V + 1) for t in range(1, self.H + 1) )

        #Term 7: Load conserveation through x variables (depot):
        model.addConstrs( 
            (-1) * quicksum( modelVars['l_{}_{}_{}'.format(j, 0, t)] for j in range(1, self.V + 1) if j != 0 ) +
            quicksum( modelVars['l_{}_{}_{}'.format(0, j, t)] for j in range(1, self.V + 1) if j != 0 ) ==
            quicksum( self.q[i] * modelVars['x_{}_{}_{}'.format(i, j, t)] for j in range(self.V + 1) for i in range(self.V + 1) if j != i )
         for t in range(1, self.H + 1) )

        #Term 8: Loads do not exceed vehicle's capacity.
        model.addConstrs( modelVars['l_{}_{}_{}'.format(i, j, t)] <= self.Q * modelVars['x_{}_{}_{}'.format(i, j, t)]
         for i in range(self.V + 1) for j in range(self.V + 1) for t in range(1, self.H + 1) if i != j )

        #Term 9&10: Variable types are declared in the creation of the variables.

        #Term 10: l is nonnegative
        model.addConstrs( modelVars['l_{}_{}_{}'.format(i, j, t)] >= 0
         for t in range(1, self.H + 1) for i in range(self.V + 1) for j in range(self.V + 1) if i != j )
        
        # Clients outside its time window are set to zero.
        model.addConstrs( modelVars['x_{}_{}_{}'.format(i, j, t)] == 0
         for t in range(1, self.H + 1) for j in range(self.V + 1) for i in range(self.V + 1) if i != j and i > 0 and (
         t > self.dueDates[i] or t < self.release[i] ) )

        ## the sets calS (caligraphic S) and qtij (q_(t_i)_(t_j)) are defined.
        calS = {
            (t1, t2) : [i for i in range(1, self.V + 1) if self.release[i] >= t1 and self.dueDates[i] <= t2 and i not in self.C ]
             for t1 in range(1, self.H + 1) for t2 in range(1, self.H + 1) if t1 <= t2
        }
        qt1t2 = {}
        for t1 in range(1, self.H + 1):
            for t2 in range(1, self.H + 1):
                if t1 <= t2:
                    if len(calS[(t1, t2)]) == 0:
                        qt1t2[(t1, t2)] = 0
                    else:
                        qt1t2[(t1, t2)] = sum(list(map(lambda x : self.q[x], calS[(t1, t2)])))

        #Term 11: Tightening Cut, added from Archetti et al 2015.
        model.addConstrs( quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)] * self.q[i]
         for i in range(self.V + 1) for j in range(self.V + 1) for t in range(1, tPrime + 1) if i != j )
         <= self.m * self.Q * tPrime
         for tPrime in range(1, self.H + 1) )

        #Term 12: Tightening Cut, added from Archetti et al 2015.
        model.addConstrs(
            quicksum( self.q[i] for i in range(self.V + 1) ) - 
            quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)] * self.q[i]
             for i in range(self.V + 1) for j in range(self.V + 1) for t in range(1, tPrime + 1) if i != j ) -
            quicksum( 
                self.q[i] * (1 - quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)] for t in range(self.release[i], self.H + 1)
                 for j in range(self.V + 1) if i != j ) )
             for i in self.C )
            <= self.m * self.Q * (self.H - tPrime)
             for tPrime in range(1, self.H + 1)
        )

        #Term 13: Tightening Cut, added from Archetti et al 2015.
        model.addConstrs(
            quicksum( self.q[i] for i in range(self.V + 1) ) - 
            quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)] * self.q[i]
             for i in range(self.V + 1) for j in range(self.V + 1) for t in range(1, tPrime + 1) if i != j ) -
            quicksum( 
                self.q[i] * (1 - quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)] for t in range(self.release[i], self.H + 1)
                 for j in range(self.V + 1) if i != j ) )
             for i in self.C )
            <= self.Q * quicksum( modelVars['x_{}_{}_{}'.format(0, j, t)] for t in range(tPrime + 1, self.H + 1) for j in range(1, self.V + 1) )
             for tPrime in range(1, self.H + 1)
        )

        #Term 14: Tightening Cut, added from Archetti et al 2015.
        model.addConstrs(
            quicksum( modelVars['x_{}_{}_{}'.format(0, j, t)] for t in range(t1, t2 + 1) for j in range(1, self.V + 1) ) >= 
            ceil(qt1t2[(t1, t2)] / self.Q )
             for t1 in range(1, self.H + 1) for t2 in range(1, self.H + 1) if t1 <= t2
        )

        #Term 15: Tightening Cut, added from Archetti et al 2015.
        model.addConstrs(
            quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)] * self.q[j]
             for t in range(t1, t2 + 1) for i in range(self.V + 1) for j in calS[(t1, t2)] if i != j ) >= 
             qt1t2[(t1, t2)]
             for t1 in range(1, self.H + 1) for t2 in range(1, self.H + 1) if t1 <= t2 and len(calS[(t1, t2)]) > 0
        )

        #Term 16: Tightening Cut, added from Archetti et al 2015.
        model.addConstrs(
            quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)] * self.q[j]
             for t in range(t1, t2 + 1) for i in range(self.V + 1) for j in calS[(t1, t2)] if i != j ) <= 
             self.m * self.Q * (t2 - t1 + 1)
             for t1 in range(1, self.H + 1) for t2 in range(1, self.H + 1) if t1 <= t2 and len(calS[(t1, t2)]) > 0
        )
        
        # New valid cuts proposed by Larraín et al 2019.

        #Term 17: New Valid Cut.
        model.addConstrs(
            quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)] * self.q[i]
             for t in range(1, tPrime + 1) for i in range(self.V + 1) for j in range(self.V + 1) if i != j ) <=
             self.Q * quicksum( modelVars['x_{}_{}_{}'.format(0, j, t)]
             for t in range(1, tPrime + 1) for j in range(1, self.V + 1) if 0 != j )
             for tPrime in range(1, self.H + 1) )

        #Term 18: New Valid Cut.
        model.addConstrs(
            quicksum( modelVars['x_{}_{}_{}'.format(i, j, t)] * self.q[j]
             for t in range(t1, t2 + 1) for i in range(self.V + 1) for j in calS[(t1, t2)] if i != j ) <=
             self.Q * quicksum( modelVars['x_{}_{}_{}'.format(0, j, t)]
             for t in range(t1, t2 + 1) for j in range(1, self.V + 1) if 0 != j )
             for t1 in range(1, self.H + 1) for t2 in range(1, self.H + 1) if t1 <= t2 and len(calS[(t1, t2)]) > 0 )

        # Objective Function is set.
        model._vars = modelVars
        model.setObjective(obj, GRB.MINIMIZE)
        return model

    def exportMPS(self, writePath = os.path.join(os.path.pardir, 'MIPLIB')):
        model = self.createInstance()
        self.pathMPS = os.path.join(writePath , self.name + '.mps')
        model.write( self.pathMPS )

    def genNeighborhoods(self, k = 25, Kvicinities = False, funNbhs = False, varCluster = False):
        if varCluster:

            numClu = int(self.H * self.V / 7)

            outerNbhs = { i : (0,) for i in range(1, numClu + 1) }

            labelsDict = genClusterNeighborhoods( self.pathMPS, numClu, fNbhs = True, varFilter=lambda x: x[0] == 'x')
            def fClusterNbhs(varName, depth, param):
                return labelsDict[varName] != depth - 1          

            klist = ['x_{}_{}_{}'.format(i, j, t) for i in range(self.V + 1) for j in range(self.V + 1) for t in range(1, self.H + 1) if i != j ]

            return Neighborhoods(
                lowest = 1,
                highest = numClu,
                keysList= klist,
                randomSet=False,
                outerNeighborhoods=outerNbhs,
                useFunction=True,
                funNeighborhoods=fClusterNbhs
                )
        
        if funNbhs:
            X = self.positions
            # In larrain et al 2019 the neighbors parameter is set to 20.
            nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(X)
            indices = nbrs.kneighbors(X)[1]
            
            def fNbhs(varName, depth, param):
                elements = varName.split('_')
                if len(elements) < 4:
                    return False
                else:
                    tl = int(elements[3])
                    il = int(elements[1])
                    jl = int(elements[2])

                    if depth == 1:
                        return tl != param
                    elif depth == 2:
                        return il not in indices[param] and jl not in indices[param]
                    else:
                        print('Error 23 Nbhds Function!! ')
                        return 0
                return True

            outer = {
                1 : tuple([ tf for tf in range(1, self.H + 1) ]),
                2 : tuple([ i for i in range(1, self.V + 1) ])
            }

            klist = ['x_{}_{}_{}'.format( i, j, t )
             for t in range(1, self.H  + 1) for i in range(self.V + 1) for j in range(self.V + 1) if i != j]
            return Neighborhoods(
                lowest = 1,
                highest = 2,
                keysList= klist,
                randomSet=False,
                outerNeighborhoods=outer,
                funNeighborhoods= fNbhs,
                useFunction=True)

        if Kvicinities:
            X = self.positions
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
            indices = nbrs.kneighbors(X)[1]

            outerMVRPD = {
                1 : { # Period Neighborhood
                    tAct : tuple(['x_{}_{}_{}'.format(i, j, t) for i in range(self.V + 1)
                    for j in range(self.V + 1) for t in range(1, self.H + 1) if i != j and t != tAct ])
                    for tAct in range(1, self.H + 1)
                },
                2 : { # k Vecinity Neighborhood
                    ip : tuple(['x_{}_{}_{}'.format(i, j, t) for i in range(self.V + 1) for j in range(self.V + 1) for t in range(1, self.H + 1)
                    if i != j and ( i in indices[ip] or j in indices[ip] ) ])
                    for ip in range(self.V + 1)
                }
            }
            return Neighborhoods(lowest=1, highest = 2, keysList=None, randomSet=False, outerNeighborhoods = outerMVRPD, useFunction = False)
        
        else:
            X = self.positions

            kmp = 3 # K means parameter
            kmeans = KMeans(n_clusters = kmp, random_state=0).fit(X)
            labels = kmeans.labels_

            outerMVRPD = {
                1 : { # Period Neighborhood
                    tAct : tuple(['x_{}_{}_{}'.format(i, j, t) for i in range(self.V + 1)
                    for j in range(self.V + 1) for t in range(1, self.H + 1) if i != j and t != tAct ])
                    for tAct in range(1, self.H + 1)
                },
                2 : { # K means Neighborhood
                    (selK, tAct) : tuple([ 'x_{}_{}_{}'.format(i, j, t)
                     for i in range(self.V + 1) for j in range(self.V + 1) for t in range(1, self.H + 1)
                    if i != j and t != tAct and labels[i] != selK and labels[j] != selK ])
                    for selK in range(kmp) for tAct in range(1, self.H + 1)
                }
            }
            return Neighborhoods(lowest=1, highest = 2, keysList=None, randomSet=False, outerNeighborhoods = outerMVRPD)

    def genLazy(self):
        # No lazy constraints are required for this problem
        pass

    def genTestFunction(self):
        def checkSubTour(vals):
            vals = { keyOpMVRPD(var) : vals[var] for var in vals.keys() if var[0] == 'x' and vals[var] >= 0.999 }

            errorcnt = 0
            for t in range(1, self.H + 1):
                    
                edges = [(key[1], key[2]) for key in vals.keys() if key[3] == t and key[0] == 'x']
                if len(edges) > 0:

                    subsets = getSubsets(edges, self.V + 1)
                    if len(subsets) > 0:
                        print(t, subsets)
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
        writeResult = True,
        outPlotGapsTimes = False,
        outWriteTestLog = False
        ):
        self.exportMPS()

        if outImportedNeighborhoods == 'function':
            modelOut = solver(
                path = self.pathMPS,
                addlazy = False,
                funlazy= None,
                importNeighborhoods= True,
                importedNeighborhoods= self.genNeighborhoods(funNbhs=True),
                funTest= self.genTestFunction(),
                alpha = outAlpha,
                callback = outCallback,
                verbose = outVerbose,
                minBCTime = outMinBCTime,
                timeLimitSeconds= outTimeLimitSeconds,
                plotGapsTime = outPlotGapsTimes,
                writeTestLog = outWriteTestLog
            )
        elif outImportedNeighborhoods == 'separated':
            nbhs = self.genNeighborhoods(funNbhs=True)
            nbhs.separateParameterizations()
            modelOut = solver(
                path = self.pathMPS,
                addlazy = False,
                funlazy= None,
                importNeighborhoods= True,
                importedNeighborhoods= nbhs,
                funTest= self.genTestFunction(),
                alpha = outAlpha,
                callback = outCallback,
                verbose = outVerbose,
                minBCTime = outMinBCTime,
                timeLimitSeconds= outTimeLimitSeconds,
                plotGapsTime = outPlotGapsTimes,
                writeTestLog = outWriteTestLog
            )
        elif outImportedNeighborhoods == 'cluster':
            modelOut = solver(
                path = self.pathMPS,
                addlazy = False,
                funlazy= None,
                importNeighborhoods= True,
                importedNeighborhoods= self.genNeighborhoods(varCluster=True),
                funTest= self.genTestFunction(),
                alpha = outAlpha,
                callback = outCallback,
                verbose = outVerbose,
                minBCTime = outMinBCTime,
                timeLimitSeconds= outTimeLimitSeconds,
                plotGapsTime = outPlotGapsTimes,
                writeTestLog = outWriteTestLog
            )
        else:
            modelOut = solver(
                path = self.pathMPS,
                addlazy = False,
                funlazy= None,
                importNeighborhoods= True,
                importedNeighborhoods= self.genNeighborhoods(funNbhs=False, varCluster=False, Kvicinities=True),
                funTest= self.genTestFunction(),
                alpha = outAlpha,
                callback = outCallback,
                verbose = outVerbose,
                minBCTime = outMinBCTime,
                timeLimitSeconds= outTimeLimitSeconds,
                plotGapsTime = outPlotGapsTimes,
                writeTestLog = outWriteTestLog
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

        self.resultVars = {keyOpMVRPD(var.varName) : var.x for var in modelOut.getVars() if var.x > 0}
        self.vals ={var : modelOut._vars[var].x for var in modelOut._vars}

        return modelOut

    def analyzeRes(self):
        routingCosts = 0
        holdingCosts = 0
        penaltyCosts = 0

        for t in range(1, self.H + 1):
            for i in range(self.V + 1):
                for j in range(self.V + 1):
                    if i != j:
                        routingCosts += self.vals['x_{}_{}_{}'.format(i, j, t)] * self.cost[i][j]

        for i in range(1, self.V + 1):
            inside1 = 0
            for t in range(self.release[i], self.H + 1):
                inside1 += sum([self.vals['x_{}_{}_{}'.format(i, j, t)] for j in range(self.V + 1) if i != j]) * (t - self.release[i])
            
            holdingCosts += self.q[i] * self.h[i] * inside1
    
        for i in self.C:
            firstTerm = ( self.q[i] * self.h[i] * (self.H - self.release[i]) + self.p[i] * self.q[i])
            secondTerm = 1 - sum([ self.vals['x_{}_{}_{}'.format(i, j, t)]
             for j in range(self.V + 1) for t in range(self.release[i], self.H + 1) if i != j])
            penaltyCosts += firstTerm * secondTerm


        print('Routing Costs : {}'.format(routingCosts))
        print('Holding Costs : {}'.format(holdingCosts))
        print('Penalty Costs : {}'.format(penaltyCosts))
        print('Total Objective : {}'.format( routingCosts + holdingCosts + penaltyCosts ))

    def visualizeRes(self):
        if self.resultVars is None:
            print('The model must be run first before visualizing results! Execute first the run method')
            return 0
        outRoutes = {key : self.resultVars[key] for key in self.resultVars.keys() if self.resultVars[key] >= 0.999
         and key[0] == 'x'}
        for t in range(1, self.H + 1):
            edges = [(key[1], key[2]) for key in outRoutes.keys() if key[3] == t]
            print(edges)
            if len(edges) == 0:
                continue
            pos =  {i : (self.positions[i][0], self.positions[i][1]) for i in range(self.V + 1) }

            G_1 = nx.Graph()
            G_1.add_edges_from(edges)

            nx.draw(G_1, pos, edge_labels = True, with_labels=True, font_weight='bold')
            plt.show()


def runSeveralMVRPD(instNames, nbhs = ('normal', 'cluster'), timeLimit = 100, includePure = True):
    
    for inst in instNames:
        instAct = MVRPD(inst)

        for nbhType in nbhs:
            
            instAct.run(
                outImportNeighborhoods=True,
                outImportedNeighborhoods=nbhType,
                outVerbose=False,
                outTimeLimitSeconds=timeLimit,
                writeResult=True
            )

        if includePure:
            instAct = MVRPD(inst)
            instAct.run(
                outImportNeighborhoods=True,
                outImportedNeighborhoods='function',
                outVerbose=False,
                outTimeLimitSeconds=timeLimit,
                outCallback='pure',
                writeResult=True
            )

if __name__ == '__main__':

    inst1 = MVRPD( os.path.join( 'MVRPDInstances' , 'ajs1n25_h_3.dat' ) )
    inst1.run(
        outImportedNeighborhoods='function',
        writeResult=False,
        outVerbose=True,
        outCallback = 'vmnd',
        outTimeLimitSeconds= None,
        outWriteTestLog = True
    )
    inst1.analyzeRes()