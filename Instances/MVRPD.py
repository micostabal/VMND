from gurobipy import *
import sys
from VMNDproc import solver
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
from Functions import keyOpMVRPD


def loadMVRPD(path):
    outdict = {}
    lines = list(map(lambda x: list(filter(lambda y: y != '', x.replace(' ', '\t').strip('\n').split('\t'))), open(path, 'r').readlines()))

    outdict['V'] = int(lines[0][0]) - 1
    outdict['H'] = int(lines[0][1])
    outdict['Q'] = int(lines[0][2])
    outdict['m'] = 3
    outdict['h'] = {i : 1 for i in range(1, outdict['V'] + 1)}
    outdict['p'] = {i : 1 for i in range(1, outdict['V'] + 1)}
    outdict['demand'] = {i : 1 for i in range(1, outdict['V'] + 1)}
    outdict['release'] = {i : 1 for i in range(1, outdict['V'] + 1)}
    outdict['duedates'] = {i : 1 for i in range(1, outdict['V'] + 1)}
    outdict['positions'] = np.zeros(shape = (outdict['V'] + 1, 2))
    
    for ind, i in enumerate(lines[1:]):
        outdict['positions'][ind][0] = float(i[1])
        outdict['positions'][ind][1] = float(i[2])
        
        if ind > 0:
            outdict['demand'][ind] = float(i[3])
            outdict['h'][ind] = float(i[4])
            outdict['p'][ind] = 10 * outdict['h'][ind]
            
            outdict['duedates'][ind] = int(i[5])
            outdict['release'][ind] = int(max(1, outdict['duedates'][ind] - 2))


    outdict['cost'] = np.zeros(shape = (outdict['V'] + 1, outdict['V'] + 1))

    for i in range(0, outdict['V'] + 1):
        for j in range(i + 1, outdict['V'] + 1):
            outdict['cost'][i][j] = np.linalg.norm( np.array([outdict['positions'][i][0], outdict['positions'][i][1]]) - 
                                            np.array([outdict['positions'][j][0], outdict['positions'][j][1]]) )
    return outdict


class MVRPD:

    def __init__(self, path = ''):
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
        self.C = [ i for i in range(1, int(self.V / 2))]
        self.dueDates = dictInst['duedates']
        self.cost = dictInst['cost']
        self.positions = dictInst['positions']
        self.resultVars = None

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

                        if i > 0:
                            if t > self.dueDates[i] or t < self.release[i]:
                                model.addConstr(modelVars['x_{}_{}_{}'.format(i, j, t)] == 0)

                        modelVars['l_{}_{}_{}'.format(i, j, t)] = \
                        model.addVar(0, GRB.INFINITY, vtype = GRB.CONTINUOUS, name='l_{}_{}_{}'.format(i, j, t))

        #Term 1.1 : Objective Function: Routing Costs.
        obj = quicksum( self.cost[i][j] * modelVars['x_{}_{}_{}'.format(i, j, t)] for t in range( self.release[i] , self.H + 1 ) 
         for i in range(self.V + 1) for j in range(self.V + 1) if i != j )

        #Term 1.2 : Objective Function: Inventory holding cost for clients visited during the planning horizon.
        obj += quicksum( self.h[i] * (self.H - self.release[i] ) * modelVars['x_{}_{}_{}'.format(i, j, t)]
         for t in range( self.release[i] , self.H + 1 ) for i in range(1, self.V + 1) for j in range(self.V + 1) if i != j )

        #Term 1.3 : Objective Function: Inventory holding cost and penalty for postponed customers.
        obj += quicksum( (self.h[i] * (self.H - self.release[i] ) + self.p[i]) *
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

        ## the sets calS (caligraphic S) and qtij (q_(t_i)_(t_j)) are defined.
        calS = {
            (t1, t2) : [i for i in range(1, self.V + 1) if self.release[i] >= t1 and self.dueDates[i] <= t2 ]
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
            <= self.Q * quicksum( modelVars['x_{}_{}_{}'.format(0, j, t)] for t in range(tPrime + 1, self.H) for j in range(1, self.V + 1) )
             for tPrime in range(1, self.H + 1)
        )

        #Term 14: Tightening Cut, added from Archetti et al 2015.
        model.addConstrs(
            quicksum( modelVars['x_{}_{}_{}'.format(0, j, t)] for t in range(t1, t2 + 1) for j in range(1, self.V + 1)) >= 
            ceil(qt1t2[(t1, t2)] / self.Q)
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

    def genNeighborhoods(self, k = 20, Kvecinities = True):
        if Kvecinities:
            X = self.positions
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
            indices = nbrs.kneighbors(X)[1]

            outerMVRPD = {
                1 : { # Period Neighborhood
                    tAct : ['x_{}_{}_{}'.format(i, j, t) for i in range(self.V + 1)
                    for j in range(self.V + 1) for t in range(1, self.H + 1) if i != j and t != tAct ]
                    for tAct in range(1, self.H + 1)
                },
                2 : { # k Vecinity Neighborhood
                    ip : ['x_{}_{}_{}'.format(i, j, t) for i in range(self.V + 1) for j in range(self.V + 1) for t in range(1, self.H + 1)
                    if i != j and ( i in indices[ip] or j in indices[ip] ) ]
                    for ip in range(self.V + 1)
                }
            }
            return Neighborhoods(lowest=1, highest = 2, keysList=None, randomSet=False, outerNeighborhoods = outerMVRPD)
        else:
            X = self.positions

            kmp = 3 # K means parameter
            kmeans = KMeans(n_clusters = kmp, random_state=0).fit(X)
            labels = kmeans.labels_

            outerMVRPD = {
                1 : { # Period Neighborhood
                    tAct : ['x_{}_{}_{}'.format(i, j, t) for i in range(self.V + 1)
                    for j in range(self.V + 1) for t in range(1, self.H + 1) if i != j and t != tAct ]
                    for tAct in range(1, self.H + 1)
                },
                2 : { # K means Neighborhood
                    (selK, tAct) : ['x_{}_{}_{}'.format(i, j, t)
                     for i in range(self.V + 1) for j in range(self.V + 1) for t in range(1, self.H + 1)
                    if i != j and t != tAct and labels[i] != selK and labels[j] != selK ]
                    for selK in range(kmp) for tAct in range(1, self.H + 1)
                }
            }
            return Neighborhoods(lowest=1, highest = 2, keysList=None, randomSet=False, outerNeighborhoods = outerMVRPD)

    def genLazy(self):
        # No lazy constraints are required for this problem
        pass

    def genTestFunction(self):
        pass

    def run(self):
        self.exportMPS()

        modelOut = solver(
            path = self.pathMPS,
            addlazy = False,
            funlazy= None,
            importNeighborhoods= True,
            importedNeighborhoods= self.genNeighborhoods(k=20, Kvecinities= True),
            funTest= None,
            alpha = 1,
            callback = 'vmnd',
            verbose = True,
            minBCTime = 10
        )
        self.resultVars = {keyOpMVRPD(var.varName) : var.x for var in modelOut.getVars() if var.x > 0 }

        return modelOut

    def analyzeRes(self):
        # Not built yet.
        pass

    def visualizeRes(self):
        self.run()
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

if __name__ == '__main__':

    mvrpd1 = MVRPD(os.path.join( 'MVRPDInstances' , 'ajs2n50_h_3.dat'))
    mvrpd1.visualizeRes()
    
    
