import os
import sys
sys.path.append(os.path.pardir)
import numpy as np
from random import randint, random
from math import floor
from Instance import Instance
from gurobipy import Model, GRB, quicksum
from Neighborhood import genIRPneighborhoods, Neighborhoods
from Functions import genClusterNeighborhoods
from VMNDproc import solver


def generateInstanceFile(n, R, m):
    
    fileOut = open( os.path.join('OISRCInstances', 'instance_{}_{}_{}_3.oisrc'.format(m, R, n) ), 'w+')
    fileOut.write('{}\n'.format(m))
    fileOut.write('{}\n'.format(R))
    fileOut.write('{}\n'.format(n))

    for j in range(n):
        sj = randint(0, 10 * n - 1)
        dj = randint(1, 5 * n)
        rj = randint(1, R)
        uj = random()
        vj = floor( rj * dj * (0.5 + uj) )

        fileOut.write('{} {} {} {} {}\n'.format(j, sj, sj + dj, rj, vj))

    fileOut.close()


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


        if outImportedNeighborhoods is 'cluster':
            modelOut = solver(
                self.pathMPS,
                verbose = outVerbose,
                addlazy= False, 
                funlazy= None,
                importNeighborhoods= outImportNeighborhoods, 
                importedNeighborhoods= self.genNeighborhoods(),
                funTest= None, 
                callback = outCallback,
                alpha = outAlpha,
                minBCTime= outMinBCTime,
                timeLimitSeconds= outTimeLimitSeconds
            )
        else:
            print('Cluster Neighborhoods are the only available in this model.')
        

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

        self.outputvars = {var.varName : var.x for var in modelOut.getVars() if var.x > 0 }
        return modelOut

    def analyzeRes(self): pass

    def visualizeRes(self): pass


def runSeveralOISRC(instNames = [os.path.join('OISRCInstances', 'instance_15_2_170_1.oisrc')], nbhs = ('cluster', ), timeLimit = 100 ):

    for inst in instNames:
        instAct = OISRC(inst)

        for nbhType in nbhs:
            
            instAct.run(
                outImportNeighborhoods=True,
                outImportedNeighborhoods=nbhType,
                outVerbose=False,
                outTimeLimitSeconds=timeLimit,
                writeResult=True
            )
        instAct = OISRC(inst)
        instAct.run(
            outImportNeighborhoods=True,
            outImportedNeighborhoods='cluster',
            outVerbose=False,
            outTimeLimitSeconds=timeLimit,
            outCallback='pure',
            writeResult=True
        )

if __name__ == '__main__':
    inst1 = OISRC(os.path.join('OISRCInstances', 'instance_45_6_400_3.oisrc'))
    inst1.run(
            outImportNeighborhoods=True,
            outImportedNeighborhoods='cluster',
            outVerbose=True,
            outTimeLimitSeconds=500,
            outCallback='vmnd',
            writeResult=True
    )
    #generateInstanceFile(400, 6, 45)