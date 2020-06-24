from random import sample, randint, sample
from Others import loadMPS
from functools import reduce


class Neighborhoods:

    def __init__(
        self,
        lowest = 1,
        highest = 5,
        keysList= [],
        randomSet=True,
        outerNeighborhoods = None,
        useFunction = False,
        funNeighborhoods = None):
        self.keysList = keysList
        self.lowest = lowest
        self.highest = highest
        self.neighborhoods = { i : {} for i in range(lowest, highest + 1) }
        if randomSet:
            self.createRandomNeighborhoods()
        else:
            self.importNeighborhoods(outerNeighborhoods)
        self._depth = lowest
        self.useFunction = useFunction
        self.funNeighborhoods = None
        if useFunction:
            self.funNeighborhoods = funNeighborhoods

    @property
    def depth(self, _depth):
        return self._depth
    
    @depth.setter
    def depth(self, new_val):
        if new_val <= self.lowest:
            print('Moving towards the {} level of depth.'.format(new_val))
            self._depth = new_val
        else:
            print('Can\'t go to upper neighborhoods.')

    def canIncNeigh(self):
        return self._depth + 1 <= self.highest

    def resetDepth(self):
        print('Depth set to first: {}'.format(self.lowest))
        self._depth = self.lowest

    def createRandomNeighborhoods(self):
        for depthAct in range(self.lowest, self.highest + 1):
            nParam = 3
            for actParam in range(nParam):
                
                freeVars = sample(self.keysList, int(len(self.keysList) * 0.01) + 2)

                self.neighborhoods[depthAct][actParam] = [var for var in self.keysList if var not in freeVars]

    def importNeighborhoods(self, outerNeighborhoods):
        self.neighborhoods = outerNeighborhoods

    def exportNeighborhood(self, name):
        
        strout = ''
        strout += str(name) +'\n'
        strout += str(self.lowest) +'\n'
        strout += str(self.highest) +'\n'
        for neighborhood in self.neighborhoods:
            strout+= str(neighborhood) + '\n'
            strout+= str(len(self.neighborhoods[neighborhood].keys())) + '\n'
            for param in self.neighborhoods[neighborhood]:
                strout+= str(param) + '\n'
                strout+= str(len(self.neighborhoods[neighborhood][param])) + '\n'
                strout += reduce(lambda x, y: str(x) + ' ' + str(y), self.neighborhoods[neighborhood][param])
                strout += '\n'
        file = open( name + '.txt', 'w')
        file.write(strout)
        file.close()


def genIRPneighborhoods(n, H, K):
    return {
        2 : {(kf, tf) : [ ('y', i, j, k, t) 
            for k in range(1, K + 1) for t in range(1, H + 1) for i in range(n + 1) for j in range(n + 1)
            if i < j and (k, t) != (kf, tf) ]
                for kf in range(1, K + 1) for tf in range(1, H + 1)
        },
        3: {
            tf : [ ('y', i, j, k, t) 
            for k in range(1, K + 1) for t in range(1, H + 1) for i in range(n + 1) for j in range(n + 1)
            if i < j and t != tf]
            for tf in range(1, H + 1)
        },
        
        4: {
            kf : [ ('y', i, j, k, t) 
            for k in range(1, K + 1) for t in range(1, H + 1) for i in range(n + 1) for j in range(n + 1)
            if i < j and k != kf ]
            for kf in range(1, K + 1)
        },
        5: {
            (tf1, tf2) : [ ('y', i, j, k, t) 
            for k in range(1, K + 1) for t in range(1, H + 1) for i in range(n + 1) for j in range(n + 1)
            if i < j and tf1 != t and tf2 != t]
            for tf1 in range(1, H + 1) for tf2 in range(1, H + 1) if tf1 < tf2
        }
    }

def genIRPneigh(n, H, K, stVarName = 'y'):
    return {
        2 : {(kf, tf) : [ '{}_{}_{}_{}_{}'.format(stVarName, i, j, k, t)
            for k in range(1, K + 1) for t in range(1, H + 1) for i in range(n + 1) for j in range(n + 1)
            if i < j and (k, t) != (kf, tf) ]
                for kf in range(1, K + 1) for tf in range(1, H + 1)
        },
        3: {
            tf : [ '{}_{}_{}_{}_{}'.format(stVarName, i, j, k, t)
            for k in range(1, K + 1) for t in range(1, H + 1) for i in range(n + 1) for j in range(n + 1)
            if i < j and t != tf]
            for tf in range(1, H + 1)
        },
        
        4: {
            kf : [ '{}_{}_{}_{}_{}'.format(stVarName, i, j, k, t)
            for k in range(1, K + 1) for t in range(1, H + 1) for i in range(n + 1) for j in range(n + 1)
            if i < j and k != kf ]
            for kf in range(1, K + 1)
        },
        5: {
            (tf1, tf2) : [ '{}_{}_{}_{}_{}'.format(stVarName, i, j, k, t)
            for k in range(1, K + 1) for t in range(1, H + 1) for i in range(n + 1) for j in range(n + 1)
            if i < j and tf1 != t and tf2 != t]
            for tf1 in range(1, H + 1) for tf2 in range(1, H + 1) if tf1 < tf2
        }
    }

def importNeighborhoods(fileName = 'Neighborhoods//IRP10.txt'):
    file = open(fileName, 'r')
    #lines = list(map(lambda x: x.rstrip('\n'), file.readlines()))
    
    nameFile = str(file.readline().rstrip('\n'))
    lowestFile = int(file.readline().rstrip('\n'))
    highestFile = int(file.readline().rstrip('\n'))
    neighborhoods = {}
    for n in range(lowestFile, highestFile + 1):
        actN = str(file.readline().rstrip('\n'))
        neighborhoods[n] = {}
        params = int(file.readline().rstrip('\n'))
        for paramNumber in range(params):
            actParam = str(file.readline().rstrip('\n'))
            numVars = int(file.readline().rstrip('\n'))
            fixedX = file.readline().rstrip('\n').split(' ')
            neighborhoods[n][actParam] = fixedX
    
    nhOutput = Neighborhoods(lowest = lowestFile, highest = highestFile, randomSet = False, outerNeighborhoods= neighborhoods)
    file.close()
    return nhOutput

if __name__ == '__main__':
    #m = loadMPS('MIPLIB//binkar10_1.mps')
    #ns1 = Neighborhoods(list(m._vars.keys()), randomSet=True)
    #nsIRP10.exportNeighborhood('IRP10')
    print(importNeighborhoods().neighborhoods[4]['2'])