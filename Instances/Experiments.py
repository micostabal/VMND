import os
import sys
from IRP import runSeveralIRP
from IRPCS import runSeveralIRPCS
from MVRPD import runSeveralMVRPD
from VRP import runSeveralVRP
from OISRC import runSeveralOISRC

createdMIP = ['IRP', 'IRPCS', 'MVRPD', 'OISRC', 'VRP']

def checkfiles(listFiles):
    for link in listFiles:
        if not os.path.isfile(link):
            return False
    return True

def loadExperiment(path = os.path.join(os.path.pardir, 'Experiments', 'exptest.txt')):
    paths = {prob : [] for prob in createdMIP}
    lines = list( map( lambda x: x.rstrip('\n'), filter(lambda y : y != '\n' and y != '', open(path, 'r').readlines() ) ) )

    for line in lines:
        elements = line.split(' ')
        probType = elements[0]
        if probType not in createdMIP:
            print('Problem Name is not correct: {}'.format(probType))
            continue
        
        thisPath = os.path.join(*elements[1:])

        if not os.path.exists(thisPath):
            print('Path does not exist: {}'.format(thisPath))
            continue
        
        paths[probType].append(thisPath)
    return paths


class Experiment:

    def __init__(self, pathFile, totalTime = 7200):
        self.pathInstances = loadExperiment(pathFile)
        self.totalTime = totalTime

    def runSeveral(self):
        for prob in self.pathInstances.keys():
            if prob == 'IRP':
                runSeveralIRP(
                    self.pathInstances[prob],
                    nbhs = ('separated', 'function'),
                    timeLimit = self.totalTime,
                    includePure = True
                )
            elif prob == 'IRPCS':
                runSeveralIRPCS(
                    self.pathInstances[prob],
                    nbhs = ('separated', 'function'),
                    timeLimit = self.totalTime,
                    outVtrunc = 70,
                    outHtrunc = 3,
                    outKtrunc = 12,
                    includePure = True
                )
            elif prob == 'VRP':
                print('VRP is not yet implemented')
            elif prob == 'OISRC':
                runSeveralOISRC(
                    self.pathInstances[prob],
                    timeLimit = self.totalTime
                )
            elif prob == 'MVRPD':
                runSeveralMVRPD(
                    self.pathInstances[prob],
                    nbhs = ('separated', 'function'),
                    timeLimit = self.totalTime,
                    includePure = True
                )
            else:
                print('Problem name not in the list')


if __name__ == '__main__':
    arguments = sys.argv
    if len(arguments) > 1:
        fileName = arguments[1]
        print(fileName)
        firstExperiment = Experiment(pathFile = os.path.join(os.path.pardir, 'Experiments', fileName))
        firstExperiment.runSeveral()