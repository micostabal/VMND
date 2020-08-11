import unittest
import sys
import os
import pytest
import pprint
from collections import deque


class Test:

    def __init__(self, name):
        self.name = name
        self._totalCases = 0
        self._failures = 0

    def addCase(self, failure = True):
        self._totalCases += 1
        if failure:
            self._failures += 1
    
    def printResults(self):
        strout = ''
        if self._failures == 0:
            if self.name == 'testEnd':
                strout += '{}\t\t OK... TOTAL {}'.format(self.name, self._totalCases)
            else:
                strout += '{}\t OK... TOTAL {}'.format(self.name, self._totalCases)
        else:
            strout += '{}\t|FAILURES {} TOTAL {}'.format(self.name, self._failures, self._totalCases)
        print(strout)
        

class Log:

    def __init__(
        self,
        nameInstance = 'thisInstance',
        lazy = False,
        filePath = os.path.join('Logs', 'testfile.testlog'),
        nbhdLowest = 1,
        nbhdHighest = 1000
        ):
        self.name = nameInstance
        self.executed = False
        self.lazyConstraints = lazy
        self.nbhsLowest = nbhdLowest
        self.nbhsHighest = nbhdHighest
        self.incumbentBC = None
        self.lastLSObjective = 0
        self.inBC = True
        self.inLS = False
        self.LSimproved = False
        self.LSFirstExplored = True
        self.BCImproved = False
        self.currentNbhd = self.nbhsLowest
        self.events = []
        self.previousLog = None
        if filePath is not None:
            file = open(filePath)
            lines = list( filter(lambda x : x != '', file.read().split('\n') ) )
            file.close()
            self.events = deque(lines)

        testNames = [
            'testBegin',
            'testEnd',
            'testBeginBC',
            'testEndBC',
            'testBeginLS',
            'testEndLS',
            'testNewIncBC',
            'testNewIncLS',
            'testExploreNbhd'
        ]
        self.tests = { name : Test(name) for name in testNames}
        self.errors = []
        self.effectiveness = { i : { 'total': 0, 'improved' : 0 } for i in range(self.nbhsLowest, self.nbhsHighest + 1)}

    def LogAssertEqual(self, nameTest, val1, val2, msg = ''): # Assert Equal
        if val1 == val2:
            self.tests[nameTest].addCase(failure = False)
        else:
            self.tests[nameTest].addCase(failure = True)
            self.errors.append(nameTest + '::' + msg)

    def LogAssertTrue(self, nameTest, val, msg = ''):
        if val:
            self.tests[nameTest].addCase(failure = False)
        else:
            self.tests[nameTest].addCase(failure = True)
            self.errors.append(nameTest + '::' + msg)

    def LogAssertFalse(self, nameTest, val, msg = ''):
        if val:
            self.tests[nameTest].addCase(failure = True)
            self.errors.append(nameTest + '::' + msg)
        else:
            self.tests[nameTest].addCase(failure = False)

    def testBegin(self):
        self.LogAssertTrue('testBegin', True, msg= 'The program should have started.')
    
    def testBeginBC(self):
        if self.incumbentBC is not None:
            self.LogAssertFalse('testBeginBC', self.inBC, msg= 'It shouldn\'t be in BC.')
        self.LogAssertFalse('testBeginBC', self.inLS, msg= 'It shouldn\'t be in LS.')
        self.inBC = True
        self.BCImproved = False

    def testEndBC(self):
        self.LogAssertTrue('testEndBC', self.inBC, msg= 'It should be in BC.')
        self.LogAssertFalse('testEndBC', self.inLS, msg= 'It shouldn\'t be in LS.')
        self.inBC = False

    def testBeginLS(self):
        self.LogAssertFalse('testBeginLS', self.inBC, msg= 'It shouldn\'t be in BC.')
        self.LogAssertFalse('testBeginLS', self.inLS, msg= 'It shouldn\'t be in LS.')
        self.inLS = True
        self.LSimproved = False
    
    def testEndLS(self):
        self.LogAssertTrue('testEndLS', self.inLS, msg= 'It should be in LS.')
        self.LogAssertFalse('testEndLS', self.inBC, msg= 'It shouldn\'t be in BC.')

        self.LogAssertTrue(
            'testEndLS',
            (int(self.currentNbhd) == self.nbhsHighest) or self.LSimproved,
            msg = 'Local Search shouldn\'t have ended (no improvement and highest hasn\'t been explored).'
        )


        self.inLS = False
        self.BCImproved = False
        
    def testNewIncBC(self, newInc):
        if self.incumbentBC is not None:
            self.LogAssertTrue(
                'testNewIncBC',
                newInc <= self.incumbentBC,
                msg= 'New incumbent({}) should be lower than curretn incumbent ({})'.format(newInc, self.incumbentBC))
        self.BCImproved = True
        self.incumbentBC = newInc
    
    def testNewIncLS(self, newInc):
        self.LogAssertTrue(
            'testNewIncLS',
            newInc <= self.incumbentBC,
            msg= 'The Local Search objective {} should be lower than the current incumbent.'.format(newInc, self.incumbentBC)
        )
        self.LSimproved = True

    def testExploreNbhd(self, newNbhd):
        self.LogAssertTrue(
            'testExploreNbhd',
            self.nbhsHighest >= int(newNbhd),
            msg = 'Neighborhood Explored ({}) mustn\'t be greater than the highest {}.'.format(newNbhd, self.nbhsHighest)
        )
        self.LogAssertTrue(
            'testExploreNbhd',
            self.nbhsLowest <= int(newNbhd),
            msg = 'Neighborhood Explored ({}) mustn\'t be lower than the lowest ({}).'.format(newNbhd, self.nbhsLowest)
        )

        self.LogAssertFalse(
            'testExploreNbhd',
            self.LSimproved,
            msg = 'Local Search already improved objective, BC should have been resumed.'
        )

        if not self.BCImproved and int(newNbhd) == self.nbhsLowest:
            self.LogAssertTrue(
                'testExploreNbhd',
                False,
                msg = 'Entering Local Search without improving in BC phase.'
            )

        if self.BCImproved:
            self.LogAssertTrue(
                'testExploreNbhd',
                int(newNbhd) == self.nbhsLowest,
                msg = 'After improving in BC, Local Search should have started in Lowest Neighborhood.'
            )
            self.BCImproved = False
        elif newNbhd != self.nbhsHighest and self.currentNbhd != self.nbhsHighest:
            self.LogAssertTrue(
                'testExploreNbhd',
                int(self.currentNbhd) + 1 == int(newNbhd),
                msg = 'Neighborhood Exploration should have increased in 1 (not from {} to {}).'.format(self.currentNbhd, newNbhd)
            )
        else:
            self.LogAssertTrue(
                'testExploreNbhd',
                self.currentNbhd != self.nbhsHighest,
                msg = 'The current Negihborhood shouln\'t be the highest.'
            )
        self.currentNbhd = newNbhd
        
    def testEnd(self):
        self.LogAssertTrue('testEnd', True, msg= 'The execution of the program should have ended.')

    def run(self, printState = True):

        while (len(self.events) > 0):

            thisEvent = self.events.popleft()
            elements = thisEvent.split(' ')

            typeLog = elements[0]

            if typeLog == 'LOWEST':
                self.nbhsLowest = int(elements[1])
            elif typeLog == 'HIGHEST':
                self.nbhsHighest = int(elements[1])
            elif typeLog == 'BC':
                if elements[1] == 'BEGIN':
                    self.testBeginBC()
                elif elements[1] == 'NEWINCUMBENT':
                    newInc = float(elements[2])
                    self.testNewIncBC(newInc)
                elif elements[1] == 'END':
                    self.testEndBC()
            elif typeLog == 'LS':
                if elements[1] == 'BEGIN':
                    self.testBeginLS()
                elif elements[1] == 'NEWINCUMBENT':
                    nbhd = int(elements[2])
                    newInc = float(elements[3])
                    self.effectiveness[int(newNbhd)]['improved'] += 1
                    self.testNewIncLS(newInc)
                elif elements[1] == 'EXPLORE':
                    newNbhd = elements[2]
                    self.effectiveness[int(newNbhd)]['total'] += 1
                    self.testExploreNbhd(newNbhd)
                elif elements[1] == 'END':
                    self.testEndLS()
            elif typeLog == 'GEN':
                if elements[1] == 'BEGIN':
                    self.testBegin()
                elif elements[1] == 'END':
                    self.testEnd()
            else:
                raise(AssertionError('Testlog File format is not correct.'))

            if printState:
                pp = pprint.PrettyPrinter(indent = 4)
                print('---- Summary ---- {}'.format(elements))
                pp.pprint('inBC: {}'.format(self.inBC))
                pp.pprint('inLS: {}'.format(self.inLS))
                pp.pprint('incumbent: {}'.format(self.incumbentBC))
                pp.pprint('current Nbhd: {}'.format(self.currentNbhd))
                pp.pprint('Improved BC: {}'.format(self.BCImproved))
                print('-----------------')

            self.previousLog = thisEvent
        
        self.executed =True
    
    def printResults(self):
        if not self.executed:
            print('The run Method must be executed first!')
            return(0)

        # The results are printed...
        print('------ VMND TESTING --------')
        for test in self.tests:
            self.tests[test].printResults()
        print('---VMND FAILURES/WARNINGS---')
        for error in filter(lambda x : x != '', self.errors):
            print(error)
        print('-- NEIGHBORHOOD\'S EFFECTIVENESS --')
        for nbh in self.effectiveness:
            strNbh = 'NBH {}: -- TOTAL EXPLORED : {} -- TOTAL IMPROVEMENTS : {}'.format(
                nbh,
                self.effectiveness[nbh]['total'], 
                self.effectiveness[nbh]['improved']
            )
            print(strNbh)


if __name__ == '__main__':
    log1 = Log(filePath=os.path.join('Logs', 'binkar10_1.testlog'), nbhdHighest= 5, nbhdLowest= 1)
    log1.run(printState=False)
    log1.printResults()
    
    