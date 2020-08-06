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
                strout += '{}\t\t| OK... |TOTAL {}|'.format(self.name, self._totalCases)
            else:
                strout += '{}\t| OK... |TOTAL {}|'.format(self.name, self._totalCases)
        else:
            strout += '{}\t|FAILURES {}| TOTAL {}|'.format(self.name, self._failures, self._totalCases)
        print(strout)
        

class Log:

    def __init__(self, nameInstance = 'thisInstance', lazy = False, filePath = 'testfile.testlog'):
        self.name = nameInstance
        self.lazyConstraints = lazy
        self.nbhsLowest = 0
        self.nbhsHighest = 5
        self.incumbentBC = None
        self.lastLSObjective = 0
        self.inBC = True
        self.inLS = False
        self.LSimproved = False
        self.BCImproved = False
        self.currentNbhd = self.nbhsLowest
        self.events = []
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

    def LogAssertEqual(self, nameTest, val1, val2): # Assert Equal
        if val1 == val2:
            self.tests[nameTest].addCase(failure = False)
        else:
            self.tests[nameTest].addCase(failure = True)

    def LogAssertTrue(self, nameTest, val):
        if val:
            self.tests[nameTest].addCase(failure = False)
        else:
            self.tests[nameTest].addCase(failure = True)

    def LogAssertFalse(self, nameTest, val):
        if val:
            self.tests[nameTest].addCase(failure = True)
        else:
            self.tests[nameTest].addCase(failure = False)

    def testBegin(self):
        self.LogAssertTrue('testBegin', True)
    
    def testBeginBC(self):
        if self.incumbentBC is not None:
            self.LogAssertFalse('testBeginBC', self.inBC)
        self.LogAssertFalse('testBeginBC', self.inLS)
        self.inBC = True
        self.BCImproved = False

    def testEndBC(self):
        self.LogAssertTrue('testEndBC', self.inBC)
        self.LogAssertFalse('testEndBC', self.inLS)
        self.inBC = False

    def testBeginLS(self):
        self.LogAssertFalse('testBeginLS', self.inBC)
        self.LogAssertFalse('testBeginLS', self.inLS)
        self.inLS = True
    
    def testEndLS(self):
        self.LogAssertTrue('testEndLS', self.inLS)
        self.LogAssertFalse('testEndLS', self.inBC)
        self.inLS = False
        
    def testNewIncBC(self, newInc):
        if self.incumbentBC is not None:
            self.LogAssertTrue('testNewIncBC', newInc <= self.incumbentBC)
            self.BCImproved = True
        else:
            self.incumbentBC = newInc
    
    def testNewIncLS(self, newInc):
        self.LogAssertTrue('testNewIncLS', newInc <= self.incumbentBC)

    def testExploreNbhd(self, newNbhd):
        if self.BCImproved:
            self.LogAssertEqual('testExploreNbhd', newNbhd, self.nbhsLowest)
        else:
            self.LogAssertTrue('testExploreNbhd', self.currentNbhd != self.nbhsHighest )

        self.currentNbhd = newNbhd
    
    def testEnd(self):
        self.LogAssertTrue('testEnd', True)

    def run(self):

        while (len(self.events) > 0):
            thisEvent = self.events.popleft()
            elements = thisEvent.split(' ')

            typeLog = elements[0]

            if typeLog == 'BC':
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
                    self.testNewIncLS(newInc)
                elif elements[1] == 'EXPLORE':
                    newNbhd = elements[2]
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
        
        # The results are printed...
        for test in self.tests:
            self.tests[test].printResults()




if __name__ == '__main__':
    log1 = Log()
    log1.run()

    
    