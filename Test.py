import unittest
from gurobipy import *
from Neighborhood import Neighborhoods, genIRPneigh
from Cuts import genSubtourLazy
from VMNDproc import solver
from ConComp import getSubsets
from Functions import transformKey
from Cuts import getCheckSubTour


class VMNDTest(unittest.TestCase):
    
    def setUp(self):
        self.n = 5
        self.H = 3
        self.K = 2
        self.nbh = Neighborhoods(lowest = 2, highest = 5, randomSet = False, outerNeighborhoods = genIRPneigh(self.n, self.H, self.K))

        self.path = 'MIPLIB//abs1n{}_{}.mps'.format(5, 1)
        self.finalModel = None

    def subtourTestCase(self):
        for instance in range(1, 5):
            self.path = 'MIPLIB//abs1n{}_{}.mps'.format(5, instance)
            self.finalModel = solver(
            self.path,
            addlazy= True,
            funlazy= genSubtourLazy(self.n, self.H, self.K),
            importNeighborhoods= True,
            importedNeighborhoods= self.nbh,
            funTest= getCheckSubTour(self.n, self.H, self.K),
            alpha = 2,
            callback = 'vmnd',
            verbose = False
            )

            outputVals = { var.VarName : var.X for var in self.finalModel.getVars() if var.X > 0}
            self.assertEqual(getCheckSubTour(self.n, self.H, self.K)(outputVals), True, 'SUBTOUR ERRORS IN SOLUTION' )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(VMNDTest('subtourTestCase'))
    return suite


if __name__ == '__main__':
    pass
    #runner = unittest.TextTestRunner()
    #runner.run(suite())
    