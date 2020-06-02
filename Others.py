from gurobipy import *
import numpy as np
#from itertools import chain, combinations
#from ConComp import getSubsets
#from Neighborhood import genIRPneighborhoods
import time
from random import choice


def loadMPS(path = 'MIPLIB//binkar10_1.mps'):
    model = Model()
    model = read(path)
    variables = model.getVars()
    
    dictVars = {}
    for var in variables:
        #Binary Type of Variable
        if (var.VType == 'I' and var.LB == 0 and var.UB == 1) or var.VType == 'B':
            var.VType = 'B'
            dictVars[var.VarName] = model.addVar(name = var.VarName, vtype = GRB.BINARY)
        #Integer Type of Variable
        elif var.VType == 'I' and (var.LB, var.UB) != (0, 1):
            dictVars[var.VarName] = model.addVar(var.LB, var.UB, name = var.VarName, vtype = GRB.INTEGER)
        #Continuous Type of Variable
        elif var.VType == 'C':
            dictVars[var.VarName] = model.addVar(var.LB, var.UB, name = var.VarName, vtype = GRB.CONTINUOUS)
        else :
            print('------ Format Error!! ------')
    
    model._vars = dictVars
    return model



if __name__ == '__main__':
    loadMPS('MIPLIB//drayage-100-23.mps')