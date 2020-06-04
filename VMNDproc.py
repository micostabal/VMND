from gurobipy import *
import numpy as np
from ConComp import getSubsets
from Neighborhood import Neighborhoods, genIRPneighborhoods, genIRPneigh
from Others import loadMPS
from Functions import transformKey
from Cuts import genSubtourLazy, Cut, getCheckSubTour
import time
import os

def checkVariables(modelVars, neighborhoods):
    totalVarsN = set()
    setModelVars = set(modelVars)
    for n in neighborhoods.keys():
        for param in neighborhoods[n]:
            totalVarsN = totalVarsN.union(set(neighborhoods[n][param]))

    if len(totalVarsN.difference(setModelVars)) > 0:
        return False
    else:
        return True

def SubtourElimCallback(model, where):

    if where == GRB.Callback.MIPSOL:
        if model._addLazy and model._funLazy is not None:
            model._vals = model.cbGetSolution(model._vars)
            vals = model.cbGetSolution(model._vars)
            newLazy = model._funLazy(model._vals)

            if len(newLazy) > 0:
                for cut in newLazy:
                    model.cbLazy( quicksum( model._vars[key] * cut.nonzero[key] for key in cut.nonzero.keys() )
                    , model._senseDict[cut.sense], cut.rhs)
                    model._BCLazyAdded.append(cut)


            if time.time() - model._LastGapTime >= 10:

                bound = GRB.Callback.MIPSOL_OBJBST
                obj = GRB.Callback.MIPSOL_OBJBND
                gap = abs(bound - obj) / abs(bound)

                model._gapsTimes.append( (time.time() - model._initialTime, gap ) )
                model._LastGapTime = time.time()

def VMNDCallback(model, where):

    if where == GRB.Callback.MIPSOL:
        # Necessary (subtour) cuts need to be added.
        if model._addLazy and model._funLazy is not None:
            
            vals = model.cbGetSolution(model._vars)
            model._vals = vals
            v1 = {}
            for varname in model._vars.keys():
                v1[varname] = vals[varname]
            model._BCVals = v1

            newLazy = model._funLazy(model._vals)
            
            if len(newLazy) > 0:
                for cut in newLazy:
                    model.cbLazy( quicksum( model._vars[key] * cut.nonzero[key] for key in cut.nonzero.keys() )
                    , model._senseDict[cut.sense], cut.rhs)
                    model._BCLazyAdded.append(cut)
        
        if not model._incFound:
            model._incFound = True
            model._restrTime = True
            model._BCLastObj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            model._vals = model.cbGetSolution(model._vars)

            ## Here we generate heuristic local search solutions.
            if model._verbose:
                print('Starting Local Search Phase')
            starting_local_search = time.time()
            localSearch(model)
            model._LSLastTime = time.time() - starting_local_search

            if model._verbose:
                print('Local Search Phase is finished.')

            #print('Starting B&C Search')
            # Time starting new B&C phase
            model._BCLastStart = time.time()
        
        if time.time() - model._LastGapTim >= 10:
            model._gapsTimes.append( (time.time() - model._initialTime, model.getAttr(GRB.Attr.MIPGap) ) )
            model._LastGapTime = time.time()

    # Check B&C time.
    tactBC = (time.time() - model._BCLastStart)

    # Time is being restricted and BC phase hasn't reached its timelimit and we are now in a MIP Node.
    if where == GRB.Callback.MIPNODE and model._incFound and (not model._restrTime or
     (model._restrTime and tactBC <= model._alpha * model._LSLastTime )):
        # We inject heuristic solution to B&C procedure.
        if model._LSImprSols is not None and model._restrTime:
            
            for key in model._LSImprovedDict:
                model.cbSetSolution(model.getVarByName(key), model._LSImprovedDict[key])

                #model._BCHeuristicCounter += 1
                #model.cbSetSolution( model.getVars(), model._LSImprSols)
                #objval = model.cbUseSolution()

    # Time is being restricted and BC phase hasn't reached its timelimit and we have found a MIP Solution.
    if where == GRB.Callback.MIPSOL and model._incFound and (not model._restrTime or (model._restrTime and tactBC <= model._alpha * model._LSLastTime )):
        
        if model.cbGet(GRB.Callback.MIPSOL_OBJ) < model._BCLastObj and model._LSNeighborhoods._depth > model._LSNeighborhoods.lowest:
            if model._verbose:
                print("Valid B&C improvement.")
            model._LSNeighborhoods.resetDepth()
            model._restrTime = True
                
        model._BCLastObj = model.cbGet(GRB.Callback.MIPSOL_OBJ)

    # We are now in an arbitrary place in the search tree and B&C timelimit has already ocurred.
    if model._incFound and model._restrTime and tactBC > model._alpha * model._LSLastTime:

        if where == GRB.Callback.MIPSOL:
            if model.cbGet(GRB.Callback.MIPSOL_OBJ) < model._BCLastObj and model._LSNeighborhoods._depth > model._LSNeighborhoods.lowest:
                if model._verbose:
                    print("Valid B&C improvement.")
                model._LSNeighborhoods.resetDepth()
                model._restrTime = True

            if model._BCLastObj > model.cbGet(GRB.Callback.MIPSOL_OBJ):
                model._restrTime = True
            model._BCLastObj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                
        # Local search is performed again and parameters are updated.
        if model._verbose:
            print('Starting Local Search Phase')
        starting_local_search = time.time()
        localSearch(model)
            
        if model._LSNeighborhoods._depth == model._LSNeighborhoods.highest and not model._LSImproved:
            model._restrTime = False
        
        model._LSLastTime = time.time() - starting_local_search

        if model._verbose:
            print('Starting B&C Search')
        
        # Time starting new B&C phase
        model._BCLastStart = time.time()

    # Final Subtour Check
    if model._addLazy and model._funLazy is not None and where == GRB.Callback.MIPSOL:
    
        vals = model.cbGetSolution(model._vars)
        model._vals = vals
        v1 = {}
        for varname in model._vars.keys():
            v1[varname] = vals[varname]
        model._BCVals = v1

        newLazy = model._funLazy(model._vals)
        
        if len(newLazy) > 0:
            for cut in newLazy:
                model.cbLazy( quicksum( model._vars[key] * cut.nonzero[key] for key in cut.nonzero.keys() )
                , model._senseDict[cut.sense], cut.rhs)
                model._BCLazyAdded.append(cut)

def localSearch(model):
    model._LSImproved = False
    locModel = loadMPS(model._path)
    locModel.setParam('OutputFlag', 0)
    premodelconstrs = locModel.NumConstrs

    ## Model Variables are set within its dictionary
    locModelVars = {}
    for var in locModel.getVars():
        locModelVars[var.VarName] = var
    locModel._vars = locModelVars

    # We add previously added cuts.
    if len(model._BCLazyAdded) > 0:
        if model._verbose:
            print('NewSubcuts Detected to be added to local search')
        for cut in model._BCLazyAdded:
            locModel.addConstr( quicksum( locModel._vars[key] * cut.nonzero[key] for key in cut.nonzero.keys() )
            , model._senseDict[cut.sense], cut.rhs)
        locModel.update()

    outerConstrs = locModel.NumConstrs
    if outerConstrs == premodelconstrs and model._verbose:
        pass
        #print('--------- POSSIBLE ERROR 0!! -------------')

    for act_depth in model._LSNeighborhoods.neighborhoods:
        if model._verbose:
            print('Searching in depth : {}'.format(act_depth))
        model._LSNeighborhoods._depth = act_depth
        for param_act in model._LSNeighborhoods.neighborhoods[act_depth]:
            if model._verbose:
                print('------- Searching in {} parametrization of depth : {} -------'.format(param_act, act_depth))
            locModel.reset()


            innerconstrs = locModel.NumConstrs
            if innerconstrs != outerConstrs and model._verbose:
                print('--------- CONSTRAINT ERROR 1!! -------------')
            
            # Contains the string keys of the fixed variables
            addedFixedVarsKeys = []

            for keyAct in model._LSNeighborhoods.neighborhoods[act_depth][param_act]:
                locModel.addConstr(locModel._vars[keyAct] == model._BCVals[keyAct], name=keyAct)
                addedFixedVarsKeys.append(keyAct)
            locModel.update()
            
            if outerConstrs >= locModel.NumConstrs and verbose:
                print('--------- CONSTRAINT ERROR 2!! -------------')

            locModel.optimize()

            if locModel.status == GRB.OPTIMAL:
                if model._verbose:
                    print('Local Search Phase has feasible solution')
                if model._BCLastObj > locModel.objVal:
                    model._LSImproved = True
                    model._BCHeuristicCounter = 0
                    model._LSImprSols = locModel.getAttr('X')
                    model._LSImprovedDict = {}

                    totalvars = len(model._vars.keys())
                    distinct = 0
                    for key in locModel._vars.keys():
                        model._LSImprovedDict[key] = locModel._vars[key].X
                        if locModel._vars[key].X != model._BCVals[key]:
                            distinct += 1

                    model._LSNeighborhoods._depth = act_depth
                    if model._verbose:
                        print( 'MIP Incumbent: {} --'.format(model._BCLastObj) + 'Local Search Objective: {}'.format(locModel.objVal))
                        print('--------- Changed {} variables from {}, a  {}% ----------'.format(
                         distinct, totalvars, round(100 * distinct/totalvars, 4)))
                    break
            
            for fixedVar in addedFixedVarsKeys:
                cAct = locModel.getConstrByName(fixedVar)
                locModel.remove(cAct)
            locModel.update()

            finalconstrs = locModel.NumConstrs
            if finalconstrs != outerConstrs and model._verbose:
                print('--------- CONSTRAINT ERROR 3!! -------------')
        
        if model._LSImproved: break
    
    if model._verbose:
        if model._LSImproved:
            print('--- Objective was reduced ---')
        else:
            print('--- Objective was not reduced ---')
        print('Finished Local Search phase')

    if model._LSNeighborhoods._depth == model._LSNeighborhoods.highest and not model._LSImproved:
        if model._verbose:
            print('Local Search phase lead to no improvement.')
        model._restrTime = False

def solver(
    path,
    verbose =False,
    addlazy = False,
    funlazy = None,
    importNeighborhoods = False,
    importedNeighborhoods = None,
    funTest = None,
    callback = 'vmnd',
    alpha = 2
    ):
    model = Model()
    model = loadMPS(path)

    modelVars = {}
    for var in model.getVars():
        modelVars[var.VarName] = var
    model._vars = modelVars

    # Model VMND attributes.
    model._alpha = alpha
    model._incFound = False
    model._restrTime = True
    model._vals = None
    model._realObj = 0
    model._path = path
    model._verbose = verbose
    model._initialTime = time.time()
    model._LastGapTime = time.time()
    model._gapsTimes = []
    

    # Local Search Attriutes.
    if not importNeighborhoods or importedNeighborhoods is None:
        model._LSNeighborhoods = Neighborhoods(lowest = 1, highest = 5, keysList = model._vars.keys(), randomSet=True)
    else:
        model._LSNeighborhoods = importedNeighborhoods
        
    model._LSLastTime = 1000
    model._LSImproved = False
    model._LSImprovedDict = None
    model._LSImprSols = None
    

    # Branch and Cut Attributes.
    model._addLazy = addlazy
    model._funLazy = funlazy
    model._BCLastStart = time.time()
    model._BCLastObj = 0
    model._BCVals = None
    model._BCHeuristicCounter = 0
    model._BCimproved = False
    model._BCLazyAdded = []
    model._senseDict = {
        '<=' : GRB.LESS_EQUAL,
        '==' : GRB.EQUAL,
        '>=' : GRB.GREATER_EQUAL
    }
    model._line = ''

    allTestsPassed = False
    if importedNeighborhoods is not None and importNeighborhoods:
        if checkVariables(model._vars.keys(), importedNeighborhoods.neighborhoods):
            print('Variables are in accordance with their neighborhoods.')
            allTestsPassed = True
        else:
            print('[TEST] Neighborhood\'s variable not belonging to model varaibles found.')    

    if allTestsPassed or not importNeighborhoods:
        model.setParam("LazyConstraints", 1)
        model.setParam('MIPFocus', 3)
        if not verbose:
            model.setParam('OutputFlag', 0)
        if callback == 'vmnd':
            model.optimize(VMNDCallback)
        else:
            model.optimize(SubtourElimCallback)

        model._line += ' ,RUNTIME : {}, '.format(round(model.RUNTIME, 3))
        
        if funTest is not None:
            outputVals = { var.VarName : var.X for var in model.getVars() if var.X > 0}
            correct = funTest(outputVals)
            if correct:
                model._line += 'SUBTOUR : CORRECT'
            else:
                model._line += 'SUBTOUR : ERROR'
        

    return model

def creator(path):
    return loadMPS(path)

def runSeveral(heuristic = 'vmnd'):
    for nodes in [5]:
        for vers in [1, 2, 3]:
            for i in range(1, 5):
                path = os.path.join('MIPLIB', 'abs{}n{}_{}.mps'.format(vers, nodes, i))
                line = path
                n , H, K = nodes, 3, 2
                nsAct = Neighborhoods(lowest = 2, highest = 5, randomSet = False, outerNeighborhoods = genIRPneigh(n, H, K))
                mout = solver (
                        path,
                        addlazy= True,
                        funlazy= genSubtourLazy(n, H, K),
                        importNeighborhoods= True,
                        importedNeighborhoods= nsAct,
                        funTest= getCheckSubTour(n, H, K),
                        alpha = 2,
                        callback = heuristic,
                        verbose = True
                )
                if mout .status == GRB.OPTIMAL:
                    file = open('{}results.txt'.format(heuristic.upper()), 'a')
                    line += mout._line + '\n'
                    file.write(line.lstrip('MIPLIB//'))
                    file.close()
                    print('Finished Model {}'.format(line.lstrip('MIPLIB//')))
                else:
                    file = open('{}results.txt'.format(heuristic.upper()), 'a')
                    line += ' OPTIMZLITY WAS NOT REACHED' + '\n'
                    file.write(line.lstrip('MIPLIB//'))
                    file.close()

def compareGaps(path):
    # Initial parameters
    n, H, K = 15, 3, 2

    # Neighborhoods are set
    nsAct = Neighborhoods(lowest = 2, highest = 5, randomSet = False, outerNeighborhoods = genIRPneigh(n, H, K))
    # Mdoel is executed
    vmndModel = solver(
        path,
        addlazy = True,
        funlazy = genSubtourLazy(n, H, K),
        importNeighborhoods = True,
        importedNeighborhoods = nsAct,
        funTest = getCheckSubTour(n, H, K),
        alpha = 3,
        callback = 'pure',
        verbose = True
        )
    vmndGapList = vmndModel._gapsTimes
    print(vmndGapList)





if __name__ == '__main__':
    #runSeveral('vmnd')

    compareGaps(os.path.join('MIPLIB', 'abs1n15_3.mps'))