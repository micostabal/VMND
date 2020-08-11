from gurobipy import *
import numpy as np
from math import trunc
from ConComp import getSubsets
from Neighborhood import Neighborhoods, varClusterFromMPS
from Others import loadMPS
from Functions import transformKey, genClusterNeighborhoods
from Cuts import genSubtourLazy, Cut, getCheckSubTour 
import matplotlib.pyplot as plt
import time
import os

def testlogUpdate(path, newLog):
    testLogFile = open(path, 'a')
    testLogFile.write('\n' + newLog)
    testLogFile.close()

def checkVariables(modelVars, neighborhoods):
    totalVarsN = set()
    setModelVars = set(modelVars)

    if neighborhoods.useFunction:
        totalVarsN = set(neighborhoods.keysList)
    else:
        for n in neighborhoods.neighborhoods.keys():
            for param in neighborhoods.neighborhoods[n]:
                totalVarsN = totalVarsN.union(set(neighborhoods.neighborhoods[n][param]))

    if len(totalVarsN.difference(setModelVars)) > 0:
        return False
    else:
        return True

def GapTimePlot(gapsTimes):
    plt.plot(
        [elem[1] for elem in gapsTimes],
        [elem[0] for elem in gapsTimes],
        scaley=True,
        scalex = True
    )
    plt.title('Gap (%) vs Time (sec)')
    plt.ylabel('Gap (%)')
    plt.xlabel('Time (s)')
    plt.show()

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

    if where == GRB.Callback.MIP:
        if time.time() - model._LastGapTime >= 5 and model._verbose and model._plotGapsTimes:

            try:
                bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                bst = model.cbGet(GRB.Callback.MIP_OBJBST)
                gap = abs(bst - bnd)/ bst

                model._gapsTimes.append( ( round(gap, 8) , round( model.cbGet(GRB.Callback.RUNTIME), 3)) )
                model._LastGapTime = time.time()
            except:
                print('Can\'t get this parameters with this comands')

def VMNDCallback(model, where):

    if where == GRB.Callback.MIPSOL:

        # Variables are stored.
        vals = model.cbGetSolution(model._vars)
        model._vals = vals


        if time.time() - model._LastGapTime >= 5 and model._verbose and model._plotGapsTimes:

            bnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            bst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            gap = abs(bst - bnd)/ bst

            model._gapsTimes.append( ( round(gap, 8) , round( model.cbGet(GRB.Callback.RUNTIME), 3)) )
            model._LastGapTime = time.time()


        bestObj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        thisObj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        
        # Necessary LAZY CONSTRAINTS needed.
        if model._addLazy and model._funLazy is not None:
            newLazy = model._funLazy(model._vals)

            if len(newLazy) > 0:
                for cut in newLazy:
                    model.cbLazy( quicksum( model._vars[key] * cut.nonzero[key] for key in cut.nonzero.keys() )
                    , model._senseDict[cut.sense], cut.rhs)
                    model._BCLazyAdded.append(cut)

            else:
                if bestObj >= thisObj:
                    model._newBest = True
                    if model._verbose:
                        print('-- NEW B&C INCUMBENT FOUND -- INC :{} --'.format(thisObj))

                    if model._writeTestLog:
                        testlogUpdate(model._testLogPath, 'BC NEWINCUMBENT {}'.format(round(thisObj, 7)) )

        else:
            if bestObj >= thisObj:
                model._newBest = True

                if model._writeTestLog:
                        testlogUpdate(model._testLogPath, 'BC NEWINCUMBENT {}'.format(round(thisObj, 7)) )
                
                if model._verbose:
                    print('-- NEW B&C INCUMBENT FOUND -- INC :{} --'.format(thisObj))
        
        # Parameter are initialized with the first "Integer" Incumbent.
        if not model._incFound:
            v1 = {}
            for varname in model._vars.keys():
                v1[varname] = vals[varname]
            model._BCVals = v1

            model._incFound = True
            model._restrTime = True
            model._IncBeforeLS = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            model._vals = model.cbGetSolution(model._vars)

            if model._writeTestLog:
                testlogUpdate(model._testLogPath, 'BC END')

            #### Local search is performed ####
            localSearch(model)

            if model._writeTestLog:
                testlogUpdate(model._testLogPath, 'BC BEGIN')

            if model._verbose:
                print('Starting B&C Search')
            # Time starting new B&C phase
            model._BCLastStart = time.time()
        else:
            if model._newBest:
                
                # Time is restricted.
                if not model._restrTime:
                    model._restrTime = True
                    if model._verbose:
                        print("Valid B&C improvement, time will be restricted.")

                ## New incumbent is stored inside BCVals
                v1 = {}
                for varname in model._vars.keys():
                    v1[varname] = vals[varname]
                model._BCVals = v1
                
                # If the current depth is not the lowest, it is reset.
                if model._LSNeighborhoods._depth > model._LSNeighborhoods.lowest:
                    model._LSNeighborhoods.resetDepth()

                # The varaible newBest si set to False
                model._newBest = False
    
    # Check B&C time.
    tactBC = (time.time() - model._BCLastStart)
    # The time in B&C must be at least the minimum provided by the user (minBCTime) and alpha times the last LS phase.
    totalTimeBC = max(model._alpha * model._LSLastTime, model._minBCTime )

    # A MIP NODE is being explored.
    if where == GRB.Callback.MIPNODE and model._incFound and (not model._restrTime or
     (model._restrTime and tactBC <= totalTimeBC )):

        # We set heuristic solution to B&C procedure.
        if model._LSImproved and model._LSImprovedDict is not None:

            ## Error handilng if: Checks whether the improved and current variables have the same "length"
            if len(model._LSImprovedDict.keys()) != len(model._vars.keys()):
                print('No match between vars and LS solutions')
            else:
                for key in model._vars.keys():
                    model.cbSetSolution(model.getVarByName(key), model._LSImprovedDict[key])
                    
                # Solution is inmediatly used.
                model.cbUseSolution()

                # We stop suggesting this solution.
                model._LSImprovedDict = None

    # Timelimit has already ocurred.
    if model._incFound and model._restrTime and tactBC > totalTimeBC and ( where == GRB.Callback.MIPNODE or where == GRB.Callback.MIPSOL ):

        # Incumbent before last Local search is updated. In MIPNODE or MIPSOL
        if where == GRB.Callback.MIPSOL:
            model._IncBeforeLS = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        if where == GRB.Callback.MIPNODE:
            model._IncBeforeLS = model.cbGet(GRB.Callback.MIPNODE_OBJBST)

        # Local Search must be performed again.
        if where == GRB.Callback.MIPNODE or where == GRB.Callback.MIPSOL:
            if model._writeTestLog:
                testlogUpdate(model._testLogPath, 'BC END' )

            #### Local Search is performed. ####
            localSearch(model)

            if model._writeTestLog:
                testlogUpdate(model._testLogPath, 'BC BEGIN' )

            # B&C is started again.
            if model._verbose:
                print('Starting B&C Search')
            
            # Time starting new B&C phase
            model._BCLastStart = time.time()

def localSearch(model):

    # Time is measured
    if model._verbose:
        print('Starting Local Search Phase')
    starting_local_search = time.time()

    if model._writeTestLog:
        testlogUpdate(model._testLogPath, 'LS BEGIN' )

    # Improved is always set to false.
    model._LSImproved = False
    
    
    # Model is loaded.
    locModel = loadMPS(model._path)
    locModel.setParam('OutputFlag', 0)
    premodelconstrs = locModel.NumConstrs

    ## Model Variables are set within its dictionary
    locModelVars = {}
    for var in locModel.getVars():
        locModelVars[var.VarName] = var
    locModel._vars = locModelVars

    if len(locModel._vars.keys()) != len(model._vars.keys()):
        print('ERROR LOCMODEL AND MODEL VARS ARE NOT THE SAME')

    # We add previously added cuts.
    if len(model._BCLazyAdded) > 0:
        if model._verbose:
            print('NewSubcuts Detected to be added to local search')
        for cut in model._BCLazyAdded:
            locModel.addConstr( quicksum( locModel._vars[key] * cut.nonzero[key] for key in cut.nonzero.keys() )
            , model._senseDict[cut.sense], cut.rhs)
        locModel.update()

    # The amount of constraints of the model is measured for checking purposes.
    outerConstrs = locModel.NumConstrs
    if outerConstrs == premodelconstrs and model._verbose:
        pass
        #print('--------- POSSIBLE ERROR 0!! -------------')


    while model._LSNeighborhoods._depth <= model._LSNeighborhoods.highest:

        # The best objetive found in the neihgborhood is started as the last B&C objective.
        bestLSObjSoFar = model._IncBeforeLS
        locModel.params.BestObjStop = bestLSObjSoFar

        act_depth = model._LSNeighborhoods._depth

        if model._writeTestLog:
            testlogUpdate(model._testLogPath, 'LS EXPLORE {}'.format(act_depth) )
 
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

            # Local Search model is instructed to stop as soon as it finds a better improvement.
            locModel.params.BestObjStop = bestLSObjSoFar

            # The varaibles are fixed according to the declared neighborhoods or to a function.
            if not model._LSNeighborhoods.useFunction:
                for keyAct in model._LSNeighborhoods.neighborhoods[act_depth][param_act]:
                    locModel.addConstr(locModel._vars[keyAct] == model._BCVals[keyAct], name=keyAct)
                    addedFixedVarsKeys.append(keyAct)
            else:
                for keyAct in model._LSNeighborhoods.keysList:
                    if model._LSNeighborhoods.funNeighborhoods(keyAct, act_depth, param_act):
                        locModel.addConstr(locModel._vars[keyAct] == model._BCVals[keyAct], name=keyAct)
                        addedFixedVarsKeys.append(keyAct)
            locModel.update()

            
            if outerConstrs >= locModel.NumConstrs and model._verbose:
                print('--------- CONSTRAINT ERROR 2!! -------------')

            locModel.optimize()

            if locModel.status == GRB.OPTIMAL or locModel.status == GRB.USER_OBJ_LIMIT:

                if model._verbose:
                    print('Local Search Phase has feasible solution')
                # if bestLSObjSoFar > locModel.objVal and model._LSLastObj > locModel.objVal:
                if bestLSObjSoFar > locModel.objVal:
                    model._LSImproved = True
                    model._LSLastObj = locModel.objVal
                    model._BCHeuristicCounter = 0
                    model._LSImprSols = locModel.getAttr('X')
                    model._LSImprovedDict = {}

                    # The best obj so far is updated.
                    bestLSObjSoFar = locModel.objVal

                    totalvars = len(model._vars.keys())
                    distinct = 0
                    for key in locModel._vars.keys():
                        model._LSImprovedDict[key] = locModel._vars[key].X
                        if locModel._vars[key].X != model._BCVals[key]:
                            distinct += 1

                    model._LSNeighborhoods._depth = act_depth
                    if model._verbose:
                        print( 'MIP Incumbent: {} --'.format(model._IncBeforeLS) + 'Local Search Objective: {}'.format(locModel.objVal))
                        print('--------- Changed {} variables from {}, a  {}% ----------'.format(
                         distinct, totalvars, round(100 * distinct/totalvars, 4)))

                    if model._writeTestLog:
                        testlogUpdate(model._testLogPath, 'LS NEWINCUMBENT {} {}'.format(act_depth, round(locModel.objVal, 6) ) )
            else:
                pass
                #print(locModel.status)
                    
            # Viriables fixed in this parameterization are set free.
            for fixedVar in addedFixedVarsKeys:
                cAct = locModel.getConstrByName(fixedVar)
                locModel.remove(cAct)
            locModel.update()

            # An Error is shown if constraints don't match.
            finalconstrs = locModel.NumConstrs
            if finalconstrs != outerConstrs and model._verbose:
                print('--------- CONSTRAINT ERROR 3!! -------------')

        # Neighborhood is increased for the next local search.
        if not model._LSNeighborhoods.canIncNeigh():
            break
        else:
            model._LSNeighborhoods._depth += 1
        
        if model._LSImproved: break
    
    # If the current neighborhood is the last B&C time is restricted, and solutions cannot de set.
    if model._LSNeighborhoods._depth == model._LSNeighborhoods.highest:
        if model._verbose and not model._LSImproved:
            print('Local Search phase lead to no improvement.')
        model._restrTime = False
        print('Time is not restricted')

    # Time is stored inside the model attribute.
    model._LSLastTime = time.time() - starting_local_search

    # If Local Search has not improved, solutions cannot be set in MIPNODE.
    if not model._LSImproved:
        model._LSImprovedDict = None

    if model._verbose:
        if model._LSImproved:
            print('--- Objective was reduced ---')
        else:
            print('--- Objective was not reduced ---')
        print('Finished Local Search phase')

    if model._writeTestLog:
        testlogUpdate(model._testLogPath, 'LS END' )

def solver(
    path,
    verbose =True,
    addlazy = False,
    funlazy = None,
    importNeighborhoods = False,
    importedNeighborhoods = None,
    funTest = None,
    callback = 'vmnd',
    alpha = 2,
    minBCTime = 7,
    timeLimitSeconds = 300,
    plotGapsTime = False,
    writeTestLog = False
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
    model._plotGapsTimes = plotGapsTime
    model._writeTestLog = writeTestLog
    model._testLogPath = None
    model._gapsTimes = []

    if writeTestLog:
        model._testLogPath = os.path.join ( 'Testing', 'Logs', os.path.basename(path).rstrip('.mps') + '.testlog')
        testFile = open(model._testLogPath, 'w')
        testFile.write('\nLOWEST {}'.format(importedNeighborhoods.lowest))
        testFile.write('\nHIGHEST {}'.format(importedNeighborhoods.highest))
        testFile.write('\nGEN BEGIN')
        testFile.close()
    

    # Local Search Attriutes.
    if not importNeighborhoods or importedNeighborhoods is None:
        model._LSNeighborhoods = Neighborhoods(lowest = 1, highest = 5, keysList = model._vars.keys(), randomSet=True)
    else:
        model._LSNeighborhoods = importedNeighborhoods
        
    model._LSLastTime = 1000
    model._IncBeforeLS = None
    model._LSImproved = False
    model._LSImprovedDict = None
    model._LSImprSols = None
    model._LSLastObj = None
    

    # Branch and Cut Attributes.
    model._addLazy = addlazy
    model._funLazy = funlazy
    model._minBCTime = minBCTime
    model._BCLastStart = time.time()
    model._BCLastObj = None
    model._BCVals = None
    model._BCHeuristicCounter = 0
    model._BCimproved = False
    model._BCLazyAdded = []
    model._newBest = False
    model._senseDict = {
        '<=' : GRB.LESS_EQUAL,
        '==' : GRB.EQUAL,
        '>=' : GRB.GREATER_EQUAL
    }
    model._line = ''

    allTestsPassed = False
    if importedNeighborhoods is not None and importNeighborhoods:
        if checkVariables(model._vars.keys(), importedNeighborhoods):
            print('Variables are in accordance with their neighborhoods.')
            allTestsPassed = True
        else:
            print('[TEST] Neighborhood\'s variable not belonging to model variables found.')    

    if allTestsPassed or not importNeighborhoods:
        model.setParam("LazyConstraints", 1)

        if callback == 'vmnd':
            model.setParam('ImproveStartNodes', 200)
            model.setParam('MIPFocus', 3)
        else:
            model.setParam('ImproveStartNodes', 10)
            model.setParam('MIPFocus', 0)

        if timeLimitSeconds is not None:
            model.setParam('TimeLimit', timeLimitSeconds)
        if not verbose:
            model.setParam('OutputFlag', 0)

        if model._writeTestLog:
            testlogUpdate(model._testLogPath, 'BC BEGIN')
    
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

        if verbose and plotGapsTime:
            GapTimePlot(model._gapsTimes)
    
    if writeTestLog:
        testFile = open(model._testLogPath, 'a')
        testFile.write('\nGEN END')
        testFile.close()
    
    print(model._gapsTimes)
    return model

def creator(path):
    return loadMPS(path)


if __name__ == '__main__':
    path = os.path.join('MIPLIB', 'binkar10_1.mps')

    #nbhs = varClusterFromMPS(path, numClu = 5, varFilter = None)
    nbhs = Neighborhoods(
        lowest = 1,
        highest = 5,
        keysList=[f"C{i}" for i in range(1000, 2250)],
        randomSet=True,
        outerNeighborhoods = None,
        useFunction = False,
        funNeighborhoods = None)
    mout = solver(
        path,
        verbose = True,
        addlazy= False,
        funlazy = None,
        importNeighborhoods=True,
        importedNeighborhoods= nbhs,
        funTest= None,
        callback = 'pure',
        alpha = 1,
        minBCTime= 15,
        timeLimitSeconds= None,
        plotGapsTime= True,
        writeTestLog=False
    )
    
    