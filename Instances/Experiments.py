import os
from IRP import runSeveralIRP
from IRPCS import runSeveralIRPCS
from MVRPD import runSeveralMVRPD
from VRP import runSeveralVRP
from OISRC import runSeveralOISRC


official = True


# Official Experiment
if official:
    IRPinst = [
        'abs3n50_1.dat',
        'abs2n50_3.dat' ,
        'abs3n100_2.dat'
    ]
    IRPCSinst = [ 
        os.path.join( 'IRPCSInstances', 'Inst1.txt'),
        os.path.join( 'IRPCSInstances', 'Inst2.txt'),
        os.path.join( 'IRPCSInstances', 'Inst3.txt')
    ]
    MVRPDinst = [ 
        os.path.join( 'MVRPDInstances' , 'ajs4n75_h_6.dat' ),
        os.path.join( 'MVRPDInstances' , 'ajs4n100_l_3.dat' ),
        os.path.join( 'MVRPDInstances' , 'ajs3n75_h_6.dat' )
        
    ]
    OISRCinst = [
        os.path.join('OISRCInstances', 'instance_20_4_200_1.oisrc'),
        os.path.join('OISRCInstances', 'instance_20_4_190_1.oisrc') 
    ]

# Easy instances, just for testing purposes.
if not official:
    
    IRPinst = [ 
        'abs1n5_1.dat',
        'abs2n10_3.dat'
    ]
    IRPCSinst = [ 
        os.path.join( 'IRPCSInstances', 'Inst1.txt'),
        os.path.join( 'IRPCSInstances', 'Inst2.txt')
    ]
    MVRPDinst = [ 
        os.path.join( 'MVRPDInstances' , 'ajs1n25_h_3.dat' ),
        os.path.join( 'MVRPDInstances' , 'ajs1n25_l_3.dat' )
    ]
    OISRCinst = [
        os.path.join('OISRCInstances', 'instance_4_2_100_1.oisrc')
    ]


# Instances are run. Results saved in Results/results.txt

if official:
    elapsedTime = 7210
else:
    elapsedTime = 50




# 1.- IRPCS
runSeveralIRPCS(
    IRPCSinst,
    nbhs = ('separated', 'function'),
    timeLimit = elapsedTime,
    outVtrunc = 70,
    outHtrunc = 3,
    outKtrunc = 12,
    includePure = True
)

# 2.- MVRPD
runSeveralMVRPD(
    MVRPDinst,
    nbhs = ('separated', 'function'),
    timeLimit = elapsedTime,
    includePure = True
)

# 3.- OISRC
runSeveralOISRC(
    OISRCinst,
    timeLimit = elapsedTime
)

# 4.- IRP
runSeveralIRP(
    IRPinst,
    nbhs = ('separated', 'function'),
    timeLimit = elapsedTime,
    includePure = True
)

if __name__ == '__main__': pass