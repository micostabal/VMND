import os
from IRP import runSeveralIRP
from IRPCS import runSeveralIRPCS
from MVRPD import runSeveralMVRPD
from VRP import runSeveralVRP
from OISRC import runSeveralOISRC


# Official Experiment
IRPinst = [
    'abs3n100_2.dat',
    'abs4n100_3.dat' ,
    'abs9n200_2.dat'
]
IRPCSinst = [ 
    os.path.join( 'IRPCSInstances', 'Inst1.txt'),
    os.path.join( 'IRPCSInstances', 'Inst2.txt'),
    os.path.join( 'IRPCSInstances', 'Inst3.txt')
]
MVRPDinst = [ 
    os.path.join( 'MVRPDInstances' , 'ajs1n100_l_6.dat' ),
    os.path.join( 'MVRPDInstances' , 'ajs5n100_h_6.dat' ),
    os.path.join( 'MVRPDInstances' , 'ajs4n100_l_6.dat' )
]
OISRCinst = [
    os.path.join('OISRCInstances', 'instance_20_4_200_1.oisrc'),
    os.path.join('OISRCInstances', 'instance_20_4_190_1.oisrc'),
    os.path.join('OISRCInstances', 'instance_15_2_170_1.oisrc')
]

# Easy instances, just for testing purposes.
    
"""IRPinst = [ 
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
]"""


# Instances are run. Results saved in Results/results.txt
elapsedTime = 7205

#elapsedTime = 50


# 1.- OISRC
runSeveralOISRC(
    OISRCinst,
    timeLimit = elapsedTime
)

"""# 2.- MVRPD
runSeveralMVRPD(
    MVRPDinst,
    nbhs = ('separated', 'function'),
    timeLimit = elapsedTime,
    includePure = True
)

# 3.- IRPCS
runSeveralIRPCS(
    IRPCSinst,
    nbhs = ('separated', 'function'),
    timeLimit = elapsedTime,
    outVtrunc = 70,
    outHtrunc = 3,
    outKtrunc = 12,
    includePure = True
)

# 4.- IRP
runSeveralIRP(
    IRPinst,
    nbhs = ('separated', 'function'),
    timeLimit = elapsedTime,
    includePure = True
)"""

if __name__ == '__main__': pass