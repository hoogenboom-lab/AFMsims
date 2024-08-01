
# ----------------------------------------------Load Modules-----------------------------------------------------------
import sys
from odbAccess import *
from types import IntType
import numpy as np 
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import *
from part import *
from material import *
from section import *
from assembly import *
from interaction import *
from mesh import *
from visualization import *
import visualization
import odbAccess
from connectorBehavior import *
import cProfile, pstats, io
import regionToolset
#from abaqus import getInput
executeOnCaeStartup()

# ------------------------------------------------Set variables-------------------------------------------------------
variables = np.loadtxt('variables.csv', delimiter=",")
rackPos    = np.loadtxt('rackPos.csv', delimiter=",")

timePeriod, timeInterval, binSize, indentionDepth, meshIndenter, meshSurface = variables

N = int(timePeriod/ timeInterval)+1
RF = np.zeros([len(rackPos),N])
U2 = np.zeros([len(rackPos),N])

# --------------------------------------------Set Data extraction---------------------------------------------------
for i in range(len(rackPos)):
    jobName = 'AFMtestRasterScan-Pos' + str(int(i))
    try :
        # Opening the odb
        odb    = openOdb(jobName +'.odb', readOnly=True)
        region = odb.rootAssembly.nodeSets.values()[1]
    except:        
        with open('Errors.txt', 'a') as f:
            f.write('ERROR for'+str(i)+'\n')
    else:          
        # Extracting Step 1, this analysis only had one step
        step1 = odb.steps.values()[0]
        
        j,k = 0, 0 
        # Creating a for loop to iterate through all frames in the step
        for x in odb.steps[step1.name].frames:
            # Reading stress and strain data from the model 
            fieldRF = x.fieldOutputs['RF'].getSubset(region= region)
            fieldU  = x.fieldOutputs['U'].getSubset(region= region)    

            # Storing Stress and strain values for the current frame
            for rf in fieldRF.values:
                RF[i,j] = np.sqrt(rf.data[1]**2)
                j+=1

            for u in fieldU.values:
                U2[i,k] = u.data[1] 
                k+=1   

# Writing to a .csv file
np.savetxt('U2_Results.csv', U2 , delimiter=",")
np.savetxt('RF_Results.csv', RF , delimiter=",")   

# Close the odb
odb.close()    
