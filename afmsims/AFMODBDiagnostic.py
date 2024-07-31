# ----------------------------------------------Load Modules-----------------------------------------------------------
import sys
import os
from os.path import exists

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
# import regionToolset
executeOnCaeStartup()

# ------------------------------------------------Set variables-------------------------------------------------------
# Import predefined variables from files set in current directory
variables        = np.loadtxt('variables.csv', delimiter=",")
clipped_scanPos  = np.loadtxt('clipped_scanPos.csv', delimiter=",")

timePeriod, timeInterval, binSize, meshSurface, meshBase, meshIndentor, indentionDepth, surfaceHeight = variables
# Set array for indentation data: RF - Indentor reaction force , U2 - Indentor displacement, N - no. timesteps

N = int(timePeriod/ timeInterval)+1
RFtemp = np.zeros([len(clipped_scanPos),N])
U2temp = np.zeros([len(clipped_scanPos),N])

ErrorMask = np.zeros([len(clipped_scanPos)])
    
# ----------------------------------------------Set Data extraction---------------------------------------------------
for i in range(len(clipped_scanPos)):
    # Log Analysis Progression in text file
    with open('Progress.txt', 'a') as f:
        f.write('File '+str(i)+'\n')

    jobName = 'AFMRasterScan-Pos'+str(int(i))  
    
    # Check if odb file is veiwable and no corrupted data which my throw an error
    try :
        # Open odb and retrieve data for indentor reference point(2nd value in models nodal sets/ region)
        odb    = openOdb(jobName +'.odb', readOnly=True)
        region = odb.rootAssembly.nodeSets.values()[1]
    except:        
        # If theres an error log odb file value in text file
        with open('Null.txt', 'a') as f:
            f.write('Error for'+str(i)+'\n')
    else:          
        # Extracting data for Step 1, this analysis only had one step
        step1 = odb.steps.values()[0]

        # Counting frames/ timesteps
        j,k = 0, 0 
        
        # Creating a for loop to iterate through all frames in the step
        for x in odb.steps[step1.name].frames:
            # Reading force and displacement data from the model at reference point
            fieldRF = x.fieldOutputs['RF'].getSubset(region= region)
            fieldU  = x.fieldOutputs['U'].getSubset(region= region)    

            # Storing reaction force and displacement values for the current frame
            for rf in fieldRF.values:
                RFtemp[i,j] = np.sqrt(rf.data[2]**2)
                j+=1         

            for u in fieldU.values:
                U2temp[i,k] = u.data[2] 
                k+=1 
                
        if np.count_nonzero(RFtemp[i])==0:
            
            with open('Null.txt', 'a') as f:
                f.write('Null for {0} Array: \n {1} \n {2}'.format(i,U2temp[i], RFtemp[i]))

# Close the odb
odb.close()    
