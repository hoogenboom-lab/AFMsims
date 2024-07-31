Running Simulator
===================================
The code calculates scan variables and export them to csv files then runs ABAQUS using seperate python scripts that import the variable data. ABAQUS can be run locally, however, they are designed to be run on remote servers, using SSH to upload files and run ABAQUS on HPC queues. Cloning the git page and pip installing 'afmsims' will add all packages/ modules to your python enviroment. All Jupyter notebooks(.ipynb) are self contained, they produce the input files, in the specified local working directory, for each simulation so best run from own self contained directory. The notebooks contain breakdown and description of code function. Seperate Python(.py) files for the AFM simulation are available in the 'Python Scripts' folder. For more lightweight code the simulator can be run from separate python kernal/notebook by importing the AFM_ABAQUS_Simulation_Code.py file (the ABAQUS scripts will need to be copied into the working directory (localPath) specified in simulator).

Importing Python files
===================================

Within a seperate python script the simulator code can be imported by either appending the package using system command and path to directory holding the files:

.. code-block:: python

    import sys
    sys.path.insert(1, 'C:\\path\\to\\directory\\afmsims') 
    
Or by either copying the afmsims package to the same directory or to the main python path (for jupyter notebook/spyder this will be main anaconda directory). Packages can be imported in various ways importing as:

.. code-block:: python

    import afmsims

    afmsims.afm.AFMSimulation(...)
    afmsims.wave.WaveSimulation(...)
    afmsims.hemisphere.HemisphereSimulation(...)

Alternative:

.. code-block:: python

    from afmsims import *

    afm.AFMSimulation(...)
    wave.WaveSimulation(...)
    hemisphere.HemisphereSimulation(...)

Alternative (can have conflicting functions do not do for all as shown):

.. code-block:: python

    from afmsims.afm import *
    from afmsims.wave import *
    from afmsims.hemisphere import *
    
    AFMSimulation(...) 
    WaveSimulation(...) 
    HemisphereSimulation(...)

Then, the simulator can simply be run by defining the required variables and running main function:

.. code-block:: python

        host, port, username, password, None, localPath, afmCommand, fileName, subData,              
        pdb, rotation, surfaceApprox, indentorType, rIndentor, theta_degrees, tip_length,             
        indentionDepth, forceRef, contrast, binSize, clearance, meshSurface, meshBase, meshIndentor,   
        timePeriod, timeInterval = ...
        
         ...AFMSimulation(host, port, username, password, None, localPath, afmCommand, fileName, subData,
        pdb, rotation, surfaceApprox, indentorType, rIndentor, theta_degrees, tip_length,
        indentionDepth, forceRef, contrast, binSize, clearance, meshSurface, meshBase, meshIndentor,
        timePeriod, timeInterval)


Common Errors
===================================
 * ABAQUS scripts/ package files not located in working directory or system path
 * Some modules may require Python 3.9 or newer. 
 * You must be careful to change path syntaax if using mac or linux.
 * Require the following modules: py3Dmol, nglview, biopython, mendeleev, pyabaqus==2022, paramiko (view requirements.txt)