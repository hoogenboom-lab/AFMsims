
abqDir     = '/ABAQUS/AFM_Simulations/Simulation_Code-Test-0'
scratch    = '/scratch/scratch/zcapjgi'
remotePath = scratch + abqDir
fileName = 'AFMRasterScan-Pos'
subData  = ['24:0:0', '2.5G', '120']

# Create set of submission comands for each scan locations
files = [1, 8, 9, 10, 15, 16, 18, 19, 20, 21, 23, 25, 26, 29, 32, 33, 34, 36, 37, 38, 40, 42, 44, 45, 46, 47, 48, 51, 53, 55, 56, 57, 60, 61, 62, 63, 64, 67, 69, 73, 75, 77, 81, 82, 85, 86, 87, 94, 98, 100, 102, 103, 105, 106]

jobs   = ['abaqus interactive cpus=$NSLOTS memory="90%" mp_mode=mpi standard_parallel=all job='+fileName+str(i)+' input='+fileName+str(i)+'.inp scratch=$ABAQUS_PARALLELSCRATCH' 
            for i in files]

# Produce preamble to used to set up bash script
lines = ['#!/bin/bash -l',
            '#$ -S /bin/bash',
            '#$ -l h_rt='+ subData[0],
            '#$ -l mem=' + subData[1],
            '#$ -pe mpi ' + subData[2],
            '#$ -wd /home/zcapjgi/Scratch/ABAQUS',
            'module load abaqus/2017 ',
            'ABAQUS_PARALLELSCRATCH="/home/zcapjgi/Scratch/ABAQUS" ',
            'cd ' + remotePath ]
# Combine to produce total  script
lines+=jobs

# Create script file in current directory by writing each line to file
with open('batchScript.sh', 'w', newline = '\n') as f:
    for line in lines:
        f.write(line)
        f.write('\n')