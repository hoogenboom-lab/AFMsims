# # Simulation Interface

# %%

import sys
sys.path.insert(1, '/home/jgiblinburnham/Documents/ABAQUS-AFM-Simulations/')

from abqsims.afm import *



# %%
#  ------------------------------------------Remote Sever and Submission Variables------------------------------------------
remote_server = np.loadtxt('/home/jgiblinburnham/Documents/ABAQUS-AFM-Simulations/ssh/myriad.csv', dtype=str, delimiter=",")
proxy_server = np.loadtxt('/home/jgiblinburnham/Documents/ABAQUS-AFM-Simulations/ssh/ssh-gateway.csv', dtype=str, delimiter=",")

# Set up commands
bashCommand = 'qsub'
abqCommand  = 'module load abaqus/2017 \n abaqus cae -noGUI'

# Set up directories
home, scratch = remote_server[-2], remote_server[-1]
wrkDir     = '/ABAQUS/AFM_Simulations/DNASimulation-18A-2D-2'
localPath  = '/home/jgiblinburnham/Documents/ABAQUS-AFM-Simulations/example-scripts/afm' #os.getcwd()
remotePath = scratch + wrkDir
dataPath = '/home/jgiblinburnham/Documents/ABAQUS-AFM-Simulations/example-scripts/afm/experimental-data/'

# Set up submission variables
abqscripts = ('AFMSurfaceModel.py','AFMRasterScan.py','AFMODBAnalysis.py')
fileName = 'AFMRasterScan-Pos'
subData  = ['48:0:0', '5G', '16']

# %%
#  -------------------------------------------Simulation Variables---------------------------------------------------------
# Surface variables                              #
pdb = 'bdna8'                                    # '1bna' 'bdna8' '4dqy' 'pyn1bna'
rotation = [0, 270, 60]                          # degrees
surfaceApprox = 0.2                              # arb
E_true, v = 10, 0.3                            # (10^10 Pa / 10GPa) 1000
elasticProperties = np.array([E_true, v])        #
                                                 #
# Indentor variables                             #
indentorType = 'Capped'                          #
rIndentor = 18 # 2                               # (x10^-10 m / Angstroms)
theta_degrees = 5                                # degrees
tip_length = 50                                  # (x10^-10 m / Angstroms)
                                                 #
# Scan variables                                 #
clearance = 0.15  # 2.5                           # (x10^-10 m / Angstroms)
indentionDepth = clearance + 2                   # (x10^-10 m / Angstroms)
binSize  = 5.5  # 6                              # (x10^-10 m / Angstroms)
forceRef = 15                                     # (x10^-10 N / pN)
contrast = 1.61                                  # arb
                                                 # 
# ABAQUS variable                                #
timePeriod   = 1                                 # s
timeInterval = 0.1                               # s
meshSurface  = 2.5 #0.3                          # (x10^-10 m / Angstroms)
meshBase     = 2                                 # (x10^-10 m / Angstroms)
meshIndentor = 0.6 #0.3                          # (x10^-10 m / Angstroms)

# In[]
#  -------------------------------------------Simulation Script----------------------------------------------------------
U2, RF, ErrorMask, scanPos, scanDims, baseDims, variables = AFMSimulation(remote_server, remotePath, localPath, abqscripts, abqCommand, fileName, subData, 
            pdb, rotation, surfaceApprox, indentorType, rIndentor, theta_degrees, tip_length, indentionDepth, forceRef, contrast, binSize, 
            clearance, elasticProperties, meshSurface, meshBase, meshIndentor,  timePeriod, timeInterval,

            Preprocess  = True,
              
            Submission  = 'scanlines',
            ProxyJump   = proxy_server,
            Transfer    = False, 
            Part        = False, 
            Input       = False, 
            Batch       = False, 
            Queue       = False, 
            Analysis    = False, 
            Retrieval   = True, 

            Postprocess = True, 
            ReturnData  = True,
              
            HSPlot        = False,
            MolecularView = False,
            DotPlot       = False,
            DataPlot      = False,                
              
            ClippedScan  = [0.35,1],
            # PowerNorm    = 0.85,
            # NLinearScale = True,
            # ImagePadding = 1, 
            # Noise        = [0,0,0.15],
            # SaveImages   = '/mnt/c/Users/Joshg/Downloads/'
            )

# %%
# Import experimental data
force_height_data = np.loadtxt(dataPath+'Force-height-data.csv', dtype=float, delimiter=",", skiprows=3)[:,:2]*np.array([1,10])[None,:]

dirFile = list(os.listdir(dataPath))
dirFile.remove('Force-height-data.csv')

force_data, data_length = np.zeros([len(dirFile)]), np.zeros(len(dirFile), dtype=int)
cross_section_height_data = np.zeros([len(dirFile),190,2])

for i, file in enumerate(dirFile):
    force_data[i] = float(np.loadtxt(dataPath + file, dtype=str, delimiter=",", skiprows=2, max_rows=1)[1][:-2])
    temp_data = np.loadtxt(dataPath+ file, dtype=float, delimiter=",", skiprows=3)
    data_length[i] = len(temp_data)
    cross_section_height_data[i,:len(temp_data)] = temp_data*10
    
force_data = np.sort(force_data)
cross_section_height_data = np.array([x for _, x in sorted(zip(force_data, cross_section_height_data))])

# %%
# Produce analysis on experimental data
N = len(force_data)
FWHM_exp, AreaX_exp = np.zeros(N), np.zeros(N)
for i in range(N):
    # Extract y coordinates and z compontents of force contour across x domain
    Fy, Fz = cross_section_height_data[i,:data_length[i],0],  cross_section_height_data[i,:data_length[i],1]

    # Connect contour points smoothly with a spline
    forceSpline = UnivariateSpline(Fy, Fz, s = 0.1)         

    # Half maxima can be calculated by finding roots of spline 
    roots = UnivariateSpline(Fy, Fz - (Fz.min() + Fz.max())/2, s = 0.01).roots()             
    FWHM_exp[i] = roots[1]-roots[0]

    AreaX_exp[i] = np.trapz(Fy, Fz)      


# %% [markdown]
# ## Postprocessing

# %%
# Produce true surface geometry, with small tip
surfaceSize = 1
temp_U2, temp_RF, temp_ErrorMask, surfacePos, surfaceDims, temp_baseDims, temp_variables = AFMSimulation(remote_server, remotePath, localPath, abqscripts, abqCommand, fileName, subData, 
pdb, rotation, surfaceApprox, indentorType, 1, 5, tip_length, indentionDepth, forceRef, contrast, surfaceSize, 0, elasticProperties, meshSurface, meshBase, 
meshIndentor,  timePeriod, timeInterval, Preprocess  = True, Submission  = False, Postprocess = False, ReturnData  = True, ClippedScan  = [0.45,1])

# %%
waveLength, Nmax, binSize = 50, 125, variables[2]
courseGrain = 2.5

# %%
indentationForce, structuralData, XSectionData, YSectionData = DataAnalysis(U2, RF, force_data, scanPos, scanDims, binSize, clearance, waveLength, Nmax, courseGrain, rIndentor, elasticProperties, timeInterval, timePeriod)

# %%
indentationForce, XSectionData, YSectionData = DataAnalysis(U2, RF, force_data, scanPos, scanDims, binSize, clearance, waveLength, Nmax, courseGrain, rIndentor, elasticProperties, timeInterval, timePeriod, NStructural=True)

# %% 
# ## Data Plots
plotTip1 = TipStructure(18, 10, 50)
plotTip2 = TipStructure(18, 10, 60) 

# %% 
# ### Manuscript AFM Image 
ManuscriptAFMContourPlot(U2, RF, scanPos, scanDims, binSize, clearance, ErrorMask, forceRef, 1.4, pdb, SaveImages='/mnt/c/Users/Joshg/Documents/Manuscript_Figures/')

# %%
# ### Illustrations Plot
ManuscriptDiagram(scanPos, scanDims, binSize, surfacePos, surfaceDims, surfaceSize, plotTip1, SaveImages = '/mnt/c/Users/Joshg/Documents/Manuscript_Figures/')

SurfacePlot(scanPos, scanDims, binSize, surfacePos, surfaceDims, surfaceSize, plotTip2, SaveImages = '/mnt/c/Users/Joshg/Documents/Manuscript_Figures' )

# %% 
# ### Profile Plots
ForceProfiles(scanPos, scanDims, binSize, structuralData[0], structuralData[1], indentationForce, force_data, cross_section_height_data, data_length, SaveImages = '/mnt/c/Users/Joshg/Documents/Manuscript_Figures/')

# %% 
# ### Structural Plots

StructureAnalysisPlot(structuralData, indentationForce, force_data, force_height_data, FWHM_exp, AreaX_exp, SaveImages = '/mnt/c/Users/Joshg/Documents/Manuscript_Figures/')


# %% 
# ### Contour Plots
AFMForceContours(RF, XSectionData, YSectionData, surfacePos, surfaceDims, surfaceSize, plotTip1, clearance, elasticProperties, contrast, 1, SaveImages = '/mnt/c/Users/Joshg/Documents/Manuscript_Figures/')

# %% 
# ### Youngs Modulus

# YoungPlot(XSectionData, rIndentor, elasticProperties, 37)
YoungPlot(YSectionData, rIndentor, elasticProperties, 3)


