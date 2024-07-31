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


# %%
# Import experimental data
force_height_data = np.loadtxt(dataPath+'Force-height-data.csv', dtype=float, delimiter=",", skiprows=3)[:,:2]*np.array([1,10])[None,:]

dirFile = list(os.listdir(dataPath))
dirFile.remove('Force-height-data.csv')
dirFile.remove('7mv-sqz-cross section.txt')
dirFile.remove('7mv-sqz-retrace-profile.txt')
dirFile.remove('7mv-sqz-trace-profile.txt')


force_data, data_length = np.zeros([len(dirFile)]), np.zeros(len(dirFile), dtype=int)
cross_section_height_data = np.zeros([len(dirFile),190,2])

for i, file in enumerate(dirFile):
    force_data[i] = float(np.loadtxt(dataPath + file, dtype=str, delimiter=",", skiprows=2, max_rows=1)[1][:-2])
    temp_data = np.loadtxt(dataPath + file, dtype=float, delimiter=",", skiprows=3)
    data_length[i] = len(temp_data)
    cross_section_height_data[i,:len(temp_data)] = temp_data*10
    
force_data = np.sort(force_data)
cross_section_height_data = np.array([x for _, x in sorted(zip(force_data, cross_section_height_data))])
  

# %%
N = 20
tip_size  = np.linspace(10,30, N)
FWHM_test = np.zeros(N)

for i, RIndentor in enumerate(tip_size) :
    temp1, temp2, temp3, temp_scanPos, temp_scanDims, temp_baseDims, temp4 = AFMSimulation(remote_server, remotePath, localPath, abqscripts, abqCommand, fileName, subData, 
            pdb, rotation, surfaceApprox, indentorType, RIndentor, theta_degrees, tip_length, indentionDepth, forceRef, contrast, binSize, 
            0, elasticProperties, meshSurface, meshBase, meshIndentor,  timePeriod, timeInterval,
            Preprocess  = True, Submission  = False,Postprocess = False, ReturnData  = True )

    
    xNum, yNum = int(temp_scanDims[0]/binSize)+1, int(temp_scanDims[1]/binSize)+1
    X_temp, Y_temp, Z_temp = temp_scanPos.reshape(yNum, xNum, 3)[:,:,0], temp_scanPos.reshape(yNum, xNum, 3)[:,:,1], temp_scanPos.reshape(yNum, xNum, 3)[:,:,2] 

    FWHM_temp = np.zeros(xNum)
    
    for n in range(xNum):
        # Extract y coordinates and z compontents of force contour across x domain
        Fy, Fz = Y_temp[:,0], Z_temp[:,n]
        
        # Connect contour points smoothly with a spline
        forceSpline = UnivariateSpline(Fy, Fz, s = 0.01)         
        
        # Use try loop to avoid error for contours that cannot produce splines  
        try: 
            # Half maxima can be calculated by finding roots of spline 
            roots = UnivariateSpline(Fy, Fz - (Fz.min() + Fz.max())/2, s = 0.01).roots()             
            FWHM_temp[n] = roots[1]-roots[0]
        except:
            None 
    
    FWHM_test[i] = FWHM_temp[FWHM_temp!=0].mean()

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

# %%
# Half maxima can be calculated by finding roots of spline 
intercept = UnivariateSpline(tip_size, FWHM_test-FWHM_exp[0], s = 0.01).roots()             

fig, ax = plt.subplots(figsize = (linewidth, 1/1.61*linewidth))
ax.axhline(FWHM_exp[0], color='k', ls='--', label = 'Experimental FWHM' )
ax.axvline(intercept, color='r', ls=':', label = 'Intercept')
ax.plot(tip_size,FWHM_test) 

ticks = list(ax.get_xticks())[1:-1] + [intercept[0]]
ticks.remove(20)
ticks.sort()
print(ticks)

ax.set_xlabel('Tip Radius')
ax.set_ylabel('FWHM')
ax.set_xticks(ticks)
fig.savefig('AFMIntercept.png', bbox_inches = 'tight')

# plt.show()

# %%
# Produce array of fitted elastic modulus over scan positions for each indentor
u2, rf      = abs(1.*temp_scanPos[:,2].max()-force_height_data[:-1,1])*1e-10, abs(force_height_data[:-1,0]*1e-12)
E_DNA, E_err = curve_fit(lambda x, E: F_Hertz(x, E, 20*1e-10, elasticProperties), u2, rf)

print(E_DNA,E_err)
x = np.linspace(u2.min(), u2.max(), 25)

fig, ax = plt.subplots(figsize = (linewidth, 1/1.61*linewidth))
ax.plot(u2, rf, 'x' ) 
ax.plot(x, F_Hertz(x, E_DNA, 17.88*1e-10, elasticProperties), ':' )
ax.set_xlabel('DNA Height')
ax.set_ylabel('Force')
fig.savefig('AFMHertzFit.png', bbox_inches = 'tight')

# plt.show()
