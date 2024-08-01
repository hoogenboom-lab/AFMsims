# %%
import sys
sys.path.insert(1, '..')

from afmsims.afm import *
from scipy import interpolate




# %%
#  ------------------------------------------Remote Sever and Submission Variables------------------------------------------
remote_server = np.loadtxt('../ssh/myriad.csv', dtype=str, delimiter=",")
proxy_server  = np.loadtxt('../ssh/ssh-gateway.csv', dtype=str, delimiter=",")

# Set up commands
bashCommand = 'qsub'
abqCommand  = 'module load abaqus/2017 \n abaqus cae -noGUI'

# Set up directories
home, scratch = remote_server[-2], remote_server[-1]
wrkDirs = ['/ABAQUS/AFM_Simulations/DNASimulation-18A-2D','/ABAQUS/AFM_Simulations/DNASimulation-18A-10D','/ABAQUS/AFM_Simulations/DNASimulation-18A-6D','/ABAQUS/AFM_Simulations/DNASimulation-18A-2D-2']
localPath  = os.getcwd() + os.sep + 'afm' 

dataPath = os.getcwd() + os.sep + 'afm/experimental-data/'
imagePath = '/mnt/c/Users/Joshg/Documents/Manuscript_Figures/'

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

tipDims = TipStructure(rIndentor, theta_degrees, tip_length)


# %%

AFMSimulation(remote_server, '' '', localPath, abqscripts, abqCommand, fileName, subData, pdb, rotation, surfaceApprox, indentorType, 
              rIndentor, theta_degrees, tip_length, indentionDepth, forceRef, contrast, binSize,  clearance, 
              elasticProperties, meshSurface, meshBase, meshIndentor,  timePeriod, timeInterval, Preprocess  = True, 
              Submission  = False, Postprocess = False, ReturnData  = False, ClippedScan  = [0.35,1])

atom_coord, atom_element, atom_radius, variables, baseDims, scanPos, clipped_scanPos, scanDims = ImportVariables(localPath)
timePeriod, timeInterval, binSize, meshSurface, meshBase, meshIndentor, indentionDepth, surfaceHeight = variables
clearance = 0.15


xNum, yNum = int(scanDims[0]/binSize)+1, int(scanDims[1]/binSize)+1
xi, yi = int(xNum/2), int(yNum/2)

scanPosX, scanPosY = scanPos.reshape(yNum, xNum, 3)[yi,:,[0,2]].T, scanPos.reshape(yNum, xNum, 3)[:,xi,[1,2]]







# %%
# Create array of indices for grid, extract 2D indices across two scanlines (xi, yi), then map to a 1d array indices and join
index_array= np.indices([yNum, xNum])
indices = np.sort(np.concatenate((
    np.ravel_multi_index([index_array[0,yi],index_array[1,yi]],[yNum, xNum]),
    np.ravel_multi_index([index_array[0,:,xi],index_array[1,:,xi]],[yNum, xNum])  
    )) 
    )

# Loop to remove clipped positions and map indices to clipped ist indices 
clipped_indices, j = [], 0  
for i in range(len(scanPos)):
    if scanPos[i,2] != clearance: 
        # Extract indices for indices 
        if i in indices:
            clipped_indices.append(j) 
        # Count indices in clipped array
        j+=1 

N =len(clipped_indices)
split_index = [ np.arange(clipped_indices[int((N+1)/2)]-int((clipped_indices[1]-clipped_indices[0])/2),clipped_indices[int((N+1)/2)]+int((clipped_indices[1]-clipped_indices[0])/2), 1 ),
               np.arange(clipped_indices[0],clipped_indices[-1]+int(clipped_indices[1]-clipped_indices[0]), int(clipped_indices[1]-clipped_indices[0]) ) ]
# print(split_index)
# print(N, clipped_indices)

U2 = [[] for n in range(N)]
RF = [[] for n in range(N)]

for k, wrkDir in enumerate(wrkDirs):
    remotePath = scratch + wrkDir

    # Retrieve variables used for given simulation (in case variables redefined when skip kwargs used) 
    dataFiles = ('U2_Results.csv','RF_Results.csv')

    RemoteFTPFiles(remote_server, dataFiles[0], remotePath, localPath, ProxyJump = proxy_server)
    RemoteFTPFiles(remote_server, dataFiles[1], remotePath, localPath, ProxyJump = proxy_server)

    if k>=2: 
        clipped_U2 = abs(np.array(np.loadtxt(localPath+os.sep+'data' + os.sep + 'U2_Results.csv', delimiter=",")))
        clipped_RF = 100*abs(np.array(np.loadtxt(localPath+os.sep+'data' + os.sep + 'RF_Results.csv', delimiter=",")))
    
    else:
        clipped_U2 = abs(np.array(np.loadtxt(localPath+os.sep+'data' + os.sep + 'U2_Results.csv', delimiter=",")))
        clipped_RF = abs(np.array(np.loadtxt(localPath+os.sep+'data' + os.sep + 'RF_Results.csv', delimiter=",")))

    for i, n in enumerate(clipped_indices):
        for j, v in enumerate(clipped_RF[n]):
            if v != 0.0:
                U2[i].append(clipped_scanPos[n,2]-clipped_U2[n,j])
                RF[i].append(v)











# %%#########################################################################################
# Create force spline
sortedU2 = [np.sort(np.array(U2[i])) for i in range(N)]
sortedRF = RF.copy()

for i in range(N):
    sortedRF[i] = np.array([x for _, x in sorted(zip(U2[i],RF[i]))])

    temp = sortedU2[i].copy()

    sortedU2[i] = np.array([v for j, v in enumerate(sortedU2[i]) if v != sortedU2[i][j-1] ])
    sortedRF[i] = np.array([v for j, v in enumerate(sortedRF[i]) if temp[j] != temp[j-1] ])/10


g = {}
for i, v in enumerate(clipped_indices):
    g[v] = UnivariateSpline(sortedU2[i], sortedRF[i], s=2, k=1)


fig, ax = plt.subplots(1,1,figsize=(10,5))
for i, v in enumerate(clipped_indices):
    x = np.linspace(np.min(sortedU2[i]), np.max(sortedU2[i]),50)
    im = ax.plot(x, g[v](x))
    ax.plot(sortedU2[i], sortedRF[i], ':', color=im[0].get_color())
plt.show()



#########################################################################################

# Create force spline
sortedRF = [np.sort(np.array(RF[i])) for i in range(N)]
sortedU2 = U2.copy()

for i in range(N):
    sortedU2[i] = np.array([x for _, x in sorted(zip(RF[i],U2[i]))])

    temp = sortedRF[i].copy()

    sortedRF[i] = np.array([v for j, v in enumerate(sortedRF[i]) if v != sortedRF[i][j-1] ])/10
    sortedU2[i] = np.array([v for j, v in enumerate(sortedU2[i]) if temp[j] != temp[j-1] ])

f = {}
for i, v in enumerate(clipped_indices):
    # print(sortedU2[i])
    f[v] = UnivariateSpline(sortedRF[i], sortedU2[i], s=2, k=1)


fig, ax = plt.subplots(1,1,figsize=(10,5))
for i, v in enumerate(clipped_indices):
    x = np.linspace(np.min(sortedRF[i]),np.max(sortedRF[i]),50)
    im = ax.plot(x, f[v](x))
    ax.plot(sortedRF[i], sortedU2[i], ':', color= im[0].get_color())
    ax.set_ylim(0,20)
plt.show()



#########################################################################################

# fig, ax = plt.subplots(3,1,figsize=(10,5))
# for i in range(N):
#     ax[0].plot(sortedRF[i], sortedU2[i])
#     ax[1].bar(i,len(sortedRF[i]))
#     ax[2].bar(i,np.max(sortedRF[i]))
# plt.show()

# for i, v in enumerate(clipped_indices):
#     x = np.linspace(np.min(sortedRF[i]),np.max(sortedRF[i]),50)
#     fig, ax = plt.subplots(1,1,figsize=(10,5))
#     im = ax.plot(x, f[v](x))
#     ax.plot(sortedRF[i], sortedU2[i], ':', color= im[0].get_color())
#     ax.set_ylim(0,np.max(sortedU2[i]))
#     plt.show()

#########################################################################################









# %%#########################################################################################

# Import experimental data
force_height_data = np.loadtxt(dataPath+'Force-height-data.csv', dtype=float, delimiter=",", skiprows=3)[:,:2]*np.array([1,10])[None,:]

longitude_crosssection =np.loadtxt(dataPath+'7mv-sqz-trace-profile.txt', dtype=float, delimiter=",", skiprows=3)*np.array([10,10**10])[None,:]
# print(longitude_crosssection)

dirFile = list(os.listdir(dataPath))
dirFile.remove('Force-height-data.csv')
dirFile.remove('7mv-sqz-cross section.txt')
dirFile.remove('7mv-sqz-retrace-profile.txt')
dirFile.remove('7mv-sqz-trace-profile.txt')

force_data, data_length = np.zeros([len(dirFile)]), np.zeros(len(dirFile), dtype=int)
cross_section_height_data = np.zeros([len(dirFile),190,2])
for i, file in enumerate(dirFile):
    force_data[i] = float(np.loadtxt(dataPath + file, dtype=str, delimiter=",", skiprows=2, max_rows=1)[1][:-2])
    temp_data = np.loadtxt(dataPath+ file, dtype=float, delimiter=",", skiprows=3)
    data_length[i] = len(temp_data)
    cross_section_height_data[i,:len(temp_data)] = temp_data*10 # turn to angstroms


# Sort data  
cross_section_height_data = np.array([x for _, x in sorted(zip(force_data, cross_section_height_data))])
force_data = np.sort(force_data)


#%%
# Produce true surface scan
surfaceSize = 1
temp_U2, temp_RF, temp_ErrorMask, surfacePos, surfaceDims, temp_baseDims, temp_variables = AFMSimulation(remote_server, remotePath, localPath, abqscripts, abqCommand, fileName, subData, 
pdb, rotation, surfaceApprox, indentorType, 1, 5, tip_length, indentionDepth, forceRef, contrast, surfaceSize, 0, elasticProperties, meshSurface, meshBase, 
meshIndentor,  timePeriod, timeInterval, Preprocess  = True, Submission  = False, Postprocess = False, ReturnData  = True, ClippedScan  = [0.45,1])

# Initialise surface structure
xNumS, yNumS = int(surfaceDims[0]/surfaceSize)+1, int(surfaceDims[1]/surfaceSize)+1
xsi, ysi = int(xNumS/2), int(yNumS/2-2)
surfacePosX, surfacePosY = surfacePos.reshape(yNumS, xNumS, 3)[ysi,:,[0,2]].T, surfacePos.reshape(yNumS, xNumS, 3)[:,xsi,[1,2]]





# %%#########################################################################################
# Force profiles
# print(list(longitude_crosssection))

Nf = len(force_data)
indentationForce = np.sort(np.copy(force_data))*10

X, Y   = scanPos.reshape(yNum, xNum, 3)[yi,:,[0,2]].T[:,0], scanPos.reshape(yNum, xNum, 3)[:,xi,[1,2]][:,0]
X0, Y0 = np.linspace(X[0], X[-1], 250), np.linspace(Y[0], Y[-1], 250)
XZ, YZ = np.zeros([int(Nf)+1,len(X)]), np.zeros([int(Nf)+1,len(Y)])
Xi, Yi = [clipped_scanPos[i,0] for i in split_index[0]], [clipped_scanPos[i,1] for i in split_index[1]]

for j, forceRef in enumerate(indentationForce):
    XZ[j] = np.array([f[split_index[0][Xi.index(x)]](forceRef/10)  if x in Xi else 0  for x in X ])
    YZ[j] = np.array([f[split_index[1][Yi.index(y)]](forceRef)     if y in Yi else 0  for y in Y ])    


DNAheight = np.array([[np.max(XZ[i]),np.max(YZ[i])] for i, forceRef in enumerate(indentationForce)])





# %%
#  --------------------------------------------X Profile Plot---------------------------------------------------
plt.rcParams['font.size'] = 11
fig, ax = plt.subplots(1, 1, figsize = (linewidth/2, (1/1.61)*linewidth/2))
for i, forceRef in enumerate(indentationForce[1::3]):
    Fx, Fxz = X, np.clip(XZ[1::3][i], 0, 100)
    forceSpline = UnivariateSpline(Fx, Fxz, s =.2,k=3) 
    
    plot= ax.plot(X0, forceSpline(X0), '-',  label='{0:.2f}'.format(forceRef/10), lw=0.75)
    # ax.plot(Fx,Fxz, 'x', color = plot[0].get_color() )
# ax.plot(scanPosX[:,0],    scanPosX[:,1]-clearance, ':', color = 'r', lw = 1, label = 'Hard sphere\nboundary ')
ax.fill_between(surfacePosX[:,0][::-1], surfacePosX[:,1], interpolate=True, color='grey', alpha=0.2, label = 'DNA Surface ' )

data = longitude_crosssection[:,0]-np.max(longitude_crosssection[:,0])/2
ax.plot(0.875*(data[:]-17), longitude_crosssection[:][:,1], '^', color = ax.lines[0].get_color(), ms=1.5, lw=1)


ax.set_xlabel(r'x (${\AA}$)')
ax.set_ylabel(r'z (${\AA}$)')
ax.set_xlim(X.min(),X.max())
ax.set_ylim(13, 22)
ax.set_xticks([-50,0,50])
ax.set_yticks([15,20])
ax.axes.set_aspect(6) 
# ax.legend(title = 'Force (pN):', frameon=False, ncol=1, labelspacing=0, loc=[0.95,0.175])

fig.savefig(imagePath + os.sep + 'AFM_XProfile.pdf', bbox_inches = 'tight') 
plt.show()







# %%
#  --------------------------------------------Y Profile Plot---------------------------------------------------

fig, ax = plt.subplots(1, 1, figsize = (linewidth/2, (1/1.61)*linewidth/2))
n=1

for i, forceRef in enumerate(indentationForce[n::3]):
    Fy, Fyz      = Y,  np.clip(YZ[n::3][i], 0, 100)
    forceSpline  = UnivariateSpline(Fy,Fyz, s = 2, k = 3)
    
    plot = ax.plot(Y0[::-1], forceSpline(Y0), '-', label= '{0:0.2f}'.format(forceRef/10), lw=0.85)
    # ax.plot(Fy[::-1], Fyz, 'x', color = plot[0].get_color(), label= '{0:0.2f}'.format(forceRef/10), lw=0.65, ms=2)

# for i, forceRef in enumerate(force_data):
    if i==0:
        Y_exp, Z_exp = cross_section_height_data[n::3][i,:data_length[i],0]-90, cross_section_height_data[n::3][i,:data_length[i],1]
    if i == 1:
        Y_exp, Z_exp = cross_section_height_data[n::3][i,:data_length[i],0]-92.5, cross_section_height_data[n::3][i,:data_length[i],1]
    else: 
        Y_exp, Z_exp = cross_section_height_data[n::3][i,:data_length[i],0]-87.5, cross_section_height_data[n::3][i,:data_length[i],1]

    ax.plot(Y_exp[::3], Z_exp[::3],  '^', color = plot[0].get_color(), lw=1, ms=1.5)
    # ax.plot(Y_exp[::5], Z_exp[::5], ':', color = plot[0].get_color(),  lw=0.85)
# ax.plot(surfacePosY[:,0][::-1], surfacePosY[:,1], ':', color = 'k', lw = 1, label = 'DNA Surface ') 
# ax.plot(scanPosY[:,0],    scanPosY[:,1]-clearance, 'x', color = 'r', lw = 1, ms=2)#, label = 'Hard Sphere boundary')
# ax.plot(scanPosY[:,0],    scanPosY[:,1]-clearance, ':', color = 'r', lw = 1, label = 'Hard sphere\nboundary ')


ax.fill_between(surfacePosY[:,0][::-1], surfacePosY[:,1], interpolate=True, color='grey', alpha=0.2, label = 'DNA surface ' )

ax.set_xlabel(r'y (${\AA}$)')
ax.set_ylabel(r'z (${\AA}$)')
ax.set_xlim(-75, 75)
ax.set_ylim(0, 22)
ax.set_xticks([-50,0,50])
ax.set_yticks([0,10,20])
ax.axes.set_aspect(3) 

# ax.legend(title = 'Force (pN):', title_fontsize = 8, fontsize=8, frameon=False, ncol=1, labelspacing=0, loc=[0.62,0.32])

handles, labels = ax.get_legend_handles_labels()
handles.append(mpl.lines.Line2D([0], [0], marker='^', linestyle='None', markersize=1.5, color='gray'))
labels.append('Experimental data')

# Create a legend for the first line.
first_legend = ax.legend(handles=handles[:3], labels=labels[:3],title = 'Force (pN):', title_fontsize = 8, 
                         fontsize=8, frameon=False, ncol=1, labelspacing=0,loc=[0.65,0.5] )

# Add the legend manually to the Axes.
ax.add_artist(first_legend)

# Create another legend for the second line.
ax.legend(handles=handles[-2:][::-1], labels=labels[-2:][::-1], fontsize=7, frameon=False, 
          ncol=1, labelspacing=0, loc=[0.0,0.7])

fig.savefig(imagePath+ os.sep + 'AFM_YProfile.pdf', bbox_inches = 'tight') 
# plt.show()
plt.rcParams['font.size'] = 13    
















# %%

#  --------------------------------------------DNA Height Plot---------------------------------------------------
fig, ax = plt.subplots(figsize = (linewidth/2, 1/1.61*linewidth/2))

plot0 = ax.plot(indentationForce/10, DNAheight[:,1], label = 'DNA Height', color = 'k')
ax.plot(force_height_data[:,0], force_height_data[:,1], '^', label = 'Experimental DNA Height', color = 'k', ms=2.5)

ax.axes.set_aspect(10) 
ax.set_xlabel("Indentation force (pN)")
ax.set_ylabel(r"DNA height(${\AA}$)")
# ax.yaxis.set_label_coords(-0.14, 0.4)
                            
handles0, labels0 = ax.get_legend_handles_labels()

fig.savefig(imagePath + os.sep + 'AFM_StructureAnalysisPlot.pdf', bbox_inches = 'tight') 
plt.show()










#%%
# -----------------------------------------------------------Set Variable--------------------------------------------------------- 
# Tip variables
rIndentor, theta, tip_length, r_int, z_int,r_top = tipDims[:-1]
# Produce spherical tip with polar coordinates
r, phi = np.linspace(-r_top, r_top, 100), np.linspace(-np.pi/2,np.pi/2, 100)   

# Set material properties
E_true, v = elasticProperties 
E_eff     = E_true/(1-v**2) 
# Set constant to normalise dimensionaless forces
F_dim, maxRF = (E_eff*rIndentor**2), max(RF)/(E_eff*rIndentor**2)    
normalizer   = mpl.colors.PowerNorm(0.35, 0, contrast*(maxRF).max() )
# Set plot colormap
colormap = mpl.colormaps.get_cmap('coolwarm')
colormap.set_bad('grey') 



Zmax=40
Nz = 50
ZX, ZY = np.zeros([Nz,len(X)]), np.zeros([Nz,len(Y)])

for i , z in enumerate(np.linspace(0,Zmax,Nz)):
    ZX[i] = np.array([g[split_index[0][Xi.index(x)]](z) if x in Xi else 0 for x in X ])
    ZY[i] = np.array([g[split_index[1][Yi.index(y)]](z) if y in Yi else 0 for y in Y ])    

# Connect contour points smoothly with a spline
Fx, Fxz = X, np.clip(XZ[::3][2], 0, 100)
Fy, Fyz = Y,  np.clip(YZ[::3][2], 0, 100)
forceSplineX, forceSplineY = UnivariateSpline(Fx, Fxz,  s = .2,k=3), UnivariateSpline(Fy,Fyz, s = 2, k = 3)


# plt.rcParams['font.size'] = 14



# %%
# Plot of force heatmaps using imshow to directly visualise 2D array
# ----------------------------------------------------2D Plots X axis--------------------------------------------------------         
fig, ax = plt.subplots(1,1, figsize = (linewidth/2, (1/1.61)*linewidth/2)) 
    
# 2D heat map plot without interpolation
im = ax.imshow(ZX/F_dim, origin= 'lower', cmap=colormap, interpolation='bicubic', norm= normalizer, extent = (X[0], X[-1], 0, Zmax), interpolation_stage = 'rgba')
    
# Plot spline force for contour points, contour points themselves, surface boundary using polar coordinates, and hard sphere tip convolution
ax.plot(X,                forceSplineX(X),         '-', color = 'r', lw = 1, label = 'Fitted Fource Contour')
ax.plot(scanPosX[:,0],    scanPosX[:,1]-clearance, ':', color = 'r', lw = 1, label = 'Hard Sphere boundary')
ax.plot(surfacePosX[:,0], surfacePosX[:,1],        ':', color = 'w', lw = 1, label = 'Surface boundary') 

# Plot indentor geometry
ax.plot(r + scanPosX[int(len(scanPosX)/2),0], Zconical(r, 0, r_int, z_int, theta, rIndentor, tip_length) + scanPosX[int(len(scanPosX)/2),1] -clearance,  color = 'w', lw = 1, label = 'Indentor boundary') 

# Set legend and axis labels, limits and title
ax.set_xlabel(r'x (${\AA}$)')
ax.set_ylabel(r'z (${\AA}$)', rotation=90,  labelpad = 5)
ax.set_xlim(X[0], X[-1])
ax.set_ylim(0, Zmax )
ax.set_yticks( np.round(10*np.linspace(0, Zmax, 3))/10 )
ax.axes.set_aspect(1.2) 
# ------------------------------------------------Plot color bar ------------------------------------------------------------
cbar= fig.colorbar(im, ax= ax , orientation = 'vertical', fraction=0.01675, pad=0.025)
cbar.set_ticks(np.array([0, 0.01, 0.03]))
cbar.set_label(r'$\frac{F}{E*R^2}$', rotation=0)
cbar.ax.yaxis.set_label_coords(7.5, 0.5)
cbar.ax.set_ylim(0, 0.03)
cbar.minorticks_on() 

fig.savefig(imagePath+ os.sep + 'AFM_CrossSectionHeatMap-X.pdf', bbox_inches = 'tight') 
plt.show()

# ----------------------------------------------------2D Plots Y axis--------------------------------------------------------    
fig, ax = plt.subplots(1,1, figsize = (linewidth/2, (1/1.61)*linewidth/2)) 

# 2D heat map plot without interpolation
im = ax.imshow(ZY/F_dim, origin= 'lower', cmap=colormap, interpolation='bicubic', norm= normalizer, extent = (Y[0], Y[-1], 0, Zmax), interpolation_stage = 'rgba')
    
# Plot spline force for contour points, contour points themselves, surface boundary using polar coordinates, and hard sphere tip convolution
ax.plot(Y,                forceSplineY(Y),         '-', color = 'r', lw = 1, label = 'Fitted Fource Contour')
ax.plot(scanPosY[:,0],    scanPosY[:,1]-clearance, ':', color = 'r', lw = 1, label = 'Hard Sphere boundary')
ax.plot(surfacePosY[:,0], surfacePosY[:,1],        ':', color = 'w', lw = 1, label = 'Surface boundary') 

# Plot indentor geometry
ax.plot(r + scanPosY[int(len(scanPosY)/2),0], Zconical(r, 0, r_int, z_int, theta, rIndentor, tip_length) + scanPosY[int(len(scanPosY)/2),1] -clearance,  color = 'w', lw = 1, label = 'Indentor boundary') 

# Set legend and axis labels, limits and title
ax.set_xlabel(r'y (${\AA}$)')
ax.set_ylabel(r'z (${\AA}$)', rotation=90,  labelpad = 5)
ax.set_xlim(Y[0], Y[-1])
ax.set_ylim(0, Zmax )
ax.set_yticks( np.round(10*np.linspace(0, Zmax, 3))/10 )
ax.axes.set_aspect('equal') 
# ax.tick_params(labelleft=False)

# ------------------------------------------------Plot color bar ------------------------------------------------------------
cbar= fig.colorbar(im, ax= ax , orientation = 'vertical', fraction=0.01675, pad=0.025)
cbar.set_ticks(np.array([0, 0.01, 0.03]))
cbar.set_label(r'$\frac{F}{E*R^2}$', rotation=0)
cbar.ax.yaxis.set_label_coords(7.5, 0.5)
cbar.ax.set_ylim(0, 0.03)
cbar.minorticks_on() 

fig.savefig(imagePath+ os.sep + 'AFM_CrossSectionHeatMap-Y.pdf', bbox_inches = 'tight') 
# plt.rcParams['font.size'] = 13    
plt.show()



# %%
