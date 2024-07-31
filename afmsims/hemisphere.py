# # Introduction
# Authors: J. Giblin-Burnham
# 


# ## Imports
# %%
# --------------------------------------------------System Imports-----------------------------------------------------
import os
import sys
import time
import subprocess
from datetime import timedelta
import paramiko
import socket
from scp import SCPClient

# ---------------------------------------------Mathematical/Plotting Imports--------------------------------------------
# Importing relevant maths and graphing modules
import numpy as np 
import math
from numpy import random   
from random import randrange

# Interpolation/ Fittting modules
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splrep, PPoly
from scipy.optimize import curve_fit

# Plotting import and settinngs
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import MaxNLocator

linewidth = 11.69/2 # inch

plt.rcParams["figure.figsize"] = (linewidth/3, (1/1.61)*linewidth/3)
plt.rcParams['figure.dpi'] = 256
plt.rcParams['font.size'] = 13
plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


# For displaying images in Markdown and  animation import
from IPython.display import Image 
from IPython.display import clear_output
 

# %% [markdown]
# ## Simulation Code Functions
# Functionalised code to automate scan and geometry calculations, remote server access, remote script submission, data anlaysis and postprocessing required to produce AFM image.

# %% [markdown]
# ### Pre-Processing Function
# Functions used in preprocessing step of simulation, including calculating scan positions and exporting variables.

# %% [markdown]
# #### Tip Functions
# Functions to produce list of tip structural parameters, alongside function to calculates and returns tip surface heights from radial  position r.

# %%
def TipStructure(rIndentor, theta_degrees, tip_length): 
    '''Produce list of tip structural parameters. 
    
    Change principle angle to radian. Calculate tangent point where 
    sphere smoothly transitions to cone for capped conical indentor.
    
    Args:
        theta_degrees (float) : Principle conical angle from z axis in degrees
        rIndentor (float)     : Radius of spherical tip portion
        tip_length (float)    : Total cone height
        
    Returns:
        tipDims (list): Geometric parameters for defining capped tip structure     
    '''
    theta = theta_degrees*(np.pi/180)
    
    # Intercept of spherical and conical section of indentor (Tangent point) 
    r_int, z_int = rIndentor*abs(np.cos(theta)), -rIndentor*abs(np.sin(theta))
    # Total radius/ footprint of indentor/ top coordinates
    r_top, z_top = (r_int+(tip_length-r_int)*abs(np.tan(theta))), tip_length-rIndentor
    
    return [rIndentor, theta, tip_length, r_int, z_int, r_top, z_top]

# %%
def Fconical(r, r0, r_int, z_int, theta, R, tip_length):
    '''Calculates and returns spherically capped conical tip surface heights from radial  position r. 
    
    Uses radial coordinate along xz plane from centre as tip is axisymmetric around z axis (bottom of tip set as zero point such z0 = R).
    
    Args:
        r (float/1D arr)   : xz radial coordinate location for tip height to be found
        r0 (float)         : xz radial coordinate for centre of tip
        r_int (float)      : xz radial coordinate of tangent point (point where sphere smoothly transitions to cone)
        z_int (float)      : Height of tangent point, where sphere smoothly transitions to cone (defined for tip centred at spheres center, as calculations assume tip centred at indentors bottom the value must be corrected to, R-z_int) 
        theta (float)      : Principle conical angle from z axis in radians
        R (float)          : Radius of spherical tip portion
        tip_length (float) : Total cone height
        
    Returns:
        Z (float/1D arr): Height of tip at xz radial coordinate 
    '''
    
    ### Constructing conical and spherical parts boundaries of tip using arrays for computation speed
    
    # ------------------------------------------------Spherical Boundary------------------------------------------------
    # For r <= r_int, z <= z_int : (z-z0)^2 +  (r-r0)^2 = R^2 --> z = z0  + ( R^2 - (r-r0)^2 )^1/2   
    
    # Using equation of sphere compute height (points outside sphere radius are complex and return nan, 
    # nan_to_num is used to set these points to max value R). The heights are clip to height of tangent point, R-z_int. 
    # Producing spherical portion for r below tangent point r_int and constant height R-zint for r values above r_int.
    
    z1 = np.clip( np.nan_to_num(R - np.sqrt(R**2 - (r-r0)**2), copy=False, nan=R ), a_min = 0 , a_max = R-abs(z_int))
    # z1 = np.clip( np.where( np.isnan( R - np.sqrt(R**2 - (r-r0)**2) ) , R, R - np.sqrt(R**2 - (r-r0)**2) ), a_min = 0 , a_max = R-np.abs(z_int))

    # -------------------------------------------------Conical Boundary-------------------------------------------------
    # r > r_int, z > z_int : z = m*abs(x-x0);  where x = r, x0 = r0 + r_int,  m = 1/tan(theta)
    
    # Using equation of cone (line) to compute height for r values larger than tangent point r_int (using where condition) 
    # For r values below r_int the height is set to zero
    
    z2 =np.where(abs(r-r0)>=r_int, (abs(r-r0)-r_int)/abs(np.tan(theta)), 0)
    
    # ------------------------------------------------Combing Boundaries------------------------------------------------
    # For r values less than r_int, combines spherical portion with zero values from conical, producing spherical section
    # For r values more than r_int, combines linear conical portion with R-z_int values from spherical, producing cone section
    Z = z1 + z2 
    
    # Optional mask values greater than tip length
    Z = np.ma.masked_greater(z1+z2, tip_length )
    return Z

# %% [markdown]
# #### Scan Functions
# Calculate scan positions of tip over surface and vertical set points above surface for each position. In addition, function to plot and visualise molecules surface and scan position.

# %%
def F_TipConvolution(x, rSurface, rIndentor):
    '''Function to calculate hard sphere tip convolution of tip and semi-sphere using mathematics above'''
    return ( np.sqrt( (rSurface + rIndentor)**2 - (abs(x).clip(max=rSurface+rIndentor))**2)- rIndentor).clip(min=0)

# %%
def ScanGeometry(indentorType, tipDims, baseDims, rSurface, Nb, clearance):
    ''' Produces array of scan locations and corresponding heights/ tip positions above surface in Angstroms (x10-10 m).
    
    The scan positions are produced creating a straight line along the centre of the surface with positions spaced by the bin size. Heights, at each position, 
    are calculated for conical indentor by set tip above sample and calculating vertical distance between of tip and molecules surface over the indnenters area. 
    Subsequently, the minimum vertical distance corresponds to the position where tip is tangential. Spherical indentors are calculated explicitly.
    
        Args:
            indentorType (str) : String defining indentor type (Spherical or Capped)
            tipDims (list)     : Geometric parameters for defining capped tip structure     
            baseDims (list)    : Geometric parameters for defining base/ substrate structure [width, height, depth] 
            rSurface (float)   : Radius of semi-sphere surface
            Nb (int)           : Number of scan positions along x axis of base
            clearance (float)  : Clearance above molecules surface indentor is set to during scan
            
        Returns:
            rackPos (arr) : Array of coordinates [x,z] of scan positions to image biomolecule 
    '''
    #  -------------------------------------Set Rack Positions from Scan Geometry---------------------------------------------
    # Intialise array of raster scan positions
    rackPos = np.zeros([Nb,2])
    
    # Create linear set of scan positions over base, along x axis
    rackPos[:,0] = np.linspace(-baseDims[0]/2, baseDims[0]/2, Nb)     
    
    [rIndentor, theta, tip_length, r_int, z_int, r_top, z_top] = tipDims
    
    # ------------------------------------------------For Conically-Capped tip------------------------------------------------
    if indentorType == 'Capped':
        
        for i, rPos in enumerate(rackPos[:,0]):
            # Array of radial positions along indentor radial extent. Set indentor position/ coordinate origin at surface height 
            # (z' = z + surfaceHeight) and calculate  vertical heights along the radial extent of indentor at position. 
            r0 = np.linspace(rPos-r_top, rPos+r_top, 1000)
            z0 = Fconical(r0, rPos, r_int, z_int, theta, rIndentor, tip_length) + rSurface
            
            # Using equation of sphere compute top heights of atoms surface along indentors radial extent (points outside sphere 
            # radius are complex and return nan, nan_to_num is used to set these points to the min value of bases surface z=0).
            z  = np.sqrt(rSurface**2 - (r0**2).clip(max = rSurface**2))
        
            # The difference in the indentor height and the surface at each point along indenoter extent, produces a dz
            # array of all the height differences between indentor and surface within the indentors boundary around this position.
            # Therefore, z' = -dz  gives an array of indentor positions when each individual part of surface atoms contacts the tip portion above.
            # Translating from z' basis (with origin at z = surfaceHeight) to z basis (with origin at the top of the base) is achieved by
            # perform translation z = z' + surfaceheight. Therefore, these tip position are given by  dz = surfaceheight - dz'. The initial height 
            # corresponds to the maximum value of dz/ min value of dz' where the tip is tangential to the surface. I.e. when dz' is minimised 
            # all others dz' tip positions will be above/ further from the surface. Therefore, at this position, the rest of the indentor wil 
            # not be in contact with the surface and it is tangential. 
            
            rackPos[i,1] = rSurface - abs((z0-z).min()) + clearance 

    # -----------------------------------------------------Else Spherical tip----------------------------------------------------
    else:
        # For spherical indentor use geometry to define scan positions
        rackPos[:,1] = F_TipConvolution(rackPos[:,0], rSurface, rIndentor)
        rackPos[:,1] += clearance             
    
    return rackPos

# %% [markdown]
# ### Remote Functions
# Functions for working on remote serve, including transfering files, submitting bash commands, submiting bash scripts for batch input files and check queue statis.

# %% [markdown]
# #### File Import/ Export Function

# %%
def ExportVariables(localPath, rackPos, variables, baseDims, tipDims, indentorType, elasticProperties ):
    '''Export simulation variables as csv and txt files to load in abaqus python scripts.

    Args:
        localPath (str)         : Path to local file/directory
        rackPos (arr)           : Array of coordinates [x,z] of scan positions to image biomolecule 
        variables (list)        : List of simulation variables: [timePeriod, timeInterval, binSize, meshSurface, meshBase, meshIndentor, indentionDepth, surfaceHeight]
        baseDims (list)         : Geometric parameters for defining base/ substrate structure [width, height, depth] 
        tipDims (list)          : Geometric parameters for defining capped tip structure     
        indentorType (str)      : String defining indentor type (Spherical or Capped)
        elasticProperties (arr) : Array of surface material properties, for elastic surface [Youngs Modulus, Poisson Ratio]
    '''
    ### Creating a folder on the Users system
    os.makedirs(localPath + os.sep + 'data', exist_ok=True)

    np.savetxt(localPath + os.sep + "data"+os.sep+"elasticProperties.csv", elasticProperties, fmt='%s', delimiter=",")
    np.savetxt(localPath + os.sep + "data"+os.sep+"variables.csv", variables, fmt='%s', delimiter=",")
    np.savetxt(localPath + os.sep + "data"+os.sep+"rackPos.csv", rackPos, fmt='%s', delimiter=",")
    np.savetxt(localPath + os.sep + "data"+os.sep+"baseDims.csv", baseDims, fmt='%s', delimiter=",")
    np.savetxt(localPath + os.sep + "data"+os.sep+"tipDims.csv", tipDims, fmt='%s', delimiter=",")
    
    with open(localPath + os.sep + 'data'+os.sep+'indentorType.txt', 'w', newline = '\n') as f:
        f.write(indentorType)

# %%
def ImportVariables(localPath):
    '''Import simulation geometry variables from csv files.

    Args:
        localPath (str)  : Path to local file/directory

    Returns:
        variables (list) : List of simulation variables: [timePeriod, timeInterval, binSize, meshSurface, meshBase, meshIndentor, indentionDepth, surfaceHeight]
        baseDims (list)  : Geometric parameters for defining base/ substrate structure [width, height, depth]             
        rackPos (arr)    : Array of coordinates [x,z] of scan positions to image biomolecule  
    '''
    variables = np.loadtxt(localPath + os.sep +"data"+os.sep+"variables.csv", delimiter=",")
    baseDims  = np.loadtxt(localPath + os.sep +"data"+os.sep+"baseDims.csv", delimiter=",")
    rackPos   = np.loadtxt(localPath + os.sep +"data"+os.sep+"rackPos.csv", delimiter=",")
    
    return variables, baseDims, rackPos

# %% [markdown]
# #### Remote Connect

# %%
def SSHconnect(remote_server, **kwargs):
    ''' Function to open ssh connecction to remote server. 
    
    A new Channel is opened and allows requested command to be executed in other functions. The function allows for ProxyJumpp/Port Forwarding/SSH Tunelling.

    Args:
        remote_server (list) : Contains varibles for remote server in list format [host, port, username, password, sshkey, home, scratch]:
                                \n - host (str):     Hostname of the server to connect to
                                \n - port (int):     Server port to connect to 
                                \n - username (str): Username to authenticate as (defaults to the current local username)        
                                \n - password (str): Used for password authentication, None if ssh-key is used; is also used for private key decryption if passphrase is not given.
                                \n - sshkey (str):   Path to private key for keyexchange if password not used, None if not used
                                \n - home (str):     Path to home directory on remote server
                                \n - scratch (str):  Path to scratch directory on remote server
    Keyword Args:
        ProxyJump (proxy_server) : Optional define whether to use a Proxy Jump to ssh through firewall; 
                                        defines varibles for proxy server in list format [host, port, username, password, sshkey, home, scratch]
                                        
    Returns: 
        ssh_client (obj) : SHH client object which allows for bash command execution and file transfer.
    '''

    host, port, username, password, sshkey, home, scratch = remote_server

    if 'ProxyJump' in kwargs:
        # Set variables for proxy port
        proxy_host, proxy_port, proxy_username, proxy_password, proxy_sshkey, proxy_home, proxy_scratch = kwargs['ProxyJump']
        hostname = socket.getfqdn()
        remote_addr = (host, int(port))
        local_addr  = (socket.gethostbyname_ex(hostname)[2][0], 22)

        # Create proxy jump/ ssh tunnel
        proxy_client = paramiko.SSHClient()
        proxy_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        proxy_client.connect(proxy_host, int(proxy_port), proxy_username, proxy_password, key_filename=proxy_sshkey)
        transport = proxy_client.get_transport()
        channel = transport.open_channel("direct-tcpip", remote_addr, local_addr)

        # SSH to clusters using paramiko module
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(host, int(port), username, password, key_filename=sshkey, sock=channel)
    
    else: 
        # SSH to clusters using paramiko module
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(host, int(port), username, password, key_filename=sshkey)

    return ssh_client

# %% [markdown]
# #### File Transfer

# %%
def RemoteSCPFiles(remote_server, files, remotePath, **kwargs):
    '''Function to make directory and transfer files to SSH server.
     
    A new Channel is opened and the files are transfered.The commands input and output streams are returned as Python file-like objects representing stdin, stdout, and stderr.
    
    Args:
        remote_server (list) : Contains varibles for remote server in list format [host, port, username, password, sshkey, home, scratch]
        files (str/list)     : File or list of file to transfer
        remotePath (str)     : Path to remote file/directory
    
    Keywords Args:
        ProxyJump (proxy_server)  : Optional define whether to use a Proxy Jump to ssh through firewall; defines varibles for proxy server in list format [host, port, username, password, sshkey, home, scratch]
        path                      : Path to data files
    '''
    # SHH to clusters
    ssh_client = SSHconnect(remote_server, **kwargs)
    stdin, stdout, stderr = ssh_client.exec_command('mkdir -p ' + remotePath)

    # SCPCLient takes a paramiko transport as an argument- Uploading content to remote directory
    scp_client = SCPClient(ssh_client.get_transport())

    if 'path' in kwargs and isinstance(kwargs['path'], str):
        scp_client.put([kwargs['path']+os.sep+file for file in files], recursive=True, remote_path = remotePath)
    else:
        scp_client.put(files, recursive=True, remote_path = remotePath)
        
    scp_client.close()
    
    ssh_client.close()

# ##### Bash Command Submission

# In[23]:

def RemoteCommand(remote_server, script, remotePath, command, **kwargs):
    '''Function to execute a command/ script submission on the SSH server. 
    
    A new Channel is opened and the requested command is executed. The commands input and output streams are returned as Python file-like objects representing stdin, stdout, and stderr.
    
    Args:
        remote_server (list) : Contains varibles for remote server in list format [host, port, username, password, sshkey, home, scratch]
        script (str)         : Script to run via bash command 
        remotePath (str)     : Path to remote file/directory
        command (str)        : Abaqus command to execute and run script   
                     
    Keywords Args:
        ProxyJump (proxy_server)  : Optional define whether to use a Proxy Jump to ssh through firewall; defines varibles for proxy server in list format [host, port, username, password, sshkey, home, scratch]
    '''
    
    ssh_client = SSHconnect(remote_server, **kwargs)
    # Execute command
    stdin, stdout, stderr = ssh_client.exec_command('cd ' + remotePath + ' \n '+ command +' '+ script +' & \n')
    lines = stdout.readlines()

    ssh_client.close()
    
    for line in lines:
        print(line)

# In[24]:
# ##### Batch File Submission

# In[25]:

def BatchSubmission(remote_server, fileName, subData, scanPos, remotePath, **kwargs):
    ''' Function to create bash script for batch submission of input file, and run them on remote server.
        
    Args:
        remote_server (list) : Contains varibles for remote server in list format [host, port, username, password, sshkey, home, scratch]
        fileName (str)       : Base File name for abaqus input files
        subData (str)        : Data for submission to serve queue [walltime, memory, cpus]
        scanPos (arr)        : Array of coordinates [x,y] of scan positions to image biomolecule (can be clipped or full) 
        remotePath (str)     : Path to remote file/directory
            
    Keywords Args:
        ProxyJump (proxy_server)          : Optional define whether to use a Proxy Jump to ssh through firewall. Defines varibles for proxy server in list format [host, port, username, password, sshkey, home, scratch]
        Submission ('serial'/ 'paralell') : Optional define whether single serial script or seperate paralell submission to queue {Default: 'serial'}  
    '''

    # For paralell mode create bash script to runs for single scan location, then loop used to submit individual scripts for each location which run in paralell
    if 'Submission' in kwargs and kwargs['Submission'] == 'paralell':
        jobs = 'abaqus interactive cpus=$NSLOTS mp_mode=mpi job=$JOB_NAME input=$JOB_NAME.inp scratch=$ABAQUS_PARALLELSCRATCH resultsformat=odb'
        
    # Otherwise, create script to run serial analysis consecutively with single submission
    else:
        # Create set of submission comands for each scan locations
        jobs = ['abaqus interactive cpus=$NSLOTS memory="90%" mp_mode=mpi standard_parallel=all job='+fileName+str(int(i))+' input='+fileName+str(int(i))+'.inp scratch=$ABAQUS_PARALLELSCRATCH' 
                for i in range(len(scanPos))]
    
    # Produce preamble to used to set up bash script
    scratch = remote_server[-1]
    lines = ['#!/bin/bash -l',
             '#$ -S /bin/bash',
             '#$ -l h_rt='+ subData[0],
             '#$ -l mem=' + subData[1],
             '#$ -pe mpi '+ subData[2],
             '#$ -wd '+scratch,
             'module load abaqus/2017 ',
             'ABAQUS_PARALLELSCRATCH="'+scratch+'" ',
             'cd ' + remotePath 
            ]
        
    # Combine to produce total  script
    lines+=jobs

    # Create script file in current directory by writing each line to file
    with open('batchScript.sh', 'w', newline = '\n') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    # SSH to clusters 
    ssh_client = SSHconnect(remote_server, **kwargs)
    stdin, stdout, stderr = ssh_client.exec_command('mkdir -p ' + remotePath)

    # SCPCLient takes a paramiko transport as an argument- Uploading content to remote directory
    scp_client = SCPClient(ssh_client.get_transport())
    scp_client.put('batchScript.sh', recursive=True, remote_path = remotePath)
    scp_client.close()
    
    # If paralell mode, submit  individual scripts for individual scan locations
    if 'Submission' in kwargs and kwargs['Submission'] == 'paralell':
        for i in range(len(scanPos)):
            # Job name set as each input file name as -N jobname is used as input variable in script
            jobName = fileName+str(int(i))
            # Command to run individual jobs
            batchCommand = 'cd ' + remotePath + ' \n qsub -N '+ jobName +' batchScript.sh \n'

            # Execute command
            stdin, stdout, stderr = ssh_client.exec_command(batchCommand)
            lines = stdout.readlines()
            print(lines)
    
    # Otherwise submit single serial scripts
    else:
        # Job name set as current directory name (change / to \\ for windows)
        jobName = remotePath.split('/')[-1]
        batchCommand = 'cd ' + remotePath + ' \n qsub -N '+ jobName +' batchScript.sh \n'

        # Execute command
        stdin, stdout, stderr = ssh_client.exec_command(batchCommand)
        lines = stdout.readlines()
        print(lines)
        
    ssh_client.close() 

# In[26]:
# ##### Queue Status Function

# In[27]:

def QueueCompletion(remote_server, **kwargs):
    '''Function to check queue statis and complete when queue is empty.

    Args:
        remote_server (list) : Contains varibles for remote server in list format [host, port, username, password, sshkey, home, scratch]
                       
    Keywords Args:
        ProxyJump (proxy_server) : Optional define whether to use a Proxy Jump to ssh through firewall; defines varibles for proxy server in list format [host, port, username, password, sshkey, home, scratch]
    '''
    # Log time
    t0 = time.time()
    complete= False

    while complete == False:
        # SSH to clusters 
        ssh_client = SSHconnect(remote_server, **kwargs)

        # Execute command to view the queue
        stdin, stdout, stderr = ssh_client.exec_command('qstat')
        lines = stdout.readlines()
        
        # Check if queue is empty
        if len(lines)==0:
            print('Complete')
            complete = True
            ssh_client.close()   
        
        # Otherwis close and wait 2 mins before checking again
        else:
            ssh_client.close() 
            time.sleep(120)

    # Return total time
    t1 = time.time()
    print(t1-t0)

# In[28]:
# ##### File Retrieval

# In[29]:

def RemoteFTPFiles(remote_server, files, remotePath, localPath, **kwargs):
    '''  Function to transfer files from directory on SSH server to local machine. 
    
    A new Channel is opened and the files are transfered. The function uses FTP file transfer.
    
    Args:
        remote_server (list) : Contains varibles for remote server in list format [host, port, username, password, sshkey, home, scratch]
        files (str )         : File to transfer
        remotePath (str)     : Path to remote file/directory
        localPath (str)      : Path to local file/directory

    Keywords Args:
        ProxyJump (proxy_server) : Optional define whether to use a Proxy Jump to ssh through firewall; defines varibles for proxy server in list format [host, port, username, password, sshkey, home, scratch]
    '''
    ### Creating a folder on the Users system
    os.makedirs(localPath + os.sep + 'data', exist_ok=True)
    
    # SSH to cluster
    ssh_client = SSHconnect(remote_server, **kwargs)

    # FTPCLient takes a paramiko transport as an argument- copy content from remote directory
    ftp_client=ssh_client.open_sftp()
    ftp_client.get(remotePath+'/'+files, localPath + os.sep + 'data'+ os.sep + files)  
    ftp_client.close()


# In[30]:
# ##### Remote Terminal

# In[31]:

def Remote_Terminal(remote_server, **kwargs):
    ''' Function to emulate cluster terminal. 
    
    Channel is opened and commands given are executed. The commands input and output streams are returned as Python file-like objects representing  
    stdin, stdout, and stderr.
    
    Args:
        remote_server (list) : Contains varibles for remote server in list format [host, port, username, password, sshkey, home, scratch]
                
    Keywords Args:
        ProxyJump (proxy_server) : Optional define whether to use a Proxy Jump to ssh through firewall; defines varibles for proxy server in list format [host, port, username, password, sshkey, home, scratch]
    '''
    
    # SHH to cluster
    ssh_client = SSHconnect(remote_server, **kwargs)
    
    # Create channel to keep connection open
    ssh_channel = ssh_client.get_transport().open_session()
    ssh_channel.get_pty()
    ssh_channel.invoke_shell()
    
    # While open accept user input commands
    while True:
        command = input('$ ')
        if command == 'exit':
            break

        ssh_channel.send(command + "\n")
        
        # Return bash output from command
        while True:
            if ssh_channel.recv_ready():
                output = ssh_channel.recv(1024)
                print(output)
            else:
                time.sleep(0.5)
                if not(ssh_channel.recv_ready()):
                    break
    # Close cluster connection
    ssh_client.close()

# %% [markdown]
# ### Submission Functions
# Function to run simulation and scripts on the remote servers. Files for variables are transfered, ABAQUS scripts are run to create parts and input files. A bash file is created and submitted to run simulation for batch of inputs. Analysis of odb files is performed and data transfered back to local machine. Using keyword arguments invidual parts of simulation previously completed can be skipped.

# %%
def RemoteSubmission(remote_server, remotePath, localPath, csvfiles, abqscripts, abqCommand, fileName, subData, rackPos, **kwargs):
    '''Function to run simulation and scripts on the remote servers. 
    
    Files for variables are transfered, ABAQUS scripts are run to create parts and input files. A bash file is created and submitted to run simulation for 
    batch of inputs. Analysis of odb files is performed and data transfered back to local machine. Using keyword arguments can submitt the submission files in parrallel.
    
    Args:
        remote_server (list) : Contains varibles for remote server in list format [host, port, username, password, sshkey, home, scratch]
        remotePath (str)  : Path to remote file/directory
        localPath (str)   : Path to local file/directory
        csvfiles (list)   : List of csv and txt files to transfer to remote server
        abqscripts (list) : List of abaqus script files to transfer to remote server
        abqCommand (str)  : Abaqus command to execute and run script
        fileName (str)    : Base File name for abaqus input files
        subData (str)     : Data for submission to serve queue [walltime, memory, cpus]
        rackPos (arr)     : Array of scan positions and initial height [x,z] to image 
    
    Keyword Args:           
        ProxyJump (proxy_server) : Optional define whether to use a Proxy Jump to ssh through firewall; defines varibles for proxy server in list format [host, port, username, password, sshkey, home, scratch]
        Submission ('serial'/ 'paralell') : Type of submission, submit pararlell scripts or single serial script for scan locations {Default: 'serial'}
    '''
    #  ---------------------------------------------File Transfer----------------------------------------------------------       
    # Transfer scripts and variable files to remote server
    RemoteSCPFiles(remote_server, csvfiles, remotePath, path = localPath+os.sep+'data', **kwargs)
    RemoteSCPFiles(remote_server, abqscripts, remotePath, **kwargs)

    print('File Transfer Complete')

    #  ----------------------------------------------Input File Creation----------------------------------------------------    
    t0 = time.time()
    print('Producing Input Files ...')

    # Produce simulation and input files
    RemoteCommand(remote_server, abqscripts[0], remotePath, abqCommand, **kwargs)

    t1 = time.time()
    print('Input File Complete - ' + str(timedelta(seconds=t1-t0)) )

    #  --------------------------------------------Batch File Submission----------------------------------------------------
    t0 = time.time()
    print('Submitting Batch Scripts ...')

    # Submit bash scripts to remote queue to carry out batch abaqus analysis
    BatchSubmission(remote_server, fileName, subData, rackPos, remotePath, **kwargs) 

    t1 = time.time()
    print('Batch Submission Complete - '+ str(timedelta(seconds=t1-t0)) + '\n' )

# %%
def DataRetrieval(remote_server, wrkDir, localPath, csvfiles, dataFiles, indentorRadius, **kwargs):
    '''Function to retrieve simulation data transfered back to local machine.
     
    Using keyword arguments to change to compilation of simulations data.
    
    Args:
        remote_server (list) : Contains varibles for remote server in list format [host, port, username, password, sshkey, home, scratch]
        remotePath (str)     : Path to remote file/directory
        localPath (str)      : Path to local file/directory
        csvfiles (list)      : List of csv and txt files to transfer to remote server
        datafiles (list)     : List of abaqus script files to transfer to remote server
        indentorRadius (arr) : Array of indentor radii of spherical tip portion varied for seperate  simulations

    Keyword Args:
        Compile(int): If passed, simulation data is compiled from seperate sets of simulations in directory in remote server to combine complete indentations. Value is set as int representing the range of directories to compile from (directories must have same root naming convention with int denoting individual directories)
        ProxyJump (proxy_server) : Optional define whether to use a Proxy Jump to ssh through firewall; defines varibles for proxy server in list format [host, port, username, password, sshkey, home, scratch]

    Returns:
        variables (list) : List of simulation variables: [timePeriod, timeInterval, binSize, meshSurface, meshIndentor, indentionDepth]
        TotalU2 (arr)    : Array of indentors z displacement in time over scan position and  for all indenter [Ni, Nb, Nt]
        TotalRF (arr)    : Array of reaction force in time on indentor reference point over scan position  and for all indenter [Ni, Nb, Nt]
        NrackPos (arr)   : Array of initial scan positions for each indenter [Ni, Nb, [x, z] ]    
    '''  
    #  -------------------------------------------------Remote Variable------------------------------------------------------------
    # Import variables from remote server used for the simulations
    for file in csvfiles:    
        RemoteFTPFiles(remote_server, file, wrkDir+'/IndenterRadius7', localPath, **kwargs)
    
    # Set simulation variables used to process data
    variables, baseDims, rackPos  = ImportVariables(localPath)
    timePeriod, timeInterval, binSize, indentionDepth, meshIndentor, meshSurface, rSurface = variables 
    
    # Set array size variables
    Nb, Nt= int(baseDims[0]/binSize)+1, int(timePeriod/ timeInterval)+1

    #  -------------------------------------------Initialise data arrays----------------------------------------------------------
    NrackPos = np.zeros([len(indentorRadius), Nb, 2])
    TotalRF  = np.zeros([len(indentorRadius), Nb, Nt])
    TotalU2  = np.zeros([len(indentorRadius), Nb, Nt])    
    
    
    #  -------------------------------------------Compiled data retrieval----------------------------------------------------------   
    if 'Compile' in kwargs.keys():  
        # For number of directories set to compile
        for n in range(kwargs['Compile']):
            
            
            for index, rIndentor in enumerate(indentorRadius):
                # Set path to file
                remotePath = wrkDir[:-1] + str(n) + '/IndenterRadius'+str(int(rIndentor))
                
                # Check file is available
                try :
                    RemoteFTPFiles(remote_server, dataFiles[0], remotePath, localPath, **kwargs)
                    RemoteFTPFiles(remote_server, dataFiles[1], remotePath, localPath, **kwargs)
                    RemoteFTPFiles(remote_server, dataFiles[2], remotePath, localPath, **kwargs)

                except:
                    None

                else:
                    # If files are available load data in to temperary variable 
                    U2 = np.array(np.loadtxt(localPath + os.sep +"data"+os.sep+dataFiles[0], delimiter=","))
                    RF = np.array(np.loadtxt(localPath + os.sep +"data"+os.sep+dataFiles[1], delimiter=","))  
                    NrackPos[index] = np.array(np.loadtxt(localPath + os.sep +"data"+os.sep+dataFiles[2], delimiter=",")) 
                    
                    # Loop through data and store indentations with less zeros/ higher sums of forces
                    for i in range(len(RF)):
                        if np.all(TotalRF[index,i]==0)  == True or np.count_nonzero(TotalRF[index,i]==0)>np.count_nonzero(RF[i]==0) or sum(RF[i]) > sum(TotalRF[index,i]): 
                            TotalU2[index,i]  = U2[i]
                            TotalRF[index,i]  = RF[i]   
                            
    #  -------------------------------------------Single Directory retrieval----------------------------------------------------------
    else: 
        # For each indentor
        for index, rIndentor in enumerate(indentorRadius):
            # Define path to file
            remotePath = wrkDir + '/IndenterRadius'+str(int(rIndentor))
            
            # Retrive data files and store in curent directory
            try :
                RemoteFTPFiles(remote_server, dataFiles[0], remotePath, localPath, **kwargs)
                RemoteFTPFiles(remote_server, dataFiles[1], remotePath, localPath, **kwargs)
                RemoteFTPFiles(remote_server, dataFiles[2], remotePath, localPath, **kwargs)

            except:
                None

            else:
                # Load and set data in array for all indentors
                TotalU2[index]  = np.array(np.loadtxt(localPath + os.sep +"data"+os.sep+dataFiles[0], delimiter=","))
                TotalRF[index]  = np.array(np.loadtxt(localPath + os.sep +"data"+os.sep+dataFiles[1], delimiter=","))  
                NrackPos[index] =  np.array(np.loadtxt(localPath + os.sep +"data"+os.sep+dataFiles[2], delimiter=","))  

    return variables, TotalU2, TotalRF, NrackPos


# %% [markdown]
# ### Post-Processing Functions
# Function for postprocessing ABAQUS simulation data, loading variables from files in current directory and process data from simulation in U2/RF files. Process data from scan position to include full data range over all scan positions. Alongside, function to plot and visualise data. Then, calculates contours/z heights of constant force in simulation data for given threshold force and visualise. Produce data analysis for simulation data.

# %% [markdown]
# #### AFM Image Function
# Function to produce force heat map over scan domain and calculate contours/z heights of constant force in simulation data for given threshold force.
# 

# %%
def ForceGrid2D(X, Z, U2, RF, rackPos, courseGrain, **kwargs):
    '''Function to produce force heat map over scan domain.

    Args:
        X (arr)             : 1D array of postions over x domain of scan positions
        Z (arr)             : 1D array of postions over z domain of scan positions, discretised into bins of courseGrain value
        U2 (arr)            : Array of indentors y indentor position over scan, discretised into bins of courseGrain value ( As opposed to displacement into surface given from simulation and used elsewhere)
        RF (arr)            : Array of reaction force on indentor reference point
        rackPos (arr)       : Array of coordinates (x,z) of scan positions to image biomolecule [Nb,[x,z]]
        courseGrain (float) : Width of bins that subdivid xz domain of raster scanning/ spacing of the positions sampled over

    Keyword Args:
        Symmetric: If false skip postprocessing step to ensure AFM image is symmetric about z-axis, instead produce image from raw data {Default: True}
    
    Returns:
        forceGrid (arr)        : 2D Array of force heatmap over xz domain of scan i.e. grid of xz positions with associated force [Nx,Nz] 
        forceGridmask (arr)    : 2D boolean array giving mask for force grid with exclude postions with no indentation data [Nx,Nz] 
        forceContour (arr)     : 2D Array of coordinates for contours of constant force given by reference force across scan positons 
        forceContourmask (arr) : 2D boolean array giving mask for force contour for zero values in which no reference force 
    '''
    #  ----------------------------------------------------Force Grid calculation------------------------------------------------------        
    # Intialise force grid array
    forceGrid = np.zeros([len(X),len(Z)])
    
    # For all x and y coordinates in course grained/binned domain
    for i in range(len(X)):
        for j in range(len(Z)):
            
            # For each indentation coordinate
            for k in range(U2.shape[1]):
                
                # If equal to Y position then set corresponding forcee grid array value to force value for that position
                if U2[i,k] == Z[j]:
                    # If symmetric kwargs, use same value for both x/-x position, i.e. x[i]  = -x[i] = x[-i-1]. Use data from position with indentation data with less zeros
                    if 'Symmetric' in kwargs.keys() and kwargs['Symmetric'] == True and np.count_nonzero(RF[i]==0) <= np.count_nonzero(RF[-i-1]==0) :
                        forceGrid[i,j] = RF[i,k]
                        forceGrid[-i-1,j] = RF[i,k]
                        
                    # Otherwise use on y values for corresponding x position in scan
                    else:
                        forceGrid[i,j] = RF[i,k]
                        

    #  -----------------------------------------------------Create Force Grid mask---------------------------------------------------      
    # Initialise mask array, 0 values include 1 excludes
    forceGridmask = np.zeros([len(X),len(Z)])  
    
    # For scan positions in force array/ same as positions in X array
    for i in range(len(RF)):
        
        # Check how maNz non-zero values there are for each postion
        k = [ k for k,v in enumerate(forceGrid[i]) if v != 0]
        
        # If there are non zero values
        if len(k)!=0:
            # Mask all grid values upto the first non zero force value position 
            for j in range(k[0]):
                forceGridmask[i,j] = 1
                
        # If all force values are zero 
        else:
            # Mask all y positions in force grid for those forces 
            k = [ k for k,v in enumerate(forceGrid[i]) if Z[k] == U2[i,0] ]
            for j in range(k[0]):
                forceGridmask[i,j] = 1

    return forceGrid, forceGridmask

# %%
def ForceContour2D(U2, RF, rackPos, forceRef, **kwargs):
    '''Function to calculate contours/z heights of constant force in simulation data for given threshold force.
    
    Args:
        U2 (arr)         : Array of indentors y indentor position over scan ( As opposed to displacement into surface given from simulation and used elsewhere)
        RF (arr)         : Array of reaction force on indentor reference point
        rackPos (arr)    : Array of coordinates (x,z) of scan positions to image biomolecule [Nb,[x,z]]
        forceRef (float) : Threshold force to evaluate indentation contours at (pN)
        
    Keyword Args:
        Symmetric: If false skip postprocessing step to ensure AFM image is symmetric about z-axis, instead produce image from raw data {Default: True}
            
    Returns:
        forceContour (arr)     : 2D Array of coordinates for contours of constant force given by reference force across scan positons 
        forceContourmask (arr) : 2D boolean array giving mask for force contour for zero values in which no reference force 
    '''
    #  ----------------------------------------------------Force Contour Calculation-------------------------------------------------         
    # Initialise arrays
    forceContour = np.zeros([len(RF),2])
    forceContourmask = np.zeros([len(RF), 2])
    
    # For scan positions in force array/ same as positions in X array
    for i in range(len(RF)):
        
        # If maximum at this position is greater than Reference force
        if np.max(RF[i]) >= forceRef:
            
            # Return index at which force is greater than force threshold
            j = [ k for k,v in enumerate(RF[i]) if v >= forceRef][0]
                        
            if 'Symmetric' in kwargs.keys() and kwargs['Symmetric'] == True and np.count_nonzero(RF[i]==0) <= np.count_nonzero(RF[-i-1]==0) :
                # Store corrsponding depth/ Y position and X position for the index
                forceContour[i] = np.array([ rackPos[i,0], U2[i,j] ])  
                forceContour[-i-1] = np.array([ rackPos[-i-1,0], U2[i,j] ]) 
                        
            elif 'Symmetric' not in kwargs.keys() or kwargs['Symmetric'] == False:
                # Store corrsponding depth/ Y position and X position for the index
                forceContour[i] = np.array([ rackPos[i,0], U2[i,j] ])  

        else:
            # If not symmetrical the position is automatically masked
            if 'Symmetric' not in kwargs.keys() or kwargs['Symmetric'] == False:
                # Otherwise position not above force reference, therefore set mask values to 1
                forceContourmask[i] = np.ones(2)
            
            # However, if symmetrical the position is only masked if the symmetric data is also less than force(as data can be take from either position to produce contour)
            elif 'Symmetric' in kwargs.keys() and kwargs['Symmetric'] == True and np.max(RF[-i-1]) < forceRef:
                # Otherwise position not above force reference, therefore set mask values to 1
                forceContourmask[i] = np.ones(2)
    
    return forceContour, forceContourmask

# %% [markdown]
# #### Force Interpolation Function
# Calculate a 2D force heatmap over the xz domain, produced from interpolated forces using Hertz model.

# %%
def F_Hertz(U, E, rIndentor, rSurface, elasticProperties):
    '''Hertzian fit for indentation data'''   
    E_true, v =  elasticProperties
    R_eff = 1/(1/rSurface +1/rIndentor)
    return (2/3) * (E/(1-v**2)) * np.sqrt(R_eff) * U**(3/2)

# %%
def ForceInterpolation(Xgrid, Zgrid, U2, RF, rackPos, rIndentor, rSurface, elasticProperties, Nt, **kwargs):
    '''Calculate a 2D force heatmap over the xz domain, produced from interpolated forces using Hertz model.
    
    Args:             
        Xgrid (arr)             : 2D array/ grid of postions over xz domain of scan positions
        Zgrid (arr)             : 2D array/ grid of postions over xz domain of scan positions       
        U2 (arr)                : Array of indentors y displacement in time over scan position and for one indenter [Ni, Nb, Nt]
        RF (arr)                : Array of reaction force in time on indentor reference point over scan position  and for one indenter [Nb, Nt]
        rackPos (arr)           : Array of initial scan positions for one indenter [Nb, [x, z]] 
        rIndentor (float)       : Indentor radius of spherical tip portion varied for seperate  simulations
        rSurface (float)        : Radius of semi-sphere surface
        elasticProperties (arr) : Array of surface material properties, for elastic surface [Youngs Modulus, Poisson Ratio]
        Nt (int)                : Number of time steps

    Keyword Args:
        Symmetric : If false skip postprocessing step to ensure AFM image is symmetric about z-axis, instead produce image from raw data {Default: True}
            
    Return:
        E_hertz (arr)      : Array of fitted elastic modulus for an indentation force value over each scan positions [Nb,Nt]
        E_contour (arr)    : Array of fitted elastic modulus (upto clipped force) across the contour of the sample [Nb]        
        F (arr)            : Array of interpolated force values over xz grid for all indentors and reference force [Ni, Nb, Nz] 
    '''

    # Initialise array to hold elastic modulus
    Nb = len(rackPos)
    E_hertz   = np.zeros([Nb, Nt])
    E_contour = np.zeros(Nb)


    # Fit Hertz equation to force/indentation for each x scan positon, use lambda function to pass constant parameters(rIndentor/ elasticProperties )
    for i, value in enumerate(rackPos):
        for t in range(1, Nt):
            u2, rf     = abs(U2[i,:t]), abs(RF[i,:t])
            popt, pcov = curve_fit(lambda x, E: F_Hertz(x, E, rIndentor, rSurface, elasticProperties), u2, rf)

            # Produce array of fitted elastic modulus over scan positions for each indentor
            E_hertz[i,t]  = popt

    forceRef = np.max(RF,axis=1) 
    forceRef = forceRef[forceRef>0].min()
   
    # Find E across scan positions all to same depth -  loop over X  positions
    for i in range(Nb):

        # If maximum at this position is greater than Reference force
        if np.max(RF[i]) >= forceRef:
            
            # Return index at which force is greater than force threshold
            j = [ k for k,v in enumerate(RF[i]) if v >= forceRef][0]   
            
            if 'Symmetric' in kwargs.keys() and kwargs['Symmetric'] == True and np.count_nonzero(RF[i]==0) <= np.count_nonzero(RF[-i-1]==0) :
                # Store corrsponding E value for the index
                E_contour[i] = E_hertz[i,j] 
                E_contour[-i-1] = E_hertz[i,j]
                        
            elif 'Symmetric' not in kwargs.keys() or kwargs['Symmetric'] == False:
                # Store corrsponding E value for the index
                E_contour[i] = E_hertz[i,j]


    # Use Elastic modulus over scan position to produce continous spline
    ESpline = UnivariateSpline(rackPos[:,0],  E_hertz[:,-1], s=1)
    # From spline interpolate youngs modulus over x domain
    E = ESpline(abs(Xgrid))
    
    # Create spline for initial scan positions
    rackSpline = UnivariateSpline(rackPos[:,0], rackPos[:,1], s = 0.001)
    # Calculate initial scan positions of x domain using scan position spline
    Zinit = F_TipConvolution(Xgrid, rSurface, rIndentor)

    # Use Hertz Eq to interpolate force over xz grid: (Yinit-Ygrid) gives indentation depths over grid
    F = F_Hertz(Zinit - Zgrid, E, rIndentor, rSurface, elasticProperties)
    
    return F, E_hertz, E_contour

# %% [markdown]
# #### FWHM and Volume

# %%
def FWHM_Volume(forceContour, NrackPos, Nf, Ni, indentorRadius, rSurface):
    '''Calculate Full Width Half Maxima and Volume for Force Contours of varying reference force using splines
    
    Args:          
        forceContour (arr)   : 2D Array of coordinates for contours of constant force given by reference force across scan positons for all indentor and reference force [Nf,Ni, Nb, [x,z]] (With mask applied).
        NrackPos (arr)       : Array of initial scan positions for each indenter [Ni, Nb, [x, z]] 
        Nf                   : Number if reference force values
        Ni                   : Number if indentor radii/ values
        indentorRadius (arr) : Array of indentor radii of spherical tip portion varied for seperate  simulations
        rSurface (float)     : Radius of semi-sphere surface
        
    Returns:
        FWHM (arr)   : Array of full width half maxima of force contour for corresponding indentor and reference force [Nf,Ni]
        Volume (arr) : Array of volume under force contour for corresponding indentor and reference force [Nf,Ni]
    '''
    #  -----------------------------------------Calculate Volume and Full Width Half Maxima--------------------------------------------  
    # Intialise arrays for to store volume and FWHM, for each indentor size and reference forces
    FWHM, Volume = np.zeros([Nf+1, Ni]), np.zeros([Nf+1, Ni])

    # Loop  for each indentor size and reference forces
    for n, rIndentor in enumerate(indentorRadius):
        
        # Calculate a overestimate of boundary of integration for volume using geometry to find root of hard sphere
        a = 1.1*np.sqrt( (rSurface + rIndentor)**2 - rIndentor**2 )   
        x = np.linspace(-a,a,100)
        
        # --------------------Set first values as hardsphere boundary for each fwhm and volume --------------------------------------- 
        # Create spline of hard sphere intiial heights
        Fx, Fz = NrackPos[n,:,0], NrackPos[n,:,1]
        forceSpline = UnivariateSpline(Fx, Fz, s = 0.001)

        # Volume can be found by integrating contour spline over bounds - cannot integrate force spline itsels as integral will 
        # only be over data range in that case
        Volume[0,n] =  UnivariateSpline(x, forceSpline(x), s = 0.001).integral(-a/1.1, a/1.1) 
        
        # Half maxima can be calculated by finding roots of spline that is tanslated vertically by half the maximum  y value
        FWHM[0,n] = np.sqrt((rSurface + rIndentor)**2 - (rSurface/2+rIndentor)**2)
        
    
        # -------------------------------------Set fwhm and volume values for each force contour--------------------------------------
        for m  in range(Nf):
            # Extract xz compontents of force contour - compressed removees masked values
            Fx, Fz = forceContour[m,n,:,0].compressed(), forceContour[m,n,:,1].compressed()
            
            # Use try loop to avoid error for contours that cannot produce splines theat 
            try: 
                # Half maxima can be calculated by finding roots of spline that is tanslated vertically by half the maximum  y value
                # FWHM[m+1,n] = abs(UnivariateSpline(Fx, Fz-Fz.max()/2, s = 0.1).roots()[0]) 
                # Spline may align incorrectly and miss root, alternative method
                tck = splrep(Fx, Fz-Fz.max()/2, s = 0)
                FWHM[m+1,n] = abs(PPoly.from_spline(tck).roots(extrapolate=True)[1])
                
            except:             
                None
            
            try:                            
                # Create a spline through contour points for integral
                forceSpline = UnivariateSpline(Fx, Fz, s = 0.1)
    
                # Find definite bounds of integral
                b = UnivariateSpline(x, forceSpline(x), s = 0.1).roots()
            
                # Volume can be found by integrating contour spline over bounds- cannot integrate force spline itsels as integral 
                # will only be over data range in that case
                Volume[m+1,n] =  UnivariateSpline(x, forceSpline(x), s = 0.01).integral(b[0], b[1]) 
                
            except:
                None

    return FWHM, Volume

# %% [markdown]
# #### Postprocessing

# %%
def Postprocessing(TotalU2, TotalRF, NrackPos, Nb, Nt, courseGrain, refForces, indentorRadius, baseDims, rSurface, elasticProperties, **kwargs):
    '''Calculate a 2D force heatmap produced from simulation over the xz domain.
    
    Args:          
        TotalU2 (arr)           : Array of indentors y displacement in time over scan position and  for all indenter [Ni, Nb, Nt]
        TotalRF (arr)           : Array of reaction force in time on indentor reference point over scan position  and for all indenter [Ni, Nb, Nt]
        NrackPos (arr)          : Array of initial scan positions for each indenter [Ni, Nb, [x, z]] 
        Nb (int)                : Number of scan positions along x axis of base
        Nt (int)                : Number of time steps
        courseGrain (float)     : Width of bins that subdivid xz domain of raster scanning/ spacing of the positions sampled over
        refForces (arr)         : Array of threshold force to evaluate indentation contours at (pN)
        indentorRadius (arr)    : Array of indentor radii of spherical tip portion varied for seperate  simulations
        baseDims (list)         : Geometric parameters for defining base/ substrate structure [width, height, depth]
        rSurface (float)        : Radius of semi-sphere surface
        elasticProperties (arr) : Array of surface material properties, for elastic surface [Youngs Modulus, Poisson Ratio]

    Keyword Args:
        Symmetric : If false skip postprocessing step to ensure AFM image is symmetric about z-axis, instead produce image from raw data {Default: True}
                    
    Returns:
        X (arr)            : 1D array of postions over x domain of scan positions
        Z (arr)            : 1D array of postions over z domain of scan positions, discretised into bins of courseGrain value
        forceGrid (arr)    : 2D Array of force heatmap over xz domain of scan i.e. grid of xz positions with associated force for all indentors and reference force [Nf, Ni, Nb, Nz] (With mask applied). 
        forceContour (arr) : 2D Array of coordinates for contours of constant force given by reference force across scan positons for all indentor and reference force [Nf,Ni, Nb, [x,z]] (With mask applied).
        FWHM (arr)         : Array of full width half maxima of force contour for corresponding indentor and reference force [Nf,Ni]
        Volume (arr)       : Array of volume under force contour for corresponding indentor and reference force [Nf,Ni]
        E_hertz (arr)      : Array of fitted elastic modulus for each indentation force value over each scan positions for each indentor [Ni,Nb,Nt]
        E_contour (arr)    : Array of fitted elastic modulus (upto clipped force) across the contour of the sample for each indenter [Ni,Nb]
        F (arr)            : Array of interpolated force values over xz grid for all indentors and reference force [Ni, Nb, Nz] 
    '''

    #  -----------------------------------------------------Initialise variables------------------------------------------------------  
    Nf = len(refForces)
    Ni = len(indentorRadius)
    
    # Convert indentation data to indentor Y displacement and discretise values into bins of width given by course grain value
    zIndentor = (TotalU2 + NrackPos[:,:,1, None])
    U2 = courseGrain*np.round(zIndentor/courseGrain)
    
    # Set X arrays of scan positions
    X = NrackPos[0,:,0]
    
    # Produce Y arrays over all Y domain of indentor position for all indentors
    Z = np.round(np.arange(U2.min(initial=0), U2.max()+courseGrain,courseGrain)*10)/10 
    

    #  -----------------------------------------------------Set force grid and force contour-----------------------------------------
    # Intialise arrays for all indentor size and reference forces
    forceContour, forceContourmask = np.zeros([Nf, Ni , Nb, 2]),     np.zeros([Nf, Ni , Nb, 2])
    forceGrid,    forceGridmask    = np.zeros([Nf, Ni, Nb, len(Z)]), np.zeros([Nf, Ni, Nb, len(Z)])
    
    # Set force grid and force contour for each indentor and refence force 
    for m  in range(Nf):
        for n in range(Ni):
            forceGrid[m,n],    forceGridmask[m,n]    = ForceGrid2D(    X, Z, U2[n], TotalRF[n], NrackPos[n], courseGrain, **kwargs)
            forceContour[m,n], forceContourmask[m,n] = ForceContour2D(zIndentor[n], TotalRF[n], NrackPos[n], refForces[m], **kwargs)
            
    # Mask force grid excluding postions with no indentation data [Nx,Nz] and mask force contour for zero values in which below reference force 
    forceGrid    = np.ma.masked_array(forceGrid, mask=forceGridmask)   
    forceContour = np.ma.masked_array(forceContour, mask=forceContourmask) 
    
    
    #  -----------------------------------------Calculate Volume and Full Width Half Maxima--------------------------------------------  
    # Intialise arrays for to store volume and FWHM, for each indentor size and reference forces
    FWHM, Volume = FWHM_Volume(forceContour, NrackPos, Nf, Ni, indentorRadius, rSurface)
                
    # Mask values equal to zero
    FWHM   = np.ma.masked_equal(FWHM, 0)            
    Volume = np.ma.masked_equal(Volume, 0)
    
    #  -----------------------------------------------------Calculate Hertz fit and interpolate force from the fit------------------------------------------------------     
    # Initialise grid arrays over xz domain
    X0     = np.linspace(-baseDims[0]/2, baseDims[0]/2, 250)
    Z0     = np.linspace(0, rSurface, 250)
    Xgrid, Zgrid = np.meshgrid(X0,Z0)   
    
    # Initialise array holding Fitted Elastic modulus and Interpolated force    
    E_hertz   = np.zeros([Ni,Nb,Nt])
    E_contour = np.zeros([Ni,Nb])
    F = np.zeros([Ni, len(X0), len(Z0)])
    
    # For each indentor
    for n, rIndentor in enumerate(indentorRadius):
        F[n], E_hertz[n], E_contour[n] = ForceInterpolation(Xgrid, Zgrid, TotalU2[n], TotalRF[n], NrackPos[n], rIndentor, rSurface, elasticProperties, Nt,**kwargs)    
        
    return X, Z, forceContour, forceGrid, FWHM, Volume, E_hertz, E_contour, F

# %% [markdown]
# ### Simulation Function
# Final simulation function

# %%
def  HemisphereSimulation(remote_server, wrkDir, localPath, abqscripts, abqCommand, fileName, subData, 
                  indentorType, indentorRadius, theta_degrees, tip_length, indentionDepths, baseDims, rSurface, 
                  refForces, courseGrain, binSize, clearance, meshSurface, meshIndentor, timePeriod, timeInterval, 
                  elasticProperties, **kwargs):
    '''Final function to automate simulation. 
    
    User inputs all variables and all results are outputted. The user gets a optionally get a surface plot of scan positions.
    Produces a heatmap of the AFM image, and 3D plots of the sample surface for given force threshold.
    
    Args:
        remote_server (list)    : Contains varibles for remote server in list format [host, port, username, password, sshkey, home, scratch]
        wrkDir (str)            : Working directory extension
        localPath (str)         : Path to local file/directory
        abqscripts (list)       : List of abaqus script files to transfer to remote server
        abqCommand (str)        : Abaqus command to execute and run script
        fileName (str)          : Base File name for abaqus input files
        subData (str)           : Data for submission to serve queue [walltime, memory, cpus]
        
        indentorType (str)      : String defining indentor type (Spherical or Capped)
        indentorRadius (arr)    : Array of indentor radii of spherical tip portion varied for seperate  simulations
        theta_degrees (float)   : Principle conical angle from z axis in degrees
        tip_length (float)      : Total cone height
        indentionDepths (arr)   : Array of maximum indentation depth into surface 
        baseDims (list)         : Dimension of base
        rSurface (float)        : Radius of semi-sphere
        
        refForces (float)       : Threshold force to evaluate indentation contours at, mimics feedback force in AFM (pN)
        courseGrain (float)     : Width of bins that subdivid xz domain of raster scanning/ spacing of the positions sampled over
        binSize(float)          : Width of bins that subdivid xz domain during raster scanning/ spacing of the positions sampled over
        clearance (float)       : Clearance above molecules surface indentor is set to during scan
        meshSurface (float)     : Value of indentor mesh given as bin size for vertices of geometry in nm (x10-9 m)
        meshIndentor (float)    : Value of indentor mesh given as bin size for vertices of geometry in nm (x10-9 m) 
        timePeriod(float)       : Total time length for ABAQUS simulation/ time step (T)
        timeInterval(float)     : Time steps data sampled over for ABAQUS simulation/ time step (dt)
        elasticProperties (arr) : Array of surface material properties, for elastic surface [Youngs Modulus, Poisson Ratio]
        
    Keyword Args:
        ProxyJump (proxy_server)          : Optional define whether to use a Proxy Jump to ssh through firewall; defines varibles for proxy server in list format [host, port, username, password, sshkey, home, scratch]
        Submission ('serial'/ 'paralell') : Type of submission, submit pararlell scripts or single serial script for scan locations {Default: 'serial'}
        Main (bool)        : If false skip preprocessing step of simulation {Default: True}
        SurfacePlot (bool) : If false skip surface plot of biomolecule and scan positions {Default: True}
        Queue (bool)       : If false skip queue completion step of simulation {Default: True}
        Analysis (bool)    : If false skip odb analysis step of simulation {Default: True}
        Retrieval (bool)   : If false skip data file retrivial from remote serve {Default: True}
        Compile(int)       : If passed, simulation data is compiled from seperate sets of simulations in directory in remote server to combine complete indentations. Value is set as int representing the range of directories to compile from (directories must have same root naming convention with int denoting individual directories)                     : 
        Postprocess (bool) : If false skip postprocessing step to produce AFM image from data {Default: True}
        DataPlot (bool)    : If false skip scatter plot of simulation data {Default: True}
        Symmetric          : If false skip postprocessing step to ensure AFM image is symmetric about z-axis, instead produce image from raw data {Default: True}
            
    Returns:
        X (arr)            : 1D array of postions over x domain of scan positions, discretised into bins of courseGrain value [Nx]
        Z (arr)            : 1D array of postions over z domain of scan positions, discretised into bins of courseGrain value [Nz]
        TotalU2 (arr)      : Array of indentors z displacement in time over scan position and  for all indenter [Ni, Nb, Nt]
        TotalRF (arr)      : Array of reaction force in time on indentor reference point over scan position  and for all indenter [Ni, Nb, Nt]
        NrackPos (arr)     : Array of initial scan positions for each indenter [Ni, Nb, [x, z]] 
        forceGrid (arr)    : 2D Array of force heatmap over xz domain of scan i.e. grid of xz positions with associated force [Nx,Nz] (With mask applied). 
        forceContour (arr) : 2D Array of coordinates for contours of constant force given by reference force across scan positons (With mask applied).
        FWHM (arr)         : Array of full width half maxima of force contour for corresponding indentor and reference force [Nf,Ni]
        Volume (arr)       : Array of volume under force contour for corresponding indentor and reference force [Nf,Ni]
        E_hertz (arr)      : Array of fitted elastic modulus for each indentation force value over each scan positions for each indentor [Ni,Nb,Nt]
        E_contour (arr)    : Array of fitted elastic modulus (upto clipped force) across the contour of the sample for each indenter [Ni,Nb]
        F (arr)            : Array of interpolated force values over xz grid for all indentors and reference force [Ni, Nb, Nz] 
    '''
    # Set intial time
    T0 = time.time()
        
    #  -----------------------------------------------------Main Simulations------------------------------------------------------  
    if 'Main' not in kwargs.keys() or kwargs['Main'] == True:                
        t0 = time.time()
        
        # For each indentor radius prdoduce main simulation and submit
        for index, rIndentor in enumerate(indentorRadius): 
            print('Indentor Radius - ', rIndentor + '\n')
               
            #  ---------------------------------------Raster Scan Positions ---------------------------------------------------
            # Calculate tip geometry to create indentor and calculate scan positions over molecule for imaging
            indentionDepth = clearance + indentionDepths[index]    
            Nb, Nt= int(baseDims[0]/binSize)+1, int(timePeriod/ timeInterval)+1

            tipDims = TipStructure(rIndentor, theta_degrees, tip_length)
            rackPos = ScanGeometry(indentorType, tipDims, baseDims, rSurface, Nb, clearance)
                
            #  -------------------------------------------Export Variable-----------------------------------------------------
            # Set list of simulation variables and export to current directory
            variables = [timePeriod, timeInterval, binSize, indentionDepth, meshIndentor, meshSurface, rSurface]
            ExportVariables(localPath, rackPos, variables, baseDims, tipDims, indentorType, elasticProperties )

            
            #  -------------------------------------------Remote Submission---------------------------------------------------
            remotePath = wrkDir +'/IndenterRadius'+str(int(rIndentor))
            csvfiles = ( "rackPos.csv", "variables.csv","baseDims.csv", "tipDims.csv", "indentorType.txt", "elasticProperties.csv" )

            RemoteSubmission(remote_server, remotePath, localPath,  csvfiles, abqscripts, abqCommand, fileName, subData, rackPos, **kwargs)
   
        t1 = time.time()
        print('Main Submission Complete - ' + str(timedelta(seconds=t1-t0)) + '\n' )     
        
        
    #  -------------------------------------------------Queue Status----------------------------------------------------------
    if 'Queue' not in kwargs.keys() or kwargs['Queue'] == True:
        t0 = time.time()
        print('Simulations Processing ...')

        # Wait for completion when queue is empty
        QueueCompletion(remote_server)

        t1 = time.time()
        print('ABAQUS Simulation Complete - '+ str(timedelta(seconds=t1-t0)) + '\n')
         
            
    #  -------------------------------------------ODB Analysis Submission----------------------------------------------------
    if 'Analysis' not in kwargs.keys() or kwargs['Analysis'] == True:
        t0 = time.time()
        print('Running ODB Analysis...')
        
        
        # For each indentor radius
        for index, rIndentor in enumerate(indentorRadius):
            print('Indentor Radius:', rIndentor) 
            
            # ODB analysis script to run, extracts data from simulation and sets it in csv file on server
            remotePath = wrkDir + '/IndenterRadius'+str(int(rIndentor))
            
            RemoteCommand(remote_server, abqscripts[1], remotePath, abqCommand)
        
        t1 = time.time()
        print('ODB Analysis Complete - ' + str(timedelta(seconds=t1-t0)) + '\n')

        
    #  -----------------------------------------------File Retrieval----------------------------------------------------------
    if 'Retrieval' not in kwargs.keys() or kwargs['Retrieval'] == True:
        t0 = time.time()
        print('Running File Retrieval...')
        
        # Retrieve variables used for given simulation (in case variables redefined when skip kwargs used) 
        dataFiles = ('U2_Results.csv','RF_Results.csv', 'rackPos.csv')
        csvfiles  = ( "rackPos.csv", "variables.csv","baseDims.csv", "tipDims.csv")

        variables, TotalU2, TotalRF, NrackPos = DataRetrieval(remote_server, wrkDir, localPath, csvfiles, dataFiles, indentorRadius, **kwargs)

        # Export simulation data so it is saved in current directory for future use (save as a 2d array instead of 3d)
        np.savetxt(localPath+os.sep+"data"+os.sep+"variables.csv", variables, fmt='%s', delimiter=",")
        np.savetxt(localPath+os.sep+"data"+os.sep+"TotalU2.csv", TotalU2.reshape(TotalU2.shape[0], -1), fmt='%s', delimiter=",")
        np.savetxt(localPath+os.sep+"data"+os.sep+"TotalRF.csv", TotalRF.reshape(TotalRF.shape[0], -1), fmt='%s', delimiter=",")
        np.savetxt(localPath+os.sep+"data"+os.sep+"NrackPos.csv", NrackPos.reshape(NrackPos.shape[0], -1), fmt='%s', delimiter=",")

        t1 = time.time()
        print('File Retrevial Complete' + '\n')  

        
    #  -------------------------------------------------- Post-Processing-------------------------------------------------------
    if  'Postprocess' not in kwargs.keys() or kwargs['Postprocess'] == True:
        
        t0 = time.time()
        print('Running Postprocessing...')
        
        # Check if simulation files are accessible in curent directory to use if pre=processing skipped
        try:
            variables, baseDims, rackPos  = ImportVariables(localPath)
            timePeriod, timeInterval, binSize, indentionDepth, meshIndentor, meshSurface, rSurface = variables
            Nb, Nt= int(baseDims[0]/binSize)+1, int(timePeriod/ timeInterval)+1
            
            # Load saved simulation data and reshape as data sved as 2d array,  true shape is 3d
            TotalU2  = np.array(np.loadtxt(localPath+os.sep+"data"+os.sep+'TotalU2.csv', delimiter=",")).reshape(len(indentorRadius), Nb, Nt)
            TotalRF  = np.array(np.loadtxt(localPath+os.sep+"data"+os.sep+'TotalRF.csv', delimiter=",")).reshape(len(indentorRadius), Nb, Nt)    
            NrackPos = np.array(np.loadtxt(localPath+os.sep+"data"+os.sep+'NrackPos.csv', delimiter=",")).reshape(len(indentorRadius), Nb, 2)  

        # If file missing prompt user to import/ produce files 
        except:
            print('No Simulation files available, run preprocessing or import data' + '\n')
        
        
        #  ---------------------------------------------------- Data-Processing---------------------------------------------------     
        # Visualise data if set in kwargs
        if 'DataPlot' in kwargs.keys(): 
            n =  kwargs['DataPlot']
            DataPlot(NrackPos, TotalU2, TotalRF, Nb, Nt, n)
              
        # Process simulation data to produce heat map, analyse force contours, full width half maximum, volume and youngs modulus
        X, Z, forceContour, forceGrid, FWHM, Volume, E_hertz, E_contour, F = Postprocessing(TotalU2, TotalRF, NrackPos, Nb, Nt, courseGrain, refForces, indentorRadius,  baseDims, rSurface, elasticProperties, **kwargs)
        
        t1 = time.time()
        print('Postprocessing Complete' + '\n')
        
        
        # Return final time of simulation and variables
        T1 = time.time()
        print('Simulation Complete - ' + str(timedelta(seconds=T1-T0)) )
        return X, Z, TotalU2, TotalRF, NrackPos, forceContour, forceGrid, FWHM, Volume, E_hertz, E_contour, F
    
    else:
        # Return final time of simulation
        T1 = time.time()
        print('Simulation Complete - ' + str(timedelta(seconds=T1-T0)) )
        return None, None, None, None, None, None, None, None, None, None, None, None

# %% [markdown]
# ## Plot Functions
# Code to plot data from the simulation

# %% [markdown]
# ### Manuscript Contour Plot

# %%
def ContourPlotMan(X, Z, forceGrid, forceContour, indentorRadius, clearance, rSurface, baseDims, theta_degrees, tip_length, binSize, 
                   elasticProperties, normalizer, maxRF, contrast, n0, n1, n2):
    '''Function to plot a 2D force heatmap produced from simulation over the xz domain for single indenter and refereance force.
    
    Args:          
        X (arr)                 : 1D array of x coordinates over scan positions 
        Z (arr)                 : 1D array of z coordinates over scan positions 
        rackPos (arr)           : Array of initial scan positions for indenter [Nb, [x, z] ] 
        forceGrid (arr)         : 2D Array of force grid of xz positions 
        forceContour( arr)      : 2D Array of coordinates for contours of constant force given by reference force 
        indentorRadius (arr)    : Array of indentor radii of spherical tip portion varied for seperate  simulations
        clearance(float)        : Clearance above molecules surface indentor is set to during scan
        rSurface (float)        : Radius of semi-sphere
        baseDims (list)         : Dimension of base
        waveDims (list)         : Geometric parameters for defining base/ substrate structure [width, height, depth]
        theta_degrees (float)   : Principle conical angle from z axis in degrees
        tip_length (float)      : Total cone height
        binSize (float)         : Width of bins that subdivid xz domain during raster scanning/ spacing of the positions sampled over
        elasticProperties (arr) : Array of surface material properties, for elastic surface [Youngs Modulus, Poisson Ratio]
        normalizer (obj)        : Normalisation of cmap
        maxRF (float)           : Maximum Force value
        contrast (float)        : Contrast between high and low values in AFM heat map (0-1)
    '''
    # -----------------------------------------------------------Set Variable---------------------------------------------------------      
    # Set material properties
    E_true, v = elasticProperties 
    E_eff = E_true/(1-v**2) 
    # Tip variables
    rIndentor, theta, tip_length, r_int, z_int, r_top, z_top = TipStructure(indentorRadius[n1], theta_degrees, tip_length)
    
        
    # Extract xz compontents of force contour - compressed removes masked values
    Fx, Fz = np.array(forceContour[n1,:,0].compressed()), np.array(forceContour[n1,:,1].compressed())    
    # Connect contour points smoothly with a spline
    forceSpline = UnivariateSpline(Fx, Fz, s = 0.8)
    # Set constant to normalise dimensionaless forces and colormap
    F_dim = (E_eff*rIndentor**2)    
    colormap = mpl.colormaps.get_cmap('coolwarm')
    colormap.set_bad('grey') 
    
    # Increase padding to add above surface
    hPadding = 1
    # Produce spherical tip with polar coordinates
    x = np.linspace(-np.pi/2,np.pi/2, 100)
    
    plt.rcParams['font.size'] = 14
    # ---------------------------------------------------------Plots------------------------------------------------------------------- 
    # Plot of force heatmaps using imshow to directly visualise 2D array
    fig, ax = plt.subplots(1,1, figsize = (linewidth/3, (1/1.61)*linewidth/3)) 
       
    # ----------------------------------------------------2D Plots Indentor 1--------------------------------------------------------         
    # 2D heat map plot without interpolation
    im = ax.imshow(forceGrid[n1].T/F_dim, origin= 'lower', cmap=colormap, interpolation='bicubic', norm= normalizer,
                      extent = (X[0]/rSurface, X[-1]/rSurface, Z[0]/rSurface, Z[-1]/rSurface), interpolation_stage = 'rgba')
    
    # Plot spline force for contour points, contour points themselves, surface boundary using polar coordinates, and hard sphere tip convolution
    ax.plot(X/rSurface,  forceSpline(X)/rSurface,                                        color = 'r', lw = 1, label = 'Fitted Fource Contour')
    ax.plot(X/rSurface,  F_TipConvolution(X, rSurface, rIndentor)/rSurface, ':',         color = 'r', lw = 1, label = 'Hard Sphere boundary')
    ax.plot(np.sin(x),   np.cos(x), ':',                                                 color = 'w', lw = 1, label = 'Surface boundary') 

    # Plot indentor geometry
    ax.plot((7.5*2/np.pi*x)/rSurface, (Fconical(7.5*2/np.pi*x, 0, r_int, z_int, theta, rIndentor, tip_length)+rSurface)/rSurface, 
            color = 'w', lw = 1, label = 'Indentor boundary') 
    
    # Add 0 values in image for region above surface to have continous colour
    ax.imshow(np.zeros([10,10]), origin= 'lower', cmap='coolwarm', interpolation='bicubic', norm= normalizer, 
              extent = (X[0]/rSurface, X[-1]/rSurface, 1, ((1+hPadding)*rSurface)/rSurface) ) 
    
    # Set legend and axis labels, limits and title
    ax.set_xlabel(r'$x/r$')
    ax.set_ylabel(r'$z/r$', rotation=0,  labelpad = 15)
    ax.set_xlim(X[0]/rSurface, X[-1]/rSurface)
    ax.set_ylim(0, ((1+hPadding)*rSurface)/rSurface)
    ax.set_yticks(np.round(20*np.linspace(0, ((1+hPadding)*rSurface)/rSurface, 3))/20)
    ax.axes.set_aspect('equal')  
    
    
    # -----------------------------------------------------Change Indentor 2-------------------------------------------------------- 
    # Tip variables
    rIndentor, theta, tip_length, r_int, z_int, r_top, z_top = TipStructure(indentorRadius[n2], theta_degrees, tip_length)
    
    # Extract xz compontents of force contour - compressed removes masked values
    Fx, Fz = np.array(forceContour[n2,:,0].compressed()), np.array(forceContour[n2,:,1].compressed())    
    # Connect contour points smoothly with a spline
    forceSpline = UnivariateSpline(Fx, Fz, s = 0.1)
    
    # Set constant to normalise dimensionaless forces
    F_dim = (E_eff*rIndentor**2)
    
    # ----------------------------------------------------2D Plots Indentor 2--------------------------------------------------------       
    # Plot of force heatmaps using imshow to directly visualise 2D array
    fig, ax = plt.subplots(1,1, figsize = (linewidth/3, (1/1.61)*linewidth/3)) 
    # 2D heat map plot without interpolation
    im = ax.imshow(forceGrid[n1].T/F_dim, origin= 'lower', cmap=colormap, interpolation='bicubic', norm= normalizer,
                      extent = (X[0]/rSurface, X[-1]/rSurface, Z[0]/rSurface, Z[-1]/rSurface), interpolation_stage = 'rgba')
    
    # Plot spline force for contour points, contour points themselves, surface boundary using polar coordinates, and hard sphere tip convolution
    ax.plot(X/rSurface,  forceSpline(X)/rSurface,                                        color = 'r', lw = 1, label = 'Fitted Fource Contour')
    ax.plot(X/rSurface,  F_TipConvolution(X, rSurface, rIndentor)/rSurface, ':',         color = 'r', lw = 1, label = 'Hard Sphere boundary')
    ax.plot(np.sin(x),   np.cos(x), ':',                                                 color = 'w', lw = 1, label = 'Surface boundary') 

    # Plot indentor geometry
    ax.plot((7.5*2/np.pi*x)/rSurface, (Fconical(7.5*2/np.pi*x, 0, r_int, z_int, theta, rIndentor, tip_length)+rSurface)/rSurface, 
            color = 'w', lw = 1, label = 'Indentor boundary') 
    
    # Add 0 values in image for region above surface to have continous colour
    ax.imshow(np.zeros([10,10]), origin= 'lower', cmap='coolwarm', interpolation='bicubic', norm= normalizer, 
              extent = (X[0]/rSurface, X[-1]/rSurface, 1, ((1+hPadding)*rSurface)/rSurface) ) 
    
    # Set legend and axis labels, limits and title
    ax.set_xlabel(r'$x/r$')
    ax.set_xlim(X[0]/rSurface, X[-1]/rSurface)
    ax.set_ylim(0, ((1+hPadding)*rSurface)/rSurface)
    ax.set_yticks(np.round(20*np.linspace(0, ((1+hPadding)*rSurface)/rSurface, 3))/20)
    ax.axes.set_aspect('equal')  
    ax.tick_params(labelleft=False)
    
    # ------------------------------------------------Plot color bar ------------------------------------------------------------
    cbar= fig.colorbar(im, ax= ax , orientation = 'vertical', fraction=0.0315, pad=0.025)
    ax.text(1.8,0.9, r'$\frac{F}{E*R^2}$', rotation=0)
    # cbar.set_label(r'$\frac{F}{E*R^2}$', rotation=0)
    cbar.set_ticks(np.round(10*np.array([0, maxRF*0.25**(1/0.45), maxRF*0.75**(1/0.45), maxRF]))/10)
    cbar.ax.yaxis.set_label_coords(4, 0.6)
    
    
    #  ----------------------------------------------Raster Scan positions ------------------------------------------------------    
    Nb = int((baseDims[0]/binSize) + 1)
    
    tipDims =  TipStructure(2, theta_degrees, tip_length)
    rIndentor, theta, tip_length, r_int, z_int, r_top, z_top = tipDims

    # Raster scan grid positions for spherical and capped tip 
    rackPosCone = ScanGeometry('Capped', tipDims, baseDims, rSurface, Nb, clearance)/rSurface

    # Set index for scan position to plot tip at
    i, j = int(Nb/2), -6
    
    # Produce spherical tip with polar coordinates
    x = np.linspace(-7.5, 7.5, 1000)/rSurface
    t = np.linspace(-np.pi, np.pi, 1000)
    x1 = rIndentor*np.cos(t)/rSurface
    y1 = rIndentor*np.sin(t)/rSurface
    
    
    #  --------------------------------------------------Plot Points ----------------------------------------------------------    
    fig, ax = plt.subplots(1,1, figsize = (linewidth/3, (1/1.61)*linewidth/3)) 
    # Plot spherical and conical scan position as points
    ax.plot(rackPosCone[:,0], rackPosCone[:,1], '.', ms=3, color = 'r')

    # Plot lines for the semi-sphere surface and base using same array as indentor
    ax.plot(np.cos(t[500:]),np.sin(t[500:]),'k')
    ax.plot(x, 0*x, 'k')

    # Plot the geometry of spherical and conical tip at index i
    ax.plot(x+rackPosCone[i,0], Fconical(x*rSurface, 0, r_int, z_int, theta, rIndentor, 7)/rSurface + rackPosCone[i,1], color = '#5a71ad')
    ax.plot(x1+rackPosCone[i,0], y1+(rIndentor/rSurface)+rackPosCone[i,1], ':', color = '#fc8535')

    # Plot the geometry of spherical and conical tip at index j
    ax.plot(x+rackPosCone[j,0], Fconical(x*rSurface, 0, r_int, z_int, theta, rIndentor, 10)/rSurface + rackPosCone[j,1], color = '#5a71ad')
    ax.plot(x1+rackPosCone[j,0], y1+(rIndentor/rSurface)+rackPosCone[j,1], ':', color = '#fc8535')

    # Set axis labels, title and legend
    # ax.set_xlabel(r'$\frac{x}{r}$', color='w')
    # ax.set_ylabel(r'$\frac{z}{r}$', rotation=0,  labelpad = 15, color='w')
    ax.set_xlim(X[0]/rSurface, X[-1]/rSurface)
    ax.set_ylim(0, ((1+hPadding)*rSurface)/rSurface)
    ax.axes.set_aspect('equal')
     
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(colors='white')
    # ax.set_axis_off()
    
    # Annotating Diagram
    ax.text( rIndentor/(2*rSurface)-0.2, 1 + rIndentor/rSurface+0.1, 'R', color = '#fc8535')
    ax.text(1/2, 0.15, 'r') 
    
    plt.rcParams['font.size'] = 8    
    ax.annotate('', color = '#fc8535',
                xy=(rIndentor/rSurface, 1 + rIndentor/rSurface), xycoords='data',
                xytext=(-0.025, 1 + rIndentor/rSurface), textcoords='data',
                arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color='#fc8535'))
    ax.annotate('', xy=(1, 0.1), xycoords='data', 
                xytext=(0, 0.1), textcoords='data',
                arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color='k'))
    
    plt.rcParams['font.size'] = 13
    plt.show()

# %% [markdown]
# ### Illustrative Surface Plot

# %%
def SurfacePlot(rackPos, Nb, baseDims, tipDims, rSurface, binSize, clearance):
    '''Plot the surfaces and scan positions to visualise and check positions. 
    
    Args:
        rackPos (arr)      : Array of coordinates [x,z] of scan positions to image biomolecule  
        Nb (int)           : Number of scan positions along x axis of base
        baseDims (list)    : Geometric parameters for defining base/ substrate structure [width, height, depth] 
        tipDims (list)     : Geometric parameters for defining capped tip structure  
        rSurface (float)   : Radius of spherical surface
        clearance (float)  : Clearance above molecules surface indentor is set to during scan
    '''
    
    #  ----------------------------------------------Raster Scan positions ------------------------------------------------------
    # Set tip dimensional variables 
    rIndentor, theta, tip_length, r_int, z_int, r_top, z_top = tipDims
    Nb = int(baseDims[0]/binSize)+1
    
    # Raster scan grid positions for spherical and capped tip 
    rackPosSphere = ScanGeometry('Spherical', tipDims, baseDims, rSurface, Nb, clearance)
    rackPosCone = ScanGeometry('Capped', tipDims, baseDims, rSurface, Nb, clearance)

    # Set index for scan position to plot tip at
    i, j = int(Nb/2), -6
    
    # Produce spherical tip with polar coordinates
    x = np.linspace(-7.5, 7.5, 1000)
    t = np.linspace(-np.pi, np.pi, 1000)
    x1 = rIndentor*np.cos(t)
    y1 = rIndentor*np.sin(t)
    
    
    #  --------------------------------------------------Plot Points ----------------------------------------------------------
    # Create figure for scan positions
    fig, ax = plt.subplots(1,1, figsize = (11.69/3, 8.27/3 ))
    
    # Plot spherical and conical scan position as points
    ax.plot(rackPosCone[:,0], rackPosCone[:,1], '.', color = 'r')

    # Plot lines for the semi-sphere surface and base using same array as indentor
    ax.plot(rSurface*np.cos(t[500:]),rSurface*np.sin(t[500:]),'k')
    ax.plot(x, 0*x, 'k')

    # Plot the geometry of spherical and conical tip at index i
    ax.plot(x+rackPosCone[i,0], Fconical(x, 0, r_int, z_int, theta, rIndentor, 7)+rackPosCone[i,1], color = '#5a71ad')
    ax.plot(x1+rackPosSphere[i,0], y1+rIndentor+rackPosSphere[i,1], ':', color = '#fc8535')
    
    # Plot the geometry of spherical and conical tip at index j
    ax.plot(x+rackPosCone[j,0], Fconical(x, 0, r_int, z_int, theta, rIndentor, 10)+rackPosCone[j,1], color = '#5a71ad')
    ax.plot(x1+rackPosSphere[j,0], y1+rIndentor+rackPosSphere[j,1], ':', color = '#fc8535')

    # Set axis labels, title and legend
    ax.set_xlim(-baseDims[0]/2, baseDims[0]/2)
    ax.set_ylim(0, (2*rSurface))
    ax.set_xlabel(r'$\frac{x}{r}$', color='white')
    ax.set_ylabel(r'$\frac{z}{r}$', rotation=0,  labelpad = 10, color='white')
    ax.axes.set_aspect('equal')
    
            
    # Place axis on right for desired spacing
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
     
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(colors='white')
        
    # Annotating Diagram
    ax.text( -1, rSurface + rIndentor/2, 'R', color = '#fc8535')
    ax.annotate('', color = '#fc8535',
                xy=(0, rSurface), xycoords='data',
                xytext=(0, rSurface+rIndentor), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color='#fc8535'))
    
    ax.text(rSurface/2, 0.75, 'r')
    ax.annotate('', xy=(rSurface, 0.5), xycoords='data', 
                xytext=(0, 0.5), textcoords='data',
                arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color='k'))
    plt.show()

# %% [markdown]
# ### Data Plot 
# Function to produces scatter plot of indentation depth and reaction force to visualise and check simulation data.

# %%
def DataPlot(NrackPos, TotalU2, TotalRF, Nb, Nt, n):
    '''Produces scatter plot of indentation depth and reaction force to visualise and check simulation data.
    
    Args:
        NrackPos (arr) : Array of initial scan positions for each indenter [Ni, Nb, [x, z] ]              
        TotalU2 (arr)  : Array of indentors z displacement in time over scan position and  for all indenter [Ni, Nb, Nt]
        TotalRF (arr)  : Array of reaction force in time on indentor reference point over scan position  and for all indenter [Ni, Nb, Nt]
        Nb (int)       : Number of scan positions along x axis of base
        Nt(int)        : Number of frames in  ABAQUS simulation/ time step 
        n (int)        : Index of indenter data to plot corresponding to indices in indenterRadius
            
    '''
    # Force Curves for all the data
    fig, ax = plt.subplots(1,1)
    for i in range(len(TotalRF)):
        ax.plot(TotalU2[i],TotalRF[i])

        
    # Initialise array for indentor force and displacement        
    tipPos   = np.zeros([Nb*Nt, 2])
    tipForce = np.zeros(Nb*Nt)
    
    # Initialise count
    k = 0
    
    # Loop over array indices
    for i in range(Nb):
        for j in range( Nt ):
            #  Set array values for tip force and displacement             
            tipPos[k]   = [NrackPos[n,i,0], NrackPos[n,i,1] + TotalU2[n,i,j]] 
            tipForce[k] = TotalRF[n,i,j]            
            
            # Count array index
            k += 1

    # Scatter plot indentor displacement over scan positions
    fig ,ax = plt.subplots(1,1)
    ax.plot(tipPos[:,0], tipPos[:,1], '.')

    ax.set_xlabel(r'x (nm)', labelpad = 25)
    ax.set_ylabel(r'y (nm)', labelpad = 25)
    ax.set_title('Tip Position for Raster Scan')
    plt.show()
    
    
    # Scatter plot of force over scan positions
    fig, ax = plt.subplots(1,1)
    ax.plot(tipPos[:,0], tipForce, '.')
    
    ax.set_xlabel(r'x (nm)', labelpad = 25)
    ax.set_ylabel('F (pN)',labelpad = 25)
    ax.set_title('Force Scatter Plot for Raster Scan')
    plt.show()

# %% [markdown]
# ### Contour Plot

# %% [markdown]
# #### Interpolate

# %%
def ContourPlot(X, Z, forceGrid, forceContour, refForces, baseDims, tipDims, rSurface, elasticProperties, normalizer, maxRF, contrast):
    ''' Function to plot a 2D force heatmap produced from simulation over the xz domain for single indenter and refereance force.
    
    Args:          
        X (arr)                 : 1D array of x coordinates over scan positions 
        Z (arr)                 : 1D array of z coordinates over scan positions 
        forceGrid (arr)         : 2D Array of force grid of xz positions 
        forceContour( arr)      : 2D Array of coordinates for contours of constant force given by reference force 
        refForces (float)       : Threshold force to evaluate indentation contours at (pN)
        baseDims (list)         : Dimension of base
        tipDims (list)          : Geometric parameters for defining capped tip structure     
        rSurface (float)        : Radius of semi-sphere
        elasticProperties (arr) : Array of surface material properties, for elastic surface [Youngs Modulus, Poisson Ratio]
        normalizer (obj)        :  Normalisation of cmap
        maxRF (float)           : Maximum Force value
        contrast (float)        : Contrast between high and low values in AFM heat map (0-1)
    '''
    #  ----------------------------------------------------Set Variable-----------------------------------------------------      
    # Set material and tip  Propeties
    E_true, v = elasticProperties 
    E_eff = E_true/(1-v**2) 
    rIndentor, theta, tip_length, r_int, z_int, r_top, z_top = tipDims
    
    # Set constant to normalise dimensionaless forces
    F_dim = (E_eff*rIndentor**2)
    # Produce spherical tip with polar coordinates
    x = np.linspace(-np.pi/2,np.pi/2, 100)
    
    # Extract xz compontents of force contour - compressed removes masked values
    Fx, Fz = np.array(forceContour[:,0].compressed()), np.array(forceContour[:,1].compressed())
    # Connect contour points smoothly with a spline
    forceSpline = UnivariateSpline(Fx, Fz, s = 0.1)
    
    # Increase padding to add above surface to show indentor
    hPadding = 1
    

    #  ----------------------------------------------------2D Plots--------------------------------------------------------
    # Plot of force heatmaps using imshow to directly visualise 2D array
    fig, ax = plt.subplots(1, 1, figsize = (11.69/3, 8.27/3)) 
    
    # 2D heat map plot without interpolation
    im = ax.imshow(forceGrid.T/F_dim, origin= 'lower', cmap='coolwarm', interpolation='bicubic', norm= normalizer,
                      extent = (X[0]/rSurface, X[-1]/rSurface, Z[0]/rSurface, Z[-1]/rSurface), interpolation_stage = 'rgba')
    
    # Plot spline force for contour points, contour points themselves, surface boundary using polar coordinates, and hard sphere tip convolution
    ax.plot(X/rSurface,  forceSpline(X)/rSurface,                                        color = 'r', lw = 1, label = 'Fitted Fource Contour')
    ax.plot(X/rSurface,  F_TipConvolution(X, rSurface, rIndentor)/rSurface, ':',         color = 'r', lw = 1, label = 'Hard Sphere boundary')
    ax.plot(np.sin(x),   np.cos(x), ':',                                                 color = 'w', lw = 1, label = 'Surface boundary') 

    # Plot indentor geometry
    ax.plot((7.5*2/np.pi*x)/rSurface, (Fconical(7.5*2/np.pi*x, 0, r_int, z_int, theta, rIndentor, tip_length)+rSurface)/rSurface, 
            color = 'w', lw = 1, label = 'Indentor boundary') 
    
    # Add 0 values in image for region above surface to have continous colour
    ax.imshow(np.zeros([10,10]), origin= 'lower', cmap='coolwarm', interpolation='bicubic', norm= normalizer, 
              extent = (X[0]/rSurface, X[-1]/rSurface, 1, ((1+hPadding)*rSurface)/rSurface) ) 
    
    # Set legend and axis labels, limits and title
    ax.set_xlabel(r'$\frac{x}{r}$')
    ax.set_ylabel(r'$\frac{z}{r}$', rotation=0,  labelpad = 15)
    ax.set_xlim(X[0]/rSurface, X[-1]/rSurface)
    ax.set_ylim(0, ((1+hPadding)*rSurface)/rSurface)
    ax.set_facecolor("grey")
    ax.axes.set_aspect('equal')
        
    # --------------------------------------------Plot color bar ----------------------------------------------------------
    # plt.legend(loc = [1.35,0.5], facecolor='#d1d1d1', edgecolor = '#d1d1d1') 

    cbar= fig.colorbar(im, ax= ax , orientation = 'vertical', fraction=0.0315, pad=0.02)
    cbar.set_label(r'$\frac{F}{E*R^2}$', rotation=0)
    cbar.set_ticks(np.round(10*np.array([0, maxRF*0.25**(1/0.45), maxRF*0.75**(1/0.45), maxRF]))/10)
    cbar.ax.yaxis.set_label_coords(4, 0.6)
       
    plt.show()

# %%
def ContourPlot2(X, Z, forceGrid, forceContour, refForces, baseDims, tipDims, rSurface, elasticProperties, normalizer, maxRF, contrast):
    '''Function to plot a 2D force heatmap produced from simulation over the xz domain for single indenter and refereance force.
    
    Args:          
        X (arr)                 : 1D array of x coordinates over scan positions 
        Z (arr)                 : 1D array of z coordinates over scan positions 
        forceGrid (arr)         : 2D Array of force grid of xz positions 
        forceContour( arr)      : 2D Array of coordinates for contours of constant force given by reference force 
        refForces (float)       : Threshold force to evaluate indentation contours at (pN)
        baseDims (list)         : Dimension of base
        tipDims (list)          : Geometric parameters for defining capped tip structure     
        rSurface (float)        : Radius of semi-sphere
        elasticProperties (arr) : Array of surface material properties, for elastic surface [Youngs Modulus, Poisson Ratio]
        normalizer (obj)        :  Normalisation of cmap
        maxRF (float)           : Maximum Force value
        contrast (float)        : Contrast between high and low values in AFM heat map (0-1)
    '''
    #  ----------------------------------------------------Set Variable-----------------------------------------------------      
    # Set material and tip  Propeties
    E_true, v = elasticProperties 
    E_eff = E_true/(1-v**2) 
    rIndentor, theta, tip_length, r_int, z_int, r_top, z_top = tipDims
    
    # Set constant to normalise dimensionaless forces
    F_dim = (E_eff*rIndentor**2)
    # Produce spherical tip with polar coordinates
    x = np.linspace(-np.pi/2,np.pi/2, 100)
    
    # Extract xz compontents of force contour - compressed removes masked values
    Fx, Fz = np.array(forceContour[:,0].compressed()), np.array(forceContour[:,1].compressed())
    # Connect contour points smoothly with a spline
    forceSpline = UnivariateSpline(Fx, Fz, s = 0.1)
    
    # Increase padding to add above surface to show indentor
    hPadding = 1
    

    #  ----------------------------------------------------2D Plots--------------------------------------------------------
    # Plot of force heatmaps using imshow to directly visualise 2D array
    fig, ax = plt.subplots(1, 2, figsize = (11.69/3, 8.27/3)) 
    
    # 2D heat map plot without interpolation
    im = ax[0].imshow(forceGrid.T/F_dim, origin= 'lower', cmap='coolwarm', interpolation='bicubic', norm= normalizer,
                      extent = (X[0]/rSurface, X[-1]/rSurface, Z[0]/rSurface, Z[-1]/rSurface), interpolation_stage = 'rgba')
    
    # Plot spline force for contour points, contour points themselves, surface boundary using polar coordinates, and hard sphere tip convolution
    ax[0].plot(X/rSurface,  forceSpline(X)/rSurface,                                        color = 'r', lw = 1, label = 'Fitted Fource Contour')
    ax[0].plot(X/rSurface,  F_TipConvolution(X, rSurface, rIndentor)/rSurface, ':',         color = 'r', lw = 1, label = 'Hard Sphere boundary')
    ax[0].plot(np.sin(x),   np.cos(x), ':',                                                 color = 'w', lw = 1, label = 'Surface boundary') 

    # Plot indentor geometry
    ax[0].plot((7.5*2/np.pi*x)/rSurface, (Fconical(7.5*2/np.pi*x, 0, r_int, z_int, theta, rIndentor, tip_length)+rSurface)/rSurface, 
            color = 'w', lw = 1, label = 'Indentor boundary') 
    
    # Add 0 values in image for region above surface to have continous colour
    ax[0].imshow(np.zeros([10,10]), origin= 'lower', cmap='coolwarm', interpolation='bicubic', norm= normalizer, 
              extent = (X[0]/rSurface, X[-1]/rSurface, 1, ((1+hPadding)*rSurface)/rSurface) ) 
    
    # Set legend and axis labels, limits and title
    ax[0].set_xlabel(r'$\frac{x}{r}$')
    ax[0].set_ylabel(r'$\frac{z}{r}$', rotation=0,  labelpad = 15)
    ax[0].set_xlim(X[0]/rSurface, X[-1]/rSurface)
    ax[0].set_ylim(0, ((1+hPadding)*rSurface)/rSurface)
    ax[0].set_facecolor("grey")
    ax[0].axes.set_aspect('equal')
    
    
    # --------------------------------------------Plot second indentor ----------------------------------------------------------
    # 2D heat map plot without interpolation
    im = ax[0].imshow(forceGrid.T/F_dim, origin= 'lower', cmap='coolwarm', interpolation='bicubic', norm= normalizer,
                      extent = (X[0]/rSurface, X[-1]/rSurface, Z[0]/rSurface, Z[-1]/rSurface), interpolation_stage = 'rgba')
    
    # Plot spline force for contour points, contour points themselves, surface boundary using polar coordinates, and hard sphere tip convolution
    ax[0].plot(X/rSurface,  forceSpline(X)/rSurface,                                        color = 'r', lw = 1, label = 'Fitted Fource Contour')
    ax[0].plot(X/rSurface,  F_TipConvolution(X, rSurface, rIndentor)/rSurface, ':',         color = 'r', lw = 1, label = 'Hard Sphere boundary')
    ax[0].plot(np.sin(x),   np.cos(x), ':',                                                 color = 'w', lw = 1, label = 'Surface boundary') 

    # Plot indentor geometry
    ax[0].plot((7.5*2/np.pi*x)/rSurface, (Fconical(7.5*2/np.pi*x, 0, r_int, z_int, theta, rIndentor, tip_length)+rSurface)/rSurface, 
            color = 'w', lw = 1, label = 'Indentor boundary') 
    
    # Add 0 values in image for region above surface to have continous colour
    ax[0].imshow(np.zeros([10,10]), origin= 'lower', cmap='coolwarm', interpolation='bicubic', norm= normalizer, 
              extent = (X[0]/rSurface, X[-1]/rSurface, 1, ((1+hPadding)*rSurface)/rSurface) ) 
    
    # Set legend and axis labels, limits and title
    ax[0].set_xlabel(r'$\frac{x}{r}$')
    ax[0].set_ylabel(r'$\frac{z}{r}$', rotation=0,  labelpad = 15)
    ax[0].set_xlim(X[0]/rSurface, X[-1]/rSurface)
    ax[0].set_ylim(0, ((1+hPadding)*rSurface)/rSurface)
    ax[0].set_facecolor("grey")
    ax[0].axes.set_aspect('equal')
        
    # --------------------------------------------Plot color bar ----------------------------------------------------------
    # plt.legend(loc = [1.35,0.5], facecolor='#d1d1d1', edgecolor = '#d1d1d1') 

    cbar= fig.colorbar(im, ax= ax , orientation = 'vertical', fraction=0.0315, pad=0.02)
    cbar.set_label(r'$\frac{F}{E*R^2}$', rotation=0)
    cbar.set_ticks(np.round(10*np.array([0, maxRF*0.25**(1/0.45), maxRF*0.75**(1/0.45), maxRF]))/10)
    cbar.ax.yaxis.set_label_coords(4, 0.6)
       
    plt.show()

# %% [markdown]
# #### No Interpolation

# %%
def ContourPlotNI(X, Z, forceGrid, forceContour, refForces, baseDims, tipDims, rSurface, elasticProperties, normalizer, maxRF, contrast):
    '''Function to plot a 2D force heatmap produced from simulation over the xz domain for single indenter and refereance force.
    
    Args:          
        X (arr)                 : 1D array of x coordinates over scan positions 
        Z (arr)                 : 1D array of z coordinates over scan positions 
        RF(arr)                 : Array of reaction force on indentor reference point
        forceGrid (arr)         : 2D Array of force grid of xz positions 
        forceContour( arr)      : 2D Array of coordinates for contours of constant force given by reference force 
        refForces (float)       : Threshold force to evaluate indentation contours at (pN)
        baseDims (list)         : Dimension of base
        tipDims (list)          : Geometric parameters for defining capped tip structure     
        rSurface (float)        : Radius of semi-sphere
        elasticProperties (arr) : Array of surface material properties, for elastic surface [Youngs Modulus, Poisson Ratio]
        normalizer (obj)        :  Normalisation of cmap
        maxRF (float)           : Maximum Force value
        contrast (float)        : Contrast between high and low values in AFM heat map (0-1)
    '''
    #  ----------------------------------------------------Set Variable-----------------------------------------------------      
    # Set material and tip  Propeties
    E_true, v = elasticProperties 
    E_eff = E_true/(1-v**2) 
    rIndentor, theta, tip_length, r_int, z_int, r_top, z_top = tipDims
    
    # Set constant to normalise dimensionaless forces
    F_dim = (E_eff*rIndentor**2)
    # Produce spherical tip with polar coordinates
    x = np.linspace(-np.pi/2,np.pi/2, 100)
    
    # Extract xz compontents of force contour - compressed removes masked values
    Fx, Fz = np.array(forceContour[:,0].compressed()), np.array(forceContour[:,1].compressed())
    # Connect contour points smoothly with a spline
    forceSpline = UnivariateSpline(Fx, Fz, s = 0.1)
    
    # Increase padding to add above surface to show indentor
    hPadding = 1

    #  ----------------------------------------------------2D Plots--------------------------------------------------------
    # Plot of force heatmaps using imshow to directly visualise 2D array
    fig, ax = plt.subplots(1, 1, figsize = (11.69/3, 8.27/3)) 
    
    # 2D heat map plot without interpolation
    im = ax.imshow(forceGrid.T/F_dim, origin= 'lower', cmap='coolwarm', interpolation='None', norm= normalizer,
                      extent = (X[0]/rSurface, X[-1]/rSurface, Z[0]/rSurface, Z[-1]/rSurface))
    
    # Plot spline force for contour points, contour points themselves, surface boundary using polar coordinates, and hard sphere tip convolution
    ax.plot(Fx/rSurface, Fz/rSurface, 'x', ms = 3,                                       color = 'k', lw = 2, label = 'Force contour for F= {0:.3f}'.format(refForces/F_dim))  
    ax.plot(X/rSurface,  F_TipConvolution(X, rSurface, rIndentor)/rSurface, ':',         color = 'r', lw = 1, label = 'Hard Sphere boundary')
    ax.plot(np.sin(x),   np.cos(x), ':',                                                 color = 'w', lw = 1, label = 'Surface boundary') 

    # Plot indentor geometry
    ax.plot((7.5*2/np.pi*x)/rSurface, (Fconical(7.5*2/np.pi*x, 0, r_int, z_int, theta, rIndentor, tip_length)+rSurface)/rSurface, 
            color = 'w', lw = 1, label = 'Indentor boundary') 
    
    # Add 0 values in image for region above surface to have continous colour
    ax.imshow(np.zeros([10,10]), origin= 'lower', cmap='coolwarm', interpolation='None', norm= normalizer, 
              extent = (X[0]/rSurface, X[-1]/rSurface, 1, ((1+hPadding)*rSurface)/rSurface) ) 
    
    # Set legend and axis labels, limits and title
    ax.set_xlabel(r'$\frac{x}{r}$')
    ax.set_ylabel(r'$\frac{z}{r}$', rotation=0,  labelpad = 15)
    ax.set_xlim(X[0]/rSurface, X[-1]/rSurface)
    ax.set_ylim(0, ((1+hPadding)*rSurface)/rSurface)
    ax.set_facecolor("grey")
    ax.axes.set_aspect('equal')
        
    # --------------------------------------------Plot color bar ----------------------------------------------------------
    # plt.legend(loc = [1.35,0.5], facecolor='#d1d1d1', edgecolor = '#d1d1d1') 

    cbar= fig.colorbar(im, ax= ax , orientation = 'vertical', fraction=0.0315, pad=0.02)
    cbar.set_label(r'$\frac{F}{E*R^2}$', rotation=0)
    cbar.set_ticks(np.round(10*np.array([0, maxRF*0.25**(1/0.45), maxRF*0.75**(1/0.45), maxRF]))/10)
    cbar.ax.yaxis.set_label_coords(4, 0.6)
           
    plt.show()

# %% [markdown]
# #### Line

# %%
def LineContourPlot(X, forceContour, refForces, rSurface, tipDims, elasticProperties, normalizer, maxRF, contrast):
    '''Function to plot fitted contour lines produced from simulation over the xz domain for single indenter and  range reference force.Interpolate/none.
    
    Args:          
        X (arr)                 : 1D array of x coordinates over scan positions 
        RF(arr)                 : Array of reaction force on indentor reference point
        forceContour( arr)      : 2D Array of coordinates for contours of constant force given by reference force 
        refForces (float)       : Threshold force to evaluate indentation contours at (pN)
        rIndentor (arr)         : Array of indentor radii of spherical tip portion varied for seperate  simulations
        rSurface (float)        : Radius of semi-sphere
        elasticProperties (arr) : Array of surface material properties, for elastic surface [Youngs Modulus, Poisson Ratio]
        normalizer (obj)        :  Normalisation of cmap
        maxRF (float)           : Maximum Force value
        contrast (float)        : Contrast between high and low values in AFM heat map (0-1)
    '''
    #  ----------------------------------------------------Set Variable-----------------------------------------------------      
    rIndentor, theta, tip_length, r_int, z_int, r_top, z_top = tipDims

    # Set material Propeties
    E_true, v = elasticProperties 
    E_eff = E_true/(1-v**2) 
    # Set constant to normalise dimensionaless forces
    F_dim = (E_eff*rIndentor**2)

    # Increase padding to add above surface to show indentor
    hPadding = 1

    # Set colourmap
    cmap = mpl.cm.coolwarm
    
    # Produce spherical tip with polar coordinates
    x = np.linspace(-np.pi/2,np.pi/2, 100)
    
    #  ----------------------------------------------------2D Plots--------------------------------------------------------
    # Plot of force heatmaps using imshow to directly visualise 2D array
    fig, ax = plt.subplots(1, 1, figsize =  (11.69/3, 8.27/3))  
    
    #  --------------------------------------------------Interpolation-----------------------------------------------------      
    # Plot spline force for contour points, contour points themselves, surface boundary using polar coordinates, and hard sphere tip convolution
    ax.plot(X/rSurface,  F_TipConvolution(X, rSurface, rIndentor)/rSurface, ':', color = cmap(normalizer(0)), lw = 2, label = 'Hard\nSphere')
    ax.plot(np.sin(x),   np.cos(x), ':',                                         color = 'k', lw = 2, label = 'Surface') 
    
    # Plot indentor geometry
    ax.plot((7.5*2/np.pi*x)/rSurface, (Fconical(7.5*2/np.pi*x, 0, r_int, z_int, theta, rIndentor, tip_length)+rSurface)/rSurface, 
            color = 'k', lw = 1, label = 'Indentor') 
    
    # Plot contour lines
    for m in range(len(refForces)):
        # Extract xz compontents of force contour - compressed removes masked values
        Fx, Fz = np.array(forceContour[m,:,0].compressed()), np.array(forceContour[m,:,1].compressed())        
        try:
            # Connect contour points smoothly with a spline
            forceSpline = UnivariateSpline(Fx, Fz, s = 0.1)
        except:
            None
        else:
            ax.plot(X/rSurface,  forceSpline(X)/rSurface, color = cmap(normalizer(refForces[m]/F_dim)), lw = 1 )

    # Set legend and axis labels, limits and title
    ax.set_xlabel(r'$\frac{x}{r}$')
    ax.set_ylabel(r'$\frac{z}{r}$', rotation=0,  labelpad = 15)
    ax.set_xlim(X[0]/rSurface, X[-1]/rSurface)
    ax.set_ylim(0, ((1+hPadding)*rSurface)/rSurface)
    ax.axes.set_aspect('equal')

    # --------------------------------------------Plot color bar -----------------------------------------------------------
    # plt.legend( frameon=False) 
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=normalizer), ax=ax, fraction=0.0315, pad=0.02)
    cbar.set_label(r'$\frac{F}{E*R^2}$', rotation=0)
    cbar.set_ticks(np.round(10*np.array([0, maxRF*0.25**(1/0.45), maxRF*0.75**(1/0.45), maxRF]))/10)
    cbar.ax.yaxis.set_label_coords(4, 0.6)
       
    plt.show()

# %% [markdown]
# ### Force Interpolation

# %%
def FInterpolatePlot(X, Z, F, baseDims, tipDims, rSurface, elasticProperties, normalizer, maxRF, contrast):
    '''Function to plot a 2D force heatmap produced from simulation over the xz domain for single indenter and refereance force.
    
    Args:          
        X (arr)                 : 1D array of x coordinates over scan positions 
        Z (arr)                 : 1D array of z coordinates over scan positions 
        F (arr)                 : Array of interpolated force values over xz grid for all indentors and reference force [Ni, Nb, Nz] 
        baseDims (list)         : Dimension of base
        tipDims (list)          : Geometric parameters for defining capped tip structure     
        rSurface (float)        : Radius of semi-sphere
        elasticProperties (arr) : Array of surface material properties, for elastic surface [Youngs Modulus, Poisson Ratio]
        normalizer (obj)        :  Normalisation of cmap
        maxRF (float)           : Maximum Force value
        contrast (float)        : Contrast between high and low values in AFM heat map (0-1)
    '''
    #  ----------------------------------------------------Set Variable-----------------------------------------------------      
    # Set material and tip  Propeties
    E_true, v = elasticProperties 
    E_eff = E_true/(1-v**2) 
    rIndentor, theta, tip_length, r_int, z_int, r_top, z_top = tipDims
    
    # Set constant to normalise dimensionaless forces
    F_dim = (E_eff*rIndentor**2)
    # Produce spherical tip with polar coordinates
    x = np.linspace(-np.pi/2,np.pi/2, 100)
    
    # Increase padding to add above surface to show indentor
    hPadding = 1

    #  ----------------------------------------------------2D Plots--------------------------------------------------------
    # Plot of force heatmaps using imshow to directly visualise 2D array
    fig, ax = plt.subplots(1, 1, figsize = (11.69/3, 8.27/3)) 
    
    # 2D heat map plot without interpolation
    im = ax.imshow(F/F_dim, origin= 'lower', cmap='coolwarm', interpolation='bicubic', norm= normalizer,
                      extent = (-baseDims[0]/(2*rSurface), baseDims[0]/(2*rSurface), 0, Z[-1]/rSurface), interpolation_stage = 'rgba')
    
    # Plot spline force for contour points, contour points themselves, surface boundary using polar coordinates, and hard sphere tip convolution
    ax.plot(X/rSurface,  F_TipConvolution(X, rSurface, rIndentor)/rSurface, ':', color = 'r', lw = 1, label = 'Hard Sphere boundary')
    ax.plot(np.sin(x),   np.cos(x), ':',                                                 color = 'w', lw = 1, label = 'Surface boundary') 

    # Plot indentor geometry
    ax.plot((7.5*2/np.pi*x)/rSurface, (Fconical(7.5*2/np.pi*x, 0, r_int, z_int, theta, rIndentor, tip_length)+rSurface)/rSurface, 
            color = 'w', lw = 1, label = 'Indentor boundary') 
    
    # Set legend and axis labels, limits and title
    ax.set_xlabel(r'$\frac{x}{r}$')
    ax.set_ylabel(r'$\frac{z}{r}$', rotation=0,  labelpad = 15)
    ax.set_xlim(X[0]/rSurface, X[-1]/rSurface)
    ax.set_ylim(0, ((1+hPadding)*rSurface)/rSurface)
    ax.set_facecolor("grey")
    ax.axes.set_aspect('equal')
        
    # --------------------------------------------Plot color bar ----------------------------------------------------------
    # plt.legend(loc = [1.35,0.5], facecolor='#d1d1d1', edgecolor = '#d1d1d1') 

    cbar= fig.colorbar(im, ax= ax , orientation = 'vertical', fraction=0.0315, pad=0.02)
    cbar.set_label(r'$\frac{F}{E*R^2}$', rotation=0)
    cbar.set_ticks(np.round(10*np.array([0, maxRF*0.25**(1/0.45), maxRF*0.75**(1/0.45), maxRF]))/10)
    cbar.ax.yaxis.set_label_coords(4, 0.6)
           
    plt.show()

# %% [markdown]
# ### Full Width Half Maxima

# %%
def FWHMPlot(FWHM, indentorRadius, refForces, rSurface, elasticProperties):
    ''' Function to plot Full Width Half Maxima of force contour for each indentor for varying reference force.
    
    Args:          
        FWHM (arr)           : 2D array of y coordinates over grid positions 
        indentorRadius (arr) : 2D array of z coordinates of force contour over grid positions 
        refForces (float)    : Threshold force to evaluate indentation contours at (pN)
        rSurface (float)     : Radius of semi-sphere
        elasticProperties (arr) : Array of surface material properties, for elastic surface [Youngs Modulus, Poisson Ratio]
    '''
    
    # Set material Propeties
    E_true, v = elasticProperties 
    E_eff = E_true/(1-v**2) 
    # Set constant to normalise dimensionaless forces
    F_dim = (E_eff*indentorRadius**2)

    # Calculate fwhm for hard sphere convolution, using geometric equation above
    halfMax = rSurface/2
    hsFWHM  = np.sqrt(3/4*rSurface**2)   

    # ------------------------------------------------Plot ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize = (linewidth/2, (1/1.61)*linewidth/2) ) 
    for n in range(len(indentorRadius)):
        # Plot fwhm for each indentor as they vary over reference force - add zero value to force array with insert (for hard sphere, F=0)
        ax.plot(np.insert((refForces)/F_dim[n],0,1e-5), FWHM[:,n]/hsFWHM, '-', lw = 1, label = r'$\frac{R}{r}$= '+str(indentorRadius[n]/rSurface))
        # ax.plot(refForces, hsFWHM[n]*refForces**0, ':', label = 'Hard Sphere Radius = '+str(indentorRadius[n]))
        
    # Expected FWHM value
    ax.plot([0,3], [1,1], ':', color = 'k', lw = 1)
    
    # Set axis label and legend    
    ax.set_xlabel(r'$\frac{F}{E^*R^2}$', labelpad=5, fontsize=16)
    ax.set_ylabel(r'$\frac{FWHM_{AFM}}{FWHM_{Sample}}$', fontsize=16)
    ax.set_xscale('log')    
    ax.set_xlim(1e-3,5)
    ax.set_yticks(np.round(10*np.linspace(0,1.5, 3))/10)
    
    fig.subplots_adjust(bottom=0.15)
    plt.show()

# %% [markdown]
# ### Volume

# %%
def VolumePlot(Volume, indentorRadius, refForces, rSurface, elasticProperties):
    '''Function to plot volume under force contour for each indentor for varying reference force.

    Args: 
        Volume (arr)            : Array of volume under force contour for corresponding indentor and reference force [Nf,Ni]
        indentorRadius (arr)    : Array of indentor radii of spherical tip portion varied for seperate  simulations
        refForces (float)       : Threshold force to evaluate indentation contours at, mimics feedback force in AFM 
        rSurface (float)        : Radius of semi-sphere
        elasticProperties (arr) : Array of surface material properties, for elastic surface [Youngs Modulus, Poisson Ratio]
    '''
    # Set material Propeties
    E_true, v = elasticProperties 
    E_eff = E_true/(1-v**2)
    
    # Set constant to normalise dimensionaless forces
    F_dim = (E_eff*indentorRadius**2)
    
    # Calculate volume of surface portion
    surfaceVolume = 1/2*np.pi*rSurface**2
    
    fig, ax = plt.subplots(1, 1, figsize = (linewidth/2, (1/1.61)*linewidth/2)) 
    for n, rIndentor in enumerate(indentorRadius):
        # Plot for each indentor variation of volume over reference force- add zero value to force array with insert (for hard sphere, F=0)  
        ax.plot(np.insert((refForces)/F_dim[n],0,1e-5), Volume[:,n]/surfaceVolume, lw = 1, label = r'$\frac{R}{r}$= '+ str(rIndentor/rSurface))
    
    # Expected volume value
    ax.plot([0,3], [1,1], ':', color = 'k', lw = 1)
    
    # Set axis label and legend  
    ax.set_xlabel(r'$\frac{F}{E*R^2}$', labelpad=5, fontsize=16)
    ax.set_ylabel(r'$\frac{V_{AFM}}{V_{Sample}}$', fontsize=16)
    ax.set_xlim(1e-3,5)
    ax.set_xscale('log')
    ax.set_yticks(np.round(10*np.linspace(0,1.5, 3))/10)

    fig.subplots_adjust(bottom=0.15)
    plt.show()

# %% [markdown]
# ### Youngs Modulus

# %%
def YoungPlot(E_hertz, E_contour, TotalRF, indentorRadius, NrackPos, rSurface, elasticProperties, basePos):
    '''Function to plot elastic modulus over scan position for each indentor.
    
    Args:          
        E_hertz (arr)           : Array of fitted elastic modulus for each indentation force value over each scan positions for each indentor [Ni,Nb,Nt]
        E_contour (arr)         : Array of fitted elastic modulus (upto clipped force) across the contour of the sample for each indenter [Ni,Nb]
        TotalRF (arr)           : Array of reaction force in time on indentor reference point over scan position  and for all indenter [Ni, Nb, Nt]
        indentorRadius (arr)    : Array of indentor radii of spherical tip portion varied for seperate  simulations
        NrackPos (arr)          : Array of initial scan positions for each indenter [Ni, Nb, [x, z]] 
        rSurface (float)        : Radius of semi-sphere
        elasticProperties (arr) : Array of surface material properties, for elastic surface [Youngs Modulus, Poisson Ratio]
        basePos                 : Index of position along scan to consider vatioation in fitted E against force
    '''
    # Set constant to normalise dimensionaless forces
    E_true, v = elasticProperties    
    E_eff = E_true/(1-v**2)
    F_dim = (E_eff*indentorRadius**2)
    
    # ------------------------------------------------Plot 1------------------------------------------------------------------   
    fig, ax = plt.subplots(1, 1, figsize = (linewidth/2, (1/1.61)*linewidth/2) )
    for n, rIndentor in enumerate(indentorRadius):
        Nb2 = int(len(E_hertz[n])/2)
        Edata = np.append(E_hertz[n,Nb2:,-1][::-1], E_hertz[n,Nb2+1:,-1] )/E_true
    
        # Plot for each indentor variation of elastic modulus over scan positions
        plot = ax.plot(NrackPos[n,:,0]/rSurface, E_contour[n]/E_true, lw=1, label = r'$R/r$='+ str(rIndentor/rSurface))
        # ax.plot(NrackPos[n,:,0]/rSurface, Edata, ':', color = plot[0].get_color(), lw=0.5)
    
    # Expected elastic modulus value
    ax.plot(NrackPos[0,:,0]/rSurface, NrackPos[1,:,0]**0, ':', color = 'k', lw = 1)

    # Set axis label and legend  
    ax.set_xlabel(r'$x/r$', labelpad=5)
    ax.set_ylabel(r'$\frac{E_{AFM}}{E_{Sample}}$', fontsize=16)
    ax.set_ylim(0, 1.3)
    ax.set_yticks(np.round(10*np.linspace(0,1,2))/10)
    ax.set_yticks(np.round(10*np.linspace(0,1,2))/10)
    ax.legend(frameon=False, loc=[0.25, -0.05], labelspacing=0.2, handletextpad=0.3) 
    
    fig.subplots_adjust(bottom=0.15)
    plt.show()

    # ------------------------------------------------Plot 2 ------------------------------------------------------------------   
    fig, ax = plt.subplots(1, 1, figsize = (linewidth/2, (1/1.61)*linewidth/2) )
    for n, rIndentor in enumerate(indentorRadius):
        # Masking zero values in force
        RF = np.ma.masked_equal(TotalRF[n,basePos], 0)
        E1 = np.ma.masked_array(E_hertz[n,basePos], mask = np.ma.getmask(RF))
        E2 = np.ma.masked_array(E_hertz[n,basePos-10], mask = np.ma.getmask(RF))

        # Plot fitted youngs modulus for given indentation force
        ax.plot(RF/F_dim[n], E1/E_true, lw=1, label = r'$\frac{R}{r}$='+ str(rIndentor/rSurface))
        ax.plot(RF/F_dim[n], E2/E_true, ':', color = plt.gca().lines[-1].get_color(), lw=1, label = r'$\frac{R}{r}$='+ str(rIndentor/rSurface))
        
    # Expected elastic modulus value
    ax.plot([0,1000], [1,1], ':', color = 'k', lw = 1)
    
    # Set axis label and legend  
    ax.set_xlabel(r'$\frac{F}{E^*R^2}$', labelpad=5, fontsize=16)
    ax.set_ylabel(r'$\frac{E_{AFM}}{E_{Sample}}$', fontsize=16)
    ax.set_xlim(5e-3,5)
    ax.set_ylim(0, 1.35)
    
    ax.set_yticks(np.round(10*np.linspace(0,1,2))/10)
    ax.set_xscale('log')
    
    plt.show()