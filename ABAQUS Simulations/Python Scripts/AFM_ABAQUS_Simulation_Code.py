#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Authors: J. Giblin-Burnham


# In[1]:
# --------------------------------------------------System Imports-----------------------------------------------------
import os
import time
from datetime import timedelta


# -----------------------------------------------Server commands--------------------------------------------------------
import paramiko
from scp import SCPClient

# --------------------------------------------------Mathematical Imports------------------------------------------------
# Importing relevant maths and graphing modules
import numpy as np    
import matplotlib.pyplot as plt

linewidth = 5.92765763889 # inch

plt.rcParams["figure.figsize"] = (1.61*linewidth, linewidth)
plt.rcParams['figure.dpi'] = 256
plt.rcParams['font.size'] = 16
plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


# -----------------------------------------------Specific Imports-------------------------------------------------------
# PDB stuff:From video: youtube.com/watch?v=mL8NPpRxgJA&ab_channel=CarlosG.Oliver
from Bio.PDB import *
from Bio.PDB.PDBParser import PDBParser

# Atomic properties amd molecule visualistion
from mendeleev import element
from mendeleev.fetch import fetch_table
import nglview as nv
import py3Dmol


# In[2]:

# ## Simulation Script Functions
# Functionalised code to automate scan and geometry calculations, remote server access, remote script submission, data anlaysis and postprocessing required to produce AFM image.

# In[3]:
# ### Pre-Processing Functions
# Functions used in preprocessing step of simulation, including calculating scan positions and exporting variables.

# In[4]:
# #### Biomolecule PDB Function
# Function to imports the relevant PDB file (and takes care of the directory) in which it is saved etc for the user, returning the structure and the view using a widget.

# In[5]:

### Function to get structure from pdb file 4 digit code
def PDB(pdbid, localPath, **kwargs):
    '''
    This function imports the relevant PDB file (and takes care of the directory) in which it is saved etc for the user, 
    returning the structure and the view using a widget.
    
        Parameters:
            pdbid (str) - PDB (or CSV) file name of desired biomolecule
            
            kwargs:
                CustomPDB - Extract data from local custom pd as opposed to from PDB online
            
        Returns:
            structure (class) - Class containing proteins structural data (Atom coords/positions and masses etc...)
            view (class)      - Class for visualising the protein
    '''
    # Set biopython variables
    pdbl       = PDBList()
    parser     = MMCIFParser(QUIET=True)
    pdb_parser = PDBParser(QUIET=True,PERMISSIVE=1)
    
    # Retrieves PDB file from 4 letter code using Bio.python
    pdbl.retrieve_pdb_file(pdbid)
    
    ### Creating a folder on the Users system- location is the same as the Notebook file's
    split_pdbid = list(pdbid)
    structure_file_folder = str(split_pdbid[1]) + str(split_pdbid[2])
    
    if 'CustomPDB' in kwargs.keys() and kwargs['CustomPDB'] == True:
        # Retrieving file from the location it is saved in. Set `the_slashes` as '/' for MAC and Google Colab or '//' for Windows
        file_loc = localPath + '\\' + pdbid + '.pdb'
        
        # Defining structure i.e. '4 letter PDB ID code' and 'location'
        structure = pdb_parser.get_structure(pdbid, file_loc) 
        
    else:
        # Retrieving file from the location it is saved in. Set `the_slashes` as '/' for MAC and Google Colab or '//' for Windows
        file_loc = localPath + '\\' + structure_file_folder + '\\' + pdbid + '.cif'
        
        # Defining structure i.e. '4 letter PDB ID code' and 'location'
        structure = parser.get_structure(pdbid, file_loc) 
    
    # Plotting relevant structure using py3Dmol 
    viewer_name = 'pdb:' + pdbid
    view = py3Dmol.view(query=viewer_name).setStyle({'cartoon':{'color':'spectrum'}})
    
    return(structure, view)

# In[6]:
    
# #### Tip Functions
# Functions to produce list of tip structural parameters, alongside function to calculates and returns tip surface heights from radial  position r.

# In[7]

def TipStructure(rIndentor, theta_degrees, tip_length): 
    '''
    Produce list of tip structural parameters. Change principle angle to radian. Calculate tangent point where 
    sphere smoothly transitions to cone for capped conical indentor.
    
        Parameters:
            theta_degrees (float) - Principle conical angle from z axis in degrees
            rIndentor (float)     - Radius of spherical tip portion
            tip_length (float)    - Total cone height
            
        Returns:
            tipDims (list) - Geometric parameters for defining capped tip structure     
    '''
    theta = theta_degrees*(np.pi/180)
    
    # Intercept of spherical and conical section of indentor (Tangent point) 
    r_int, z_int = rIndentor*abs(np.cos(theta)), -rIndentor*abs(np.sin(theta))
    # Total radius/ footprint of indentor/ top coordinates
    r_top, z_top = (r_int+(tip_length-r_int)*abs(np.tan(theta))), tip_length-rIndentor
    
    return [rIndentor, theta, tip_length, r_int, z_int, r_top, z_top]


# In[8]:

def Zconical(r, r0, r_int, z_int, theta, R, tip_length):
    '''
    Calculates and returns spherically capped conical tip surface heights from radial  position r. Uses radial coordinate along
    xy plane from centre as tip is axisymmetric around z axis (bottom of tip set as zero point such z0 = R).
    
        Parameters:
            r (float/1D arr)   - xy radial coordinate location for tip height to be found
            r0 (float)         - xy radial coordinate for centre of tip
            r_int (float)      - xy radial coordinate of tangent point (point where sphere smoothly transitions to cone)
            z_int (float)      - Height of tangent point, where sphere smoothly transitions to cone (defined for tip centred at spheres 
                                 center, as calculations assume tip centred at indentors bottom the value must be corrected to, R-z_int) 
            theta (float)      - Principle conical angle from z axis in radians
            R (float)          - Radius of spherical tip portion
            tip_length (float) - Total cone height
            
        Returns:
            Z (float/1D arr)- Height of tip at xy radial coordinate 
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
    # Z = np.ma.masked_greater(z1+z2, tip_length )
    return Z


# In[9]:

def Zspherical(r, r0, r_int, z_int, theta, R, tip_length):
    '''
    Calculates and returns spherical tip surface heights from radial  position r. Uses radial coordinate along xy plane from 
    centre as tip is axisymmetric around z axis (bottom of tip set as zero point such z0 = R).
    
        Parameters:
            r (float/1D arr)   - xy radial coordinate location for tip height to be found
            r0 (float)         - xy radial coordinate for centre of tip
            r_int (float)      - xy radial coordinate for tangent point (point where sphere smoothly transitions to cone)
            z_int (float)      - Height of tangent point (point where sphere smoothly transitions to cone)
            theta (float)      - Principle conical angle from z axis in radians
            R (float)          - Radius of spherical tip portion
            tip_length (float) - Total cone height
            
        Returns:
            Z (float/1D arr)- Height of tip at xy radial coordinate 
    '''
    # Simple spherical equation: (z-z0)^2 +  (r-r0)^2 = R^2 --> z = z0  - ( R^2 - (r-r0)^2 )^1/2  
    return ( R - np.sqrt(R**2 - (r-r0)**2) ) 

# In[10]:
    
# #### Surface Functions
# Functions to orientate biomolecule, extract atomic positions and elements from structural data and calculate bse/substrate dimensions.

# In[11]:
def Rotate(domain, rotation):
    '''
    Rotate coordinates of a domain around each coordinate axis by angles given.
        Parameters:
            domain (arr)    - Array of [x,y,z] coordinates in domain to be rotated (Shape: (3) or (N,3) )
            rotation (list) - Array of [xtheta, ytheta, ztheta] rotational angle around coordinate axis:
                              # xtheta(float), angle in degrees for rotation around x axis (Row)
                              # ytheta(float), angle in degrees for rotation around y axis (Pitch)
                              # ztheta(float), angle in degrees for rotation around z axis (Yaw)
        Returns:
             rotate_domain(arr) - Rotated coordinate array
    '''
    xtheta, ytheta, ztheta = (np.pi/180)*np.array(rotation)
    
    # Row, Pitch, Yaw rotation matrices
    R_x = np.matrix( [[1,0,0],[0,np.cos(xtheta),-np.sin(xtheta)],[0,np.sin(xtheta),np.cos(xtheta)]] )   
    R_y = np.matrix( [[np.cos(ytheta),0,np.sin(ytheta)],[0,1,0],[-np.sin(ytheta),0,np.cos(ytheta)]] )
    R_z = np.matrix( [[np.cos(ztheta),-np.sin(ztheta),0],[np.sin(ztheta),np.cos(ztheta),0],[0,0,1]] )
    
    # Complete rotational matrix, from matrix multiplication
    R = R_x * R_y * R_z
    
    return np.array((R*np.asmatrix(domain).T).T)


# In[12]:
    
def MolecularStructure(structure, rotation, tipDims, indentorType, binSize, surfaceApprox):
    ''' 
    Extracts molecular data from structure class and returns array of molecules atomic coordinate and element names. Alongside, producing dictionary
    of element radii and calculating base dimensions. All distances given in Angstroms (x10-10 m).
    
        Parameters:
            structure (class)     - Class containing proteins structural data (Atom coords/positions and masses etc...)
            rotation (list)       - Array of [x,y,z] rotational angle around coordinate axis'
            tipDims (list)        - Geometric parameters for defining capped tip structure     
            indentorType (str)    - String defining indentor type (Spherical or Capped)
            binSize (float)       - Width of bins that subdivid xy domain during raster scanning/ spacing of the positions sampled over
            surfaceApprox (float) - Percentage of biomolecule assumed to be not imbedded in base/ substrate. Range: 0-1 
       
        Returns:
            atom_coord (arr)      - Array of coordinates [x,y,z] for atoms in biomolecule 
            atom_element (arr)    - Array of elements names(str) for atoms in biomolecule 
            atom_radius (dict)    - Dictionary containing van der waals radii each the element in the biomolecule 
            surfaceHeight (float) - Maximum height of biomolecule in z direction
            baseDims (arr)        - Geometric parameters for defining base/ substrate structure [width, height, depth]           
    '''
    
    #  --------------------------------------Setup Molecule Elements---------------------------------------------------
    # Extract atom element list as array
    atom_list  = structure.get_atoms()
    elements = list(fetch_table('elements')['symbol'])    
    atom_element = np.array([atom.element[0] if len(atom.element)==1 else atom.element[0] + atom.element[1:].lower() for atom in atom_list])
    # atom_element = np.array([atom.element[0] if len(atom.element)==1 else atom.element[0] + atom.element[1:].lower() if len(atom.element)==2 else 'C' for atom in atom_list])

    print(np.array([atom for atom in atom_element if atom not in elements]))
    
    # Produce dictionary of element radii in angstrom (using van de waals or vdw_radius_dreiding vdw_radius_mm3 vdw_radius_uff )
    atom_radius = dict.fromkeys(np.sort(atom_element))
    for atom in list(atom_radius.keys()):  
        if atom in elements:
            atom_radius[atom] = element(atom).vdw_radius_uff/100  
        else:
            atom_radius[atom] = element('C').vdw_radius_uff/100 
    
    #  --------------------------------------Setup Molecule Geometry---------------------------------------------------
    # Extract atom coordinates list as array in angstrom 
    atom_list  = structure.get_atoms()
    atom_coord = np.array([atom.coord for atom in atom_list]) 
    
    # Rotate coordinates of molecule
    atom_coord = Rotate(atom_coord, rotation)

    # Find extent of molecule extent
    surfaceMaxX, surfaceMinX = atom_coord[:,0].max(), atom_coord[:,0].min()
    surfaceMaxY, surfaceMinY = atom_coord[:,1].max(), atom_coord[:,1].min()
    surfaceMaxZ, surfaceMinZ = atom_coord[:,2].max(), atom_coord[:,2].min()
    
    surfaceWidthX = abs(surfaceMaxX-surfaceMinX)
    surfaceWidthY = abs(surfaceMaxY-surfaceMinY)
    surfaceWidthZ = abs(surfaceMaxZ-surfaceMinZ)
    
    # Centre molecule geometry in xy and set z=0 at the top of the base with percentage of height not imbedded
    atom_coord[:,0] = atom_coord[:,0] - surfaceMinX - surfaceWidthX/2
    atom_coord[:,1] = atom_coord[:,1] - surfaceMinY - surfaceWidthY/2
    atom_coord[:,2] = atom_coord[:,2] - surfaceMinZ - surfaceWidthZ*surfaceApprox

    
    #  --------------------------------------Setup Base/Surface Geometry---------------------------------------------------
    # Extract tip dimensions
    rIndentor, theta, tip_length, r_int, z_int, r_top, z_top = tipDims
    # Set indentor height functions and indentor radial extent/boundry for z scanPos calculation.
    if indentorType == 'Capped':
        # Extent of conical indentor is the radius of the top portion
        rBoundary  =  r_top
    else:
        # Extent of spherical indentor is the radius
        rBoundary  = rIndentor
    
    # Calculate maximum surface height with added clearance. Define substrate/Base dimensions using biomolecules extent in x and y and boundary of indentor
    surfaceHeight = 1.5*(atom_coord[:,2].max()) 
    baseDims      = np.rint([surfaceWidthX+4*rBoundary+binSize, surfaceWidthY+4*rBoundary+binSize, 2*np.max(list(atom_radius.values()))+1])
        
    return atom_coord, atom_element, atom_radius, surfaceHeight, baseDims

# In[13]:
    
# #### Scan Functions
# Calculate scan positions of tip over surface and vertical set points above surface for each position. In addition, function to plot and visualise molecules surface and scan position.

# In[14]:

def ScanGeometry(atom_coord, atom_radius, atom_element, indentorType, tipDims, baseDims, surfaceHeight, binSize, clearance):
    ''' 
    Produces array of scan locations and corresponding heights/ tip positions above surface in Angstroms (x10-10 m). Also return an array including 
    only positions where tip interact with the sample. The scan positions are produced creating a rectangular grid over bases extent with widths bin size.
    Heightss, at each position, are calculated by set tip above sample and calculating vertical distance between of tip and molecules surface over the indnenters 
    area. Subsequently, the minimum vertical distance corresponds to the position where tip is tangential.
    
        Parameters:
            atom_coord (arr)      - Array of coordinates [x,y,z] for atoms in biomolecule 
            atom_radius (dict)    - Dictionary containing van der waals radii each the element in the biomolecule 
            atom_element (arr)    - Array of elements names(str) for atoms in biomolecule 
            indentorType (str)    - String defining indentor type (Spherical or Capped)
            tipDims (list)        - Geometric parameters for defining capped tip structure     
            baseDims (arr)        - Geometric parameters for defining base/ substrate structure [width, height, depth] 
            surfaceHeight (float) - Maximum height of biomolecule in z direction
            binSize (float)       - Width of bins that subdivid xy domain during raster scanning/ spacing of the positions sampled over
            clearance (float)     - Clearance above molecules surface indentor is set to during scan
            
        Returns:
            scanPos (arr)         - Array of coordinates [x,y,z] of scan positions to image biomolecule and initial heights/ hard sphere boundary
            clipped_scanPos (arr) - Array of clipped (containing only positions where tip and molecule interact) scan positions and 
                                      initial heights [x,y,z] to image biomolecule
    '''
    #  ------------------------------------Set Scan Positions from Scan Geometry---------------------------------------------
    xNum = int(baseDims[0]/binSize)+1
    yNum = int(baseDims[1]/binSize)+1
    # Create rectangular grid of xy scan positions over base using meshgrid. 
    x = np.linspace(-baseDims[0]/2, baseDims[0]/2, xNum)
    y = np.linspace(-baseDims[1]/2, baseDims[1]/2, yNum)
        
    # Produce xy scan positions of indentor, set initial z height as clearance
    scanPos = np.array([ [x[i], y[j], clearance] for j in range(len(y)) for i in range(len(x)) ])
    
    #  --------------------------------------Set Vertical Scan Positions Positions -------------------------------------------   
    # Extract each atoms radius using radius dictionary [Natoms]
    rElement  = np.vectorize(atom_radius.get)(atom_element)    
    # Extract tip dimensions
    rIndentor, theta, tip_length, r_int, z_int, r_top, z_top = tipDims
    
   # Set indentor height functions and indentor radial extent/boundry for z scanPos calculation.
    if indentorType == 'Capped':
        # Extent of conical indentor is the radius of the top portion
        rBoundary  =  r_top
        Zstructure = Zconical
    else:
        # Extent of spherical indentor is the radius
        rBoundary  = rIndentor
        Zstructure = Zspherical                  
            
    # Array of radial positions along indentor radial extent. Set indentor position/ coordinate origin at surface height 
    # (z' = z + surfaceHeight) and calculate  vertical heights along the radial extent. 
    r = np.linspace(-rBoundary, rBoundary, 50)
    zIndentor = Zstructure(r, 0, r_int, z_int, theta, rIndentor, tip_length) + surfaceHeight
    
    N = np.linspace(0, xNum*yNum, int(np.ceil(xNum*yNum*len(atom_coord)/1.1e8)+1), dtype=int)
    for j in range(len(N)-1):
        
        # Calculate radial distance from scan position to each atom centre giving array of  [NscanPos, Natoms]
        rInteract = np.sqrt( (atom_coord[:,0]-scanPos[N[j]:N[j+1],0,None])**2 + (atom_coord[:,1]-scanPos[N[j]:N[j+1],1,None])**2 ) 

        # Mask atoms outside the indenter boundary for each scan position and produce corresponding element radiujs and z positions array. Compress to remove masked values
        rInteractMasked = np.ma.masked_greater(rInteract, rBoundary+rElement)
        mask = np.ma.getmask(rInteractMasked) 
        rInteractMasked = [ rInteractMasked[i].compressed() for i in range(N[j+1]-N[j]) ]
        zAtomMasked     = [ np.ma.masked_array(atom_coord[:,2], mask = mask[i]).compressed() for i in range(N[j+1]-N[j]) ]
        rElementMasked  = [ np.ma.masked_array( rElement, mask = mask[i] ).compressed() for i in range(N[j+1]-N[j]) ]

        # Find vertical distances from atoms to indentor surface over all scan positions inside np.nan_num(nan_num removes any infinites). Minus from zIndentor to calculate the 
        # difference in the indentor height and the atoms surface at each point along indenoter extent, produces a dz array of all the height differences between indentor and 
        # surface atoms within the indentors boundary around this position. Find the minimum (ensurring maximum is surface height with initial). Therefore, z' = -dz  gives an 
        # array of indentor positions when each individual part of surface atoms contacts the tip portion above. Translating from z' basis (with origin at z = surfaceHeight) to 
        # z basis (with origin at the top of the base) is achieved by perform translation z = z' + surfaceheight. Therefore, these tip position are given by dz = surfaceheight-dz'. 
        # The initial height corresponds to the maximum value of dz/ min value of dz' where the tip is tangential to the surface. I.e. when dz' is minimised all others dz' tip 
        # positions will be above/ further from the surface. Therefore, at this position, the rest of the indentor wil  not be in contact with the surface and it is tangential.    

        dz = np.array([(zIndentor[:,None] - np.nan_to_num((zAtomMasked[i] + np.sqrt( rElementMasked[i]**2 - (r[:,None]-rInteractMasked[i])**2)), 
                                                          copy=False, nan=0 )).min(initial=surfaceHeight)   for i in range(N[j+1]-N[j]) ])
        scanPos[N[j]:N[j+1],2] = surfaceHeight - abs(dz) + clearance 
    
    #  ---------------------------------------------Clip Scan position ---------------------------------------------------------    
    # Include only positions where tip interact with the sample. Scan position equal clearance, corresponds to indentor at base height 
    # therfore, can't indent surface (where all dz' heights were greater than surface height )
    clipped_scanPos = np.array([ [ scanPos[i,0], scanPos[i,1], scanPos[i,2] ] for i in range(len(scanPos)) if scanPos[i,2] != clearance ])
            
    return scanPos, clipped_scanPos

# In[15]:
    
def DotPlot(atom_coord, atom_radius, atom_element, scanPos, clipped_scanPos, pdb, **kwargs):
    ''' 
    Plot the molecules atoms surfaces and scan positions to visualise and check positions.
    
        Parameters:
            atom_coord (arr)        - Array of coordinates [x,y,z] for atoms in biomolecule 
            atom_radius (dict)      - Dictionary containing van der waals radii each the element in the biomolecule 
            atom_element (arr)      - Array of elements names(str) for atoms in biomolecule 
            scanPos (arr)           - Array of coordinates [x,y,z] of scan positions to image biomolecule and initial heights/ hard sphere boundary
            clipped_scanPos (arr)   - Array of clipped (containing only positions where tip and molecule interact) scan positions and 
                                      initial heights [x,y,z] to image biomolecule
            pdb (str)               - PDB (or CSV) file name of desired biomolecule
            
            kwargs: 
                        SaveImages (str)  - If Contour images to be saved include kwarg specifying the file path to folder
    '''
    # Set range of polar/ azimuthal angles for setting atoms surface positions  
    polar     = np.linspace(-np.pi, np.pi, 16)
    azimuthal = np.linspace(0, np.pi, 16)
    
    # Initialise count variable
    k=-1
    
    # Create array of all atom surface positions includung embedded
    surfacatomPos = np.zeros([ len(atom_coord)*len(polar)*len(azimuthal), 3 ])
    # For each atom, loop over polar angles and azimuthal angles
    for i, r in enumerate(atom_coord):
        for phi in polar:
            for theta in azimuthal:
                # Count array index
                k+=1
                
                # Unpack coordinates of atom centre and atom radius
                x0, y0, z0 = r
                R = atom_radius[atom_element[i]]
                
                # Calculate surface coordinate using spherical coordinates
                surfacatomPos[k,0] = x0 - R*np.cos(phi)*np.sin(theta)
                surfacatomPos[k,1] = y0 - R*np.sin(phi)*np.sin(theta)
                surfacatomPos[k,2] = z0 - R*np.cos(theta)
    
    
    # Initialise count variables           
    nB=0   
    k=-1
    
    # Create array of all atom surface positions above base
    clipped_surfacatomPos = np.zeros([ len(atom_coord)*len(polar)*len(azimuthal), 3 ])
    for i, r in enumerate(atom_coord):
        
        #  Set atom radius for atoms with surface above base 
        R = atom_radius[atom_element[i]]   
        if r[2] >= -R:
            # Count atom
            nB+=1
            # For each atom, loop over polar angles and azimuthal angles
            for phi in polar:
                for theta in azimuthal:
                    # Count array index
                    k+=1
                    
                    # Unpack coordinates of atom centre
                    x0, y0, z0 = r

                    # Calculate surface coordinate using spherical coordinates
                    clipped_surfacatomPos[k,0] = x0 - R*np.cos(phi)*np.sin(theta)
                    clipped_surfacatomPos[k,1] = y0 - R*np.sin(phi)*np.sin(theta)
                    clipped_surfacatomPos[k,2] = z0 - R*np.cos(theta)   
    
    # Return number of atoms and scan positions
    print('Number of Atoms in Molecuel:', nB)
    
    # Plot Surface incuding imbedded portin and all scan positions  
    fig1 = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(surfacatomPos[:,0], surfacatomPos[:,1], surfacatomPos[:,2], label = 'All Atom Surfaces')
    ax1.scatter3D(scanPos[:,0], scanPos[:,1], scanPos[:,2], label = 'All Scan Positons')
    ax1.set_xlabel(r'x (${\AA}$)')
    ax1.set_ylabel(r'y (${\AA}$)')
    ax1.set_zlabel(r'z (${\AA}$)')
    ax1.view_init(50, 145)    

    # Optionally save image
    if 'SaveImages' in kwargs.keys():
        fig1.savefig(kwargs['SaveImages'] + '\\AFMSimulationScanPos-'+pdb+'1.png', bbox_inches = 'tight', pad_inches=0.5) # change to backslash for mac/google colab
    
    plt.show()

    # Plot clipped surface and clipped scan positions
    fig2 = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.scatter3D(clipped_surfacatomPos[::8,0], clipped_surfacatomPos[::8,1], clipped_surfacatomPos[::8,2],  label = 'Clipped Atom Surfaces')
    ax2.scatter3D(clipped_scanPos[:,0], clipped_scanPos[:,1], clipped_scanPos[:,2],  label = 'Clipped Scan Positons')
    ax2.set_xlabel(r'x (${\AA}$)')
    ax2.set_ylabel(r'y (${\AA}$)')
    ax2.set_zlabel(r'z (${\AA}$)')
    ax2.view_init(90, 0)
    # Optionally save image
    if 'SaveImages' in kwargs.keys():
        fig2.savefig(kwargs['SaveImages'] + '\\AFMSimulationScanPos-'+pdb+'2.png', bbox_inches = 'tight', pad_inches=0.5) # change to backslash for mac/google colab
    plt.show()

# In[16]:
    
# ### Submission Functions

# In[17]:
# #### File Import/ Export 

# In[18]:

def ExportVariables(atom_coord, atom_element, atom_radius, clipped_scanPos, scanPos, variables, baseDims, tipDims, indentorType):
    ''' 
    Export simulation variables as csv and txt files to load in abaqus python scripts.
    
        Parameters:
            atom_coord (arr)      - Array of coordinates [x,y,z] for atoms in biomolecule 
            atom_element (arr)    - Array of elements names(str) for atoms in biomolecule 
            atom_radius (dict)    - Dictionary containing van der waals radii each the element in the biomolecule 
            clipped_scanPos (arr) - Array of clipped (containing only positions where tip and molecule interact) scan positions and 
                                    initial heights [x,y,z] to image biomolecule            
            scanPos (arr)         - Array of coordinates [x,y,z] of scan positions to image biomolecule and initial heights/ hard sphere boundary
            variables (list)      - List of simulation variables: [timePeriod, timeInterval, binSize, meshSurface, meshBase, meshIndentor, 
                                    indentionDepth, surfaceHeight]
            baseDims (arr)        - Geometric parameters for defining base/ substrate structure [width, height, depth] 
            tipDims (list)        - Geometric parameters for defining capped tip structure     
            indentorType (str)    - String defining indentor type (Spherical or Capped)
    '''
    
    np.savetxt("atom_coords.csv", atom_coord, delimiter=",")
    np.savetxt("atom_elements.csv", atom_element, fmt='%s', delimiter=",")

    np.savetxt("atom_radius_keys.csv", list(atom_radius.keys()), fmt='%s', delimiter=",")
    np.savetxt("atom_radius_values.csv", list(atom_radius.values()), delimiter=",")

    np.savetxt("clipped_scanPos.csv", clipped_scanPos, delimiter=",")
    np.savetxt("scanPos.csv", scanPos, fmt='%s', delimiter=",")

    np.savetxt("variables.csv", variables, fmt='%s', delimiter=",")
    np.savetxt("baseDims.csv", baseDims, fmt='%s', delimiter=",")
    np.savetxt("tipDims.csv", tipDims, fmt='%s', delimiter=",")

    with open('indentorType.txt', 'w', newline = '\n') as f:
        f.write(indentorType)


# In[19]:


def ImportVariables():
    ''' 
    Import simulation geometry variables from csv files.
    
        Return:
            atom_coord (arr)        - Array of coordinates [x,y,z] for atoms in biomolecule 
            atom_element (arr)      - Array of elements names(str) for atoms in biomolecule 
            atom_radius (dict)      - Dictionary containing van der waals radii each the element in the biomolecule 
            variables (list)        - List of simulation variables: [timePeriod, timeInterval, binSize, meshSurface, meshBase, meshIndentor, 
                                      indentionDepth, surfaceHeight]
            baseDims (arr)          - Geometric parameters for defining base/ substrate structure [width, height, depth]             
            scanPos (arr)           - Array of coordinates [x,y,z] of scan positions to image biomolecule and initial heights/ hard sphere boundary
            clipped_scanPos (arr)   - Array of clipped (containing only positions where tip and molecule interact) scan positions and 
                                      initial heights [x,y,z] to image biomolecule
    '''
    
    atom_coord      = np.loadtxt('atom_coords.csv', delimiter=",")
    atom_element    = np.loadtxt("atom_elements.csv", dtype = 'str', delimiter=",")

    keys            = np.loadtxt('atom_radius_keys.csv', dtype = 'str', delimiter=",")
    values          = np.loadtxt('atom_radius_values.csv', delimiter=",")
    atom_radius     = {keys[i]:values[i] for i in range(len(keys))}

    variables       = np.loadtxt('variables.csv', delimiter=",")
    baseDims        = np.loadtxt('baseDims.csv', delimiter=",")
    scanPos         = np.loadtxt('scanPos.csv', delimiter=",")
    clipped_scanPos = np.loadtxt('clipped_scanPos.csv', delimiter=",")

    return atom_coord, atom_element, atom_radius, variables, baseDims, scanPos, clipped_scanPos

# In[20]:
# #### Remote Functions
# Functions for working on remote serve, including transfering files, submitting bash commands, submiting bash scripts for batch input files and check queue statis.

# In[21]:
# ##### File Transfer`

# In[22]:

def RemoteSCPFiles(host, port, username, password, files, remotePath):
    '''    
    Function to make directory and transfer files to SSH server. A new Channel is opened and the files are transfered. 
    The command’s input and output streams are returned as Python file-like objects representing stdin, stdout, and stderr.
    
        Parameters:
            host (str)       - Hostname of the server to connect to
            port (int)       – Server port to connect to 
            username (str)   – username to authenticate as (defaults to the current local username)        -  
            password (str)   - password (str) – Used for password authentication; is also used for private key decryption if passphrase is not given.
            files (str/list) - File or list of file to transfer
            remotePath (str) - Path to remote file/directory
    '''
    # SHH to clusters
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(host, port, username, password)

    stdin, stdout, stderr = ssh_client.exec_command('mkdir -p ' + remotePath)

    # SCPCLient takes a paramiko transport as an argument- Uploading content to remote directory
    scp_client = SCPClient(ssh_client.get_transport())
    scp_client.put(files, recursive=True, remote_path = remotePath)
    scp_client.close()
    
    ssh_client.close()


# ##### Bash Command Submission

# In[23]:

def RemoteCommand(host, port, username, password, script, remotePath, command):
    '''
    Function to execute a command/ script submission on the SSH server. A new Channel is opened and the requested command is executed. 
    The command’s input and output streams are returned as Python file-like objects representing stdin, stdout, and stderr.
    
        Parameters:
            host (str)       - Hostname of the server to connect to
            port (int)       – Server port to connect to 
            username (str)   – username to authenticate as (defaults to the current local username)        -  
            password (str)   - password (str) – Used for password authentication; is also used for private key decryption if passphrase is not given.
            script (str)     - Script to run via bash command 
            remotePath (str) - Path to remote file/directory
            command (str)    - Abaqus command to execute and run script
    '''
    # SSH to clusters using paramiko module
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(host, port, username, password)
    
    # Execute command
    stdin, stdout, stderr = ssh_client.exec_command('cd ' + remotePath + ' \n '+ command +' '+ script +' & \n')
    lines = stdout.readlines()

    ssh_client.close()
    
    for line in lines:
        print(line)

# In[24]:
# ##### Batch File Submission

# In[25]:

def BatchSubmission(host, port, username, password, fileName, subData, scanPos, remotePath, **kwargs):
    ''' 
    Function to create bash script for batch submission of input file, and run them on remote server.
        Parameters:
            host (str)       - Hostname of the server to connect to
            port (int)       – Server port to connect to 
            username (str)   – username to authenticate as (defaults to the current local username)        -  
            password (str)   - password (str) – Used for password authentication; is also used for private key decryption if passphrase is not given.
            fileName (str)   - Base File name for abaqus input files
            subData (str)    - Data for submission to serve queue [walltime, memory, cpus]
            scanPos (arr)    - Array of coordinates [x,y] of scan positions to image biomolecule (can be clipped or full) 
            remotePath (str) - Path to remote file/directory
            
            kwargs:
                Submission ('serial'/ 'paralell') - optional define whether single serial script or seperate paralell submission to queue {Default: 'serial'}  
    '''
    # For paralell mode create bash script to runs for single scan location, then loop used to submit individual scripts for each location which run in paralell
    if 'Submission' in kwargs and kwargs['Submission'] == 'paralell':
        lines = ['#!/bin/bash -l',
                 '#$ -S /bin/bash',
                 '#$ -l h_rt='+ subData[0],
                 '#$ -l mem=' + subData[1],
                 '#$ -pe mpi ' + subData[2],
                 '#$ -wd /scratch/scratch/zcapjgi/ABAQUS',
                 'module load abaqus/2017 ',
                 'ABAQUS_PARALLELSCRATCH = "/scratch/scratch/zcapjgi/ABAQUS" ',
                 'cd ' + remotePath,
                 'gerun abaqus interactive cpus=$NSLOTS mp_mode=mpi job=$JOB_NAME input=$JOB_NAME.inp scratch=$ABAQUS_PARALLELSCRATCH resultsformat=odb'
                ]
        
    # Otherwise, create script to run serial analysis consecutively with single submission
    else:
        # Create set of submission comands for each scan locations
        jobs   = ['gerun abaqus interactive cpus=$NSLOTS mp_mode=mpi job='+fileName+str(int(i))+' input='+fileName+str(int(i))+'.inp scratch=$ABAQUS_PARALLELSCRATCH resultsformat=odb' 
                  for i in range(len(scanPos))]
        
        # Produce preamble to used to set up bash script
        lines = ['#!/bin/bash -l',
                    '#$ -S /bin/bash',
                    '#$ -l h_rt='+ subData[0],
                    '#$ -l mem=' + subData[1],
                    '#$ -pe mpi ' + subData[2],
                    '#$ -wd /home/zcapjgi/Scratch/ABAQUS',
                    'module load abaqus/2017 ',
                    'ABAQUS_PARALLELSCRATCH = "/home/zcapjgi/Scratch/ABAQUS" ',
                    'cd ' + remotePath ]
        # Combine to produce total  script
        lines+=jobs

    # Create script file in current directory by writing each line to file
    with open('batchScript.sh', 'w', newline = '\n') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    # SSH to clusters 
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(host, port, username, password)

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

def QueueCompletion(host, port, username, password):
    '''
    Function to check queue statis and complete when queue is empty.
        Parameters:
            host (str)       - Hostname of the server to connect to
            port (int)       – Server port to connect to 
            username (str)   – username to authenticate as (defaults to the current local username)        -  
            password (str)   - password (str) – Used for password authentication; is also used for private key decryption if passphrase is not given.
    '''
    # Log time
    t0 = time.time()
    complete= False

    while complete == False:
        # SSH to clusters 
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(host, port, username, password)

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

def RemoteFTPFiles(host, port, username, password, files, remotePath, localPath):
    ''' 
    Function to transfer files from directory on SSH server to local machine. A new Channel is opened and the files are transfered. 
    The function uses FTP file transfer.
    
        Parameters:
            host (str)       - Hostname of the server to connect to
            port (int)       – Server port to connect to 
            username (str)   – username to authenticate as (defaults to the current local username)        -  
            password (str)   - password (str) – Used for password authentication; is also used for private key decryption if passphrase is not given.
            files (str )     - File to transfer
            remotePath (str) - Path to remote file/directory
            localPath (str)  - Path to local file/directory
    '''
    # SSH to cluster
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(host, port, username, password)

    # FTPCLient takes a paramiko transport as an argument- copy content from remote directory
    ftp_client=ssh_client.open_sftp()
    ftp_client.get(remotePath+'/'+files, localPath +'\\'+ files)  
    ftp_client.close()


# In[30]:
# ##### Remote Terminal

# In[31]:

def Remote_Terminal(host, port, username, password):
    '''    
    Function to emulate cluster terminal. Channel is opened and commands given are executed. The command’s input 
    and output streams are returned as Python file-like objects representing stdin, stdout, and stderr.
    
        Parameters:
            host (str)       - Hostname of the server to connect to
            port (int)       – Server port to connect to 
            username (str)   – username to authenticate as (defaults to the current local username)        -  
            password (str)   - password (str) – Used for password authentication; is also used for private key decryption 
                               if passphrase is not given.
    '''
    
    # SHH to cluster
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(host, port, username, password)
    
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

# In[32]:
# #### Remote/ Local Submission 
# Function to run simulation and scripts on the remote servers. Files for variables are transfered, ABAQUS scripts are run to create parts and input files. A bash file is created and submitted to run simulation for batch of inputs. Analysis of odb files is performed and data transfered back to local machine. Using keyword arguments invidual parts of simulation previously completed can be skipped.

# In[33]:

def LocalSubmission():
    ''' Submit Abaqus scripts locally'''
    get_ipython().system('abaqus fetch job=AFMSurfaceModel')
    get_ipython().system('abaqus cae -noGUI AFMSurfaceModel.py')

    get_ipython().system('abaqus fetch job=AFMRasterScan')
    get_ipython().system('abaqus cae -noGUI AFMRasterScan.py')

    get_ipython().system('abaqus fetch job=AFMODBAnalysis')
    get_ipython().system('abaqus cae -noGUI AFMODBAnalysis.py')


# In[34]:

def RemoteSubmission(host, port, username, password, remotePath, localPath,  csvfiles, abqfiles, abqCommand, fileName, subData, clipped_scanPos, **kwargs):
    '''
    Function to run simulation and scripts on the remote servers. Files for variables are transfered, ABAQUS scripts are run to create parts and input files. 
    A bash file is created and submitted to run simulation for batch of inputs. Analysis of odb files is performed and data transfered back to local machine.
    Using keyword arguments invidual parts of simulation previously completed can be skipped.
    
        Parameters:
            host (str)              - Hostname of the server to connect to
            port (int)              – Server port to connect to 
            username (str)          – Username to authenticate as (defaults to the current local username)        -  
            password (str)          - password (str) – Used for password authentication; is also used for private key decryption if passphrase is not given.
            remotePath (str)        - Path to remote file/directory
            localPath (str)         - Path to local file/directory
            csvfiles (list)         - List of csv and txt files to transfer to remote server
            abqfiles (list)         - List of abaqus script files to transfer to remote server
            abqCommand (str)        - Abaqus command to execute and run script
            fileName (str)          - Base File name for abaqus input files
            subData (str)           - Data for submission to serve queue [walltime, memory, cpus]
            clipped_scanPos (arr)   - Array of clipped (containing only positions where tip and molecule interact) scan positions and  initial heights [x,y,z] 
                                      to image biomolecule    
            kwargs:
                submission ('serial'/ 'paralell') - Type of submission, submit pararlell scripts or single serial script for scan locations {Default: 'serial'}
                Transfer (bool)                   - If false skip file transfer step of simulation {Default: True}
                Part (bool)                       - If false skip part creation step of simulation {Default: True}
                Input (bool)                      - If false skip input file creation step of simulation {Default: True}
                Batch (bool)                      - If false skip batch submission step of simulation {Default: True}
                Queue (bool)                      - If false skip queue completion step of simulation {Default: True}
                Analysis (bool)                   - If false skip odb analysis step of simulation {Default: True}
                Retrieval (bool)                  - If false skip data file retrivial from remote serve {Default: True}
    '''
    #  ---------------------------------------------File Transfer----------------------------------------------------------
    if 'Transfer' not in kwargs.keys() or kwargs['Transfer'] == True:
        
        # Transfer scripts and variable files to remote server
        RemoteSCPFiles(host, port, username, password, csvfiles, remotePath)
        RemoteSCPFiles(host, port, username, password, abqfiles, remotePath)
        
        print('File Transfer Complete')

    #  ----------------------------------------------Input File Creation----------------------------------------------------
    if 'Part' not in kwargs.keys() or kwargs['Part'] == True:
        t0 = time.time()
        print('Creating Parts ...')
        
        # Create Molecule and Tip
        script = 'AFMSurfaceModel.py'
        RemoteCommand(host, port, username, password, script, remotePath, abqCommand)
        
        t1 = time.time()
        print('Part Creation Complete - ' + str(timedelta(seconds=t1-t0)) )
    
    if 'Input' not in kwargs.keys() or kwargs['Input'] == True:
        t0 = time.time()
        print('Producing Input Files ...')
        
        # Produce simulation and input files
        script = 'AFMRasterScan.py'
        RemoteCommand(host, port, username, password, script, remotePath, abqCommand)
        
        t1 = time.time()
        print('Input File Complete - ' + str(timedelta(seconds=t1-t0)) )

    #  --------------------------------------------Batch File Submission----------------------------------------------------
    if 'Batch' not in kwargs.keys() or kwargs['Batch'] == True:
        t0 = time.time()
        print('Submitting Batch Scripts ...')
        
        # Submit bash scripts to remote queue to carry out batch abaqus analysis
        BatchSubmission(host, port, username, password, fileName, subData, clipped_scanPos, remotePath, **kwargs) 
        
        t1 = time.time()
        print('Batch Submission Complete - '+ str(timedelta(seconds=t1-t0)) )
    
    if 'Queue' not in kwargs.keys() or kwargs['Queue'] == True:
        t0 = time.time()
        print('Simulations Processing ...')
        
        # Wait for completion when queue is empty
        QueueCompletion(host, port, username, password)
        
        t1 = time.time()
        print('ABAQUS Simulation Complete - '+ str(timedelta(seconds=t1-t0)) )

    #  -------------------------------------------ODB Analysis Submission----------------------------------------------------
    if 'Analysis' not in kwargs.keys() or kwargs['Analysis'] == True:
        t0 = time.time()
        print('Running ODB Analysis...')
        
        # ODB analysis script to run, extracts data from simulation and sets it in csv file on server
        script = 'AFMODBAnalysis.py'
        RemoteCommand(host, port, username, password, script, remotePath, abqCommand)
        
        t1 = time.time()
        print('ODB Analysis Complete - ' + str(timedelta(seconds=t1-t0)) )

    #  -----------------------------------------------File Retrieval----------------------------------------------------------
    if 'Retrieval' not in kwargs.keys() or kwargs['Retrieval'] == True:
        t0 = time.time()
        # Retrieve variables used for given simulation (in case variables redefined when skip kwargs used) 
        csvfiles = ("clipped_scanPos.csv", "scanPos.csv","variables.csv","baseDims.csv", "tipDims.csv")
        dataFiles = ('U2_Results.csv','RF_Results.csv','ErrorMask.csv')
        
        # Files retrievals from remote server
        for file in csvfiles:
            RemoteFTPFiles(host, port, username, password, file, remotePath, localPath)
        RemoteFTPFiles(host, port, username, password, dataFiles[0], remotePath, localPath)
        RemoteFTPFiles(host, port, username, password, dataFiles[1], remotePath, localPath)
        RemoteFTPFiles(host, port, username, password, dataFiles[2], remotePath, localPath)

            
        t1 = time.time()
        print('File Retrevial Complete')

# In[35]:
# ### Post-Processing Functions
# Function for postprocessing ABAQUS simulation data, loading variables from files in current directory and process data from simulation in U2/RF files. Process data from clipped scan positions to include full data range over all scan positions. Alongside, function to plot and visualise data. Then, calculates contours/z heights of constant force in simulation data for given threshold force and visualise.

# In[36]:
# #### Data Processing 
# Function to load variables from fil~es in current directory and process data from simulation in U2/RF files. Process data from clipped scanpositions to include full data range over all scan positions. Alongside, function to plot and visualise data.

# In[37]:

def DataProcessing(clipped_RF, clipped_U2, scanPos, clipped_scanPos, clipped_ErrorMask, indentionDepth, timePeriod, timeInterval):   
    '''
    Function to load variables from files in current directory and process data from simulation in U2/RF files. Process data from clipped scan positions
    to include full data range over all scan positions.
        Parameters:
            clipped_RF              - Array of indentors z displacement over clipped scan position
            clipped_U2              - Array of reaction force on indentor reference point over clipped scan positions
            scanPos (arr)           - Array of coordinates [x,y,z] of scan positions to image biomolecule and initial heights/ hard sphere boundary
            clipped_scanPos (arr)   - Array of clipped (containing only positions where tip and molecule interact) scan positions and 
                                      initial heights [x,y,z] to image biomolecule
            clipped_ErrorMask (arr) - Boolean array specifying mask for clipped scan positions which errored in ABAQUS
            indentionDepth (float)  - Maximum indentation depth into surface 
            timePeriod(float)       - Total time length for ABAQUS simulation/ time step (T)
            timeInterval(float)     - Time steps data sampled over for ABAQUS simulation/ time step (dt)
            
        Return:
            U2 (arr)        - Array of indentors z displacement over scan position
            RF (arr)        - Array of reaction force on indentor reference point
            ErrorMask (arr) - Boolean array specifying mask for all scan positions which errored in ABAQUS
            N (int)         - Number of frames in ABAQUS simulation/ time step  
    '''
    
    # Set number if frames in ABAQUS simulation step -  N = T/dt + 1 for intial frame
    N = int(timePeriod/ timeInterval)+1
    
    # Initialise reaction force RF and z indentation depth U2
    RF = np.zeros([len(scanPos),N])
    U2 = np.zeros([len(scanPos),N])
    ErrorMask = np.zeros([len(scanPos)])


    # Loop over scan positions and clipped scanPos positions
    for i in range(len(scanPos)):
        for j in range(len(clipped_scanPos)):
            
            # If scan position is in clipped set points extract corresponding simulation for position
            if scanPos[i,0]==clipped_scanPos[j,0] and  scanPos[i,1]==clipped_scanPos[j,1] and scanPos[i,2]==clipped_scanPos[j,2]:
                RF[i] = abs(clipped_RF[j])
                U2[i] = clipped_U2[j]
                ErrorMask[i] = clipped_ErrorMask[j]
            
            # Otherwise indentor does not contact molecules surface and force left as zero for linear indentor displacement
            else:
                U2[i] = np.linspace(0,-indentionDepth,N)
                
    return U2, RF, ErrorMask, N


# In[38]:

def DataPlot(scanPos, U2, RF, N):
    ''' 
    Produces scatter plot of indentation depth and reaction force to visualise and check simulation data.
    
        Parameters:
            scanPos (arr) - Array of coordinates [x,y] of scan positions to image biomolecule 
            U2 (arr)      - Array of indentors z displacement over scan position
            RF (arr)      - Array of reaction force on indentor reference point
            N (int)       - Number of frames in  ABAQUS simulation/ time step 
    '''
    
    # Initialise array for indentor force and displacement
    tipPos   = np.zeros([len(scanPos)*N,3])
    tipForce = np.zeros(len(scanPos)*N)

    # print(scanPos.shape)
    # print(RF.shape, U2.shape)
    # print(tipPos.shape, tipForce.shape)

    # Initialise count
    k = 0
    
    # Loop over array indices
    for i in range(len(scanPos)):
        for j in range( N ):
            #  Set array values for tip force and displacement 
            tipPos[k]   = [scanPos[i,0], scanPos[i,1] , U2[i,j]] 
            tipForce[k] = abs(RF[i,j])
            
            # Count array index
            k+=1

    # Scatter plot indentor displacement over scan positions
    fig1 = plt.figure()
    ax1 = plt.axes(projection='3d')

    ax1.scatter3D(tipPos[:,0], tipPos[:,1], tipPos[:,2])

    ax1.set_xlabel(r'x coordinate/ Length (nm)')
    ax1.set_ylabel(r'y coordinate/ Width(nm)')
    ax1.set_zlabel(r'z coordinate/ Height (nm)')
    ax1.set_title('Tip Position for Raster Scan')
    plt.show()

    # Scatter plot of force over scan positions
    fig2 = plt.figure()
    ax2 = plt.axes(projection='3d')

    ax2.scatter3D(tipPos[:,0], tipPos[:,1], tipForce)

    ax2.set_xlabel(r'x coordinate/ Length (nm)')
    ax2.set_ylabel(r'y coordinate/ Width(nm)')
    ax2.set_zlabel('Force N')
    ax2.set_title('Force Scatter Plot for Raster Scan')
    ax2.view_init(50, 35)
    plt.show()

# In[39]:
# #### AFM Image Functions
# Calculate contours/z heights of constant force in simulation data for given threshold force and visualise.

# In[40]:

def ForceContours(U2, RF,forceRef, scanPos, baseDims, binSize):
    ''' 
    Function to calculate contours/z heights of constant force in simulation data for given threshold force.
    
        Parameters:
            U2 (arr)         - Array of indentors z displacement over scan position
            RF (arr)         - Array of reaction force on indentor reference point
            forceRef (float) - Threshold force to evaluate indentation contours at (pN)
            scanPos (arr)    - Array of coordinates [x,y,z] of scan positions to image biomolecule 
            baseDims (arr)   - Geometric parameters for defining base/ substrate structure [width, height, depth]           
            binSize (float)  - Width of bins that subdivid xy domain during raster scanning/ spacing of the positions sampled over
            
        Return:
            X (arr) - 2D array of x coordinates over grid positions 
            Y (arr) - 2D array of y coordinates over grid positions 
            Z (arr) - 2D array of z coordinates of force contour over grid positions  
    '''
    
    # Initialise dimensional variables
    xNum = int(baseDims[0]/binSize)+1
    yNum = int(baseDims[1]/binSize)+1
    
    # Initialise contour array
    forceContour = np.zeros(len(RF))

    # Loop over each reaction force array, i.e. each scan positions
    for i in range(len(RF)):
        
        # If maximum for at this position is greater than Reference force
        if np.max(RF[i]) > forceRef:
            # Return index of force threshold and store related depth
            j = [ k for k,v in enumerate(RF[i]) if v > forceRef][0]
            
            # Set surface height for reference height
            forceContour[i] = scanPos[i,2] + U2[i,j] 
        
        # If no value above freshold set value at bottom height
        else: 
            forceContour[i] = scanPos[i,2] + U2[i,-1]
            
    # Format x,y,z position for force contour       
    X  = scanPos.reshape(yNum, xNum, 3)[:,:,0]
    Y  = scanPos.reshape(yNum, xNum, 3)[:,:,1]
    Z  = forceContour.reshape(yNum, xNum)  
    
    return X, Y, Z 


# In[41]:

def ContourPlot(X, Y, Z, ErrorMask, baseDims, binSize, forceRef, contrast, pdb, **kwargs):
    ''' 
    Function to plot force contor produced from simulation. Plots 3D wire frame image and a 2D AFM image.
    
        Parameters:          
            X (arr)          - 2D array of x coordinates over grid positions 
            Y (arr)          - 2D array of y coordinates over grid positions 
            Z (arr)          - 2D array of z coordinates of force contour over grid positions 
            ErrorMask (arr)  - Boolean array specifying mask for all scan positions which errored in ABAQUS
            baseDims (arr)   - Geometric parameters for defining base/ substrate structure [width, height, depth]
            binSize (float)  - Width of bins that subdivid xy domain during raster scanning/ spacing of the positions sampled over
            forceRef (float) - Threshold force to evaluate indentation contours at (pN)
            contrast (float) - Contrast between high and low values in AFM heat map (0-1)
            pdb (str)        - PDB (or CSV) file name of desired biomolecule
            
        kwargs:
            Noise (list)         - If listed adds noise to AFM images [strength, mean, standard deviation]
            ImagePadding (float) - Black space / padding around image as percentage of dimensions of molecule extent
            SaveImages (str)     - If Contour images to be saved include kwarg specifying the file path to folder
    '''    
    # ------------------------------------Add noise and padding to image if in kwargs--------------------------------   
    # Current data shape
    xNum,  yNum  = int(baseDims[0]/binSize)+1,  int(baseDims[1]/binSize)+1
    
    if 'ImagePadding' in kwargs.keys():
        imagePadding = kwargs['ImagePadding']
        
        padDims      = imagePadding*baseDims
        xPad,  yPad  = int(padDims[0]/binSize)+1, int(padDims[1]/binSize)+1 
        xDiff, yDiff = abs((xPad-xNum)/2), abs((yPad-yNum)/2)
        
        X, Y = np.meshgrid(np.linspace(-padDims[0]/2, padDims[0]/2, xPad),
                           np.linspace(-padDims[1]/2, padDims[1]/2, yPad))   
        
        if imagePadding >= 1:            
            Z = np.pad(Z, pad_width= ( (round(yDiff-0.25), round(yDiff+0.25)), (round(xDiff-0.25), round(xDiff+0.25)) ), mode='constant')
        else:
            Z = np.delete(Z, np.arange(-round(yDiff-0.25), round(yDiff+0.25)), axis = 0)
            Z = np.delete(Z, np.arange(-round(xDiff-0.25), round(xDiff+0.25)), axis = 1)
            
        imageDims = padDims
        
    else:
        imageDims = baseDims
        
    if 'Noise' in kwargs.keys():
        noise_strength, noise_mean, noise_variance = kwargs['Noise'] 
        noise = noise_strength*np.random.normal(noise_mean, noise_variance, [Z.shape[0], Z.shape[1]])
        Z+=noise      
    else:
        None      
        
    # Reshape image mask and apply to data
    X = np.ma.masked_array(X, mask = ErrorMask.reshape(yNum, xNum) )
    Y = np.ma.masked_array(Y, mask = ErrorMask.reshape(yNum, xNum) )
    Z = np.ma.masked_array(Z, mask = ErrorMask.reshape(yNum, xNum) )
    
    #  -------------------------------------------------3D Plots-----------------------------------------------------      
    # Plot 3D Contour Plot
    fig = plt.figure()
    ax = plt.axes(projection = "3d") 
    ax.contour3D(X,Y, Z, 30, cmap='afmhot')
    # ax.plot_wireframe(X,Y, Z)
    ax.plot_surface(X,Y, Z, cmap='afmhot')

    ax.set_xlabel(r'x coordinate/ Length (${\AA}$)')
    ax.set_ylabel(r'y coordinate/ Width(${\AA}$)')
    ax.set_zlabel(r'z coordinate/ Height (${\AA}$)')
    ax.set_title('Contour Plot for Force of {0}pN'.format(forceRef))
    ax.view_init(60, 35)
    # ax.view_init(90, 0)
    plt.show()
    
    #  -------------------------------------------------2D Plots-----------------------------------------------------      
    # 2D heat map/ contour plot with interpolation
    fig, ax = plt.subplots(1, 2) 
    im = ax[0].imshow(Z, origin= 'lower', cmap='afmhot', interpolation='bicubic',vmin=0, vmax= (contrast)*Z.compressed().max(initial = 1e-10), 
                      extent=(-imageDims[0]/2, imageDims[0]/2,-imageDims[1]/2, imageDims[1]/2), interpolation_stage = 'rgba' )
    ax[0].set_xlabel(r'x (${\AA}$)')
    ax[0].set_ylabel(r'y (${\AA}$)')
    ax[0].axes.set_aspect('equal')
    ax[0].set_facecolor("grey")
    
    # 2D heat map/ contour plot without interpolation
    im = ax[1].imshow(Z, origin= 'lower', cmap='afmhot',vmin=0, vmax= (contrast)*Z.compressed().max(initial = 1e-10),
                      extent=(-imageDims[0]/2, imageDims[0]/2, -imageDims[1]/2, imageDims[1]/2), interpolation_stage = 'rgba'  )
    ax[1].set_xlabel(r'x (${\AA}$)')
    ax[1].set_ylabel(r'y (${\AA}$)')
    ax[1].axes.set_aspect('equal')
    ax[1].set_facecolor("grey")
    
    plt.subplots_adjust(wspace = 0.5)
    cbar= fig.colorbar(im, ax= ax.ravel().tolist(), orientation='horizontal')
    cbar.set_label(r'z (${\AA}$)')
    
    # Optionally save image
    if 'SaveImages' in kwargs.keys():
        fig.savefig(kwargs['SaveImages'] + '\\AFMSimulationMolecule-'+pdb+'.png', bbox_inches = 'tight') # change to backslash for mac/google colab
    
    plt.show()


# In[42]:

def HardSphereAFM(scanPos, baseDims, binSize, clearance, contrast,  pdb, **kwargs):
    ''' 
    Plot the molecules atoms surfaces and scan positions to visualise and check positions.
    
        Parameters:
            scanPos (arr)      - Array of coordinates [x,y,z] of scan positions to image biomolecule and initial heights/ hard sphere boundary
            baseDims (arr)     - Geometric parameters for defining base/ substrate structure [width, height, depth] 
            binSize (float)    - Width of bins that subdivid xy domain during raster scanning/ spacing of the positions sampled over
            clearance (float)  - Clearance above molecules surface indentor is set to during scan
            contrast (float)   - Contrast between high and low values in AFM heat map (0-1)  
            pdb (str)          - PDB (or CSV) file name of desired biomolecule
       
        kwargs:
            Noise (list)         - If listed adds noise to AFM images [strength, mean, standard deviation]
            ImagePadding (float) - Black space / padding around image as percentage of dimensions of molecule extent
            SaveImages (str)     - If Contour images to be saved include kwarg specifying the file path to folder
    '''     
    #  ------------------------------------------------------------------------------------------------------------        
    # Initialise dimensional variables
    xNum, yNum = int(baseDims[0]/binSize)+1, int(baseDims[1]/binSize)+1
    X, Y, Z = scanPos.reshape(yNum, xNum, 3)[:,:,0], scanPos.reshape(yNum, xNum, 3)[:,:,1], scanPos.reshape(yNum, xNum, 3)[:,:,2] - clearance
    
    
    # ----------------------------------Add noise and padding to image if in kwargs--------------------------------     
    if 'ImagePadding' in kwargs.keys():
        imagePadding = kwargs['ImagePadding']
        
        padDims      = imagePadding*baseDims
        xNum,  yNum  = int(baseDims[0]/binSize)+1,  int(baseDims[1]/binSize)+1
        xPad,  yPad  = int(padDims[0]/binSize)+1, int(padDims[1]/binSize)+1 
        xDiff, yDiff = abs((xPad-xNum)/2), abs((yPad-yNum)/2)
        
        X, Y = np.meshgrid(np.linspace(-padDims[0]/2, padDims[0]/2, xPad),
                           np.linspace(-padDims[1]/2, padDims[1]/2, yPad))   
        
        if imagePadding >= 1:            
            Z = np.pad(Z, pad_width= ( (round(yDiff-0.25), round(yDiff+0.25)), (round(xDiff-0.25), round(xDiff+0.25)) ), mode='constant')
        else:
            Z = np.delete(Z, np.arange(-round(yDiff-0.25), round(yDiff+0.25)), axis = 0)
            Z = np.delete(Z, np.arange(-round(xDiff-0.25), round(xDiff+0.25)), axis = 1)
            
        imageDims = padDims
        
    else:
        imageDims = baseDims
        
    if 'Noise' in kwargs.keys():
        noise_strength, noise_mean, noise_variance = kwargs['Noise'] 
        noise = noise_strength*np.random.normal(noise_mean, noise_variance, [Z.shape[0], Z.shape[1]])
        Z+=noise      
    else:
        None      
            
    #  -------------------------------------------------3D Plots-----------------------------------------------------      
    # Plot 3D Contour Plot
    fig = plt.figure()
    ax = plt.axes(projection = "3d") 
    ax.contour3D(X, Y, Z, 30, cmap='afmhot')
    # ax.plot_wireframe(X,Y, Z)
    ax.plot_surface(X,Y, Z, cmap='afmhot')

    ax.set_xlabel(r'x coordinate/ Length (${\AA}$)')
    ax.set_ylabel(r'y coordinate/ Width(${\AA}$)')
    ax.set_zlabel(r'z coordinate/ Height (${\AA}$)')
    ax.set_title('Contour Plot for Force of {0}pN'.format(0))
    ax.view_init(60, 35)
    # ax.view_init(90, 0)
    plt.show()
    

    #  -------------------------------------------------2D Plots-----------------------------------------------------      
    # 2D heat map/ contour plot with interpolation
    fig, ax = plt.subplots(1, 2) 
    im = ax[0].imshow(Z, origin= 'lower', cmap='afmhot', interpolation='bicubic',vmin=0,  vmax= contrast*Z.max(initial = 1e-10)   ,
                      extent=(-imageDims[0]/2,imageDims[0]/2,-imageDims[1]/2,imageDims[1]/2)  )
    ax[0].set_xlabel(r'x (${\AA}$)')
    ax[0].set_ylabel(r'y (${\AA}$)')
    ax[0].axes.set_aspect('equal')
    
    # 2D heat map/ contour plot without interpolation
    im = ax[1].imshow(Z, origin= 'lower', cmap='afmhot',vmin=0,  vmax= contrast*Z.max(initial = 1e-10),   
                      extent=(-imageDims[0]/2,imageDims[0]/2,-imageDims[1]/2,imageDims[1]/2)  )
    ax[1].set_xlabel(r'x (${\AA}$)')
    ax[1].set_ylabel(r'y (${\AA}$)')
    ax[1].axes.set_aspect('equal')
    
    plt.subplots_adjust(wspace = 0.5)
    cbar= fig.colorbar(im, ax= ax.ravel().tolist(),  orientation='horizontal')
    cbar.set_label(r'z (${\AA}$)')
    
    # Optionally save image
    if 'SaveImages' in kwargs.keys():
        fig.savefig(kwargs['SaveImages'] + '\\AFMSimulationMolecule-'+pdb+'.png', bbox_inches = 'tight') # change to backslash for mac/google colab
        
    plt.show()

# In[43]:
# ## Simulation Script
# Final simulation function

# In[44]:

def AFMSimulation(host, port, username, password, remotePath, localPath, abqCommand, fileName, subData, 
                  pdb, rotation, surfaceApprox, indentorType, rIndentor, theta_degrees, tip_length, 
                  indentionDepth, forceRef, contrast, binSize, clearance, meshSurface, meshBase, meshIndentor, 
                  timePeriod, timeInterval, **kwargs):
    '''
    Final function to automate simulation. User inputs all variables and all results are outputted. The user gets a optionally get a surface plot of scan positions.
    Produces a heatmap of the AFM image, and 3D plots of the sample surface for given force threshold.
    
        Parameters:
            host (str)             - Hostname of the server to connect to
            port (int)             - Server port to connect to 
            username (str)         - Username to authenticate as (defaults to the current local username)        -  
            password (str)         - password (str) – Used for password authentication; is also used for private key decryption if passphrase is not given.
            remotePath (str)       - Path to remote file/directory
            localPath (str)        - Path to local file/directory
            abqCommand (str)       - Abaqus command to execute and run script
            fileName (str)         - Base File name for abaqus input files
            subData (list)         - Data for submission to serve queue [walltime, memory, cpus]
            pdb (str)              - PDB (or CSV) file name of desired biomolecule
            rotation (list)        - Array of [xtheta, ytheta, ztheta] rotational angle around coordinate axis'
            surfaceApprox (float)  - Percentage of biomolecule assumed to be not imbedded in base/ substrate. Range: 0-1 
            indentorType (str)     - String defining indentor type (Spherical or Capped)
            rIndentor (float)      - Radius of spherical tip portion
            theta_degrees (float)  - Principle conical angle from z axis in degrees
            tip_length (float)     - Total cone height
            indentionDepth (float) - Maximum indentation depth into surface 
            forceRef (float)       - Threshold force to evaluate indentation contours at, mimics feedback force in AFM (pN)
            contrast (float)       - Contrast between high and low values in AFM heat map (0-1)
            binSize(float)         - Width of bins that subdivid xy domain during raster scanning/ spacing of the positions sampled over
            clearance(type:float)  - Clearance above molecules surface indentor is set to during scan
            meshSurface (float)    - Value of indentor mesh given as bin size for vertices of geometry in Angstrom (x10-10 m)
            meshBase (float)       - Value of indentor mesh given as bin size for vertices of geometry in Angstrom (x10-10 m)
            meshIndentor (float)   - Value of indentor mesh given as bin size for vertices of geometry in Angstrom (x10-10 m) 
            timePeriod(float)      - Total time length for ABAQUS simulation/ time step (T)
            timeInterval(float)    - Time steps data sampled over for ABAQUS simulation/ time step (dt)
            
            kwargs:
                Submission ('serial'/ 'paralell') - Type of submission, submit pararlell scripts or single serial script for scan locations {Default: 'serial'}
                CustomPDB - Extract data from local custom pd as opposed to from PDB online
                
                Preprocess (bool)  - If false skip preprocessing step of simulation {Default: True}
                DotPlot (bool)     - If false skip surface plot of biomolecule and scan positions {Default: False}
                HSPlot (bool)      - If false skip Hard Sphere AFM plot of biomolecule {Default: False}
                MoleculeView(bool) - If false skip interactive sphere model of biomolecule {Default: False}
                
                Transfer (bool)    - If false skip file transfer step of simulation {Default: True}
                Part (bool)        - If false skip part creation step of simulation {Default: True}
                Input (bool)       - If false skip input file creation step of simulation {Default: True}
                Batch (bool)       - If false skip batch submission step of simulation {Default: True}
                Queue (bool)       - If false skip queue completion step of simulation {Default: True}
                Analysis (bool)    - If false skip odb analysis step of simulation {Default: True}
                Retrieval (bool)   - If false skip data file retrivial from remote serve {Default: True}
                
                Postprocess (bool) - If false skip postprocessing step to produce AFM image from data {Default: True}
                DataPlot (bool)    - If false skip scatter plot of simulation data {Default: True} 
                ReturnData (bool)  - If true returns simulation data to analysis {Default: False} 
                
                Noise (list)         - If listed adds noise to AFM images [strength, mean, standard deviation]
                imagePadding (float) - Black space / padding around image as percentage of dimensions of molecule extent
                SaveImages (str)     - If Contour images to be saved include kwarg specifying the file path to folder
                
    '''
    T0 = time.time()
    
    #  -------------------------------------------------Pre-Processing-----------------------------------------------------  
    if 'Preprocess' not in kwargs.keys() or kwargs['Preprocess'] == True:
        t0 = time.time()
        
        # Extract tip geometry and molecule structure
        structure, view = PDB(pdb, localPath, **kwargs) 
        tipDims = TipStructure(rIndentor, theta_degrees, tip_length)
                
        # Produce array of atoom coordinates, element, radius and dimension of base/substrate and calculate scan positions over molecule for imaging
        atom_coord, atom_element, atom_radius, surfaceHeight, baseDims = MolecularStructure(structure, rotation, tipDims, indentorType, binSize, surfaceApprox)        
        scanPos, clipped_scanPos = ScanGeometry(atom_coord, atom_radius, atom_element, indentorType, tipDims, baseDims, surfaceHeight, binSize, clearance)
    
        # Set list of simulation variables and export to current directory
        variables = [timePeriod, timeInterval, binSize, meshSurface, meshBase, meshIndentor, indentionDepth, surfaceHeight]
        ExportVariables(atom_coord, atom_element, atom_radius, clipped_scanPos, scanPos, variables, baseDims, tipDims, indentorType)

        # Set return variables as None if postprocessing not run but want to return scan positons
        X, Y, Z, U2, RF = None, None, None, None, None
        
        t1 = time.time()

        # Print data
        print('Preprocessing Complete - ' + str(timedelta(seconds=t1-t0)) )     
        print('Number of Scan Positions:', len(clipped_scanPos))
        
        
        #  -------------------------------------------------Plot data-----------------------------------------------------          
        if 'HSPlot' in kwargs.keys() and kwargs['HSPlot'] == True:        
            HardSphereAFM(scanPos, baseDims, binSize, clearance, contrast, pdb, **kwargs)
            
        # Option plot for surface visualisation
        if 'DotPlot' in kwargs.keys() and kwargs['DotPlot'] == True:
            DotPlot(atom_coord, atom_radius, atom_element, scanPos, clipped_scanPos, **kwargs)

        # Interactive molecular view
        if 'MolecularView' in kwargs.keys() and kwargs['MolecularView'] == True:
            molecular_view = nv.show_biopython(structure)        
            molecular_view.control.spin([1,0,0],rotation[0])
            molecular_view.control.spin([0,1,0],rotation[1])
            molecular_view.control.spin([0,0,1],rotation[2])
            molecular_view.add_representation('spacefill', selection='all')
            return(molecular_view)  
            
    # Condition to skip preprocessing step if files already generated previously
    else:
        # Check if simulation files are accessible in curent directory to use if pre=processing skipped
        try:
            atom_coord, atom_element, atom_radius, variables, baseDims, scanPos, clipped_scanPos     = ImportVariables()
            timePeriod, timeInterval, binSize, meshSurface, meshBase, meshIndentor, indentionDepth, surfaceHeight = variables
            
        # If file missing prompt user to import/ produce files 
        except:
            print('No Simulation files available, run preprocessing or import data')
              

    #  ----------------------------------------------------Remote Simulation-------------------------------------------------------  
    if 'Submission' not in kwargs.keys() or kwargs['Submission'] != False:         
        # SSH to remote cluster to perform ABAQUS simulation and analysis from scripts and data files 
        csvfiles = ("atom_coords.csv","atom_elements.csv","atom_radius_keys.csv", "atom_radius_values.csv", 
                    "clipped_scanPos.csv", "scanPos.csv","variables.csv","baseDims.csv", "tipDims.csv", "indentorType.txt")
        abqfiles = ('AFMSurfaceModel.py', 'AFMRasterScan.py', 'AFMODBAnalysis.py')
    
        RemoteSubmission(host, port, username, password, remotePath, localPath, csvfiles, abqfiles, abqCommand, fileName, subData, clipped_scanPos, **kwargs)  
        
        
    #  -------------------------------------------------- Post-Processing--------------------------------------------------------  
    if 'Postprocess' not in kwargs.keys() or kwargs['Postprocess'] == True:
        
        # Check if all simulation files are accessible in curent directory for post-processing
        try:
            atom_coord, atom_element, atom_radius, variables, baseDims, scanPos, clipped_scanPos                  = ImportVariables()
            timePeriod, timeInterval, binSize, meshSurface, meshBase, meshIndentor, indentionDepth, surfaceHeight = variables
            clipped_U2 = np.array(np.loadtxt('U2_Results.csv', delimiter=","))
            clipped_RF = np.array(np.loadtxt('RF_Results.csv', delimiter=","))
            clipped_ErrorMask = np.array(np.loadtxt('ErrorMask.csv', delimiter=","))
            
        # If file missing prompt user to import/ produce files 
        except:
            print('Missing Simulation files, run preprocessing or import data')
       
        #  ---------------------------------------------------- Data-Processing---------------------------------------------------  
        # Process simulation data to include full range of scan positions
        U2, RF, ErrorMask, N  =  DataProcessing(clipped_RF, clipped_U2, scanPos, clipped_scanPos, clipped_ErrorMask, indentionDepth, timePeriod, timeInterval)
        if 'DataPlot' in kwargs.keys() and kwargs['DataPlot'] == True:
            DataPlot(scanPos, U2, RF, N)
          
        #  ------------------------------------------------AFM Force Contour-------------------------------------------------------  
        # Return force contours and plot in AFM image
        X,Y,Z = ForceContours(U2, RF, forceRef, scanPos, baseDims, binSize)
        ContourPlot(X, Y, Z, ErrorMask, baseDims, binSize, forceRef, contrast, pdb, **kwargs)

        if 'HSPlot' in kwargs.keys() and kwargs['HSPlot'] == True:
            HardSphereAFM(scanPos, baseDims, binSize, clearance, contrast, pdb, **kwargs)
    
    # Return final time of simulation
    T1 = time.time()
    print('Simulation Complete - ' + str(timedelta(seconds=T1-T0)) )
    
    if 'returnData' in kwargs.keys() and kwargs['returnData'] == True:
        return U2, RF, X, Y, Z, scanPos, baseDims 
    else:
        None
