# ----------------------------------------------Load Modules----------------------------------------------------------
import numpy as np 
from abaqus import *
from abaqusConstants import *
from caeModules import *
from part import *
from section import *
from assembly import *
import odbAccess
import cProfile, pstats, io

# ------------------------------------------------Set variables-------------------------------------------------------
atom_coord   = np.loadtxt('atom_coords.csv', delimiter=",")
atom_element = np.loadtxt("atom_elements.csv", dtype = 'str', delimiter=",")

keys     = np.loadtxt('atom_radius_keys.csv', dtype = 'str', delimiter=",")
values   = np.loadtxt('atom_radius_values.csv', delimiter=",")
atom_radius = {keys[i]:values[i] for i in range(len(keys))}

baseDims  = np.loadtxt('baseDims.csv', delimiter=",")
tipDims   = np.loadtxt('tipDims.csv', delimiter=",")
rIndentor, theta, tip_length, r_int, z_int, r_top, z_top = tipDims  

with open('indentorType.txt', 'r') as f:
    indentorType = f.read()

#  -------------------------------------------------Model--------------------------------------------------------------
modelName = 'AFMSurfaceModel'
model = mdb.Model(name=modelName)

# ------------------------------------------------Create Atom Parts---------------------------------------------------- 
for atom in list(atom_radius.keys()):
    # Cretae base atoms (as spheres with vdw radius) to form the biomolecule  
    r = atom_radius[atom]
    sketch = model.ConstrainedSketch(name = atom, sheetSize=1.0)   
    model.sketches[atom].ConstructionLine(point1=(0,r),point2=(0,-r))
    model.sketches[atom].ArcByCenterEnds(center=(0,0),point1=(0,r),point2=(0,-r), direction = CLOCKWISE)
    model.sketches[atom].Line(point1=(0,r),point2=(0,-r))
    part = model.Part(name=atom, dimensionality=THREE_D, type=DEFORMABLE_BODY)
    model.parts[atom].BaseSolidRevolve( angle=360.0, flipRevolveDirection=OFF, sketch= model.sketches[atom] )    

    
# ----------------------------------------------Create Molecule Assembly------------------------------------------------
# Centre coordinate system
model.rootAssembly.DatumCsysByDefault(CARTESIAN)

# Cretae biomolecule by looping through atom list and merge part instances of the individual atoms
for i, coord in enumerate(atom_coord):
    atom = atom_element[i]
    # For atoms part of molecule above the base 
    if coord[2] >= -atom_radius[atom]:
        # Create new part instance for each atom in molecule and translate to position from coordinate list
        model.rootAssembly.Instance(name='instance'+str(i), part = model.parts[atom], dependent=ON)
        model.rootAssembly.translate(instanceList = ('instance'+str(i),) , vector = (coord[0],coord[1],coord[2]) )

# Merge the atomic part instances to make molecule part and part instance 
Instances_List = list(model.rootAssembly.instances.keys())
model.rootAssembly.InstanceFromBooleanMerge(name='molecule',
                                            instances=([model.rootAssembly.instances[Instances_List[i]] 
                                                        for i in range(len(Instances_List))] ), 
                                            originalInstances=DELETE) 
# Delete individual atoms
for atom in list(atom_radius.keys()):
    del model.parts[atom]

# -----------------------------------------Create Base/Substrate Part---------------------------------------------------
# Create base part using predefined base dimensions in baseDims, add width/height to accomidate radius of indenter
model.ConstrainedSketch(name = 'base', sheetSize=1.0) 
model.sketches['base'].rectangle(point1=(-baseDims[0]/2-rIndentor,-baseDims[1]/2-rIndentor), 
                                 point2=( baseDims[0]/2+rIndentor, baseDims[1]/2+rIndentor) )
model.Part(name='base', dimensionality=THREE_D, type= DEFORMABLE_BODY)
model.parts['base'].BaseSolidExtrude(sketch= model.sketches['base'], depth = baseDims[2])    

# Create as base part instance
model.rootAssembly.Instance(name='base', part = model.parts['base'], dependent=ON)
model.rootAssembly.translate(instanceList = ('base',) , vector = (0,0,-baseDims[2]) )

# ----------------------------------------------Create Surface Part-----------------------------------------------------
# Create biomolecule surface by cut any of the molecule intersecting the base  
model.rootAssembly.Instance(name='molecule', part = model.parts['molecule'], dependent=ON)
model.rootAssembly.InstanceFromBooleanCut(name= 'surface', instanceToBeCut=model.rootAssembly.instances['molecule'], 
                                          cuttingInstances=(model.rootAssembly.instances['base'],),
                                          originalInstances=DELETE)
# Delete unclipped molecule part
del model.parts['molecule'] 

# -----------------------------------------------Create Tip Part--------------------------------------------------------
# If set, create Capped-Conical Indentor using predefined dimensions in tipDims. Using rigid/ incompressible shell part
if indentorType == 'Capped':
    
    sketch = model.ConstrainedSketch(name = 'indentor', sheetSize=1.0)   
    model.sketches['indentor'].ConstructionLine(point1=(0,-rIndentor),point2=(0,z_top))
    model.sketches['indentor'].Line(point1=(r_int,z_int), point2=(r_top,z_top))
    model.sketches['indentor'].Line(point1=(0,-rIndentor), point2=(0,z_top))
    model.sketches['indentor'].Line(point1=(0,z_top), point2=(r_top,z_top))
    model.sketches['indentor'].ArcByCenterEnds(center=(0,0), point1=(r_int,z_int), point2=(0,-rIndentor),
                                               direction =CLOCKWISE)  
    model.Part(name='indentor', dimensionality=THREE_D, type= DISCRETE_RIGID_SURFACE)
    model.parts['indentor'].BaseShellRevolve(angle=360.0, flipRevolveDirection=OFF, sketch=model.sketches['indentor'])
    
    plane = model.parts['indentor'].DatumPlaneByPrincipalPlane(offset= tip_length/3, principalPlane=XZPLANE)
    model.parts['indentor'].PartitionFaceByDatumPlane(datumPlane = model.parts['indentor'].datums[plane.id], 
                                                      faces = model.parts['indentor'].faces.getSequenceFromMask(('[#2 ]', ),))
    
# Otherwise create Spherical Indentor part using rIndentor only
else:
    model.ConstrainedSketch(name = 'indentor', sheetSize=1.0)   
    model.sketches['indentor'].ConstructionLine(point1=(0,rIndentor),point2=(0,-rIndentor))
    model.sketches['indentor'].ArcByCenterEnds(center=(0,0),point1=(0,rIndentor),point2=(0,-rIndentor), 
                                               direction = CLOCKWISE)
    model.sketches['indentor'].Line(point1=(0,rIndentor),point2=(0,-rIndentor))
    model.Part(name='indentor', dimensionality=THREE_D, type=DISCRETE_RIGID_SURFACE)
    model.parts['indentor'].BaseShellRevolve( angle=360.0, flipRevolveDirection=OFF, sketch=model.sketches['indentor'])


# ----------------------------------------------Export Part Files-----------------------------------------------------
model.parts['surface'].writeAcisFile( fileName = 'surface' )
model.parts['base'].writeAcisFile( fileName = 'base' )
model.parts['indentor'].writeAcisFile( fileName = 'indentor' )

# Save Model 
mdb.saveAs(modelName +'.cae')
