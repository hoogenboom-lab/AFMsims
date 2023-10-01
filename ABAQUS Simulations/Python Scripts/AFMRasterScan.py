# ----------------------------------------------Load Modules----------------------------------------------------------
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
import subprocess
import os
import __main__
executeOnCaeStartup()


# ------------------------------------------------Set variables-------------------------------------------------------
# Import predefined variables from files set in current directory
variables = np.loadtxt('variables.csv', delimiter=",")

baseDims  = np.loadtxt('baseDims.csv', delimiter=",")
tipDims   = np.loadtxt('tipDims.csv', delimiter=",")

clipped_scanPos  = np.loadtxt('clipped_scanPos.csv', delimiter=",")

timePeriod, timeInterval, binSize, meshSurface, meshBase, meshIndentor, indentionDepth, surfaceHeight = variables
rIndentor, theta, tip_length, r_int, z_int, r_top, z_top = tipDims  

with open('indentorType.txt', 'r') as f:
    indentorType = f.read()
    
#  -----------------------------------------------Set Model-------------------------------------------------------------
modelName =  'AFMRasterScan'

# Open surface model with parts and rename to simulation model
# Linux nomeclature - for Windows use '\\AFMSurfaceModel.cae' 
mdb.openAuxMdb( pathName = os.getcwd() +'/AFMSurfaceModel.cae')
mdb.copyAuxMdbModel(fromName='AFMSurfaceModel', toName = modelName)
mdb.closeAuxMdb()

# Set model variable for briefity
model = mdb.models[modelName]


# ------------------------------------------------Set Parts----------------------------------------------------------- 
# Alternatively, model can be set and indiviadual parts can be imported however shapes detail can be lost in import
# model = mdb.Model(name=modelName)

# # Create Surface part 
# surface = mdb.openAcis(fileName = 'surface' )
# model.PartFromGeometryFile(name = 'surface', geometryFile = surface , dimensionality =THREE_D, 
#                            type = DEFORMABLE_BODY)

# # Create Base part
# base = mdb.openAcis(fileName = 'base' )
# model.PartFromGeometryFile(name= 'base', geometryFile = base, dimensionality = THREE_D,
#                            type = DEFORMABLE_BODY)   

# # Create Indentor part
# indentor = mdb.openAcis(fileName = 'indentor' )
# model.PartFromGeometryFile(name= 'indentor', geometryFile = indentor, dimensionality = THREE_D, 
#                            type = DISCRETE_RIGID_SURFACE)    
    
    
# ----------------------------------------------Set Assembly----------------------------------------------------------
model.rootAssembly.DatumCsysByDefault(CARTESIAN)

# Regenerate part instances and orientate base and indentor
model.rootAssembly.Instance(name='surface', part = model.parts['surface'],dependent=ON)
model.rootAssembly.Instance(name='indentor', part = model.parts['indentor'], dependent=ON)
model.rootAssembly.Instance(name='base', part = model.parts['base'], dependent=ON)

model.rootAssembly.rotate(instanceList = ('indentor',), axisPoint = (0,0,0), axisDirection = (1,0,0), angle = 90)
model.rootAssembly.translate(instanceList = ('base',) , vector = (0,0,-baseDims[2]) )

# Delete duplicate of part instance
del model.rootAssembly.instances['molecule-1'] 
del model.rootAssembly.instances['surface-1'] 
 
    
# ----------------------------------------------Set Geometry----------------------------------------------------------
# Create geometric sets and surfaces for each parts faces and cells - these sets are used to reference in model set up

# Surface sets and gemoetric surface for contact
model.parts['surface'].Set(cells= model.parts['surface'].cells.getSequenceFromMask(mask=('[#1fff]', ), ), 
                         name='surface_cells')

model.parts['surface'].Set( name='surface_base', faces= 
                           model.parts['surface'].faces.getByBoundingBox(-baseDims[0]/2,-baseDims[1]/2,-0.5,
                                                                         baseDims[0]/2,baseDims[1]/2,0.5))
# Base sets and surfaces
model.parts['base'].Set(faces= model.parts['base'].faces.getSequenceFromMask(mask=('[#20]', ),),name='base_faces')
model.parts['base'].Set(cells= model.parts['base'].cells.getSequenceFromMask(mask=('[#1]', ),), name='base_cells')


# Indentor sets and surfaces
if indentorType == 'Capped':
    # For Spherically Capped geometry
    model.parts['indentor'].Set(faces= model.parts['indentor'].faces.getSequenceFromMask(mask=('[#f]', ), ),
                                name='indentor_faces')    
    model.parts['indentor'].Surface(name='indentor_surface', 
                                    side1Faces = model.parts['indentor'].faces.getSequenceFromMask(mask=('[#9]', ), ))    
else:
    # For Spherical geometry
    model.parts['indentor'].Set(faces= model.parts['indentor'].faces.getSequenceFromMask(mask=('[#1]', ), ),
                                name='indentor_faces')                      
    model.parts['indentor'].Surface(name='indentor_surface', 
                                    side1Faces = model.parts['indentor'].faces.getSequenceFromMask(mask=('[#1]', ), ))    
# Create reference points for indentor
point = model.parts['indentor'].ReferencePoint((0, 0, 0))
model.parts['indentor'].Set(referencePoints = (model.parts['indentor'].referencePoints[point.id],),
                            name = 'indentor_centre')

model.rootAssembly.regenerate()


# -----------------------------------------------Set Properties-------------------------------------------------------
# Assign materials, using elastic and visoelastic properties
elastic = ((1000, 0.3), )
viscoelastic = ((0.0403,0,0.649),(0.0458,0,1.695),)

# Assign molecule surface material
model.Material(name='surface_material')
model.materials['surface_material'].Elastic(table = elastic)
# model.materials['surface_material'].Viscoelastic(domain = FREQUENCY, frequency = PRONY, table = viscoelastic )
model.HomogeneousSolidSection(name='section', material='surface_material', thickness=None)
model.parts['surface'].SectionAssignment(region=model.parts['surface'].sets['surface_cells'],sectionName='section')

# Assign base/substrate large (incompressible) material
model.Material(name='base_material')
model.materials['base_material'].Elastic(table = ((1e15,0.4),))
model.HomogeneousSolidSection(name='base_section', material='base_material', thickness=None)
model.parts['base'].SectionAssignment(region = model.parts['base'].sets['base_cells'], sectionName='base_section')


# ------------------------------------------------Set Steps-----------------------------------------------------------
model.StaticStep(name='Indentation', previous='Initial', description='', timePeriod=timePeriod, 
                 timeIncrementationMethod=AUTOMATIC, maxNumInc=int(1e5), initialInc=0.1, minInc=1e-20, maxInc=1)

model.steps['Indentation'].control.setValues(allowPropagation=OFF, resetDefaultValues=OFF, 
                                timeIncrementation=(4.0, 8.0, 9.0, 16.0, 10.0, 4.0, 12.0, 25.0, 6.0, 3.0, 50.0))

field = model.FieldOutputRequest('F-Output-1', createStepName='Indentation', variables=('RF', 'TF', 'U'), 
                                 timeInterval = timeInterval)


# ----------------------------------------------Set Interactions------------------------------------------------------
# Set  Contact Behaviour
model.ContactProperty(name ='Contact Properties')
model.interactionProperties['Contact Properties'].TangentialBehavior(formulation = FRICTIONLESS) #PENALTY FRICTION
model.interactionProperties['Contact Properties'].NormalBehavior(pressureOverclosure=HARD)

# Set Ridged Indentor                                 
model.RigidBody(name = 'indentor_constraint', 
                bodyRegion = model.rootAssembly.instances['indentor'].sets['indentor_faces'],
                refPointRegion = model.rootAssembly.instances['indentor'].sets['indentor_centre'])


# -----------------------------------------------Set Loads------------------------------------------------------------
# Create base boundary conditions
model.DisplacementBC(name = 'Base-BC', createStepName = 'Initial', 
                     region = model.rootAssembly.instances['base'].sets['base_faces'], 
                     u1 = SET, u2 = SET, u3 = SET, ur1 = SET, ur2 = SET, ur3 = SET)

# Create surface boundary conditions
model.DisplacementBC(name = 'Surface-BC', createStepName = 'Initial', 
                     region = model.rootAssembly.instances['surface'].sets['surface_base'], 
                     u1 = SET, u2 = SET, u3 = SET, ur1 = SET, ur2 = SET, ur3 = SET)

# Create indentor boundary conditions
model.DisplacementBC(name = 'Indentor-UC', createStepName = 'Indentation',                  
                     region = model.rootAssembly.instances['indentor'].sets['indentor_centre'], 
                     u1 = SET, u2 = SET, u3 = -indentionDepth,
                     ur1 = SET, ur2 = SET, ur3 = SET)

    
# ------------------------------------------------Set Mesh------------------------------------------------------------
# Assign an element type to the part instance- seed and generate
model.rootAssembly.regenerate()

# Assign surface mesh using tetrahedral elements
elemType1 = mesh.ElemType(elemCode=C3D20R, elemLibrary=STANDARD)
elemType2 = mesh.ElemType(elemCode=C3D15, elemLibrary=STANDARD)
elemType3 = mesh.ElemType(elemCode=C3D10, elemLibrary=STANDARD)
cells     =  model.parts['surface'].cells.getSequenceFromMask(mask=('[#1]', ), )

model.parts['surface'].seedPart(size=meshSurface, deviationFactor=0.01, minSizeFactor=0.95)
model.parts['surface'].setMeshControls(regions=cells, elemShape=TET, sizeGrowthRate=1.64, technique=FREE)
model.parts['surface'].setElementType(regions=(cells,), elemTypes=(elemType1, elemType2, elemType3))
model.parts['surface'].generateMesh()


# Assign indentor mesh using triangular elements
elemType1 = mesh.ElemType(elemCode=R3D4, elemLibrary=STANDARD)
elemType2 = mesh.ElemType(elemCode=R3D3, elemLibrary=STANDARD)
faces     = model.parts['indentor'].faces.getSequenceFromMask(mask=('[#1 ]', ),)

model.parts['indentor'].seedPart(size=meshIndentor, minSizeFactor=0.25)
model.parts['indentor'].setMeshControls(regions=faces, elemShape=TRI)
model.parts['indentor'].setElementType(regions=(faces,), elemTypes=(elemType1, elemType2))

if indentorType == 'Capped':
    # Long edge above partition
    model.parts['indentor'].seedEdgeBySize(constraint=FINER, size=3*meshIndentor, edges = model.parts['indentor'].edges.getSequenceFromMask(('[#20 ]', ),))
    # Long edge partition portion
    model.parts['indentor'].seedEdgeByBias(biasMethod=SINGLE, maxSize=meshIndentor, minSize=0.1*meshIndentor, constraint=FINER, 
                                           end2Edges = model.parts['indentor'].edges.getSequenceFromMask(('[#2 ]', ), ))
    # Spherical tip mesh
    model.parts['indentor'].seedEdgeBySize(constraint=FINER, size=0.1*meshIndentor, deviationFactor=0.1, 
                                           edges = model.parts['indentor'].edges.getSequenceFromMask(('[#40 ]', ), ))
model.parts['indentor'].generateMesh()



# Assign base mesh using tetrahedral elements
model.parts['base'].seedPart(size = meshBase )
model.parts['base'].setElementType(model.rootAssembly.instances['base'].sets['base_faces'], 
                                      elemTypes = (mesh.ElemType(elemCode=QUAD, elemLibrary=STANDARD),))
model.parts['base'].setMeshControls(regions=model.rootAssembly.instances['base'].sets['base_faces'].vertices, 
                                       elemShape=TET,technique=FREE)
model.parts['base'].generateMesh()


## ----------------------------------------------Set Submission--------------------------------------------------------
for i in range(len(clipped_scanPos)):  
    
    # ---------------------------------------Translate Indentor position-----------------------------------------------
    # Translate indentor to raster scan position
    model.rootAssembly.translate(instanceList = ('indentor',),
                                 vector = (clipped_scanPos[i,0], clipped_scanPos[i,1], clipped_scanPos[i,2] + rIndentor )) 
    model.rootAssembly.regenerate()
        
    # -----------------------------------------Set Surface Interactions------------------------------------------------
    # Define surface-indentor contact region
    model.parts['surface'].Surface(name       = 'surface_top', 
                                   side1Faces = model.parts['surface'].faces.getByBoundingCylinder((clipped_scanPos[i,0], clipped_scanPos[i,1], 5), 
                                                                                                   (clipped_scanPos[i,0], clipped_scanPos[i,1], clipped_scanPos[i,2]+tip_length), 
                                                                                                   1.4*r_top))

    # Define surface-surface contact for each body only around indentor
    model.SurfaceToSurfaceContactStd(name = 'surface-indentor', 
                                     createStepName = 'Initial', 
                                     master     = model.rootAssembly.instances['indentor'].surfaces['indentor_surface'], 
                                     slave = model.rootAssembly.instances['surface'].surfaces['surface_top'],
                                     interactionProperty = 'Contact Properties', 
                                     sliding = FINITE)
    
    # ----------------------------------------------Create Input file--------------------------------------------------
    # Create input file for simulation position
    jobName   = 'AFMRasterScan-Pos'+str(int(i))
    job = mdb.Job(name=jobName, model=modelName, description='AFM')
    job.writeInput()

    if i == int(len(clipped_scanPos)/2):
        mdb.saveAs('AFMRasterScan.cae')
        
    # -------------------------------------------------Reset Model-----------------------------------------------------    
    # Reset indentor position to centre
    model.rootAssembly.translate(instanceList = ('indentor',),
                                 vector = (-clipped_scanPos[i,0], -clipped_scanPos[i,1], -(clipped_scanPos[i,2] + rIndentor) ))
    
    del model.parts['surface'].surfaces['surface_top']
    del model.interactions['surface-indentor']
