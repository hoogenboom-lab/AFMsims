
# ----------------------------------------------Load Modules-----------------------------------------------------------
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
#from abaqus import getInput
executeOnCaeStartup()

# ----------------------------------------------Import variables------------------------------------------------------
spheres = [2.5, 5 ,10,15, 20, 30,40,50,60]

# Cone variables
R_indentor = 5
theta_degrees = 20
theta = theta_degrees*(np.pi/180)
tip_length = 25

# Intercept of spherical and Capped section of indentor (Tangent point) 
x_int, y_int = R_indentor*np.cos(theta), -R_indentor*np.sin(theta)

# Total radius/ footprint of indentor/ top coordinates
x_top, y_top = tip_length*np.tan(theta), tip_length

# Simulation variables
timePeriod = 2.5
timeInterval = 0.1
N = int(timePeriod/ timeInterval)

# Array for data
RFBase = np.zeros([len(spheres),N+1])
U2Base = np.zeros([len(spheres),N+1])
RFIndentor = np.zeros([len(spheres),N+1])
U2Indentor = np.zeros([len(spheres),N+1])

#  ---------------------------------------------------Model-----------------------------------------------------------
for i, r in enumerate(spheres):
    modelName = jobName = 'AFMtestShpere_Indentor-'+str(i)
    model = mdb.Model(name=modelName)

    # ------------------------------------------------Set Parts----------------------------------------------------------- 
    # Create Indentor part
    model.ConstrainedSketch(name = 'indentor', sheetSize=1.0) 
    model.sketches['indentor'].ConstructionLine(point1=(0,-R_indentor),point2=(0,y_top))
    model.sketches['indentor'].ArcByCenterEnds(center=(0,0), point1=(x_int,y_int), point2=(0,-R_indentor), 
                                               direction = CLOCKWISE)
    model.sketches['indentor'].Line(point1=(x_int,y_int), point2=(x_top,y_top))
    model.sketches['indentor'].Line(point1=(0,y_top), point2=(x_top,y_top))
    
    model.Part(name='indentor', dimensionality=AXISYMMETRIC, type= DISCRETE_RIGID_SURFACE)
    model.parts['indentor'].BaseWire(sketch = model.sketches['indentor'])

    # Create Sphere part 
    model.ConstrainedSketch(name = 'sphere', sheetSize=1.0)   
    model.sketches['sphere'].ConstructionLine(point1=(0,r),point2=(0,-r))
    model.sketches['sphere'].ArcByCenterEnds(center=(0,0),point1=(0,r),point2=(0,-r), direction = CLOCKWISE)
    model.sketches['sphere'].Line(point1=(0,r),point2=(0,-r))
    model.Part(name='sphere', dimensionality=AXISYMMETRIC, type=DEFORMABLE_BODY)
    model.parts['sphere'].BaseShell(sketch = model.sketches['sphere'])

    # Create Base Part
    model.ConstrainedSketch(name = 'base', sheetSize=1.0)   
    model.sketches['base'].ConstructionLine(point1=(0,0),point2=(0,r))
    model.sketches['base'].Line(point1=(0,0),point2=(2*r,0))
    model.Part(name='base', dimensionality=AXISYMMETRIC, type= DISCRETE_RIGID_SURFACE)
    model.parts['base'].BaseWire(sketch = model.sketches['base'])


    # ----------------------------------------------Set Geometry----------------------------------------------------------
    # Create geometric sets for referencing
    model.parts['sphere'].Set(faces= model.parts['sphere'].faces, name='sphere_edges')
    model.parts['indentor'].Set(edges= model.parts['indentor'].edges, name='indentor_edges')
    model.parts['base'].Set(edges= model.parts['base'].edges, name='base_edges')

    # Create gemoetric surface for contact
    model.parts['sphere'].Surface(name='sphere_surface', 
                                 side1Edges = model.parts['sphere'].edges.getSequenceFromMask(mask=('[#2]', ), ) )
    model.parts['indentor'].Surface(name='indentor_surface', 
                                    side1Edges = model.parts['indentor'].edges.getSequenceFromMask(
                                        mask=('[#1]','[#2]',),))
    model.parts['base'].Surface(name='base_surface', 
                                 side1Edges = model.parts['base'].edges.getSequenceFromMask(mask=('[#1]', ), ) )

    # Create reference points
    point = model.parts['indentor'].ReferencePoint((0, 0, 0))
    model.parts['indentor'].Set(referencePoints = (model.parts['indentor'].referencePoints[point.id],),
                                name= 'indentor_centre')
    
    point = model.parts['sphere'].ReferencePoint((0, -r, 0))
    model.parts['sphere'].Set(referencePoints = (model.parts['sphere'].referencePoints[point.id],), name='sphere_base')
    
    point = model.parts['base'].ReferencePoint((0, 0, 0))
    model.parts['base'].Set(referencePoints = (model.parts['base'].referencePoints[point.id],), name= 'base_centre')


    # -----------------------------------------------Set Properties-------------------------------------------------------
    # Assign materials
    model.Material(name='sphere_material')
    model.materials['sphere_material'].Elastic(table=((1000, 0.2), ))
    model.HomogeneousSolidSection(name='section', material='sphere_material', thickness=None)
    model.parts['sphere'].SectionAssignment( region = model.parts['sphere'].sets['sphere_edges'], sectionName='section')


    # ----------------------------------------------Set Assembly----------------------------------------------------------
    model.rootAssembly.regenerate()
    
    model.rootAssembly.Instance(name='sphere', part = model.parts['sphere'],dependent=ON)
    model.rootAssembly.Instance(name='indentor', part = model.parts['indentor'], dependent=ON)
    model.rootAssembly.Instance(name='base', part = model.parts['base'], dependent=ON)

    model.rootAssembly.translate(instanceList = ('indentor',) , vector = (0,2*r+R_indentor,0) )
    model.rootAssembly.translate(instanceList = ('sphere',)  , vector = (0,r,0) )

    model.rootAssembly.DatumCsysByDefault(CARTESIAN)


    # ------------------------------------------------Set Steps-----------------------------------------------------------
    step = model.StaticStep(name='Step-1', previous='Initial', description='', timePeriod=timePeriod, 
                            timeIncrementationMethod=AUTOMATIC, maxNumInc=int(1e5), 
                            initialInc=1.0, minInc=1e-25, maxInc=1)
    field = model.FieldOutputRequest('F-Output-1', createStepName='Step-1', variables=('RF', 'TF', 'U'), 
                                     timeInterval = timeInterval)
    model.steps['Step-1'].control.setValues(allowPropagation=OFF, resetDefaultValues=OFF, 
                              timeIncrementation=(4.0, 8.0, 9.0, 16.0, 10.0, 4.0, 12.0, 25.0, 6.0, 3.0, 50.0))


    # ----------------------------------------------Set Interactions------------------------------------------------------
    model.ContactProperty(name = 'Contact Properties')
    model.interactionProperties['Contact Properties'].TangentialBehavior(formulation =ROUGH)
    model.interactionProperties['Contact Properties'].NormalBehavior(pressureOverclosure=HARD)

    model.RigidBody(name= 'indentor_constraint', 
                    bodyRegion = model.rootAssembly.instances['indentor'].sets['indentor_edges'],
                    refPointRegion = model.rootAssembly.instances['indentor'].sets['indentor_centre'])

    model.RigidBody(name= 'base_constraint', 
                    bodyRegion = model.rootAssembly.instances['base'].sets['base_edges'],
                    refPointRegion = model.rootAssembly.instances['base'].sets['base_centre'])

    model.SurfaceToSurfaceContactStd(name    = 'surface-indentor', 
                                     createStepName = 'Initial', 
                                     master = model.rootAssembly.instances['indentor'].surfaces['indentor_surface'], 
                                     slave  = model.rootAssembly.instances['sphere'].surfaces['sphere_surface'],
                                     interactionProperty = 'Contact Properties', 
                                     sliding = FINITE)

    model.SurfaceToSurfaceContactStd(name    = 'base-sphere', 
                                     createStepName = 'Initial', 
                                     master =  model.rootAssembly.instances['base'].surfaces['base_surface'], 
                                     slave  =  model.rootAssembly.instances['sphere'].surfaces['sphere_surface'],
                                     interactionProperty = 'Contact Properties', 
                                     sliding = FINITE)


    # -----------------------------------------------Set Loads------------------------------------------------------------
    # Create surface boundary conditions
    model.DisplacementBC(name='Base-BC',createStepName='Initial', 
                         region= model.rootAssembly.instances['base'].sets['base_edges'], 
                         u1=SET, u2=SET, ur3= SET)
    # Create indentor boundary conditions
    model.DisplacementBC(name='Indentor-U',createStepName='Step-1',
                         region= model.rootAssembly.instances['indentor'].sets['indentor_centre'], 
                         u1=SET, u2=-6, ur3 = SET )


    # ------------------------------------------------Set Mesh------------------------------------------------------------
    #Assign an 'sphere' type to the part instance- seed and generate
    model.rootAssembly.regenerate()
    model.parts['sphere'].seedPart(size = 0.3)    
    model.parts['sphere'].setElementType(model.rootAssembly.instances['sphere'].sets['sphere_edges'], 
                                         elemTypes =(mesh.ElemType(elemCode=TRI,secondOrderAccuracy = ON),) )
    model.parts['sphere'].setMeshControls(regions= model.rootAssembly.instances['sphere'].sets['sphere_edges'].vertices,
                                          elemShape=TRI, technique=FREE)
    model.parts['sphere'].generateMesh()
    
    
    model.parts['indentor'].seedPart(size = 0.1)
    model.parts['indentor'].generateMesh()

    
    model.parts['base'].seedPart(size = 1)
    model.parts['base'].generateMesh()


    # ----------------------------------------------Set Submission--------------------------------------------------------
    # Create an analysis job for the model and submit it.
    job = mdb.Job(name=jobName, model=modelName, description='AFM Sphere')
    job.writeInput()
    job.submit()
    job.waitForCompletion()

    # ----------------------------------------------Set Data extraction--------------------------------------------------

    # Opening the odb and the output database and display a default contour plot.
    odb    = session.openOdb(jobName +'.odb', readOnly=True)
    regionBase     = odb.rootAssembly.nodeSets.values()[1]
    regionIndentor = odb.rootAssembly.nodeSets.values()[2]
    
    # Extracting Step 1, this analysis only had one step
    step1 = odb.steps.values()[0]
        
    n,m,j,k = 0,0,0,0
    # Creating a for loop to iterate through all frames in the step
    
    for x in odb.steps[step1.name].frames:
        # Reading stress and strain data from the model 
        fieldRFBase = x.fieldOutputs['RF'].getSubset(region= regionBase)
        fieldUBase  = x.fieldOutputs['U'].getSubset(region= regionBase)    
                
        fieldRFIndentor = x.fieldOutputs['RF'].getSubset(region= regionIndentor)
        fieldUIndentor  = x.fieldOutputs['U'].getSubset(region= regionIndentor) 
        
        # Storing Stress and strain values for the current frame
        for rf in fieldRFBase.values:
            RFBase[i,j] = rf.data[1]
            j+=1
            
        for u in fieldUBase.values:
            U2Base[i,k] = u.data[1] 
            k+=1    
            
        for rf in fieldRFIndentor.values:
            RFIndentor[i,n] = rf.data[1]
            n+=1
            
        for u in fieldUIndentor.values:
            U2Indentor[i,m] = u.data[1] 
            m+=1    
        
# Writing to a .csv file
np.savetxt("U2Base_Results.csv", U2Base , delimiter=",")
np.savetxt("RFBase_Results.csv", RFBase , delimiter=",")
np.savetxt("U2Indentor_Results.csv", U2Indentor , delimiter=",")
np.savetxt("RFIndentor_Results.csv", RFIndentor , delimiter=",")

# Close the odb
odb.close()

mdb.saveAs('AFMtestCapped_Sphere.cae')
