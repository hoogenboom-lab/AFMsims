.. abqsims documentation master file, created by
   sphinx-quickstart on Fri Oct 13 14:16:53 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AFMSIMS Package
===================================

Authors: J. Giblin-Burnham 

Repository for Masters project code simulating AFM imaging using ABAQUS/FEM and various FEM simulations. There are four different simulations.

Simulations implement ABAQUS (2017) software for quasi-static, implicit computations using user subroutines UMAT. Samples are modelled as continuous, homogeneous and isotropic elastic materials with Young's modulus and Poisson ratio comparable to biomolecules. To eliminate the hourglass effect, R3D10 tetrahedral elements are employed.  Simulations impliment "surface to surface contact" interaction with "hard", nonadhesive normal contact and "rough" (Coulomb friction), non-slip tangential contact. Boundary conditions fix the base of the structures, and vertical force and indentation data are mapped and sampled via reference points at the indenter's centre.

The shape of a blunt AFM tip is a simplified construct similar to the SEM image of actual AFM tips shown by Chen et al. The tip is modelled as a rigid (incompressible) cone with opening angle \theta ending in a spherical termination of radius. The spherical portion smoothly transitions to the conical segment at the tangential contact point described by,

.. math:: X_{tangent} = R\cos\theta ; Y_{tangent} = R(1-\sin\theta)


.. toctree::
   :maxdepth: 4
   :caption: Installation

   Installation


.. toctree::
   :maxdepth: 4
   :caption: Models

   Models/Axisymmetric_sphere
   Models/Hemisphere_Simulation
   Models/Wave_Simulation
   Models/AFM_Simulation


.. toctree::
   :maxdepth: 4
   :caption: Documentation

   afmsims


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`