# Simulations Notebooks

There are five different simulations. Finite element simulations implement ABAQUS (2017) software for quasi-static, implicit computations using user subroutines UMAT. Samples are modelled as continuous, homogeneous and isotropic elastic materials with Young's modulus and Poisson ratio comparable to biomolecules. To eliminate the hourglass effect, R3D10 tetrahedral elements are employed.  Simulations impliment "surface to surface contact" interaction with "hard", nonadhesive normal contact and "rough" (Coulomb friction), non-slip tangential contact. Boundary conditions fix the base of the structures, and vertical force and indentation data are mapped and sampled via reference points at the indenter's centre.

Example notebooks are availabel for : 

* Hard Sphere AFM image simulations with no indentation or finite element modelling (hard-sphere)
* Finite element AFM Image Simulator (afm)
* Axisymmetric simulation of indentation of elastic sphere (axisymmetric) 
* Three dimensional simulation of compression along scanline of a hemisphere (hemisphere)
* Three dimensional simulation of compression along scanline of a periodic structure (wave)


Also available are some C++ scripts for hard sphere calculations (cpp-calculations)