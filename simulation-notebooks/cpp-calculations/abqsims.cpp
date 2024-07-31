#include <iostream>
#include <cmath>
#include <vector>
#include <tuple>
#include <string>
#include <map>
#include <list>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <typeinfo>
// #include "matplotlib.h"

// namespace plt = matplotlibcpp;
using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Number>
vector<double> linspace(Number start_in, Number end_in, int num_in)
{
    vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0) { return linspaced;}
    if (num == 1) 
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
    return linspaced;
}

vector<vector<double>> T(vector<double> v)
{
    int N = v.size();
    vector<vector<double>> v_T( N, vector<double>(1) );

    for (int i = 0; i<N; i++)
        v_T[i][0] = v[i]; 

    return v_T ;
}
vector<vector<double>> T(vector<vector<double>> A)
{
    int N = A.size();
    int M = A[0].size();

    vector<vector<double>> A_T( M, vector<double>(N) );

    for (int i = 0; i<N; i++)
        for (int j = 0; j<M; j++)
            A_T[j][i] = A[i][j]; 

    return A_T ;
}


vector<double> times(vector<double> A, vector<double> B)
{
    int N = A.size();
    
    vector<double> AB(N);

    for (int i=0; i<N; ++i)
        AB[i] = A[i]*B[i];
    
    return AB ;
}
vector<vector<double>> times(vector<vector<double>> A, vector<vector<double>> B)
{
    int N = A.size();
    int M = A[0].size();
    
    vector<vector<double>> AB( N, vector<double>(M) );

    for (int i=0; i<N; ++i)
        for (int j = 0; j<M; j++)
            AB[i][j] = A[i][j]*B[i][j];

    return AB ;
}


vector<double> add(vector<double> A, double value)
{
    int N = A.size();
    vector<double> AB(N);

    for (int i=0; i<N; ++i)
        AB[i] = A[i] + value;

    return AB;
}


double dot(double a, double b){ return a*b; }
double dot(vector<double> u, vector<double> v)
{
    int N = u.size();
    double uv = 0.0;

    for(int i=0; i<N; ++i)
        uv += u[i]*v[i];

    return uv;
}
vector<vector<double>> dot(vector<vector<double>> A, vector<vector<double>> B)
{
    auto T_B = T(B);
    int N = A.size();
    int M = T_B.size();
    
    vector<vector<double>> AB( N, vector<double>(M) );

    for (int i=0; i<N; ++i)
        for (int j = 0; j<M; j++)
            AB[i][j] = dot(A[i],T_B[j]);

    return AB ;
}


vector<vector<double>> tensor(vector<double> u, vector<double> v)
{
    int N = u.size();
    int M = v.size();

    vector<vector<double>> A( N, vector<double>(M) );

    for(int i=0; i<N; ++i)
        for (int j = 0; j<M; j++)
            A[i][j] = u[i]*v[j]; 
    return A;
}

template<typename value>
void display(vector<value>A)
{
    cout <<  '|'<<' ' ;
    for (auto m:A){ cout << m <<  ' ' ;} 
    cout <<  '|' << endl;   
}
template<typename value>
void display(vector<vector<value>>A)
{
    for (auto n:A)
    {   
        cout <<  '|'<<' ' ;
        for (auto m:n){ cout << m <<  ' ' ;} 
        cout <<  '|' << endl;   
    }
}


vector<vector<double>> Rotate(vector<vector<double>> domain, vector<double> rotation)
{   /* Rotate coordinates of a domain around each coordinate axis by angles given.

    Args:
        domain (arr)    : Array of [x,y,z] coordinates in domain to be rotated (Shape: (3) or (N,3) )
        rotation (list) : Array of [xtheta, ytheta, ztheta] rotational angle around coordinate axis:
                            * xtheta(float), angle in degrees for rotation around x axis (Row)
                            * ytheta(float), angle in degrees for rotation around y axis (Pitch)
                            * ztheta(float), angle in degrees for rotation around z axis (Yaw)
    Returns:
        rotate_domain(arr) : Rotated coordinate array
    */

    double xtheta = (M_PI/180)*rotation[0]; 
    double ytheta = (M_PI/180)*rotation[1];
    double ztheta = (M_PI/180)*rotation[2];
    
    // Row, Pitch, Yaw rotation matrices
    vector<vector<double>> R_x = { {1,0,0}, {0,cos(xtheta),-sin(xtheta)}, {0,sin(xtheta),cos(xtheta)} };   
    vector<vector<double>> R_y = { {cos(ytheta),0,sin(ytheta)}, {0,1,0}, {-sin(ytheta),0,cos(ytheta)} };
    vector<vector<double>> R_z = { {cos(ztheta),-sin(ztheta),0}, {sin(ztheta),cos(ztheta),0}, {0,0,1} };
    
    // Complete rotational matrix, from matrix multiplication
    vector<vector<double>> R = dot(R_x, dot(R_y, R_z));
    vector<vector<double>> rotated = T( dot(R, T(domain)) );
    
    return rotated;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void retrieve_pdb_file(string  pdbid, string file_type)
{    /* Fetch PDB structure file from PDB server, and store it locally.

    The PDB structure's file name is returned as a single string.
    If obsolete ``==`` True, the file will be saved in a special file tree.
    */

    string cwd = filesystem::current_path();
    string pdbDir   = cwd + "/" + pdbid[1] + pdbid[2] ;
    string bashFile = "pdb_download_script.sh";

    filesystem::create_directory(pdbDir);

    // Create and open a text file
    ofstream script(bashFile);
    // Write to the file
    script << " #!/bin/bash \n" ;
    script << " # Script to download files from RCSB http file download services. \n" ;
    script << "\n" ; 
    script << " PROGNAME=$0 \n" ;
    script << " BASE_URL=\"https://files.rcsb.org/download\" \n" ;
    script << "\n" ;
    script << " download() { \n" ;
    script << "   url=\"$BASE_URL/$1\" \n" ;
    script << "   out=$2/$1 \n" ;
    script << "   echo \"Downloading $url to $out\" \n" ;
    script << "   curl -s -f $url -o $out || echo \"Failed to download $url\" \n" ;
    script << " } \n" ;
    script << "\n" ;
    script << "outdir="+pdbDir+"\n";
    script << "shift \"$((OPTIND - 1))\" \n" ;
    script << "contents=\""+pdbid+"\"\n";
    script << "\n" ;

    // Download a cif.gz file for each PDB id'
    script << "   download $contents."+file_type+".gz $outdir \n" ;

    // Close the file
    script.close();

    string chmod = "chmod +x " + cwd + "/" + bashFile;
    string command = cwd + "/" + bashFile;
    string zip = "gunzip -f " + pdbDir + "/" + pdbid + "." + file_type + ".gz" ;

    system(chmod.c_str());
    system(command.c_str());
    system(zip.c_str());
}


map<string, double> mendelev(void)
{   
    // Produce dictionary of element radii in angstrom (using van de waals or vdw_radius_dreiding vdw_radius_mm3 vdw_radius_uff )
    // atom_radius = dict.fromkeys(sort(atom_element))
    vector<string> elements = {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", 
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
     "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
      "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
       "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
    
    vector<double> radii = {2.886,2.362,2.451,2.745,4.083,3.8510000000000004,3.66,3.5,3.364,3.2430000000000003,2.983,3.0210000000000004,4.499,4.295,4.147,
    4.035,3.947,3.8680000000000003,3.812,3.3989999999999996,3.295,3.175,3.1439999999999997,3.023,2.9610000000000003,2.912,2.872,2.8339999999999996,3.495,
    2.763,4.383,4.28,4.23,4.205,4.189,4.141,4.114,3.641,3.345,3.1239999999999997,3.165,3.052,2.998,2.963,2.929,2.8989999999999996,3.148,2.8480000000000003,
    4.463,4.3919999999999995,4.42,4.47,4.5,4.404,4.5169999999999995,3.7030000000000003,3.522,3.556,3.6060000000000003,3.575,3.5469999999999997,3.52,3.4930000000000003,
    3.3680000000000003,3.451,3.428,3.409,3.391,3.3739999999999997,3.355,3.64,3.141,3.17,3.096,2.9539999999999997,3.12,2.84,2.7539999999999996,3.293,2.705,
    4.3469999999999995,4.297,4.37,4.709,4.75,4.765,4.9,3.677,3.478,3.3960000000000004,3.424,3.395,3.424,3.424,3.3810000000000002,3.326,3.339,3.313,3.299,3.286,
    3.2739999999999996,3.248,3.236,3.8510000000000004,3.8510000000000004,3.8510000000000004,3.8510000000000004,3.8510000000000004,3.8510000000000004,3.8510000000000004,
    3.8510000000000004,3.8510000000000004,3.8510000000000004,3.8510000000000004,3.8510000000000004,3.8510000000000004,3.8510000000000004,3.8510000000000004};

    map<string, double> atom_radii;
    for (int i = 0 ; i < elements.size(); i++)
    { 
        atom_radii[elements[i]] = radii[i] ;
    }

    return atom_radii;
}


class Structure
{
    public:             
        vector<string> atoms;
        vector<vector<double>> coords;
        map<string, double> atom_radii;
        double surfaceHeight;
        vector<double> baseDims;

        double rIndentor; 
        double theta; 
        double tip_length ;  
        double r_int ; 
        double z_int ; 
        double r_top ; 
        double z_top ;
};


Structure MolecularStructure(string pdbid, string file_type, vector<double> rotation, Structure tipDims, string indentorType, double binSize, double surfaceApprox)
{   /* his function imports the relevant PDB file and extracts molecular data from structure class.
    
    It takes care of the directory in which it is saved etc for the user, returning the structure.
    Returns array of molecules atomic coordinate and element names. Alongside, producing dictionary
    of element radii and calculating base dimensions. All distances given in Angstroms (x10-10 m).
    
    Args:
        pdbid (str)           : PDB (or CSV) file name of desired biomolecule
        file_type(str)        : PDB file type
        structure (class)     : Class containing proteins structural data (Atom coords/positions and masses etc...)
        rotation (list)       : Array of [x,y,z] rotational angle around coordinate axis'
        tipDims (list)        : Geometric parameters for defining capped tip structure     
        indentorType (str)    : String defining indentor type (Spherical or Capped)
        binSize (float)       : Width of bins that subdivid xy domain during raster scanning/ spacing of the positions sampled over
        surfaceApprox (float) : Percentage of biomolecule assumed to be not imbedded in base/ substrate. Range: 0-1 
    
    Returns:
        atom_coords (arr)      : Array of coordinates [x,y,z] for atoms in biomolecule 
        atom_element (arr)    : Array of elements names(str) for atoms in biomolecule 
        atom_radius (dict)    : Dictionary containing van der waals radii each the element in the biomolecule 
        surfaceHeight (float) : Maximum height of biomolecule in z direction
        baseDims (arr)        : Geometric parameters for defining base/ substrate structure [width, height, depth]           
        
    Keywords Args:
        CustomPDB (str): Extract data from local custom pd as opposed to from PDB online
        
    Returns:
        structure (obj) : Class containing proteins structural data (Atom coords/positions and masses etc...)
    */

    //  ------------------------------------Initialize vectors and load PDB---------------------------------------------
    retrieve_pdb_file(pdbid, file_type);

    string cwd = filesystem::current_path() ;
    string pdbFile = cwd + "/" + pdbid[1] + pdbid[2] + "/" + pdbid + "." + file_type ;

    string line;
    int Natoms;

    vector<string> atom_list = {} ; 
    vector<vector<double>> atom_coords = {} ; 

    //  --------------------------------------Extracting Molecule Data from pdb---------------------------------------------------
    // Read from the text file
    ifstream File(pdbFile);
    // Use a while loop together with the getline() function to read the file line by line
    while (getline (File, line)) 
    {   
        if (line.substr(0,31) == "_refine_hist.number_atoms_total"){Natoms = stoi(line.substr(31,*line.end()-1));}
        else if ((line.substr(0,4) == "ATOM")||(line.substr(0,6) =="HETATM"))
        {
            // Extract atom element list as array
            int Ndigit = to_string(Natoms).size() + 8 ;
            string atom = (string(1,line[Ndigit+1]) == " ") ? string(1,line[Ndigit])  : string(1,line[Ndigit]) + string(1,tolower(line[Ndigit+1])); 
            atom_list.push_back(atom);
            
            stringstream token( line.substr((int)line.find("?")+1,(int)line.find("?")) );
            string word;
            
            // Extract atom coordinates list as array in angstrom 
            vector<double> coord(3); int i = 0;
            while (token >> word) { if(i<=2){ coord[i] = stold(word) ; i += 1;} }
            atom_coords.push_back(coord);
        }        
    }
    // Close the file
    File.close(); 

    //  --------------------------------------Setup Molecule Elements---------------------------------------------------
    map<string, double> vdw_radius = mendelev();
    map<string, double> atom_radii;

    for (auto atom : atom_list)
    {
        if(vdw_radius.count(atom)){atom_radii.insert_or_assign(atom, vdw_radius[atom]);}
        else{atom_radii.insert_or_assign(atom, vdw_radius["c"]);}
    }
    
    //  --------------------------------------Setup Molecule Geometry---------------------------------------------------    
    // Rotate coordinates of molecule
    atom_coords = Rotate(atom_coords, rotation);
    atom_coords = T(atom_coords);

    // Find extent of molecule extent
    double surfaceMaxX = *max_element(atom_coords[0].begin(), atom_coords[0].end()) ;
    double surfaceMinX = *min_element(atom_coords[0].begin(), atom_coords[0].end()) ;
    double surfaceMaxY = *max_element(atom_coords[1].begin(), atom_coords[1].end()) ;
    double surfaceMinY = *min_element(atom_coords[1].begin(), atom_coords[1].end()) ;
    double surfaceMaxZ = *max_element(atom_coords[2].begin(), atom_coords[2].end()) ;
    double surfaceMinZ = *min_element(atom_coords[2].begin(), atom_coords[2].end()) ;
    
    double surfaceWidthX = abs(surfaceMaxX-surfaceMinX);
    double surfaceWidthY = abs(surfaceMaxY-surfaceMinY);
    double surfaceWidthZ = abs(surfaceMaxZ-surfaceMinZ);
    
    // Centre molecule geometry in xy and set z=0 at the top of the base with percentage of height not imbedded
    atom_coords[0] = add(atom_coords[0], -surfaceMinX-surfaceWidthX/2) ;
    atom_coords[1] = add(atom_coords[1], -surfaceMinY-surfaceWidthY/2) ;
    atom_coords[2] = add(atom_coords[2], -surfaceMinZ-surfaceWidthZ*surfaceApprox) ; 

    //  --------------------------------------Setup Base/Surface Geometry---------------------------------------------------
    // Set indentor height functions and indentor radial extent/boundry for z scanPos calculation.
    // Extent of conical indentor is the radius of the top portion
    // Extent of spherical indentor is the radius
    double rBoundary = (indentorType == "Capped") ?  tipDims.r_top : tipDims.rIndentor;
        
    // // Calculate maximum surface height with added clearance. Define substrate/Base dimensions using biomolecules extent in x and y and boundary of indentor
    double surfaceHeight    = *max_element(atom_coords[2].begin(), atom_coords[2].end()) * 1.5 ;
    vector<double> baseDims = {surfaceWidthX+4*rBoundary+binSize, surfaceWidthY+4*rBoundary+binSize, 15 };
    
    //  --------------------------------------Setup Base/Surface Geometry---------------------------------------------------
    Structure structure;
    
    structure.coords = T(atom_coords);
    structure.atoms  = atom_list;
    structure.atom_radii    = atom_radii;
    structure.surfaceHeight = surfaceHeight;
    structure.baseDims      = baseDims;
    
    return structure;
}


Structure TipStructure(double theta_degrees, double rIndentor, double tip_length)
{   /* Produce list of tip structural para meters. 

    Change principle angle to radian. Calculate tangent point where sphere smoothly transitions to cone for capped conical indentor.
    
    Args:
        theta_degrees (float) : Principle conical angle from z axis in degrees
        rIndentor (float)     : Radius of spherical tip portion
        tip_length (float)    : Total cone height
        
    Returns:
        tipDims (Class): Geometric parameters for defining capped tip structure     
    */

    double theta = theta_degrees*(M_PI/1800.);

    // Intercept of spherical and conical section of indentor (Tangent point) 
    const double r_int = rIndentor*abs(cos(theta));
    const double z_int = -rIndentor*abs( sin(theta) );

    // Total radius/ footprint of indentor/ top coordinates
    const double r_top = (r_int+(tip_length-r_int)*abs(tan(theta))) ;
    const double z_top = tip_length-rIndentor ; 

    Structure tipDims ;   
    tipDims.rIndentor  = rIndentor ;
    tipDims.theta      = theta ;
    tipDims.tip_length = tip_length ;
    tipDims.r_int = r_int ;
    tipDims.z_int = z_int ;
    tipDims.r_top = r_top ;
    tipDims.z_top = z_top ;

    return tipDims;
}


vector<double> Zconical(vector<double> r, double r0, double r_int, double z_int, double theta, double R, double tip_length)
{   /* Calculates and returns spherically capped conical tip surface heights from radial  position r. 
    
    Uses radial coordinate along xy plane from centre as tip is axisymmetric around z axis (bottom of tip set as zero point such z0 = R).
    
    Args:
        r (float/1D arr)   : xy radial coordinate location for tip height to be found
        r0 (float)         : xy radial coordinate for centre of tip
        r_int (float)      : xy radial coordinate of tangent point (point where sphere smoothly transitions to cone)
        z_int (float)      : Height of tangent point, where sphere smoothly transitions to cone (defined for tip centred at spheres center, as calculations assume tip centred at indentors bottom the value must be corrected to, R-z_int) 
        theta (float)      : Principle conical angle from z axis in radians
        R (float)          : Radius of spherical tip portion
        tip_length (float) : Total cone height
        
    Returns:
        Z (float/1D arr) : Height of tip at xy radial coordinate    
    */

    const int N = r.size();
    vector<double> Z(N) ; 

    for (int i =0; i< N; ++i)
    {
        // ------------------------------------------------Spherical Boundary------------------------------------------------
        //  For r <= r_int, z <= z_int : (z-z0)^2 +  (r-r0)^2 = R^2 --> z = z0  + ( R^2 - (r-r0)^2 )^1/2   
        //  Using equation of sphere compute height (points outside sphere radius are complex and return nan, 
        //  nan_to_num is used to set these points to max value R). The heights are clip to height of tangent point, R-z_int. 
        //  Producing spherical portion for r below tangent point r_int and constant height R-zint for r values above r_int.
        double z = R - sqrt( R*R - (r[i]-r0)*(r[i]-r0) );

        if (z!=z)
        {
            Z[i] = R-abs(z_int); 
        } 
        else if(z > R-abs(z_int))
        { 
            Z[i] = R-abs(z_int); 
        } 
        else
        {
            Z[i] = z;\
        }

        //  -------------------------------------------------Conical Boundary-------------------------------------------------
        //  r > r_int, z > z_int : z = m*abs(x-x0);  where x = r, x0 = r0 + r_int,  m = 1/tan(theta)
        //  Using equation of cone (line) to compute height for r values larger than tangent point r_int (using where condition) 
        //  For r values below r_int the height is set to zero.
        //  For r values less than r_int, combines spherical portion with zero values from conical, producing spherical section
        //  For r values more than r_int, combines linear conical portion with R-z_int values from spherical, producing cone section
            
        if (abs(r[i]-r0) >= r_int)
        { 
            Z[i] += (abs(r[i]-r0)-r_int)/abs(tan(theta)); 
        } 
        else
        {
            Z[i] += 0;
        }

     }   

    //  Optional mask values greater than tip length
    //  Z = ma.masked_greater(z1+z2, tip_length )
    return Z;
}


class Geometry
{
    public:             
        vector<vector<double>> clipped;
        vector<vector<double>> full;
};


Geometry ScanGeometry(Structure structure, string indentorType, Structure tipDims, double binSize, double clearance)
{   /* Produces array of scan locations and corresponding heights/ tip positions above surface in Angstroms (x10-10 m). 
    
    Also return an array including only positions where tip interact with the sample. The scan positions are produced creating a 
    rectangular grid over bases extent with widths bin size. Heights, at each position, are calculated by set tip above sample and 
    calculating vertical distance between of tip and molecules surface over the indnenters area. Subsequently, the minimum vertical 
    distance corresponds to the position where tip is tangential.
    
    Args:
        atom_coord (arr)      : Array of coordinates [x,y,z] for atoms in biomolecule 
        atom_radius (dict)    : Dictionary containing van der waals radii each the element in the biomolecule 
        atom_element (arr)    : Array of elements names(str) for atoms in biomolecule 
        indentorType (str)    : String defining indentor type (Spherical or Capped)
        tipDims (Class)        : Geometric parameters for defining capped tip structure     
        baseDims (arr)        : Geometric parameters for defining base/ substrate structure [width, height, depth] 
        surfaceHeight (float) : Maximum height of biomolecule in z direction
        binSize (float)       : Width of bins that subdivid xy domain during raster scanning/ spacing of the positions sampled over
        clearance (float)     : Clearance above molecules surface indentor is set to during scan
        
    Returns:
        scanPos (arr)         : Array of coordinates [x,y,z] of scan positions to image biomolecule and initial heights/ hard sphere boundary
        clipped_scanPos (arr) : Array of clipped (containing only positions where tip and molecule interact) scan positions and initial heights [x,y,z] to image biomolecule
    */
    // ------------------------------------Set Scan Positions from Scan Geometry---------------------------------------------
    int xNum = (int)round( (structure.baseDims[0]/binSize)+1 );
    int yNum = (int)round( (structure.baseDims[1]/binSize)+1 );

    // Create rectangular grid of xy scan positions over base using meshgrid. 
    vector<double> x = linspace(-structure.baseDims[0]/2, structure.baseDims[0]/2, xNum);
    vector<double> y = linspace(-structure.baseDims[1]/2, structure.baseDims[1]/2, yNum);
        
    //  --------------------------------------Set Vertical Scan Positions Positions -------------------------------------------   
    // Extract each atoms radius using radius dictionary [Natoms]
    vector<string> atom_list = structure.atoms;    
    map<string, double> atom_radii = structure.atom_radii;

    // Set indentor height functions and indentor radial extent/boundry for z scanPos calculation.
    // Extent of conical indentor is the radius of the top portion
    double rBoundary  =  tipDims.r_top;
    vector<vector<double>> atom_coords = structure.coords;
         
    // Array of radial positions along indentor radial extent. Set indentor position/ coordinate origin at surface height 
    // (z' = z + surfaceHeight) and calculate  vertical heights along the radial extent. 
    vector<double> r = linspace(-rBoundary, rBoundary, 50) ;

    vector<double> zIndentor = add(Zconical(r, 0, tipDims.r_int, tipDims.z_int, tipDims.theta, tipDims.rIndentor, tipDims.tip_length), structure.surfaceHeight) ;
    
    // Produce xy scan positions of indentor, set initial z height as clearance
    vector<vector<double>> scanPos = {};
    for(int i=0; i<xNum; i++)
    {   
        for(int j=0; j<yNum; j++)
        {   
            double dzmin = structure.surfaceHeight; 

            for (int k; k < atom_coords.size(); k++ )
            {   // Calculate radial distance from scan position to each atom centre giving array of  [NscanPos, Natoms]
                double rInteract = sqrt( (atom_coords[k][0] - x[i])*(atom_coords[k][0] - y[j]) + (atom_coords[k][1] -scanPos[i][1])*(atom_coords[k][1] -scanPos[i][1]) ) ;
                double rElement = atom_radii[atom_list[k]] ;

                if (rInteract <= rBoundary+rElement)
                {   for (int k; k < r.size(); k++ )
                    {   if ( abs(rElement) <= abs(r[k]-rInteract) )
                        {   // Find vertical distances from atoms to indentor surface over all scan positions inside np.nan_num(nan_num removes any infinites). Minus from zIndentor to calculate the 
                            // difference in the indentor height and the atoms surface at each point along indenoter extent, produces a dz array of all the height differences between indentor and 
                            // surface atoms within the indentors boundary around this position. Find the minimum (ensurring maximum is surface height with initial). Therefore, z' = -dz  gives an 
                            // array of indentor positions when each individual part of surface atoms contacts the tip portion above. Translating from z' basis (with origin at z = surfaceHeight) to 
                            // z basis (with origin at the top of the base) is achieved by perform translation z = z' + surfaceheight. Therefore, these tip position are given by dz = surfaceheight-dz'. 
                            // The initial height corresponds to the maximum value of dz/ min value of dz' where the tip is tangential to the surface. I.e. when dz' is minimised all others dz' tip 
                            // positions will be above/ further from the surface. Therefore, at this position, the rest of the indentor wil  not be in contact with the surface and it is tangential.    
                            double dz = zIndentor[k] - (atom_coords[k][2] + sqrt( rElement*rElement - (r[k]-rInteract)*(r[k]-rInteract) ) );
                            dzmin = (dz<dzmin) ? dz : dzmin;
                        }   
                    }
                }
            }

            double z = structure.surfaceHeight - abs(dzmin) + clearance ;
            scanPos.push_back({x[i], y[j], z})                          ;
            cout << x[i] << "," << y[j] << "," << z  << endl                           ;
        }
    }

    //  ---------------------------------------------Clip Scan position ---------------------------------------------------------    
    // Include only positions where tip interact with the sample. Scan position equal clearance, corresponds to indentor at base height 
    // therfore, can't indent surface (where all dz' heights were greater than surface height )
    vector<vector<double>> clipped_scanPos ={};
    for( int i=0; i< scanPos.size(); i++){   
        if (scanPos[i][2] != clearance){ clipped_scanPos.push_back({scanPos[i][0], scanPos[i][1], scanPos[i][2]}); }   } 
    
    Geometry scanGeometry;
    scanGeometry.clipped = clipped_scanPos;
    scanGeometry.full = scanPos;

    return scanGeometry ;
}


int main()
{
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //  ----------------------------------Simulation Variables------------------------------------------------
    // Surface variables                                                  //
    const string pdbid = "1bna";                                          // '1bna' 'bdna8' '4dqy' 'pyn1bna'
    const string file_type = "cif";                                       // pdb, cif, xml
    vector<double> rotation = {0.0,270.0,60.0};                           // degrees
    const double surfaceApprox = 0.2;                                     // arb
                                                                          //
    // Indentor variables                                                 //
    const string indentorType = "capped";                                 // string
    const double rIndentor     = 2.0 ;                                    // (x10^-10 m / Angstroms)
    const double theta_degrees = 5.0 ;                                    // degrees
    const double tip_length    = 50.0 ;                                   // (x10^-10 m / Angstroms)
                                                                          //
    // Scan variables                                                     //
    double clearance = 0.5 ;                                              // (x10^-10 m / Angstroms)
    const double binSize = 6.0;                                           // (x10^-10 m / Angstroms)
    const double indentionDepth = clearance + 2.0;                        // (x10^-10 m / Angstroms)
    const double forceRef = 1.0 ;                                         // (x10^-10 N / pN)
    const double contrast = 1.61 ;                                        // arb
                                                                          //
    // ABAQUS variable                                                    //
    const double timePeriod   = 1.0 ; //1.5                               // s
    const double timeInterval = 0.1 ;                                     // s
    const double meshSurface  = 2.5 ; // 0.6                              // (x10^-10 m / Angstroms)
    const double meshBase     = 2.0 ;                                     // (x10^-10 m / Angstroms)
    const double meshIndentor = 0.6 ;   //0.35                            // (x10^-10 m / Angstroms)

    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    Structure tipDims = TipStructure(theta_degrees, rIndentor, tip_length);
    Structure structure = MolecularStructure(pdbid, file_type, rotation, tipDims, indentorType, binSize, surfaceApprox);
    Geometry scanGeometry = ScanGeometry(structure, indentorType, tipDims, binSize, clearance);

    // display<string>(structure.atoms);
    // display<double>(structure.coords);

    // display<double>(scanGeometry.full);
    cout << "structure.coords.size()" << endl ;
}