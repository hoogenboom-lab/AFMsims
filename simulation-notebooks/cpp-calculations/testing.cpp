#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <filesystem>

using namespace std;

void test1(void)
{
    int myNum = 5;               // Integer (whole number without decimals)
    double myFloatNum = 5.99;    // Floating point number (with decimals)
    char myLetter  = 'D';         // Character
    string myText  = "Hello";     // String (text)
    bool myBoolean = true;       // Boolean (true or false)

    std::cout << "\n Hello World" << std::endl; //
}


void test2(void)
{

    int x, y, z;

    cout << "Type a number: "; // Type a number and press enter
    cin >> x; // Get user input from the keyboard
    cout << "Your number is: " << x << endl; // Display the input value

    x = y = z = 1;
    y = 10;

    cout << x << endl;
    cout << x + y + z << endl;

    vector<string> msg {"Hello\t", "C++", "World", "from", "VS Code", "and the C++ extension!"};
    
    for (const string& word : msg)
    {
        cout << word << " ";
    }
}

void task1(void)
{
    const int N = 10;
    const int sum = 0;

    vector<double> circumference(N);

    for (int i=0; i<N; i++){
        const double radius = static_cast<double>(i);
        circumference[i] = 2*M_PI*radius;
    }

    sort(circumference.begin(), circumference.end(), greater<int>());

    cout <<"largest circumference is: " << circumference[0] << endl;
}

double R(double x, double y)
{
    return sqrt( pow(x,2) + pow(y,2) );
}

void task2(int N)
{
    double x0 = 0.3;
    double y0 = 0.4;
    double r0 = R(x0,y0);

    int count = 0;
    
    default_random_engine generator;
    uniform_real_distribution<double> uniform(-1.0,10);


    for (int i=0; i<N; ++i)
    {
        const double x = uniform(generator) ;
        const double y = uniform(generator) ;
        const double r = R(x,y) ;

        if (r<1.0){ ++count;}
    }

    double pi = 4.0*static_cast<double>(count)/static_cast<double>(N);

    cout << pi << endl ;   
}

double dot(vector<double> u, vector<double> v)
{
    int N = u.size();
    double uv = 0.0;

    for(int i=0; i<N; ++i)
        uv += u[i]*v[i];

    return uv;
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


vector<double> multiple(vector<double> A, vector<double> B)
{
    int N = A.size();
    
    vector<double> AB(N);

    for (int i=0; i<N; ++i)
        AB[i] = A[i]*B[i];
    
    return AB ;
}

vector<vector<double>> conjugate(vector<vector<double>> A, vector<vector<double>> B)
{
    int N = A.size();
    int M = A[0].size();
    
    vector<vector<double>> AB( N, vector<double>(M) );

    for (int i=0; i<N; ++i)
        for (int j = 0; j<M; j++)
            AB[i][j] = A[i][j]*B[i][j];

    return AB ;
}

vector<vector<double>> product(vector<vector<double>> A, vector<vector<double>> B)
{
    int N = A.size();
    int M = B[0].size();
    
    vector<vector<double>> AB( N, vector<double>(M) );

    for (int i=0; i<N; ++i)
        for (int j = 0; j<M; j++)
            AB[i][j] = dot(A[i],T(B)[j]);
    
    return AB ;
}

void matrix(vector<vector<double>>A)
{
    for (auto n:A)
    {   
        cout << '|'<<' ';
        for (auto m:n){ cout << m << ' ' ;} 
        cout << '|' << endl;   
    }
}

int main(void)
{    
    // vector<double> a = {1,2,3};
    // vector<double> b = multiple(a,{2,1,4});
    // vector<double> c = {3,5,2};

    // vector<vector<double>> A = {c,b,a};
    // // vector<vector<double>> B = conjugate(A, T(A));  
    // vector<vector<double>> B = {a,c,b};  

    // vector<vector<double>> AB = product(A, B);   
      

    // for (auto v:a){cout<<'|'<<v<<'|'<<endl;}
    // cout<<endl;
    // for (auto v:b){cout<<'|'<<v<<'|'<<endl;}
    // cout<<endl;
    // matrix(A);
    // cout<<endl;
    // matrix(B);
    // cout<<endl;
    // matrix(AB);

    cout << filesystem::current_path() << endl;

    string dir =  "C";

    filesystem::create_directory(dir);
}

