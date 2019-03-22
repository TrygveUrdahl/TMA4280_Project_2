#include <iostream>
#include <cmath>
#include <vector>

// Initialize vector with all values initialized to 0
// Input
// n: size of vector
std::vector<double> makeVector(int n) {
  std::vector<double> array(n, 0);
  return array;
}


// TODO: make memory contiguous?
// Input
// n1: first dimension of matrix
// n2: second dimension of matrix
std::vector< std::vector<double> > makeMatrix(int n1, int n2) {
  std::vector< std::vector<double> > mat(n1);
  for (auto vec : mat) {
    vec.resize(n2);
    std::fill(vec.begin(), vec.end(), 0);
  }
  return mat;
}


// Calculate transpose of matrix in-place
// Input
// mat: matrix to transpose in-place
void transpose(std::vector< std::vector<double> > &mat) {

}


// Calculate right hand side from passed function pointer
// Input
// function: pointer to function for evaluating rhs
// x: first parameter of function
// y: second paramenter of function
double rhs(double (*function)(double, double), double x, double y) {
  return function(x, y);
}

// Export matrix to file
// Input
// mat: matrix to export
void exportMatrix(std::vector< std::vector<double> > &mat) {

}
