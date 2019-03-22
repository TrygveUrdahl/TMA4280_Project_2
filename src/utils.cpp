#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

typedef std::vector< std::vector<double> > matrix_t;

// Export matrix to file
// Input
// mat: matrix to export
void exportMatrix(matrix_t &mat, std::string fname = "../output/mat.txt") {

}

// Print matrix to console for debug
// Input
// mat: matrix to print
void printMatrix(matrix_t &mat) {
  int n1 = mat.size();
  int n2 = mat.at(0).size();

  std::cout << "Printing matrix: " << std::endl;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      std::cout << mat.at(i).at(j) << "\t";
    }
    std::cout << std::endl;
  }
}

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
matrix_t makeMatrix(int n1, int n2) {
  matrix_t mat;
  mat.resize(n2);
  for (int i = 0; i < n1; i++) {
    std::vector<double> vec(n1, 0);
    mat.at(i) = vec;
  }
  return mat;
}

// Calculate transpose of matrix in-place
// Input
// mat: matrix to transpose in-place
void transpose(matrix_t &mat) {

}

// Calculate right hand side from passed function pointer
// Input
// function: pointer to function for evaluating rhs
// x: first parameter of function
// y: second paramenter of function
double rhs(double (*function)(double, double), double x, double y) {
  return function(x, y);
}

// Generate 1D Laplace operator matrix
// Input
// n: dimension of matrix
matrix_t make1DLaplaceOperator(int n) {
  matrix_t mat = makeMatrix(n, n);
  for (int i = 0; i < n; i++) {
    mat.at(i).at(i) = 2;
    if (i > 0) {
      mat.at(i).at(i - 1) = -1;
    }
    if (i < (n - 1)) {
      mat.at(i).at(i + 1) = -1;
    }
  }
  return mat;
}
