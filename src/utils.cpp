#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include "omp.h"

#include "structs.hpp"

inline int matIdx(matrix_t &mat, int x, int y) {
  return y + mat.n1 * x;
}

// Export matrix to file
// Input
// mat: matrix to export
void exportMatrix(matrix_t &mat, std::string fname = "../output/mat.txt") {

}

// Print matrix to console for debug
// Input
// mat: matrix to print
void printMatrix(matrix_t &mat) {
  int n1 = mat.n1;
  int n2 = mat.n2;
  std::cout << "Printing matrix: " << std::endl;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      std::cout << mat.vec.at(matIdx(mat, i,j)) << "\t";
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

// Input
// n1: first dimension of matrix
// n2: second dimension of matrix
matrix_t makeMatrix(int n1, int n2) {
  std::vector<double> vec(n1*n2,0);
  matrix_t mat;
  mat.vec = vec;
  mat.n1 = n1;
  mat.n2 = n2;
  return mat;
}

// Calculate transpose of matrix in-place sequential
// Input
// mat: matrix to transpose in-place
void transposeSeq(matrix_t &mat) {
  int n1 = mat.n1;
  int n2 = mat.n2;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      if (j>i) {
        std::swap(mat.vec.at(matIdx(mat, i, j)), mat.vec.at(matIdx(mat, j, i)));
      }
    }
  }
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
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    mat.vec.at(matIdx(mat, i,i)) = 2;
    if (i > 0) {
      mat.vec.at(matIdx(mat, i,i-1)) = -1;
    }
    if (i < (n - 1)) {
      mat.vec.at(matIdx(mat, i,i+1)) = -1;
    }
  }
  return mat;
}
