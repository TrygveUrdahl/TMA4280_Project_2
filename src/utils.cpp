#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include "omp.h"
#include "mpi.h"

#include "structs.hpp"
#include "utils.hpp"

int matIdx(matrix_t &mat, int x, int y) {
  return y + x * mat.n1;
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
vector_t makeVector(int n) {
  vector_t vec;
  std::vector<double> array(n, 0);
  vec.vec = array;
  vec.n1 = n;
  return vec;
}

// Initialize matrix with all values initialized to 0
// Input
// n1: first dimension of matrix
// n2: second dimension of matrix
matrix_t makeMatrix(int n1, int n2) {
  std::vector<double> vec(n1*n2, 0);
  matrix_t mat;
  mat.vec = vec;
  mat.n1 = n1;
  mat.n2 = n2;
  return mat;
}

// Calculate right hand side from passed function pointer
// Input
// function: pointer to function for evaluating rhs
// x: first parameter of function
// y: second paramenter of function
double rhs(double (*function)(double, double), double x, double y) {
  return function(x, y);
}

// Generate 1D Laplace operator matrix (which it turns out is not needed)
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


void transpose(matrix_t &bt, matrix_t &b, std::vector<int> &bsize, std::vector<int> &displacement,
                std::vector<int> &nPerRankVec, int rank, int size, MPI_Comm myComm){

  MPI_Alltoallv(b.vec.data(), bsize.data(), displacement.data(), MPI_DOUBLE,
                bt.vec.data(), bsize.data(), displacement.data(), MPI_DOUBLE, myComm);

  for (int i = 0; i < size; i++) {
    int d = 0;
    for (int j = 0; j < i; j++) {
      d = d + nPerRankVec.at(j);
    }
    for (int column = 0; column < nPerRankVec.at(rank); column++) {
      for (int row = column + 1; row < nPerRankVec.at(i); row++) {
        double *elem1 = bt.vec.data() + matIdx(bt, column, row + d);
        double *elem2 = bt.vec.data() + matIdx(bt, row, column + d);
        std::swap(*elem1, *elem2);
      }
    }
  }
 }
