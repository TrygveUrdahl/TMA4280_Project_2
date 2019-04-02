#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include "omp.h"
#include "mpi.h"

#include "structs.hpp"


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

// Parallel transpose of matrix
// Inputs
// b: matrix
// bt: matrix transposed (result)
// send: send buffer
// recv: receive buffer
// n: total number of rows
// nPerRank: number of columns per rank
// rank: index of node in communicator
// size: total number of nodes
// myComm: MPI Comm to use
void transpose(matrix_t &b, matrix_t &bt, vector_t &send, vector_t &recv,
              int n, int nPerRank, int rank, int size, MPI_Comm myComm) {

  for (int i = 0; i < nPerRank; i++) {
    for (int j = 0; j < n; j++) {
      int idx = nPerRank * i + (j / nPerRank) * (nPerRank * nPerRank) + j % nPerRank;
      send.vec.at(idx) = b.vec.at(matIdx(b, i, j));
    }
  }

  MPI_Alltoall(send.vec.data(), nPerRank*nPerRank, MPI_DOUBLE, recv.vec.data(),
    nPerRank*nPerRank, MPI_DOUBLE, myComm);

  int index = 0;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < nPerRank; i++) {
      bt.vec.at(matIdx(bt, i, j)) = recv.vec.at(index);
      index++;
    }
  }
}
