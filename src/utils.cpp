#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include "omp.h"
#include "mpi.h"

#include "structs.hpp"
#include "rhs.hpp"

// Return id of element (x, y) in matrix
// Input
// mat: matrix to look up element in
// x, y: indices
int matIdx(matrix_t &mat, int x, int y) {
  return y + x * mat.n2;
}

// Export matrix to file
// Input
// mat: matrix to export
void exportMatrix(matrix_t &mat, std::string fname = "../output/mat.txt") {
  std::ofstream file;
  file.open(fname);
  if (file.is_open()) {
    for (int i = 0; i < mat.n1; i++) {
      for (int j = 0; j < mat.n2; j++) {
        file << mat.vec.at(matIdx(mat, i, j)) << " ";
      }
      file << std::endl;
    }
  }
  file.close();
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


// Gather all b matrices to result on root node 0
void gatherMatrix(matrix_t &result, matrix_t &b, std::vector<int> &bsize,
          std::vector<int> &displacement, int rank, MPI_Comm myComm) {
	MPI_Gatherv(b.vec.data(), b.vec.size(), MPI_DOUBLE, result.vec.data(),
              bsize.data(), displacement.data(), MPI_DOUBLE, 0, myComm);
}

// Parallel transpose of matrix
void transpose(matrix_t &bt, matrix_t &b, vector_t &send, vector_t &recv, std::vector<int> &nPerRankVec, std::vector<int> &bsize, std::vector<int> &displacement, int n, int rank, int size, MPI_Comm myComm) {
	int nPerRank = nPerRankVec.at(rank);
	int block_M, current_j = 0, count = 0;
	for (int block_idx = 0; block_idx < size; block_idx++) {
		block_M = bsize.at(block_idx) / nPerRank;
		for (int i = 0; i < nPerRank; i++) {
			for (int j = 0; j < block_M; j++) {
				send.vec.at(count++) = b.vec.at(matIdx(b, i, j+current_j));
			}
		}
		current_j += block_M;
	}

	MPI_Alltoallv(send.vec.data(), bsize.data(), displacement.data(), MPI_DOUBLE,
                recv.vec.data(), bsize.data(), displacement.data(), MPI_DOUBLE, myComm);

  int val = 0;
  for (int j=0; j < n; j++) {
    for (int i=0; i < nPerRank; i++) {
      bt.vec.at(matIdx(bt,i,j)) = recv.vec.at(val++);
    }
  }
}

// Fill matrix with dummy values
void fillMatrix(matrix_t &b, std::vector<int> &displacementgather, int n1, int n2, int rank) {
	for (int i = 0; i < n1 * n2; i++) {
		b.vec.at(i) = i + n1*displacementgather.at(rank);
	}
}

// Test the transpose function
void testTranspose(matrix_t &bt, matrix_t &b, int m, std::vector<int> &bsize, std::vector<int> &bsizegather, std::vector<int> &displacement, std::vector<int> &displacementgather, std::vector<int> &nPerRankVec, int rank, int size, MPI_Comm myComm, vector_t &send, vector_t &recv) {
	matrix_t result = makeMatrix(m,m);

	fillMatrix(b, displacementgather, nPerRankVec.at(rank), m, rank);

	gatherMatrix(result, b, bsizegather, displacementgather, rank, myComm);
	if(rank == 0) std::cout << "Before transpose: " << std::endl;
	if(rank == 0) printMatrix(result);
	// if (rank == 0) exportMatrix(result);

	transpose(bt, b, send, recv, nPerRankVec, bsize, displacement, m, rank, size, myComm);
	gatherMatrix(result, bt, bsizegather, displacementgather, rank, myComm);
	if(rank == 0) std::cout << "After transpose: " << std::endl;
	if(rank == 0) printMatrix(result);
	// if (rank == 0) exportMatrix(result);
}

// Pick which rhs function to use (yay for trailing return type)
auto rhsPicker(int p) -> double(*)(double, double)
{
    if (p == 0) {
			return fRhs;
		}
		if (p == 1) {
			return singularRhs;
		}
		if (p == 2) {
			return invRadialRhs;
		}
		if (p == 3) {
			return expRhs;
		}
		if (p == 4) {
			return trigRhs;
		}
		if (p == 5) {
			return discreteRhs;
		}
		if (p == 99) {
			return convergenceTestRhs;
		}
		return fRhs; // Default
}
