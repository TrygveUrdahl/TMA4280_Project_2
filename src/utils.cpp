#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include "omp.h"
#include "mpi.h"

#include "structs.hpp"

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


// Parallel transpose matrix b into matrix bt
// Input
void transpose(matrix_t &bt, matrix_t &b, std::vector<int> &bsize, std::vector<int> &displacement,
  						std::vector<int> &nPerRankVec, int rank, int size, MPI_Comm myComm){


	MPI_Alltoallv(b.vec.data(), bsize.data(), displacement.data(), MPI_DOUBLE,
                bt.vec.data(), bsize.data(), displacement.data(), MPI_DOUBLE, myComm);

	int d = 0;
  for (int i = 0; i < size; i++) {
    d = 0;
    for (int j = 0; j < i; j++) {
      d = d + nPerRankVec.at(j);
    }
    for (int column = 0; column < nPerRankVec.at(rank); column++) {
      for (int row = column + 1; row < nPerRankVec.at(i); row++) {
				//double *elem1 = bt.vec.data() + matIdx(bt, column, row + d);
        //double *elem2 = bt.vec.data() + matIdx(bt, row, column + d);
				double temp = bt.vec.at(matIdx(bt, column, row + d));
				bt.vec.at(matIdx(bt, column, row + d)) = bt.vec.at(matIdx(bt, column, row + d));
				bt.vec.at(matIdx(bt, column, row + d)) = temp;
        //std::swap(bt.vec.at(matIdx(bt, column, row + d)), bt.vec.at(matIdx(bt, row, column + d)));

      }
    }
  }
 }


// Gather all b matrices to result on root node 0
// Input
void gatherMatrix(matrix_t &result, matrix_t &b, std::vector<int> &bsize,
          std::vector<int> &displacement, int rank, MPI_Comm myComm) {
	MPI_Gatherv(b.vec.data(), b.vec.size(), MPI_DOUBLE, result.vec.data(),
              bsize.data(), displacement.data(), MPI_DOUBLE, 0, myComm);
}



void transpose_p(matrix_t &bt, matrix_t &b, std::vector<int> bsize, std::vector<int> displ, std::vector<int> nPerProcVec, int rank, int size, MPI_Comm myComm) {
  double* Apck = bt.vec.data();
	int* nrows = nPerProcVec.data();
	int np = nPerProcVec.at(rank);
  for (int p = 0, off_rp = 0; p < size; ++p, off_rp+=nrows[p]) {
	  for (int i = 0; i < np; ++i, Apck+=nrows[p]) {
			double* A = b.vec.data();
			memcpy(Apck, &(A[i]) + off_rp, nrows[p]*sizeof(double));
	  }
	}

  /* Exchange blocks */
  MPI_Alltoallv(bt.vec.data(), bsize.data(), displ.data(), MPI_DOUBLE, b.vec.data(), bsize.data(), displ.data(), MPI_DOUBLE, myComm);

  /* Transpose blocks */
  Apck = b.vec.data();
  for (int p = 0, off_rp = 0; p < size; ++p, off_rp+=nrows[p], Apck +=bsize[p]) {
	  for (int i = 0; i < np; ++i) {
			for (int j = 0; j < nrows[p]; ++j) {
				bt.vec.at(matIdx(bt, i, off_rp + j)) = Apck[j * nrows[p]+i];
      }
	  }
	}
}


void fillMatrix(matrix_t &b, std::vector<int> &displacementgather, int n1, int n2, int rank) {
	for (int i = 0; i < n1 * n2; i++) {
		b.vec.at(i) = i + n1*displacementgather.at(rank);
	}
}


void testTranspose(matrix_t &bt, matrix_t &b, int m, std::vector<int> &bsize, std::vector<int> &bsizegather, std::vector<int> &displacement, std::vector<int> &displacementgather, std::vector<int> &nPerRankVec, int rank, int size, MPI_Comm myComm) {
	matrix_t result = makeMatrix(m,m);

	fillMatrix(b, displacementgather, nPerRankVec.at(rank), m, rank);

	gatherMatrix(result, b, bsizegather, displacementgather, rank, myComm);
	if(rank == 0) std::cout << "Before transpose: " << std::endl;
	if(rank == 0) printMatrix(result);

	transpose_p(bt, b, bsize, displacement, nPerRankVec, rank, size, myComm);
	gatherMatrix(result, b, bsizegather, displacementgather, rank, myComm);
	if(rank == 0) std::cout << "After transpose: " << std::endl;
	if(rank == 0) printMatrix(result);
}
