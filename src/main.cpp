#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include "mpi.h"
#include "omp.h"


// Own include files
#include "extern.hpp"
#include "utils.hpp"
#include "structs.hpp"

double fRhs(double x, double y) {
  return 1.0;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  if (argc < 2) {
    std::cout << "Requires argument(s): " << std::endl;
    std::cout << "\tint n: size of grid (power of two)" << std::endl;
    MPI_Finalize();
    return 1;
  }
  int n = atoi(argv[1]);
  if (!((n != 0) && ((n &(n - 1)) == 0))) {
    std::cout << "\"n\" must be power of two! " << std::endl;
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_DIMS);
    return 1;
  }

  // Setup MPI variables
  int rank, size;
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  MPI_Comm_size(myComm, &size);
  MPI_Comm_rank(myComm, &rank);

  if (!((size != 0) && ((size &(size - 1)) == 0))) {
    if (rank == 0) std::cout << "World size must be power of two! " << std::endl;
    MPI_Abort(myComm, MPI_ERR_DIMS);
    return 1;
  }


  // Setup for program parameters completed

                          // Grid points per direction is n + 1
  int m = n - 1;          // Degrees of freedom in each direction
  double h = 1.0/n;       // Mesh size / step size
  int nPerRank = n/size;  // Cols per rank
  int nn = 4 * n;

  // Setup matrixes and send/recv buffers
  matrix_t b = makeMatrix(nPerRank, m);
  matrix_t bt = makeMatrix(nPerRank, m);
  vector_t send = makeVector(nPerRank * n);
  vector_t recv = makeVector(nPerRank * n);
  vector_t xAxis = makeVector(nPerRank);
  vector_t yAxis = makeVector(n);
  vector_t z = makeVector(nn);
  vector_t diag = makeVector(n);

  // Create local x and y axis vectors for convenience
  for (int i = 0; i < nPerRank; i++) {
    xAxis.vec.at(i) = (i + 1 + nPerRank*rank) * h;
  }
  for (int i = 0; i < n + 1; i++) {
    yAxis.vec.at(i) = (i + 1) * h;
  }

  for (int i = 0; i < n; i++) {
    diag.vec.at(i) = 2.0 * (1.0 - cos((i+1) * M_PI / n));
  }

  for (int i = 0; i < nPerRank; i++) {
    for (int j = 0; j < n; j++) {
      b.vec.at(matIdx(b, i, j)) = h * h * rhs(fRhs, xAxis.vec.at(i), yAxis.vec.at(j));
    }
  }


  // Start solving
  for (int i = 0; i < nPerRank; i++) {
    //fst_(b.vec.data()[n*i], &n, z.vec.data(), &nn);
  }

  // TODO: Transpose
  // transpose(bt, b, ...);

  for (int i = 0; i < nPerRank; i++) {
    //fstinv_(bt.vec.data()[n*i], &n, z.vec.data(), &nn);
  }

  for (int i = 0; i < nPerRank; i++) {
    for (int j = 0; j < n; j++) {
      bt.vec.at(matIdx(bt, i, j)) /= diag.vec.at(i + rank * nPerRank) + diag.vec.at(j);
    }
  }

  for (int i = 0; i < nPerRank; i++) {
    //fst_(bt.vec.data()[n*i], &n, z.vec.data(), &nn);
  }

  // TODO: Transpose
  // transpose(b, bt, ...);

  for (int i = 0; i < nPerRank; i++) {
    //fstinv_(b.vec.data()[n*i], &n, z.vec.data(), &nn);
  }

/*
  auto start = std::chrono::high_resolution_clock::now();
  auto mat = make1DLaplaceOperator(n);
  auto end = std::chrono::high_resolution_clock::now();
  printMatrix(mat);
  //transposeSeq(mat);
  //printMatrix(mat);
  std::chrono::duration<double> diff = end-start;
  std::cout << "Time taken to create Laplace Operator: " << diff.count() << "s" << std::endl;
*/




  MPI_Finalize();
  return 0;
}
