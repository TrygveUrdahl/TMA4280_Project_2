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
  return 1;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  if (argc < 2) {
    std::cout << "Requires argument(s): " << std::endl;
    std::cout << "\tint n: size of grid (power of two)" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
    return 1;
  }
  int n = atoi(argv[1]);

  if (!((n != 0) && ((n &(n - 1)) == 0))) {
    // std::cout << "\"n\" must be power of two! " << std::endl;
    // MPI_Abort(MPI_COMM_WORLD, MPI_ERR_DIMS);
    // return 1;
  }

  // Setup MPI variables
  int rank, size;
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  MPI_Comm_size(myComm, &size);
  MPI_Comm_rank(myComm, &rank);

                                          // Grid points per direction is n + 1
  int m = n - 1;                          // Degrees of freedom in each direction
  double h = 1.0/n;                       // Mesh size / step size
  std::vector<int> nPerRankVec(size, 0);  // Cols per rank
  int nn = 4 * n;

  // Initialize nPerRankVec
  for (int i = 0; i < size; i++) {
    nPerRankVec.at(i) = m/size;
  }
  // Distribute the rest of columns
  int rest = m % size;
  for (int i = 0; i < rest; i++) {
    nPerRankVec.at(i)++;
  }

  int nPerRank = nPerRankVec.at(rank); // For simplicity typing

  // Setup matrixes and send/recv buffers
  matrix_t b = makeMatrix(nPerRank, m);
  matrix_t bt = makeMatrix(nPerRank, m);
  vector_t xAxis = makeVector(nPerRank);
  vector_t yAxis = makeVector(n + 1);
  vector_t z = makeVector(nn);
  vector_t diag = makeVector(n);
  std::vector<int> bsize(size, 0);
  std::vector<int> bsizegather(size, 0);
  std::vector<int> displacement(size, 0);
  std::vector<int> displacementgather(size, 0);

  // Initialize vectors for transpose logic
  if (rank==0) std::cout << "bsize: ";
  for (int i = 0; i < size; i++) {
    bsize.at(i) = nPerRankVec.at(rank) * nPerRankVec.at(i);
    bsizegather.at(i) = nPerRankVec.at(i) * m;
    if (rank == 0) std::cout << bsize.at(i) << " ";
  }
  if (rank==0) std::cout << std::endl;
  for (int i = 1; i < size; i++) {
    displacement.at(i) = displacement.at(i - 1) + bsize.at(i - 1);
    displacementgather.at(i) = displacementgather.at(i - 1) + bsizegather.at(i - 1);
  }

  testTranspose(bt, b, m, bsize, bsizegather, displacement, displacementgather, nPerRankVec, rank, size, myComm);


  /*
  // Create local x and y axis index vectors for convenience
  for (int i = 0; i < nPerRank; i++) {
    xAxis.vec.at(i) = (i + 1 + nPerRank*rank) * h;
  }
  for (int i = 0; i < n + 1; i++) {
    yAxis.vec.at(i) = (i + 1) * h;
  }

  for (int i = 0; i < n; i++) {
    diag.vec.at(i) = 2.0 * (1.0 - cos((i + 1) * M_PI / n));
  }

  for (int i = 0; i < nPerRank; i++) {
    for (int j = 0; j < n; j++) {
      double* elem = b.vec.data() + matIdx(b, i, j);
      *elem = h * h * rhs(fRhs, xAxis.vec.at(i), yAxis.vec.at(j));
    }
  }

  // Start solving, one column FST per iteration
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    fst_(b.vec.data() + (m * i), &n, z.vec.data(), &nn);
  }

  // Transpose
  //transpose_p(bt, b, bsize, displacement, nPerRankVec, rank, size, myComm);
  transpose(bt, b, bsize, displacement, nPerRankVec, rank, size, myComm);

  // Inverse FST per column
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    fstinv_(bt.vec.data() + (m * i), &n, z.vec.data(), &nn);
  }

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    for (int j = 0; j < m; j++) {
      double* elem = bt.vec.data() + matIdx(b, i, j);
      *elem /= diag.vec.at(i + rank * nPerRank) + diag.vec.at(j);
    }
  }

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    fst_(bt.vec.data() + (m * i), &n, z.vec.data(), &nn);
  }

  // Transpose
  //transpose_p(b, bt, bsize, displacement, nPerRankVec, rank, size, myComm);
  transpose(b, bt, bsize, displacement, nPerRankVec, rank, size, myComm);

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    fstinv_(b.vec.data() + (m * i), &n, z.vec.data(), &nn);
  }


  matrix_t result = makeMatrix(m,m);
  gatherMatrix(result, b, bsizegather, displacementgather, rank, myComm);

  if(rank == 0) exportMatrix(result);
  */
/*
  // Test matrix export
  if (rank==0) {
    auto matrix = makeMatrix(n,n);
    for (int i = 0; i < n*n; i++) {
      matrix.vec.at(i) = i;
    }
    exportMatrix(matrix);
  }
*/



/*
  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-start;
  std::cout << "Time taken: " << diff.count() << "s" << std::endl;
*/




  MPI_Finalize();
  return 0;
}
